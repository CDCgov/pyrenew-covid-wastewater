# Basic Imports
import datetime as dt
import json
import logging
import os
import subprocess
import warnings
from enum import StrEnum
from pathlib import Path
from zoneinfo import ZoneInfo

# Direct use of dagster
import dagster as dg

# Helper Libraries
from cfa.stf.forecasttools import LOCATION_LIST
from cfa_dagster import (
    ADLS2PickleIOManager,
    ExecutionConfig,
    GraphDimension,
    GraphDimensionExclusion,
    SelectorConfig,
    azure_batch_executor,
    collect_definitions,
    docker_executor,
    dynamic_executor,
    dynamic_graph_asset,
    start_dev_env,
)
from cfa_dagster import (
    is_production as is_prod,
)
from pydantic import BaseModel, Field
from pygit2.repository import Repository
from pyrenew_multisignal.hew.utils import flags_from_hew_letters

# Model Code
from cfa.stf.routine._paths import PRODUCTION_PRIORS
from cfa.stf.routine.data.data_access import DataResolution
from cfa.stf.routine.epiautogp.forecast_epiautogp import main as forecast_epiautogp
from cfa.stf.routine.fable.forecast_fable import main as forecast_fable
from cfa.stf.routine.forecast_window import ForecastWindow
from cfa.stf.routine.pyrenew_hew.forecast_pyrenew import main as forecast_pyrenew
from cfa.stf.routine.utils.postprocess_forecast_batches import main as postprocess
from cfa.stf.routine.utils.prop_utils import create_prop_fusion_model
from cfa.stf.routine.utils.r_utils import (
    make_figures_from_model_fit_dir,
    model_fit_dir_to_hub_tbl,
)

log = logging.getLogger(__name__)

# ignore beta automation condition sensor warnings
warnings.filterwarnings(
    "ignore",
    message=r".*AutomationConditionSensorDefinition.*is currently in beta.*",
)

# ============================================================================
# DAGSTER INITIALIZATION
# ============================================================================

# get the user running the Dagster instance
user = os.getenv("DAGSTER_USER")

is_production = is_prod()

start_dev_env(__name__)

# ============================================================================
# RUNTIME CONFIGURATION: WORKING DIRECTORY, EXECUTORS, VOLUME MOUNTS
# ============================================================================
# Executors define the runtime-location of an asset job
# See later on for Asset job definitions

# ---------- Working Directory, Branch, and Image Tag ----------


def _find_project_root() -> Path:
    """Find the checkout containing the Dagster project configuration."""
    working_dir = Path.cwd().resolve()
    module_dir = Path(__file__).resolve().parent
    for candidate in (
        working_dir,
        *working_dir.parents,
        module_dir,
        *module_dir.parents,
    ):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return working_dir


local_workdir = _find_project_root()
container_workdir = Path(
    f"/{local_workdir.name}"
)  # in the container, workdir is mounted at /

# Get branch name from git, defaulting to main if not in a git repo
try:
    current_branch_name = os.environ.get("GITHUB_HEAD_REF") or str(
        Repository(local_workdir).head.shorthand
    )
    log.debug(f"Branch name from git: {current_branch_name}")
except Exception:
    current_branch_name = "main"
    log.warning("No .git folder detected; using main as the branch name")

# Use 'latest' tag for production or main branch, otherwise use branch name
registry = "cfaprdbatchcr.azurecr.io"
tag = (
    "latest"
    if (is_production or current_branch_name == "main")
    else current_branch_name.replace("/", "-")
)
image = f"{registry}/{local_workdir.name}:{tag}"

# ----------- Output volume mount strings ---------------

# Azure Batch writes outputs directly to blob storage.
azure_blob_mounts = [
    f"stf-routine-forecasting-prod-output:{container_workdir}/output",
    f"stf-routine-forecasting-test-output:{container_workdir}/test-output",
]

# Local runs are non-production, so they use one output directory.
local_output_mount = (
    f"{local_workdir / 'test-output'}:{container_workdir / 'test-output'}"
)

# ---------- Execution Configuration ----------

# Launches locally in a new system process
# Used for lightweight assets and jobs, etc. where volume mounts are not needed
basic_execution_config = ExecutionConfig(
    executor=SelectorConfig(class_name=dg.multiprocess_executor.__name__),
)

# Launches locally, executes in a docker container as configured below
# Allows for rapid local testing in a similar-to-batch environment
docker_execution_config = ExecutionConfig(
    executor=SelectorConfig(
        class_name=docker_executor.__name__,
        config={
            "image": image,
            "retries": {"enabled": {}},
            "container_kwargs": {
                "volumes": [
                    # bind the ~/.azure folder for optional cli login
                    f"/home/{user}/.azure:/root/.azure",
                    # bind current file so we don't have to rebuild
                    # the container image for workflow changes
                    f"{__file__}:{container_workdir / 'src/cfa/stf/routine/dagster_defs.py'}",
                    # Store outputs on the host so they persist after the
                    # container exits.
                ]
                + [local_output_mount]
            },
        },
    ),
)

# Cloud execution. This is what we want for any model run.
# Shared config for all Azure Batch pools; only pool_name differs between them.
_azure_batch_shared_config = {
    **(
        {}
        if is_production  # image will come from the code location in prod
        else {"image": image}
    ),
    "container_kwargs": {
        "volumes": [
            # bind the ~/.azure folder for optional cli login
            # f"/home/{user}/.azure:/root/.azure",
            # bind current file so we don't have to rebuild
            # the container image for workflow changes
            # Azure blob output mounts
        ]
        + azure_blob_mounts,
        "working_dir": f"{container_workdir}",
    },
}

azure_batch_2cpu_execution_config = ExecutionConfig(
    executor=SelectorConfig(
        class_name=azure_batch_executor.__name__,
        config={
            "pool_name": "stf-routine-2cpu",
            **_azure_batch_shared_config,
        },
    ),
)

azure_batch_4cpu_execution_config = ExecutionConfig(
    executor=SelectorConfig(
        class_name=azure_batch_executor.__name__,
        config={
            "pool_name": "stf-routine-4cpu",
            **_azure_batch_shared_config,
        },
    ),
)

azure_batch_64cpu_execution_config = ExecutionConfig(
    executor=SelectorConfig(
        class_name=azure_batch_executor.__name__,
        config={
            "pool_name": "stf-routine-64cpu",
            **_azure_batch_shared_config,
        },
    ),
)

# ============================================================================
# GRAPH DIMENSIONS AND PARTITIONS
# How are the data split and processed in Azure Batch?
# ============================================================================

DEFAULT_EXCLUDED_LOCATIONS = ["AS", "GU", "MP", "PR", "UM", "VI"]
SUPPORTED_DISEASES = ["covid", "flu", "rsv"]

# Disease dimensions
DISEASES = SUPPORTED_DISEASES
Disease = StrEnum("Disease", {v: v for v in DISEASES})

# Location dimensions
LOCATIONS = [
    location for location in LOCATION_LIST if location not in DEFAULT_EXCLUDED_LOCATIONS
]
Location = StrEnum("Location", {v: v for v in LOCATIONS})

# Daily Partitions
tz = "America/New_York"
daily_partitions_def = dg.DailyPartitionsDefinition(
    start_date=dt.datetime.now(ZoneInfo(tz)) - dt.timedelta(days=1),
    end_offset=1,
    timezone=tz,
)

# ============================================================================
# ASSET CONFIGURATIONS
# ============================================================================


# Use default_factory to prevent ConfigOverrides from populating fields in the
# Launchpad unless the user explicitly sets them.
class _SharedModelConfigFields(BaseModel):
    output_basedir: str = Field(
        default_factory=lambda: "",
        description="Output directory used by all forecast models.",
    )
    exclude_last_n_days: int = Field(
        default_factory=lambda: 0,
        description="Requested recent-data omission used by all forecast models.",
    )
    fail_on_stale_data: bool = Field(
        default_factory=lambda: is_production,
        description="Stale-input policy used by all forecast models.",
    )


class ConfigOverride(_SharedModelConfigFields, dg.Config):
    location: Location  # type: ignore[reportInvalidTypeForm]
    n_lookback_days: int | None = Field(
        default_factory=lambda: None,
        description=(
            "Training lookback applied to all forecast models for this location."
        ),
    )

    def as_dict(self) -> dict:  # type: ignore[reportInvalidTypeForm]
        return self.model_dump(mode="json", exclude_unset=True)


class ModelBaseConfig(_SharedModelConfigFields, dg.ConfigurableResource):
    """
    Shared configuration and explicitly model-scoped lookbacks for model assets.
    """

    output_basedir: str = Field(
        default="output" if is_production else "test-output",
        description="Output directory used by all forecast models.",
    )
    fable_pyrenew_n_lookback_days: int | None = Field(
        default=150,
        description="Training lookback used only by Fable and PyRenew models.",
    )
    epiautogp_n_lookback_days: int | None = Field(
        default=None if is_production else 150,
        description="Training lookback used only by EpiAutoGP models.",
    )
    exclude_last_n_days: int = Field(
        default=1,
        description="Requested recent-data omission used by all forecast models.",
    )
    fail_on_stale_data: bool = Field(
        default=is_production,
        description="Stale-input policy used by all forecast models.",
    )
    diseases: GraphDimension[Disease] = Field(  # type: ignore[reportInvalidTypeForm]
        default=GraphDimension(DISEASES),
        description="Diseases run by all selected forecast models.",
    )
    locations: GraphDimension[Location] = Field(  # type: ignore[reportInvalidTypeForm]
        default=GraphDimension(LOCATIONS),
        description="Locations run by all selected forecast models.",
    )
    # Add defaults here, or add in the launchpad with ctrl+space
    config_overrides: list[ConfigOverride] = Field(
        default=[
            # ConfigOverride(location="GA", exclude_last_n_days=2).as_dict(),
        ],
        description=(
            "Provide location-specific overrides as a list of dicts. "
            "An explicitly provided n_lookback_days applies to all models, "
            "which otherwise retain their model-specific defaults. "
            "The Launchpad accepts both YAML and JSON-style lists, e.g. "
            "config_overrides: [{ location: GA, n_lookback_days: 120 }]."
        ),
    )  # type: ignore[reportInvalidTypeForm]

    def get_by_location(self, loc: Location) -> "ModelBaseConfig":  # type: ignore[reportInvalidTypeForm]
        overrides = {}
        for entry in self.config_overrides:
            if isinstance(entry, dict):
                entry = ConfigOverride(**entry)
            if entry.location == loc:
                overrides = entry.model_dump(exclude={"location"}, exclude_unset=True)
                if "n_lookback_days" in overrides:
                    n_lookback_days = overrides.pop("n_lookback_days")
                    overrides["fable_pyrenew_n_lookback_days"] = n_lookback_days
                    overrides["epiautogp_n_lookback_days"] = n_lookback_days
                break
        return self.model_copy(update=overrides)


class FableEOtherConfig(dg.ConfigurableResource):
    """
    Configuration for fable E-other model assets
    (fable_e_other, epiweekly_fable_e_other).
    These default values can be modified in the Dagster asset materialization launchpad.
    """

    n_samples: int = 400 if not is_production else 2000


class PyrenewConfig(dg.ConfigurableResource):
    """
    Configuration for Pyrenew model assets (pyrenew_e, pyrenew_h, pyrenew_he, etc.).
    These default values can be modified in the Dagster asset materialization launchpad.
    """

    n_warmup: int = 200 if not is_production else 1000
    n_samples: int = 200 if not is_production else 500
    n_chains: int = 2 if not is_production else 4
    rng_key: int = 12345
    additional_forecast_letters: str = ""


class EpiAutoGPEPctEpiweeklyConfig(dg.ConfigurableResource):
    """Configuration for the epiweekly EpiAutoGP E-pct model asset."""

    n_particles: int = 64 if is_production else 4
    n_mcmc: int = 200 if is_production else 100
    n_hmc: int = 50 if is_production else 25
    n_forecast_draws: int = 2000
    smc_data_proportion: float = 0.1
    n_threads: str = "auto"


class EModelExclusions(dg.ConfigurableResource):
    # filter out WY
    locations: GraphDimensionExclusion[Location] = GraphDimensionExclusion(["WY"])  # type: ignore[reportInvalidTypeForm]


class WModelExclusions(dg.ConfigurableResource):
    # only covid is valid for W
    diseases: GraphDimension[Disease] = GraphDimension(["covid"])  # type: ignore[reportInvalidTypeForm]


class PostProcessConfig(dg.Config):
    """
    Configuration for the Post-Processing asset.
    """

    output_basedir: str = "output" if is_production else "test-output"
    skip_existing: bool = False
    postprocess_diseases: list[str] = ["covid", "flu", "rsv"]


# ============================================================================
# MODEL CONSTRUCTOR FUNCTIONS - these are used later, in Asset Definitions
# ============================================================================


def _throw_if_backfill(
    context: dg.OpExecutionContext | dg.AssetExecutionContext,
    partition_def: dg.PartitionsDefinition,
):
    current_partition = context.partition_key
    latest_partition = partition_def.get_last_partition_key()
    if current_partition != latest_partition:
        raise RuntimeError("STF forecast models do not support backfills")


def _run_fable_e_other(
    context: dg.OpExecutionContext,
    fable_e_other_config: FableEOtherConfig,
    model_base_config: ModelBaseConfig,
    ed_visit_input_resolution: DataResolution,
) -> str | None:
    """Run a Fable E-other model at the requested ED-visit resolution."""
    _throw_if_backfill(context, daily_partitions_def)

    disease = model_base_config.diseases.current_value
    location = model_base_config.locations.current_value
    run_date = dt.datetime.strptime(context.partition_key, "%Y-%m-%d").date()

    loc_config = model_base_config.get_by_location(location)
    context.log.debug(f"loc_config: '{loc_config}'")

    # We let the user potentially override the basedir, but the subdirectory is
    # locked to the partition date.
    daily_forecast_output_dir: Path = Path(
        loc_config.output_basedir,
        f"{context.partition_key}_forecasts",
    )

    context.log.info(f"fable_e_other_config: '{fable_e_other_config}'")
    context.log.info(f"Will write to: {daily_forecast_output_dir}")
    forecast_fable(
        disease=disease,
        loc=location,
        output_dir=daily_forecast_output_dir,
        n_lookback_days=loc_config.fable_pyrenew_n_lookback_days,
        n_samples=fable_e_other_config.n_samples,
        exclude_last_n_days=loc_config.exclude_last_n_days,
        ed_visit_input_resolution=ed_visit_input_resolution,
        run_date=run_date,
        fail_on_stale_data=loc_config.fail_on_stale_data,
    )


def _run_pyrenew_model(
    context: dg.OpExecutionContext,
    pyrenew_config: PyrenewConfig,
    model_base_config: ModelBaseConfig,
    model_letters: str,
) -> str | None:
    """
    Helper to run Pyrenew models with common arguments.
    """
    _throw_if_backfill(context, daily_partitions_def)

    disease = model_base_config.diseases.current_value
    location = model_base_config.locations.current_value
    run_date = dt.datetime.strptime(context.partition_key, "%Y-%m-%d").date()

    loc_config = model_base_config.get_by_location(location)
    context.log.debug(f"loc_config: '{loc_config}'")

    # We let the user potentially override the basedir, but the subdirectory is
    # locked to the partition date.
    daily_forecast_output_dir: Path = Path(
        loc_config.output_basedir, f"{context.partition_key}_forecasts"
    )

    fit_flags = flags_from_hew_letters(model_letters)
    forecast_flags = flags_from_hew_letters(
        f"{model_letters}{pyrenew_config.additional_forecast_letters}",
        flag_prefix="forecast",
    )
    context.log.info(f"config: '{pyrenew_config}'")
    context.log.info(f"Will write to: {daily_forecast_output_dir}")
    forecast_pyrenew(
        disease=disease,
        loc=location,
        priors_path=PRODUCTION_PRIORS,
        output_dir=daily_forecast_output_dir,
        n_lookback_days=loc_config.fable_pyrenew_n_lookback_days,
        n_chains=pyrenew_config.n_chains,
        n_warmup=pyrenew_config.n_warmup,
        n_samples=pyrenew_config.n_samples,
        exclude_last_n_days=loc_config.exclude_last_n_days,
        rng_key=pyrenew_config.rng_key,
        run_date=run_date,
        fail_on_stale_data=loc_config.fail_on_stale_data,
        **fit_flags,
        **forecast_flags,
    )


def _run_epiautogp_e_pct_epiweekly(
    context: dg.OpExecutionContext,
    epiautogp_e_pct_epiweekly_config: EpiAutoGPEPctEpiweeklyConfig,
    model_base_config: ModelBaseConfig,
) -> None:
    """Run EpiAutoGP directly on epiweekly NSSP percentage data."""
    _throw_if_backfill(context, daily_partitions_def)

    disease = model_base_config.diseases.current_value
    location = model_base_config.locations.current_value
    run_date = dt.datetime.strptime(context.partition_key, "%Y-%m-%d").date()
    loc_config = model_base_config.get_by_location(location)
    context.log.debug(f"loc_config: '{loc_config}'")
    daily_forecast_output_dir = Path(
        loc_config.output_basedir,
        f"{context.partition_key}_forecasts",
    )
    context.log.info(
        f"epiautogp_e_pct_epiweekly_config: '{epiautogp_e_pct_epiweekly_config}'"
    )
    context.log.info(f"Will write to: {daily_forecast_output_dir}")

    forecast_epiautogp(
        disease=disease,
        loc=location,
        output_dir=daily_forecast_output_dir,
        n_lookback_days=loc_config.epiautogp_n_lookback_days,
        target="nssp",
        frequency="epiweekly",
        ed_visit_type="pct",
        exclude_last_n_days=loc_config.exclude_last_n_days,
        n_particles=epiautogp_e_pct_epiweekly_config.n_particles,
        n_mcmc=epiautogp_e_pct_epiweekly_config.n_mcmc,
        n_hmc=epiautogp_e_pct_epiweekly_config.n_hmc,
        n_forecast_draws=epiautogp_e_pct_epiweekly_config.n_forecast_draws,
        smc_data_proportion=epiautogp_e_pct_epiweekly_config.smc_data_proportion,
        n_threads=epiautogp_e_pct_epiweekly_config.n_threads,
        nowcast_source_name="none",
        run_date=run_date,
        fail_on_stale_data=loc_config.fail_on_stale_data,
        logger=context.log,
    )


def get_model_loc_dir(
    context: dg.OpExecutionContext,
    model_base_config: ModelBaseConfig,
) -> Path:
    disease = model_base_config.diseases.current_value
    location = model_base_config.locations.current_value

    loc_config = model_base_config.get_by_location(location)
    context.log.debug(f"loc_config: '{loc_config}'")

    run_date = dt.datetime.strptime(context.partition_key, "%Y-%m-%d").date()
    forecast_window = ForecastWindow(
        report_date=run_date,
        n_lookback_days=loc_config.fable_pyrenew_n_lookback_days,
        exclude_last_n_days=loc_config.exclude_last_n_days,
    )
    model_batch_dir_name = forecast_window.model_batch_dir_name(disease)

    model_loc_dir = Path(
        loc_config.output_basedir,
        f"{context.partition_key}_forecasts",
        model_batch_dir_name,
        "model_runs",
        location,
    )
    return model_loc_dir


def _run_fusion_model(
    context: dg.OpExecutionContext,
    model_base_config: ModelBaseConfig,
    num_model_name,
    other_model_name,
    aggregate_num,
    aggregate_other,
    fusion_model_name,
) -> str | None:
    """
    Helper function to run fusion model.
    """
    _throw_if_backfill(context, daily_partitions_def)
    model_loc_dir = get_model_loc_dir(context, model_base_config)
    create_prop_fusion_model(
        model_run_dir=model_loc_dir,
        num_model_name=num_model_name,
        other_model_name=other_model_name,
        aggregate_num=aggregate_num,
        aggregate_other=aggregate_other,
    )

    fusion_model_fit_dir = Path(model_loc_dir, fusion_model_name)

    make_figures_from_model_fit_dir(fusion_model_fit_dir)

    make_figures_from_model_fit_dir(
        fusion_model_fit_dir,
        save_figs=True,
        save_ci=True,
    )
    run_date = dt.datetime.strptime(context.partition_key, "%Y-%m-%d").date()
    model_fit_dir_to_hub_tbl(fusion_model_fit_dir, report_date=run_date)

    context.log.debug(f"config: '{model_base_config}'")


def _fuse_pyrenew_fable_e_other(
    context,
    model_base_config: ModelBaseConfig,
    pyrenew_model_name,
    epiweekly: bool,
):
    other_model_name = "epiweekly_fable_e_other" if epiweekly else "daily_fable_e_other"
    fusion_model_name = (
        f"prop_epiweekly_aggregated_{pyrenew_model_name}_epiweekly_fable_e_other"
        if epiweekly
        else f"prop_{pyrenew_model_name}_daily_fable_e_other"
    )
    aggregate_num = epiweekly
    _run_fusion_model(
        context=context,
        model_base_config=model_base_config,
        num_model_name=pyrenew_model_name,
        other_model_name=other_model_name,
        aggregate_num=aggregate_num,
        aggregate_other=False,
        fusion_model_name=fusion_model_name,
    )


# ============================================================================
# SCHEDULES AND AUTOMATION CONDITION SENSORS
# ============================================================================


# Custom Automation Condition. Relies on use_user_code_server=True on the sensor
class IsWeekday(dg.AutomationCondition):
    def __init__(self, weekday: int):
        """
        Check if evaluation time falls on a specific weekday.
        This is is a simple evaluation, rather than a stateful operation,
        such as with cron_tick_passed().

        Args:
            weekday: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday,
                    4=Friday, 5=Saturday, 6=Sunday
        """
        self.weekday = weekday
        super().__init__()

    def evaluate(self, context: dg.AutomationContext) -> dg.AutomationResult:
        # If the current weekday is equal to the desired weekday,
        # return the candidate_subset -> a dagster context's "true" case
        if context.evaluation_time.weekday() == self.weekday:
            true_subset = context.candidate_subset
        else:
            true_subset = context.get_empty_subset()

        return dg.AutomationResult(context=context, true_subset=true_subset)

    @property
    def name(self) -> str:
        """Define the label that will appear in the UI"""
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        return f"is_{days[self.weekday].lower()}"


eager_on_wed = (
    # We specifically don't want these to run unless it's Wednesday
    # 0=monday,1=tuesday,2=wednesday,etc.
    # Note this is different from cron which is 1-indexed
    dg.AutomationCondition.eager() & IsWeekday(2)
).with_label("eager_on_wed")


weekly_fable_sensor = dg.AutomationConditionSensorDefinition(
    name="Fable",
    target=dg.AssetSelection.groups("Fable"),
    run_tags=azure_batch_2cpu_execution_config.to_run_tags(),
    use_user_code_server=True,  # allows for custom automation conditions
)

weekly_pyrenew_sensor = dg.AutomationConditionSensorDefinition(
    name="Pyrenew",
    target=dg.AssetSelection.groups("Pyrenew"),
    run_tags=azure_batch_4cpu_execution_config.to_run_tags(),
    use_user_code_server=True,  # allows for custom automation conditions
)

weekly_fusion_sensor = dg.AutomationConditionSensorDefinition(
    name="Fusion",
    target=dg.AssetSelection.groups("Fusion"),
    run_tags=azure_batch_2cpu_execution_config.to_run_tags(),
    use_user_code_server=True,  # allows for custom automation conditions
)

epiautogp_sensor = dg.AutomationConditionSensorDefinition(
    name="EpiAutoGP",
    # add a group_name="EpiAutoGP" to an epiautogp asset to include it
    # in the rules and configuration this sensor provides
    target=dg.AssetSelection.groups("EpiAutoGP"),
    run_tags=azure_batch_64cpu_execution_config.to_run_tags(),
    use_user_code_server=True,  # allows for custom automation conditions
)


# ---------- Shared Asset Decorator Arguments ----------

# It's helpful (and helps reduce DRY issues) to specify some common
# arguments that we give to the asset decorators, as well as some tags

common_asset_args = {
    "partitions_def": daily_partitions_def,  # every asset uses this partitions def
    "retry_policy": dg.RetryPolicy(),  # allow the assets to retry once on failure
}

# Dagster tag keys cannot contain spaces. These tags make it easy to select all
# assets that need rerunning after their corresponding source data changes.
E_DATA_RERUN_TAGS = {"e-data-rerun": ""}
H_DATA_RERUN_TAGS = {"h-data-rerun": ""}
HE_DATA_RERUN_TAGS = E_DATA_RERUN_TAGS | H_DATA_RERUN_TAGS

# ============================================================================
# ASSET DEFINITIONS
# ============================================================================
# These are the core of Dagster - functions that specify data

# ---------- External Asset Specs -------------

# These allow us to model external assets we do not have locally
# while in development. They do not materialize.
# They are replaced with true assets in production where
# other code locations are able to be referenced.

comprehensive_nssp_gold = dg.AssetSpec(
    "comprehensive_nssp_gold",
    partitions_def=daily_partitions_def,
    group_name="Upstream",
)

nhsn_hrd_prelim = dg.AssetSpec(
    "nhsn_hrd_prelim", partitions_def=daily_partitions_def, group_name="Upstream"
)


# ----------------  Forecasts --------------


# Fable E Other
@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=eager_on_wed,
    group_name="Fable",
    ins={"comprehensive_nssp_gold": dg.In(dg.Nothing)},
    tags=E_DATA_RERUN_TAGS,
)
def fable_e_other(
    context: dg.OpExecutionContext,
    fable_e_other_config: FableEOtherConfig,
    model_base_config: ModelBaseConfig,
):
    _run_fable_e_other(
        context,
        fable_e_other_config,
        model_base_config,
        ed_visit_input_resolution="daily",
    )


# Epiweekly Fable E Other
@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=eager_on_wed,
    group_name="Fable",
    ins={"comprehensive_nssp_gold": dg.In(dg.Nothing)},
    tags=E_DATA_RERUN_TAGS,
)
def epiweekly_fable_e_other(
    context: dg.OpExecutionContext,
    fable_e_other_config: FableEOtherConfig,
    model_base_config: ModelBaseConfig,
):
    _run_fable_e_other(
        context,
        fable_e_other_config,
        model_base_config,
        ed_visit_input_resolution="epiweekly",
    )


# Pyrenew E
@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=eager_on_wed,
    group_name="Pyrenew",
    ins={
        "comprehensive_nssp_gold": dg.In(dg.Nothing),
    },
    tags=E_DATA_RERUN_TAGS,
)
def pyrenew_e(
    context: dg.OpExecutionContext,
    pyrenew_config: PyrenewConfig,
    model_base_config: ModelBaseConfig,
    e_model_exclusions: EModelExclusions,
):
    _run_pyrenew_model(context, pyrenew_config, model_base_config, "e")


# Pyrenew H
@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=eager_on_wed,
    group_name="Pyrenew",
    ins={
        "nhsn_hrd_prelim": dg.In(dg.Nothing),
    },
    tags=H_DATA_RERUN_TAGS,
)
def pyrenew_h(
    context: dg.OpExecutionContext,
    pyrenew_config: PyrenewConfig,
    model_base_config: ModelBaseConfig,
):
    _run_pyrenew_model(context, pyrenew_config, model_base_config, "h")


# Pyrenew HE
@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=eager_on_wed,
    group_name="Pyrenew",
    ins={
        "comprehensive_nssp_gold": dg.In(dg.Nothing),
        "nhsn_hrd_prelim": dg.In(dg.Nothing),
    },
    tags=HE_DATA_RERUN_TAGS,
)
def pyrenew_he(
    context: dg.OpExecutionContext,
    pyrenew_config: PyrenewConfig,
    model_base_config: ModelBaseConfig,
    e_model_exclusions: EModelExclusions,
):
    _run_pyrenew_model(context, pyrenew_config, model_base_config, "he")


# EpiAutoGP E-pct (epiweekly)
@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=eager_on_wed,
    group_name="EpiAutoGP",
    ins={"comprehensive_nssp_gold": dg.In(dg.Nothing)},
    tags=E_DATA_RERUN_TAGS,
)
def epiautogp_e_pct_epiweekly(
    context: dg.OpExecutionContext,
    epiautogp_e_pct_epiweekly_config: EpiAutoGPEPctEpiweeklyConfig,
    model_base_config: ModelBaseConfig,
):
    _run_epiautogp_e_pct_epiweekly(
        context,
        epiautogp_e_pct_epiweekly_config,
        model_base_config,
    )


# ---------- Fusion Forecasts ----------


@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=dg.AutomationCondition.eager(),
    group_name="Fusion",
    ins={"pyrenew_e": dg.In(dg.Nothing), "fable_e_other": dg.In(dg.Nothing)},
    tags=E_DATA_RERUN_TAGS,
)
def fuse_pyrenew_e_ts(
    context: dg.OpExecutionContext,
    model_base_config: ModelBaseConfig,
    e_model_exclusions: EModelExclusions,
):
    _fuse_pyrenew_fable_e_other(
        context,
        model_base_config,
        pyrenew_model_name="pyrenew_e",
        epiweekly=False,
    )


@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=dg.AutomationCondition.eager(),
    group_name="Fusion",
    ins={
        "pyrenew_e": dg.In(dg.Nothing),
        "epiweekly_fable_e_other": dg.In(dg.Nothing),
    },
    tags=E_DATA_RERUN_TAGS,
)
def fuse_pyrenew_e_ts_epiweekly(
    context: dg.OpExecutionContext,
    model_base_config: ModelBaseConfig,
    e_model_exclusions: EModelExclusions,
):
    _fuse_pyrenew_fable_e_other(
        context,
        model_base_config,
        pyrenew_model_name="pyrenew_e",
        epiweekly=True,
    )


@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=dg.AutomationCondition.eager(),
    group_name="Fusion",
    ins={"pyrenew_he": dg.In(dg.Nothing), "fable_e_other": dg.In(dg.Nothing)},
    tags=HE_DATA_RERUN_TAGS,
)
def fuse_pyrenew_he_ts(
    context: dg.OpExecutionContext,
    model_base_config: ModelBaseConfig,
    e_model_exclusions: EModelExclusions,
):
    _fuse_pyrenew_fable_e_other(
        context,
        model_base_config,
        pyrenew_model_name="pyrenew_he",
        epiweekly=False,
    )


@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=dg.AutomationCondition.eager(),
    group_name="Fusion",
    ins={
        "pyrenew_he": dg.In(dg.Nothing),
        "epiweekly_fable_e_other": dg.In(dg.Nothing),
    },
    tags=HE_DATA_RERUN_TAGS,
)
def fuse_pyrenew_he_ts_epiweekly(
    context: dg.OpExecutionContext,
    model_base_config: ModelBaseConfig,
    e_model_exclusions: EModelExclusions,
):
    _fuse_pyrenew_fable_e_other(
        context,
        model_base_config,
        pyrenew_model_name="pyrenew_he",
        epiweekly=True,
    )


# ---------- Postprocessing Forecast Batches ----------


@dg.asset(
    deps=[
        "fuse_pyrenew_e_ts",
        "fuse_pyrenew_e_ts_epiweekly",
        "fuse_pyrenew_he_ts",
        "fuse_pyrenew_he_ts_epiweekly",
        "pyrenew_h",
        "epiautogp_e_pct_epiweekly",
    ],
    partitions_def=daily_partitions_def,
    # Runs when any dependency has been updated as long as at least one exists
    automation_condition=(
        dg.AutomationCondition.eager().replace(
            old=~dg.AutomationCondition.any_deps_missing(),
            new=dg.AutomationCondition.any_deps_match(
                ~dg.AutomationCondition.missing()
                | dg.AutomationCondition.will_be_requested()
            ),
        )
    ).with_label("postprocess_custom_eager"),
    group_name="Fusion",  # included with the fusion assets, but should be separate
    retry_policy=dg.RetryPolicy(),  # allow the asset to retry once on failure
    tags=HE_DATA_RERUN_TAGS,
)
def postprocess_forecasts(
    context: dg.AssetExecutionContext,
    config: PostProcessConfig,
):
    """
    Postprocess forecast batches
    """

    _throw_if_backfill(context, daily_partitions_def)

    daily_forecast_output_dir: Path = Path(
        config.output_basedir, f"{context.partition_key}_forecasts"
    )

    context.log.info(f"config: '{config}'")
    postprocess(
        base_forecast_dir=daily_forecast_output_dir,
        diseases=config.postprocess_diseases,
        skip_existing=config.skip_existing,
        local_copy_dir=daily_forecast_output_dir,
    )


# ============================================================================
# JOBS AND OPS
# These can create images.
# ============================================================================

update_script_url = (
    # repo
    "https://raw.githubusercontent.com/CDCgov/cfa-dagster/"
    # ref
    "refs/heads/main/"
    # file
    "scripts/update_code_location.py"
)

prod_server_image = f"{registry}/{local_workdir.name}:latest"


# Plain helper so the deploy logic can be reused directly (e.g. from build_image_op)
# without invoking the op imperatively, which Dagster does not support.
def _refresh_prod_server_image(logger, image_to_deploy: str):
    logger.info(f"Deploying {image_to_deploy} to the dagster prod server.")
    subprocess.run(
        ["uv", "run", update_script_url, "--registry_image", image_to_deploy],
        check=True,
    )


# Used both in the schedule and in the build_image_op
@dg.op
def refresh_prod_server_image_op(context: dg.OpExecutionContext, image_to_deploy: str):
    """
    Deploys the dagster image to the prod server. Can deploy a working branch's image or the latest image.
    """
    _refresh_prod_server_image(context.log, image_to_deploy)


refresh_prod_server_image_config = dg.RunConfig(
    ops={
        "refresh_prod_server_image_op": {
            "inputs": {
                "image_to_deploy": prod_server_image,
            }
        }
    },
    # configure this job to run on your computer
    execution=basic_execution_config.to_run_config(),
)


@dg.job(
    description=(
        "Standalone job that simply (re)deploys the latest image to the prod server (you can override the tag if necessary). "
        "Note - the build_image job that is available in dev can be passed a flag that executes this when complete. "
    ),
    config=refresh_prod_server_image_config,
    executor_def=dynamic_executor(),
)
def refresh_prod_server_image():
    refresh_prod_server_image_op()


E2E_LOCATIONS = ["CA", "US"]


def e2e_config() -> dg.RunConfig:
    return dg.RunConfig(
        resources={
            "model_base_config": ModelBaseConfig(
                locations=GraphDimension(E2E_LOCATIONS)
            ),
        },
        execution=azure_batch_4cpu_execution_config.to_run_config(),
    )


def e2e_json() -> str:
    return json.dumps(e2e_config().to_config_dict())


end_to_end = dg.define_asset_job(
    name="end_to_end",
    selection=dg.AssetSelection.groups("Fable", "Pyrenew", "EpiAutoGP", "Fusion"),
    config=e2e_config(),
)


@dg.schedule(
    cron_schedule="00 23 * * TUE",
    execution_timezone=tz,
    job_name="refresh_prod_server_image",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def reset_prod_server_image_for_wednesday():
    return dg.RunRequest(run_config=refresh_prod_server_image_config)


# These are only used in dev - they should not appear on the production webserver
if not is_production:
    # Build and Push Image ---------------------------

    @dg.op
    def build_image_op(
        context: dg.OpExecutionContext,
        should_push: bool,
        should_deploy_to_prod: bool,
        dockerfile_path: str,
        build_context: str,
        image: str,
    ):
        """
        Builds the image used by dagster. Requires that your VM be registered with an Azure managed identity.

        should_push: bool - should the image be pushed to the Container Registry?
        should_deploy_to_prod: bool - should the prod server be updated with the newest image? (usually you do not want to do this)
        dockerfile_path: str - where is the Dockerfile located locally? (has a default)
        build_context: str - where should we build from? (has a default)
        image: str - the full name (including registry and tag) of the image
        """

        build_command = [
            "docker",
            "buildx",
            "build",
            "-t",
            image,
            "-f",
            dockerfile_path,
            build_context,
        ]

        if should_push:
            subprocess.run(
                ["az", "login", "--identity"],
                check=True,
            )
            subprocess.run(["az", "acr", "login", "-n", registry], check=True)
            build_command.append("--push")

        context.log.info(f"Running {' '.join(build_command)}")
        subprocess.run(build_command, check=True)

        if should_deploy_to_prod:
            _refresh_prod_server_image(context.log, image_to_deploy=image)

    @dg.job(
        description=(
            "Build the container image used by dagster to run this project's asset cfa.stf.routine."
            "Run after making any change and before running the cfa.stf.routine."
        ),
        config=dg.RunConfig(
            ops={
                "build_image_op": {
                    "inputs": {
                        "should_push": True,
                        "should_deploy_to_prod": False,
                        "dockerfile_path": f"{local_workdir}/Dockerfile",
                        # the build context should be the top level of the repo
                        "build_context": str(local_workdir),
                        "image": image,
                    }
                }
            },
            # configure this job to run on your computer
            execution=basic_execution_config.to_run_config(),
        ),
        executor_def=dynamic_executor(),
    )
    def build_image():
        build_image_op()

    # Explore the image you built as it will be run with dagster ---------------------------

    @dg.op
    def explore_image_op(
        context: dg.OpExecutionContext,
    ):
        """
        Allows you to run the container you previously built and explore the filesystem that will be used by dagster.
        """
        context.log.info(
            "Check the terminal from which you ran the webserver to interact; stdout from your terminal will appear below."
        )
        explore_cmd = (
            ["docker", "run", "-it"]
            + ["-v", local_output_mount]
            + ["--rm", image, "bash"]
        )
        subprocess.run(explore_cmd, check=True)

    @dg.job(
        description=(
            "Interactively navigate the filesystem of your last-built container, "
            "as it would be used in Docker or Azure Batch execution."
        ),
        executor_def=dg.in_process_executor,
    )
    def explore_image():
        explore_image_op()

# ============================================================================
# DAGSTER DEFINITIONS OBJECT
# ============================================================================
# This code allows us to collect all of the above definitions into a single
# Definitions object for Dagster to read. By doing this, we can keep our
# Dagster code in a single file instead of splitting it across multiple files.

# collect Dagster definitions from the current file
collected_defs = collect_definitions(globals())

# Set Azure HTTP Logging Level
# this will limit excessive IO logs in stderr
# for any assets making azure http requests
azure_http_logger = logging.getLogger(
    "azure.core.pipeline.policies.http_logging_policy"
)
azure_http_logger.setLevel(logging.WARNING)

# Create Definitions object
defs = dg.Definitions(
    **collected_defs,
    resources={
        # These IOManagers let Dagster serialize asset outputs and store them
        # in Azure to pass between assets
        "io_manager": ADLS2PickleIOManager(),
        # Shared resources for model assets
        "model_base_config": ModelBaseConfig(),
        "pyrenew_config": PyrenewConfig(),
        "epiautogp_e_pct_epiweekly_config": EpiAutoGPEPctEpiweeklyConfig(),
        "fable_e_other_config": FableEOtherConfig(),
        "e_model_exclusions": EModelExclusions(),
        "w_model_exclusions": WModelExclusions(),
    },
    executor=dynamic_executor(
        default_config=azure_batch_4cpu_execution_config,
        # default_config=basic_execution_config,
        # default_config=docker_execution_config,
        alternate_configs=[
            basic_execution_config,
            docker_execution_config,
        ],
    ),
)
