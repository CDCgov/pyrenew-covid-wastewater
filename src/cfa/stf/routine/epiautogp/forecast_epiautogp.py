import datetime as dt
import logging
from pathlib import Path
from typing import Literal, get_args

from cfa.stf.data import get_nnh_right_truncation_pmf

from cfa.stf.routine._paths import EPIAUTOGP_DIR
from cfa.stf.routine.data.data_access import DataResolution, ForecastSourceName
from cfa.stf.routine.data.hubverse_nowcast import HubverseNowcast
from cfa.stf.routine.data.nowcast import NowcastSource
from cfa.stf.routine.epiautogp.config import EpiAutoGPConfig
from cfa.stf.routine.epiautogp.prep_epiautogp_data import (
    _validate_epiautogp_parameters,
    convert_to_epiautogp_json,
)
from cfa.stf.routine.epiautogp.reporting_delay_nowcast import ReportingDelayNowcast
from cfa.stf.routine.forecast_pipeline import ForecastPipeline
from cfa.stf.routine.forecast_run import ForecastRun
from cfa.stf.routine.utils.date_utils import parse_exclude_date_ranges
from cfa.stf.routine.utils.language_utils import run_julia_script

_FIT_SCRIPT = Path(__file__).parent / "fit_epiautogp.jl"
NowcastSourceName = Literal["none", "reporting-delay", "hubverse"]
VALID_NOWCAST_SOURCE_NAMES: tuple[str, ...] = get_args(NowcastSourceName)


def run_epiautogp_forecast(
    json_input_path: Path,
    model_dir: Path,
    *,
    n_particles: int,
    n_mcmc: int,
    n_hmc: int,
    n_forecast_draws: int,
    transformation: str,
    smc_data_proportion: float,
    n_threads: int | str,
) -> None:
    """Run the repository's Julia adapter around NowcastAutoGP."""
    model_dir.mkdir(parents=True, exist_ok=True)
    args = [
        f"--json-input={json_input_path}",
        f"--output-dir={model_dir}",
        f"--n-particles={n_particles}",
        f"--n-mcmc={n_mcmc}",
        f"--n-hmc={n_hmc}",
        f"--n-forecast-draws={n_forecast_draws}",
        f"--transformation={transformation}",
        f"--smc-data-proportion={smc_data_proportion}",
    ]
    run_julia_script(
        _FIT_SCRIPT,
        args,
        executor_flags=[
            f"--project={EPIAUTOGP_DIR}",
            f"--threads={n_threads}",
        ],
        function_name="run_epiautogp_forecast",
        capture_output=False,
    )


def _build_reporting_delay_nowcast(
    *,
    forecast_run: ForecastRun,
    reporting_delay_pmf: list[float] | None,
) -> ReportingDelayNowcast:
    """Build a reporting-delay source, fetching the PMF when necessary."""
    if reporting_delay_pmf is None:
        reporting_delay_pmf = get_nnh_right_truncation_pmf(
            state_abb=forecast_run.loc,
            disease=forecast_run.disease,
            as_of=forecast_run.report_date,
            reference_date=forecast_run.report_date,
        )
    return ReportingDelayNowcast(reporting_delay_pmf=reporting_delay_pmf)


def _resolve_nowcast_source(
    *,
    forecast_run: ForecastRun,
    config: EpiAutoGPConfig,
    nowcast_source_name: NowcastSourceName,
    reporting_delay_pmf: list[float] | None = None,
    hubverse_nowcast_dir: Path | str | None = None,
) -> NowcastSource | None:
    """Resolve the requested nowcast source for one EpiAutoGP run."""
    if reporting_delay_pmf is not None and hubverse_nowcast_dir is not None:
        raise ValueError(
            "reporting_delay_pmf and hubverse_nowcast_dir are mutually exclusive."
        )

    match nowcast_source_name:
        case "none":
            return None
        case "reporting-delay":
            ReportingDelayNowcast.ensure_applicable(config=config)
            return _build_reporting_delay_nowcast(
                forecast_run=forecast_run,
                reporting_delay_pmf=reporting_delay_pmf,
            )
        case "hubverse":
            HubverseNowcast.ensure_applicable(config=config)
            if hubverse_nowcast_dir is None:
                raise ValueError(
                    "hubverse_nowcast_dir is required when Hubverse nowcasting is "
                    "requested."
                )
            return HubverseNowcast(
                containing_dir=Path(hubverse_nowcast_dir),
                forecast_run=forecast_run,
                config=config,
            )
        case _:
            raise ValueError(
                f"nowcast_source_name must be one of "
                f"{list(VALID_NOWCAST_SOURCE_NAMES)}, got {nowcast_source_name!r}"
            )


class EpiAutoGPPipeline(ForecastPipeline):
    """Single-location EpiAutoGP forecast pipeline."""

    def __init__(
        self,
        *,
        target: str,
        frequency: str,
        ed_visit_type: str = "observed",
        exclude_date_ranges: list[tuple[dt.date, dt.date]] | None = None,
        n_particles: int = 24,
        n_mcmc: int = 100,
        n_hmc: int = 50,
        n_forecast_draws: int = 2000,
        smc_data_proportion: float = 0.1,
        n_threads: int | str = "auto",
        nowcast_source_name: NowcastSourceName = "none",
        reporting_delay_pmf: list[float] | None = None,
        hubverse_nowcast_dir: Path | str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.config = EpiAutoGPConfig(
            target=target,
            frequency=frequency,
            ed_visit_type=ed_visit_type,
            exclude_date_ranges=exclude_date_ranges,
        )
        self.n_particles = n_particles
        self.n_mcmc = n_mcmc
        self.n_hmc = n_hmc
        self.n_forecast_draws = n_forecast_draws
        self.smc_data_proportion = smc_data_proportion
        self.n_threads = n_threads
        self.nowcast_source_name = nowcast_source_name
        self.reporting_delay_pmf = reporting_delay_pmf
        self.hubverse_nowcast_dir = hubverse_nowcast_dir

    @property
    def model_name(self) -> str:
        model_name = f"epiautogp_{self.config.target}_{self.config.frequency}"
        if self.config.ed_visit_type == "pct":
            model_name += "_pct"
        if self.config.ed_visit_type == "other":
            model_name += "_other"
        return model_name

    @property
    def sources(self) -> set[ForecastSourceName]:
        return {self.config.target}

    @property
    def ed_visit_input_resolution(self) -> DataResolution:
        return self.config.frequency

    @property
    def minimum_exclude_last_n_days(self) -> int:
        return 3 if self.nowcast_source_name == "none" else 0

    def validate_configuration(self) -> None:
        _validate_epiautogp_parameters(
            self.config.target,
            self.config.frequency,
            self.config.ed_visit_type,
        )

    def prepare_model_artifacts(self, run: ForecastRun) -> None:
        nowcast_source = _resolve_nowcast_source(
            forecast_run=run,
            config=self.config,
            nowcast_source_name=self.nowcast_source_name,
            reporting_delay_pmf=self.reporting_delay_pmf,
            hubverse_nowcast_dir=self.hubverse_nowcast_dir,
        )
        self.logger.info("Converting data to EpiAutoGP JSON format...")
        convert_to_epiautogp_json(
            forecast_run=run,
            config=self.config,
            nowcast_source=nowcast_source,
            logger=self.logger,
        )

    def run_model(self, run: ForecastRun) -> None:
        transformation = (
            "percentage" if self.config.ed_visit_type == "pct" else "boxcox"
        )
        self.logger.info("Performing EpiAutoGP forecasting...")
        run_epiautogp_forecast(
            json_input_path=run.model_dir / f"{run.model_name}_input.json",
            model_dir=run.model_dir,
            n_particles=self.n_particles,
            n_mcmc=self.n_mcmc,
            n_hmc=self.n_hmc,
            n_forecast_draws=self.n_forecast_draws,
            transformation=transformation,
            smc_data_proportion=self.smc_data_proportion,
            n_threads=self.n_threads,
        )


def main(
    disease: str,
    run_date: dt.date,
    loc: str,
    output_dir: Path | str,
    n_lookback_days: int | None,
    target: str,
    frequency: str,
    ed_visit_type: str = "observed",
    exclude_last_n_days: int = 0,
    exclude_date_ranges: str | None = None,
    n_particles: int = 24,
    n_mcmc: int = 100,
    n_hmc: int = 50,
    n_forecast_draws: int = 2000,
    smc_data_proportion: float = 0.1,
    n_threads: int | str = "auto",
    nowcast_source_name: NowcastSourceName = "none",
    reporting_delay_pmf: list[float] | None = None,
    hubverse_nowcast_dir: Path | str | None = None,
    fail_on_stale_data: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Run the complete EpiAutoGP pipeline for one location."""
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    parsed_exclude_date_ranges = parse_exclude_date_ranges(exclude_date_ranges)
    if parsed_exclude_date_ranges:
        logger.info(
            f"Excluding {len(parsed_exclude_date_ranges)} date range(s): "
            f"{parsed_exclude_date_ranges}"
        )

    EpiAutoGPPipeline(
        disease=disease,
        loc=loc,
        target=target,
        frequency=frequency,
        ed_visit_type=ed_visit_type,
        output_dir=output_dir,
        n_lookback_days=n_lookback_days,
        exclude_last_n_days=exclude_last_n_days,
        exclude_date_ranges=parsed_exclude_date_ranges,
        logger=logger,
        nowcast_source_name=nowcast_source_name,
        reporting_delay_pmf=reporting_delay_pmf,
        hubverse_nowcast_dir=hubverse_nowcast_dir,
        run_date=run_date,
        fail_on_stale_data=fail_on_stale_data,
        n_particles=n_particles,
        n_mcmc=n_mcmc,
        n_hmc=n_hmc,
        n_forecast_draws=n_forecast_draws,
        smc_data_proportion=smc_data_proportion,
        n_threads=n_threads,
    ).execute()
