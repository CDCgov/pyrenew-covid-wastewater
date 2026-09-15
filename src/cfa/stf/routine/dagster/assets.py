from pathlib import Path

import dagster as dg
from cfa_dagster import dynamic_graph_asset

from cfa.stf.routine.dagster.automation import eager_on_wed
from cfa.stf.routine.dagster.config import (
    EModelExclusions,
    FableEOtherConfig,
    ModelBaseConfig,
    PostProcessConfig,
    PyrenewConfig,
    daily_partitions_def,
)
from cfa.stf.routine.dagster.functions import (
    _fuse_pyrenew_fable_e_other,
    _run_fable_e_other,
    _run_pyrenew_model,
    _throw_if_backfill,
)
from cfa.stf.routine.utils.postprocess_forecast_batches import main as postprocess

common_asset_args = {
    "partitions_def": daily_partitions_def,
    "retry_policy": dg.RetryPolicy(),
}
E_DATA_RERUN_TAGS = {"e-data-rerun": ""}
H_DATA_RERUN_TAGS = {"h-data-rerun": ""}
HE_DATA_RERUN_TAGS = E_DATA_RERUN_TAGS | H_DATA_RERUN_TAGS

nssp_gold_v1 = dg.AssetSpec(
    "nssp_gold_v1", partitions_def=daily_partitions_def, group_name="Upstream"
)
nhsn_hrd_prelim = dg.AssetSpec(
    "nhsn_hrd_prelim", partitions_def=daily_partitions_def, group_name="Upstream"
)


@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=eager_on_wed,
    group_name="Fable",
    ins={"nssp_gold_v1": dg.In(dg.Nothing)},
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


@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=eager_on_wed,
    group_name="Fable",
    ins={"nssp_gold_v1": dg.In(dg.Nothing)},
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


@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=eager_on_wed,
    group_name="Pyrenew",
    ins={"nssp_gold_v1": dg.In(dg.Nothing)},
    tags=E_DATA_RERUN_TAGS,
)
def pyrenew_e(
    context: dg.OpExecutionContext,
    pyrenew_config: PyrenewConfig,
    model_base_config: ModelBaseConfig,
    e_model_exclusions: EModelExclusions,
):
    _run_pyrenew_model(context, pyrenew_config, model_base_config, "e")


@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=eager_on_wed,
    group_name="Pyrenew",
    ins={"nhsn_hrd_prelim": dg.In(dg.Nothing)},
    tags=H_DATA_RERUN_TAGS,
)
def pyrenew_h(
    context: dg.OpExecutionContext,
    pyrenew_config: PyrenewConfig,
    model_base_config: ModelBaseConfig,
):
    _run_pyrenew_model(context, pyrenew_config, model_base_config, "h")


@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=eager_on_wed,
    group_name="Pyrenew",
    ins={"nssp_gold_v1": dg.In(dg.Nothing), "nhsn_hrd_prelim": dg.In(dg.Nothing)},
    tags=HE_DATA_RERUN_TAGS,
)
def pyrenew_he(
    context: dg.OpExecutionContext,
    pyrenew_config: PyrenewConfig,
    model_base_config: ModelBaseConfig,
    e_model_exclusions: EModelExclusions,
):
    _run_pyrenew_model(context, pyrenew_config, model_base_config, "he")


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
    _fuse_pyrenew_fable_e_other(context, model_base_config, "pyrenew_e", False)


@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=dg.AutomationCondition.eager(),
    group_name="Fusion",
    ins={"pyrenew_e": dg.In(dg.Nothing), "epiweekly_fable_e_other": dg.In(dg.Nothing)},
    tags=E_DATA_RERUN_TAGS,
)
def fuse_pyrenew_e_ts_epiweekly(
    context: dg.OpExecutionContext,
    model_base_config: ModelBaseConfig,
    e_model_exclusions: EModelExclusions,
):
    _fuse_pyrenew_fable_e_other(context, model_base_config, "pyrenew_e", True)


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
    _fuse_pyrenew_fable_e_other(context, model_base_config, "pyrenew_he", False)


@dynamic_graph_asset(
    **common_asset_args,
    automation_condition=dg.AutomationCondition.eager(),
    group_name="Fusion",
    ins={"pyrenew_he": dg.In(dg.Nothing), "epiweekly_fable_e_other": dg.In(dg.Nothing)},
    tags=HE_DATA_RERUN_TAGS,
)
def fuse_pyrenew_he_ts_epiweekly(
    context: dg.OpExecutionContext,
    model_base_config: ModelBaseConfig,
    e_model_exclusions: EModelExclusions,
):
    _fuse_pyrenew_fable_e_other(context, model_base_config, "pyrenew_he", True)


@dg.asset(
    deps=[
        "fuse_pyrenew_e_ts",
        "fuse_pyrenew_e_ts_epiweekly",
        "fuse_pyrenew_he_ts",
        "fuse_pyrenew_he_ts_epiweekly",
        "pyrenew_h",
    ],
    partitions_def=daily_partitions_def,
    automation_condition=(
        dg.AutomationCondition.eager().replace(
            old=~dg.AutomationCondition.any_deps_missing(),
            new=dg.AutomationCondition.any_deps_match(
                ~dg.AutomationCondition.missing()
                | dg.AutomationCondition.will_be_requested()
            ),
        )
    ).with_label("postprocess_custom_eager"),
    group_name="Fusion",
    retry_policy=dg.RetryPolicy(),
    tags=HE_DATA_RERUN_TAGS,
)
def postprocess_forecasts(context: dg.AssetExecutionContext, config: PostProcessConfig):
    _throw_if_backfill(context, daily_partitions_def)
    daily_forecast_output_dir = Path(
        config.output_basedir, f"{context.partition_key}_forecasts"
    )
    context.log.info(f"config: '{config}'")
    postprocess(
        base_forecast_dir=daily_forecast_output_dir,
        diseases=config.postprocess_diseases,
        skip_existing=config.skip_existing,
        local_copy_dir=daily_forecast_output_dir,
    )
