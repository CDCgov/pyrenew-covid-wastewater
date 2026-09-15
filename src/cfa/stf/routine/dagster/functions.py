import datetime as dt
from pathlib import Path

import dagster as dg
from pyrenew_multisignal.hew.utils import flags_from_hew_letters

from cfa.stf.routine._paths import PRODUCTION_PRIORS
from cfa.stf.routine.dagster.config import (
    FableEOtherConfig,
    ModelBaseConfig,
    PyrenewConfig,
    daily_partitions_def,
)
from cfa.stf.routine.data.data_access import DataResolution
from cfa.stf.routine.fable.forecast_fable import main as forecast_fable
from cfa.stf.routine.pyrenew_hew.forecast_pyrenew import main as forecast_pyrenew
from cfa.stf.routine.utils.date_utils import calculate_training_dates
from cfa.stf.routine.utils.directory_utils import get_model_batch_dir_name
from cfa.stf.routine.utils.prop_utils import create_prop_fusion_model
from cfa.stf.routine.utils.r_utils import (
    make_figures_from_model_fit_dir,
    model_fit_dir_to_hub_tbl,
)


def _throw_if_backfill(
    context: dg.OpExecutionContext | dg.AssetExecutionContext,
    partition_def: dg.PartitionsDefinition,
):
    if context.partition_key != partition_def.get_last_partition_key():
        raise RuntimeError("STF forecast models do not support backfills")


def _run_fable_e_other(
    context: dg.OpExecutionContext,
    fable_e_other_config: FableEOtherConfig,
    model_base_config: ModelBaseConfig,
    ed_visit_input_resolution: DataResolution,
) -> None:
    _throw_if_backfill(context, daily_partitions_def)
    disease = model_base_config.diseases.current_value
    location = model_base_config.locations.current_value
    run_date = dt.datetime.strptime(context.partition_key, "%Y-%m-%d").date()
    daily_forecast_output_dir = Path(
        model_base_config.output_basedir, f"{context.partition_key}_forecasts"
    )
    loc_config = model_base_config.get_by_location(location)
    context.log.info(f"Will write to: {daily_forecast_output_dir}")
    forecast_fable(
        disease=disease,
        loc=location,
        output_dir=daily_forecast_output_dir,
        n_training_days=model_base_config.n_training_days,
        n_forecast_days=28,
        n_samples=fable_e_other_config.n_samples,
        exclude_last_n_days=loc_config.exclude_last_n_days,
        ed_visit_input_resolution=ed_visit_input_resolution,
        run_date=run_date,
        fail_on_stale_data=model_base_config.fail_on_stale_data,
    )


def _run_pyrenew_model(
    context: dg.OpExecutionContext,
    pyrenew_config: PyrenewConfig,
    model_base_config: ModelBaseConfig,
    model_letters: str,
) -> None:
    _throw_if_backfill(context, daily_partitions_def)
    disease = model_base_config.diseases.current_value
    location = model_base_config.locations.current_value
    run_date = dt.datetime.strptime(context.partition_key, "%Y-%m-%d").date()
    daily_forecast_output_dir = Path(
        model_base_config.output_basedir, f"{context.partition_key}_forecasts"
    )
    fit_flags = flags_from_hew_letters(model_letters)
    forecast_flags = flags_from_hew_letters(
        f"{model_letters}{pyrenew_config.additional_forecast_letters}",
        flag_prefix="forecast",
    )
    loc_config = model_base_config.get_by_location(location)
    context.log.info(f"Will write to: {daily_forecast_output_dir}")
    forecast_pyrenew(
        disease=disease,
        loc=location,
        priors_path=PRODUCTION_PRIORS,
        output_dir=daily_forecast_output_dir,
        n_training_days=model_base_config.n_training_days,
        n_forecast_days=28,
        n_chains=pyrenew_config.n_chains,
        n_warmup=pyrenew_config.n_warmup,
        n_samples=pyrenew_config.n_samples,
        exclude_last_n_days=loc_config.exclude_last_n_days,
        rng_key=pyrenew_config.rng_key,
        run_date=run_date,
        fail_on_stale_data=model_base_config.fail_on_stale_data,
        **fit_flags,
        **forecast_flags,
    )


def get_model_loc_dir(
    context: dg.OpExecutionContext, model_base_config: ModelBaseConfig
) -> Path:
    disease = model_base_config.diseases.current_value
    location = model_base_config.locations.current_value
    loc_config = model_base_config.get_by_location(location)
    run_date = dt.datetime.strptime(context.partition_key, "%Y-%m-%d").date()
    first_training_date, last_training_date = calculate_training_dates(
        report_date=run_date,
        n_training_days=model_base_config.n_training_days,
        exclude_last_n_days=loc_config.exclude_last_n_days,
        logger=context.log,
    )
    model_batch_dir_name = get_model_batch_dir_name(
        disease=disease,
        report_date=run_date,
        first_training_date=first_training_date,
        last_training_date=last_training_date,
    )
    return Path(
        model_base_config.output_basedir,
        f"{context.partition_key}_forecasts",
        model_batch_dir_name,
        "model_runs",
        location,
    )


def _run_fusion_model(
    context: dg.OpExecutionContext,
    model_base_config: ModelBaseConfig,
    num_model_name: str,
    other_model_name: str,
    aggregate_num: bool,
    aggregate_other: bool,
    fusion_model_name: str,
) -> None:
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
    make_figures_from_model_fit_dir(fusion_model_fit_dir, save_figs=True, save_ci=True)
    model_fit_dir_to_hub_tbl(fusion_model_fit_dir)


def _fuse_pyrenew_fable_e_other(
    context: dg.OpExecutionContext,
    model_base_config: ModelBaseConfig,
    pyrenew_model_name: str,
    epiweekly: bool,
) -> None:
    other_model_name = "epiweekly_fable_e_other" if epiweekly else "daily_fable_e_other"
    fusion_model_name = (
        f"prop_epiweekly_aggregated_{pyrenew_model_name}_epiweekly_fable_e_other"
        if epiweekly
        else f"prop_{pyrenew_model_name}_daily_fable_e_other"
    )
    _run_fusion_model(
        context=context,
        model_base_config=model_base_config,
        num_model_name=pyrenew_model_name,
        other_model_name=other_model_name,
        aggregate_num=epiweekly,
        aggregate_other=False,
        fusion_model_name=fusion_model_name,
    )
