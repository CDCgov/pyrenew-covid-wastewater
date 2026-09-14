import datetime as dt
import json
import logging
from dataclasses import replace
from unittest.mock import patch

import polars as pl
import pytest
from tests.factories import make_test_forecast_run

from cfa.stf.routine.data.hubverse_nowcast import HubverseNowcast
from cfa.stf.routine.epiautogp.config import EpiAutoGPConfig
from cfa.stf.routine.epiautogp.forecast_epiautogp import (
    EpiAutoGPPipeline,
    _resolve_nowcast_source,
    run_epiautogp_forecast,
)
from cfa.stf.routine.epiautogp.reporting_delay_nowcast import ReportingDelayNowcast


def _run(tmp_path, *, sources=("nssp", "nhsn")):
    return make_test_forecast_run(
        output_dir=tmp_path,
        model_name="epiautogp_nssp_daily",
        sources=sources,
    )


def _pipeline(tmp_path, **overrides):
    kwargs = {
        "disease": "covid",
        "loc": "CA",
        "target": "nssp",
        "frequency": "daily",
        "ed_visit_type": "observed",
        "output_dir": tmp_path,
        "n_lookback_days": 90,
        "run_date": dt.date(2024, 12, 20),
        "logger": logging.getLogger("test-epiautogp-pipeline"),
    }
    kwargs.update(overrides)
    return EpiAutoGPPipeline(**kwargs)


@pytest.mark.parametrize(
    ("target", "frequency", "ed_visit_type", "expected"),
    [
        ("nssp", "daily", "observed", "epiautogp_nssp_daily"),
        ("nssp", "epiweekly", "pct", "epiautogp_nssp_epiweekly_pct"),
        ("nssp", "daily", "other", "epiautogp_nssp_daily_other"),
        ("nhsn", "epiweekly", "observed", "epiautogp_nhsn_epiweekly"),
    ],
)
def test_pipeline_declares_model_name_and_source(
    tmp_path, target, frequency, ed_visit_type, expected
):
    pipeline = _pipeline(
        tmp_path,
        target=target,
        frequency=frequency,
        ed_visit_type=ed_visit_type,
    )

    assert pipeline.model_name == expected
    assert pipeline.sources == {target}
    assert pipeline.ed_visit_input_resolution == frequency


def test_pipeline_validates_configuration_before_loading(tmp_path):
    pipeline = _pipeline(tmp_path, target="nhsn", frequency="daily")
    with pytest.raises(ValueError, match="only available in epiweekly"):
        pipeline.validate_configuration()


@pytest.mark.parametrize(
    ("nowcast_source_name", "expected"),
    [("none", 3), ("reporting-delay", 0), ("hubverse", 0)],
)
def test_pipeline_declares_minimum_exclusion(tmp_path, nowcast_source_name, expected):
    pipeline = _pipeline(tmp_path, nowcast_source_name=nowcast_source_name)

    assert pipeline.minimum_exclude_last_n_days == expected


@patch("cfa.stf.routine.epiautogp.forecast_epiautogp.convert_to_epiautogp_json")
def test_prepare_model_artifacts_resolves_nowcast_without_mutating_state(
    mock_convert,
    tmp_path,
):
    pipeline = _pipeline(
        tmp_path,
        nowcast_source_name="reporting-delay",
        reporting_delay_pmf=[1.0],
    )
    run = _run(tmp_path)

    pipeline.prepare_model_artifacts(run)

    assert mock_convert.call_args.kwargs["forecast_run"] is run
    assert mock_convert.call_args.kwargs["config"] is pipeline.config
    assert isinstance(
        mock_convert.call_args.kwargs["nowcast_source"],
        ReportingDelayNowcast,
    )
    assert not hasattr(pipeline, "nowcast_source")


@patch("cfa.stf.routine.epiautogp.forecast_epiautogp.run_epiautogp_forecast")
def test_run_model_passes_prepared_input_and_model_options(
    mock_forecast,
    tmp_path,
):
    pipeline = _pipeline(
        tmp_path,
        ed_visit_type="pct",
        n_particles=2,
        n_mcmc=3,
        n_hmc=4,
        n_forecast_draws=5,
        smc_data_proportion=0.2,
        n_threads=6,
    )
    run = _run(tmp_path)

    pipeline.run_model(run)

    assert mock_forecast.call_args.kwargs == {
        "json_input_path": run.model_dir / f"{run.model_name}_input.json",
        "model_dir": run.model_dir,
        "n_particles": 2,
        "n_mcmc": 3,
        "n_hmc": 4,
        "n_forecast_draws": 5,
        "transformation": "percentage",
        "smc_data_proportion": 0.2,
        "n_threads": 6,
    }


def test_epiweekly_input_includes_forecast_through(tmp_path):
    pipeline = _pipeline(tmp_path, frequency="epiweekly")
    run = make_test_forecast_run(
        output_dir=tmp_path,
        report_date=dt.date(2026, 9, 8),
        exclude_last_n_days=2,
        model_name=pipeline.model_name,
        sources=("nssp",),
    )
    weekly_data = run.nssp.data.with_columns(
        date=pl.lit(dt.date(2026, 8, 29)),
        resolution=pl.lit("epiweekly"),
    )
    run = replace(
        run,
        surveillance=replace(
            run.surveillance,
            nssp=replace(run.nssp, data=weekly_data, resolution="epiweekly"),
        ),
    )

    pipeline.prepare_model_artifacts(run)

    assert run.last_training_date == dt.date(2026, 8, 29)
    assert run.forecast_through == dt.date(2026, 10, 3)
    input_path = run.model_dir / f"{run.model_name}_input.json"
    assert json.loads(input_path.read_text())["forecast_through"] == "2026-10-03"


@patch("cfa.stf.routine.epiautogp.forecast_epiautogp.run_julia_script")
def test_runner_builds_explicit_julia_command(mock_run_julia, tmp_path):
    input_path = tmp_path / "input.json"
    model_dir = tmp_path / "model"

    run_epiautogp_forecast(
        input_path,
        model_dir,
        n_particles=2,
        n_mcmc=3,
        n_hmc=4,
        n_forecast_draws=5,
        transformation="boxcox",
        smc_data_proportion=0.25,
        n_threads=6,
    )

    assert model_dir.is_dir()
    assert mock_run_julia.call_args.args[1] == [
        f"--json-input={input_path}",
        f"--output-dir={model_dir}",
        "--n-particles=2",
        "--n-mcmc=3",
        "--n-hmc=4",
        "--n-forecast-draws=5",
        "--transformation=boxcox",
        "--smc-data-proportion=0.25",
    ]
    assert mock_run_julia.call_args.kwargs["executor_flags"][1] == "--threads=6"


def test_reporting_delay_fetches_pmf_from_run(tmp_path):
    get_pmf = patch(
        "cfa.stf.routine.epiautogp.forecast_epiautogp.get_nnh_right_truncation_pmf",
        return_value=[0.25, 0.75],
    )
    with get_pmf as mock_get_pmf:
        source = _resolve_nowcast_source(
            forecast_run=_run(tmp_path),
            config=EpiAutoGPConfig("nssp", "daily", "observed"),
            nowcast_source_name="reporting-delay",
        )

    assert isinstance(source, ReportingDelayNowcast)
    assert source.reporting_delay_pmf == [0.25, 0.75]
    mock_get_pmf.assert_called_once_with(
        state_abb="CA",
        disease="covid",
        as_of=dt.date(2024, 12, 20),
        reference_date=dt.date(2024, 12, 20),
    )


def test_direct_reporting_delay_pmf_skips_fetch(tmp_path):
    with patch(
        "cfa.stf.routine.epiautogp.forecast_epiautogp.get_nnh_right_truncation_pmf"
    ) as mock_get_pmf:
        source = _resolve_nowcast_source(
            forecast_run=_run(tmp_path),
            config=EpiAutoGPConfig("nssp", "daily", "other"),
            nowcast_source_name="reporting-delay",
            reporting_delay_pmf=[0.4, 0.6],
        )

    assert isinstance(source, ReportingDelayNowcast)
    assert source.reporting_delay_pmf == [0.4, 0.6]
    mock_get_pmf.assert_not_called()


def test_reporting_delay_rejects_percentage_target(tmp_path):
    with pytest.raises(ValueError, match="not applicable"):
        _resolve_nowcast_source(
            forecast_run=_run(tmp_path),
            config=EpiAutoGPConfig("nssp", "daily", "pct"),
            nowcast_source_name="reporting-delay",
            reporting_delay_pmf=[1.0],
        )


def test_reporting_delay_warns_for_non_daily_frequency(caplog, tmp_path):
    with caplog.at_level(logging.WARNING):
        source = _resolve_nowcast_source(
            forecast_run=_run(tmp_path),
            config=EpiAutoGPConfig("nssp", "epiweekly", "observed"),
            nowcast_source_name="reporting-delay",
            reporting_delay_pmf=[1.0],
        )

    assert isinstance(source, ReportingDelayNowcast)
    assert "PMF support matches the model cadence" in caplog.text


def test_hubverse_resolution_uses_run_and_config(tmp_path):
    run = _run(tmp_path)
    config = EpiAutoGPConfig("nhsn", "epiweekly", "observed")
    source = _resolve_nowcast_source(
        forecast_run=run,
        config=config,
        nowcast_source_name="hubverse",
        hubverse_nowcast_dir=tmp_path,
    )

    assert isinstance(source, HubverseNowcast)
    assert source.containing_dir == tmp_path
    assert source.forecast_run is run
    assert source.config is config


@pytest.mark.parametrize(
    "config",
    [
        EpiAutoGPConfig("nssp", "epiweekly", "observed"),
        EpiAutoGPConfig("nhsn", "daily", "observed"),
        EpiAutoGPConfig("nhsn", "epiweekly", "pct"),
    ],
)
def test_hubverse_resolution_propagates_applicability_errors(tmp_path, config):
    with pytest.raises(ValueError, match="only applicable"):
        _resolve_nowcast_source(
            forecast_run=_run(tmp_path),
            config=config,
            nowcast_source_name="hubverse",
            hubverse_nowcast_dir=tmp_path,
        )


def test_hubverse_resolution_requires_containing_directory(tmp_path):
    with pytest.raises(ValueError, match="hubverse_nowcast_dir is required"):
        _resolve_nowcast_source(
            forecast_run=_run(tmp_path),
            config=EpiAutoGPConfig("nhsn", "epiweekly", "observed"),
            nowcast_source_name="hubverse",
        )


def test_nowcast_resolution_rejects_mutually_exclusive_inputs(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve_nowcast_source(
            forecast_run=_run(tmp_path),
            config=EpiAutoGPConfig("nhsn", "epiweekly", "observed"),
            nowcast_source_name="hubverse",
            reporting_delay_pmf=[0.5, 0.5],
            hubverse_nowcast_dir=tmp_path,
        )


def test_none_and_invalid_nowcast_names(tmp_path):
    run = _run(tmp_path)
    config = EpiAutoGPConfig("nssp", "daily", "observed")
    assert (
        _resolve_nowcast_source(
            forecast_run=run,
            config=config,
            nowcast_source_name="none",
        )
        is None
    )
    with pytest.raises(ValueError, match="must be one of"):
        _resolve_nowcast_source(
            forecast_run=run,
            config=config,
            nowcast_source_name="auto",
        )
