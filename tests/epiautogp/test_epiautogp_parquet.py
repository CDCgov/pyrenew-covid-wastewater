"""Checks for the direct NowcastAutoGP Julia runner parquet output."""

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest

from cfa.stf.routine._paths import EPIAUTOGP_DIR
from cfa.stf.routine.utils.language_utils import run_julia_script


def _write_synthetic_input(path: Path) -> None:
    start_date = dt.date(2024, 1, 1)
    dates = [start_date + dt.timedelta(days=i) for i in range(36)]
    reports = [12.0 + (i % 7) * 0.4 + i * 0.05 for i in range(len(dates))]
    input_data = {
        "dates": [date.isoformat() for date in dates],
        "reports": reports,
        "pathogen": "covid",
        "location": "US",
        "target": "nssp",
        "frequency": "daily",
        "ed_visit_type": "pct",
        "forecast_through": dt.date(2024, 2, 7).isoformat(),
        "nowcast_dates": [],
        "nowcast_reports": [],
    }
    path.write_text(json.dumps(input_data), encoding="utf-8")


def test_direct_nowcastautogp_runner_writes_pipeline_parquet(tmp_path) -> None:
    input_path = tmp_path / "epiautogp-input.json"
    output_dir = tmp_path / "model-fit"
    _write_synthetic_input(input_path)

    try:
        run_julia_script(
            EPIAUTOGP_DIR / "fit_epiautogp.jl",
            [
                f"--json-input={input_path}",
                f"--output-dir={output_dir}",
                "--n-particles=2",
                "--n-mcmc=1",
                "--n-hmc=1",
                "--n-forecast-draws=4",
                "--transformation=percentage",
                "--smc-data-proportion=0.5",
            ],
            executor_flags=[f"--project={EPIAUTOGP_DIR}", "--startup-file=no"],
            function_name="test_direct_nowcastautogp_runner_writes_pipeline_parquet",
            text=True,
        )
    except FileNotFoundError:
        pytest.skip("julia is not available")

    samples_path = output_dir / "samples.parquet"
    assert samples_path.is_file()

    samples = pl.read_parquet(samples_path)
    assert samples.schema["date"] == pl.Date
    assert samples.schema[".draw"] == pl.Int32
    expected_dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(38)]
    assert samples["date"].to_list() == expected_dates * 4
    assert samples[".draw"].to_list() == [
        draw for draw in range(1, 5) for _ in range(38)
    ]
    assert samples[".variable"].unique().to_list() == ["prop_disease_ed_visits"]
    assert samples["resolution"].unique().to_list() == ["daily"]
    assert samples["geo_value"].unique().to_list() == ["US"]
    assert samples["disease"].unique().to_list() == ["covid"]
    assert samples[".value"].min() >= 0.0
    assert samples[".value"].max() <= 1.0


def test_direct_runner_rejects_partial_epiweekly_horizon(tmp_path) -> None:
    input_path = tmp_path / "epiautogp-input.json"
    output_dir = tmp_path / "model-fit"
    _write_synthetic_input(input_path)
    input_data = json.loads(input_path.read_text())
    input_data["frequency"] = "epiweekly"
    input_data["forecast_through"] = dt.date(2024, 2, 10).isoformat()
    input_path.write_text(json.dumps(input_data), encoding="utf-8")

    try:
        with pytest.raises(RuntimeError, match="whole number of model steps"):
            run_julia_script(
                EPIAUTOGP_DIR / "fit_epiautogp.jl",
                [
                    f"--json-input={input_path}",
                    f"--output-dir={output_dir}",
                ],
                executor_flags=[f"--project={EPIAUTOGP_DIR}", "--startup-file=no"],
                function_name="test_direct_runner_rejects_partial_epiweekly_horizon",
                text=True,
            )
    except FileNotFoundError:
        pytest.skip("julia is not available")
