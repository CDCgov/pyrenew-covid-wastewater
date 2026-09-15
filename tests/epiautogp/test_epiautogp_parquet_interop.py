"""Cross-language checks for EpiAutoGP parquet output.

This module guards the schema boundary that failed in pipeline-run-check
postprocessing: EpiAutoGP wrote samples through Julia/DuckDB, R converted each
model's samples to a hubverse table, and Python/Polars combined those hubverse
tables across model directories. The failure was caused by one producer writing
draw IDs as floats while the rest wrote integer sample identifiers.
Those draw IDs are `.draw` in `samples.parquet`; the hubverse conversion renames
them to `output_type_id`, which is where the CI failure surfaced.

The fixtures build tiny but realistic model-fit directories for each producer
family: EpiAutoGP samples from the direct NowcastAutoGP Julia runner,
fable-style samples from R, and PyRenew-style posterior predictive output
converted by the real Polars sample conversion function. The final test then
runs the same hubverse conversion and batch combine functions
used by CI so producer schema drift is caught before postprocessing reaches the
expensive end-to-end workflow.
"""

import datetime as dt
import json
import shutil
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from cfa.stf.routine._paths import EPIAUTOGP_DIR
from cfa.stf.routine.pyrenew_hew.forecast_pyrenew import (
    format_pyrenew_samples,
)
from cfa.stf.routine.utils.language_utils import (
    run_julia_script,
    run_r_code,
)
from cfa.stf.routine.utils.postprocess_forecast_batches import combine_hubverse_tables
from cfa.stf.routine.utils.r_utils import model_fit_dir_to_hub_tbl

FORECAST_DATES = [
    dt.date(2024, 2, 6),
    dt.date(2024, 2, 7),
]
TRAINING_DATES = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(36)]
REPORT_DATE = dt.date(2024, 2, 3)
EXPECTED_DATES = FORECAST_DATES * 2
EPIAUTOGP_EXPECTED_DATES = (TRAINING_DATES + FORECAST_DATES) * 2
EPIAUTOGP_EXPECTED_DRAWS = [
    draw for draw in range(1, 3) for _ in range(len(TRAINING_DATES + FORECAST_DATES))
]


@dataclass(frozen=True)
class EpiAutoGPInteropPaths:
    batch_dir: Path
    model_fit_dir: Path
    samples_path: Path


def _quote_for_embedded_code(value: str | Path) -> str:
    return json.dumps(str(value))


def _skip_if_r_packages_missing(*packages: str) -> None:
    package_list = ", ".join(_quote_for_embedded_code(package) for package in packages)
    code = textwrap.dedent(
        f"""
        pkgs <- c({package_list})
        missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
        if (length(missing) > 0) {{
          cat(paste(missing, collapse = ","))
          quit(status = 1)
        }}
        """
    )
    try:
        run_r_code(code, executor_flags=["--vanilla"], text=True)
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f"R packages are not available: {exc}")


def _write_epiautogp_input(path: Path) -> None:
    reports = [12.0 + (i % 7) * 0.4 + i * 0.05 for i in range(len(TRAINING_DATES))]
    input_data = {
        "dates": [date.isoformat() for date in TRAINING_DATES],
        "reports": reports,
        "pathogen": "covid",
        "location": "US",
        "target": "nssp",
        "frequency": "daily",
        "ed_visit_type": "pct",
        "forecast_through": FORECAST_DATES[-1].isoformat(),
        "nowcast_dates": [],
        "nowcast_reports": [],
    }
    path.write_text(json.dumps(input_data), encoding="utf-8")


@pytest.fixture(scope="module")
def epiautogp_interop_paths(tmp_path_factory) -> Iterator[EpiAutoGPInteropPaths]:
    tmp_dir = tmp_path_factory.mktemp("epiautogp-parquet-interop")
    try:
        batch_dir = tmp_dir / "covid_lookback-33_omit-1"
        model_fit_dir = batch_dir / "model_runs" / "US" / "epiautogp_nssp_daily_pct"
        input_path = tmp_dir / "epiautogp-input.json"
        _write_epiautogp_input(input_path)

        try:
            run_julia_script(
                EPIAUTOGP_DIR / "fit_epiautogp.jl",
                [
                    f"--json-input={input_path}",
                    f"--output-dir={model_fit_dir}",
                    "--n-particles=2",
                    "--n-mcmc=1",
                    "--n-hmc=1",
                    "--n-forecast-draws=2",
                    "--transformation=percentage",
                    "--smc-data-proportion=0.5",
                ],
                executor_flags=[f"--project={EPIAUTOGP_DIR}", "--startup-file=no"],
                function_name="run_epiautogp_interop_fixture",
                text=True,
            )
        except FileNotFoundError:
            pytest.skip("julia is not available")

        samples_path = model_fit_dir / "samples.parquet"
        assert samples_path.is_file()
        yield EpiAutoGPInteropPaths(
            batch_dir=batch_dir,
            model_fit_dir=model_fit_dir,
            samples_path=samples_path,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _write_fable_reference_model_samples(batch_dir: Path) -> Path:
    fable_model_fit_dir = batch_dir / "model_runs" / "US" / "fable_daily"
    r_code = textwrap.dedent(
        f"""
        library(dplyr)
        library(forecasttools)
        library(fs)
        library(stfroutineforecasting)

        fable_model_fit_dir <- {_quote_for_embedded_code(fable_model_fit_dir)}
        dir_create(fable_model_fit_dir, recurse = TRUE)

        dates <- as.Date(c("2024-02-04", "2024-02-10", "2024-02-04", "2024-02-10"))
        draws <- as.integer(c(1, 1, 2, 2))

        fable_samples <- tibble::tibble(
          date = dates,
          .draw = draws,
          observed_ed_visits = c(101, 201, 102, 202),
          other_ed_visits = c(1001, 2001, 1002, 2002)
        ) |>
          stfroutineforecasting::format_timeseries_output(
            geo_value = "US",
            disease = "covid",
            resolution = "daily",
            output_type_id = ".draw"
          )
        forecasttools::write_tabular(
          fable_samples,
          fs::path(fable_model_fit_dir, "samples", ext = "parquet")
        )
        """
    )
    run_r_code(r_code, executor_flags=["--vanilla"], text=True)
    return fable_model_fit_dir


def _write_pyrenew_reference_model_samples(batch_dir: Path) -> Path:
    pyrenew_model_fit_dir = batch_dir / "model_runs" / "US" / "pyrenew_e"
    mcmc_output_dir = pyrenew_model_fit_dir / "mcmc_output"
    mcmc_output_dir.mkdir(parents=True, exist_ok=True)
    posterior_predictive = pl.DataFrame(
        {
            "chain": [0, 0, 0, 0],
            "draw": [0, 0, 1, 1],
            "variable": ["observed_ed_visits"] * len(EXPECTED_DATES),
            "value": [101.0, 201.0, 102.0, 202.0],
            "date": EXPECTED_DATES,
        }
    )
    posterior_predictive.write_parquet(
        mcmc_output_dir / "tidy_posterior_predictive.parquet"
    )

    samples = format_pyrenew_samples(
        posterior_predictive,
        geo_value="US",
        disease="covid",
    )
    samples.write_parquet(pyrenew_model_fit_dir / "samples.parquet")
    return pyrenew_model_fit_dir


def test_epiautogp_samples_parquet_date_is_discovered_by_polars(
    epiautogp_interop_paths,
):
    samples = pl.read_parquet(epiautogp_interop_paths.samples_path)

    assert samples.schema["date"] == pl.Date
    assert samples.schema[".draw"] == pl.Int32
    assert samples["date"].to_list() == EPIAUTOGP_EXPECTED_DATES
    assert samples[".draw"].to_list() == EPIAUTOGP_EXPECTED_DRAWS

    lazy_schema = (
        pl.scan_parquet(epiautogp_interop_paths.samples_path)
        .select(pl.col("date").min())
        .collect()
        .schema
    )
    assert lazy_schema["date"] == pl.Date


def test_epiautogp_samples_parquet_date_is_discovered_by_r_dplyr(
    epiautogp_interop_paths,
):
    _skip_if_r_packages_missing("forecasttools", "dplyr", "lubridate")
    expected_dates_r = ", ".join(
        _quote_for_embedded_code(expected_date.isoformat())
        for expected_date in EPIAUTOGP_EXPECTED_DATES
    )
    r_code = textwrap.dedent(
        f"""
        samples <- forecasttools::read_tabular({_quote_for_embedded_code(epiautogp_interop_paths.samples_path)})
        stopifnot(inherits(samples$date, "Date"))
        stopifnot(is.integer(samples[[".draw"]]))
        stopifnot(identical(
          as.character(samples$date),
          c({expected_dates_r})
        ))

        mutated <- dplyr::mutate(
          samples,
          date_plus_one = .data$date + lubridate::days(1)
        )
        stopifnot(inherits(mutated$date_plus_one, "Date"))

        summarised <- dplyr::summarise(mutated, min_date = min(.data$date))
        stopifnot(inherits(summarised$min_date, "Date"))
        """
    )

    try:
        run_r_code(r_code, executor_flags=["--vanilla"], text=True)
    except FileNotFoundError:
        pytest.skip("Rscript is not available")


def test_epiautogp_hubverse_table_combines_with_fable_and_pyrenew_outputs(
    epiautogp_interop_paths,
):
    _skip_if_r_packages_missing(
        "argparser",
        "dplyr",
        "forecasttools",
        "fs",
        "stfroutineforecasting",
    )
    try:
        model_fit_dirs = [
            epiautogp_interop_paths.model_fit_dir,
            _write_fable_reference_model_samples(epiautogp_interop_paths.batch_dir),
            _write_pyrenew_reference_model_samples(epiautogp_interop_paths.batch_dir),
        ]
        for model_fit_dir in model_fit_dirs:
            model_fit_dir_to_hub_tbl(model_fit_dir, report_date=REPORT_DATE)
            hubverse_table_path = model_fit_dir / "hubverse_table.parquet"
            assert hubverse_table_path.is_file()
            hubverse_table = pl.read_parquet(hubverse_table_path)
            # Sample .draw values become hubverse output_type_id values.
            assert hubverse_table.schema["output_type_id"] == pl.Int32

        combine_hubverse_tables(epiautogp_interop_paths.batch_dir)
    except FileNotFoundError:
        pytest.skip("Rscript is not available")

    combined_path = (
        epiautogp_interop_paths.batch_dir / "2024-02-03-covid-hubverse-table.parquet"
    )
    assert combined_path.is_file()
    combined = pl.read_parquet(combined_path)
    assert combined.schema["output_type_id"] == pl.Int32
