# Adding a forecasting model

This package contains the single-location model pipelines used by routine forecasting.
New models should extend [`ForecastPipeline`](forecast_pipeline.py), which owns the common lifecycle:

1. validate the model configuration;
2. calculate the training window and load the requested surveillance data;
3. create a [`ForecastRun`](forecast_run.py) with canonical paths and run metadata;
4. serialize common model inputs;
5. fit the model and write forecast samples; and
6. create standard plots, credible intervals, and a Hubverse table.

Existing implementations are useful references:

- [`fable/forecast_fable.py`](fable/forecast_fable.py) is the smallest example.
- [`pyrenew_hew/forecast_pyrenew.py`](pyrenew_hew/forecast_pyrenew.py) shows model-specific inputs and every optional lifecycle hook.
- [`epiautogp/forecast_epiautogp.py`](epiautogp/forecast_epiautogp.py) shows a model with its own configuration, input conversion, and external runtime.

## 1. Create the model package

Add a directory such as `src/cfa/stf/routine/my_model/`, including an `__init__.py` and a `forecast_my_model.py`.
Keep bundled scripts and other runtime assets in the same directory.
If another module needs stable paths to those assets, add them to [`_paths.py`](_paths.py).

A minimal pipeline looks like this:

```python
import datetime as dt
import logging
from pathlib import Path

import polars as pl

from cfa.stf.routine.data.data_access import ForecastSourceName
from cfa.stf.routine.forecast_pipeline import ForecastPipeline
from cfa.stf.routine.forecast_run import ForecastRun


class MyModelPipeline(ForecastPipeline):
    def __init__(self, *, n_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_samples = n_samples

    @property
    def model_name(self) -> str:
        # This becomes the model output directory name and must be stable.
        return "my_model"

    @property
    def sources(self) -> set[ForecastSourceName]:
        # Valid sources are currently "nssp" and "nhsn".
        return {"nssp"}

    def run_model(self, run: ForecastRun) -> None:
        # Replace this example with the real fit and forecast implementation.
        samples = pl.DataFrame(...)
        samples.write_parquet(run.model_dir / "samples.parquet")


def main(
    disease: str,
    loc: str,
    output_dir: Path | str,
    n_lookback_days: int | None,
    n_samples: int,
    run_date: dt.date,
    exclude_last_n_days: int = 0,
    fail_on_stale_data: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    MyModelPipeline(
        disease=disease,
        loc=loc,
        output_dir=output_dir,
        n_lookback_days=n_lookback_days,
        run_date=run_date,
        exclude_last_n_days=exclude_last_n_days,
        fail_on_stale_data=fail_on_stale_data,
        n_samples=n_samples,
        logger=logger,
    ).execute()
```

The base constructor arguments in the example are shared by all models.
Put model-specific settings on the subclass, preferably as keyword-only arguments.
The module-level `main()` function is the boundary used by orchestration and integration tests; it should construct the pipeline and call `execute()`.
Accept an optional logger so orchestrators can preserve their structured logging context.
Configure the default Python logger only when the caller does not provide one.

## 2. Implement the required contracts

Every subclass must implement:

- `model_name`: a stable, filesystem-safe name.
  Outputs are grouped under this name, so changing it changes downstream paths.
- `sources`: the surveillance inputs to load.
  The currently supported values are `"nssp"` (daily emergency-department visits) and `"nhsn"` (epiweekly hospital admissions).
  At least one source is required.
  Supporting a new source also requires changes to [`data/data_access.py`](data/data_access.py) and [`data/prep_data.py`](data/prep_data.py).
- `run_model(run)`: run the model and create a standardized `run.model_dir / "samples.parquet"`.

`ForecastWindow` is the source of truth for the report-anchored training bounds, final forecast date, and model-batch directory name.
Set `n_lookback_days` to `None` to retain all available training history; a positive integer limits training data to that many days before the report date.
`ForecastRun` combines that configured window with the surveillance observations that were actually retained, the population, and the model output paths.
In particular, use `run.model_dir`, `run.data_dir`, and `run.model_run_dir` rather than rebuilding paths in model code.
Models can override `minimum_exclude_last_n_days` when they require a longer omitted tail than the caller requested.
The pipeline's `requested_window` preserves that caller configuration; the run's `forecast_window` records the effective omission, while `run.requested_window` determines the batch directory so all models remain in the same batch.
The resulting layout is:

```text
<output_dir>/<disease>_lookback-<days-or-all>_omit-<days>/
  model_runs/<location>/<model_name>/
    data/
      combined_data.tsv
      data_for_model_fit.json
    samples.parquet
    ci.parquet
    hubverse_table.parquet
```

The base pipeline creates `data/`, `combined_data.tsv`, and `data_for_model_fit.json`.
The model creates `samples.parquet`; common post-processing creates `ci.parquet`, figures, and `hubverse_table.parquet`.

### Forecast sample format

Standard post-processing expects `samples.parquet` in long format with these columns:

  | Column       | Meaning                                              |
  | ------------ | ---------------------------------------------------- |
  | `date`       | Forecast target date, stored as a date               |
  | `.draw`      | Integer draw identifier                              |
  | `geo_value`  | Location abbreviation                                |
  | `disease`    | Disease identifier                                   |
  | `.variable`  | Forecast variable name, such as `observed_ed_visits` |
  | `.value`     | Numeric forecast value                               |
  | `resolution` | `daily` or `epiweekly`                               |

Additional identifiers such as `.chain` or `.iteration` are allowed when relevant.
Draws should form coherent trajectories: the same `.draw` identifies values belonging to one sample across forecast dates.
See [`fable/fit_fable.R`](fable/fit_fable.R) for an R writer and EpiAutoGP's [`forecast_epiautogp.py`](epiautogp/forecast_epiautogp.py) for a non-R writer.

## 3. Declare input resolution and use hooks only where needed

The base pipeline prepares and serializes daily ED-visit inputs by default.
Override the `ed_visit_input_resolution` property with `"epiweekly"` when a model requires ED visits aggregated to complete MMWR weeks.
The declared resolution applies consistently to both `combined_data.tsv` and `data_for_model_fit.json`.

The base class also provides optional hooks for model-specific preparation:

  | Hook                           | Use it to                                                |
  | ------------------------------ | -------------------------------------------------------- |
  | `validate_configuration()`     | Reject invalid option combinations before loading data   |
  | `prepare_model_artifacts(run)` | Create model-specific inputs, such as JSON or parameters |
  | `minimum_exclude_last_n_days`  | Require a longer recent-data omission for this model     |

Do not override `execute()`, `build_forecast_run()`, `prepare_input_artifacts()`, or `publish_outputs()` unless the shared lifecycle itself must change.
Keeping those methods common preserves data freshness checks and compatible outputs.
`run_model()` is responsible for any native-output conversion needed to leave a standardized `samples.parquet` for publishing.

When fitting stops before the report date because `exclude_last_n_days` is nonzero, decide whether the model must predict through that excluded tail.
`run.forecast_window.min_allowed_training_date` and `run.forecast_window.max_allowed_training_date` are the inclusive bounds used while loading and labeling observations for training.
`run.first_training_date` and `run.last_training_date` are instead derived from the observations actually retained.
For daily models, forecasting often means generating `run.n_forecast_days` days from that observed last training date through `run.forecast_through`.
See the Fable and PyRenew implementations.
Fable requires at least three omitted calendar days.
EpiAutoGP also requires at least three when it runs without a nowcast source.

## 4. Add orchestration explicitly

Pipeline subclasses are not discovered automatically.
To run the new model in production, update [`dagster_defs.py`](dagster_defs.py) to:

1. import the new module-level `main()` function;
2. define any model-specific Dagster config;
3. add a helper or asset that passes the shared and model-specific options; and
4. connect the asset to fusion and batch post-processing dependencies when its output is consumed there.

Declare the correct upstream data asset dependencies for the pipeline's `sources`.
Keep this orchestration change separate from the model class so the single-location pipeline remains directly testable.

## 5. Test the model

At minimum, add tests that cover:

- construction through `main()` and propagation of all shared options, modeled on [`tests/test_forecast_entrypoints.py`](../../../../tests/test_forecast_entrypoints.py);
- `model_name`, `sources`, configuration validation, hook behavior, and calls into the model runtime, modeled on [`tests/test_model_pipelines.py`](../../../../tests/test_model_pipelines.py);
- one small single-location integration run that asserts `samples.parquet` and `hubverse_table.parquet` are produced, modeled on the files in [`tests/integration/`](../../../../tests/integration/); and
- output schema and cross-language interoperability if the model writes Parquet outside Python/Polars.

Run the fast suite while developing:

```shell
uv run pytest -m "not pipeline_e2e and not model_integration"
uv run ruff check src tests
uv run ruff format --check src tests
```

Then add a model-specific recipe to the [`Justfile`](../../../../Justfile) and run its integration test with small sample and iteration counts.
Run the reduced end-to-end pipeline when the model participates in fusion or batch post-processing:

```shell
just e2e mock
```
