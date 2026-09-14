import datetime as dt
import logging
from pathlib import Path

from cfa.stf.routine._paths import FABLE_DIR
from cfa.stf.routine.data.data_access import DataResolution, ForecastSourceName
from cfa.stf.routine.forecast_pipeline import ForecastPipeline
from cfa.stf.routine.forecast_run import ForecastRun
from cfa.stf.routine.utils.language_utils import run_r_script


def fable_e_other_forecasts(
    model_dir: Path, n_forecast_days: int, n_samples: int
) -> None:
    script_args = [
        "--model-dir",
        f"{model_dir}",
        "--n-forecast-days",
        f"{n_forecast_days}",
        "--n-samples",
        f"{n_samples}",
    ]
    run_r_script(
        FABLE_DIR / "fit_fable.R",
        script_args,
        function_name="fit_fable",
        capture_output=False,
    )
    return None


class FablePipeline(ForecastPipeline):
    """Single-location Fable E-other forecast pipeline."""

    def __init__(
        self,
        *,
        n_samples: int,
        ed_visit_input_resolution: DataResolution = "daily",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self._ed_visit_input_resolution = ed_visit_input_resolution

    @property
    def model_name(self) -> str:
        return f"{self.ed_visit_input_resolution}_fable_e_other"

    @property
    def sources(self) -> set[ForecastSourceName]:
        return {"nssp"}

    @property
    def ed_visit_input_resolution(self) -> DataResolution:
        return self._ed_visit_input_resolution

    @property
    def minimum_exclude_last_n_days(self) -> int:
        return 3

    def run_model(self, run: ForecastRun) -> None:
        self.logger.info("Performing fable E-other forecasting")
        fable_e_other_forecasts(
            run.model_dir,
            run.n_forecast_days,
            self.n_samples,
        )


def main(
    disease: str,
    loc: str,
    output_dir: Path | str,
    n_lookback_days: int | None,
    n_samples: int,
    run_date: dt.date,
    exclude_last_n_days: int = 0,
    ed_visit_input_resolution: DataResolution = "daily",
    fail_on_stale_data: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    FablePipeline(
        disease=disease,
        loc=loc,
        output_dir=output_dir,
        n_lookback_days=n_lookback_days,
        run_date=run_date,
        exclude_last_n_days=exclude_last_n_days,
        fail_on_stale_data=fail_on_stale_data,
        logger=logger,
        n_samples=n_samples,
        ed_visit_input_resolution=ed_visit_input_resolution,
    ).execute()
    return None
