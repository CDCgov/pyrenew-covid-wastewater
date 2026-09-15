import datetime as dt
from enum import StrEnum
from zoneinfo import ZoneInfo

import dagster as dg
from cfa.stf.forecasttools import LOCATION_LIST
from cfa_dagster import GraphDimension, GraphDimensionExclusion
from pydantic import BaseModel, Field

from cfa.stf.routine.dagster.execution import is_production

DEFAULT_EXCLUDED_LOCATIONS = ["AS", "GU", "MP", "PR", "UM", "VI"]
SUPPORTED_DISEASES = ["covid", "flu", "rsv"]
DISEASES = SUPPORTED_DISEASES
Disease = StrEnum("Disease", {value: value for value in DISEASES})
LOCATIONS = [
    location for location in LOCATION_LIST if location not in DEFAULT_EXCLUDED_LOCATIONS
]
Location = StrEnum("Location", {value: value for value in LOCATIONS})

tz = "America/New_York"
daily_partitions_def = dg.DailyPartitionsDefinition(
    start_date=dt.datetime.now(ZoneInfo(tz)) - dt.timedelta(days=1),
    end_offset=1,
    timezone=tz,
)


class _ModelTrainingFields(BaseModel):
    output_basedir: str = Field(default_factory=lambda: "")
    n_training_days: int = Field(default_factory=lambda: 0)
    exclude_last_n_days: int = Field(default_factory=lambda: 0)
    fail_on_stale_data: bool = Field(default_factory=lambda: is_production)


class ConfigOverride(_ModelTrainingFields, dg.Config):
    location: Location  # type: ignore[reportInvalidTypeForm]

    def as_dict(self) -> dict:  # type: ignore[reportInvalidTypeForm]
        return self.model_dump(mode="json", exclude_unset=True)


class ModelBaseConfig(_ModelTrainingFields, dg.ConfigurableResource):
    """Base configuration shared by Fable and Pyrenew model assets."""

    output_basedir: str = "output" if is_production else "test-output"
    n_training_days: int = 150
    exclude_last_n_days: int = 1
    fail_on_stale_data: bool = is_production
    diseases: GraphDimension[Disease] = GraphDimension(DISEASES)  # type: ignore[reportInvalidTypeForm]
    locations: GraphDimension[Location] = GraphDimension(LOCATIONS)  # type: ignore[reportInvalidTypeForm]
    config_overrides: list[ConfigOverride] = Field(
        default=[],
        description=(
            "Provide location-specific overrides as a list of dicts. "
            "The Launchpad accepts both yaml and json-style lists e.g."
            "config_overrides: [{ location: GA, exclude_last_n_days: 2 }]"
        ),
    )  # type: ignore[reportInvalidTypeForm]

    def get_by_location(self, loc: Location) -> "ModelBaseConfig":  # type: ignore[reportInvalidTypeForm]
        overrides = {}
        for entry in self.config_overrides:
            if isinstance(entry, dict):
                entry = ConfigOverride(**entry)
            if entry.location == loc:
                overrides = entry.model_dump(exclude={"location"}, exclude_unset=True)
                break
        return self.model_copy(update=overrides)


class FableEOtherConfig(dg.ConfigurableResource):
    n_samples: int = 400 if not is_production else 2000


class PyrenewConfig(dg.ConfigurableResource):
    n_warmup: int = 200 if not is_production else 1000
    n_samples: int = 200 if not is_production else 500
    n_chains: int = 2 if not is_production else 4
    rng_key: int = 12345
    additional_forecast_letters: str = ""


class EModelExclusions(dg.ConfigurableResource):
    locations: GraphDimensionExclusion[Location] = GraphDimensionExclusion(["WY"])  # type: ignore[reportInvalidTypeForm]


class WModelExclusions(dg.ConfigurableResource):
    diseases: GraphDimension[Disease] = GraphDimension(["covid"])  # type: ignore[reportInvalidTypeForm]


class PostProcessConfig(dg.Config):
    output_basedir: str = "output" if is_production else "test-output"
    skip_existing: bool = False
    save_local_copy: bool = False
    local_copy_dir: str = ""
    postprocess_diseases: list[str] = ["covid", "flu", "rsv"]
