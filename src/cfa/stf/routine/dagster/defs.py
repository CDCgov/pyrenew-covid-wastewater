"""Assemble the project's modular Dagster definitions."""

import logging

import dagster as dg
from cfa_dagster import (
    ADLS2PickleIOManager,
    collect_definitions,
    dynamic_executor,
    start_dev_env,
)

from cfa.stf.routine.dagster import assets, automation, jobs
from cfa.stf.routine.dagster.config import (
    EModelExclusions,
    FableEOtherConfig,
    ModelBaseConfig,
    PyrenewConfig,
    WModelExclusions,
)
from cfa.stf.routine.dagster.execution import (
    azure_batch_4cpu_execution_config,
    basic_execution_config,
    docker_execution_config,
)

start_dev_env(__name__)

collected_defs = collect_definitions(vars(assets) | vars(automation) | vars(jobs))
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)

defs = dg.Definitions(
    **collected_defs,
    resources={
        "io_manager": ADLS2PickleIOManager(),
        "model_base_config": ModelBaseConfig(),
        "pyrenew_config": PyrenewConfig(),
        "fable_e_other_config": FableEOtherConfig(),
        "e_model_exclusions": EModelExclusions(),
        "w_model_exclusions": WModelExclusions(),
    },
    executor=dynamic_executor(
        default_config=azure_batch_4cpu_execution_config,
        alternate_configs=[basic_execution_config, docker_execution_config],
    ),
)
