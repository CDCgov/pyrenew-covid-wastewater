import logging
import os
from pathlib import Path

import dagster as dg
from cfa_dagster import (
    ExecutionConfig,
    SelectorConfig,
    azure_batch_executor,
    docker_executor,
)
from cfa_dagster import (
    is_production as is_prod,
)
from pygit2.repository import Repository

log = logging.getLogger(__name__)

user = os.getenv("DAGSTER_USER")
is_production = is_prod()


def _find_project_root() -> Path:
    """Find the checkout containing the Dagster project configuration."""
    working_dir = Path.cwd().resolve()
    module_dir = Path(__file__).resolve().parent
    for candidate in (
        working_dir,
        *working_dir.parents,
        module_dir,
        *module_dir.parents,
    ):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return working_dir


local_workdir = _find_project_root()
container_workdir = Path(f"/{local_workdir.name}")

try:
    current_branch_name = os.environ.get("GITHUB_HEAD_REF") or str(
        Repository(local_workdir).head.shorthand
    )
    log.debug(f"Branch name from git: {current_branch_name}")
except Exception:
    current_branch_name = "main"
    log.warning("No .git folder detected; using main as the branch name")

registry = "cfaprdbatchcr.azurecr.io"
tag = (
    "latest"
    if (is_production or current_branch_name == "main")
    else current_branch_name.replace("/", "-")
)
image = f"{registry}/{local_workdir.name}:{tag}"

azure_blob_mounts = [
    f"stf-routine-forecasting-prod-output:{container_workdir}/output",
    f"stf-routine-forecasting-test-output:{container_workdir}/test-output",
]
local_output_mount = (
    f"{local_workdir / 'test-output'}:{container_workdir / 'test-output'}"
)

basic_execution_config = ExecutionConfig(
    executor=SelectorConfig(class_name=dg.multiprocess_executor.__name__),
)

docker_execution_config = ExecutionConfig(
    executor=SelectorConfig(
        class_name=docker_executor.__name__,
        config={
            "image": image,
            "retries": {"enabled": {}},
            "container_kwargs": {
                "volumes": [
                    f"/home/{user}/.azure:/root/.azure",
                    f"{local_workdir / 'src'}:{container_workdir / 'src'}",
                ]
                + [local_output_mount]
            },
        },
    ),
)

_azure_batch_shared_config = {
    **({} if is_production else {"image": image}),
    "container_kwargs": {
        "volumes": azure_blob_mounts,
        "working_dir": f"{container_workdir}",
    },
}

azure_batch_2cpu_execution_config = ExecutionConfig(
    executor=SelectorConfig(
        class_name=azure_batch_executor.__name__,
        config={
            "pool_name": "stf-routine-2cpu",
            **_azure_batch_shared_config,
        },
    ),
)

azure_batch_4cpu_execution_config = ExecutionConfig(
    executor=SelectorConfig(
        class_name=azure_batch_executor.__name__,
        config={
            "pool_name": "stf-routine-4cpu",
            **_azure_batch_shared_config,
        },
    ),
)

azure_batch_64cpu_execution_config = ExecutionConfig(
    executor=SelectorConfig(
        class_name=azure_batch_executor.__name__,
        config={
            "pool_name": "stf-routine-64cpu",
            **_azure_batch_shared_config,
        },
    ),
)
