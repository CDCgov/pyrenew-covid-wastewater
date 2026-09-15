import json
import subprocess

import dagster as dg
from cfa_dagster import GraphDimension, dynamic_executor

from cfa.stf.routine.dagster.config import ModelBaseConfig, tz
from cfa.stf.routine.dagster.execution import (
    azure_batch_4cpu_execution_config,
    basic_execution_config,
    image,
    is_production,
    local_output_mount,
    local_workdir,
    registry,
)

update_script_url = (
    "https://raw.githubusercontent.com/CDCgov/cfa-dagster/"
    "refs/heads/main/"
    "scripts/update_code_location.py"
)
prod_server_image = f"{registry}/{local_workdir.name}:latest"


def _refresh_prod_server_image(logger, image_to_deploy: str):
    logger.info(f"Deploying {image_to_deploy} to the dagster prod server.")
    subprocess.run(
        ["uv", "run", update_script_url, "--registry_image", image_to_deploy],
        check=True,
    )


@dg.op
def refresh_prod_server_image_op(context: dg.OpExecutionContext, image_to_deploy: str):
    _refresh_prod_server_image(context.log, image_to_deploy)


refresh_prod_server_image_config = dg.RunConfig(
    ops={
        "refresh_prod_server_image_op": {
            "inputs": {"image_to_deploy": prod_server_image}
        }
    },
    execution=basic_execution_config.to_run_config(),
)


@dg.job(config=refresh_prod_server_image_config, executor_def=dynamic_executor())
def refresh_prod_server_image():
    refresh_prod_server_image_op()


E2E_LOCATIONS = ["CA", "US"]


def e2e_config() -> dg.RunConfig:
    return dg.RunConfig(
        resources={
            "model_base_config": ModelBaseConfig(
                locations=GraphDimension(E2E_LOCATIONS)
            )
        },
        execution=azure_batch_4cpu_execution_config.to_run_config(),
    )


def e2e_json() -> str:
    return json.dumps(e2e_config().to_config_dict())


end_to_end = dg.define_asset_job(
    name="end_to_end",
    selection=dg.AssetSelection.groups("Fable", "Pyrenew", "Fusion"),
    config=e2e_config(),
)


@dg.schedule(
    cron_schedule="00 23 * * TUE",
    execution_timezone=tz,
    job_name="refresh_prod_server_image",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def reset_prod_server_image_for_wednesday():
    return dg.RunRequest(run_config=refresh_prod_server_image_config)


if not is_production:

    @dg.op
    def build_image_op(
        context: dg.OpExecutionContext,
        should_push: bool,
        should_deploy_to_prod: bool,
        dockerfile_path: str,
        build_context: str,
        image: str,
    ):
        build_command = [
            "docker",
            "buildx",
            "build",
            "-t",
            image,
            "-f",
            dockerfile_path,
            build_context,
        ]
        if should_push:
            subprocess.run(["az", "login", "--identity"], check=True)
            subprocess.run(["az", "acr", "login", "-n", registry], check=True)
            build_command.append("--push")
        context.log.info(f"Running {' '.join(build_command)}")
        subprocess.run(build_command, check=True)
        if should_deploy_to_prod:
            _refresh_prod_server_image(context.log, image_to_deploy=image)

    @dg.job(
        config=dg.RunConfig(
            ops={
                "build_image_op": {
                    "inputs": {
                        "should_push": True,
                        "should_deploy_to_prod": False,
                        "dockerfile_path": f"{local_workdir}/Dockerfile",
                        "build_context": str(local_workdir),
                        "image": image,
                    }
                }
            },
            execution=basic_execution_config.to_run_config(),
        ),
        executor_def=dynamic_executor(),
    )
    def build_image():
        build_image_op()

    @dg.op
    def explore_image_op(context: dg.OpExecutionContext):
        context.log.info("Use the originating terminal to interact with the container.")
        subprocess.run(
            ["docker", "run", "-it", "-v", local_output_mount, "--rm", image, "bash"],
            check=True,
        )

    @dg.job(executor_def=dg.in_process_executor)
    def explore_image():
        explore_image_op()
