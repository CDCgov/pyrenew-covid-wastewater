import warnings

import dagster as dg

from cfa.stf.routine.dagster.execution import (
    azure_batch_2cpu_execution_config,
    azure_batch_4cpu_execution_config,
    azure_batch_64cpu_execution_config,
)


class IsWeekday(dg.AutomationCondition):
    def __init__(self, weekday: int):
        self.weekday = weekday
        super().__init__()

    def evaluate(self, context: dg.AutomationContext) -> dg.AutomationResult:
        true_subset = (
            context.candidate_subset
            if context.evaluation_time.weekday() == self.weekday
            else context.get_empty_subset()
        )
        return dg.AutomationResult(context=context, true_subset=true_subset)

    @property
    def name(self) -> str:
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        return f"is_{days[self.weekday].lower()}"


eager_on_wed = (dg.AutomationCondition.eager() & IsWeekday(2)).with_label(
    "eager_on_wed"
)

warnings.filterwarnings(
    "ignore",
    message=r".*AutomationConditionSensorDefinition.*is currently in beta.*",
)

weekly_fable_sensor = dg.AutomationConditionSensorDefinition(
    name="Fable",
    target=dg.AssetSelection.groups("Fable"),
    run_tags=azure_batch_2cpu_execution_config.to_run_tags(),
    use_user_code_server=True,
)
weekly_pyrenew_sensor = dg.AutomationConditionSensorDefinition(
    name="Pyrenew",
    target=dg.AssetSelection.groups("Pyrenew"),
    run_tags=azure_batch_4cpu_execution_config.to_run_tags(),
    use_user_code_server=True,
)
weekly_fusion_sensor = dg.AutomationConditionSensorDefinition(
    name="Fusion",
    target=dg.AssetSelection.groups("Fusion"),
    run_tags=azure_batch_2cpu_execution_config.to_run_tags(),
    use_user_code_server=True,
)
epiautogp_sensor = dg.AutomationConditionSensorDefinition(
    name="EpiAutoGP",
    target=dg.AssetSelection.groups("EpiAutoGP"),
    run_tags=azure_batch_64cpu_execution_config.to_run_tags(),
    use_user_code_server=True,
)
