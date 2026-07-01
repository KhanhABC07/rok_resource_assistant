from __future__ import annotations

from collections.abc import Sequence

from rok_assistant.db.models import Task, TaskStep
from rok_assistant.workflow_types import (
    WorkflowDefinitionSpec,
    WorkflowStepSpec,
    WorkflowValidationError,
    _int_value,
    _optional_float,
)


class LegacyAutomationTaskAdapter:
    def to_workflow(self, task: Task, steps: Sequence[TaskStep]) -> WorkflowDefinitionSpec:
        ordered_steps = sorted(steps, key=lambda item: item.order)
        converted, next_index = self._convert_block(ordered_steps, 0, stop_actions=set())
        if next_index != len(ordered_steps):
            step = ordered_steps[next_index]
            raise WorkflowValidationError(
                f"Unexpected {step.action_type} at step {step.order}."
            )
        return WorkflowDefinitionSpec(
            workflow_key=f"legacy-task-{task.id or 'unsaved'}",
            name=task.name,
            schema_version=2,
            version=1,
            enabled=task.enabled,
            steps=converted,
        )

    def validation_error(self, steps: Sequence[TaskStep]) -> str:
        try:
            self._convert_block(sorted(steps, key=lambda item: item.order), 0, stop_actions=set())
        except WorkflowValidationError as exc:
            return str(exc)
        return ""

    def _convert_block(
        self,
        steps: Sequence[TaskStep],
        index: int,
        *,
        stop_actions: set[str],
    ) -> tuple[list[WorkflowStepSpec], int]:
        converted: list[WorkflowStepSpec] = []
        while index < len(steps):
            step = steps[index]
            if step.action_type in stop_actions:
                return converted, index
            if step.action_type == "RepeatEnd":
                raise WorkflowValidationError(
                    f"RepeatEnd without RepeatStart at step {step.order}."
                )
            if step.action_type == "Else":
                raise WorkflowValidationError(
                    f"Else without IfTemplateExists at step {step.order}."
                )
            if step.action_type == "EndIf":
                raise WorkflowValidationError(
                    f"EndIf without IfTemplateExists at step {step.order}."
                )
            if step.action_type == "RepeatStart":
                child_steps, repeat_end_index = self._convert_block(
                    steps,
                    index + 1,
                    stop_actions={"RepeatEnd"},
                )
                if repeat_end_index >= len(steps) or steps[repeat_end_index].action_type != "RepeatEnd":
                    raise WorkflowValidationError(
                        f"Missing RepeatEnd for RepeatStart at step {step.order}."
                    )
                converted.append(self._repeat_step(step, child_steps))
                index = repeat_end_index + 1
                continue
            if step.action_type == "IfTemplateExists":
                converted.append(self._if_step(steps, index))
                index = self._find_matching_endif(steps, index) + 1
                continue
            converted.append(self._regular_step(step))
            index += 1
        return converted, index

    def _if_step(self, steps: Sequence[TaskStep], index: int) -> WorkflowStepSpec:
        if_step = steps[index]
        else_index, endif_index = self._find_if_bounds(steps, index)
        if endif_index < 0:
            raise WorkflowValidationError(
                f"Missing EndIf for IfTemplateExists at step {if_step.order}."
            )
        then_stop = {"Else", "EndIf"}
        then_steps, then_end_index = self._convert_block(
            steps,
            index + 1,
            stop_actions=then_stop,
        )
        if else_index >= 0:
            if then_end_index != else_index:
                raise WorkflowValidationError(
                    f"Invalid IfTemplateExists block at step {if_step.order}."
                )
            else_steps, else_end_index = self._convert_block(
                steps,
                else_index + 1,
                stop_actions={"EndIf"},
            )
            if else_end_index != endif_index:
                raise WorkflowValidationError(
                    f"Invalid Else block at step {steps[else_index].order}."
                )
        else:
            if then_end_index != endif_index:
                raise WorkflowValidationError(
                    f"Invalid IfTemplateExists block at step {if_step.order}."
                )
            else_steps = []
        parameters = dict(if_step.parameters or {})
        parameters["condition_type"] = "template_exists"
        return WorkflowStepSpec(
            step_key=self._legacy_key(if_step),
            action_type="if_else",
            parameters=parameters,
            then_steps=then_steps,
            else_steps=else_steps,
            timeout_seconds=_optional_float(parameters.get("timeout_seconds")),
            legacy_order=if_step.order,
            legacy_action_type=if_step.action_type,
        )

    def _repeat_step(
        self,
        step: TaskStep,
        child_steps: list[WorkflowStepSpec],
    ) -> WorkflowStepSpec:
        parameters = dict(step.parameters or {})
        count = max(0, _int_value(parameters.get("count"), 1))
        parameters["count"] = count
        parameters.setdefault("max_count", count)
        return WorkflowStepSpec(
            step_key=self._legacy_key(step),
            action_type="bounded_repeat",
            parameters=parameters,
            steps=child_steps,
            legacy_order=step.order,
            legacy_action_type=step.action_type,
        )

    def _regular_step(self, step: TaskStep) -> WorkflowStepSpec:
        parameters = dict(step.parameters or {})
        action_type = step.action_type
        mapped_type = {
            "WaitTemplate": "wait",
            "ClickTemplate": "click_semantic_template",
            "ClickCoordinates": "tap",
            "SwipeCoordinates": "swipe",
            "Delay": "delay",
            "AbortTask": "cancel",
        }.get(action_type, action_type)
        if action_type == "WaitTemplate":
            parameters["condition_type"] = "template_exists"
        return WorkflowStepSpec(
            step_key=self._legacy_key(step),
            action_type=mapped_type,
            parameters=parameters,
            timeout_seconds=_optional_float(parameters.get("timeout_seconds")),
            legacy_order=step.order,
            legacy_action_type=step.action_type,
        )

    def _find_if_bounds(self, steps: Sequence[TaskStep], index: int) -> tuple[int, int]:
        depth = 0
        else_index = -1
        for candidate_index in range(index + 1, len(steps)):
            action_type = steps[candidate_index].action_type
            if action_type == "IfTemplateExists":
                depth += 1
            elif action_type == "EndIf":
                if depth == 0:
                    return else_index, candidate_index
                depth -= 1
            elif action_type == "Else" and depth == 0:
                if else_index >= 0:
                    raise WorkflowValidationError(
                        f"Duplicate Else inside If block at step {steps[candidate_index].order}."
                    )
                else_index = candidate_index
        return else_index, -1

    def _find_matching_endif(self, steps: Sequence[TaskStep], index: int) -> int:
        _else_index, endif_index = self._find_if_bounds(steps, index)
        if endif_index < 0:
            raise WorkflowValidationError(
                f"Missing EndIf for IfTemplateExists at step {steps[index].order}."
            )
        return endif_index

    @staticmethod
    def _legacy_key(step: TaskStep) -> str:
        return f"legacy-{step.order:04d}-{step.action_type}"
