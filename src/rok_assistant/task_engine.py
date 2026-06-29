from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from rok_assistant.action_engine import DEFAULT_ABORT_REASON, ActionEngine
from rok_assistant.db.models import Task, TaskRunHistory, TaskStep, utc_now_iso
from rok_assistant.emulator import MEmuAdbManager
from rok_assistant.task_result import TaskResult


ActionEngineFactory = Callable[[int, str], ActionEngine]
Sleeper = Callable[[float], None]


class TaskRunHistoryWriter(Protocol):
    def create(self, history: TaskRunHistory) -> int:
        ...


class TaskAbortException(Exception):
    pass


@dataclass
class StepExecutionResult:
    order: int
    action_type: str
    success: bool
    message: str = ""
    result: dict[str, object] = field(default_factory=dict)


@dataclass
class TaskExecutionResult:
    task_id: int | None
    task_name: str
    success: bool
    message: str = ""
    steps: list[StepExecutionResult] = field(default_factory=list)
    result: TaskResult | None = None

    def __post_init__(self) -> None:
        if self.result is None:
            self.result = TaskResult.SUCCESS if self.success else TaskResult.FAILED


class TaskRunner:
    def __init__(
        self,
        adb_manager: MEmuAdbManager,
        *,
        action_engine_factory: ActionEngineFactory | None = None,
        sleeper: Sleeper = time.sleep,
        history_repository: TaskRunHistoryWriter | None = None,
    ):
        self.adb_manager = adb_manager
        self.action_engine_factory = action_engine_factory
        self.sleeper = sleeper
        self.history_repository = history_repository
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_task(
        self,
        task: Task,
        steps: list[TaskStep],
        *,
        instance_index: int,
        instance_name: str,
    ) -> TaskExecutionResult:
        started_at = utc_now_iso()
        self.logger.info("[TaskEngine] Task Started: %s", task.name)
        try:
            result = self._run_task(
                task,
                steps,
                instance_index=instance_index,
                instance_name=instance_name,
            )
        except Exception as exc:
            self.logger.exception("[TaskEngine] Task crashed: %s", task.name)
            result = self._task_failed(task, str(exc))
        finished_at = utc_now_iso()
        self._record_run_history(
            result,
            instance_index=instance_index,
            instance_name=instance_name,
            started_at=started_at,
            finished_at=finished_at,
        )
        return result

    def _run_task(
        self,
        task: Task,
        steps: list[TaskStep],
        *,
        instance_index: int,
        instance_name: str,
    ) -> TaskExecutionResult:
        if not task.enabled:
            message = "Task is disabled."
            return self._task_failed(task, message)

        ordered_steps = sorted(steps, key=lambda item: item.order)
        validation_error = self.validate_steps(ordered_steps)
        if validation_error:
            return self._task_failed(task, validation_error)

        engine = self._create_action_engine(instance_index, instance_name)
        step_results: list[StepExecutionResult] = []
        try:
            success, message, next_index = self._execute_steps(
                engine,
                ordered_steps,
                0,
                len(ordered_steps),
                task.name,
                step_results,
            )
        except TaskAbortException as exc:
            return self._task_aborted(task, str(exc), step_results)
        except Exception as exc:
            self.logger.exception("[TaskEngine] Task crashed: %s", task.name)
            return self._task_failed(task, str(exc), step_results)
        if not success:
            return self._task_failed(task, message, step_results)
        if next_index != len(ordered_steps):
            message = "Task execution stopped before all steps were processed."
            return self._task_failed(task, message, step_results)

        result = TaskResult.SUCCESS
        self.logger.info(
            "[TaskEngine] Task Finished: %s result=%s",
            task.name,
            result.name,
        )
        return TaskExecutionResult(task.id, task.name, True, "", step_results, result)

    def _record_run_history(
        self,
        result: TaskExecutionResult,
        *,
        instance_index: int,
        instance_name: str,
        started_at: str,
        finished_at: str,
    ) -> None:
        if self.history_repository is None:
            return

        task_result = result.result or (
            TaskResult.SUCCESS if result.success else TaskResult.FAILED
        )
        try:
            self.history_repository.create(
                TaskRunHistory(
                    task_id=result.task_id,
                    task_name=result.task_name,
                    instance_index=instance_index,
                    instance_name=instance_name,
                    started_at=started_at,
                    finished_at=finished_at,
                    result=task_result.value,
                    error_message=(
                        result.message if task_result == TaskResult.FAILED else ""
                    ),
                    abort_reason=(
                        result.message if task_result == TaskResult.ABORTED else ""
                    ),
                )
            )
        except Exception:
            self.logger.exception(
                "[TaskEngine] Failed to store task run history: %s",
                result.task_name,
            )

    def _task_aborted(
        self,
        task: Task,
        message: str,
        steps: list[StepExecutionResult] | None = None,
    ) -> TaskExecutionResult:
        result = TaskResult.ABORTED
        message = message.strip() or DEFAULT_ABORT_REASON
        self.logger.info("Task aborted intentionally: %s", message)
        self.logger.info(
            "[TaskEngine] Task Aborted: %s - %s result=%s",
            task.name,
            message,
            result.name,
        )
        return TaskExecutionResult(
            task.id,
            task.name,
            False,
            message,
            steps or [],
            result,
        )

    def _task_failed(
        self,
        task: Task,
        message: str,
        steps: list[StepExecutionResult] | None = None,
    ) -> TaskExecutionResult:
        result = TaskResult.FAILED
        if message:
            self.logger.error(
                "[TaskEngine] Task Failed: %s - %s result=%s",
                task.name,
                message,
                result.name,
            )
        else:
            self.logger.error(
                "[TaskEngine] Task Failed: %s result=%s",
                task.name,
                result.name,
            )
        return TaskExecutionResult(
            task.id,
            task.name,
            False,
            message,
            steps or [],
            result,
        )

    def validate_steps(self, steps: list[TaskStep]) -> str:
        repeat_start: TaskStep | None = None
        if_start: TaskStep | None = None
        if_has_else = False
        for step in sorted(steps, key=lambda item: item.order):
            if step.action_type == "RepeatStart":
                if repeat_start is not None:
                    return (
                        "Nested repeat blocks are not supported "
                        f"(step {step.order})."
                    )
                repeat_start = step
            elif step.action_type == "RepeatEnd":
                if repeat_start is None:
                    return f"RepeatEnd without RepeatStart at step {step.order}."
                repeat_start = None
            elif step.action_type == "IfTemplateExists":
                if if_start is not None:
                    return f"Nested If blocks are not supported (step {step.order})."
                if_start = step
                if_has_else = False
            elif step.action_type == "Else":
                if if_start is None:
                    return f"Else without IfTemplateExists at step {step.order}."
                if if_has_else:
                    return f"Duplicate Else inside If block at step {step.order}."
                if_has_else = True
            elif step.action_type == "EndIf":
                if if_start is None:
                    return f"EndIf without IfTemplateExists at step {step.order}."
                if_start = None
                if_has_else = False
        if repeat_start is not None:
            return f"Missing RepeatEnd for RepeatStart at step {repeat_start.order}."
        if if_start is not None:
            return f"Missing EndIf for IfTemplateExists at step {if_start.order}."
        return ""

    def _execute_steps(
        self,
        engine: ActionEngine,
        steps: list[TaskStep],
        start_index: int,
        end_index: int,
        task_name: str,
        step_results: list[StepExecutionResult],
    ) -> tuple[bool, str, int]:
        index = start_index
        while index < end_index:
            step = steps[index]
            if step.action_type == "RepeatStart":
                repeat_end_index = self._find_repeat_end(steps, index + 1, end_index)
                if repeat_end_index < 0:
                    return False, f"Missing RepeatEnd for RepeatStart at step {step.order}.", index
                count = max(0, self._int(step.parameters or {}, "count", 1))
                self.logger.info(
                    "[TaskEngine] Step Started: task=%s order=%s action=%s",
                    task_name,
                    step.order,
                    step.action_type,
                )
                self.logger.info(
                    "[TaskEngine] loop start: task=%s step=%s count=%s",
                    task_name,
                    step.order,
                    count,
                )
                step_results.append(
                    StepExecutionResult(
                        order=step.order,
                        action_type=step.action_type,
                        success=True,
                        result={"count": count},
                    )
                )
                for iteration in range(1, count + 1):
                    self.logger.info(
                        "[TaskEngine] current iteration: task=%s step=%s iteration=%s/%s",
                        task_name,
                        step.order,
                        iteration,
                        count,
                    )
                    success, message, _next_index = self._execute_steps(
                        engine,
                        steps,
                        index + 1,
                        repeat_end_index,
                        task_name,
                        step_results,
                    )
                    if not success:
                        self.logger.error(
                            "[TaskEngine] loop end: task=%s step=%s success=False",
                            task_name,
                            step.order,
                        )
                        return False, message, repeat_end_index + 1
                repeat_end = steps[repeat_end_index]
                self.logger.info(
                    "[TaskEngine] Step Started: task=%s order=%s action=%s",
                    task_name,
                    repeat_end.order,
                    repeat_end.action_type,
                )
                step_results.append(
                    StepExecutionResult(
                        order=repeat_end.order,
                        action_type=repeat_end.action_type,
                        success=True,
                    )
                )
                self.logger.info(
                    "[TaskEngine] loop end: task=%s step=%s success=True",
                    task_name,
                    step.order,
                )
                self.logger.info(
                    "[TaskEngine] Step Finished: task=%s order=%s action=%s",
                    task_name,
                    repeat_end.order,
                    repeat_end.action_type,
                )
                self.logger.info(
                    "[TaskEngine] Step Finished: task=%s order=%s action=%s",
                    task_name,
                    step.order,
                    step.action_type,
                )
                index = repeat_end_index + 1
                continue

            if step.action_type == "IfTemplateExists":
                else_index, endif_index = self._find_if_bounds(steps, index + 1, end_index)
                if endif_index < 0:
                    return False, f"Missing EndIf for IfTemplateExists at step {step.order}.", index
                condition_result = self._execute_if_condition(engine, step, task_name)
                step_results.append(condition_result)
                if not condition_result.success:
                    return False, condition_result.message, endif_index + 1

                branch_result = bool(condition_result.result.get("condition_result"))
                self.logger.info(
                    "[TaskEngine] IF condition result: task=%s step=%s result=%s",
                    task_name,
                    step.order,
                    branch_result,
                )
                if branch_result:
                    self.logger.info(
                        "[TaskEngine] entering TRUE branch: task=%s step=%s",
                        task_name,
                        step.order,
                    )
                    branch_start = index + 1
                    branch_end = else_index if else_index >= 0 else endif_index
                else:
                    self.logger.info(
                        "[TaskEngine] entering FALSE branch: task=%s step=%s",
                        task_name,
                        step.order,
                    )
                    branch_start = else_index + 1 if else_index >= 0 else endif_index
                    branch_end = endif_index

                success, message, _next_index = self._execute_steps(
                    engine,
                    steps,
                    branch_start,
                    branch_end,
                    task_name,
                    step_results,
                )
                if not success:
                    return False, message, endif_index + 1

                endif_step = steps[endif_index]
                self.logger.info(
                    "[TaskEngine] EndIf reached: task=%s step=%s",
                    task_name,
                    endif_step.order,
                )
                step_results.append(
                    StepExecutionResult(
                        order=endif_step.order,
                        action_type=endif_step.action_type,
                        success=True,
                    )
                )
                index = endif_index + 1
                continue

            if step.action_type in {"RepeatEnd", "Else", "EndIf"}:
                return True, "", index

            step_result = self._execute_regular_step(engine, step, task_name)
            step_results.append(step_result)
            if step_result.result.get("aborted"):
                raise TaskAbortException(step_result.message)
            if not step_result.success:
                return False, step_result.message, index + 1
            index += 1
        return True, "", index

    def _execute_regular_step(
        self,
        engine: ActionEngine,
        step: TaskStep,
        task_name: str,
    ) -> StepExecutionResult:
        self.logger.info(
            "[TaskEngine] Step Started: task=%s order=%s action=%s",
            task_name,
            step.order,
            step.action_type,
        )
        step_result = self._execute_step(engine, step)
        if not step_result.success:
            self.logger.error(
                "[TaskEngine] Step Failed: task=%s order=%s action=%s message=%s",
                task_name,
                step.order,
                step.action_type,
                step_result.message,
            )
            return step_result
        self.logger.info(
            "[TaskEngine] Step Finished: task=%s order=%s action=%s",
            task_name,
            step.order,
            step.action_type,
        )
        return step_result

    @staticmethod
    def _find_repeat_end(steps: list[TaskStep], start_index: int, end_index: int) -> int:
        for index in range(start_index, end_index):
            if steps[index].action_type == "RepeatEnd":
                return index
        return -1

    @staticmethod
    def _find_if_bounds(
        steps: list[TaskStep],
        start_index: int,
        end_index: int,
    ) -> tuple[int, int]:
        else_index = -1
        for index in range(start_index, end_index):
            if steps[index].action_type == "Else" and else_index < 0:
                else_index = index
            elif steps[index].action_type == "EndIf":
                return else_index, index
        return else_index, -1

    def _execute_if_condition(
        self,
        engine: ActionEngine,
        step: TaskStep,
        task_name: str,
    ) -> StepExecutionResult:
        self.logger.info(
            "[TaskEngine] Step Started: task=%s order=%s action=%s",
            task_name,
            step.order,
            step.action_type,
        )
        parameters = step.parameters or {}
        try:
            result = engine.wait_for_template(
                str(parameters.get("template_path", "")),
                threshold=self._float(parameters, "threshold", 0.8),
                timeout_seconds=self._float(parameters, "timeout_seconds", 10.0),
                retry_interval_seconds=self._float(
                    parameters,
                    "retry_interval_seconds",
                    1.0,
                ),
            )
        except Exception as exc:
            result = {"success": False, "fatal": True, "message": str(exc)}

        if result.get("fatal"):
            message = str(result.get("message") or "IfTemplateExists failed.")
            self.logger.error(
                "[TaskEngine] Step Failed: task=%s order=%s action=%s message=%s",
                task_name,
                step.order,
                step.action_type,
                message,
            )
            return StepExecutionResult(
                order=step.order,
                action_type=step.action_type,
                success=False,
                message=message,
                result=dict(result),
            )

        condition_result = bool(result.get("success"))
        step_result = dict(result)
        step_result["condition_result"] = condition_result
        self.logger.info(
            "[TaskEngine] Step Finished: task=%s order=%s action=%s",
            task_name,
            step.order,
            step.action_type,
        )
        return StepExecutionResult(
            order=step.order,
            action_type=step.action_type,
            success=True,
            result=step_result,
        )

    def _execute_step(self, engine: ActionEngine, step: TaskStep) -> StepExecutionResult:
        parameters = step.parameters or {}
        try:
            if step.action_type == "WaitTemplate":
                result = engine.wait_for_template(
                    str(parameters.get("template_path", "")),
                    threshold=self._float(parameters, "threshold", 0.8),
                    timeout_seconds=self._float(parameters, "timeout_seconds", 10.0),
                    retry_interval_seconds=self._float(
                        parameters,
                        "retry_interval_seconds",
                        1.0,
                    ),
                )
            elif step.action_type == "ClickTemplate":
                result = engine.click_template(
                    str(parameters.get("template_path", "")),
                    threshold=self._float(parameters, "threshold", 0.8),
                )
            elif step.action_type == "ClickCoordinates":
                result = engine.click_coordinates(
                    self._int(parameters, "x", 0),
                    self._int(parameters, "y", 0),
                )
            elif step.action_type == "SwipeCoordinates":
                result = engine.swipe_coordinates(
                    self._int(parameters, "x1", 0),
                    self._int(parameters, "y1", 0),
                    self._int(parameters, "x2", 0),
                    self._int(parameters, "y2", 0),
                    self._int(parameters, "duration_ms", 500),
                )
            elif step.action_type == "Delay":
                seconds = max(0.0, self._float(parameters, "seconds", 1.0))
                self.sleeper(seconds)
                result = {"success": True, "elapsed_time": seconds}
            elif step.action_type == "AbortTask":
                if "reason" in parameters:
                    result = engine.abort_task(str(parameters.get("reason") or ""))
                else:
                    result = engine.abort_task()
            else:
                result = {
                    "success": False,
                    "message": f"Unsupported action type: {step.action_type}",
                }
        except Exception as exc:
            result = {"success": False, "message": str(exc)}

        success = bool(result.get("success"))
        if result.get("aborted"):
            message = str(result.get("message") or DEFAULT_ABORT_REASON)
        else:
            message = "" if success else str(result.get("message") or "Step failed.")
        return StepExecutionResult(
            order=step.order,
            action_type=step.action_type,
            success=success,
            message=message,
            result=dict(result),
        )

    def _create_action_engine(self, instance_index: int, instance_name: str) -> ActionEngine:
        if self.action_engine_factory is not None:
            return self.action_engine_factory(instance_index, instance_name)
        return ActionEngine(self.adb_manager, instance_index, instance_name)

    @staticmethod
    def _int(parameters: dict[str, object], key: str, default: int) -> int:
        try:
            return int(parameters.get(key, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _float(parameters: dict[str, object], key: str, default: float) -> float:
        try:
            return float(parameters.get(key, default))
        except (TypeError, ValueError):
            return default
