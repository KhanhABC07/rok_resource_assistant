from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from rok_assistant.action_engine import DEFAULT_ABORT_REASON, ActionEngine
from rok_assistant.db.models import Task, TaskRunHistory, TaskStep, utc_now_iso
from rok_assistant.emulator import MEmuAdbManager
from rok_assistant.task_result import TaskResult
from rok_assistant.workflow_engine import (
    LegacyAutomationTaskAdapter,
    WorkflowEngine,
    WorkflowExecutionContext,
    WorkflowExecutionResult,
    WorkflowOutcome,
    WorkflowValidationError,
)


ActionEngineFactory = Callable[[int, str], ActionEngine]
Sleeper = Callable[[float], None]


class TaskRunHistoryWriter(Protocol):
    def create(self, history: TaskRunHistory) -> int:
        ...


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
        adapter = LegacyAutomationTaskAdapter()
        try:
            workflow = adapter.to_workflow(task, ordered_steps)
        except WorkflowValidationError as exc:
            validation_error = str(exc)
            return self._task_failed(task, validation_error)

        engine = self._create_action_engine(instance_index, instance_name)
        result = WorkflowEngine().execute(
            workflow,
            WorkflowExecutionContext(
                action_engine=engine,
                sleeper=self.sleeper,
            ),
        )
        step_results = self._legacy_step_results(result)
        if result.outcome == WorkflowOutcome.CANCELLED:
            return self._task_aborted(task, result.message, step_results)
        if result.outcome not in {WorkflowOutcome.SUCCESS, WorkflowOutcome.SKIPPED}:
            return self._task_failed(task, result.message, step_results)

        task_result = TaskResult.SUCCESS
        self.logger.info(
            "[TaskEngine] Task Finished: %s result=%s",
            task.name,
            task_result.name,
        )
        return TaskExecutionResult(task.id, task.name, True, "", step_results, task_result)

    def _legacy_step_results(
        self,
        result: WorkflowExecutionResult,
    ) -> list[StepExecutionResult]:
        step_results: list[StepExecutionResult] = []
        for index, step in enumerate(result.steps, start=1):
            order = self._int(step.data, "legacy_order", index)
            action_type = str(step.data.get("legacy_action_type") or step.action_type)
            success = step.outcome in {
                WorkflowOutcome.SUCCESS,
                WorkflowOutcome.SKIPPED,
                WorkflowOutcome.CANCELLED,
            }
            data = dict(step.data)
            data["workflow_outcome"] = step.outcome.value
            step_results.append(
                StepExecutionResult(
                    order=order,
                    action_type=action_type,
                    success=success,
                    message=step.message,
                    result=data,
                )
            )
        return sorted(step_results, key=lambda item: item.order)

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
        return LegacyAutomationTaskAdapter().validation_error(steps)

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
