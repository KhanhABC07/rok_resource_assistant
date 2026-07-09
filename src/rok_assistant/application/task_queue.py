from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from rok_assistant.db.models import Instance, ScheduledTask, Task, TaskRunHistory, TaskStep
from rok_assistant.paths import PROJECT_ROOT
from rok_assistant.task_engine import TaskExecutionResult, TaskResult, TaskRunner
from rok_assistant.tasks.resource_search_workflow import (
    MAX_RESOURCE_LEVEL,
    MIN_RESOURCE_LEVEL,
    ResourceSearchWorkflow,
    ResourceType,
    check_template_readiness,
)


ACTION_PARAMETER_FIELDS: dict[str, tuple[str, ...]] = {
    "WaitTemplate": ("template", "threshold", "timeout", "polling_interval"),
    "ClickTemplate": ("template", "threshold"),
    "ClickCoordinates": ("x", "y"),
    "SwipeCoordinates": ("x1", "y1", "x2", "y2", "swipe_duration"),
    "Delay": ("delay_duration",),
    "RepeatStart": ("count",),
    "RepeatEnd": (),
    "IfTemplateExists": ("template", "threshold"),
    "Else": (),
    "EndIf": (),
    "AbortTask": ("reason",),
}

FIELD_PARAMETER_KEYS: dict[str, str] = {
    "template": "template_path",
    "threshold": "threshold",
    "timeout": "timeout_seconds",
    "polling_interval": "retry_interval_seconds",
    "x": "x",
    "y": "y",
    "x1": "x1",
    "y1": "y1",
    "x2": "x2",
    "y2": "y2",
    "swipe_duration": "duration_ms",
    "delay_duration": "seconds",
    "count": "count",
    "reason": "reason",
}

FIELD_DEFAULTS: dict[str, object] = {
    "template": "",
    "threshold": 0.8,
    "timeout": 10.0,
    "polling_interval": 1.0,
    "x": 540,
    "y": 960,
    "x1": 540,
    "y1": 1500,
    "x2": 540,
    "y2": 600,
    "swipe_duration": 500,
    "delay_duration": 1.0,
    "count": 5,
    "reason": "",
}

TEMPLATE_ACTION_TYPES = {"WaitTemplate", "ClickTemplate", "IfTemplateExists"}


class AutomationTaskRepositoryPort(Protocol):
    def list_all(self) -> list[Task]: ...

    def get(self, item_id: int) -> Task | None: ...

    def save_task(self, task: Task) -> int: ...

    def delete_task(self, task_id: int) -> None: ...

    def duplicate_task(self, task_id: int) -> int: ...

    def list_steps(self, task_id: int) -> list[TaskStep]: ...

    def get_step(self, step_id: int) -> TaskStep | None: ...

    def add_step(
        self,
        task_id: int,
        action_type: str,
        parameters: dict[str, object],
    ) -> int: ...

    def save_step(self, step: TaskStep) -> int: ...

    def delete_step(self, step_id: int) -> None: ...

    def move_step_up(self, step_id: int) -> None: ...

    def move_step_down(self, step_id: int) -> None: ...


class InstanceRepositoryPort(Protocol):
    def list_all(self) -> list[Instance]: ...

    def get(self, item_id: int) -> Instance | None: ...


class ScheduledTaskRepositoryPort(Protocol):
    def list_recent(self, limit: int = 300) -> list[ScheduledTask]: ...


class TaskRunHistoryRepositoryPort(Protocol):
    def list_recent(self, limit: int = 200) -> list[TaskRunHistory]: ...


@dataclass(frozen=True)
class TaskRow:
    id: int | None
    name: str
    enabled: str
    created_at: str


@dataclass(frozen=True)
class StepRow:
    id: int | None
    order: int
    action_type: str
    parameters: dict[str, object]
    status: str
    status_kind: str


@dataclass(frozen=True)
class TemplateReadinessRow:
    template_name: str
    template_path: str
    status: str
    status_kind: str
    step_ids: tuple[int, ...]
    invalid_step_id: int | None = None


@dataclass(frozen=True)
class ReadinessView:
    ready: bool
    missing_count: int
    invalid_count: int
    rows: tuple[TemplateReadinessRow, ...]


@dataclass(frozen=True)
class TargetInstanceRow:
    id: int | None
    label: str


@dataclass(frozen=True)
class SchedulerTaskRow:
    id: int | None
    instance_name: str
    character_name: str
    task_type: str
    march_slot: int | str
    priority: int
    status: str
    scheduled_for: str
    attempts: int
    error_message: str


@dataclass(frozen=True)
class RunHistoryRow:
    task_name: str
    instance_index: int | str
    instance_name: str
    started_at: str
    finished_at: str
    result: str
    error_or_abort_reason: str


@dataclass(frozen=True)
class PreparedTaskRun:
    task: Task
    steps: list[TaskStep]
    instance_index: int
    instance_name: str


@dataclass(frozen=True)
class TaskRunPreparation:
    run: PreparedTaskRun | None = None
    warning_title: str = ""
    warning_message: str = ""
    readiness: ReadinessView | None = None

    @property
    def ready(self) -> bool:
        return self.run is not None


class TaskExecutionService:
    def __init__(
        self,
        adb_manager: object,
        *,
        history_repository: object | None = None,
    ) -> None:
        self.adb_manager = adb_manager
        self.history_repository = history_repository

    def run_task(self, run: PreparedTaskRun) -> TaskExecutionResult:
        return TaskRunner(
            self.adb_manager,
            history_repository=self.history_repository,
        ).run_task(
            run.task,
            run.steps,
            instance_index=run.instance_index,
            instance_name=run.instance_name,
        )


class TaskQueueViewModel:
    def __init__(
        self,
        automation_tasks: AutomationTaskRepositoryPort,
        instances: InstanceRepositoryPort,
        scheduled_tasks: ScheduledTaskRepositoryPort,
        *,
        schedule_enabled_work: Callable[[], int],
        task_run_history: TaskRunHistoryRepositoryPort | None = None,
    ) -> None:
        self.automation_tasks = automation_tasks
        self.instances = instances
        self.scheduled_tasks = scheduled_tasks
        self.schedule_enabled_work = schedule_enabled_work
        self.task_run_history = task_run_history

    def list_task_rows(self) -> list[TaskRow]:
        return [
            TaskRow(
                id=task.id,
                name=task.name,
                enabled="Yes" if task.enabled else "No",
                created_at=task.created_at,
            )
            for task in self.automation_tasks.list_all()
        ]

    def task_exists(self, task_id: int) -> bool:
        return self.automation_tasks.get(task_id) is not None

    def get_task(self, task_id: int) -> Task | None:
        return self.automation_tasks.get(task_id)

    def create_task(self) -> int:
        next_number = len(self.automation_tasks.list_all()) + 1
        return self.automation_tasks.save_task(
            Task(name=f"Task {next_number}", enabled=True)
        )

    def create_resource_workflow(
        self,
        *,
        resource_type: ResourceType,
        target_level: int,
        march_required: bool,
        fallback_enabled: bool,
    ) -> int:
        workflow = ResourceSearchWorkflow(
            resource_type=resource_type,
            target_level=target_level,
            march_required=march_required,
            fallback_enabled=fallback_enabled,
        )
        task_id = self.automation_tasks.save_task(
            Task(
                name=self.resource_workflow_task_name(workflow),
                enabled=True,
                template_readiness_required=True,
            )
        )
        for step in workflow.to_task_steps():
            self.automation_tasks.add_step(
                task_id,
                step.action_type,
                step.parameters or {},
            )
        return task_id

    @staticmethod
    def resource_workflow_task_name(workflow: ResourceSearchWorkflow) -> str:
        resource_type = workflow.resource_type.value.title()
        return f"Resource Workflow - {resource_type} L{workflow.target_level}"

    def save_task(self, task_id: int, *, name: str, enabled: bool) -> bool:
        existing = self.automation_tasks.get(task_id)
        if existing is None:
            return False
        self.automation_tasks.save_task(
            Task(
                id=task_id,
                name=name,
                enabled=enabled,
                template_readiness_required=existing.template_readiness_required,
            )
        )
        return True

    def delete_task(self, task_id: int) -> None:
        self.automation_tasks.delete_task(task_id)

    def duplicate_task(self, task_id: int) -> int:
        return self.automation_tasks.duplicate_task(task_id)

    def list_step_rows(self, task_id: int) -> list[StepRow]:
        steps = self.automation_tasks.list_steps(task_id)
        readiness_required = self._template_readiness_required(task_id)
        return [
            StepRow(
                id=step.id,
                order=step.order,
                action_type=step.action_type,
                parameters=step.parameters or {},
                status=self.step_template_status(step, readiness_required),
                status_kind=self.step_template_status_kind(step, readiness_required),
            )
            for step in steps
        ]

    def list_steps(self, task_id: int) -> list[TaskStep]:
        return self.automation_tasks.list_steps(task_id)

    def get_step(self, step_id: int) -> TaskStep | None:
        return self.automation_tasks.get_step(step_id)

    def add_step(
        self,
        task_id: int,
        action_type: str,
        parameters: dict[str, object],
    ) -> int:
        return self.automation_tasks.add_step(task_id, action_type, parameters)

    def save_step(
        self,
        *,
        task_id: int,
        step_id: int,
        action_type: str,
        parameters: dict[str, object],
    ) -> bool:
        existing = self.automation_tasks.get_step(step_id)
        if existing is None:
            return False
        self.automation_tasks.save_step(
            TaskStep(
                id=existing.id,
                task_id=task_id,
                order=existing.order,
                action_type=action_type,
                parameters=parameters,
            )
        )
        return True

    def delete_step(self, step_id: int) -> None:
        self.automation_tasks.delete_step(step_id)

    def move_step_up(self, step_id: int) -> None:
        self.automation_tasks.move_step_up(step_id)

    def move_step_down(self, step_id: int) -> None:
        self.automation_tasks.move_step_down(step_id)

    def readiness_view(self, task_id: int | None) -> ReadinessView:
        if task_id is None:
            return ReadinessView(True, 0, 0, ())
        steps = self.automation_tasks.list_steps(task_id)
        if not self._template_readiness_required(task_id):
            return ReadinessView(True, 0, 0, self.template_readiness_rows(steps))

        readiness = check_template_readiness(steps)
        invalid_steps = [
            step
            for step in steps
            if step.action_type in TEMPLATE_ACTION_TYPES
            and not str((step.parameters or {}).get("template_path", "")).strip()
        ]
        ready = readiness.ready and not invalid_steps
        return ReadinessView(
            ready=ready,
            missing_count=len(readiness.missing_templates),
            invalid_count=len(invalid_steps),
            rows=self.template_readiness_rows(steps),
        )

    def template_readiness_rows(
        self,
        steps: list[TaskStep],
    ) -> tuple[TemplateReadinessRow, ...]:
        template_steps: dict[str, list[TaskStep]] = {}
        invalid_steps: list[TaskStep] = []
        for step in sorted(steps, key=lambda item: item.order):
            if step.action_type not in TEMPLATE_ACTION_TYPES:
                continue
            template_path = str((step.parameters or {}).get("template_path", "")).strip()
            if template_path:
                template_steps.setdefault(template_path, []).append(step)
            else:
                invalid_steps.append(step)

        rows: list[TemplateReadinessRow] = []
        for template_path, matching_steps in template_steps.items():
            exists = self.template_path_exists(template_path)
            rows.append(
                TemplateReadinessRow(
                    template_name=Path(template_path).stem.replace("_", " ").title(),
                    template_path=template_path,
                    status="Ready" if exists else "Missing",
                    status_kind="ready" if exists else "missing",
                    step_ids=tuple(
                        step.id for step in matching_steps if step.id is not None
                    ),
                )
            )
        for step in invalid_steps:
            rows.append(
                TemplateReadinessRow(
                    template_name=f"Step {step.order}",
                    template_path="",
                    status="Invalid",
                    status_kind="invalid",
                    step_ids=(),
                    invalid_step_id=step.id,
                )
            )
        return tuple(rows)

    def update_template_paths(
        self,
        *,
        task_id: int,
        selected_path: str,
        template_path: str = "",
        step_id: int | None = None,
    ) -> None:
        for step in self.automation_tasks.list_steps(task_id):
            current_path = str((step.parameters or {}).get("template_path", "")).strip()
            matches = step.id == step_id if step_id is not None else current_path == template_path
            if not matches:
                continue
            parameters = dict(step.parameters or {})
            parameters["template_path"] = selected_path
            self.automation_tasks.save_step(
                TaskStep(
                    id=step.id,
                    task_id=step.task_id,
                    order=step.order,
                    action_type=step.action_type,
                    parameters=parameters,
                )
            )

    def list_target_rows(self) -> list[TargetInstanceRow]:
        return [
            TargetInstanceRow(
                id=instance.id,
                label=(
                    f"{instance.instance_index} - "
                    f"{instance.instance_name or instance.name}"
                ),
            )
            for instance in self.instances.list_all()
            if instance.instance_index is not None
        ]

    def get_instance(self, instance_id: int) -> Instance | None:
        return self.instances.get(instance_id)

    def create_scheduled_tasks(self) -> int:
        return self.schedule_enabled_work()

    def list_scheduler_rows(self, limit: int = 300) -> list[SchedulerTaskRow]:
        return [
            SchedulerTaskRow(
                id=task.id,
                instance_name=task.instance_name,
                character_name=task.character_name,
                task_type=task.task_type,
                march_slot=task.march_slot or "",
                priority=task.priority,
                status=task.status,
                scheduled_for=task.scheduled_for,
                attempts=task.attempts,
                error_message=task.error_message,
            )
            for task in self.scheduled_tasks.list_recent(limit=limit)
        ]

    def list_run_history_rows(self, limit: int = 200) -> list[RunHistoryRow]:
        if self.task_run_history is None:
            return []
        return [
            RunHistoryRow(
                task_name=run.task_name,
                instance_index=run.instance_index if run.instance_index is not None else "",
                instance_name=run.instance_name,
                started_at=run.started_at,
                finished_at=run.finished_at,
                result=run.result,
                error_or_abort_reason=run.error_message or run.abort_reason,
            )
            for run in self.task_run_history.list_recent(limit=limit)
        ]

    def prepare_task_run(
        self,
        *,
        task_id: int | None,
        instance_id: int | None,
    ) -> TaskRunPreparation:
        if task_id is None:
            return TaskRunPreparation(
                warning_title="Tasks",
                warning_message="Select a task first.",
            )
        task = self.automation_tasks.get(task_id)
        if task is None:
            return TaskRunPreparation()
        steps = self.automation_tasks.list_steps(task_id)
        if not steps:
            return TaskRunPreparation(
                warning_title="Tasks",
                warning_message="Add at least one step before running.",
            )
        if task.template_readiness_required:
            readiness = self.readiness_view(task_id)
            if not readiness.ready:
                return TaskRunPreparation(
                    warning_title="Resource Workflow",
                    warning_message=(
                        "Required templates are missing. Add the assets or edit the "
                        "workflow before running."
                    ),
                    readiness=readiness,
                )
        if instance_id is None:
            return TaskRunPreparation(
                warning_title="Tasks",
                warning_message="Select a target MEmu instance.",
            )
        instance = self.instances.get(instance_id)
        if instance is None or instance.instance_index is None:
            return TaskRunPreparation(
                warning_title="Tasks",
                warning_message="Select a target MEmu instance.",
            )
        return TaskRunPreparation(
            run=PreparedTaskRun(
                task=task,
                steps=steps,
                instance_index=instance.instance_index,
                instance_name=instance.instance_name or instance.name,
            )
        )

    def _template_readiness_required(self, task_id: int) -> bool:
        task = self.automation_tasks.get(task_id)
        return bool(task is not None and task.template_readiness_required)

    def step_template_status(self, step: TaskStep, readiness_required: bool) -> str:
        if not readiness_required or step.action_type not in TEMPLATE_ACTION_TYPES:
            return "Ready"
        template_path = str((step.parameters or {}).get("template_path", "")).strip()
        if not template_path:
            return "Invalid"
        if not self.template_path_exists(template_path):
            return "Missing Template"
        return "Ready"

    def step_template_status_kind(
        self,
        step: TaskStep,
        readiness_required: bool,
    ) -> str:
        status = self.step_template_status(step, readiness_required)
        if status == "Missing Template":
            return "missing"
        if status == "Invalid":
            return "invalid"
        return "ready"

    @staticmethod
    def template_path_exists(template_path: str) -> bool:
        path = Path(template_path)
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        return resolved.is_file()
