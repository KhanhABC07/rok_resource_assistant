from __future__ import annotations

import importlib
import logging
import pkgutil
from datetime import UTC, datetime, timedelta

from rok_assistant.db.models import ScheduledTask
from rok_assistant.db.repositories import TaskRepository
from rok_assistant.recovery import ErrorRecoveryPolicy
from rok_assistant.task_result import TaskResult as TaskOutcome

from .alliance import AllianceDonateTaskPlugin, AllianceHelpTaskPlugin, GiftCollectionTaskPlugin
from .base import TaskContext, TaskPlugin, TaskResult
from .gathering import GatheringTaskPlugin


class TaskManager:
    def __init__(
        self,
        task_repository: TaskRepository,
        context: TaskContext,
        recovery_policy: ErrorRecoveryPolicy,
        plugin_packages: list[str] | None = None,
    ):
        self.tasks = task_repository
        self.context = context
        self.recovery_policy = recovery_policy
        self.logger = logging.getLogger(self.__class__.__name__)
        self.plugins: dict[str, TaskPlugin] = {}
        self.register(GatheringTaskPlugin())
        self.register(AllianceHelpTaskPlugin())
        self.register(AllianceDonateTaskPlugin())
        self.register(GiftCollectionTaskPlugin())
        for package_name in plugin_packages or []:
            self.load_package_plugins(package_name)

        self.context.task_lookup = self._task_lookup  # type: ignore[attr-defined]

    def register(self, plugin: TaskPlugin) -> None:
        if not plugin.task_type:
            raise ValueError("Task plugin must define task_type.")
        self.logger.info("Registered task plugin: %s", plugin.task_type)
        self.plugins[plugin.task_type] = plugin

    def load_package_plugins(self, package_name: str) -> None:
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            self.logger.warning("Plugin package not importable: %s", package_name)
            return

        if not hasattr(package, "__path__"):
            return

        for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            module = importlib.import_module(module_info.name)
            for value in module.__dict__.values():
                if (
                    isinstance(value, type)
                    and issubclass(value, TaskPlugin)
                    and value is not TaskPlugin
                ):
                    self.register(value())

    def execute(self, task: ScheduledTask) -> TaskResult:
        if task.id is None:
            return TaskResult(False, "Unsaved task cannot be executed.")

        plugin = self.plugins.get(task.task_type)
        if plugin is None:
            message = f"No plugin registered for task type {task.task_type}."
            self.tasks.mark_failed(task.id, message)
            return TaskResult(False, message, result=TaskOutcome.FAILED)

        self.tasks.mark_running(task.id)
        try:
            result = plugin.run(task.id, self.context)
        except Exception as exc:
            self.logger.exception("Task plugin crashed: %s", task.task_type)
            result = TaskResult(False, str(exc), result=TaskOutcome.FAILED)

        task_result = result.result or (
            TaskOutcome.SUCCESS if result.success else TaskOutcome.FAILED
        )
        self._log_task_result(task, task_result, result.message)

        if task_result == TaskOutcome.SUCCESS:
            self.tasks.mark_completed(task.id, result.message)
            return result

        if task_result == TaskOutcome.ABORTED:
            self.tasks.mark_aborted(task.id, result.message)
            return result

        if result.retry_after_seconds is not None:
            retry_at = (
                datetime.now(UTC) + timedelta(seconds=result.retry_after_seconds)
            ).replace(tzinfo=None, microsecond=0).isoformat()
            self.tasks.schedule_retry(task.id, retry_at, result.message)
            return result

        refreshed = next(iter(self._task_lookup(task.id)), task)
        decision = self.recovery_policy.decide(refreshed.attempts, result.message)
        if decision.should_retry and decision.retry_at:
            self.tasks.schedule_retry(task.id, decision.retry_at, decision.message)
        else:
            self.tasks.mark_failed(task.id, decision.message)
        return result

    def _log_task_result(
        self,
        task: ScheduledTask,
        result: TaskOutcome,
        message: str,
    ) -> None:
        log = self.logger.error if result == TaskOutcome.FAILED else self.logger.info
        log(
            "Scheduled task result: instance=%s task=%s TaskResult=%s message=%s",
            task.instance_name or task.instance_id,
            task.task_type,
            result.name,
            message,
        )

    def _task_lookup(self, task_id: int):
        recent = self.tasks.list_recent(limit=500)
        return (task for task in recent if task.id == task_id)
