from __future__ import annotations

import logging
import threading
from typing import Callable

from rok_assistant.db.models import ScheduledTask
from rok_assistant.db.repositories import InstanceRepository, SettingsRepository, TaskRepository
from rok_assistant.emulator import EmulatorManager
from rok_assistant.scheduler.worker_pool import WorkerPool


SchedulerCallback = Callable[[str, ScheduledTask], None]


class Scheduler:
    def __init__(
        self,
        task_repository: TaskRepository,
        worker_pool: WorkerPool,
        poll_interval_seconds: int = 5,
        instance_repository: InstanceRepository | None = None,
        emulator_manager: EmulatorManager | None = None,
        settings: SettingsRepository | None = None,
    ):
        self.tasks = task_repository
        self.worker_pool = worker_pool
        self.poll_interval_seconds = max(1, poll_interval_seconds)
        self.instances = instance_repository
        self.emulator_manager = emulator_manager
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._callbacks: list[SchedulerCallback] = []

    def add_callback(self, callback: SchedulerCallback) -> None:
        self._callbacks.append(callback)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.worker_pool.start()
        self._thread = threading.Thread(target=self._run_loop, name="rok-scheduler", daemon=True)
        self._thread.start()
        self.logger.info("Scheduler started.")

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        self.worker_pool.stop()
        self.logger.info("Scheduler stopped.")

    def wake(self) -> None:
        self._wake_event.set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def active_workers(self) -> int:
        return self.worker_pool.active_count

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self.dispatch_due_tasks()
            self._wake_event.wait(self.poll_interval_seconds)
            self._wake_event.clear()

    def dispatch_due_tasks(self) -> None:
        capacity = max(1, self.worker_pool.max_workers * 2)
        due_tasks = self.tasks.list_due(limit=capacity)
        planned_launches: set[int] = set()
        for task in due_tasks:
            if task.id is None:
                continue
            if not self._can_dispatch_task(task, planned_launches):
                continue
            self.tasks.mark_queued(task.id)
            self.worker_pool.submit(task)
            self._emit("queued", task)

    def _can_dispatch_task(self, task: ScheduledTask, planned_launches: set[int]) -> bool:
        if self.instances is None or self.emulator_manager is None or self.settings is None:
            return True
        if task.instance_id is None:
            return True

        instance = self.instances.get(task.instance_id)
        if instance is None or instance.id is None:
            return True
        if self.emulator_manager.is_running(instance):
            return True
        if instance.id in planned_launches:
            return True

        maximum = self.settings.get_int("scheduler.max_active_instances", 5)
        running = self.emulator_manager.running_count()
        if running + len(planned_launches) >= maximum:
            self.logger.info(
                "[MEmu] Maximum concurrent instances reached; task %s remains pending",
                task.id,
            )
            return False

        planned_launches.add(instance.id)
        return True

    def _emit(self, status: str, task: ScheduledTask) -> None:
        for callback in list(self._callbacks):
            callback(status, task)
