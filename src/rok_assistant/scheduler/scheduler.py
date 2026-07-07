from __future__ import annotations

import logging
import math
import threading
from typing import Callable

from rok_assistant.db.models import ScheduledTask
from rok_assistant.db.repositories import (
    InstanceCircuitBreakerRepository,
    InstanceRepository,
    SettingsRepository,
    TaskRepository,
)
from rok_assistant.emulator import EmulatorManager
from rok_assistant.scheduler.service import SchedulerService
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
        v2_service: SchedulerService | None = None,
        circuit_breakers: InstanceCircuitBreakerRepository | None = None,
    ) -> None:
        self.tasks = task_repository
        self.worker_pool = worker_pool
        self.poll_interval_seconds = _validate_poll_interval(poll_interval_seconds)
        self.instances = instance_repository
        self.emulator_manager = emulator_manager
        self.settings = settings
        self.v2_service = v2_service
        self.circuit_breakers = circuit_breakers
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._state_lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._callbacks: list[SchedulerCallback] = []

    def add_callback(self, callback: SchedulerCallback) -> None:
        self._callbacks.append(callback)

    def start(self) -> None:
        with self._state_lock:
            if self._thread and self._thread.is_alive():
                self.logger.info("Scheduler start ignored; already running.")
                return
            self._stop_event.clear()
            self._wake_event.clear()
            self.worker_pool.start()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="rok-scheduler",
                daemon=True,
            )
            self._thread.start()
        self.logger.info("Scheduler started.")

    def stop(self) -> None:
        with self._state_lock:
            thread = self._thread
            self._stop_event.set()
            self._wake_event.set()
        if thread:
            thread.join(timeout=3)
        with self._state_lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None
        self.worker_pool.stop()
        self.logger.info("Scheduler stopped.")

    def wake(self) -> None:
        self._wake_event.set()
        if self.v2_service is not None:
            self.v2_service.wake()
        self.logger.info("Scheduler wake requested.")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def active_workers(self) -> int:
        return self.worker_pool.active_count

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                self.logger.error(
                    "Scheduler iteration failed: %s %s",
                    exc.__class__.__name__,
                    _safe_message(exc),
                )
            if self._stop_event.is_set():
                break
            self._wake_event.wait(self.poll_interval_seconds)
            self._wake_event.clear()

    def run_once(self) -> None:
        if self.v2_service is not None:
            self.v2_service.run_once()
        self.dispatch_due_tasks()

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
        if self.circuit_breakers is not None and self.circuit_breakers.is_open(instance.id):
            self.logger.warning(
                "[Recovery] Instance %s circuit breaker is open; task %s remains pending",
                instance.name,
                task.id,
            )
            return False
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


def _validate_poll_interval(value: int | float) -> float:
    if isinstance(value, bool):
        raise ValueError("poll_interval_seconds must be a finite positive number.")
    interval = float(value)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("poll_interval_seconds must be a finite positive number.")
    return interval


def _safe_message(exc: Exception) -> str:
    message = str(exc).strip()
    if len(message) > 300:
        return message[:297] + "..."
    return message or exc.__class__.__name__
