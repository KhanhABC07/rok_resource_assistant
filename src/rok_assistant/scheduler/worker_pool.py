from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from itertools import count
from time import monotonic
from typing import Callable

from rok_assistant.db.models import ScheduledTask
from rok_assistant.task_result import TaskResult
from rok_assistant.tasks.manager import TaskManager


@dataclass(order=True)
class WorkItem:
    priority: int
    sequence: int
    task: ScheduledTask = field(compare=False)


StatusCallback = Callable[[str, ScheduledTask], None]


class WorkerPool:
    def __init__(
        self,
        task_manager: TaskManager,
        max_workers: int = 5,
        status_callback: StatusCallback | None = None,
    ):
        self.task_manager = task_manager
        self.max_workers = max_workers
        self.status_callback = status_callback
        self.logger = logging.getLogger(self.__class__.__name__)
        self._queue: queue.PriorityQueue[WorkItem] = queue.PriorityQueue()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._sequence = count()
        self._active_count = 0
        self._active_lock = threading.RLock()

    def start(self) -> None:
        if self._threads:
            return
        self._stop_event.clear()
        for index in range(self.max_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"rok-worker-{index + 1}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        self.logger.info("Worker pool started with %s workers.", self.max_workers)

    def stop(self) -> None:
        self._stop_event.set()
        for _ in self._threads:
            self._queue.put(WorkItem(priority=999999, sequence=next(self._sequence), task=ScheduledTask()))
        for thread in self._threads:
            thread.join(timeout=2)
        self._threads.clear()
        self.logger.info("Worker pool stopped.")

    def submit(self, task: ScheduledTask) -> None:
        self._queue.put(WorkItem(task.priority, next(self._sequence), task))

    @property
    def active_count(self) -> int:
        with self._active_lock:
            return self._active_count

    @property
    def queued_count(self) -> int:
        return self._queue.qsize()

    def _set_active_delta(self, delta: int) -> None:
        with self._active_lock:
            self._active_count += delta

    def _emit(self, status: str, task: ScheduledTask) -> None:
        if self.status_callback is not None:
            self.status_callback(status, task)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            task = item.task
            if task.id is None:
                self._queue.task_done()
                continue

            started = monotonic()
            self._set_active_delta(1)
            self._emit("running", task)
            try:
                result = self.task_manager.execute(task)
                task_result = self._task_result(result)
                status = self._status_for_result(task_result)
                log = self.logger.error if task_result == TaskResult.FAILED else self.logger.info
                log(
                    "Task finished: instance=%s task=%s TaskResult=%s status=%s "
                    "duration=%.2fs message=%s",
                    task.instance_name or task.instance_id,
                    task.task_type,
                    task_result.name,
                    status,
                    monotonic() - started,
                    self._message(result),
                )
                self._emit(status, task)
            finally:
                self._set_active_delta(-1)
                self._queue.task_done()

    @staticmethod
    def _task_result(result: object) -> TaskResult:
        if isinstance(result, TaskResult):
            return result
        value = getattr(result, "result", None)
        if isinstance(value, TaskResult):
            return value
        if isinstance(value, str):
            try:
                return TaskResult(value)
            except ValueError:
                pass
        success = bool(getattr(result, "success", False))
        return TaskResult.SUCCESS if success else TaskResult.FAILED

    @staticmethod
    def _status_for_result(result: TaskResult) -> str:
        if result == TaskResult.SUCCESS:
            return "completed"
        if result == TaskResult.ABORTED:
            return "aborted"
        return "failed"

    @staticmethod
    def _message(result: object) -> str:
        return str(getattr(result, "message", ""))
