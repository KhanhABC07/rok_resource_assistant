from __future__ import annotations

import sys
import tempfile
import unittest
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.db.database import Database
from rok_assistant.db.models import Character, Instance, ScheduledTask
from rok_assistant.db.repositories import (
    CharacterRepository,
    InstanceRepository,
    MarchRepository,
    SettingsRepository,
    TaskRepository,
)
from rok_assistant.recovery import ErrorRecoveryPolicy
from rok_assistant.scheduler import Scheduler, WorkerPool
from rok_assistant.task_result import TaskResult as EngineTaskResult
from rok_assistant.tasks.base import TaskContext, TaskPlugin
from rok_assistant.tasks.base import TaskResult as ScheduledTaskResult
from rok_assistant.tasks.manager import TaskManager


class FakeEmulatorManager:
    def __init__(self, running_instance_ids: set[int] | None = None):
        self.running_instance_ids = running_instance_ids or set()

    def is_running(self, instance: Instance) -> bool:
        return instance.id in self.running_instance_ids

    def running_count(self) -> int:
        return len(self.running_instance_ids)


class FakeWorkerPool:
    def __init__(self) -> None:
        self.max_workers = 5
        self.submitted: list[ScheduledTask] = []

    @property
    def active_count(self) -> int:
        return 0

    def submit(self, task: ScheduledTask) -> None:
        self.submitted.append(task)


class FakeTaskManager:
    def __init__(self, result: object) -> None:
        self.result = result

    def execute(self, task: ScheduledTask) -> object:
        return self.result


class SequenceTaskManager:
    def __init__(self, results: list[object]) -> None:
        self.results = list(results)
        self.executed_task_ids: list[int | None] = []

    def execute(self, task: ScheduledTask) -> object:
        self.executed_task_ids.append(task.id)
        if not self.results:
            raise AssertionError("No task result configured.")
        return self.results.pop(0)


class StaticResultPlugin(TaskPlugin):
    task_type = "static_result"
    display_name = "Static Result"

    def __init__(self, result: ScheduledTaskResult) -> None:
        self.result = result

    def run(self, task_id: int, context: TaskContext) -> ScheduledTaskResult:
        return self.result


class SchedulerMEmuTest(unittest.TestCase):
    def test_dispatch_does_not_queue_new_instances_beyond_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "app.sqlite3")
            db.initialize()
            instances = InstanceRepository(db)
            characters = CharacterRepository(db)
            settings = SettingsRepository(db)
            tasks = TaskRepository(db)
            settings.set("scheduler.max_active_instances", 1)

            first_instance_id = instances.save(
                Instance(name="MEmu", instance_index=0, instance_name="MEmu")
            )
            second_instance_id = instances.save(
                Instance(name="MEmu1", instance_index=1, instance_name="MEmu1")
            )
            first_character_id = characters.save(
                Character(name="Farm01", instance_id=first_instance_id)
            )
            second_character_id = characters.save(
                Character(name="Farm02", instance_id=second_instance_id)
            )
            first_task_id = tasks.enqueue(
                ScheduledTask(
                    character_id=first_character_id,
                    task_type="gathering",
                    priority=10,
                    scheduled_for="2000-01-01T00:00:00",
                )
            )
            second_task_id = tasks.enqueue(
                ScheduledTask(
                    character_id=second_character_id,
                    task_type="gathering",
                    priority=20,
                    scheduled_for="2000-01-01T00:00:00",
                )
            )

            worker_pool = FakeWorkerPool()
            scheduler = Scheduler(
                task_repository=tasks,
                worker_pool=worker_pool,  # type: ignore[arg-type]
                instance_repository=instances,
                emulator_manager=FakeEmulatorManager(),  # type: ignore[arg-type]
                settings=settings,
            )

            scheduler.dispatch_due_tasks()

            submitted_ids = [task.id for task in worker_pool.submitted]
            self.assertEqual([first_task_id], submitted_ids)

            recent = {task.id: task.status for task in tasks.list_recent(limit=10)}
            self.assertEqual("queued", recent[first_task_id])
            self.assertEqual("pending", recent[second_task_id])
            db.close()


class SchedulerTaskResultHandlingTest(unittest.TestCase):
    def _execute_static_result(
        self, result: ScheduledTaskResult
    ) -> tuple[ScheduledTaskResult, ScheduledTask]:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "app.sqlite3")
            db.initialize()
            instances = InstanceRepository(db)
            characters = CharacterRepository(db)
            marches = MarchRepository(db)
            settings = SettingsRepository(db)
            tasks = TaskRepository(db)

            instance_id = instances.save(
                Instance(name="MEmu_0", instance_index=0, instance_name="MEmu_0")
            )
            character_id = characters.save(
                Character(name="Farm01", instance_id=instance_id)
            )
            task_id = tasks.enqueue(
                ScheduledTask(
                    character_id=character_id,
                    task_type=StaticResultPlugin.task_type,
                    scheduled_for="2000-01-01T00:00:00",
                )
            )
            scheduled_task = next(task for task in tasks.list_recent() if task.id == task_id)
            context = TaskContext(
                instances=instances,
                characters=characters,
                marches=marches,
                settings=settings,
                emulator_manager=object(),  # type: ignore[arg-type]
                character_manager=object(),  # type: ignore[arg-type]
                vision=object(),  # type: ignore[arg-type]
                logger=logging.getLogger("TaskContextTest"),
            )
            manager = TaskManager(
                task_repository=tasks,
                context=context,
                recovery_policy=ErrorRecoveryPolicy(max_attempts=1),
            )
            manager.register(StaticResultPlugin(result))

            execution_result = manager.execute(scheduled_task)
            stored_task = next(task for task in tasks.list_recent() if task.id == task_id)
            db.close()
            return execution_result, stored_task

    def _run_worker_pool(self, result: object) -> tuple[list[str], list[str]]:
        statuses: list[str] = []
        worker_pool = WorkerPool(
            task_manager=FakeTaskManager(result),  # type: ignore[arg-type]
            max_workers=1,
            status_callback=lambda status, _task: statuses.append(status),
        )
        task = ScheduledTask(
            id=1,
            task_type="static_result",
            instance_name="MEmu_0",
            priority=1,
        )
        with self.assertLogs("WorkerPool", level="INFO") as logs:
            worker_pool.start()
            try:
                worker_pool.submit(task)
                worker_pool._queue.join()
            finally:
                worker_pool.stop()
        return statuses, logs.output

    def _run_two_worker_tasks(
        self, first_result: object, second_result: object
    ) -> tuple[SequenceTaskManager, list[tuple[str, int | None]], int]:
        statuses: list[tuple[str, int | None]] = []
        task_manager = SequenceTaskManager([first_result, second_result])
        worker_pool = WorkerPool(
            task_manager=task_manager,  # type: ignore[arg-type]
            max_workers=1,
            status_callback=lambda status, task: statuses.append((status, task.id)),
        )
        worker_pool.start()
        try:
            worker_pool.submit(
                ScheduledTask(
                    id=1,
                    task_type="first",
                    instance_name="MEmu_0",
                    priority=1,
                )
            )
            worker_pool.submit(
                ScheduledTask(
                    id=2,
                    task_type="second",
                    instance_name="MEmu_0",
                    priority=2,
                )
            )
            worker_pool._queue.join()
            active_count = worker_pool.active_count
        finally:
            worker_pool.stop()
        return task_manager, statuses, active_count

    def _active_count_after_worker_result(self, result: object) -> int:
        worker_pool = WorkerPool(
            task_manager=FakeTaskManager(result),  # type: ignore[arg-type]
            max_workers=1,
        )
        worker_pool.start()
        try:
            worker_pool.submit(
                ScheduledTask(
                    id=1,
                    task_type="static_result",
                    instance_name="MEmu_0",
                    priority=1,
                )
            )
            worker_pool._queue.join()
            return worker_pool.active_count
        finally:
            worker_pool.stop()

    def test_scheduler_handles_success(self) -> None:
        result, stored_task = self._execute_static_result(
            ScheduledTaskResult(True, "done", result=EngineTaskResult.SUCCESS)
        )
        statuses, logs = self._run_worker_pool(result)

        self.assertEqual(EngineTaskResult.SUCCESS, result.result)
        self.assertEqual("completed", stored_task.status)
        self.assertEqual("SUCCESS", stored_task.result)
        self.assertEqual(["running", "completed"], statuses)
        self.assertTrue(any("TaskResult=SUCCESS" in message for message in logs))

    def test_scheduler_handles_failed(self) -> None:
        result, stored_task = self._execute_static_result(
            ScheduledTaskResult(False, "broken", result=EngineTaskResult.FAILED)
        )
        statuses, logs = self._run_worker_pool(result)

        self.assertEqual(EngineTaskResult.FAILED, result.result)
        self.assertEqual("failed", stored_task.status)
        self.assertEqual("FAILED", stored_task.result)
        self.assertEqual(["running", "failed"], statuses)
        self.assertTrue(any("TaskResult=FAILED" in message for message in logs))
        self.assertTrue(any(message.startswith("ERROR:WorkerPool") for message in logs))

    def test_scheduler_handles_aborted(self) -> None:
        result, stored_task = self._execute_static_result(
            ScheduledTaskResult(False, "stopped by user", result=EngineTaskResult.ABORTED)
        )
        statuses, logs = self._run_worker_pool(result)

        self.assertEqual(EngineTaskResult.ABORTED, result.result)
        self.assertEqual("aborted", stored_task.status)
        self.assertEqual("ABORTED", stored_task.result)
        self.assertEqual(["running", "aborted"], statuses)
        self.assertTrue(any("TaskResult=ABORTED" in message for message in logs))

    def test_aborted_does_not_count_as_technical_failure(self) -> None:
        result, stored_task = self._execute_static_result(
            ScheduledTaskResult(False, "stopped intentionally", result=EngineTaskResult.ABORTED)
        )
        statuses, logs = self._run_worker_pool(result)

        self.assertEqual("aborted", stored_task.status)
        self.assertNotEqual("failed", stored_task.status)
        self.assertNotIn("failed", statuses)
        self.assertTrue(all(not message.startswith("ERROR:WorkerPool") for message in logs))

    def test_worker_continues_to_next_queued_task_after_aborted_result(self) -> None:
        task_manager, statuses, active_count = self._run_two_worker_tasks(
            ScheduledTaskResult(False, "stopped", result=EngineTaskResult.ABORTED),
            ScheduledTaskResult(True, "done", result=EngineTaskResult.SUCCESS),
        )

        self.assertEqual([1, 2], task_manager.executed_task_ids)
        self.assertEqual(
            [("running", 1), ("aborted", 1), ("running", 2), ("completed", 2)],
            statuses,
        )
        self.assertEqual(0, active_count)

    def test_worker_continues_to_next_queued_task_after_failed_result(self) -> None:
        task_manager, statuses, active_count = self._run_two_worker_tasks(
            ScheduledTaskResult(False, "broken", result=EngineTaskResult.FAILED),
            ScheduledTaskResult(True, "done", result=EngineTaskResult.SUCCESS),
        )

        self.assertEqual([1, 2], task_manager.executed_task_ids)
        self.assertEqual(
            [("running", 1), ("failed", 1), ("running", 2), ("completed", 2)],
            statuses,
        )
        self.assertEqual(0, active_count)

    def test_worker_slot_is_released_after_success(self) -> None:
        active_count = self._active_count_after_worker_result(
            ScheduledTaskResult(True, "done", result=EngineTaskResult.SUCCESS)
        )

        self.assertEqual(0, active_count)

    def test_worker_slot_is_released_after_aborted(self) -> None:
        active_count = self._active_count_after_worker_result(
            ScheduledTaskResult(False, "stopped", result=EngineTaskResult.ABORTED)
        )

        self.assertEqual(0, active_count)

    def test_worker_slot_is_released_after_failed(self) -> None:
        active_count = self._active_count_after_worker_result(
            ScheduledTaskResult(False, "broken", result=EngineTaskResult.FAILED)
        )

        self.assertEqual(0, active_count)

    def test_running_memu_instance_is_reused_after_previous_task_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "app.sqlite3")
            db.initialize()
            instances = InstanceRepository(db)
            characters = CharacterRepository(db)
            settings = SettingsRepository(db)
            tasks = TaskRepository(db)
            settings.set("scheduler.max_active_instances", 1)

            instance_id = instances.save(
                Instance(name="MEmu_0", instance_index=0, instance_name="MEmu_0")
            )
            character_id = characters.save(
                Character(name="Farm01", instance_id=instance_id)
            )
            first_task_id = tasks.enqueue(
                ScheduledTask(
                    character_id=character_id,
                    task_type="gathering",
                    priority=10,
                    scheduled_for="2000-01-01T00:00:00",
                )
            )
            second_task_id = tasks.enqueue(
                ScheduledTask(
                    character_id=character_id,
                    task_type="gathering",
                    priority=20,
                    scheduled_for="2000-01-01T00:00:00",
                )
            )
            tasks.mark_completed(first_task_id, "done")

            worker_pool = FakeWorkerPool()
            scheduler = Scheduler(
                task_repository=tasks,
                worker_pool=worker_pool,  # type: ignore[arg-type]
                instance_repository=instances,
                emulator_manager=FakeEmulatorManager({instance_id}),  # type: ignore[arg-type]
                settings=settings,
            )

            scheduler.dispatch_due_tasks()

            self.assertEqual([second_task_id], [task.id for task in worker_pool.submitted])
            stored = next(task for task in tasks.list_recent() if task.id == second_task_id)
            self.assertEqual("queued", stored.status)
            db.close()


if __name__ == "__main__":
    unittest.main()
