from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.db_helpers import FakeActionEngine, FakeAdbManager, SRC_ROOT  # noqa: F401

from rok_assistant.db.database import Database
from rok_assistant.db.models import Task, TaskStep
from rok_assistant.db.repositories import TaskRunHistoryRepository
from rok_assistant.task_engine import TaskResult, TaskRunner


class TaskRunHistoryDatabaseTest(unittest.TestCase):
    def test_successful_task_run_is_stored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "app.sqlite3")
            db.initialize()
            history = TaskRunHistoryRepository(db)
            runner = TaskRunner(
                FakeAdbManager(),  # type: ignore[arg-type]
                action_engine_factory=lambda _index, _name: FakeActionEngine(),  # type: ignore[arg-type]
                history_repository=history,
            )

            result = runner.run_task(
                Task(id=1, name="Success task", enabled=True),
                [
                    TaskStep(
                        order=1,
                        action_type="ClickTemplate",
                        parameters={"template_path": "ok.png", "threshold": 0.8},
                    )
                ],
                instance_index=2,
                instance_name="MEmu2",
            )
            rows = history.list_recent()

            self.assertEqual(TaskResult.SUCCESS, result.result)
            self.assertEqual(1, len(rows))
            self.assertEqual("SUCCESS", rows[0].result)
            self.assertEqual(1, rows[0].task_id)
            self.assertEqual("Success task", rows[0].task_name)
            self.assertEqual(2, rows[0].instance_index)
            self.assertEqual("MEmu2", rows[0].instance_name)
            self.assertEqual("", rows[0].error_message)
            self.assertEqual("", rows[0].abort_reason)
            db.close()

    def test_failed_task_run_is_stored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "app.sqlite3")
            db.initialize()
            history = TaskRunHistoryRepository(db)
            runner = TaskRunner(
                FakeAdbManager(),  # type: ignore[arg-type]
                action_engine_factory=lambda _index, _name: FakeActionEngine(
                    fail_click=True
                ),  # type: ignore[arg-type]
                history_repository=history,
            )

            result = runner.run_task(
                Task(id=2, name="Failed task", enabled=True),
                [
                    TaskStep(
                        order=1,
                        action_type="ClickTemplate",
                        parameters={"template_path": "missing.png", "threshold": 0.8},
                    )
                ],
                instance_index=3,
                instance_name="MEmu3",
            )
            rows = history.list_recent()

            self.assertEqual(TaskResult.FAILED, result.result)
            self.assertEqual(1, len(rows))
            self.assertEqual("FAILED", rows[0].result)
            self.assertEqual("click failed", rows[0].error_message)
            self.assertEqual("", rows[0].abort_reason)
            db.close()

    def test_aborted_task_run_is_stored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "app.sqlite3")
            db.initialize()
            history = TaskRunHistoryRepository(db)
            runner = TaskRunner(
                FakeAdbManager(),  # type: ignore[arg-type]
                action_engine_factory=lambda _index, _name: FakeActionEngine(),  # type: ignore[arg-type]
                history_repository=history,
            )

            result = runner.run_task(
                Task(id=3, name="Aborted task", enabled=True),
                [TaskStep(order=1, action_type="AbortTask", parameters={})],
                instance_index=4,
                instance_name="MEmu4",
            )
            rows = history.list_recent()

            self.assertEqual(TaskResult.ABORTED, result.result)
            self.assertEqual(1, len(rows))
            self.assertEqual("ABORTED", rows[0].result)
            self.assertEqual("", rows[0].error_message)
            self.assertEqual("Task aborted intentionally", rows[0].abort_reason)
            db.close()

    def test_aborted_task_run_reason_is_stored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "app.sqlite3")
            db.initialize()
            history = TaskRunHistoryRepository(db)
            runner = TaskRunner(
                FakeAdbManager(),  # type: ignore[arg-type]
                action_engine_factory=lambda _index, _name: FakeActionEngine(),  # type: ignore[arg-type]
                history_repository=history,
            )

            result = runner.run_task(
                Task(id=4, name="Aborted task", enabled=True),
                [
                    TaskStep(
                        order=1,
                        action_type="AbortTask",
                        parameters={"reason": "No free march"},
                    )
                ],
                instance_index=4,
                instance_name="MEmu4",
            )
            rows = history.list_recent()

            self.assertEqual(TaskResult.ABORTED, result.result)
            self.assertEqual("No free march", result.message)
            self.assertEqual(1, len(rows))
            self.assertEqual("ABORTED", rows[0].result)
            self.assertEqual("", rows[0].error_message)
            self.assertEqual("No free march", rows[0].abort_reason)
            db.close()


if __name__ == "__main__":
    unittest.main()
