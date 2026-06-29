from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.db.database import Database
from rok_assistant.db.models import Character, Instance, March, ScheduledTask, Task, TaskStep
from rok_assistant.db.repositories import (
    AutomationTaskRepository,
    CharacterRepository,
    InstanceRepository,
    MarchRepository,
    TaskRunHistoryRepository,
    TaskRepository,
)
from rok_assistant.task_engine import TaskResult, TaskRunner


class FakeAdbManager:
    pass


class FakeActionEngine:
    def __init__(self, *, fail_click: bool = False) -> None:
        self.fail_click = fail_click

    def click_template(self, template_path: str, *, threshold: float) -> dict[str, object]:
        if self.fail_click:
            return {"success": False, "message": "click failed"}
        return {"success": True, "template_path": template_path, "threshold": threshold}

    def abort_task(self, reason: str | None = None) -> dict[str, object]:
        abort_reason = str(reason or "").strip() or "Task aborted intentionally"
        return {
            "success": True,
            "aborted": True,
            "message": abort_reason,
            "abort_reason": abort_reason,
        }


class DatabaseRepositoryTest(unittest.TestCase):
    def test_repositories_create_default_marches_and_due_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "app.sqlite3")
            db.initialize()
            instances = InstanceRepository(db)
            characters = CharacterRepository(db)
            marches = MarchRepository(db)
            tasks = TaskRepository(db)

            instance_id = instances.save(
                Instance(
                    name="LD01",
                    instance_index=3,
                    instance_name="MEmu3",
                    launch_command="echo launch",
                    close_command="echo close",
                )
            )
            saved_instance = instances.get(instance_id)
            self.assertIsNotNone(saved_instance)
            self.assertEqual(3, saved_instance.instance_index)
            self.assertEqual("MEmu3", saved_instance.instance_name)
            character_id = characters.save(
                Character(name="Farm01", instance_id=instance_id, account_name="Account A")
            )

            default_marches = marches.list_for_character(character_id)
            self.assertEqual(5, len(default_marches))

            marches.save(
                March(
                    character_id=character_id,
                    march_slot=1,
                    status="returning",
                )
            )

            task_id = tasks.enqueue(
                ScheduledTask(
                    character_id=character_id,
                    march_slot=1,
                    task_type="gathering",
                    priority=10,
                    scheduled_for="2000-01-01T00:00:00",
                )
            )
            self.assertTrue(tasks.open_task_exists(character_id, 1, "gathering"))
            due = tasks.list_due(limit=10)
            self.assertEqual(task_id, due[0].id)
            db.close()

    def test_legacy_march_resource_columns_still_load_and_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy_resources.sqlite3"
            db = Database(path)
            db.initialize()
            instances = InstanceRepository(db)
            characters = CharacterRepository(db)
            marches = MarchRepository(db)

            instance_id = instances.save(Instance(name="Legacy"))
            character_id = characters.save(
                Character(name="Farm01", instance_id=instance_id)
            )
            db.execute(
                """
                UPDATE marches
                SET resource_type = 'Wood',
                    resource_source = 'Alliance Resource Pit'
                WHERE character_id = ? AND march_slot = 1
                """,
                (character_id,),
            )
            db.close()

            reopened = Database(path)
            reopened.initialize()
            reopened_marches = MarchRepository(reopened)
            legacy = reopened_marches.list_for_character(character_id)[0]
            self.assertEqual("Wood", legacy.resource_type)
            self.assertEqual("Alliance Resource Pit", legacy.resource_source)

            legacy.status = "returning"
            reopened_marches.save(legacy)
            row = reopened.fetch_one(
                """
                SELECT resource_type, resource_source, status
                FROM marches
                WHERE character_id = ? AND march_slot = 1
                """,
                (character_id,),
            )
            self.assertEqual("Wood", row["resource_type"])
            self.assertEqual("Alliance Resource Pit", row["resource_source"])
            self.assertEqual("returning", row["status"])
            reopened.close()

    def test_scheduled_task_resource_type_round_trips_in_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "resource_task.sqlite3")
            db.initialize()
            instances = InstanceRepository(db)
            characters = CharacterRepository(db)
            tasks = TaskRepository(db)

            instance_id = instances.save(Instance(name="MEmu"))
            character_id = characters.save(
                Character(name="Farm01", instance_id=instance_id)
            )
            task_id = tasks.enqueue(
                ScheduledTask(
                    character_id=character_id,
                    task_type="gathering",
                    resource_type="Stone",
                    scheduled_for="2000-01-01T00:00:00",
                )
            )

            stored = next(task for task in tasks.list_recent() if task.id == task_id)
            self.assertEqual("Stone", stored.resource_type)
            self.assertEqual("Stone", json.loads(stored.payload_json)["resource_type"])
            db.close()

    def test_existing_database_migrates_memu_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute(
                """
                CREATE TABLE instances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    launch_path TEXT NOT NULL DEFAULT '',
                    launch_command TEXT NOT NULL DEFAULT '',
                    close_command TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute("INSERT INTO instances(name) VALUES ('Legacy')")
            connection.commit()
            connection.close()

            db = Database(path)
            db.initialize()
            instances = InstanceRepository(db)

            legacy = instances.list_all()[0]
            self.assertEqual("Legacy", legacy.name)
            self.assertEqual("Legacy", legacy.instance_name)
            self.assertIsNone(legacy.instance_index)
            self.assertEqual("", legacy.adb_serial)
            self.assertFalse(legacy.adb_connected)

            imported_id = instances.upsert_memu_instance(0, "MEmu")
            imported = instances.get(imported_id)
            self.assertIsNotNone(imported)
            self.assertEqual(0, imported.instance_index)
            self.assertEqual("MEmu", imported.instance_name)

            instances.update_adb_status(0, "127.0.0.1:21503", True)
            imported = instances.get(imported_id)
            self.assertIsNotNone(imported)
            self.assertEqual("127.0.0.1:21503", imported.adb_serial)
            self.assertTrue(imported.adb_connected)
            db.close()

    def test_existing_database_migrates_automation_repeat_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy_tasks.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE automation_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE automation_task_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    step_order INTEGER NOT NULL,
                    action_type TEXT NOT NULL
                        CHECK(action_type IN (
                            'WaitTemplate',
                            'ClickTemplate',
                            'ClickCoordinates',
                            'SwipeCoordinates',
                            'Delay'
                        )),
                    parameters TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(task_id) REFERENCES automation_tasks(id) ON DELETE CASCADE,
                    UNIQUE(task_id, step_order)
                );
                """
            )
            connection.execute("INSERT INTO automation_tasks(name) VALUES ('Legacy Task')")
            connection.execute(
                """
                INSERT INTO automation_task_steps(task_id, step_order, action_type, parameters)
                VALUES (1, 1, 'Delay', '{"seconds": 1}')
                """
            )
            connection.commit()
            connection.close()

            db = Database(path)
            db.initialize()
            repo = AutomationTaskRepository(db)

            repo.add_step(1, "RepeatStart", {"count": 3})
            repo.add_step(1, "RepeatEnd", {})
            repo.add_step(1, "IfTemplateExists", {"template_path": "ready.png"})
            repo.add_step(1, "Else", {})
            repo.add_step(1, "EndIf", {})
            self.assertEqual(
                ["Delay", "RepeatStart", "RepeatEnd", "IfTemplateExists", "Else", "EndIf"],
                [step.action_type for step in repo.list_steps(1)],
            )
            db.close()

    def test_task_run_history_migration_creates_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "app.sqlite3")
            db.initialize()

            table = db.fetch_one(
                """
                SELECT name
                FROM sqlite_schema
                WHERE type = 'table' AND name = 'task_run_history'
                """
            )
            columns = {
                row["name"]
                for row in db.fetch_all("PRAGMA table_info(task_run_history)")
            }

            self.assertIsNotNone(table)
            self.assertLessEqual(
                {
                    "id",
                    "task_id",
                    "task_name",
                    "instance_index",
                    "instance_name",
                    "started_at",
                    "finished_at",
                    "result",
                    "error_message",
                    "abort_reason",
                    "created_at",
                },
                columns,
            )
            db.close()

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

    def test_existing_database_migrates_task_run_history_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy_history.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute(
                """
                CREATE TABLE instances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    launch_path TEXT NOT NULL DEFAULT '',
                    launch_command TEXT NOT NULL DEFAULT '',
                    close_command TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute("INSERT INTO instances(name) VALUES ('Legacy')")
            connection.commit()
            connection.close()

            db = Database(path)
            db.initialize()
            history = TaskRunHistoryRepository(db)
            runner = TaskRunner(
                FakeAdbManager(),  # type: ignore[arg-type]
                action_engine_factory=lambda _index, _name: FakeActionEngine(),  # type: ignore[arg-type]
                history_repository=history,
            )

            runner.run_task(
                Task(id=4, name="Migrated task", enabled=True),
                [
                    TaskStep(
                        order=1,
                        action_type="ClickTemplate",
                        parameters={"template_path": "ok.png", "threshold": 0.8},
                    )
                ],
                instance_index=0,
                instance_name="Legacy",
            )

            self.assertEqual(["SUCCESS"], [row.result for row in history.list_recent()])
            db.close()


if __name__ == "__main__":
    unittest.main()
