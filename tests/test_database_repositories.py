from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.db_helpers import SRC_ROOT  # noqa: F401

from rok_assistant.db.database import Database
from rok_assistant.db.models import Character, Instance, March, ScheduledTask
from rok_assistant.db.repositories import (
    CharacterRepository,
    InstanceRepository,
    MarchRepository,
    TaskRepository,
)


class CoreDatabaseRepositoryTest(unittest.TestCase):
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
                Character(name="Farm01", instance_id=instance_id)
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


if __name__ == "__main__":
    unittest.main()
