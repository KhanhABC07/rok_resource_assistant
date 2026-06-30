from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.db_helpers import SRC_ROOT  # noqa: F401

from rok_assistant.db.database import Database
from rok_assistant.db.models import Instance
from rok_assistant.db.repositories import InstanceRepository, SettingsRepository


class DatabaseTransactionTest(unittest.TestCase):
    def test_transaction_rollback_discards_repository_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "rollback.sqlite3")
            db.initialize()
            instances = InstanceRepository(db)

            with self.assertRaises(RuntimeError):
                with db.transaction():
                    instances.save(Instance(name="Transient"))
                    raise RuntimeError("fail")

            self.assertEqual([], instances.list_all())
            db.close()

    def test_nested_repository_transaction_uses_caller_owned_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "nested.sqlite3")
            db.initialize()
            settings = SettingsRepository(db)

            with db.transaction():
                settings.set_defaults({"a": "1", "b": "2"})

            self.assertEqual({"a": "1", "b": "2"}, settings.all())
            db.close()

    def test_nested_transaction_success_releases_savepoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "nested_success.sqlite3")
            db.initialize()
            settings = SettingsRepository(db)

            with db.transaction():
                settings.set("outer", "1")
                with db.transaction():
                    settings.set("inner", "2")

            self.assertEqual({"inner": "2", "outer": "1"}, settings.all())
            db.close()

    def test_caught_inner_failure_rolls_back_inner_and_commits_outer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "caught_inner.sqlite3")
            db.initialize()
            settings = SettingsRepository(db)

            with db.transaction():
                settings.set("outer", "kept")
                try:
                    with db.transaction():
                        settings.set("inner", "rolled back")
                        raise RuntimeError("inner failed")
                except RuntimeError:
                    pass
                settings.set("after", "kept")

            self.assertEqual({"after": "kept", "outer": "kept"}, settings.all())
            db.close()

    def test_uncaught_inner_failure_rolls_back_outer_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "uncaught_inner.sqlite3")
            db.initialize()
            settings = SettingsRepository(db)

            with self.assertRaisesRegex(RuntimeError, "inner failed"):
                with db.transaction():
                    settings.set("outer", "rolled back")
                    with db.transaction():
                        settings.set("inner", "rolled back")
                        raise RuntimeError("inner failed")

            self.assertEqual({}, settings.all())
            db.close()

    def test_multiple_nested_savepoint_levels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "nested_levels.sqlite3")
            db.initialize()
            settings = SettingsRepository(db)

            with db.transaction():
                settings.set("level1", "kept")
                with db.transaction():
                    settings.set("level2", "kept")
                    try:
                        with db.transaction():
                            settings.set("level3", "rolled back")
                            raise ValueError("deep failure")
                    except ValueError:
                        pass
                    settings.set("level2_after", "kept")

            self.assertEqual(
                {"level1": "kept", "level2": "kept", "level2_after": "kept"},
                settings.all(),
            )
            db.close()

    def test_outer_failure_rolls_back_released_inner_savepoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "outer_failure.sqlite3")
            db.initialize()
            settings = SettingsRepository(db)

            with self.assertRaisesRegex(RuntimeError, "outer failed"):
                with db.transaction():
                    with db.transaction():
                        settings.set("inner", "released")
                    raise RuntimeError("outer failed")

            self.assertEqual({}, settings.all())
            db.close()

    def test_repository_operations_use_same_connection_in_nested_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "same_connection.sqlite3")
            db.initialize()
            settings = SettingsRepository(db)

            with db.transaction() as outer_connection:
                outer_connection.execute("CREATE TEMP TABLE connection_marker(id INTEGER)")
                with db.transaction() as inner_connection:
                    self.assertIs(outer_connection, inner_connection)
                    settings.set_defaults({"repository": "same connection"})
                marker = outer_connection.execute(
                    """
                    SELECT name
                    FROM sqlite_temp_schema
                    WHERE type = 'table' AND name = 'connection_marker'
                    """
                ).fetchone()

            self.assertIsNotNone(marker)
            self.assertEqual({"repository": "same connection"}, settings.all())
            db.close()


if __name__ == "__main__":
    unittest.main()
