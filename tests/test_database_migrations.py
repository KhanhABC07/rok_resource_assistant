from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tests.db_helpers import LEGACY_CURRENT_SCHEMA_SQL, SRC_ROOT  # noqa: F401

from rok_assistant.db import migrations
from rok_assistant.db.database import Database, DatabaseRestoreError
from rok_assistant.db.repositories import (
    AutomationTaskRepository,
    InstanceRepository,
    TaskRunHistoryRepository,
)
from rok_assistant.db.models import Instance, Task, TaskStep
from rok_assistant.task_engine import TaskRunner
from tests.db_helpers import FakeActionEngine, FakeAdbManager


def migration_versions(db: Database) -> list[int]:
    return [
        row["version"]
        for row in db.fetch_all("SELECT version FROM schema_migrations ORDER BY version")
    ]


class DatabaseMigrationTest(unittest.TestCase):
    def test_fresh_database_creates_v2_schema_and_records_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "fresh_v2.sqlite3")
            db.initialize()

            tables = {
                row["name"]
                for row in db.fetch_all(
                    "SELECT name FROM sqlite_schema WHERE type = 'table'"
                )
            }
            self.assertLessEqual(migrations.DATA_V2_TABLES | migrations.RECOVERY_V3_TABLES, tables)
            self.assertEqual([1, 2, 3, 4, 5], migration_versions(db))
            self.assertEqual(1, db.fetch_one("PRAGMA foreign_keys")["foreign_keys"])
            self.assertEqual("wal", db.fetch_one("PRAGMA journal_mode")["journal_mode"])
            db.close()

    def test_repeated_migration_execution_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "repeat.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript(LEGACY_CURRENT_SCHEMA_SQL)
            connection.commit()
            connection.close()

            db = Database(path)
            db.initialize()
            self.assertEqual([1, 2, 3, 4, 5], migration_versions(db))
            db.close()
            first_backups = list(Path(temp_dir).glob("repeat.backup.*.sqlite3"))

            reopened = Database(path)
            reopened.initialize()
            self.assertEqual([1, 2, 3, 4, 5], migration_versions(reopened))
            reopened.close()
            second_backups = list(Path(temp_dir).glob("repeat.backup.*.sqlite3"))
            self.assertEqual(first_backups, second_backups)

    def test_legacy_current_schema_migrates_accounts_with_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy_v2.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript(LEGACY_CURRENT_SCHEMA_SQL)
            connection.execute(
                """
                INSERT INTO instances(name, instance_name)
                VALUES ('MEmu0', 'MEmu0')
                """
            )
            connection.execute(
                """
                INSERT INTO characters(name, instance_id, account_name)
                VALUES ('Farm01', 1, 'Account A')
                """
            )
            connection.commit()
            connection.close()

            db = Database(path)
            db.initialize()

            backups = list(Path(temp_dir).glob("legacy_v2.backup.*.sqlite3"))
            self.assertEqual(1, len(backups))
            backup_connection = sqlite3.connect(backups[0])
            try:
                backup_table = backup_connection.execute(
                    """
                    SELECT name
                    FROM sqlite_schema
                    WHERE type = 'table' AND name = 'game_accounts'
                    """
                ).fetchone()
                backup_account_name = backup_connection.execute(
                    "SELECT account_name FROM characters WHERE id = 1"
                ).fetchone()[0]
            finally:
                backup_connection.close()
            self.assertIsNone(backup_table)
            self.assertEqual("Account A", backup_account_name)

            account = db.fetch_one(
                "SELECT id, account_name FROM game_accounts WHERE account_name = ?",
                ("Account A",),
            )
            self.assertIsNotNone(account)
            character = db.fetch_one(
                """
                SELECT account_name, game_account_id
                FROM characters
                WHERE name = 'Farm01'
                """
            )
            self.assertEqual("Account A", character["account_name"])
            self.assertEqual(account["id"], character["game_account_id"])
            self.assertEqual([1, 2, 3, 4, 5], migration_versions(db))
            db.close()

    def test_existing_v3_database_migrates_game_account_secret_ref(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "v3_accounts.sqlite3"
            db = Database(path)
            db.initialize()
            db.close()

            connection = sqlite3.connect(path)
            try:
                connection.execute("DELETE FROM schema_migrations WHERE version IN (4, 5)")
                connection.execute("ALTER TABLE game_accounts RENAME TO game_accounts_v4")
                connection.execute(
                    """
                    CREATE TABLE game_accounts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        account_name TEXT NOT NULL COLLATE NOCASE
                            CHECK(length(trim(account_name)) > 0),
                        display_name TEXT NOT NULL DEFAULT '',
                        provider TEXT NOT NULL DEFAULT '',
                        external_id TEXT NOT NULL DEFAULT '',
                        enabled INTEGER NOT NULL DEFAULT 1,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(account_name)
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO game_accounts(
                        id, account_name, display_name, provider, external_id,
                        enabled, metadata_json, created_at, updated_at
                    )
                    SELECT
                        id, account_name, display_name, provider, external_id,
                        enabled, metadata_json, created_at, updated_at
                    FROM game_accounts_v4
                    """
                )
                connection.execute("DROP TABLE game_accounts_v4")
                connection.execute(
                    """
                    INSERT INTO game_accounts(account_name, display_name)
                    VALUES ('Account A', 'Account A')
                    """
                )
                connection.commit()
            finally:
                connection.close()

            migrated = Database(path)
            migrated.initialize()
            columns = {
                row["name"]
                for row in migrated.fetch_all("PRAGMA table_info(game_accounts)")
            }
            account = migrated.fetch_one(
                "SELECT secret_ref FROM game_accounts WHERE account_name = ?",
                ("Account A",),
            )
            self.assertIn("secret_ref", columns)
            self.assertEqual("", account["secret_ref"])
            self.assertEqual([1, 2, 3, 4, 5], migration_versions(migrated))
            migrated.close()

    def test_existing_v4_database_migrates_character_verification_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "v4_characters.sqlite3"
            db = Database(path)
            db.initialize()
            db.close()

            connection = sqlite3.connect(path)
            try:
                connection.execute("DELETE FROM schema_migrations WHERE version = 5")
                connection.execute("DROP INDEX IF EXISTS idx_characters_account_slot")
                connection.execute("ALTER TABLE characters RENAME TO characters_v5")
                connection.execute(
                    """
                    CREATE TABLE characters (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        instance_id INTEGER NOT NULL,
                        account_name TEXT NOT NULL DEFAULT '',
                        enabled INTEGER NOT NULL DEFAULT 1,
                        alliance_help_enabled INTEGER NOT NULL DEFAULT 1,
                        alliance_donate_enabled INTEGER NOT NULL DEFAULT 1,
                        gift_collection_enabled INTEGER NOT NULL DEFAULT 1,
                        last_switch_at TEXT,
                        game_account_id INTEGER,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(instance_id) REFERENCES instances(id) ON DELETE CASCADE,
                        FOREIGN KEY(game_account_id) REFERENCES game_accounts(id) ON DELETE SET NULL,
                        UNIQUE(instance_id, name)
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO characters(
                        id, name, instance_id, account_name, enabled,
                        alliance_help_enabled, alliance_donate_enabled,
                        gift_collection_enabled, last_switch_at, game_account_id,
                        created_at, updated_at
                    )
                    SELECT
                        id, name, instance_id, account_name, enabled,
                        alliance_help_enabled, alliance_donate_enabled,
                        gift_collection_enabled, last_switch_at, game_account_id,
                        created_at, updated_at
                    FROM characters_v5
                    """
                )
                connection.execute("DROP TABLE characters_v5")
                connection.commit()
            finally:
                connection.close()

            migrated = Database(path)
            migrated.initialize()
            columns = {
                row["name"]
                for row in migrated.fetch_all("PRAGMA table_info(characters)")
            }
            self.assertLessEqual(
                {
                    "character_slot",
                    "display_fingerprint",
                    "kingdom_id",
                    "verification_metadata_json",
                },
                columns,
            )
            create_sql = migrated.fetch_one(
                """
                SELECT sql
                FROM sqlite_schema
                WHERE type = 'table' AND name = 'characters'
                """
            )["sql"]
            self.assertNotIn("UNIQUE(instance_id, name)", create_sql)
            slot_index = migrated.fetch_one(
                """
                SELECT name
                FROM sqlite_schema
                WHERE type = 'index' AND name = 'idx_characters_account_slot'
                """
            )
            self.assertIsNotNone(slot_index)
            self.assertEqual([1, 2, 3, 4, 5], migration_versions(migrated))
            migrated.close()

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

    def test_migration_rolls_back_atomic_version_on_validation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollback.sqlite3"

            def apply(connection: sqlite3.Connection) -> None:
                connection.execute("CREATE TABLE rollback_marker(id INTEGER)")

            def validate(_connection: sqlite3.Connection) -> None:
                raise RuntimeError("validation failed")

            original = migrations.MIGRATIONS
            migrations.MIGRATIONS = (
                migrations.Migration(1, "bad", apply, validate),
            )
            db = Database(path)
            try:
                with self.assertRaisesRegex(RuntimeError, "validation failed"):
                    db.initialize()
            finally:
                db.close()
                migrations.MIGRATIONS = original

            connection = sqlite3.connect(path)
            try:
                marker = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_schema
                    WHERE type = 'table' AND name = 'rollback_marker'
                    """
                ).fetchone()
                version_table = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_schema
                    WHERE type = 'table' AND name = 'schema_migrations'
                    """
                ).fetchone()
            finally:
                connection.close()
            self.assertIsNone(marker)
            self.assertIsNone(version_table)

    def test_migration_version_not_recorded_on_validation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "version_failure.sqlite3"

            def apply(connection: sqlite3.Connection) -> None:
                connection.execute("CREATE TABLE created_before_failure(id INTEGER)")

            def validate(_connection: sqlite3.Connection) -> None:
                raise RuntimeError("validation failed")

            original = migrations.MIGRATIONS
            migrations.MIGRATIONS = (
                migrations.Migration(1, "bad", apply, validate),
            )
            db = Database(path)
            try:
                with self.assertRaises(RuntimeError):
                    db.initialize()
            finally:
                db.close()
                migrations.MIGRATIONS = original

            connection = sqlite3.connect(path)
            try:
                version = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_schema
                    WHERE type = 'table' AND name = 'schema_migrations'
                    """
                ).fetchone()
            finally:
                connection.close()
            self.assertIsNone(version)

    def test_foreign_key_check_failure_rolls_back_and_restores_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fk_failure.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE instances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE
                );
                CREATE TABLE characters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    instance_id INTEGER NOT NULL,
                    account_name TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    alliance_help_enabled INTEGER NOT NULL DEFAULT 1,
                    alliance_donate_enabled INTEGER NOT NULL DEFAULT 1,
                    gift_collection_enabled INTEGER NOT NULL DEFAULT 1,
                    last_switch_at TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(instance_id) REFERENCES instances(id) ON DELETE CASCADE
                );
                INSERT INTO characters(name, instance_id) VALUES ('Broken', 999);
                """
            )
            connection.commit()
            connection.close()

            db = Database(path)
            try:
                with self.assertRaisesRegex(
                    DatabaseRestoreError,
                    "Restore database foreign key validation failed",
                ):
                    db.initialize()
            finally:
                db.close()

            restored = sqlite3.connect(path)
            try:
                version_table = restored.execute(
                    """
                    SELECT name
                    FROM sqlite_schema
                    WHERE type = 'table' AND name = 'schema_migrations'
                    """
                ).fetchone()
                character_name = restored.execute(
                    "SELECT name FROM characters WHERE id = 1"
                ).fetchone()[0]
            finally:
                restored.close()
            self.assertIsNone(version_table)
            self.assertEqual("Broken", character_name)

    def test_backup_restore_keeps_original_database_on_migration_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "restore.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE existing_data(id INTEGER PRIMARY KEY, name TEXT NOT NULL)"
            )
            connection.execute("INSERT INTO existing_data(id, name) VALUES (1, 'safe')")
            connection.commit()
            connection.close()

            def apply(connection: sqlite3.Connection) -> None:
                connection.execute("CREATE TABLE should_not_survive(id INTEGER)")

            def validate(_connection: sqlite3.Connection) -> None:
                raise RuntimeError("validation failed")

            original = migrations.MIGRATIONS
            migrations.MIGRATIONS = (
                migrations.Migration(1, "bad", apply, validate),
            )
            db = Database(path)
            try:
                with self.assertRaises(RuntimeError):
                    db.initialize()
            finally:
                db.close()
                migrations.MIGRATIONS = original

            backups = list(Path(temp_dir).glob("restore.backup.*.sqlite3"))
            self.assertEqual(1, len(backups))
            restored = sqlite3.connect(path)
            try:
                value = restored.execute(
                    "SELECT name FROM existing_data WHERE id = 1"
                ).fetchone()[0]
                marker = restored.execute(
                    """
                    SELECT name
                    FROM sqlite_schema
                    WHERE type = 'table' AND name = 'should_not_survive'
                    """
                ).fetchone()
            finally:
                restored.close()
            self.assertEqual("safe", value)
            self.assertIsNone(marker)

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
