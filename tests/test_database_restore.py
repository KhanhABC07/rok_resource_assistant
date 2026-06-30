from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from tests.db_helpers import SRC_ROOT  # noqa: F401

from rok_assistant.db import migrations
from rok_assistant.db.database import Database, DatabaseRestoreError


def create_existing_database(path: Path, value: str = "safe") -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE existing_data(id INTEGER PRIMARY KEY, name TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO existing_data(id, name) VALUES (1, ?)",
            (value,),
        )
        connection.commit()
    finally:
        connection.close()


def create_invalid_foreign_key_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE parent(id INTEGER PRIMARY KEY);
            CREATE TABLE child(
                id INTEGER PRIMARY KEY,
                parent_id INTEGER NOT NULL,
                FOREIGN KEY(parent_id) REFERENCES parent(id)
            );
            INSERT INTO child(id, parent_id) VALUES (1, 999);
            """
        )
        connection.commit()
    finally:
        connection.close()


def read_existing_value(path: Path) -> str:
    connection = sqlite3.connect(path)
    try:
        return connection.execute(
            "SELECT name FROM existing_data WHERE id = 1"
        ).fetchone()[0]
    finally:
        connection.close()


def restore_artifacts(path: Path) -> list[Path]:
    return sorted(
        list(path.parent.glob(f"{path.name}*.rollback.*"))
        + list(path.parent.glob(f"{path.name}.restore.*"))
    )


@contextmanager
def failing_migrations() -> object:
    def apply(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE should_not_survive(id INTEGER)")

    def validate(_connection: sqlite3.Connection) -> None:
        raise RuntimeError("validation failed")

    original = migrations.MIGRATIONS
    migrations.MIGRATIONS = (
        migrations.Migration(1, "bad restore migration", apply, validate),
    )
    try:
        yield
    finally:
        migrations.MIGRATIONS = original


class DatabaseRestoreTest(unittest.TestCase):
    def test_successful_restore_after_migration_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "restore_success.sqlite3"
            create_existing_database(path)

            with failing_migrations():
                db = Database(path)
                try:
                    with self.assertRaisesRegex(RuntimeError, "validation failed"):
                        db.initialize()
                finally:
                    db.close()

            self.assertEqual("safe", read_existing_value(path))
            connection = sqlite3.connect(path)
            try:
                marker = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_schema
                    WHERE type = 'table' AND name = 'should_not_survive'
                    """
                ).fetchone()
            finally:
                connection.close()
            self.assertIsNone(marker)
            self.assertEqual([], restore_artifacts(path))

    def test_replacement_failure_before_main_database_changes_preserves_original(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stage_failure.sqlite3"
            create_existing_database(path)
            real_replace = os.replace

            def fail_main_stage(src: object, dst: object) -> None:
                src_path = Path(src)
                dst_path = Path(dst)
                if src_path == path and ".rollback." in dst_path.name:
                    raise OSError("stage main failed")
                real_replace(src_path, dst_path)

            with failing_migrations(), patch(
                "rok_assistant.db.database.os.replace",
                side_effect=fail_main_stage,
            ):
                db = Database(path)
                try:
                    with self.assertRaisesRegex(DatabaseRestoreError, "another process"):
                        db.initialize()
                finally:
                    db.close()

            self.assertEqual("safe", read_existing_value(path))
            self.assertEqual([], restore_artifacts(path))

    def test_replacement_failure_after_sidecars_are_staged_restores_original(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace_failure.sqlite3"
            create_existing_database(path)
            for suffix in ("-wal", "-shm", "-journal"):
                path.with_name(f"{path.name}{suffix}").write_bytes(b"")
            real_replace = os.replace

            def fail_restore_replace(src: object, dst: object) -> None:
                src_path = Path(src)
                dst_path = Path(dst)
                if src_path.name.startswith(f"{path.name}.restore.") and dst_path == path:
                    raise OSError("restore replace failed")
                real_replace(src_path, dst_path)

            with failing_migrations(), patch(
                "rok_assistant.db.database.os.replace",
                side_effect=fail_restore_replace,
            ):
                db = Database(path)
                try:
                    with self.assertRaisesRegex(DatabaseRestoreError, "another process"):
                        db.initialize()
                finally:
                    db.close()

            self.assertEqual("safe", read_existing_value(path))
            self.assertEqual([], restore_artifacts(path))

    def test_restored_database_validation_failure_preserves_original(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid_restore.sqlite3"
            create_invalid_foreign_key_database(path)

            db = Database(path)
            try:
                with self.assertRaisesRegex(
                    DatabaseRestoreError,
                    "Restore database foreign key validation failed",
                ):
                    db.initialize()
            finally:
                db.close()

            connection = sqlite3.connect(path)
            try:
                row = connection.execute(
                    "SELECT parent_id FROM child WHERE id = 1"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(999, row[0])
            self.assertEqual([], restore_artifacts(path))

    def test_sidecar_files_are_preserved_and_restored_on_restore_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sidecars.sqlite3"
            backup_path = Path(temp_dir) / "sidecars_backup.sqlite3"
            create_existing_database(path, "current")
            create_existing_database(backup_path, "backup")
            sidecars = {
                "-wal": b"wal marker",
                "-shm": b"shm marker",
                "-journal": b"journal marker",
            }
            for suffix, content in sidecars.items():
                path.with_name(f"{path.name}{suffix}").write_bytes(content)

            real_replace = os.replace
            restored_sidecar_contents: dict[str, bytes] = {}
            sidecar_names = {f"{path.name}{suffix}" for suffix in sidecars}

            def fail_restore_replace(src: object, dst: object) -> None:
                src_path = Path(src)
                dst_path = Path(dst)
                if src_path.name.startswith(f"{path.name}.restore.") and dst_path == path:
                    raise OSError("restore replace failed")
                if ".rollback." in src_path.name and dst_path.name in sidecar_names:
                    restored_sidecar_contents[dst_path.name] = src_path.read_bytes()
                real_replace(src_path, dst_path)

            db = Database(path)
            db._initializing = True
            try:
                with patch(
                    "rok_assistant.db.database.os.replace",
                    side_effect=fail_restore_replace,
                ):
                    with self.assertRaisesRegex(
                        DatabaseRestoreError,
                        "another process",
                    ):
                        db._restore_backup(backup_path)
            finally:
                db._initializing = False
                db.close()

            self.assertEqual("current", read_existing_value(path))
            for suffix, content in sidecars.items():
                self.assertEqual(
                    content,
                    restored_sidecar_contents[f"{path.name}{suffix}"],
                )
            self.assertEqual([], restore_artifacts(path))

    def test_original_database_remains_usable_after_restore_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "usable_after_failure.sqlite3"
            backup_path = Path(temp_dir) / "usable_backup.sqlite3"
            create_existing_database(path, "current")
            create_existing_database(backup_path, "backup")
            real_replace = os.replace

            def fail_restore_replace(src: object, dst: object) -> None:
                src_path = Path(src)
                dst_path = Path(dst)
                if src_path.name.startswith(f"{path.name}.restore.") and dst_path == path:
                    raise OSError("restore replace failed")
                real_replace(src_path, dst_path)

            db = Database(path)
            db._initializing = True
            try:
                with patch(
                    "rok_assistant.db.database.os.replace",
                    side_effect=fail_restore_replace,
                ):
                    with self.assertRaises(DatabaseRestoreError):
                        db._restore_backup(backup_path)
                self.assertEqual(
                    "current",
                    db.fetch_one("SELECT name FROM existing_data WHERE id = 1")[
                        "name"
                    ],
                )
            finally:
                db._initializing = False
                db.close()

    def test_stale_rollback_file_is_detected_on_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stale.sqlite3"
            create_existing_database(path)
            path.with_name(f"{path.name}.rollback.stale").write_text("stale")

            db = Database(path)
            try:
                with self.assertRaisesRegex(
                    DatabaseRestoreError,
                    "Stale database restore files exist",
                ):
                    db.initialize()
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
