from __future__ import annotations

import os
import shutil
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import migrations


class DatabaseRestoreError(RuntimeError):
    pass


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._transaction_depth = 0
        self._savepoint_counter = 0
        self._initializing = False
        self.last_backup_path: Path | None = None
        self._connection = self._open_connection()

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def initialize(self) -> None:
        with self._lock:
            self._detect_stale_restore_files()
            self._initializing = True
            try:
                self._enable_wal()
                current_version = self._schema_version()
                latest_version = migrations.latest_schema_version()
                needs_migration = current_version < latest_version
                backup_path: Path | None = None
                if self._has_user_schema() and needs_migration:
                    self._checkpoint_wal_for_migration()
                    backup_path = self._create_backup()

                try:
                    if needs_migration:
                        self._run_pending_migrations(current_version)
                    else:
                        self._verify_foreign_keys(self._connection)
                except Exception:
                    if backup_path is not None:
                        self._restore_backup(backup_path)
                    raise
            finally:
                self._initializing = False

    def _enable_wal(self) -> None:
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 5000")

    def _checkpoint_wal_for_migration(self) -> None:
        row = self._connection.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
        if row is not None and int(row[0]) != 0:
            raise RuntimeError(
                "Cannot migrate database while another process is using it."
            )

    def _has_user_schema(self) -> bool:
        row = self._connection.execute(
            """
            SELECT name
            FROM sqlite_schema
            WHERE type IN ('table', 'index', 'trigger', 'view')
              AND name NOT LIKE 'sqlite_%'
            LIMIT 1
            """
        ).fetchone()
        return row is not None

    def _schema_version(self) -> int:
        if not self._table_exists("schema_migrations"):
            return 0
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
        ).fetchone()
        return int(row["version"] if row else 0)

    def _run_pending_migrations(self, current_version: int) -> None:
        for migration in migrations.MIGRATIONS:
            if migration.version <= current_version or self._migration_applied(
                migration.version
            ):
                continue
            with self.transaction() as connection:
                migration.apply(connection)
                migration.validate(connection)
                self._verify_foreign_keys(connection)
                self._record_migration(connection, migration.version, migration.name)

    def _migration_applied(self, version: int) -> bool:
        if not self._table_exists("schema_migrations"):
            return False
        row = self._connection.execute(
            "SELECT version FROM schema_migrations WHERE version = ?",
            (version,),
        ).fetchone()
        return row is not None

    def _ensure_schema_migrations_table(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def _record_migration(
        self,
        connection: sqlite3.Connection,
        version: int,
        name: str,
    ) -> None:
        self._ensure_schema_migrations_table(connection)
        connection.execute(
            """
            INSERT INTO schema_migrations(version, name)
            VALUES (?, ?)
            """,
            (version, name),
        )

    def _create_backup(self) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
        suffix = self.path.suffix or ".sqlite3"
        backup_path = self.path.with_name(
            f"{self.path.stem}.backup.{timestamp}{suffix}"
        )
        backup_connection = sqlite3.connect(backup_path)
        try:
            self._connection.backup(backup_connection)
        finally:
            backup_connection.close()
        self.last_backup_path = backup_path
        return backup_path

    def _restore_backup(self, backup_path: Path) -> None:
        if not self._initializing:
            raise DatabaseRestoreError(
                "Database backup restore is only allowed during database initialization."
            )

        self._close_connection_for_restore()
        operation_id = uuid.uuid4().hex
        suffix = self.path.suffix or ".sqlite3"
        temp_restore_path = self.path.with_name(
            f"{self.path.name}.restore.{operation_id}{suffix}"
        )
        rollback_paths: dict[Path, Path] = {}
        restore_replaced_target = False
        try:
            shutil.copy2(backup_path, temp_restore_path)
            self._validate_database_file(temp_restore_path)
            self._stage_current_database_files(operation_id, rollback_paths)
            os.replace(temp_restore_path, self.path)
            restore_replaced_target = True
            self._validate_database_file(self.path)
            self._connection = self._open_connection()
        except Exception as exc:
            self._recover_failed_restore(
                rollback_paths,
                temp_restore_path,
                remove_current_files=restore_replaced_target,
            )
            self._connection = self._open_connection()
            if isinstance(exc, DatabaseRestoreError):
                raise
            if isinstance(exc, OSError):
                raise DatabaseRestoreError(
                    "Cannot restore database backup. The database may be in use by "
                    "another process."
                ) from exc
            raise DatabaseRestoreError("Cannot restore database backup.") from exc
        self._delete_rollback_files(rollback_paths.values())

    def _detect_stale_restore_files(self) -> None:
        patterns = (
            f"{self.path.name}*.rollback.*",
            f"{self.path.name}.restore.*",
        )
        stale_files: list[Path] = []
        for pattern in patterns:
            stale_files.extend(self.path.parent.glob(pattern))
        if stale_files:
            details = ", ".join(str(path) for path in sorted(stale_files))
            raise DatabaseRestoreError(
                "Stale database restore files exist. Resolve them before "
                f"initializing the database: {details}"
            )

    def _close_connection_for_restore(self) -> None:
        try:
            self._connection.close()
        except sqlite3.Error as exc:
            raise DatabaseRestoreError(
                "Cannot close active database connection before restore."
            ) from exc

    def _database_file_paths(self) -> tuple[Path, ...]:
        return (
            self.path,
            self.path.with_name(f"{self.path.name}-wal"),
            self.path.with_name(f"{self.path.name}-shm"),
            self.path.with_name(f"{self.path.name}-journal"),
        )

    def _stage_current_database_files(
        self,
        operation_id: str,
        rollback_paths: dict[Path, Path],
    ) -> None:
        for current_path in self._database_file_paths():
            if not current_path.exists():
                continue
            rollback_path = current_path.with_name(
                f"{current_path.name}.rollback.{operation_id}"
            )
            if rollback_path.exists():
                raise DatabaseRestoreError(
                    f"Rollback restore file already exists: {rollback_path}"
                )
            os.replace(current_path, rollback_path)
            rollback_paths[current_path] = rollback_path

    def _recover_failed_restore(
        self,
        rollback_paths: dict[Path, Path],
        temp_restore_path: Path,
        *,
        remove_current_files: bool,
    ) -> None:
        failures: list[str] = []
        if remove_current_files:
            for current_path in self._database_file_paths():
                if current_path in rollback_paths:
                    continue
                if not current_path.exists():
                    continue
                try:
                    current_path.unlink()
                except OSError as exc:
                    failures.append(f"{current_path}: {exc}")

        for original_path, rollback_path in rollback_paths.items():
            if not rollback_path.exists():
                failures.append(f"missing rollback file: {rollback_path}")
                continue
            try:
                os.replace(rollback_path, original_path)
            except OSError as exc:
                failures.append(f"{rollback_path} -> {original_path}: {exc}")

        if temp_restore_path.exists():
            try:
                temp_restore_path.unlink()
            except OSError as exc:
                failures.append(f"{temp_restore_path}: {exc}")

        if failures:
            raise DatabaseRestoreError(
                "Failed to restore original database files after restore "
                f"failure: {'; '.join(failures)}"
            )

    def _delete_rollback_files(self, rollback_paths: Iterable[Path]) -> None:
        failures: list[str] = []
        for rollback_path in rollback_paths:
            if not rollback_path.exists():
                continue
            try:
                rollback_path.unlink()
            except OSError as exc:
                failures.append(f"{rollback_path}: {exc}")
        if failures:
            raise DatabaseRestoreError(
                "Restored database is valid, but rollback files could not be "
                f"deleted: {'; '.join(failures)}"
            )

    def _validate_database_file(self, path: Path) -> None:
        if not path.exists():
            raise DatabaseRestoreError(f"Restore database file does not exist: {path}")
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            schema_row = connection.execute(
                """
                SELECT name
                FROM sqlite_schema
                WHERE type IN ('table', 'index', 'trigger', 'view')
                  AND name NOT LIKE 'sqlite_%'
                LIMIT 1
                """
            ).fetchone()
            if schema_row is None:
                raise DatabaseRestoreError(
                    "Restore database does not contain required schema objects."
                )

            integrity_row = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity_row is None or integrity_row[0] != "ok":
                detail = integrity_row[0] if integrity_row is not None else "no result"
                raise DatabaseRestoreError(
                    f"Restore database integrity check failed: {detail}"
                )

            rows = connection.execute("PRAGMA foreign_key_check").fetchall()
            if rows:
                details = ", ".join(
                    f"{row['table']}:{row['rowid']}->{row['parent']}"
                    for row in rows[:5]
                )
                raise DatabaseRestoreError(
                    f"Restore database foreign key validation failed: {details}"
                )
        finally:
            connection.close()

    def _verify_foreign_keys(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        if rows:
            details = ", ".join(
                f"{row['table']}:{row['rowid']}->{row['parent']}" for row in rows[:5]
            )
            raise RuntimeError(f"Foreign key validation failed: {details}")

    def _table_exists(self, table_name: str) -> bool:
        row = self._connection.execute(
            """
            SELECT name
            FROM sqlite_schema
            WHERE type = 'table' AND name = ?
            """,
            (table_name,),
        ).fetchone()
        return row is not None

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def reopen(self) -> None:
        with self._lock:
            self._connection = self._open_connection()
            self.initialize()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._connection.execute(sql, tuple(params))

    def executemany(self, sql: str, params: Iterable[Iterable[Any]]) -> sqlite3.Cursor:
        with self._lock:
            return self._connection.executemany(sql, params)

    def fetch_one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._connection.execute(sql, tuple(params)).fetchone()

    def fetch_all(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._connection.execute(sql, tuple(params)).fetchall())

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            is_outer_transaction = self._transaction_depth == 0
            if is_outer_transaction:
                self._connection.execute("BEGIN IMMEDIATE")
                self._transaction_depth = 1
                try:
                    yield self._connection
                except Exception:
                    self._connection.execute("ROLLBACK")
                    raise
                else:
                    self._connection.execute("COMMIT")
                finally:
                    self._transaction_depth = 0
                return

            self._savepoint_counter += 1
            savepoint_name = f"rok_savepoint_{self._savepoint_counter}"
            self._connection.execute(f"SAVEPOINT {savepoint_name}")
            self._transaction_depth += 1
            try:
                yield self._connection
            except Exception:
                self._connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                self._connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                raise
            else:
                self._connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            finally:
                self._transaction_depth -= 1
