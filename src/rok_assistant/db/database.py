from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from .schema import SCHEMA_SQL


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = self._open_connection()

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def initialize(self) -> None:
        with self._lock:
            self._connection.executescript(SCHEMA_SQL)
            self._apply_migrations()

    def _apply_migrations(self) -> None:
        columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(instances)").fetchall()
        }
        if "instance_index" not in columns:
            self._connection.execute("ALTER TABLE instances ADD COLUMN instance_index INTEGER")
        if "instance_name" not in columns:
            self._connection.execute(
                "ALTER TABLE instances ADD COLUMN instance_name TEXT NOT NULL DEFAULT ''"
            )
        if "adb_serial" not in columns:
            self._connection.execute(
                "ALTER TABLE instances ADD COLUMN adb_serial TEXT NOT NULL DEFAULT ''"
            )
        if "adb_connected" not in columns:
            self._connection.execute(
                "ALTER TABLE instances ADD COLUMN adb_connected INTEGER NOT NULL DEFAULT 0"
            )
        self._connection.execute(
            """
            UPDATE instances
            SET instance_name = name
            WHERE instance_name = ''
            """
        )
        self._connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_instances_instance_index
                ON instances(instance_index)
                WHERE instance_index IS NOT NULL
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS automation_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                template_readiness_required INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        automation_task_columns = {
            row["name"]
            for row in self._connection.execute(
                "PRAGMA table_info(automation_tasks)"
            ).fetchall()
        }
        if "template_readiness_required" not in automation_task_columns:
            self._connection.execute(
                """
                ALTER TABLE automation_tasks
                ADD COLUMN template_readiness_required INTEGER NOT NULL DEFAULT 0
                """
            )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS automation_task_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                step_order INTEGER NOT NULL,
                action_type TEXT NOT NULL
                    CHECK(action_type IN (
                        'WaitTemplate',
                        'ClickTemplate',
                        'ClickCoordinates',
                        'SwipeCoordinates',
                        'Delay',
                        'AbortTask',
                        'RepeatStart',
                        'RepeatEnd',
                        'IfTemplateExists',
                        'Else',
                        'EndIf'
                    )),
                parameters TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(task_id) REFERENCES automation_tasks(id) ON DELETE CASCADE,
                UNIQUE(task_id, step_order)
            )
            """
        )
        self._migrate_automation_task_step_actions()
        self._migrate_scheduled_task_results()
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_automation_task_steps_task
                ON automation_task_steps(task_id, step_order)
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS task_run_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER,
                task_name TEXT NOT NULL,
                instance_index INTEGER,
                instance_name TEXT NOT NULL DEFAULT '',
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                result TEXT NOT NULL CHECK(result IN ('SUCCESS', 'FAILED', 'ABORTED')),
                error_message TEXT NOT NULL DEFAULT '',
                abort_reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_run_history_task
                ON task_run_history(task_id, started_at)
            """
        )

    def _migrate_scheduled_task_results(self) -> None:
        columns = {
            row["name"]
            for row in self._connection.execute(
                "PRAGMA table_info(scheduled_tasks)"
            ).fetchall()
        }
        if "result" not in columns:
            self._connection.execute(
                """
                ALTER TABLE scheduled_tasks
                ADD COLUMN result TEXT NOT NULL DEFAULT ''
                """
            )

        row = self._connection.execute(
            """
            SELECT sql
            FROM sqlite_schema
            WHERE type = 'table' AND name = 'scheduled_tasks'
            """
        ).fetchone()
        create_sql = row["sql"] if row else ""
        if "'aborted'" in create_sql and "result" in create_sql:
            return

        self._connection.executescript(
            """
            DROP TRIGGER IF EXISTS trg_tasks_updated;
            DROP TABLE IF EXISTS scheduled_tasks_new;

            CREATE TABLE scheduled_tasks_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                march_slot INTEGER,
                task_type TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 100,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'queued', 'running', 'completed', 'failed', 'aborted', 'retrying')),
                scheduled_for TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                error_message TEXT NOT NULL DEFAULT '',
                result TEXT NOT NULL DEFAULT ''
                    CHECK(result IN ('', 'SUCCESS', 'FAILED', 'ABORTED')),
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE CASCADE
            );

            INSERT INTO scheduled_tasks_new(
                id, character_id, march_slot, task_type, priority, status,
                scheduled_for, started_at, completed_at, attempts, error_message,
                result, payload_json, created_at, updated_at
            )
            SELECT
                id, character_id, march_slot, task_type, priority, status,
                scheduled_for, started_at, completed_at, attempts, error_message,
                CASE
                    WHEN result IN ('SUCCESS', 'FAILED', 'ABORTED') THEN result
                    WHEN status = 'completed' THEN 'SUCCESS'
                    WHEN status = 'failed' THEN 'FAILED'
                    ELSE ''
                END,
                payload_json, created_at, updated_at
            FROM scheduled_tasks;

            DROP TABLE scheduled_tasks;
            ALTER TABLE scheduled_tasks_new RENAME TO scheduled_tasks;

            CREATE INDEX IF NOT EXISTS idx_tasks_due
                ON scheduled_tasks(status, scheduled_for, priority);

            CREATE TRIGGER IF NOT EXISTS trg_tasks_updated
            AFTER UPDATE ON scheduled_tasks
            BEGIN
                UPDATE scheduled_tasks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
            """
        )

    def _migrate_automation_task_step_actions(self) -> None:
        row = self._connection.execute(
            """
            SELECT sql
            FROM sqlite_schema
            WHERE type = 'table' AND name = 'automation_task_steps'
            """
        ).fetchone()
        create_sql = row["sql"] if row else ""
        required_actions = (
            "RepeatStart",
            "RepeatEnd",
            "IfTemplateExists",
            "Else",
            "EndIf",
            "AbortTask",
        )
        if all(action in create_sql for action in required_actions):
            return

        self._connection.executescript(
            """
            DROP TRIGGER IF EXISTS trg_automation_task_steps_updated;
            DROP TABLE IF EXISTS automation_task_steps_new;

            CREATE TABLE automation_task_steps_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                step_order INTEGER NOT NULL,
                action_type TEXT NOT NULL
                    CHECK(action_type IN (
                        'WaitTemplate',
                        'ClickTemplate',
                        'ClickCoordinates',
                        'SwipeCoordinates',
                        'Delay',
                        'AbortTask',
                        'RepeatStart',
                        'RepeatEnd',
                        'IfTemplateExists',
                        'Else',
                        'EndIf'
                    )),
                parameters TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(task_id) REFERENCES automation_tasks(id) ON DELETE CASCADE,
                UNIQUE(task_id, step_order)
            );

            INSERT INTO automation_task_steps_new(
                id, task_id, step_order, action_type, parameters, created_at, updated_at
            )
            SELECT id, task_id, step_order, action_type, parameters, created_at, updated_at
            FROM automation_task_steps;

            DROP TABLE automation_task_steps;
            ALTER TABLE automation_task_steps_new RENAME TO automation_task_steps;
            """
        )
        self._connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_automation_tasks_updated
            AFTER UPDATE ON automation_tasks
            BEGIN
                UPDATE automation_tasks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
            """
        )
        self._connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_automation_task_steps_updated
            AFTER UPDATE ON automation_task_steps
            BEGIN
                UPDATE automation_task_steps
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = NEW.id;
            END;
            """
        )

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
            try:
                self._connection.execute("BEGIN")
                yield self._connection
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
