from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from .schema import SCHEMA_SQL
from .sql import execute_script


MigrationCallback = Callable[[sqlite3.Connection], None]


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: MigrationCallback
    validate: MigrationCallback


LEGACY_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    instance_index INTEGER,
    instance_name TEXT NOT NULL DEFAULT '',
    adb_serial TEXT NOT NULL DEFAULT '',
    adb_connected INTEGER NOT NULL DEFAULT 0,
    launch_path TEXT NOT NULL DEFAULT '',
    launch_command TEXT NOT NULL DEFAULT '',
    close_command TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS characters (
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
    FOREIGN KEY(instance_id) REFERENCES instances(id) ON DELETE CASCADE,
    UNIQUE(instance_id, name)
);

CREATE TABLE IF NOT EXISTS marches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_id INTEGER NOT NULL,
    march_slot INTEGER NOT NULL CHECK(march_slot BETWEEN 1 AND 5),
    resource_type TEXT NOT NULL DEFAULT 'Gold'
        CHECK(resource_type IN ('Gold', 'Stone', 'Wood', 'Food')),
    resource_source TEXT NOT NULL DEFAULT 'Disabled'
        CHECK(resource_source IN ('Alliance Resource Pit', 'Wild Resource Node', 'Disabled')),
    status TEXT NOT NULL DEFAULT 'idle',
    next_action_time TEXT,
    expected_return_time TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE CASCADE,
    UNIQUE(character_id, march_slot)
);

CREATE TABLE IF NOT EXISTS scheduled_tasks (
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

CREATE TABLE IF NOT EXISTS automation_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    template_readiness_required INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

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
);

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
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tasks_due
    ON scheduled_tasks(status, scheduled_for, priority);

CREATE INDEX IF NOT EXISTS idx_automation_task_steps_task
    ON automation_task_steps(task_id, step_order);

CREATE INDEX IF NOT EXISTS idx_task_run_history_task
    ON task_run_history(task_id, started_at);

CREATE INDEX IF NOT EXISTS idx_marches_character
    ON marches(character_id, march_slot);

CREATE TRIGGER IF NOT EXISTS trg_instances_updated
AFTER UPDATE ON instances
BEGIN
    UPDATE instances SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_characters_updated
AFTER UPDATE ON characters
BEGIN
    UPDATE characters SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_marches_updated
AFTER UPDATE ON marches
BEGIN
    UPDATE marches SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_tasks_updated
AFTER UPDATE ON scheduled_tasks
BEGIN
    UPDATE scheduled_tasks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_automation_tasks_updated
AFTER UPDATE ON automation_tasks
BEGIN
    UPDATE automation_tasks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_automation_task_steps_updated
AFTER UPDATE ON automation_task_steps
BEGIN
    UPDATE automation_task_steps SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
"""


LEGACY_TABLES = {
    "instances",
    "characters",
    "marches",
    "scheduled_tasks",
    "automation_tasks",
    "automation_task_steps",
    "task_run_history",
    "settings",
}

DATA_V2_TABLES = {
    "game_accounts",
    "instance_sessions",
    "automation_profiles",
    "feature_configs",
    "schedule_definitions",
    "workflow_definitions",
    "workflow_steps",
    "jobs",
    "job_runs",
    "step_runs",
    "template_packs",
    "templates",
    "screen_observations",
    "incidents",
    "audit_logs",
}


def latest_schema_version() -> int:
    return max(migration.version for migration in MIGRATIONS)


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        """
        SELECT name
        FROM sqlite_schema
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    if not table_exists(connection, table_name):
        return set()
    return {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def _require_tables(connection: sqlite3.Connection, table_names: set[str]) -> None:
    existing = {
        row["name"]
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table'"
        ).fetchall()
    }
    missing = sorted(table_names - existing)
    if missing:
        raise RuntimeError(f"Missing database table(s): {', '.join(missing)}")


def _require_columns(
    connection: sqlite3.Connection, table_name: str, column_names: set[str]
) -> None:
    missing = sorted(column_names - table_columns(connection, table_name))
    if missing:
        raise RuntimeError(
            f"Missing database column(s) on {table_name}: {', '.join(missing)}"
        )


def migrate_legacy_schema(connection: sqlite3.Connection) -> None:
    ensure_legacy_instance_columns(connection)
    execute_script(connection, LEGACY_SCHEMA_SQL)
    apply_legacy_migrations(connection)


def validate_legacy_schema(connection: sqlite3.Connection) -> None:
    _require_tables(connection, LEGACY_TABLES)
    _require_columns(
        connection,
        "instances",
        {"instance_index", "instance_name", "adb_serial", "adb_connected"},
    )
    _require_columns(connection, "scheduled_tasks", {"status", "result"})
    _require_columns(
        connection,
        "automation_tasks",
        {"template_readiness_required"},
    )
    _require_columns(
        connection,
        "task_run_history",
        {
            "task_id",
            "task_name",
            "instance_index",
            "instance_name",
            "started_at",
            "finished_at",
            "result",
        },
    )


def migrate_data_v2_schema(connection: sqlite3.Connection) -> None:
    ensure_character_account_relationship(connection)
    execute_script(connection, SCHEMA_SQL)
    migrate_character_accounts(connection)


def validate_data_v2_schema(connection: sqlite3.Connection) -> None:
    _require_tables(connection, DATA_V2_TABLES | LEGACY_TABLES | {"schema_migrations"})
    _require_columns(connection, "characters", {"account_name", "game_account_id"})
    _require_columns(connection, "game_accounts", {"account_name", "metadata_json"})
    _require_columns(connection, "jobs", {"idempotency_key", "status", "scheduled_for"})
    _require_columns(connection, "audit_logs", {"audit_key", "details_json"})


def ensure_legacy_instance_columns(connection: sqlite3.Connection) -> None:
    if not table_exists(connection, "instances"):
        return
    columns = table_columns(connection, "instances")
    if "created_at" not in columns:
        connection.execute(
            "ALTER TABLE instances ADD COLUMN created_at TEXT NOT NULL DEFAULT ''"
        )
    if "updated_at" not in columns:
        connection.execute(
            "ALTER TABLE instances ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''"
        )
    if "instance_index" not in columns:
        connection.execute("ALTER TABLE instances ADD COLUMN instance_index INTEGER")
    if "instance_name" not in columns:
        connection.execute(
            "ALTER TABLE instances ADD COLUMN instance_name TEXT NOT NULL DEFAULT ''"
        )
    if "adb_serial" not in columns:
        connection.execute(
            "ALTER TABLE instances ADD COLUMN adb_serial TEXT NOT NULL DEFAULT ''"
        )
    if "adb_connected" not in columns:
        connection.execute(
            "ALTER TABLE instances ADD COLUMN adb_connected INTEGER NOT NULL DEFAULT 0"
        )


def ensure_character_account_relationship(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS game_accounts (
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
    if table_exists(connection, "characters") and "game_account_id" not in table_columns(
        connection, "characters"
    ):
        connection.execute(
            """
            ALTER TABLE characters
            ADD COLUMN game_account_id INTEGER
                REFERENCES game_accounts(id) ON DELETE SET NULL
            """
        )


def migrate_character_accounts(connection: sqlite3.Connection) -> None:
    if not table_exists(connection, "characters"):
        return
    columns = table_columns(connection, "characters")
    if "account_name" not in columns or "game_account_id" not in columns:
        return
    connection.execute(
        """
        INSERT OR IGNORE INTO game_accounts(account_name, display_name)
        SELECT DISTINCT trim(account_name), trim(account_name)
        FROM characters
        WHERE trim(account_name) <> ''
        """
    )
    connection.execute(
        """
        UPDATE characters
        SET game_account_id = (
            SELECT game_accounts.id
            FROM game_accounts
            WHERE game_accounts.account_name = trim(characters.account_name)
        )
        WHERE trim(account_name) <> ''
        """
    )


def apply_legacy_migrations(connection: sqlite3.Connection) -> None:
    if table_exists(connection, "instances"):
        connection.execute(
            """
            UPDATE instances
            SET instance_name = name
            WHERE instance_name = ''
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_instances_instance_index
                ON instances(instance_index)
                WHERE instance_index IS NOT NULL
            """
        )
    if table_exists(connection, "automation_tasks"):
        columns = table_columns(connection, "automation_tasks")
        if "template_readiness_required" not in columns:
            connection.execute(
                """
                ALTER TABLE automation_tasks
                ADD COLUMN template_readiness_required INTEGER NOT NULL DEFAULT 0
                """
            )
    migrate_automation_task_step_actions(connection)
    migrate_scheduled_task_results(connection)


def migrate_scheduled_task_results(connection: sqlite3.Connection) -> None:
    if not table_exists(connection, "scheduled_tasks"):
        return
    columns = table_columns(connection, "scheduled_tasks")
    if "result" not in columns:
        connection.execute(
            """
            ALTER TABLE scheduled_tasks
            ADD COLUMN result TEXT NOT NULL DEFAULT ''
            """
        )

    row = connection.execute(
        """
        SELECT sql
        FROM sqlite_schema
        WHERE type = 'table' AND name = 'scheduled_tasks'
        """
    ).fetchone()
    create_sql = row["sql"] if row else ""
    if "'aborted'" in create_sql and "result" in create_sql:
        return

    for statement in (
        "DROP TRIGGER IF EXISTS trg_tasks_updated",
        "DROP TABLE IF EXISTS scheduled_tasks_new",
        """
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
        )
        """,
        """
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
        FROM scheduled_tasks
        """,
        "DROP TABLE scheduled_tasks",
        "ALTER TABLE scheduled_tasks_new RENAME TO scheduled_tasks",
        """
        CREATE INDEX IF NOT EXISTS idx_tasks_due
            ON scheduled_tasks(status, scheduled_for, priority)
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_tasks_updated
        AFTER UPDATE ON scheduled_tasks
        BEGIN
            UPDATE scheduled_tasks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END
        """,
    ):
        connection.execute(statement)


def migrate_automation_task_step_actions(connection: sqlite3.Connection) -> None:
    if not table_exists(connection, "automation_task_steps"):
        return
    row = connection.execute(
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

    for statement in (
        "DROP TRIGGER IF EXISTS trg_automation_task_steps_updated",
        "DROP TABLE IF EXISTS automation_task_steps_new",
        """
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
        )
        """,
        """
        INSERT INTO automation_task_steps_new(
            id, task_id, step_order, action_type, parameters, created_at, updated_at
        )
        SELECT id, task_id, step_order, action_type, parameters, created_at, updated_at
        FROM automation_task_steps
        """,
        "DROP TABLE automation_task_steps",
        "ALTER TABLE automation_task_steps_new RENAME TO automation_task_steps",
        """
        CREATE TRIGGER IF NOT EXISTS trg_automation_tasks_updated
        AFTER UPDATE ON automation_tasks
        BEGIN
            UPDATE automation_tasks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_automation_task_steps_updated
        AFTER UPDATE ON automation_task_steps
        BEGIN
            UPDATE automation_task_steps
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = NEW.id;
        END
        """,
    ):
        connection.execute(statement)


MIGRATIONS = (
    Migration(1, "legacy schema compatibility", migrate_legacy_schema, validate_legacy_schema),
    Migration(2, "data v2 schema", migrate_data_v2_schema, validate_data_v2_schema),
)
