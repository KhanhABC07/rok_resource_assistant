SCHEMA_SQL = """
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
