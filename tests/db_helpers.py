from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


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


LEGACY_CURRENT_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE instances (
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
    FOREIGN KEY(instance_id) REFERENCES instances(id) ON DELETE CASCADE,
    UNIQUE(instance_id, name)
);

CREATE TABLE marches (
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

CREATE TABLE scheduled_tasks (
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

CREATE TABLE automation_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    template_readiness_required INTEGER NOT NULL DEFAULT 0,
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

CREATE TABLE task_run_history (
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

CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tasks_due
    ON scheduled_tasks(status, scheduled_for, priority);

CREATE INDEX idx_automation_task_steps_task
    ON automation_task_steps(task_id, step_order);

CREATE INDEX idx_task_run_history_task
    ON task_run_history(task_id, started_at);

CREATE INDEX idx_marches_character
    ON marches(character_id, march_slot);
"""
