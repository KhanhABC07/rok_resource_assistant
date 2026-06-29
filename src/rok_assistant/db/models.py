from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


TASK_STATUSES = (
    "pending",
    "queued",
    "running",
    "completed",
    "failed",
    "aborted",
    "retrying",
)
AUTOMATION_ACTION_TYPES = (
    "WaitTemplate",
    "ClickTemplate",
    "ClickCoordinates",
    "SwipeCoordinates",
    "Delay",
    "AbortTask",
    "RepeatStart",
    "RepeatEnd",
    "IfTemplateExists",
    "Else",
    "EndIf",
)


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0).isoformat()


def row_bool(value: Any) -> bool:
    return bool(int(value or 0))


@dataclass
class Instance:
    id: int | None = None
    name: str = ""
    instance_index: int | None = None
    instance_name: str = ""
    adb_serial: str = ""
    adb_connected: bool = False
    launch_path: str = ""
    launch_command: str = ""
    close_command: str = ""
    enabled: bool = True


@dataclass
class Character:
    id: int | None = None
    name: str = ""
    instance_id: int | None = None
    account_name: str = ""
    enabled: bool = True
    alliance_help_enabled: bool = True
    alliance_donate_enabled: bool = True
    gift_collection_enabled: bool = True
    instance_name: str = ""


@dataclass
class March:
    id: int | None = None
    character_id: int | None = None
    march_slot: int = 1
    # Retained only to load databases created before march resource configuration
    # moved into natural-resource tasks.
    resource_type: str = "Gold"
    resource_source: str = "Disabled"
    status: str = "idle"
    next_action_time: str | None = None
    expected_return_time: str | None = None


@dataclass
class ScheduledTask:
    id: int | None = None
    character_id: int | None = None
    march_slot: int | None = None
    task_type: str = "gathering"
    priority: int = 100
    status: str = "pending"
    scheduled_for: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    attempts: int = 0
    error_message: str = ""
    result: str = ""
    payload_json: str = "{}"
    resource_type: str = "Gold"
    character_name: str = ""
    instance_id: int | None = None
    instance_name: str = ""


@dataclass
class Task:
    id: int | None = None
    name: str = ""
    enabled: bool = True
    template_readiness_required: bool = False
    created_at: str = ""


@dataclass
class TaskStep:
    id: int | None = None
    task_id: int | None = None
    order: int = 1
    action_type: str = "Delay"
    parameters: dict[str, Any] | None = None


@dataclass
class TaskRunHistory:
    id: int | None = None
    task_id: int | None = None
    task_name: str = ""
    instance_index: int | None = None
    instance_name: str = ""
    started_at: str = ""
    finished_at: str = ""
    result: str = ""
    error_message: str = ""
    abort_reason: str = ""
    created_at: str = ""


@dataclass
class DashboardStats:
    active_workers: int = 0
    running_instances: int = 0
    total_characters: int = 0
    pending_tasks: int = 0
    next_scheduled_task: str = "-"
