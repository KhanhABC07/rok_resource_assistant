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
INSTANCE_SESSION_STATUSES = (
    "created",
    "starting",
    "running",
    "stopping",
    "stopped",
    "failed",
)
JOB_STATUSES = (
    "pending",
    "queued",
    "running",
    "completed",
    "failed",
    "aborted",
    "cancelled",
)
RUN_STATUSES = (
    "running",
    "completed",
    "failed",
    "aborted",
    "cancelled",
)
INCIDENT_SEVERITIES = ("info", "warning", "error", "critical")
INCIDENT_STATUSES = ("open", "acknowledged", "resolved")
RECOVERY_ATTEMPT_STATES = ("started", "succeeded", "failed", "skipped")
CIRCUIT_BREAKER_STATUSES = ("open", "closed")


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
    game_account_id: int | None = None


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
class GameAccount:
    id: int | None = None
    account_name: str = ""
    display_name: str = ""
    provider: str = ""
    external_id: str = ""
    enabled: bool = True
    metadata_json: str = "{}"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class InstanceSession:
    id: int | None = None
    instance_id: int | None = None
    session_key: str = ""
    status: str = "created"
    started_at: str | None = None
    ended_at: str | None = None
    emulator_pid: int | None = None
    adb_serial: str = ""
    metadata_json: str = "{}"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class AutomationProfile:
    id: int | None = None
    name: str = ""
    description: str = ""
    enabled: bool = True
    metadata_json: str = "{}"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class FeatureConfig:
    id: int | None = None
    profile_id: int | None = None
    feature_key: str = ""
    enabled: bool = True
    config_json: str = "{}"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ScheduleDefinition:
    id: int | None = None
    profile_id: int | None = None
    schedule_key: str = ""
    name: str = ""
    enabled: bool = True
    cron_expression: str = ""
    interval_seconds: int | None = None
    timezone: str = "UTC"
    config_json: str = "{}"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class WorkflowDefinition:
    id: int | None = None
    profile_id: int | None = None
    workflow_key: str = ""
    name: str = ""
    version: int = 1
    enabled: bool = True
    trigger_type: str = "manual"
    config_json: str = "{}"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class WorkflowStep:
    id: int | None = None
    workflow_id: int | None = None
    step_order: int = 1
    step_key: str = ""
    action_type: str = ""
    parameters_json: str = "{}"
    timeout_seconds: int | None = None
    retry_limit: int = 0
    enabled: bool = True
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Job:
    id: int | None = None
    workflow_id: int | None = None
    schedule_id: int | None = None
    character_id: int | None = None
    idempotency_key: str = ""
    job_type: str = ""
    status: str = "pending"
    priority: int = 100
    scheduled_for: str = ""
    payload_json: str = "{}"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class JobRun:
    id: int | None = None
    job_id: int | None = None
    run_key: str = ""
    status: str = "running"
    attempt: int = 1
    started_at: str = ""
    finished_at: str | None = None
    result_json: str = "{}"
    error_message: str = ""
    screenshot_path: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass
class StepRun:
    id: int | None = None
    job_run_id: int | None = None
    workflow_step_id: int | None = None
    step_key: str = ""
    status: str = "running"
    attempt: int = 1
    started_at: str = ""
    finished_at: str | None = None
    result_json: str = "{}"
    error_message: str = ""
    screenshot_path: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass
class TemplatePack:
    id: int | None = None
    pack_key: str = ""
    name: str = ""
    version: str = "1"
    source_path: str = ""
    enabled: bool = True
    metadata_json: str = "{}"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Template:
    id: int | None = None
    pack_id: int | None = None
    template_key: str = ""
    name: str = ""
    file_path: str = ""
    image_hash: str = ""
    threshold: float = 0.8
    enabled: bool = True
    metadata_json: str = "{}"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ScreenObservation:
    id: int | None = None
    observation_key: str = ""
    instance_id: int | None = None
    character_id: int | None = None
    job_run_id: int | None = None
    observed_at: str = ""
    scene_name: str = ""
    screenshot_path: str = ""
    ocr_text: str = ""
    metadata_json: str = "{}"
    created_at: str = ""


@dataclass
class Incident:
    id: int | None = None
    incident_key: str = ""
    severity: str = "error"
    status: str = "open"
    title: str = ""
    details: str = ""
    job_run_id: int | None = None
    step_run_id: int | None = None
    screenshot_path: str = ""
    created_at: str = ""
    resolved_at: str | None = None
    updated_at: str = ""


@dataclass
class RecoveryAttempt:
    id: int | None = None
    attempt_key: str = ""
    instance_id: int | None = None
    job_run_id: int | None = None
    phase: str = ""
    state: str = "started"
    started_at: str = ""
    finished_at: str | None = None
    success: bool = False
    reason: str = ""
    screenshot_path: str = ""
    metadata_json: str = "{}"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class InstanceCircuitBreaker:
    id: int | None = None
    instance_id: int | None = None
    status: str = "open"
    opened_at: str = ""
    closed_at: str | None = None
    reason: str = ""
    incident_id: int | None = None
    metadata_json: str = "{}"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class AuditLog:
    id: int | None = None
    audit_key: str = ""
    actor: str = "system"
    action: str = ""
    entity_type: str = ""
    entity_id: int | None = None
    occurred_at: str = ""
    details_json: str = "{}"
    created_at: str = ""


@dataclass
class DashboardStats:
    active_workers: int = 0
    running_instances: int = 0
    total_characters: int = 0
    pending_tasks: int = 0
    next_scheduled_task: str = "-"
