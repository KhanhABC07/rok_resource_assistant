from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from rok_assistant.db.models import Job, ScheduleDefinition, WorkflowDefinition
from rok_assistant.db.repositories import (
    JobRepository,
    ScheduleDefinitionRepository,
    WorkflowDefinitionRepository,
    WorkflowStepRepository,
)
from rok_assistant.scheduler.clock import (
    parse_persisted_utc,
    require_aware_utc,
    utc_datetime_to_text,
)
from rok_assistant.scheduler.models import (
    OccurrenceResult,
    OccurrenceState,
    ScheduleDiagnostic,
    ScheduleEvaluationResult,
)
from rok_assistant.workflow_engine import (
    WorkflowDefinitionSpec,
    WorkflowEngine,
    WorkflowValidationError,
)


class WorkflowDefinitionValidator(Protocol):
    def validation_errors(self, workflow: WorkflowDefinitionSpec) -> list[str]:
        ...


class WorkflowEngineDefinitionValidator:
    def __init__(self, engine: WorkflowEngine | None = None) -> None:
        self.engine = engine or WorkflowEngine()

    def validation_errors(self, workflow: WorkflowDefinitionSpec) -> list[str]:
        return self.engine.validation_errors(workflow)


@dataclass(frozen=True)
class WorkflowReference:
    workflow_key: str
    version: int


@dataclass(frozen=True)
class ScheduleOccurrenceTiming:
    occurrence_at: datetime | None
    next_occurrence_at: datetime
    start_at: datetime


@dataclass(frozen=True)
class ScheduleTarget:
    character_id: int | None
    identity: str
    payload: dict[str, object]


class SchedulePlanner:
    def __init__(
        self,
        schedules: ScheduleDefinitionRepository,
        workflows: WorkflowDefinitionRepository,
        workflow_steps: WorkflowStepRepository,
        jobs: JobRepository,
        *,
        validator: WorkflowDefinitionValidator | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.schedules = schedules
        self.workflows = workflows
        self.workflow_steps = workflow_steps
        self.jobs = jobs
        self.validator = validator or WorkflowEngineDefinitionValidator()
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def evaluate(self, now: datetime) -> ScheduleEvaluationResult:
        now = require_aware_utc(now, "now")
        occurrences: list[OccurrenceResult] = []
        diagnostics: list[ScheduleDiagnostic] = []
        for schedule in self.schedules.list_enabled():
            if not schedule.enabled:
                continue
            try:
                occurrence, diagnostic = self._evaluate_schedule(schedule, now)
            except Exception as exc:
                diagnostic = ScheduleDiagnostic(
                    category="schedule_evaluation_error",
                    schedule_id=schedule.id,
                    schedule_key=schedule.schedule_key,
                    message=_safe_message(exc),
                )
                occurrence = None
            if diagnostic is not None:
                diagnostics.append(diagnostic)
                self.logger.warning(
                    "scheduler_schedule_diagnostic schedule_id=%s schedule_key=%s category=%s",
                    diagnostic.schedule_id,
                    diagnostic.schedule_key,
                    diagnostic.category,
                )
            if occurrence is not None:
                occurrences.append(occurrence)
        return ScheduleEvaluationResult(tuple(occurrences), tuple(diagnostics))

    def _evaluate_schedule(
        self,
        schedule: ScheduleDefinition,
        now: datetime,
    ) -> tuple[OccurrenceResult | None, ScheduleDiagnostic | None]:
        if schedule.id is None:
            return None, self._diagnostic(schedule, "invalid_schedule", "Schedule id is required.")
        try:
            config = _json_object(schedule.config_json, "config_json")
            workflow_ref = _workflow_reference(config)
            target = _schedule_target(config)
            timing = _occurrence_timing(schedule, config, now)
            job_type = _string_config(config, "job_type", default="workflow")
            priority = _int_config(config, "priority", default=100)
        except ValueError as exc:
            return None, self._diagnostic(schedule, "invalid_schedule", str(exc))

        if timing.occurrence_at is None:
            self._persist_schedule_state(schedule, config, timing, now)
            self.logger.info(
                "scheduler_schedule_evaluated schedule_id=%s schedule_key=%s due=false",
                schedule.id,
                schedule.schedule_key,
            )
            return None, None

        workflow = self.workflows.get_by_key(
            int(schedule.profile_id or 0),
            workflow_ref.workflow_key,
            workflow_ref.version,
        )
        if workflow is None:
            self._persist_schedule_state(schedule, config, timing, now)
            return None, self._diagnostic(
                schedule,
                "missing_workflow",
                "Referenced workflow definition was not found.",
                workflow_ref.workflow_key,
            )
        if not workflow.enabled:
            self._persist_schedule_state(schedule, config, timing, now)
            return None, self._diagnostic(
                schedule,
                "disabled_workflow",
                "Referenced workflow definition is disabled.",
                workflow.workflow_key,
            )
        invalid = self._workflow_validation_errors(workflow)
        if invalid:
            self._persist_schedule_state(schedule, config, timing, now)
            return None, self._diagnostic(
                schedule,
                "invalid_workflow",
                "; ".join(invalid[:3]),
                workflow.workflow_key,
            )

        scheduled_for = utc_datetime_to_text(timing.occurrence_at)
        occurrence_key = build_occurrence_key(
            schedule,
            workflow,
            target.identity,
            timing.occurrence_at,
        )
        payload_json = _job_payload_json(
            schedule,
            workflow,
            occurrence_key,
            scheduled_for,
            target.payload,
        )
        job = Job(
            workflow_id=workflow.id,
            schedule_id=schedule.id,
            character_id=target.character_id,
            idempotency_key=occurrence_key,
            job_type=job_type,
            status="pending",
            priority=priority,
            scheduled_for=scheduled_for,
            payload_json=payload_json,
        )
        try:
            with self.jobs.db.transaction():
                persisted, created = self.jobs.create_if_absent(job)
                self._persist_schedule_state(schedule, config, timing, now)
        except ValueError as exc:
            return None, self._diagnostic(
                schedule,
                "occurrence_conflict",
                str(exc),
                workflow.workflow_key,
            )

        state = OccurrenceState.CREATED if created else OccurrenceState.ALREADY_EXISTS
        self.logger.info(
            "scheduler_occurrence_%s schedule_id=%s job_id=%s workflow_key=%s occurrence_at=%s",
            state.value,
            schedule.id,
            persisted.id,
            workflow.workflow_key,
            scheduled_for,
        )
        return (
            OccurrenceResult(
                state=state,
                schedule_id=schedule.id,
                schedule_key=schedule.schedule_key,
                occurrence_key=occurrence_key,
                scheduled_for=scheduled_for,
                job=persisted,
            ),
            None,
        )

    def _workflow_validation_errors(self, workflow: WorkflowDefinition) -> list[str]:
        if workflow.id is None:
            return ["workflow id is required."]
        try:
            spec = WorkflowDefinitionSpec.from_persisted(
                workflow,
                self.workflow_steps.list_for_workflow(workflow.id),
            )
        except WorkflowValidationError as exc:
            return [str(exc)]
        except ValueError as exc:
            return [str(exc)]
        return self.validator.validation_errors(spec)

    def _persist_schedule_state(
        self,
        schedule: ScheduleDefinition,
        config: dict[str, object],
        timing: ScheduleOccurrenceTiming,
        now: datetime,
    ) -> None:
        if schedule.id is None:
            return
        updated = dict(config)
        state_value = updated.get("scheduler")
        state = dict(state_value) if isinstance(state_value, dict) else {}
        state["start_at"] = utc_datetime_to_text(timing.start_at)
        state["next_occurrence_at"] = utc_datetime_to_text(timing.next_occurrence_at)
        state["last_evaluated_at"] = utc_datetime_to_text(now)
        if timing.occurrence_at is not None:
            state["last_occurrence_at"] = utc_datetime_to_text(timing.occurrence_at)
        updated["scheduler"] = state
        self.schedules.update_config_json(schedule.id, json.dumps(updated, sort_keys=True))

    def _diagnostic(
        self,
        schedule: ScheduleDefinition,
        category: str,
        message: str,
        workflow_key: str = "",
    ) -> ScheduleDiagnostic:
        return ScheduleDiagnostic(
            category=category,
            schedule_id=schedule.id,
            schedule_key=schedule.schedule_key,
            message=message,
            workflow_key=workflow_key,
        )


def build_occurrence_key(
    schedule: ScheduleDefinition,
    workflow: WorkflowDefinition,
    target_identity: str,
    occurrence_at: datetime,
) -> str:
    scheduled_for = utc_datetime_to_text(occurrence_at)
    schedule_key = schedule.schedule_key.strip().lower()
    workflow_key = workflow.workflow_key.strip().lower()
    return (
        f"scheduled-workflow:"
        f"schedule={schedule.id}:{schedule_key}|"
        f"workflow={workflow.id}:{workflow_key}:v{workflow.version}|"
        f"target={target_identity}|"
        f"at={scheduled_for}"
    )


def _occurrence_timing(
    schedule: ScheduleDefinition,
    config: dict[str, object],
    now: datetime,
) -> ScheduleOccurrenceTiming:
    if schedule.cron_expression.strip():
        raise ValueError("cron_expression schedules are not supported in SCHED-001A.")
    interval = schedule.interval_seconds
    if interval is None or interval <= 0:
        raise ValueError("interval_seconds must be greater than zero.")

    state_value = config.get("scheduler")
    state = state_value if isinstance(state_value, dict) else {}
    start_text = _optional_string(state.get("start_at"))
    if not start_text:
        start_text = _optional_string(config.get("start_at"))
    start_at = (
        parse_persisted_utc(start_text, "start_at")
        if start_text
        else parse_persisted_utc(schedule.created_at or utc_datetime_to_text(now), "created_at")
    )

    next_text = _optional_string(state.get("next_occurrence_at"))
    if not next_text:
        next_text = _optional_string(config.get("next_occurrence_at"))
    if next_text:
        next_at = parse_persisted_utc(next_text, "next_occurrence_at")
        if next_at > now:
            return ScheduleOccurrenceTiming(None, next_at, start_at)
        return ScheduleOccurrenceTiming(
            next_at,
            _advance_after(next_at, interval, now),
            start_at,
        )

    if start_at > now:
        return ScheduleOccurrenceTiming(None, start_at, start_at)
    elapsed = int((now - start_at).total_seconds())
    intervals = max(0, elapsed // interval)
    occurrence_at = start_at + timedelta(seconds=intervals * interval)
    return ScheduleOccurrenceTiming(
        occurrence_at,
        occurrence_at + timedelta(seconds=interval),
        start_at,
    )


def _advance_after(value: datetime, interval_seconds: int, now: datetime) -> datetime:
    next_value = value + timedelta(seconds=interval_seconds)
    if next_value > now:
        return next_value
    missed = int((now - next_value).total_seconds() // interval_seconds) + 1
    return next_value + timedelta(seconds=missed * interval_seconds)


def _workflow_reference(config: dict[str, object]) -> WorkflowReference:
    workflow_value = config.get("workflow")
    workflow_config = workflow_value if isinstance(workflow_value, dict) else {}
    workflow_key = _optional_string(workflow_config.get("key"))
    if not workflow_key:
        workflow_key = _optional_string(workflow_config.get("workflow_key"))
    if not workflow_key:
        workflow_key = _optional_string(config.get("workflow_key"))
    if not workflow_key:
        raise ValueError("workflow_key is required.")
    version_value = workflow_config.get("version", config.get("workflow_version", 1))
    return WorkflowReference(workflow_key=workflow_key, version=_int_value(version_value, "workflow_version"))


def _schedule_target(config: dict[str, object]) -> ScheduleTarget:
    target_value = config.get("target")
    target = target_value if isinstance(target_value, dict) else {}
    if target_value is not None and not isinstance(target_value, dict):
        raise ValueError("target must be a JSON object.")
    character_id = _optional_positive_int(config.get("character_id"), "character_id")
    if character_id is None:
        character_id = _optional_positive_int(target.get("character_id"), "target.character_id")
    payload: dict[str, object] = dict(target)
    if character_id is not None:
        payload["character_id"] = character_id
    identity_text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if identity_text == "{}":
        return ScheduleTarget(None, "none", {})
    digest = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()[:16]
    prefix = f"character:{character_id}" if character_id is not None else "target"
    return ScheduleTarget(character_id, f"{prefix}:{digest}", payload)


def _job_payload_json(
    schedule: ScheduleDefinition,
    workflow: WorkflowDefinition,
    occurrence_key: str,
    scheduled_for: str,
    target: dict[str, object],
) -> str:
    payload = {
        "source": "schedule",
        "schedule_id": schedule.id,
        "schedule_key": schedule.schedule_key,
        "workflow_id": workflow.id,
        "workflow_key": workflow.workflow_key,
        "workflow_version": workflow.version,
        "occurrence_key": occurrence_key,
        "scheduled_for": scheduled_for,
        "target": target,
    }
    return json.dumps(payload, sort_keys=True)


def _json_object(value: str, field_name: str) -> dict[str, object]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return parsed


def _string_config(config: dict[str, object], key: str, *, default: str) -> str:
    value = config.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    cleaned = value.strip()
    return cleaned or default


def _int_config(config: dict[str, object], key: str, *, default: int) -> int:
    return _int_value(config.get(key, default), key)


def _int_value(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


def _optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _int_value(value, field_name)


def _optional_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _safe_message(exc: Exception) -> str:
    message = str(exc).strip()
    if len(message) > 300:
        return message[:297] + "..."
    return message or exc.__class__.__name__
