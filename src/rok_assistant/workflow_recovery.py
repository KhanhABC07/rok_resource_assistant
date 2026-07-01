from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from rok_assistant.workflow_serialization import safe_json_payload
from rok_assistant.workflow_types import WorkflowOutcome, WorkflowStepResult


class RecoveryPhase(str, Enum):
    NOT_STARTED = "not_started"
    PRECONDITION_VERIFIED = "precondition_verified"
    SIDE_EFFECT_STARTED = "side_effect_started"
    SIDE_EFFECT_UNCERTAIN = "side_effect_uncertain"
    POSTCONDITION_VERIFIED = "postcondition_verified"
    COMPLETED = "completed"


SAFE_RESUME_PHASES = {
    RecoveryPhase.NOT_STARTED,
    RecoveryPhase.PRECONDITION_VERIFIED,
}
UNCERTAIN_SIDE_EFFECT_PHASES = {
    RecoveryPhase.SIDE_EFFECT_STARTED,
    RecoveryPhase.SIDE_EFFECT_UNCERTAIN,
}
TERMINAL_RUN_STATUSES = {"completed", "failed", "aborted", "cancelled"}


@dataclass(frozen=True)
class PersistedPayload:
    ok: bool
    value: dict[str, object] = field(default_factory=dict)
    message: str = ""


@dataclass(frozen=True)
class StepRecoveryDecision:
    step_run_id: int
    phase: RecoveryPhase
    result: WorkflowStepResult | None = None
    can_resume: bool = False
    requires_postcondition: bool = False
    payload: dict[str, object] = field(default_factory=dict)


def parse_persisted_payload(text: str, *, source: str) -> PersistedPayload:
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError as exc:
        return PersistedPayload(False, message=f"{source} contains malformed JSON: {exc.msg}.")
    if not isinstance(value, dict):
        return PersistedPayload(False, message=f"{source} must contain a JSON object.")
    return PersistedPayload(True, dict(value))


def payload_with_recovery(
    payload: Mapping[str, object],
    phase: RecoveryPhase,
    *,
    source: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    output = dict(payload)
    recovery: dict[str, object] = {}
    existing = output.get("recovery")
    if isinstance(existing, Mapping):
        recovery.update(dict(existing))
    recovery["phase"] = phase.value
    if extra:
        recovery.update(dict(extra))
    output["recovery"] = recovery
    return safe_json_payload(output, source=source)


def recovery_phase_from_payload(
    payload: Mapping[str, object],
    *,
    default: RecoveryPhase,
) -> RecoveryPhase | None:
    recovery = payload.get("recovery")
    if not isinstance(recovery, Mapping):
        return default
    raw_phase = str(recovery.get("phase") or "").strip()
    if not raw_phase:
        return default
    try:
        return RecoveryPhase(raw_phase)
    except ValueError:
        return None


def outcome_from_status(status: str) -> WorkflowOutcome:
    if status == "completed":
        return WorkflowOutcome.SUCCESS
    if status == "cancelled":
        return WorkflowOutcome.CANCELLED
    if status == "aborted":
        return WorkflowOutcome.CANCELLED
    return WorkflowOutcome.FATAL_FAILURE


def outcome_from_payload(
    payload: Mapping[str, object],
    *,
    fallback_status: str,
) -> WorkflowOutcome:
    raw_outcome = str(payload.get("outcome") or "").strip()
    if raw_outcome:
        try:
            return WorkflowOutcome(raw_outcome)
        except ValueError:
            return WorkflowOutcome.FATAL_FAILURE
    return outcome_from_status(fallback_status)
