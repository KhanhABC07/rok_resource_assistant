from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from rok_assistant.db.models import Character, Incident, Job, JobRun, utc_now_iso
from rok_assistant.scheduler.clock import utc_datetime_to_text
from rok_assistant.workflow_engine import (
    CancellationToken,
    WorkflowExecutionResult,
    WorkflowOutcome,
    WorkflowStepResult,
)


PEACE_SHIELD_WORKFLOW_KEY = "peace-shield"
PEACE_SHIELD_STATES = (
    "validate_input",
    "load_character",
    "verify_active_city",
    "evaluate_policy",
    "open_shield_menu",
    "select_shield",
    "apply_shield",
    "verify_shield_active",
    "complete",
    "failed",
    "cancelled",
)
PEACE_SHIELD_TEMPLATE_KEYS = (
    "city.attack_warning",
    "city.scene",
    "city.buffs.button",
    "city.peace_shield.menu",
    "city.peace_shield.item",
    "city.peace_shield.apply",
    "city.peace_shield.active",
)
EMERGENCY_PEACE_SHIELD_PRIORITY = 1


class ShieldSource(StrEnum):
    ITEM = "ITEM"
    BUFF = "BUFF"
    GEM_PURCHASE = "GEM_PURCHASE"


class AttackSignalStatus(StrEnum):
    DETECTED = "DETECTED"
    NOT_DETECTED = "NOT_DETECTED"
    UNKNOWN = "UNKNOWN"


class AttackMonitorDecision(StrEnum):
    ENQUEUED = "ENQUEUED"
    ALREADY_PENDING = "ALREADY_PENDING"
    DEBOUNCED = "DEBOUNCED"
    COOLDOWN = "COOLDOWN"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    BLOCKED = "BLOCKED"


def _require_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _require_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be zero or greater.")
    return value


def _require_non_negative_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or float(value) < 0:
        raise ValueError(f"{field_name} must be zero or greater.")
    return float(value)


@dataclass(frozen=True)
class ShieldSpendLimit:
    max_gems_per_activation: int = 0
    used_gems_today: int = 0
    max_gems_per_day: int = 0

    def normalized(self) -> ShieldSpendLimit:
        return ShieldSpendLimit(
            max_gems_per_activation=_require_non_negative_int(
                self.max_gems_per_activation,
                "max_gems_per_activation",
            ),
            used_gems_today=_require_non_negative_int(self.used_gems_today, "used_gems_today"),
            max_gems_per_day=_require_non_negative_int(self.max_gems_per_day, "max_gems_per_day"),
        )

    def allows(self, gem_cost: int) -> bool:
        normalized = self.normalized()
        cost = _require_non_negative_int(gem_cost, "gem_cost")
        if cost == 0:
            return True
        if normalized.max_gems_per_activation <= 0:
            return False
        if cost > normalized.max_gems_per_activation:
            return False
        return normalized.used_gems_today + cost <= normalized.max_gems_per_day

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "max_gems_per_activation": normalized.max_gems_per_activation,
            "used_gems_today": normalized.used_gems_today,
            "max_gems_per_day": normalized.max_gems_per_day,
            "remaining_gems_today": max(
                0,
                normalized.max_gems_per_day - normalized.used_gems_today,
            ),
        }


@dataclass(frozen=True)
class PeaceShieldPolicy:
    allowed_durations_hours: tuple[int, ...] = (8, 24)
    allow_inventory_items: bool = True
    allow_buff_activation: bool = True
    allow_gem_spend: bool = False
    manual_override: bool = False
    spend_limit: ShieldSpendLimit = field(default_factory=ShieldSpendLimit)

    def normalized(self) -> PeaceShieldPolicy:
        durations = tuple(
            sorted({_require_positive_int(value, "allowed_durations_hours") for value in self.allowed_durations_hours})
        )
        if not durations:
            raise ValueError("At least one shield duration must be allowed.")
        return PeaceShieldPolicy(
            allowed_durations_hours=durations,
            allow_inventory_items=bool(self.allow_inventory_items),
            allow_buff_activation=bool(self.allow_buff_activation),
            allow_gem_spend=bool(self.allow_gem_spend),
            manual_override=bool(self.manual_override),
            spend_limit=self.spend_limit.normalized(),
        )

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "allowed_durations_hours": list(normalized.allowed_durations_hours),
            "allow_inventory_items": normalized.allow_inventory_items,
            "allow_buff_activation": normalized.allow_buff_activation,
            "allow_gem_spend": normalized.allow_gem_spend,
            "manual_override": normalized.manual_override,
            "spend_limit": normalized.spend_limit.to_json(),
        }


@dataclass(frozen=True)
class PeaceShieldRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    policy: PeaceShieldPolicy = field(default_factory=PeaceShieldPolicy)
    target_account_id: int | None = None
    session_key: str = ""
    attack_signal_id: str = ""
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "incoming-attack-monitor"


@dataclass(frozen=True)
class PeaceShieldConfig:
    workflow_timeout_seconds: float = 90.0
    step_timeout_seconds: float = 10.0


@dataclass(frozen=True)
class CityVerification:
    verified: bool
    city_scene: bool = False
    active_character_id: int | None = None
    shield_active: bool = False
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "verified": self.verified,
            "city_scene": self.city_scene,
            "active_character_id": self.active_character_id,
            "shield_active": self.shield_active,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class ShieldOption:
    duration_hours: int
    source: ShieldSource | str = ShieldSource.ITEM
    available: bool = True
    quantity: int = 1
    gem_cost: int = 0
    confidence: float = 1.0
    target: tuple[int, int] | None = None
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_source(self) -> ShieldSource:
        if isinstance(self.source, ShieldSource):
            return self.source
        return ShieldSource(str(self.source).strip().upper())

    def to_json(self) -> dict[str, object]:
        return {
            "duration_hours": self.duration_hours,
            "source": self.normalized_source().value,
            "available": self.available,
            "quantity": self.quantity,
            "gem_cost": self.gem_cost,
            "confidence": self.confidence,
            "target": None if self.target is None else {"x": self.target[0], "y": self.target[1]},
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class ShieldInventoryScan:
    options: tuple[ShieldOption, ...] = ()
    scene_verified: bool = True
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "options": [option.to_json() for option in self.options],
            "scene_verified": self.scene_verified,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class ShieldActionResult:
    success: bool
    verified: bool = False
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "verified": self.verified,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class AttackSignal:
    status: AttackSignalStatus | str
    signal_id: str
    instance_id: int
    character_id: int
    confidence: float = 1.0
    observed_at: datetime | None = None
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_status(self) -> AttackSignalStatus:
        if isinstance(self.status, AttackSignalStatus):
            return self.status
        return AttackSignalStatus(str(self.status).strip().upper())

    def to_json(self) -> dict[str, object]:
        observed_at = self.observed_at or datetime.now(UTC)
        return {
            "status": self.normalized_status().value,
            "signal_id": self.signal_id,
            "instance_id": self.instance_id,
            "character_id": self.character_id,
            "confidence": self.confidence,
            "observed_at": utc_datetime_to_text(observed_at),
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class AttackMonitorConfig:
    debounce_seconds: float = 30.0
    cooldown_seconds: float = 300.0
    priority: int = EMERGENCY_PEACE_SHIELD_PRIORITY
    workflow_version: int = 1

    def __post_init__(self) -> None:
        _require_non_negative_float(self.debounce_seconds, "debounce_seconds")
        _require_non_negative_float(self.cooldown_seconds, "cooldown_seconds")
        _require_positive_int(self.priority, "priority")
        _require_positive_int(self.workflow_version, "workflow_version")


@dataclass(frozen=True)
class AttackMonitorResult:
    decision: AttackMonitorDecision
    job: Job | None = None
    message: str = ""
    signal: AttackSignal | None = None


class CharacterRepository(Protocol):
    def get(self, character_id: int) -> Character | None:
        ...


class JobRepository(Protocol):
    def create_if_absent(self, job: Job) -> tuple[Job, bool]:
        ...


class JobRunRepository(Protocol):
    def get(self, run_id: int) -> JobRun | None:
        ...

    def save(self, run: JobRun) -> int:
        ...


class IncidentRepository(Protocol):
    def save(self, incident: Incident) -> int:
        ...


class OperatorNotifier(Protocol):
    def notify_critical(self, *, title: str, message: str, data: dict[str, object]) -> None:
        ...


class SchedulerWakeup(Protocol):
    def wake(self) -> None:
        ...


class IncomingAttackDetector(Protocol):
    def detect(self) -> AttackSignal:
        ...


class PeaceShieldDriver(Protocol):
    def verify_active_city(
        self,
        request: PeaceShieldRequest,
        character: Character,
    ) -> CityVerification:
        ...

    def open_shield_menu(
        self,
        request: PeaceShieldRequest,
        character: Character,
        policy: PeaceShieldPolicy,
    ) -> ShieldInventoryScan:
        ...

    def select_shield(
        self,
        request: PeaceShieldRequest,
        character: Character,
        option: ShieldOption,
        policy: PeaceShieldPolicy,
    ) -> ShieldActionResult:
        ...

    def apply_shield(
        self,
        request: PeaceShieldRequest,
        character: Character,
        option: ShieldOption,
        policy: PeaceShieldPolicy,
    ) -> ShieldActionResult:
        ...

    def verify_shield_active(
        self,
        request: PeaceShieldRequest,
        character: Character,
        option: ShieldOption,
        policy: PeaceShieldPolicy,
    ) -> CityVerification:
        ...


@dataclass
class _PeaceShieldState:
    request: PeaceShieldRequest
    character: Character | None = None
    policy: PeaceShieldPolicy | None = None
    city_verification: CityVerification | None = None
    scan: ShieldInventoryScan | None = None
    selected_option: ShieldOption | None = None
    selection: ShieldActionResult | None = None
    application: ShieldActionResult | None = None
    postcondition: CityVerification | None = None
    ignored_options: list[dict[str, object]] = field(default_factory=list)
    terminal_outcome: WorkflowOutcome = WorkflowOutcome.SUCCESS
    terminal_state: str = ""
    terminal_reason: str = ""
    screenshot_path: str = ""
    incident_opened: bool = False
    operator_notified: bool = False

    @property
    def failed(self) -> bool:
        return self.terminal_outcome.is_failure

    def fail(
        self,
        state: str,
        reason: str,
        *,
        outcome: WorkflowOutcome = WorkflowOutcome.BLOCKED,
        screenshot_path: str = "",
    ) -> None:
        self.terminal_state = state
        self.terminal_reason = reason
        self.terminal_outcome = outcome
        if screenshot_path:
            self.screenshot_path = screenshot_path


class PeaceShieldWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: PeaceShieldDriver,
        job_runs: JobRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        notifier: OperatorNotifier | None = None,
        config: PeaceShieldConfig | None = None,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.job_runs = job_runs
        self.incidents = incidents
        self.notifier = notifier
        self.config = config or PeaceShieldConfig()

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return PEACE_SHIELD_STATES

    def execute(
        self,
        request: PeaceShieldRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        state = _PeaceShieldState(request=request)
        token = cancellation_token or CancellationToken()
        started_at = utc_now_iso()
        job_run_id = self._start_job_run(request, started_at)
        steps: list[WorkflowStepResult] = []
        for step_key, handler in (
            ("validate_input", self._validate_input),
            ("load_character", self._load_character),
            ("verify_active_city", self._verify_active_city),
            ("evaluate_policy", self._evaluate_policy),
            ("open_shield_menu", self._open_shield_menu),
            ("select_shield", self._select_shield),
            ("apply_shield", self._apply_shield),
            ("verify_shield_active", self._verify_shield_active),
            ("complete", self._complete),
        ):
            if token.is_cancelled:
                state.fail(step_key, token.reason or "Workflow cancelled.", outcome=WorkflowOutcome.CANCELLED)
                steps.append(_step_result(step_key, WorkflowOutcome.CANCELLED, state.terminal_reason))
                break
            if state.failed:
                break
            steps.append(handler(step_key, state))
        if state.failed:
            self._open_incident_and_notify(state, job_run_id)
            steps.append(_step_result("failed", WorkflowOutcome.FATAL_FAILURE, state.terminal_reason, screenshot_path=state.screenshot_path))
        outcome = state.terminal_outcome
        message = state.terminal_reason
        if not state.failed and outcome != WorkflowOutcome.CANCELLED:
            outcome = WorkflowOutcome.SUCCESS
            message = "Peace shield activated and verified."
        result = WorkflowExecutionResult(
            workflow_key=PEACE_SHIELD_WORKFLOW_KEY,
            schema_version=1,
            outcome=outcome,
            message=message,
            steps=steps,
            started_at=started_at,
            finished_at=utc_now_iso(),
            job_run_id=job_run_id,
            result=self._payload(state),
        )
        self._finish_job_run(result, state)
        return result

    def _validate_input(self, step_key: str, state: _PeaceShieldState) -> WorkflowStepResult:
        request = state.request
        try:
            _require_positive_int(request.instance_id, "instance_id")
            _require_positive_int(request.character_id, "character_id")
            state.policy = request.policy.normalized()
        except ValueError as exc:
            state.fail(step_key, str(exc), outcome=WorkflowOutcome.VALIDATION_FAILURE)
            return _step_result(step_key, WorkflowOutcome.VALIDATION_FAILURE, str(exc))
        return _step_result(step_key, WorkflowOutcome.SUCCESS, data={"policy": state.policy.to_json()})

    def _load_character(self, step_key: str, state: _PeaceShieldState) -> WorkflowStepResult:
        character = self.characters.get(state.request.character_id)
        if character is None or not character.enabled:
            state.fail(step_key, "Target character is missing or disabled.")
            return _step_result(step_key, WorkflowOutcome.BLOCKED, state.terminal_reason)
        state.character = character
        return _step_result(step_key, WorkflowOutcome.SUCCESS, data={"character_id": character.id or 0})

    def _verify_active_city(self, step_key: str, state: _PeaceShieldState) -> WorkflowStepResult:
        verification = self.driver.verify_active_city(state.request, _require_character(state))
        state.city_verification = verification
        if verification.screenshot_path:
            state.screenshot_path = verification.screenshot_path
        if not verification.verified or not verification.city_scene:
            state.fail(
                step_key,
                verification.message or "Active city state could not be verified.",
                screenshot_path=verification.screenshot_path,
            )
            return _step_result(step_key, WorkflowOutcome.BLOCKED, state.terminal_reason, data=verification.to_json(), screenshot_path=verification.screenshot_path)
        if verification.active_character_id not in (None, state.request.character_id):
            state.fail(step_key, "Active character does not match shield request.", screenshot_path=verification.screenshot_path)
            return _step_result(step_key, WorkflowOutcome.BLOCKED, state.terminal_reason, data=verification.to_json(), screenshot_path=verification.screenshot_path)
        if verification.shield_active:
            state.terminal_reason = "Peace shield is already active."
        return _step_result(step_key, WorkflowOutcome.SUCCESS, data=verification.to_json(), screenshot_path=verification.screenshot_path)

    def _evaluate_policy(self, step_key: str, state: _PeaceShieldState) -> WorkflowStepResult:
        policy = _require_policy(state)
        if (
            not policy.allow_inventory_items
            and not policy.allow_buff_activation
            and not policy.allow_gem_spend
            and not policy.manual_override
        ):
            state.fail(step_key, "Peace shield policy denies all shield sources.")
            return _step_result(step_key, WorkflowOutcome.BLOCKED, state.terminal_reason)
        return _step_result(step_key, WorkflowOutcome.SUCCESS, data=policy.to_json())

    def _open_shield_menu(self, step_key: str, state: _PeaceShieldState) -> WorkflowStepResult:
        scan = self.driver.open_shield_menu(state.request, _require_character(state), _require_policy(state))
        state.scan = scan
        if scan.screenshot_path:
            state.screenshot_path = scan.screenshot_path
        if not scan.scene_verified:
            state.fail(step_key, scan.message or "Peace shield menu could not be verified.", screenshot_path=scan.screenshot_path)
            return _step_result(step_key, WorkflowOutcome.BLOCKED, state.terminal_reason, data=scan.to_json(), screenshot_path=scan.screenshot_path)
        return _step_result(step_key, WorkflowOutcome.SUCCESS, data=scan.to_json(), screenshot_path=scan.screenshot_path)

    def _select_shield(self, step_key: str, state: _PeaceShieldState) -> WorkflowStepResult:
        option, ignored = _choose_option(_require_scan(state), _require_policy(state))
        state.ignored_options = ignored
        if option is None:
            state.fail(step_key, "No allowed peace shield item or buff is available.", screenshot_path=state.screenshot_path)
            return _step_result(step_key, WorkflowOutcome.BLOCKED, state.terminal_reason, data={"ignored_options": ignored}, screenshot_path=state.screenshot_path)
        state.selected_option = option
        result = self.driver.select_shield(state.request, _require_character(state), option, _require_policy(state))
        state.selection = result
        if result.screenshot_path:
            state.screenshot_path = result.screenshot_path
        if not result.success:
            state.fail(step_key, result.message or "Allowed peace shield could not be selected.", screenshot_path=result.screenshot_path)
            return _step_result(step_key, WorkflowOutcome.BLOCKED, state.terminal_reason, data=result.to_json(), screenshot_path=result.screenshot_path)
        return _step_result(step_key, WorkflowOutcome.SUCCESS, data={"selected_option": option.to_json(), "selection": result.to_json()})

    def _apply_shield(self, step_key: str, state: _PeaceShieldState) -> WorkflowStepResult:
        result = self.driver.apply_shield(state.request, _require_character(state), _require_option(state), _require_policy(state))
        state.application = result
        if result.screenshot_path:
            state.screenshot_path = result.screenshot_path
        if not result.success:
            state.fail(step_key, result.message or "Peace shield application failed.", screenshot_path=result.screenshot_path)
            return _step_result(step_key, WorkflowOutcome.BLOCKED, state.terminal_reason, data=result.to_json(), screenshot_path=result.screenshot_path)
        return _step_result(step_key, WorkflowOutcome.SUCCESS, data=result.to_json(), screenshot_path=result.screenshot_path)

    def _verify_shield_active(self, step_key: str, state: _PeaceShieldState) -> WorkflowStepResult:
        verification = self.driver.verify_shield_active(state.request, _require_character(state), _require_option(state), _require_policy(state))
        state.postcondition = verification
        if verification.screenshot_path:
            state.screenshot_path = verification.screenshot_path
        if not verification.verified or not verification.shield_active:
            state.fail(step_key, verification.message or "Peace shield active postcondition was not verified.", screenshot_path=verification.screenshot_path)
            return _step_result(step_key, WorkflowOutcome.BLOCKED, state.terminal_reason, data=verification.to_json(), screenshot_path=verification.screenshot_path)
        return _step_result(step_key, WorkflowOutcome.SUCCESS, data=verification.to_json(), screenshot_path=verification.screenshot_path)

    def _complete(self, step_key: str, state: _PeaceShieldState) -> WorkflowStepResult:
        return _step_result(step_key, WorkflowOutcome.SUCCESS, data=self._payload(state))

    def _payload(self, state: _PeaceShieldState) -> dict[str, object]:
        return {
            "policy": state.policy.to_json() if state.policy is not None else {},
            "city_verification": state.city_verification.to_json() if state.city_verification else {},
            "scan": state.scan.to_json() if state.scan else {},
            "selected_option": state.selected_option.to_json() if state.selected_option else {},
            "ignored_options": state.ignored_options,
            "application": state.application.to_json() if state.application else {},
            "postcondition": state.postcondition.to_json() if state.postcondition else {},
            "terminal_state": state.terminal_state,
            "terminal_reason": state.terminal_reason,
            "failure_evidence": {"screenshot_path": state.screenshot_path},
            "incident_opened": state.incident_opened,
            "operator_notified": state.operator_notified,
        }

    def _open_incident_and_notify(self, state: _PeaceShieldState, job_run_id: int | None) -> None:
        title = "Peace shield emergency failed"
        if self.incidents is not None and not state.incident_opened:
            self.incidents.save(
                Incident(
                    incident_key=f"peace-shield:{state.request.instance_id}:{state.request.character_id}:{uuid4().hex}",
                    severity="critical",
                    status="open",
                    title=title,
                    details=state.terminal_reason,
                    job_run_id=job_run_id,
                    screenshot_path=state.screenshot_path,
                )
            )
            state.incident_opened = True
        if self.notifier is not None and not state.operator_notified:
            self.notifier.notify_critical(
                title=title,
                message=state.terminal_reason,
                data=self._payload(state),
            )
            state.operator_notified = True

    def _start_job_run(self, request: PeaceShieldRequest, started_at: str) -> int | None:
        if self.job_runs is None or request.job_id is None:
            return None
        return self.job_runs.save(
            JobRun(
                job_id=request.job_id,
                run_key=request.run_key or f"peace-shield:{request.instance_id}:{uuid4().hex}",
                status="running",
                attempt=request.run_attempt,
                started_at=started_at,
            )
        )

    def _finish_job_run(self, result: WorkflowExecutionResult, state: _PeaceShieldState) -> None:
        if self.job_runs is None or result.job_run_id is None:
            return
        run = self.job_runs.get(result.job_run_id)
        if run is None:
            return
        run.status = "failed" if result.outcome.is_failure else "completed"
        run.finished_at = result.finished_at
        run.result_json = json.dumps(result.to_json_dict(), sort_keys=True)
        run.error_message = result.message if result.outcome.is_failure else ""
        run.screenshot_path = state.screenshot_path
        self.job_runs.save(run)


class IncomingAttackMonitor:
    def __init__(
        self,
        *,
        detector: IncomingAttackDetector,
        jobs: JobRepository,
        scheduler: SchedulerWakeup | None = None,
        config: AttackMonitorConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.detector = detector
        self.jobs = jobs
        self.scheduler = scheduler
        self.config = config or AttackMonitorConfig()
        self.clock = clock or (lambda: datetime.now(UTC))
        self._last_signal_at_by_key: dict[tuple[int, int, str], datetime] = {}
        self._cooldown_until_by_target: dict[tuple[int, int], datetime] = {}

    def poll(self) -> AttackMonitorResult:
        signal = self.detector.detect()
        now = self.clock()
        status = signal.normalized_status()
        if status != AttackSignalStatus.DETECTED:
            return AttackMonitorResult(AttackMonitorDecision.FALSE_POSITIVE, message=signal.message, signal=signal)
        target_key = (signal.instance_id, signal.character_id)
        signal_key = (signal.instance_id, signal.character_id, signal.signal_id)
        last_signal = self._last_signal_at_by_key.get(signal_key)
        if last_signal is not None and now - last_signal < timedelta(seconds=self.config.debounce_seconds):
            return AttackMonitorResult(AttackMonitorDecision.DEBOUNCED, message="Duplicate attack signal ignored.", signal=signal)
        cooldown_until = self._cooldown_until_by_target.get(target_key)
        if cooldown_until is not None and now < cooldown_until:
            self._last_signal_at_by_key[signal_key] = now
            return AttackMonitorResult(AttackMonitorDecision.COOLDOWN, message="Attack shield emergency job is cooling down.", signal=signal)
        self._last_signal_at_by_key[signal_key] = now
        payload = {
            "source": "incoming_attack_monitor",
            "workflow_key": PEACE_SHIELD_WORKFLOW_KEY,
            "workflow_version": self.config.workflow_version,
            "target": {
                "instance_id": signal.instance_id,
                "character_id": signal.character_id,
            },
            "attack_signal": signal.to_json(),
        }
        job = Job(
            character_id=signal.character_id,
            idempotency_key=f"emergency-peace-shield:{signal.instance_id}:{signal.character_id}:{signal.signal_id}",
            job_type="workflow",
            status="pending",
            priority=self.config.priority,
            scheduled_for=utc_datetime_to_text(now),
            payload_json=json.dumps(payload, sort_keys=True),
        )
        try:
            persisted, created = self.jobs.create_if_absent(job)
        except ValueError as exc:
            return AttackMonitorResult(AttackMonitorDecision.BLOCKED, message=str(exc), signal=signal)
        self._cooldown_until_by_target[target_key] = now + timedelta(seconds=self.config.cooldown_seconds)
        if created and self.scheduler is not None:
            self.scheduler.wake()
        return AttackMonitorResult(
            AttackMonitorDecision.ENQUEUED if created else AttackMonitorDecision.ALREADY_PENDING,
            job=persisted,
            signal=signal,
        )


def _choose_option(
    scan: ShieldInventoryScan,
    policy: PeaceShieldPolicy,
) -> tuple[ShieldOption | None, list[dict[str, object]]]:
    ignored: list[dict[str, object]] = []
    for option in sorted(scan.options, key=lambda item: item.duration_hours):
        reason = _option_denial_reason(option, policy)
        if reason:
            ignored.append({**option.to_json(), "ignored_reason": reason})
            continue
        return option, ignored
    return None, ignored


def _option_denial_reason(option: ShieldOption, policy: PeaceShieldPolicy) -> str:
    try:
        source = option.normalized_source()
    except ValueError:
        return "unknown_shield_source"
    if option.duration_hours not in policy.allowed_durations_hours:
        return "duration_not_allowed"
    if not option.available or option.quantity <= 0:
        return "not_available"
    if source == ShieldSource.ITEM and not policy.allow_inventory_items:
        return "inventory_items_denied"
    if source == ShieldSource.BUFF and not policy.allow_buff_activation:
        return "buff_activation_denied"
    if source == ShieldSource.GEM_PURCHASE:
        if not policy.allow_gem_spend and not policy.manual_override:
            return "gem_spend_denied"
        if not policy.spend_limit.allows(option.gem_cost) and not policy.manual_override:
            return "spend_limit_denied"
    return ""


def _step_result(
    step_key: str,
    outcome: WorkflowOutcome,
    message: str = "",
    *,
    data: dict[str, object] | None = None,
    screenshot_path: str = "",
) -> WorkflowStepResult:
    return WorkflowStepResult(
        step_key=step_key,
        action_type=f"peace_shield.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _PeaceShieldState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _PeaceShieldState) -> PeaceShieldPolicy:
    if state.policy is None:
        raise RuntimeError("Peace shield policy has not been validated.")
    return state.policy


def _require_scan(state: _PeaceShieldState) -> ShieldInventoryScan:
    if state.scan is None:
        raise RuntimeError("Peace shield options have not been scanned.")
    return state.scan


def _require_option(state: _PeaceShieldState) -> ShieldOption:
    if state.selected_option is None:
        raise RuntimeError("Peace shield option has not been selected.")
    return state.selected_option
