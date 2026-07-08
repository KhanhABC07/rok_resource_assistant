from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from rok_assistant.db.models import Character, Incident, JobRun
from rok_assistant.tasks.resource_search_workflow import ResourceGatheringActionResult
from rok_assistant.workflow_context import (
    CancellationToken,
    StepBudget,
    WorkflowDeadline,
    WorkflowExecutionContext,
)
from rok_assistant.workflow_engine import (
    ActionRegistry,
    WorkflowEngine,
    WorkflowExecutionResult,
    WorkflowOutcome,
    WorkflowRunRepositoryRecorder,
    WorkflowStepResult,
    WorkflowStepSpec,
)


ALLIANCE_TECHNOLOGY_DONATION_WORKFLOW_KEY = "alliance-technology-donation"
ALLIANCE_TECHNOLOGY_DONATION_TEMPLATE_KEYS = (
    "city.alliance.button",
    "alliance.menu.technology",
    "alliance.technology.panel",
    "alliance.technology.recommended_badge",
    "alliance.technology.donate_button",
    "alliance.technology.resource_cost",
    "alliance.technology.gem_confirmation",
    "alliance.technology.contribution_limit",
    "alliance.technology.verification_required",
)
ALLIANCE_TECHNOLOGY_DONATION_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "open_alliance",
    "open_alliance_technology",
    "select_technology",
    "scan_donation_state",
    "donate",
    "handle_confirmation",
    "verify_donation_state",
    "complete",
    "skipped",
    "recover",
    "failed",
    "cancelled",
)


class AllianceTechnologyScanStatus(StrEnum):
    READY = "READY"
    NONE_AVAILABLE = "NONE_AVAILABLE"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


class AllianceDonationStateStatus(StrEnum):
    READY = "READY"
    NO_DONATION_AVAILABLE = "NO_DONATION_AVAILABLE"
    CONTRIBUTION_LIMIT_REACHED = "CONTRIBUTION_LIMIT_REACHED"
    INSUFFICIENT_RESOURCES = "INSUFFICIENT_RESOURCES"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


class AllianceDonationCostType(StrEnum):
    FOOD = "FOOD"
    WOOD = "WOOD"
    STONE = "STONE"
    GOLD = "GOLD"
    RESOURCE = "RESOURCE"
    FREE = "FREE"
    GEMS = "GEMS"
    UNKNOWN = "UNKNOWN"


class AllianceDonationConfirmation(StrEnum):
    NONE = "NONE"
    SAFE_RESOURCE = "SAFE_RESOURCE"
    PREMIUM_CURRENCY = "PREMIUM_CURRENCY"
    UNKNOWN = "UNKNOWN"


def _technology_key(value: str) -> str:
    key = str(value).strip().upper()
    if not key:
        raise ValueError("Alliance technology key must be configured.")
    return key


def _cost_type(value: AllianceDonationCostType | str) -> AllianceDonationCostType:
    if isinstance(value, AllianceDonationCostType):
        return value
    try:
        return AllianceDonationCostType(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in AllianceDonationCostType)
        raise ValueError(f"Invalid donation cost type: {value!r}. Expected one of: {valid}.") from exc


def _confirmation(value: AllianceDonationConfirmation | str) -> AllianceDonationConfirmation:
    if isinstance(value, AllianceDonationConfirmation):
        return value
    try:
        return AllianceDonationConfirmation(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in AllianceDonationConfirmation)
        raise ValueError(f"Invalid donation confirmation: {value!r}. Expected one of: {valid}.") from exc


def _require_positive_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _require_non_negative_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be zero or greater.")
    return value


def _require_confidence(value: float, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    return numeric


@dataclass(frozen=True)
class AllianceTechnologyDonationPolicy:
    target_technology_key: str = ""
    recommended_only: bool = True
    minimum_detector_confidence: float = 0.85
    max_donations_per_run: int = 1
    max_donations_per_day: int = 20
    donations_used_today: int = 0
    allowed_cost_types: tuple[AllianceDonationCostType | str, ...] = (
        AllianceDonationCostType.FREE,
        AllianceDonationCostType.FOOD,
        AllianceDonationCostType.WOOD,
        AllianceDonationCostType.STONE,
        AllianceDonationCostType.GOLD,
        AllianceDonationCostType.RESOURCE,
    )
    allow_premium_currency: bool = False
    max_resource_cost_per_donation: int | None = None

    def normalized(self) -> AllianceTechnologyDonationPolicy:
        target = str(self.target_technology_key).strip()
        if not target and not self.recommended_only:
            raise ValueError("target_technology_key is required when recommended_only is false.")
        allowed = tuple(dict.fromkeys(_cost_type(item) for item in self.allowed_cost_types))
        if not allowed:
            raise ValueError("At least one donation cost type must be allowed.")
        if AllianceDonationCostType.GEMS in allowed and not self.allow_premium_currency:
            raise ValueError("GEMS cannot be allowed unless allow_premium_currency is true.")
        max_resource_cost = self.max_resource_cost_per_donation
        if max_resource_cost is not None:
            max_resource_cost = _require_non_negative_int(max_resource_cost, "max_resource_cost_per_donation")
        return AllianceTechnologyDonationPolicy(
            target_technology_key=_technology_key(target) if target else "",
            recommended_only=bool(self.recommended_only),
            minimum_detector_confidence=_require_confidence(
                self.minimum_detector_confidence,
                "minimum_detector_confidence",
            ),
            max_donations_per_run=_require_positive_int(self.max_donations_per_run, "max_donations_per_run"),
            max_donations_per_day=_require_positive_int(self.max_donations_per_day, "max_donations_per_day"),
            donations_used_today=_require_non_negative_int(self.donations_used_today, "donations_used_today"),
            allowed_cost_types=allowed,
            allow_premium_currency=bool(self.allow_premium_currency),
            max_resource_cost_per_donation=max_resource_cost,
        )

    @property
    def remaining_daily_budget(self) -> int:
        return max(0, self.max_donations_per_day - self.donations_used_today)

    @property
    def effective_run_budget(self) -> int:
        return min(self.max_donations_per_run, self.remaining_daily_budget)

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "target_technology_key": normalized.target_technology_key,
            "recommended_only": normalized.recommended_only,
            "minimum_detector_confidence": normalized.minimum_detector_confidence,
            "max_donations_per_run": normalized.max_donations_per_run,
            "max_donations_per_day": normalized.max_donations_per_day,
            "donations_used_today": normalized.donations_used_today,
            "remaining_daily_budget": normalized.remaining_daily_budget,
            "effective_run_budget": normalized.effective_run_budget,
            "allowed_cost_types": [item.value for item in normalized.allowed_cost_types],
            "allow_premium_currency": normalized.allow_premium_currency,
            "max_resource_cost_per_donation": normalized.max_resource_cost_per_donation,
        }


@dataclass(frozen=True)
class AllianceTechnologyDonationRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: AllianceTechnologyDonationPolicy = field(default_factory=AllianceTechnologyDonationPolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class AllianceTechnologyDonationConfig:
    workflow_timeout_seconds: float = 120.0
    step_timeout_seconds: float = 15.0
    precondition_retry_limit: int = 1
    navigation_retry_limit: int = 1
    scan_retry_limit: int = 1
    action_retry_limit: int = 0
    retry_delay_seconds: float = 0.25


@dataclass(frozen=True)
class AllianceTechnologyObservation:
    technology_key: str
    display_name: str = ""
    recommended: bool = False
    confidence: float = 1.0
    selected: bool = False
    contribution_count: int | None = None
    contribution_limit: int | None = None
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_technology_key(self) -> str:
        return _technology_key(self.technology_key)

    def to_json(self) -> dict[str, object]:
        return {
            "technology_key": self.normalized_technology_key(),
            "display_name": self.display_name,
            "recommended": self.recommended,
            "confidence": self.confidence,
            "selected": self.selected,
            "contribution_count": self.contribution_count,
            "contribution_limit": self.contribution_limit,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class AllianceTechnologyScan:
    status: AllianceTechnologyScanStatus | str
    observations: tuple[AllianceTechnologyObservation, ...] = ()
    scene_verified: bool = True
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_status(self) -> AllianceTechnologyScanStatus:
        if isinstance(self.status, AllianceTechnologyScanStatus):
            return self.status
        try:
            return AllianceTechnologyScanStatus(str(self.status).strip().upper())
        except ValueError as exc:
            valid = ", ".join(item.value for item in AllianceTechnologyScanStatus)
            raise ValueError(f"Invalid alliance technology scan status: {self.status!r}. Expected one of: {valid}.") from exc

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.normalized_status().value,
            "observations": [item.to_json() for item in self.observations],
            "scene_verified": self.scene_verified,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class AllianceDonationAction:
    action_key: str = "normal"
    cost_type: AllianceDonationCostType | str = AllianceDonationCostType.RESOURCE
    cost_amount: int = 0
    confidence: float = 1.0
    enabled: bool = True
    premium: bool = False
    message: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_cost_type(self) -> AllianceDonationCostType:
        return _cost_type(self.cost_type)

    def to_json(self) -> dict[str, object]:
        return {
            "action_key": self.action_key,
            "cost_type": self.normalized_cost_type().value,
            "cost_amount": self.cost_amount,
            "confidence": self.confidence,
            "enabled": self.enabled,
            "premium": self.premium,
            **self.data,
        }


@dataclass(frozen=True)
class AllianceDonationState:
    status: AllianceDonationStateStatus | str
    technology_key: str
    contribution_count: int | None = None
    contribution_limit: int | None = None
    available_actions: tuple[AllianceDonationAction, ...] = ()
    scene_verified: bool = True
    technology_verified: bool = True
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_status(self) -> AllianceDonationStateStatus:
        if isinstance(self.status, AllianceDonationStateStatus):
            return self.status
        try:
            return AllianceDonationStateStatus(str(self.status).strip().upper())
        except ValueError as exc:
            valid = ", ".join(item.value for item in AllianceDonationStateStatus)
            raise ValueError(f"Invalid alliance donation state status: {self.status!r}. Expected one of: {valid}.") from exc

    def normalized_technology_key(self) -> str:
        return _technology_key(self.technology_key)

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.normalized_status().value,
            "technology_key": self.normalized_technology_key(),
            "contribution_count": self.contribution_count,
            "contribution_limit": self.contribution_limit,
            "available_actions": [item.to_json() for item in self.available_actions],
            "scene_verified": self.scene_verified,
            "technology_verified": self.technology_verified,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class AllianceDonationAttemptResult:
    success: bool
    changed: bool = False
    donation_count: int = 0
    contribution_count: int | None = None
    contribution_limit: int | None = None
    confirmation: AllianceDonationConfirmation | str = AllianceDonationConfirmation.NONE
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_confirmation(self) -> AllianceDonationConfirmation:
        return _confirmation(self.confirmation)

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "changed": self.changed,
            "donation_count": self.donation_count,
            "contribution_count": self.contribution_count,
            "contribution_limit": self.contribution_limit,
            "confirmation": self.normalized_confirmation().value,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


class CharacterRepository(Protocol):
    def get(self, character_id: int) -> Character | None:
        ...


class JobRunRepository(Protocol):
    def get(self, run_id: int) -> JobRun | None:
        ...

    def save(self, run: JobRun) -> int:
        ...


class StepRunRepository(Protocol):
    ...


class IncidentRepository(Protocol):
    def save(self, incident: Incident) -> int:
        ...


class AllianceTechnologyAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: AllianceTechnologyDonationRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class AllianceTechnologyCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: AllianceTechnologyDonationRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class AllianceTechnologyDonationDriver(Protocol):
    def open_alliance(
        self,
        request: AllianceTechnologyDonationRequest,
        character: Character,
        policy: AllianceTechnologyDonationPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def open_alliance_technology(
        self,
        request: AllianceTechnologyDonationRequest,
        character: Character,
        policy: AllianceTechnologyDonationPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def scan_technologies(
        self,
        request: AllianceTechnologyDonationRequest,
        character: Character,
        policy: AllianceTechnologyDonationPolicy,
    ) -> AllianceTechnologyScan:
        ...

    def select_technology(
        self,
        request: AllianceTechnologyDonationRequest,
        character: Character,
        technology: AllianceTechnologyObservation,
        policy: AllianceTechnologyDonationPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def scan_donation_state(
        self,
        request: AllianceTechnologyDonationRequest,
        character: Character,
        technology: AllianceTechnologyObservation,
        policy: AllianceTechnologyDonationPolicy,
    ) -> AllianceDonationState:
        ...

    def donate_to_technology(
        self,
        request: AllianceTechnologyDonationRequest,
        character: Character,
        technology: AllianceTechnologyObservation,
        action: AllianceDonationAction,
        policy: AllianceTechnologyDonationPolicy,
    ) -> AllianceDonationAttemptResult:
        ...

    def handle_donation_confirmation(
        self,
        request: AllianceTechnologyDonationRequest,
        character: Character,
        technology: AllianceTechnologyObservation,
        action: AllianceDonationAction,
        attempt: AllianceDonationAttemptResult,
        policy: AllianceTechnologyDonationPolicy,
    ) -> AllianceDonationAttemptResult:
        ...

    def verify_donation_state(
        self,
        request: AllianceTechnologyDonationRequest,
        character: Character,
        technology: AllianceTechnologyObservation,
        before: AllianceDonationState,
        attempt: AllianceDonationAttemptResult,
        policy: AllianceTechnologyDonationPolicy,
    ) -> AllianceDonationAttemptResult:
        ...


class RecoveryWatchdog(Protocol):
    def monitor(
        self,
        *,
        instance_id: int,
        instance_index: int,
        instance_name: str,
        job_run_id: int | None = None,
    ) -> object:
        ...


@dataclass
class _AllianceTechnologyDonationState:
    request: AllianceTechnologyDonationRequest
    character: Character | None = None
    policy: AllianceTechnologyDonationPolicy | None = None
    technology_scan: dict[str, object] = field(default_factory=dict)
    selected_technology: AllianceTechnologyObservation | None = None
    donation_state: AllianceDonationState | None = None
    scanned_donation_states: list[dict[str, object]] = field(default_factory=list)
    donation_attempts: list[dict[str, object]] = field(default_factory=list)
    confirmation_attempts: list[dict[str, object]] = field(default_factory=list)
    verification_attempts: list[dict[str, object]] = field(default_factory=list)
    ignored_technologies: list[dict[str, object]] = field(default_factory=list)
    donation_count: int = 0
    terminal_outcome: WorkflowOutcome | None = None
    terminal_state: str = ""
    terminal_reason: str = ""
    recovery_outcome: dict[str, object] = field(default_factory=dict)
    screenshot_path: str = ""
    incident_opened: bool = False

    @property
    def stopped(self) -> bool:
        return self.terminal_outcome is not None

    @property
    def failed(self) -> bool:
        return self.terminal_outcome in {
            WorkflowOutcome.BLOCKED,
            WorkflowOutcome.FATAL_FAILURE,
            WorkflowOutcome.TIMEOUT,
            WorkflowOutcome.VALIDATION_FAILURE,
        }

    def stop(
        self,
        step_key: str,
        outcome: WorkflowOutcome,
        reason: str,
        *,
        screenshot_path: str = "",
        data: dict[str, object] | None = None,
    ) -> WorkflowStepResult:
        self.terminal_state = step_key
        self.terminal_reason = reason
        self.terminal_outcome = outcome
        if screenshot_path:
            self.screenshot_path = screenshot_path
        return _step_result(
            step_key,
            WorkflowOutcome.SUCCESS if outcome == WorkflowOutcome.SKIPPED else outcome,
            reason,
            data={
                "terminal_outcome": outcome.value,
                "terminal_state": step_key,
                "terminal_reason": reason,
                **(data or {}),
            },
            screenshot_path=screenshot_path,
        )


class AllianceTechnologyDonationWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: AllianceTechnologyDonationDriver,
        account_precondition: AllianceTechnologyAccountPrecondition | None = None,
        character_precondition: AllianceTechnologyCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: AllianceTechnologyDonationConfig | None = None,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.account_precondition = account_precondition
        self.character_precondition = character_precondition
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.incidents = incidents
        self.config = config or AllianceTechnologyDonationConfig()
        self._states: dict[str, _AllianceTechnologyDonationState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return ALLIANCE_TECHNOLOGY_DONATION_STATES

    def execute(
        self,
        request: AllianceTechnologyDonationRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _AllianceTechnologyDonationState(request=request)
        self._states[token] = state
        persistence = None
        if self.job_runs is not None and self.step_runs is not None and request.job_id is not None:
            persistence = WorkflowRunRepositoryRecorder(self.job_runs, self.step_runs)
        try:
            context = WorkflowExecutionContext(
                cancellation_token=cancellation_token or CancellationToken(),
                deadline=WorkflowDeadline.from_timeout(
                    self.config.workflow_timeout_seconds,
                    time.monotonic,
                ),
                budget=StepBudget(max_steps=len(ALLIANCE_TECHNOLOGY_DONATION_STATES) + 24),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"alliance-technology-donation:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"alliance_technology_donation_run_id": token},
            )
            result = self._engine().execute(self._definition(), context)
            self._record_engine_failure(result, state)
            self._record_recovery_for_terminal(result, state)
            self._augment_result(result, state)
            self._update_persisted_run(result, state)
            return result
        finally:
            self._states.pop(token, None)

    def _engine(self) -> WorkflowEngine:
        registry = ActionRegistry()
        for state in ALLIANCE_TECHNOLOGY_DONATION_STATES:
            registry.register(f"alliance_technology_donation.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.precondition_retry_limit,
            "ensure_character": self.config.precondition_retry_limit,
            "ensure_game_running": self.config.precondition_retry_limit,
            "open_alliance": self.config.navigation_retry_limit,
            "open_alliance_technology": self.config.navigation_retry_limit,
            "select_technology": self.config.scan_retry_limit,
            "scan_donation_state": self.config.scan_retry_limit,
            "donate": self.config.action_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=ALLIANCE_TECHNOLOGY_DONATION_WORKFLOW_KEY,
            name="Donate Alliance Technology",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"alliance_technology_donation.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in ALLIANCE_TECHNOLOGY_DONATION_STATES
            ],
        )

    def _handler_for(self, state_name: str):
        def handler(
            context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            state = self._state_from_context(context)
            if state_name == "failed":
                return self._failed(step, state)
            if state_name == "cancelled":
                return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
            if state.stopped and state_name not in {"recover", "complete", "skipped"}:
                return _step_result(
                    step.step_key,
                    WorkflowOutcome.SKIPPED,
                    data={"skipped_after_terminal_state": state.terminal_state},
                )
            if state_name == "complete":
                return self._complete(step, state)
            if state_name == "skipped":
                return self._skipped(step, state)
            if state_name == "recover":
                return self._recover(step, state, context)
            method = getattr(self, f"_{state_name}")
            return method(step, state, context)

        return handler

    def _validate_input(
        self,
        step: WorkflowStepSpec,
        state: _AllianceTechnologyDonationState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        request = state.request
        if request.instance_id <= 0:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, "instance_id must be positive.")
        if request.instance_index < 0:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, "instance_index must be zero or greater.")
        if request.character_id <= 0:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, "character_id must be positive.")
        try:
            state.policy = request.policy.normalized()
        except ValueError as exc:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, str(exc))
        if state.policy.effective_run_budget <= 0:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                "Alliance technology donation budget is already exhausted.",
                data={"skipped_reason": "budget_exhausted", "policy": state.policy.to_json()},
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "policy": state.policy.to_json(),
                "template_keys": list(ALLIANCE_TECHNOLOGY_DONATION_TEMPLATE_KEYS),
            },
        )

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _AllianceTechnologyDonationState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        character = self.characters.get(state.request.character_id)
        if character is None:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, "Target character was not found.")
        if not character.enabled:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, "Target character is disabled.")
        if character.instance_id is not None and character.instance_id != state.request.instance_id:
            return state.stop(
                step.step_key,
                WorkflowOutcome.VALIDATION_FAILURE,
                "Target character is not assigned to the requested instance.",
                data={
                    "character_instance_id": character.instance_id,
                    "request_instance_id": state.request.instance_id,
                },
            )
        state.character = character
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "character_id": character.id,
                "character_name": character.name,
                "game_account_id": character.game_account_id,
            },
        )

    def _ensure_account(
        self,
        step: WorkflowStepSpec,
        state: _AllianceTechnologyDonationState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.account_precondition is None:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"account_precondition": "not_configured"})
        return self._action_to_step(
            step,
            state,
            self.account_precondition.ensure_account(state.request, _require_character(state)),
        )

    def _ensure_character(
        self,
        step: WorkflowStepSpec,
        state: _AllianceTechnologyDonationState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.character_precondition is None:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"character_precondition": "not_configured"})
        return self._action_to_step(
            step,
            state,
            self.character_precondition.ensure_character(state.request, _require_character(state)),
        )

    def _ensure_game_running(
        self,
        step: WorkflowStepSpec,
        state: _AllianceTechnologyDonationState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.recovery_watchdog is None:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"watchdog": "not_configured"})
        result = self.recovery_watchdog.monitor(
            instance_id=state.request.instance_id,
            instance_index=state.request.instance_index,
            instance_name=state.request.instance_name,
        )
        healthy = bool(getattr(result, "healthy", False))
        if not healthy:
            observation = getattr(result, "observation", None)
            message = str(getattr(observation, "message", "") or "Game is not in a healthy running state.")
            screenshot_path = str(getattr(observation, "screenshot_path", "") or "")
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                message,
                screenshot_path=screenshot_path,
                data={"watchdog_healthy": False},
            )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"watchdog_healthy": True})

    def _open_alliance(
        self,
        step: WorkflowStepSpec,
        state: _AllianceTechnologyDonationState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.open_alliance(state.request, _require_character(state), _require_policy(state)),
        )

    def _open_alliance_technology(
        self,
        step: WorkflowStepSpec,
        state: _AllianceTechnologyDonationState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.open_alliance_technology(state.request, _require_character(state), _require_policy(state)),
        )

    def _select_technology(
        self,
        step: WorkflowStepSpec,
        state: _AllianceTechnologyDonationState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        scan = self.driver.scan_technologies(state.request, _require_character(state), _require_policy(state))
        if scan.screenshot_path:
            state.screenshot_path = scan.screenshot_path
        state.technology_scan = scan.to_json()
        status = scan.normalized_status()
        if status == AllianceTechnologyScanStatus.VERIFICATION_REQUIRED:
            return state.stop(
                step.step_key,
                WorkflowOutcome.FATAL_FAILURE,
                scan.message or "Verification screen requires manual intervention.",
                screenshot_path=scan.screenshot_path,
                data={"scan": scan.to_json()},
            )
        if status == AllianceTechnologyScanStatus.UNKNOWN or not scan.scene_verified:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                scan.message or "Alliance Technology scene could not be verified.",
                screenshot_path=scan.screenshot_path,
                data={"scan": scan.to_json()},
            )
        selected = _select_target(scan.observations, _require_policy(state))
        state.ignored_technologies = _ignored_technologies(scan.observations, _require_policy(state), selected)
        if selected is None:
            reason = _missing_target_reason(_require_policy(state))
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                reason,
                screenshot_path=scan.screenshot_path,
                data={"scan": scan.to_json(), "ignored_technologies": state.ignored_technologies},
            )
        select_result = self.driver.select_technology(state.request, _require_character(state), selected, _require_policy(state))
        if select_result.screenshot_path:
            state.screenshot_path = select_result.screenshot_path
        if not select_result.success:
            return self._action_to_step(step, state, select_result)
        state.selected_technology = selected
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "selected_technology": selected.to_json(),
                "scan": scan.to_json(),
                "ignored_technologies": state.ignored_technologies,
                **select_result.data,
            },
            screenshot_path=select_result.screenshot_path or scan.screenshot_path,
        )

    def _scan_donation_state(
        self,
        step: WorkflowStepSpec,
        state: _AllianceTechnologyDonationState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        donation_state = self.driver.scan_donation_state(
            state.request,
            _require_character(state),
            _require_selected_technology(state),
            _require_policy(state),
        )
        if donation_state.screenshot_path:
            state.screenshot_path = donation_state.screenshot_path
        state.donation_state = donation_state
        state.scanned_donation_states.append(donation_state.to_json())
        status = donation_state.normalized_status()
        if status == AllianceDonationStateStatus.VERIFICATION_REQUIRED:
            return state.stop(
                step.step_key,
                WorkflowOutcome.FATAL_FAILURE,
                donation_state.message or "Verification screen requires manual intervention.",
                screenshot_path=donation_state.screenshot_path,
                data={"donation_state": donation_state.to_json()},
            )
        if status == AllianceDonationStateStatus.UNKNOWN:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                donation_state.message or "Alliance Technology donation state could not be determined.",
                screenshot_path=donation_state.screenshot_path,
                data={"donation_state": donation_state.to_json()},
            )
        if not donation_state.scene_verified or not donation_state.technology_verified:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                "Alliance Technology scene and selected technology must be verified before donating.",
                screenshot_path=donation_state.screenshot_path,
                data={"donation_state": donation_state.to_json()},
            )
        if status in {
            AllianceDonationStateStatus.NO_DONATION_AVAILABLE,
            AllianceDonationStateStatus.CONTRIBUTION_LIMIT_REACHED,
            AllianceDonationStateStatus.INSUFFICIENT_RESOURCES,
        }:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                donation_state.message or _skip_message_for_status(status),
                screenshot_path=donation_state.screenshot_path,
                data={"donation_state": donation_state.to_json(), "skipped_reason": status.value.lower()},
            )
        safe_action = _select_safe_action(donation_state.available_actions, _require_policy(state))
        if safe_action is None:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                "No safe alliance technology donation action is available.",
                screenshot_path=donation_state.screenshot_path,
                data={"donation_state": donation_state.to_json(), "skipped_reason": "no_safe_donation"},
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"donation_state": donation_state.to_json(), "selected_action": safe_action.to_json()},
            screenshot_path=donation_state.screenshot_path,
        )

    def _donate(
        self,
        step: WorkflowStepSpec,
        state: _AllianceTechnologyDonationState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        policy = _require_policy(state)
        character = _require_character(state)
        technology = _require_selected_technology(state)
        while state.donation_count < policy.effective_run_budget:
            context.cancellation_token.throw_if_cancelled()
            before = _require_donation_state(state)
            action = _select_safe_action(before.available_actions, policy)
            if action is None:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.SKIPPED,
                    "No safe alliance technology donation action is available.",
                    screenshot_path=before.screenshot_path,
                    data={"donation_state": before.to_json(), "skipped_reason": "no_safe_donation"},
                )
            attempt = self.driver.donate_to_technology(state.request, character, technology, action, policy)
            if attempt.screenshot_path:
                state.screenshot_path = attempt.screenshot_path
            self._record_donation_attempt(state, action, attempt)
            confirmation_step = self._handle_confirmation_action(step, state, action, attempt)
            if confirmation_step.outcome != WorkflowOutcome.SUCCESS:
                return confirmation_step
            if not attempt.success:
                return self._donation_failure(step, state, attempt, "Alliance technology donation action failed.")
            verify_step = self._verify_donation_state_action(step, state, before, attempt)
            if verify_step.outcome != WorkflowOutcome.SUCCESS:
                return verify_step
            if state.donation_count >= policy.effective_run_budget:
                break
            next_state = self.driver.scan_donation_state(state.request, character, technology, policy)
            if next_state.screenshot_path:
                state.screenshot_path = next_state.screenshot_path
            state.donation_state = next_state
            state.scanned_donation_states.append(next_state.to_json())
            if next_state.normalized_status() != AllianceDonationStateStatus.READY:
                break
            if not next_state.scene_verified or not next_state.technology_verified:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    "Alliance Technology scene and selected technology must be verified before donating.",
                    screenshot_path=next_state.screenshot_path,
                    data={"donation_state": next_state.to_json()},
                )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data=self._donation_payload(state),
            screenshot_path=state.screenshot_path,
        )

    def _handle_confirmation(
        self,
        step: WorkflowStepSpec,
        _state: _AllianceTechnologyDonationState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"handled_during": "donate"})

    def _verify_donation_state(
        self,
        step: WorkflowStepSpec,
        _state: _AllianceTechnologyDonationState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"handled_during": "donate"})

    def _handle_confirmation_action(
        self,
        step: WorkflowStepSpec,
        state: _AllianceTechnologyDonationState,
        action: AllianceDonationAction,
        attempt: AllianceDonationAttemptResult,
    ) -> WorkflowStepResult:
        confirmation = attempt.normalized_confirmation()
        if confirmation == AllianceDonationConfirmation.NONE:
            state.confirmation_attempts.append(
                {
                    "action": action.to_json(),
                    "confirmation": confirmation.value,
                    "handled": False,
                }
            )
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"confirmation": confirmation.value})
        if confirmation == AllianceDonationConfirmation.PREMIUM_CURRENCY and not _require_policy(state).allow_premium_currency:
            state.confirmation_attempts.append(
                {
                    "action": action.to_json(),
                    "confirmation": confirmation.value,
                    "handled": False,
                    "blocked_reason": "premium_currency_not_allowed",
                }
            )
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                "Premium currency confirmation is not allowed for alliance technology donation.",
                screenshot_path=attempt.screenshot_path,
                data={"attempt": attempt.to_json(), "action": action.to_json()},
            )
        if confirmation == AllianceDonationConfirmation.UNKNOWN:
            state.confirmation_attempts.append(
                {
                    "action": action.to_json(),
                    "confirmation": confirmation.value,
                    "handled": False,
                    "blocked_reason": "unknown_confirmation",
                }
            )
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                "Unknown alliance technology donation confirmation cannot be handled safely.",
                screenshot_path=attempt.screenshot_path,
                data={"attempt": attempt.to_json(), "action": action.to_json()},
            )
        handled = self.driver.handle_donation_confirmation(
            state.request,
            _require_character(state),
            _require_selected_technology(state),
            action,
            attempt,
            _require_policy(state),
        )
        if handled.screenshot_path:
            state.screenshot_path = handled.screenshot_path
        state.confirmation_attempts.append(
            {
                "action": action.to_json(),
                "confirmation": confirmation.value,
                "handled": handled.success,
                **handled.to_json(),
            }
        )
        if not handled.success:
            return self._donation_failure(step, state, handled, "Alliance technology donation confirmation failed.")
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=handled.to_json(), screenshot_path=handled.screenshot_path)

    def _verify_donation_state_action(
        self,
        step: WorkflowStepSpec,
        state: _AllianceTechnologyDonationState,
        before: AllianceDonationState,
        attempt: AllianceDonationAttemptResult,
    ) -> WorkflowStepResult:
        verify = self.driver.verify_donation_state(
            state.request,
            _require_character(state),
            _require_selected_technology(state),
            before,
            attempt,
            _require_policy(state),
        )
        if verify.screenshot_path:
            state.screenshot_path = verify.screenshot_path
        state.verification_attempts.append(
            {
                "technology_key": _require_selected_technology(state).normalized_technology_key(),
                "before_contribution_count": before.contribution_count,
                **verify.to_json(),
            }
        )
        if not _donation_postcondition_verified(before, attempt, verify):
            failure = replace(
                verify,
                retryable=False,
                message=("Alliance technology donation state did not change after donation. " f"{verify.message}").strip(),
            )
            return self._donation_failure(
                step,
                state,
                failure,
                "Alliance technology donation postcondition was not verified.",
            )
        state.donation_count += max(1, verify.donation_count, attempt.donation_count)
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=verify.to_json(), screenshot_path=verify.screenshot_path)

    def _complete(self, step: WorkflowStepSpec, state: _AllianceTechnologyDonationState) -> WorkflowStepResult:
        if state.stopped:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                data={
                    "terminal_outcome": state.terminal_outcome.value if state.terminal_outcome else "",
                    "terminal_state": state.terminal_state,
                    "terminal_reason": state.terminal_reason,
                },
            )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=self._donation_payload(state))

    def _skipped(self, step: WorkflowStepSpec, state: _AllianceTechnologyDonationState) -> WorkflowStepResult:
        if state.terminal_outcome == WorkflowOutcome.SKIPPED:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data={"skipped_reason": state.terminal_reason, **self._donation_payload(state)},
                screenshot_path=state.screenshot_path,
            )
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED)

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _AllianceTechnologyDonationState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if _is_manual_stop(state):
            state.recovery_outcome = {"attempted": False, "reason": "manual_intervention_required"}
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
        state.recovery_outcome = self._monitor_recovery(state, _job_run_id(context))
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _AllianceTechnologyDonationState) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        self._open_incident(state)
        return _step_result(
            step.step_key,
            WorkflowOutcome.FATAL_FAILURE,
            state.terminal_reason,
            data={
                "terminal_state": state.terminal_state,
                "terminal_reason": state.terminal_reason,
                **self._donation_payload(state),
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _AllianceTechnologyDonationState,
        action: ResourceGatheringActionResult,
    ) -> WorkflowStepResult:
        if action.screenshot_path:
            state.screenshot_path = action.screenshot_path
        if action.success:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=action.data, screenshot_path=action.screenshot_path)
        if action.retryable:
            return _step_result(
                step.step_key,
                WorkflowOutcome.RETRYABLE_FAILURE,
                action.message or "Alliance technology donation action failed.",
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            action.message or "Alliance technology donation action failed.",
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _donation_failure(
        self,
        step: WorkflowStepSpec,
        state: _AllianceTechnologyDonationState,
        result: AllianceDonationAttemptResult,
        fallback_message: str,
    ) -> WorkflowStepResult:
        if result.retryable:
            return _step_result(
                step.step_key,
                WorkflowOutcome.RETRYABLE_FAILURE,
                result.message or fallback_message,
                data=result.to_json(),
                screenshot_path=result.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            result.message or fallback_message,
            screenshot_path=result.screenshot_path,
            data=result.to_json(),
        )

    def _record_donation_attempt(
        self,
        state: _AllianceTechnologyDonationState,
        action: AllianceDonationAction,
        result: AllianceDonationAttemptResult,
    ) -> None:
        state.donation_attempts.append(
            {
                "technology_key": _require_selected_technology(state).normalized_technology_key(),
                "action": action.to_json(),
                **result.to_json(),
            }
        )

    def _donation_payload(self, state: _AllianceTechnologyDonationState) -> dict[str, object]:
        return {
            "donation_count": state.donation_count,
            "selected_technology": (
                state.selected_technology.to_json() if state.selected_technology is not None else {}
            ),
            "technology_scan": state.technology_scan,
            "donation_states": state.scanned_donation_states,
            "donation_attempts": state.donation_attempts,
            "confirmation_attempts": state.confirmation_attempts,
            "verification_attempts": state.verification_attempts,
            "ignored_technologies": state.ignored_technologies,
            "verification_result": state.verification_attempts[-1] if state.verification_attempts else {},
            "failure_evidence": {
                "screenshot_path": state.screenshot_path,
                "terminal_state": state.terminal_state,
                "terminal_reason": state.terminal_reason,
            },
        }

    def _state_from_context(self, context: WorkflowExecutionContext) -> _AllianceTechnologyDonationState:
        token = str(context.metadata.get("alliance_technology_donation_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("Alliance technology donation runtime state is missing.") from exc

    def _open_incident(self, state: _AllianceTechnologyDonationState) -> None:
        if self.incidents is None or not state.failed or state.incident_opened:
            return
        self.incidents.save(
            Incident(
                incident_key=f"alliance-technology-donation:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="Alliance technology donation blocked",
                details=state.terminal_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )
        state.incident_opened = True

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _AllianceTechnologyDonationState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "Alliance technology donation workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(
        self,
        result: WorkflowExecutionResult,
        state: _AllianceTechnologyDonationState,
    ) -> None:
        if state.failed and not state.recovery_outcome:
            if _is_manual_stop(state):
                state.recovery_outcome = {"attempted": False, "reason": "manual_intervention_required"}
            else:
                state.recovery_outcome = self._monitor_recovery(state, result.job_run_id)
        if state.failed:
            self._open_incident(state)

    def _monitor_recovery(
        self,
        state: _AllianceTechnologyDonationState,
        job_run_id: int | None,
    ) -> dict[str, object]:
        if self.recovery_watchdog is None:
            return {"attempted": False, "reason": "watchdog_not_configured"}
        result = self.recovery_watchdog.monitor(
            instance_id=state.request.instance_id,
            instance_index=state.request.instance_index,
            instance_name=state.request.instance_name,
            job_run_id=job_run_id,
        )
        return {
            "attempted": bool(getattr(result, "recovery_attempted", False)),
            "healthy": bool(getattr(result, "healthy", False)),
            "circuit_opened": bool(getattr(result, "circuit_opened", False)),
        }

    def _augment_result(
        self,
        result: WorkflowExecutionResult,
        state: _AllianceTechnologyDonationState,
    ) -> None:
        if state.terminal_outcome in {WorkflowOutcome.SKIPPED, WorkflowOutcome.BLOCKED}:
            result.outcome = state.terminal_outcome
            result.message = state.terminal_reason
        result.result = {
            **dict(result.result),
            "policy": state.policy.to_json() if state.policy is not None else {},
            **self._donation_payload(state),
            "terminal_state": state.terminal_state,
            "terminal_reason": state.terminal_reason,
            "skipped_reason": state.terminal_reason if result.outcome == WorkflowOutcome.SKIPPED else "",
            "failure_state": state.terminal_state if result.outcome.is_failure else "",
            "failure_reason": state.terminal_reason if result.outcome.is_failure else "",
            "recovery_outcome": state.recovery_outcome,
        }

    def _update_persisted_run(
        self,
        result: WorkflowExecutionResult,
        state: _AllianceTechnologyDonationState,
    ) -> None:
        if self.job_runs is None or result.job_run_id is None:
            return
        run = self.job_runs.get(result.job_run_id)
        if run is None:
            return
        run.status = "completed" if result.outcome == WorkflowOutcome.SKIPPED else ("failed" if result.outcome.is_failure else "completed")
        run.result_json = json.dumps(result.to_json_dict(), sort_keys=True)
        run.error_message = state.terminal_reason if result.outcome.is_failure else ""
        run.screenshot_path = state.screenshot_path
        self.job_runs.save(run)


def _select_target(
    observations: tuple[AllianceTechnologyObservation, ...],
    policy: AllianceTechnologyDonationPolicy,
) -> AllianceTechnologyObservation | None:
    candidates = [
        item
        for item in observations
        if item.confidence >= policy.minimum_detector_confidence
        and (not policy.target_technology_key or item.normalized_technology_key() == policy.target_technology_key)
        and (not policy.recommended_only or item.recommended)
    ]
    if not candidates:
        return None
    selected_items = [item for item in candidates if item.selected]
    return (selected_items or candidates)[0]


def _ignored_technologies(
    observations: tuple[AllianceTechnologyObservation, ...],
    policy: AllianceTechnologyDonationPolicy,
    selected: AllianceTechnologyObservation | None,
) -> list[dict[str, object]]:
    selected_key = selected.normalized_technology_key() if selected is not None else ""
    ignored: list[dict[str, object]] = []
    for item in observations:
        reason = ""
        key = item.normalized_technology_key()
        if key == selected_key:
            continue
        if item.confidence < policy.minimum_detector_confidence:
            reason = "below_confidence_threshold"
        elif policy.target_technology_key and key != policy.target_technology_key:
            reason = "not_configured_target"
        elif policy.recommended_only and not item.recommended:
            reason = "not_recommended"
        if reason:
            ignored.append({**item.to_json(), "ignored_reason": reason})
    return ignored


def _select_safe_action(
    actions: tuple[AllianceDonationAction, ...],
    policy: AllianceTechnologyDonationPolicy,
) -> AllianceDonationAction | None:
    for action in actions:
        if _unsafe_action_reason(action, policy):
            continue
        return action
    return None


def _unsafe_action_reason(
    action: AllianceDonationAction,
    policy: AllianceTechnologyDonationPolicy,
) -> str:
    cost_type = action.normalized_cost_type()
    if not action.enabled:
        return "disabled"
    if action.confidence < policy.minimum_detector_confidence:
        return "below_confidence_threshold"
    if action.premium or cost_type == AllianceDonationCostType.GEMS:
        if not policy.allow_premium_currency:
            return "premium_currency_not_allowed"
    if cost_type not in policy.allowed_cost_types:
        return "cost_type_not_allowed"
    if (
        policy.max_resource_cost_per_donation is not None
        and cost_type != AllianceDonationCostType.FREE
        and action.cost_amount > policy.max_resource_cost_per_donation
    ):
        return "resource_cost_over_budget"
    return ""


def _donation_postcondition_verified(
    before: AllianceDonationState,
    attempt: AllianceDonationAttemptResult,
    verify: AllianceDonationAttemptResult,
) -> bool:
    if not verify.success or not verify.changed:
        return False
    if verify.donation_count > 0 or attempt.donation_count > 0:
        return True
    if before.contribution_count is None or verify.contribution_count is None:
        return False
    return verify.contribution_count > before.contribution_count


def _missing_target_reason(policy: AllianceTechnologyDonationPolicy) -> str:
    if policy.target_technology_key:
        return f"Configured alliance technology {policy.target_technology_key} was not found."
    return "No recommended alliance technology donation target was found."


def _skip_message_for_status(status: AllianceDonationStateStatus) -> str:
    if status == AllianceDonationStateStatus.CONTRIBUTION_LIMIT_REACHED:
        return "Alliance technology contribution limit is reached."
    if status == AllianceDonationStateStatus.INSUFFICIENT_RESOURCES:
        return "Insufficient resources for alliance technology donation."
    return "No alliance technology donation is available."


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
        action_type=f"alliance_technology_donation.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _AllianceTechnologyDonationState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _AllianceTechnologyDonationState) -> AllianceTechnologyDonationPolicy:
    if state.policy is None:
        raise RuntimeError("Alliance technology donation policy has not been validated.")
    return state.policy


def _require_selected_technology(state: _AllianceTechnologyDonationState) -> AllianceTechnologyObservation:
    if state.selected_technology is None:
        raise RuntimeError("Alliance technology target has not been selected.")
    return state.selected_technology


def _require_donation_state(state: _AllianceTechnologyDonationState) -> AllianceDonationState:
    if state.donation_state is None:
        raise RuntimeError("Alliance technology donation state has not been scanned.")
    return state.donation_state


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_manual_stop(state: _AllianceTechnologyDonationState) -> bool:
    text = state.terminal_reason.lower()
    return "verification" in text or "confirmation" in text or "manual" in text
