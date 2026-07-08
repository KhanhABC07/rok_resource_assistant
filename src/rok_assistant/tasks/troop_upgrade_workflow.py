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
from rok_assistant.tasks.troop_training_workflow import (
    TroopTrainingBuilding,
    TroopType,
    troop_type_for_building,
)
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


TROOP_UPGRADE_WORKFLOW_KEY = "troop-upgrade"
TROOP_UPGRADE_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "normalize_city_view",
    "open_training_building",
    "open_upgrade_tab",
    "inspect_upgrade_state",
    "select_upgrade_tiers",
    "set_upgrade_amount",
    "start_upgrade",
    "verify_upgrade_state",
    "complete",
    "skipped",
    "recover",
    "failed",
    "cancelled",
)


class TroopUpgradeQueueStatus(StrEnum):
    IDLE = "IDLE"
    BUSY = "BUSY"
    READY = "READY"
    INSUFFICIENT_RESOURCES = "INSUFFICIENT_RESOURCES"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


class TroopUpgradeConfirmation(StrEnum):
    NONE = "NONE"
    FREE = "FREE"
    SPEEDUP = "SPEEDUP"
    GEM = "GEM"
    PREMIUM = "PREMIUM"
    RESOURCE_ITEM = "RESOURCE_ITEM"
    TRAINING = "TRAINING"
    BUILDING_UPGRADE = "BUILDING_UPGRADE"
    UNKNOWN = "UNKNOWN"


def _building(value: TroopTrainingBuilding | str) -> TroopTrainingBuilding:
    if isinstance(value, TroopTrainingBuilding):
        return value
    try:
        return TroopTrainingBuilding(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in TroopTrainingBuilding)
        raise ValueError(f"Invalid troop upgrade building: {value!r}. Expected one of: {valid}.") from exc


def _queue_status(value: TroopUpgradeQueueStatus | str) -> TroopUpgradeQueueStatus:
    if isinstance(value, TroopUpgradeQueueStatus):
        return value
    try:
        return TroopUpgradeQueueStatus(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in TroopUpgradeQueueStatus)
        raise ValueError(f"Invalid troop upgrade queue status: {value!r}. Expected one of: {valid}.") from exc


def _confirmation(value: TroopUpgradeConfirmation | str) -> TroopUpgradeConfirmation:
    if isinstance(value, TroopUpgradeConfirmation):
        return value
    try:
        return TroopUpgradeConfirmation(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in TroopUpgradeConfirmation)
        raise ValueError(f"Invalid troop upgrade confirmation: {value!r}. Expected one of: {valid}.") from exc


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
class TroopUpgradePolicy:
    enabled_buildings: tuple[TroopTrainingBuilding | str, ...] = (
        TroopTrainingBuilding.BARRACKS,
        TroopTrainingBuilding.ARCHERY_RANGE,
        TroopTrainingBuilding.STABLE,
        TroopTrainingBuilding.SIEGE_WORKSHOP,
    )
    source_tier: int = 1
    target_tier: int = 2
    upgrade_amount: int = 1
    allow_all_eligible: bool = False
    max_upgrade_amount: int = 1
    max_queue_size: int = 1
    max_resource_cost: int = 0
    skip_busy_queue: bool = True
    skip_insufficient_resources: bool = True
    allow_speedups: bool = False
    allow_gem_spending: bool = False
    allow_premium_spending: bool = False
    allow_resource_items: bool = False
    minimum_detector_confidence: float = 0.85

    def normalized(self) -> TroopUpgradePolicy:
        enabled = tuple(dict.fromkeys(_building(item) for item in self.enabled_buildings))
        if not enabled:
            raise ValueError("At least one troop upgrade building must be enabled.")
        source_tier = _require_positive_int(self.source_tier, "source_tier")
        target_tier = _require_positive_int(self.target_tier, "target_tier")
        if source_tier > 5 or target_tier > 5:
            raise ValueError("source_tier and target_tier must be between 1 and 5.")
        if target_tier <= source_tier:
            raise ValueError("target_tier must be greater than source_tier.")
        if self.allow_speedups:
            raise ValueError("Speedup usage is out of scope for TROOP-002.")
        if self.allow_gem_spending:
            raise ValueError("Gem spending is out of scope for TROOP-002.")
        if self.allow_premium_spending:
            raise ValueError("Premium currency spending is out of scope for TROOP-002.")
        if self.allow_resource_items:
            raise ValueError("Resource item usage is out of scope for TROOP-002.")
        amount = _require_positive_int(self.upgrade_amount, "upgrade_amount")
        maximum = _require_positive_int(self.max_upgrade_amount, "max_upgrade_amount")
        if amount > maximum and not self.allow_all_eligible:
            raise ValueError("upgrade_amount cannot exceed max_upgrade_amount.")
        return TroopUpgradePolicy(
            enabled_buildings=enabled,
            source_tier=source_tier,
            target_tier=target_tier,
            upgrade_amount=amount,
            allow_all_eligible=bool(self.allow_all_eligible),
            max_upgrade_amount=maximum,
            max_queue_size=_require_positive_int(self.max_queue_size, "max_queue_size"),
            max_resource_cost=_require_non_negative_int(self.max_resource_cost, "max_resource_cost"),
            skip_busy_queue=bool(self.skip_busy_queue),
            skip_insufficient_resources=bool(self.skip_insufficient_resources),
            allow_speedups=False,
            allow_gem_spending=False,
            allow_premium_spending=False,
            allow_resource_items=False,
            minimum_detector_confidence=_require_confidence(
                self.minimum_detector_confidence,
                "minimum_detector_confidence",
            ),
        )

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "enabled_buildings": [item.value for item in normalized.enabled_buildings],
            "source_tier": normalized.source_tier,
            "target_tier": normalized.target_tier,
            "upgrade_amount": normalized.upgrade_amount,
            "allow_all_eligible": normalized.allow_all_eligible,
            "max_upgrade_amount": normalized.max_upgrade_amount,
            "max_queue_size": normalized.max_queue_size,
            "max_resource_cost": normalized.max_resource_cost,
            "skip_busy_queue": normalized.skip_busy_queue,
            "skip_insufficient_resources": normalized.skip_insufficient_resources,
            "allow_speedups": normalized.allow_speedups,
            "allow_gem_spending": normalized.allow_gem_spending,
            "allow_premium_spending": normalized.allow_premium_spending,
            "allow_resource_items": normalized.allow_resource_items,
            "minimum_detector_confidence": normalized.minimum_detector_confidence,
        }


@dataclass(frozen=True)
class TroopUpgradeRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: TroopUpgradePolicy = field(default_factory=TroopUpgradePolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class TroopUpgradeConfig:
    workflow_timeout_seconds: float = 180.0
    step_timeout_seconds: float = 15.0
    precondition_retry_limit: int = 1
    navigation_retry_limit: int = 1
    inspect_retry_limit: int = 1
    action_retry_limit: int = 0
    retry_delay_seconds: float = 0.25


@dataclass(frozen=True)
class TroopUpgradeOption:
    building: TroopTrainingBuilding | str
    troop_type: TroopType | str
    source_tier: int
    target_tier: int
    eligible_count: int
    enabled: bool = True
    resources_available: bool = True
    resource_cost: int = 0
    confidence: float = 1.0
    upgrade_tab_verified: bool = True
    tiers_verified: bool = True
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_building(self) -> TroopTrainingBuilding:
        return _building(self.building)

    def normalized_troop_type(self) -> TroopType:
        value = self.troop_type
        if isinstance(value, TroopType):
            return value
        return TroopType(str(value).strip().upper())

    def to_json(self) -> dict[str, object]:
        return {
            "building": self.normalized_building().value,
            "troop_type": self.normalized_troop_type().value,
            "source_tier": self.source_tier,
            "target_tier": self.target_tier,
            "eligible_count": self.eligible_count,
            "enabled": self.enabled,
            "resources_available": self.resources_available,
            "resource_cost": self.resource_cost,
            "confidence": self.confidence,
            "upgrade_tab_verified": self.upgrade_tab_verified,
            "tiers_verified": self.tiers_verified,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class TroopUpgradeState:
    building: TroopTrainingBuilding | str
    troop_type: TroopType | str
    status: TroopUpgradeQueueStatus | str
    upgrade_tab_active: bool = True
    eligible_options: tuple[TroopUpgradeOption, ...] = ()
    queue_size: int = 0
    timer_seconds: int = 0
    scene_verified: bool = True
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_building(self) -> TroopTrainingBuilding:
        return _building(self.building)

    def normalized_troop_type(self) -> TroopType:
        value = self.troop_type
        if isinstance(value, TroopType):
            return value
        return TroopType(str(value).strip().upper())

    def normalized_status(self) -> TroopUpgradeQueueStatus:
        return _queue_status(self.status)

    def to_json(self) -> dict[str, object]:
        return {
            "building": self.normalized_building().value,
            "troop_type": self.normalized_troop_type().value,
            "status": self.normalized_status().value,
            "upgrade_tab_active": self.upgrade_tab_active,
            "eligible_options": [item.to_json() for item in self.eligible_options],
            "queue_size": self.queue_size,
            "timer_seconds": self.timer_seconds,
            "scene_verified": self.scene_verified,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class TroopUpgradeAmountPlan:
    building: TroopTrainingBuilding | str
    source_tier: int
    target_tier: int
    eligible_count: int
    selected_amount: int
    resource_cost: int = 0
    all_eligible_requested: bool = False

    def normalized_building(self) -> TroopTrainingBuilding:
        return _building(self.building)

    def to_json(self) -> dict[str, object]:
        return {
            "building": self.normalized_building().value,
            "source_tier": self.source_tier,
            "target_tier": self.target_tier,
            "eligible_count": self.eligible_count,
            "selected_amount": self.selected_amount,
            "resource_cost": self.resource_cost,
            "all_eligible_requested": self.all_eligible_requested,
        }


@dataclass(frozen=True)
class TroopUpgradeStartResult:
    success: bool
    changed: bool = False
    confirmation: TroopUpgradeConfirmation | str = TroopUpgradeConfirmation.NONE
    queue_size: int | None = None
    timer_seconds: int | None = None
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_confirmation(self) -> TroopUpgradeConfirmation:
        return _confirmation(self.confirmation)

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "changed": self.changed,
            "confirmation": self.normalized_confirmation().value,
            "queue_size": self.queue_size,
            "timer_seconds": self.timer_seconds,
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


class TroopUpgradeAccountPrecondition(Protocol):
    def ensure_account(self, request: TroopUpgradeRequest, character: Character) -> ResourceGatheringActionResult:
        ...


class TroopUpgradeCharacterPrecondition(Protocol):
    def ensure_character(self, request: TroopUpgradeRequest, character: Character) -> ResourceGatheringActionResult:
        ...


class TroopUpgradeDriver(Protocol):
    def normalize_city_view(
        self,
        request: TroopUpgradeRequest,
        character: Character,
        policy: TroopUpgradePolicy,
        building: TroopTrainingBuilding,
    ) -> ResourceGatheringActionResult:
        ...

    def open_training_building(
        self,
        request: TroopUpgradeRequest,
        character: Character,
        policy: TroopUpgradePolicy,
        building: TroopTrainingBuilding,
    ) -> ResourceGatheringActionResult:
        ...

    def open_upgrade_tab(
        self,
        request: TroopUpgradeRequest,
        character: Character,
        policy: TroopUpgradePolicy,
        building: TroopTrainingBuilding,
    ) -> ResourceGatheringActionResult:
        ...

    def inspect_upgrade_state(
        self,
        request: TroopUpgradeRequest,
        character: Character,
        policy: TroopUpgradePolicy,
        building: TroopTrainingBuilding,
    ) -> TroopUpgradeState:
        ...

    def select_upgrade_tiers(
        self,
        request: TroopUpgradeRequest,
        character: Character,
        option: TroopUpgradeOption,
        policy: TroopUpgradePolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def set_upgrade_amount(
        self,
        request: TroopUpgradeRequest,
        character: Character,
        option: TroopUpgradeOption,
        amount: TroopUpgradeAmountPlan,
        policy: TroopUpgradePolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def start_upgrade(
        self,
        request: TroopUpgradeRequest,
        character: Character,
        option: TroopUpgradeOption,
        amount: TroopUpgradeAmountPlan,
        before: TroopUpgradeState,
        policy: TroopUpgradePolicy,
    ) -> TroopUpgradeStartResult:
        ...

    def verify_upgrade_state(
        self,
        request: TroopUpgradeRequest,
        character: Character,
        option: TroopUpgradeOption,
        amount: TroopUpgradeAmountPlan,
        before: TroopUpgradeState,
        start: TroopUpgradeStartResult,
        policy: TroopUpgradePolicy,
    ) -> TroopUpgradeStartResult:
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
class _TroopUpgradeRuntimeState:
    request: TroopUpgradeRequest
    character: Character | None = None
    policy: TroopUpgradePolicy | None = None
    selected_building: TroopTrainingBuilding | None = None
    upgrade_state: TroopUpgradeState | None = None
    selected_option: TroopUpgradeOption | None = None
    selected_amount: TroopUpgradeAmountPlan | None = None
    start_result: TroopUpgradeStartResult | None = None
    enabled_buildings: list[TroopTrainingBuilding] = field(default_factory=list)
    active_buildings: list[TroopTrainingBuilding] = field(default_factory=list)
    upgrade_states: dict[TroopTrainingBuilding, TroopUpgradeState] = field(default_factory=dict)
    selected_options: dict[TroopTrainingBuilding, TroopUpgradeOption] = field(default_factory=dict)
    selected_amounts: dict[TroopTrainingBuilding, TroopUpgradeAmountPlan] = field(default_factory=dict)
    start_results: dict[TroopTrainingBuilding, TroopUpgradeStartResult] = field(default_factory=dict)
    upgraded_buildings: list[TroopTrainingBuilding] = field(default_factory=list)
    skipped_buildings: list[dict[str, object]] = field(default_factory=list)
    normalized_attempts: list[dict[str, object]] = field(default_factory=list)
    open_attempts: list[dict[str, object]] = field(default_factory=list)
    tab_attempts: list[dict[str, object]] = field(default_factory=list)
    inspections: list[dict[str, object]] = field(default_factory=list)
    tier_selection_attempts: list[dict[str, object]] = field(default_factory=list)
    amount_attempts: list[dict[str, object]] = field(default_factory=list)
    upgrade_attempts: list[dict[str, object]] = field(default_factory=list)
    verification_attempts: list[dict[str, object]] = field(default_factory=list)
    ignored_options: list[dict[str, object]] = field(default_factory=list)
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


class TroopUpgradeWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: TroopUpgradeDriver,
        account_precondition: TroopUpgradeAccountPrecondition | None = None,
        character_precondition: TroopUpgradeCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: TroopUpgradeConfig | None = None,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.account_precondition = account_precondition
        self.character_precondition = character_precondition
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.incidents = incidents
        self.config = config or TroopUpgradeConfig()
        self._states: dict[str, _TroopUpgradeRuntimeState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return TROOP_UPGRADE_STATES

    def execute(
        self,
        request: TroopUpgradeRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _TroopUpgradeRuntimeState(request=request)
        self._states[token] = state
        persistence = None
        if self.job_runs is not None and self.step_runs is not None and request.job_id is not None:
            persistence = WorkflowRunRepositoryRecorder(self.job_runs, self.step_runs)
        try:
            context = WorkflowExecutionContext(
                cancellation_token=cancellation_token or CancellationToken(),
                deadline=WorkflowDeadline.from_timeout(self.config.workflow_timeout_seconds, time.monotonic),
                budget=StepBudget(max_steps=len(TROOP_UPGRADE_STATES) + 20),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"troop-upgrade:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"troop_upgrade_run_id": token},
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
        for state in TROOP_UPGRADE_STATES:
            registry.register(f"troop_upgrade.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.precondition_retry_limit,
            "ensure_character": self.config.precondition_retry_limit,
            "ensure_game_running": self.config.precondition_retry_limit,
            "normalize_city_view": self.config.navigation_retry_limit,
            "open_training_building": self.config.navigation_retry_limit,
            "open_upgrade_tab": self.config.navigation_retry_limit,
            "inspect_upgrade_state": self.config.inspect_retry_limit,
            "select_upgrade_tiers": self.config.inspect_retry_limit,
            "set_upgrade_amount": self.config.action_retry_limit,
            "start_upgrade": self.config.action_retry_limit,
            "verify_upgrade_state": self.config.inspect_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=TROOP_UPGRADE_WORKFLOW_KEY,
            name="Upgrade Troops",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"troop_upgrade.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in TROOP_UPGRADE_STATES
            ],
        )

    def _handler_for(self, state_name: str):
        def handler(context: WorkflowExecutionContext, step: WorkflowStepSpec) -> WorkflowStepResult:
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
        state: _TroopUpgradeRuntimeState,
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
        state.enabled_buildings = list(state.policy.enabled_buildings)
        state.active_buildings = list(state.policy.enabled_buildings)
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"policy": state.policy.to_json()})

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _TroopUpgradeRuntimeState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        character = self.characters.get(state.request.character_id)
        if character is None:
            return state.stop(
                step.step_key,
                WorkflowOutcome.VALIDATION_FAILURE,
                f"Character {state.request.character_id} was not found.",
            )
        state.character = character
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"character_id": character.id})

    def _ensure_account(
        self,
        step: WorkflowStepSpec,
        state: _TroopUpgradeRuntimeState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.account_precondition is None:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"skipped": True})
        return self._action_to_step(step, state, self.account_precondition.ensure_account(state.request, _require_character(state)))

    def _ensure_character(
        self,
        step: WorkflowStepSpec,
        state: _TroopUpgradeRuntimeState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.character_precondition is None:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"skipped": True})
        return self._action_to_step(step, state, self.character_precondition.ensure_character(state.request, _require_character(state)))

    def _ensure_game_running(
        self,
        step: WorkflowStepSpec,
        _state: _TroopUpgradeRuntimeState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"precondition": "delegated_to_driver"})

    def _normalize_city_view(
        self,
        step: WorkflowStepSpec,
        state: _TroopUpgradeRuntimeState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        for building in state.active_buildings:
            state.selected_building = building
            result = self.driver.normalize_city_view(state.request, _require_character(state), _require_policy(state), building)
            state.normalized_attempts.append({"building": building.value, **result.data, "screenshot_path": result.screenshot_path})
            step_result = self._action_to_step(step, state, result)
            if step_result.outcome != WorkflowOutcome.SUCCESS:
                return step_result
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"normalization_attempts": state.normalized_attempts})

    def _open_training_building(
        self,
        step: WorkflowStepSpec,
        state: _TroopUpgradeRuntimeState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        for building in state.active_buildings:
            state.selected_building = building
            result = self.driver.open_training_building(state.request, _require_character(state), _require_policy(state), building)
            state.open_attempts.append({"building": building.value, **result.data, "screenshot_path": result.screenshot_path})
            step_result = self._action_to_step(step, state, result, hard_stop_message="Training building scene could not be verified.")
            if step_result.outcome != WorkflowOutcome.SUCCESS:
                return step_result
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"open_attempts": state.open_attempts})

    def _open_upgrade_tab(
        self,
        step: WorkflowStepSpec,
        state: _TroopUpgradeRuntimeState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        for building in state.active_buildings:
            state.selected_building = building
            result = self.driver.open_upgrade_tab(state.request, _require_character(state), _require_policy(state), building)
            state.tab_attempts.append({"building": building.value, **result.data, "screenshot_path": result.screenshot_path})
            step_result = self._action_to_step(step, state, result, hard_stop_message="Troop upgrade tab could not be opened.")
            if step_result.outcome != WorkflowOutcome.SUCCESS:
                return step_result
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"tab_attempts": state.tab_attempts})

    def _inspect_upgrade_state(
        self,
        step: WorkflowStepSpec,
        state: _TroopUpgradeRuntimeState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        policy = _require_policy(state)
        active: list[TroopTrainingBuilding] = []
        for building in state.active_buildings:
            state.selected_building = building
            upgrade_state = self.driver.inspect_upgrade_state(state.request, _require_character(state), policy, building)
            state.upgrade_state = upgrade_state
            state.upgrade_states[building] = upgrade_state
            if upgrade_state.screenshot_path:
                state.screenshot_path = upgrade_state.screenshot_path
            state.inspections.append(upgrade_state.to_json())
            status = upgrade_state.normalized_status()
            if (
                not upgrade_state.scene_verified
                or not upgrade_state.upgrade_tab_active
                or status in {TroopUpgradeQueueStatus.VERIFICATION_REQUIRED, TroopUpgradeQueueStatus.UNKNOWN}
            ):
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    upgrade_state.message or "Troop upgrade tab or queue state could not be verified.",
                    screenshot_path=upgrade_state.screenshot_path,
                    data={"upgrade_state": upgrade_state.to_json()},
                )
            if status in {TroopUpgradeQueueStatus.BUSY, TroopUpgradeQueueStatus.READY}:
                if not policy.skip_busy_queue:
                    return state.stop(
                        step.step_key,
                        WorkflowOutcome.BLOCKED,
                        upgrade_state.message or "Troop upgrade queue is already busy.",
                        screenshot_path=upgrade_state.screenshot_path,
                        data={"upgrade_state": upgrade_state.to_json()},
                    )
                state.skipped_buildings.append({"building": building.value, "reason": "busy_queue", "queue_state": upgrade_state.to_json()})
                continue
            if status == TroopUpgradeQueueStatus.INSUFFICIENT_RESOURCES and not upgrade_state.eligible_options:
                if not policy.skip_insufficient_resources:
                    return state.stop(
                        step.step_key,
                        WorkflowOutcome.BLOCKED,
                        upgrade_state.message or "Insufficient resources for troop upgrade.",
                        screenshot_path=upgrade_state.screenshot_path,
                        data={"upgrade_state": upgrade_state.to_json()},
                    )
                state.skipped_buildings.append({"building": building.value, "reason": "insufficient_resources", "queue_state": upgrade_state.to_json()})
                continue
            active.append(building)
        state.active_buildings = active
        if not active:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                "No enabled troop upgrade queue is available.",
                screenshot_path=state.screenshot_path,
                data={"upgrade_inspections": state.inspections, "skipped_buildings": state.skipped_buildings},
            )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"upgrade_inspections": state.inspections})

    def _select_upgrade_tiers(
        self,
        step: WorkflowStepSpec,
        state: _TroopUpgradeRuntimeState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        policy = _require_policy(state)
        still_active: list[TroopTrainingBuilding] = []
        for building in state.active_buildings:
            upgrade_state = _require_upgrade_state(state, building)
            selected, ignored = _select_upgrade_option(upgrade_state, policy)
            state.ignored_options.extend(ignored)
            if selected is None:
                insufficient = any(item.get("ignored_reason") == "insufficient_resources" for item in ignored)
                if insufficient and policy.skip_insufficient_resources:
                    state.skipped_buildings.append({"building": building.value, "reason": "insufficient_resources", "ignored_options": ignored})
                    continue
                no_eligible = not upgrade_state.eligible_options or all(
                    item.get("ignored_reason") == "no_eligible_units" for item in ignored
                )
                if no_eligible:
                    state.skipped_buildings.append({"building": building.value, "reason": "no_eligible_units", "ignored_options": ignored})
                    continue
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    "Insufficient resources for configured troop upgrade." if insufficient else "Configured troop upgrade tiers are not available.",
                    screenshot_path=state.screenshot_path,
                    data={"upgrade_state": upgrade_state.to_json(), "ignored_options": ignored},
                )
            if not selected.upgrade_tab_verified or not selected.tiers_verified:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    selected.message or "Selected source and target tiers could not be verified before upgrade.",
                    screenshot_path=selected.screenshot_path or state.screenshot_path,
                    data={"selected_upgrade": selected.to_json(), "ignored_options": ignored},
                )
            action = self.driver.select_upgrade_tiers(state.request, _require_character(state), selected, policy)
            state.tier_selection_attempts.append({"selected_upgrade": selected.to_json(), **action.data})
            if action.screenshot_path:
                state.screenshot_path = action.screenshot_path
            if not action.success:
                return self._action_to_step(step, state, action)
            state.selected_option = selected
            state.selected_options[building] = selected
            still_active.append(building)
        state.active_buildings = still_active
        if not still_active:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                "No enabled troop upgrade building has eligible configured units.",
                screenshot_path=state.screenshot_path,
                data={"ignored_options": state.ignored_options, "skipped_buildings": state.skipped_buildings},
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"selected_upgrades": [item.to_json() for item in state.selected_options.values()], "ignored_options": state.ignored_options},
        )

    def _set_upgrade_amount(
        self,
        step: WorkflowStepSpec,
        state: _TroopUpgradeRuntimeState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        for building in state.active_buildings:
            option = _require_selected_option(state, building)
            amount = _amount_for_option(option, _require_policy(state))
            action = self.driver.set_upgrade_amount(state.request, _require_character(state), option, amount, _require_policy(state))
            state.amount_attempts.append({"selected_upgrade": option.to_json(), "amount": amount.to_json(), **action.data})
            if action.screenshot_path:
                state.screenshot_path = action.screenshot_path
            if not action.success:
                return self._action_to_step(step, state, action, hard_stop_message="Troop upgrade amount could not be set safely.")
            state.selected_amount = amount
            state.selected_amounts[building] = amount
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"amount_attempts": state.amount_attempts})

    def _start_upgrade(
        self,
        step: WorkflowStepSpec,
        state: _TroopUpgradeRuntimeState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        for building in state.active_buildings:
            option = _require_selected_option(state, building)
            amount = _require_selected_amount(state, building)
            result = self.driver.start_upgrade(
                state.request,
                _require_character(state),
                option,
                amount,
                _require_upgrade_state(state, building),
                _require_policy(state),
            )
            state.start_result = result
            state.start_results[building] = result
            if result.screenshot_path:
                state.screenshot_path = result.screenshot_path
            state.upgrade_attempts.append({"selected_upgrade": option.to_json(), "amount": amount.to_json(), **result.to_json()})
            confirmation = result.normalized_confirmation()
            if confirmation in {
                TroopUpgradeConfirmation.SPEEDUP,
                TroopUpgradeConfirmation.GEM,
                TroopUpgradeConfirmation.PREMIUM,
                TroopUpgradeConfirmation.RESOURCE_ITEM,
                TroopUpgradeConfirmation.TRAINING,
                TroopUpgradeConfirmation.BUILDING_UPGRADE,
                TroopUpgradeConfirmation.UNKNOWN,
            }:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    f"Unsafe troop upgrade confirmation cannot be handled safely: {confirmation.value}.",
                    screenshot_path=result.screenshot_path,
                    data=result.to_json(),
                )
            if not result.success:
                return self._upgrade_failure(step, state, result, "Troop upgrade could not be started.")
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"upgrade_attempts": state.upgrade_attempts})

    def _verify_upgrade_state(
        self,
        step: WorkflowStepSpec,
        state: _TroopUpgradeRuntimeState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        for building in state.active_buildings:
            option = _require_selected_option(state, building)
            amount = _require_selected_amount(state, building)
            verify = self.driver.verify_upgrade_state(
                state.request,
                _require_character(state),
                option,
                amount,
                _require_upgrade_state(state, building),
                _require_start_result(state, building),
                _require_policy(state),
            )
            if verify.screenshot_path:
                state.screenshot_path = verify.screenshot_path
            state.verification_attempts.append({"selected_upgrade": option.to_json(), "amount": amount.to_json(), **verify.to_json()})
            if not _upgrade_postcondition_verified(_require_upgrade_state(state, building), _require_start_result(state, building), verify):
                failure = replace(
                    verify,
                    retryable=False,
                    message=("Troop upgrade queue timer/state did not change after starting upgrade. " f"{verify.message}").strip(),
                )
                return self._upgrade_failure(step, state, failure, "Troop upgrade postcondition was not verified.")
            state.upgraded_buildings.append(building)
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"verification_attempts": state.verification_attempts})

    def _complete(self, step: WorkflowStepSpec, state: _TroopUpgradeRuntimeState) -> WorkflowStepResult:
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
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=self._upgrade_payload(state))

    def _skipped(self, step: WorkflowStepSpec, state: _TroopUpgradeRuntimeState) -> WorkflowStepResult:
        if state.terminal_outcome == WorkflowOutcome.SKIPPED:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data={"skipped_reason": state.terminal_reason, **self._upgrade_payload(state)},
                screenshot_path=state.screenshot_path,
            )
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED)

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _TroopUpgradeRuntimeState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if _is_manual_stop(state):
            state.recovery_outcome = {"attempted": False, "reason": "manual_intervention_required"}
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
        state.recovery_outcome = self._monitor_recovery(state, _job_run_id(context))
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _TroopUpgradeRuntimeState) -> WorkflowStepResult:
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
                **self._upgrade_payload(state),
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _TroopUpgradeRuntimeState,
        action: ResourceGatheringActionResult,
        *,
        hard_stop_message: str = "Troop upgrade action failed.",
    ) -> WorkflowStepResult:
        if action.screenshot_path:
            state.screenshot_path = action.screenshot_path
        if action.success:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=action.data, screenshot_path=action.screenshot_path)
        if action.retryable:
            return _step_result(
                step.step_key,
                WorkflowOutcome.RETRYABLE_FAILURE,
                action.message or hard_stop_message,
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            action.message or hard_stop_message,
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _upgrade_failure(
        self,
        step: WorkflowStepSpec,
        state: _TroopUpgradeRuntimeState,
        result: TroopUpgradeStartResult,
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

    def _upgrade_payload(self, state: _TroopUpgradeRuntimeState) -> dict[str, object]:
        selected = state.selected_option
        amount = state.selected_amount
        return {
            "enabled_buildings": [item.value for item in state.enabled_buildings],
            "upgraded_buildings": [item.value for item in state.upgraded_buildings],
            "upgraded_building_count": len(state.upgraded_buildings),
            "selected_building": state.selected_building.value if state.selected_building is not None else "",
            "selected_buildings": [item.value for item in state.selected_options],
            "selected_troop_type": selected.normalized_troop_type().value if selected is not None else "",
            "source_tier": selected.source_tier if selected is not None else None,
            "target_tier": selected.target_tier if selected is not None else None,
            "eligible_unit_count": selected.eligible_count if selected is not None else 0,
            "selected_upgrade_amount": amount.selected_amount if amount is not None else 0,
            "selected_upgrade": selected.to_json() if selected is not None else {},
            "selected_upgrades": [item.to_json() for item in state.selected_options.values()],
            "selected_amount": amount.to_json() if amount is not None else {},
            "selected_amounts": [item.to_json() for item in state.selected_amounts.values()],
            "queue_state": state.upgrade_state.to_json() if state.upgrade_state is not None else {},
            "queue_states": [item.to_json() for item in state.upgrade_states.values()],
            "upgrade_attempts": state.upgrade_attempts,
            "upgrade_attempt_count": len(state.upgrade_attempts),
            "verification_attempts": state.verification_attempts,
            "verification_result": state.verification_attempts[-1] if state.verification_attempts else {},
            "skipped_buildings": state.skipped_buildings,
            "skipped_reason": state.terminal_reason if state.terminal_outcome == WorkflowOutcome.SKIPPED else "",
            "recovery_outcome": state.recovery_outcome,
            "normalization_attempts": state.normalized_attempts,
            "open_attempts": state.open_attempts,
            "upgrade_tab_attempts": state.tab_attempts,
            "upgrade_inspections": state.inspections,
            "tier_selection_attempts": state.tier_selection_attempts,
            "amount_attempts": state.amount_attempts,
            "ignored_options": state.ignored_options,
            "failure_evidence": {
                "screenshot_path": state.screenshot_path,
                "terminal_state": state.terminal_state,
                "terminal_reason": state.terminal_reason,
            },
        }

    def _state_from_context(self, context: WorkflowExecutionContext) -> _TroopUpgradeRuntimeState:
        token = str(context.metadata.get("troop_upgrade_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("Troop upgrade runtime state is missing.") from exc

    def _open_incident(self, state: _TroopUpgradeRuntimeState) -> None:
        if self.incidents is None or not state.failed or state.incident_opened:
            return
        self.incidents.save(
            Incident(
                incident_key=f"troop-upgrade:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="Troop upgrade workflow blocked",
                details=state.terminal_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )
        state.incident_opened = True

    def _record_engine_failure(self, result: WorkflowExecutionResult, state: _TroopUpgradeRuntimeState) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "Troop upgrade workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(self, result: WorkflowExecutionResult, state: _TroopUpgradeRuntimeState) -> None:
        if state.failed and not state.recovery_outcome:
            if _is_manual_stop(state):
                state.recovery_outcome = {"attempted": False, "reason": "manual_intervention_required"}
            else:
                state.recovery_outcome = self._monitor_recovery(state, result.job_run_id)
        if state.failed:
            self._open_incident(state)

    def _monitor_recovery(self, state: _TroopUpgradeRuntimeState, job_run_id: int | None) -> dict[str, object]:
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

    def _augment_result(self, result: WorkflowExecutionResult, state: _TroopUpgradeRuntimeState) -> None:
        if state.terminal_outcome in {WorkflowOutcome.SKIPPED, WorkflowOutcome.BLOCKED}:
            result.outcome = state.terminal_outcome
            result.message = state.terminal_reason
        result.result = {
            **dict(result.result),
            "policy": state.policy.to_json() if state.policy is not None else {},
            **self._upgrade_payload(state),
            "terminal_state": state.terminal_state,
            "terminal_reason": state.terminal_reason,
            "failure_state": state.terminal_state if result.outcome.is_failure else "",
            "failure_reason": state.terminal_reason if result.outcome.is_failure else "",
            "recovery_outcome": state.recovery_outcome,
        }

    def _update_persisted_run(self, result: WorkflowExecutionResult, state: _TroopUpgradeRuntimeState) -> None:
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


def _select_upgrade_option(
    upgrade_state: TroopUpgradeState,
    policy: TroopUpgradePolicy,
) -> tuple[TroopUpgradeOption | None, list[dict[str, object]]]:
    ignored: list[dict[str, object]] = []
    for option in upgrade_state.eligible_options:
        reason = _option_skip_reason(option, upgrade_state, policy)
        if reason:
            ignored.append({**option.to_json(), "ignored_reason": reason})
            continue
        return option, ignored
    if not upgrade_state.eligible_options:
        ignored.append(
            {
                "building": upgrade_state.normalized_building().value,
                "source_tier": policy.source_tier,
                "target_tier": policy.target_tier,
                "ignored_reason": "no_eligible_units",
            }
        )
    return None, ignored


def _option_skip_reason(
    option: TroopUpgradeOption,
    upgrade_state: TroopUpgradeState,
    policy: TroopUpgradePolicy,
) -> str:
    building = option.normalized_building()
    if building != upgrade_state.normalized_building():
        return "wrong_building"
    if option.normalized_troop_type() != upgrade_state.normalized_troop_type():
        return "wrong_troop_type"
    if option.source_tier != policy.source_tier:
        return "source_tier_not_configured"
    if option.target_tier != policy.target_tier:
        return "target_tier_not_configured"
    if option.eligible_count <= 0:
        return "no_eligible_units"
    if not option.enabled:
        return "disabled"
    if not option.resources_available:
        return "insufficient_resources"
    if policy.max_resource_cost and option.resource_cost > policy.max_resource_cost:
        return "resource_budget_exceeded"
    if option.confidence < policy.minimum_detector_confidence:
        return "below_confidence_threshold"
    return ""


def _amount_for_option(option: TroopUpgradeOption, policy: TroopUpgradePolicy) -> TroopUpgradeAmountPlan:
    selected = min(option.eligible_count, policy.max_upgrade_amount)
    if not policy.allow_all_eligible:
        selected = min(selected, policy.upgrade_amount)
    selected = max(1, selected)
    return TroopUpgradeAmountPlan(
        building=option.normalized_building(),
        source_tier=option.source_tier,
        target_tier=option.target_tier,
        eligible_count=option.eligible_count,
        selected_amount=selected,
        resource_cost=option.resource_cost,
        all_eligible_requested=policy.allow_all_eligible,
    )


def _upgrade_postcondition_verified(
    before: TroopUpgradeState,
    start: TroopUpgradeStartResult,
    verify: TroopUpgradeStartResult,
) -> bool:
    if not verify.success or not verify.changed:
        return False
    if start.changed:
        return True
    if verify.queue_size is not None and verify.queue_size != before.queue_size:
        return True
    if verify.timer_seconds is not None and verify.timer_seconds != before.timer_seconds:
        return True
    return False


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
        action_type=f"troop_upgrade.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _TroopUpgradeRuntimeState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _TroopUpgradeRuntimeState) -> TroopUpgradePolicy:
    if state.policy is None:
        raise RuntimeError("Troop upgrade policy has not been validated.")
    return state.policy


def _require_upgrade_state(
    state: _TroopUpgradeRuntimeState,
    building: TroopTrainingBuilding | None = None,
) -> TroopUpgradeState:
    if building is not None:
        try:
            return state.upgrade_states[building]
        except KeyError as exc:
            raise RuntimeError(f"{building.value} troop upgrade state has not been inspected.") from exc
    if state.upgrade_state is None:
        raise RuntimeError("Troop upgrade state has not been inspected.")
    return state.upgrade_state


def _require_selected_option(
    state: _TroopUpgradeRuntimeState,
    building: TroopTrainingBuilding | None = None,
) -> TroopUpgradeOption:
    if building is not None:
        try:
            return state.selected_options[building]
        except KeyError as exc:
            raise RuntimeError(f"{building.value} troop upgrade tiers have not been selected.") from exc
    if state.selected_option is None:
        raise RuntimeError("Troop upgrade tiers have not been selected.")
    return state.selected_option


def _require_selected_amount(
    state: _TroopUpgradeRuntimeState,
    building: TroopTrainingBuilding | None = None,
) -> TroopUpgradeAmountPlan:
    if building is not None:
        try:
            return state.selected_amounts[building]
        except KeyError as exc:
            raise RuntimeError(f"{building.value} troop upgrade amount has not been selected.") from exc
    if state.selected_amount is None:
        raise RuntimeError("Troop upgrade amount has not been selected.")
    return state.selected_amount


def _require_start_result(
    state: _TroopUpgradeRuntimeState,
    building: TroopTrainingBuilding | None = None,
) -> TroopUpgradeStartResult:
    if building is not None:
        try:
            return state.start_results[building]
        except KeyError as exc:
            raise RuntimeError(f"{building.value} troop upgrade has not been started.") from exc
    if state.start_result is None:
        raise RuntimeError("Troop upgrade has not been started.")
    return state.start_result


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_manual_stop(state: _TroopUpgradeRuntimeState) -> bool:
    text = state.terminal_reason.lower()
    return "verification" in text or "confirmation" in text or "manual" in text
