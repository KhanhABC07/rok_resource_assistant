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


TROOP_TRAINING_WORKFLOW_KEY = "troop-training"
TROOP_TRAINING_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "normalize_city_view",
    "open_training_building",
    "inspect_training_queue",
    "select_training_tier",
    "start_training",
    "verify_training_state",
    "complete",
    "skipped",
    "recover",
    "failed",
    "cancelled",
)


class TroopType(StrEnum):
    INFANTRY = "INFANTRY"
    ARCHER = "ARCHER"
    CAVALRY = "CAVALRY"
    SIEGE = "SIEGE"


class TroopTrainingBuilding(StrEnum):
    BARRACKS = "BARRACKS"
    ARCHERY_RANGE = "ARCHERY_RANGE"
    STABLE = "STABLE"
    SIEGE_WORKSHOP = "SIEGE_WORKSHOP"


class TroopTrainingQueueStatus(StrEnum):
    IDLE = "IDLE"
    BUSY = "BUSY"
    READY = "READY"
    INSUFFICIENT_RESOURCES = "INSUFFICIENT_RESOURCES"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


class TroopTrainingConfirmation(StrEnum):
    NONE = "NONE"
    FREE = "FREE"
    SPEEDUP = "SPEEDUP"
    GEM = "GEM"
    PREMIUM = "PREMIUM"
    RESOURCE_ITEM = "RESOURCE_ITEM"
    UPGRADE = "UPGRADE"
    UNKNOWN = "UNKNOWN"


_BUILDING_TROOP_TYPES = {
    TroopTrainingBuilding.BARRACKS: TroopType.INFANTRY,
    TroopTrainingBuilding.ARCHERY_RANGE: TroopType.ARCHER,
    TroopTrainingBuilding.STABLE: TroopType.CAVALRY,
    TroopTrainingBuilding.SIEGE_WORKSHOP: TroopType.SIEGE,
}


def _building(value: TroopTrainingBuilding | str) -> TroopTrainingBuilding:
    if isinstance(value, TroopTrainingBuilding):
        return value
    try:
        return TroopTrainingBuilding(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in TroopTrainingBuilding)
        raise ValueError(f"Invalid troop training building: {value!r}. Expected one of: {valid}.") from exc


def _troop_type(value: TroopType | str) -> TroopType:
    if isinstance(value, TroopType):
        return value
    try:
        return TroopType(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in TroopType)
        raise ValueError(f"Invalid troop type: {value!r}. Expected one of: {valid}.") from exc


def _queue_status(value: TroopTrainingQueueStatus | str) -> TroopTrainingQueueStatus:
    if isinstance(value, TroopTrainingQueueStatus):
        return value
    try:
        return TroopTrainingQueueStatus(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in TroopTrainingQueueStatus)
        raise ValueError(f"Invalid troop training queue status: {value!r}. Expected one of: {valid}.") from exc


def _confirmation(value: TroopTrainingConfirmation | str) -> TroopTrainingConfirmation:
    if isinstance(value, TroopTrainingConfirmation):
        return value
    try:
        return TroopTrainingConfirmation(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in TroopTrainingConfirmation)
        raise ValueError(f"Invalid troop training confirmation: {value!r}. Expected one of: {valid}.") from exc


def _require_positive_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _require_confidence(value: float, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    return numeric


@dataclass(frozen=True)
class TroopTrainingPolicy:
    enabled_buildings: tuple[TroopTrainingBuilding | str, ...] = (
        TroopTrainingBuilding.BARRACKS,
        TroopTrainingBuilding.ARCHERY_RANGE,
        TroopTrainingBuilding.STABLE,
        TroopTrainingBuilding.SIEGE_WORKSHOP,
    )
    desired_tier: int = 1
    skip_busy_queue: bool = True
    skip_insufficient_resources: bool = True
    allow_speedups: bool = False
    allow_gem_spending: bool = False
    allow_premium_spending: bool = False
    allow_resource_items: bool = False
    minimum_detector_confidence: float = 0.85

    def normalized(self) -> TroopTrainingPolicy:
        enabled = tuple(dict.fromkeys(_building(item) for item in self.enabled_buildings))
        if not enabled:
            raise ValueError("At least one troop training building must be enabled.")
        tier = _require_positive_int(self.desired_tier, "desired_tier")
        if tier > 5:
            raise ValueError("desired_tier must be between 1 and 5.")
        if self.allow_speedups:
            raise ValueError("Speedup usage is out of scope for TROOP-001.")
        if self.allow_gem_spending:
            raise ValueError("Gem spending is out of scope for TROOP-001.")
        if self.allow_premium_spending:
            raise ValueError("Premium currency spending is out of scope for TROOP-001.")
        if self.allow_resource_items:
            raise ValueError("Resource item usage is out of scope for TROOP-001.")
        return TroopTrainingPolicy(
            enabled_buildings=enabled,
            desired_tier=tier,
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
            "desired_tier": normalized.desired_tier,
            "skip_busy_queue": normalized.skip_busy_queue,
            "skip_insufficient_resources": normalized.skip_insufficient_resources,
            "allow_speedups": normalized.allow_speedups,
            "allow_gem_spending": normalized.allow_gem_spending,
            "allow_premium_spending": normalized.allow_premium_spending,
            "allow_resource_items": normalized.allow_resource_items,
            "minimum_detector_confidence": normalized.minimum_detector_confidence,
        }


@dataclass(frozen=True)
class TroopTrainingRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: TroopTrainingPolicy = field(default_factory=TroopTrainingPolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class TroopTrainingConfig:
    workflow_timeout_seconds: float = 180.0
    step_timeout_seconds: float = 15.0
    precondition_retry_limit: int = 1
    navigation_retry_limit: int = 1
    inspect_retry_limit: int = 1
    action_retry_limit: int = 0
    retry_delay_seconds: float = 0.25


@dataclass(frozen=True)
class TroopTrainingTierOption:
    building: TroopTrainingBuilding | str
    troop_type: TroopType | str
    tier: int
    enabled: bool = True
    resources_available: bool = True
    confidence: float = 1.0
    scene_verified: bool = True
    tier_verified: bool = True
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_building(self) -> TroopTrainingBuilding:
        return _building(self.building)

    def normalized_troop_type(self) -> TroopType:
        return _troop_type(self.troop_type)

    def to_json(self) -> dict[str, object]:
        return {
            "building": self.normalized_building().value,
            "troop_type": self.normalized_troop_type().value,
            "tier": self.tier,
            "enabled": self.enabled,
            "resources_available": self.resources_available,
            "confidence": self.confidence,
            "scene_verified": self.scene_verified,
            "tier_verified": self.tier_verified,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class TroopTrainingQueueState:
    building: TroopTrainingBuilding | str
    troop_type: TroopType | str
    status: TroopTrainingQueueStatus | str
    available_tiers: tuple[TroopTrainingTierOption, ...] = ()
    active_tier: int | None = None
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
        return _troop_type(self.troop_type)

    def normalized_status(self) -> TroopTrainingQueueStatus:
        return _queue_status(self.status)

    def to_json(self) -> dict[str, object]:
        return {
            "building": self.normalized_building().value,
            "troop_type": self.normalized_troop_type().value,
            "status": self.normalized_status().value,
            "available_tiers": [item.to_json() for item in self.available_tiers],
            "active_tier": self.active_tier,
            "queue_size": self.queue_size,
            "timer_seconds": self.timer_seconds,
            "scene_verified": self.scene_verified,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class TroopTrainingStartResult:
    success: bool
    changed: bool = False
    confirmation: TroopTrainingConfirmation | str = TroopTrainingConfirmation.NONE
    queue_size: int | None = None
    timer_seconds: int | None = None
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_confirmation(self) -> TroopTrainingConfirmation:
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


class TroopTrainingAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: TroopTrainingRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class TroopTrainingCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: TroopTrainingRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class TroopTrainingDriver(Protocol):
    def normalize_city_view(
        self,
        request: TroopTrainingRequest,
        character: Character,
        policy: TroopTrainingPolicy,
        building: TroopTrainingBuilding,
    ) -> ResourceGatheringActionResult:
        ...

    def open_training_building(
        self,
        request: TroopTrainingRequest,
        character: Character,
        policy: TroopTrainingPolicy,
        building: TroopTrainingBuilding,
    ) -> ResourceGatheringActionResult:
        ...

    def inspect_training_queue(
        self,
        request: TroopTrainingRequest,
        character: Character,
        policy: TroopTrainingPolicy,
        building: TroopTrainingBuilding,
    ) -> TroopTrainingQueueState:
        ...

    def select_training_tier(
        self,
        request: TroopTrainingRequest,
        character: Character,
        option: TroopTrainingTierOption,
        policy: TroopTrainingPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def start_training(
        self,
        request: TroopTrainingRequest,
        character: Character,
        option: TroopTrainingTierOption,
        before: TroopTrainingQueueState,
        policy: TroopTrainingPolicy,
    ) -> TroopTrainingStartResult:
        ...

    def verify_training_state(
        self,
        request: TroopTrainingRequest,
        character: Character,
        option: TroopTrainingTierOption,
        before: TroopTrainingQueueState,
        start: TroopTrainingStartResult,
        policy: TroopTrainingPolicy,
    ) -> TroopTrainingStartResult:
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
class _TroopTrainingState:
    request: TroopTrainingRequest
    character: Character | None = None
    policy: TroopTrainingPolicy | None = None
    selected_building: TroopTrainingBuilding | None = None
    queue_state: TroopTrainingQueueState | None = None
    selected_tier: TroopTrainingTierOption | None = None
    start_result: TroopTrainingStartResult | None = None
    enabled_buildings: list[TroopTrainingBuilding] = field(default_factory=list)
    active_buildings: list[TroopTrainingBuilding] = field(default_factory=list)
    queue_states: dict[TroopTrainingBuilding, TroopTrainingQueueState] = field(default_factory=dict)
    selected_tiers: dict[TroopTrainingBuilding, TroopTrainingTierOption] = field(default_factory=dict)
    start_results: dict[TroopTrainingBuilding, TroopTrainingStartResult] = field(default_factory=dict)
    trained_buildings: list[TroopTrainingBuilding] = field(default_factory=list)
    skipped_buildings: list[dict[str, object]] = field(default_factory=list)
    normalized_attempts: list[dict[str, object]] = field(default_factory=list)
    open_attempts: list[dict[str, object]] = field(default_factory=list)
    queue_inspections: list[dict[str, object]] = field(default_factory=list)
    tier_selection_attempts: list[dict[str, object]] = field(default_factory=list)
    start_attempts: list[dict[str, object]] = field(default_factory=list)
    verification_attempts: list[dict[str, object]] = field(default_factory=list)
    ignored_tiers: list[dict[str, object]] = field(default_factory=list)
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


class TroopTrainingWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: TroopTrainingDriver,
        account_precondition: TroopTrainingAccountPrecondition | None = None,
        character_precondition: TroopTrainingCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: TroopTrainingConfig | None = None,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.account_precondition = account_precondition
        self.character_precondition = character_precondition
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.incidents = incidents
        self.config = config or TroopTrainingConfig()
        self._states: dict[str, _TroopTrainingState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return TROOP_TRAINING_STATES

    def execute(
        self,
        request: TroopTrainingRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _TroopTrainingState(request=request)
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
                budget=StepBudget(max_steps=len(TROOP_TRAINING_STATES) + 16),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"troop-training:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"troop_training_run_id": token},
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
        for state in TROOP_TRAINING_STATES:
            registry.register(f"troop_training.{state}", self._handler_for(state))
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
            "inspect_training_queue": self.config.inspect_retry_limit,
            "select_training_tier": self.config.inspect_retry_limit,
            "start_training": self.config.action_retry_limit,
            "verify_training_state": self.config.inspect_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=TROOP_TRAINING_WORKFLOW_KEY,
            name="Train Troops",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"troop_training.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in TROOP_TRAINING_STATES
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
        state: _TroopTrainingState,
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
        state: _TroopTrainingState,
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
        state: _TroopTrainingState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.account_precondition is None:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"skipped": True})
        return self._action_to_step(
            step,
            state,
            self.account_precondition.ensure_account(state.request, _require_character(state)),
        )

    def _ensure_character(
        self,
        step: WorkflowStepSpec,
        state: _TroopTrainingState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.character_precondition is None:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"skipped": True})
        return self._action_to_step(
            step,
            state,
            self.character_precondition.ensure_character(state.request, _require_character(state)),
        )

    def _ensure_game_running(
        self,
        step: WorkflowStepSpec,
        _state: _TroopTrainingState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"precondition": "delegated_to_driver"})

    def _normalize_city_view(
        self,
        step: WorkflowStepSpec,
        state: _TroopTrainingState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.enabled_buildings:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                "No enabled troop training building remains.",
                data={"enabled_buildings": [item.value for item in state.enabled_buildings]},
            )
        for building in state.enabled_buildings:
            state.selected_building = building
            result = self.driver.normalize_city_view(
                state.request,
                _require_character(state),
                _require_policy(state),
                building,
            )
            attempt = {"building": building.value, **result.data, "screenshot_path": result.screenshot_path}
            state.normalized_attempts.append(attempt)
            step_result = self._action_to_step(step, state, result)
            if step_result.outcome != WorkflowOutcome.SUCCESS:
                return step_result
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"normalization_attempts": state.normalized_attempts})

    def _open_training_building(
        self,
        step: WorkflowStepSpec,
        state: _TroopTrainingState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        for building in state.active_buildings:
            state.selected_building = building
            result = self.driver.open_training_building(
                state.request,
                _require_character(state),
                _require_policy(state),
                building,
            )
            state.open_attempts.append({"building": building.value, **result.data, "screenshot_path": result.screenshot_path})
            step_result = self._action_to_step(
                step,
                state,
                result,
                hard_stop_message="Training building scene could not be verified.",
            )
            if step_result.outcome != WorkflowOutcome.SUCCESS:
                return step_result
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"open_attempts": state.open_attempts})

    def _inspect_training_queue(
        self,
        step: WorkflowStepSpec,
        state: _TroopTrainingState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        policy = _require_policy(state)
        active: list[TroopTrainingBuilding] = []
        for building in state.active_buildings:
            state.selected_building = building
            queue = self.driver.inspect_training_queue(
                state.request,
                _require_character(state),
                policy,
                building,
            )
            state.queue_state = queue
            state.queue_states[building] = queue
            if queue.screenshot_path:
                state.screenshot_path = queue.screenshot_path
            state.queue_inspections.append(queue.to_json())
            status = queue.normalized_status()
            if not queue.scene_verified or status in {TroopTrainingQueueStatus.VERIFICATION_REQUIRED, TroopTrainingQueueStatus.UNKNOWN}:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    queue.message or "Training building scene or queue state could not be verified.",
                    screenshot_path=queue.screenshot_path,
                    data={"queue_state": queue.to_json()},
                )
            if status in {TroopTrainingQueueStatus.BUSY, TroopTrainingQueueStatus.READY}:
                if not policy.skip_busy_queue:
                    return state.stop(
                        step.step_key,
                        WorkflowOutcome.BLOCKED,
                        queue.message or "Training queue is already busy.",
                        screenshot_path=queue.screenshot_path,
                        data={"queue_state": queue.to_json()},
                    )
                state.skipped_buildings.append(
                    {"building": building.value, "reason": "busy_queue", "queue_state": queue.to_json()}
                )
                continue
            if status == TroopTrainingQueueStatus.INSUFFICIENT_RESOURCES and not queue.available_tiers:
                if not policy.skip_insufficient_resources:
                    return state.stop(
                        step.step_key,
                        WorkflowOutcome.BLOCKED,
                        queue.message or "Insufficient resources for troop training.",
                        screenshot_path=queue.screenshot_path,
                        data={"queue_state": queue.to_json()},
                    )
                state.skipped_buildings.append(
                    {"building": building.value, "reason": "insufficient_resources", "queue_state": queue.to_json()}
                )
                continue
            active.append(building)
        state.active_buildings = active
        if not active:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                "No enabled troop training queue is available.",
                screenshot_path=state.screenshot_path,
                data={"queue_inspections": state.queue_inspections, "skipped_buildings": state.skipped_buildings},
            )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"queue_inspections": state.queue_inspections})

    def _select_training_tier(
        self,
        step: WorkflowStepSpec,
        state: _TroopTrainingState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        policy = _require_policy(state)
        still_active: list[TroopTrainingBuilding] = []
        for building in state.active_buildings:
            queue = _require_queue_state(state, building)
            selected, ignored = _select_tier_option(queue, policy)
            state.ignored_tiers.extend(ignored)
            if selected is None:
                insufficient = any(item.get("ignored_reason") == "insufficient_resources" for item in ignored)
                if insufficient and policy.skip_insufficient_resources:
                    state.skipped_buildings.append(
                        {"building": building.value, "reason": "insufficient_resources", "ignored_tiers": ignored}
                    )
                    continue
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    "Insufficient resources for configured troop tier." if insufficient else "Configured troop tier is not available.",
                    screenshot_path=state.screenshot_path,
                    data={"queue_state": queue.to_json(), "ignored_tiers": ignored},
                )
            if not selected.scene_verified or not selected.tier_verified:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    selected.message or "Selected troop tier could not be verified before training.",
                    screenshot_path=selected.screenshot_path or state.screenshot_path,
                    data={"selected_tier": selected.to_json(), "ignored_tiers": ignored},
                )
            action = self.driver.select_training_tier(
                state.request,
                _require_character(state),
                selected,
                policy,
            )
            state.tier_selection_attempts.append({"selected_tier": selected.to_json(), **action.data})
            if action.screenshot_path:
                state.screenshot_path = action.screenshot_path
            if not action.success:
                return self._action_to_step(step, state, action)
            state.selected_tier = selected
            state.selected_tiers[building] = selected
            still_active.append(building)
        state.active_buildings = still_active
        if not still_active:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                "No enabled troop training building has trainable configured tier.",
                screenshot_path=state.screenshot_path,
                data={"ignored_tiers": state.ignored_tiers, "skipped_buildings": state.skipped_buildings},
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "selected_tiers": [item.to_json() for item in state.selected_tiers.values()],
                "ignored_tiers": state.ignored_tiers,
            },
        )

    def _start_training(
        self,
        step: WorkflowStepSpec,
        state: _TroopTrainingState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        for building in state.active_buildings:
            selected = _require_selected_tier(state, building)
            result = self.driver.start_training(
                state.request,
                _require_character(state),
                selected,
                _require_queue_state(state, building),
                _require_policy(state),
            )
            state.start_result = result
            state.start_results[building] = result
            if result.screenshot_path:
                state.screenshot_path = result.screenshot_path
            state.start_attempts.append({"selected_tier": selected.to_json(), **result.to_json()})
            confirmation = result.normalized_confirmation()
            if confirmation in {
                TroopTrainingConfirmation.SPEEDUP,
                TroopTrainingConfirmation.GEM,
                TroopTrainingConfirmation.PREMIUM,
                TroopTrainingConfirmation.RESOURCE_ITEM,
                TroopTrainingConfirmation.UPGRADE,
                TroopTrainingConfirmation.UNKNOWN,
            }:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    f"Unsafe troop training confirmation cannot be handled safely: {confirmation.value}.",
                    screenshot_path=result.screenshot_path,
                    data=result.to_json(),
                )
            if not result.success:
                return self._training_failure(step, state, result, "Troop training could not be started.")
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"start_attempts": state.start_attempts})

    def _verify_training_state(
        self,
        step: WorkflowStepSpec,
        state: _TroopTrainingState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        for building in state.active_buildings:
            selected = _require_selected_tier(state, building)
            verify = self.driver.verify_training_state(
                state.request,
                _require_character(state),
                selected,
                _require_queue_state(state, building),
                _require_start_result(state, building),
                _require_policy(state),
            )
            if verify.screenshot_path:
                state.screenshot_path = verify.screenshot_path
            state.verification_attempts.append({"selected_tier": selected.to_json(), **verify.to_json()})
            if not _training_postcondition_verified(_require_queue_state(state, building), _require_start_result(state, building), verify):
                failure = replace(
                    verify,
                    retryable=False,
                    message=("Training queue timer/state did not change after starting training. " f"{verify.message}").strip(),
                )
                return self._training_failure(
                    step,
                    state,
                    failure,
                    "Troop training postcondition was not verified.",
                )
            state.trained_buildings.append(building)
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"verification_attempts": state.verification_attempts})

    def _complete(self, step: WorkflowStepSpec, state: _TroopTrainingState) -> WorkflowStepResult:
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
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=self._training_payload(state))

    def _skipped(self, step: WorkflowStepSpec, state: _TroopTrainingState) -> WorkflowStepResult:
        if state.terminal_outcome == WorkflowOutcome.SKIPPED:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data={"skipped_reason": state.terminal_reason, **self._training_payload(state)},
                screenshot_path=state.screenshot_path,
            )
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED)

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _TroopTrainingState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if _is_manual_stop(state):
            state.recovery_outcome = {"attempted": False, "reason": "manual_intervention_required"}
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
        state.recovery_outcome = self._monitor_recovery(state, _job_run_id(context))
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _TroopTrainingState) -> WorkflowStepResult:
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
                **self._training_payload(state),
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _TroopTrainingState,
        action: ResourceGatheringActionResult,
        *,
        hard_stop_message: str = "Troop training action failed.",
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

    def _training_failure(
        self,
        step: WorkflowStepSpec,
        state: _TroopTrainingState,
        result: TroopTrainingStartResult,
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

    def _training_payload(self, state: _TroopTrainingState) -> dict[str, object]:
        selected = state.selected_tier
        return {
            "enabled_buildings": [item.value for item in state.enabled_buildings],
            "trained_buildings": [item.value for item in state.trained_buildings],
            "trained_building_count": len(state.trained_buildings),
            "selected_building": state.selected_building.value if state.selected_building is not None else "",
            "selected_buildings": [item.value for item in state.selected_tiers],
            "selected_troop_type": selected.normalized_troop_type().value if selected is not None else "",
            "selected_tier": selected.to_json() if selected is not None else {},
            "selected_tiers": [item.to_json() for item in state.selected_tiers.values()],
            "queue_state": state.queue_state.to_json() if state.queue_state is not None else {},
            "queue_states": [item.to_json() for item in state.queue_states.values()],
            "start_attempts": state.start_attempts,
            "verification_attempts": state.verification_attempts,
            "verification_result": state.verification_attempts[-1] if state.verification_attempts else {},
            "skipped_buildings": state.skipped_buildings,
            "normalization_attempts": state.normalized_attempts,
            "open_attempts": state.open_attempts,
            "queue_inspections": state.queue_inspections,
            "tier_selection_attempts": state.tier_selection_attempts,
            "ignored_tiers": state.ignored_tiers,
            "failure_evidence": {
                "screenshot_path": state.screenshot_path,
                "terminal_state": state.terminal_state,
                "terminal_reason": state.terminal_reason,
            },
        }

    def _state_from_context(self, context: WorkflowExecutionContext) -> _TroopTrainingState:
        token = str(context.metadata.get("troop_training_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("Troop training runtime state is missing.") from exc

    def _open_incident(self, state: _TroopTrainingState) -> None:
        if self.incidents is None or not state.failed or state.incident_opened:
            return
        self.incidents.save(
            Incident(
                incident_key=f"troop-training:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="Troop training workflow blocked",
                details=state.terminal_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )
        state.incident_opened = True

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _TroopTrainingState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "Troop training workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(
        self,
        result: WorkflowExecutionResult,
        state: _TroopTrainingState,
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
        state: _TroopTrainingState,
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
        state: _TroopTrainingState,
    ) -> None:
        if state.terminal_outcome in {WorkflowOutcome.SKIPPED, WorkflowOutcome.BLOCKED}:
            result.outcome = state.terminal_outcome
            result.message = state.terminal_reason
        result.result = {
            **dict(result.result),
            "policy": state.policy.to_json() if state.policy is not None else {},
            **self._training_payload(state),
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
        state: _TroopTrainingState,
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


def _select_tier_option(
    queue: TroopTrainingQueueState,
    policy: TroopTrainingPolicy,
) -> tuple[TroopTrainingTierOption | None, list[dict[str, object]]]:
    ignored: list[dict[str, object]] = []
    for option in queue.available_tiers:
        reason = _tier_skip_reason(option, queue, policy)
        if reason:
            ignored.append({**option.to_json(), "ignored_reason": reason})
            continue
        return option, ignored
    return None, ignored


def _tier_skip_reason(
    option: TroopTrainingTierOption,
    queue: TroopTrainingQueueState,
    policy: TroopTrainingPolicy,
) -> str:
    building = option.normalized_building()
    if building != queue.normalized_building():
        return "wrong_building"
    if option.normalized_troop_type() != queue.normalized_troop_type():
        return "wrong_troop_type"
    if option.tier != policy.desired_tier:
        return "tier_not_configured"
    if not option.enabled:
        return "disabled"
    if not option.resources_available:
        return "insufficient_resources"
    if option.confidence < policy.minimum_detector_confidence:
        return "below_confidence_threshold"
    return ""


def _training_postcondition_verified(
    before: TroopTrainingQueueState,
    start: TroopTrainingStartResult,
    verify: TroopTrainingStartResult,
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
        action_type=f"troop_training.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _TroopTrainingState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _TroopTrainingState) -> TroopTrainingPolicy:
    if state.policy is None:
        raise RuntimeError("Troop training policy has not been validated.")
    return state.policy


def _require_selected_building(state: _TroopTrainingState) -> TroopTrainingBuilding:
    if state.selected_building is None:
        raise RuntimeError("Troop training building has not been selected.")
    return state.selected_building


def _require_queue_state(
    state: _TroopTrainingState,
    building: TroopTrainingBuilding | None = None,
) -> TroopTrainingQueueState:
    if building is not None:
        try:
            return state.queue_states[building]
        except KeyError as exc:
            raise RuntimeError(f"{building.value} training queue has not been inspected.") from exc
    if state.queue_state is None:
        raise RuntimeError("Troop training queue has not been inspected.")
    return state.queue_state


def _require_selected_tier(
    state: _TroopTrainingState,
    building: TroopTrainingBuilding | None = None,
) -> TroopTrainingTierOption:
    if building is not None:
        try:
            return state.selected_tiers[building]
        except KeyError as exc:
            raise RuntimeError(f"{building.value} troop training tier has not been selected.") from exc
    if state.selected_tier is None:
        raise RuntimeError("Troop training tier has not been selected.")
    return state.selected_tier


def _require_start_result(
    state: _TroopTrainingState,
    building: TroopTrainingBuilding | None = None,
) -> TroopTrainingStartResult:
    if building is not None:
        try:
            return state.start_results[building]
        except KeyError as exc:
            raise RuntimeError(f"{building.value} troop training has not been started.") from exc
    if state.start_result is None:
        raise RuntimeError("Troop training has not been started.")
    return state.start_result


def troop_type_for_building(building: TroopTrainingBuilding | str) -> TroopType:
    return _BUILDING_TROOP_TYPES[_building(building)]


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_manual_stop(state: _TroopTrainingState) -> bool:
    text = state.terminal_reason.lower()
    return "verification" in text or "confirmation" in text or "manual" in text
