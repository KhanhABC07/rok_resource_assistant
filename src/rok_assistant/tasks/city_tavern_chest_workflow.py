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


CITY_TAVERN_CHEST_WORKFLOW_KEY = "city-tavern-chest"
CITY_TAVERN_CHEST_TEMPLATE_KEYS = (
    "city.tavern.button",
    "tavern.scene",
    "tavern.silver.free",
    "tavern.gold.free",
    "tavern.chest.cooldown",
    "tavern.chest.key_required",
    "tavern.chest.gem_required",
    "tavern.reward.close",
)
CITY_TAVERN_CHEST_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "open_tavern",
    "scan_chests",
    "select_free_chest",
    "open_free_chest",
    "close_reward_ui",
    "verify_chest_state",
    "complete",
    "skipped",
    "recover",
    "failed",
    "cancelled",
)


class TavernChestType(StrEnum):
    SILVER = "SILVER"
    GOLD = "GOLD"


class TavernChestStatus(StrEnum):
    FREE = "FREE"
    COOLDOWN = "COOLDOWN"
    UNAVAILABLE = "UNAVAILABLE"
    PAID = "PAID"
    KEY_REQUIRED = "KEY_REQUIRED"
    GEM_REQUIRED = "GEM_REQUIRED"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


class TavernChestConfirmation(StrEnum):
    NONE = "NONE"
    FREE = "FREE"
    PAID = "PAID"
    KEY = "KEY"
    GEM = "GEM"
    UNKNOWN = "UNKNOWN"


def _chest_type(value: TavernChestType | str) -> TavernChestType:
    if isinstance(value, TavernChestType):
        return value
    try:
        return TavernChestType(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in TavernChestType)
        raise ValueError(f"Invalid tavern chest type: {value!r}. Expected one of: {valid}.") from exc


def _chest_status(value: TavernChestStatus | str) -> TavernChestStatus:
    if isinstance(value, TavernChestStatus):
        return value
    try:
        return TavernChestStatus(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in TavernChestStatus)
        raise ValueError(f"Invalid tavern chest status: {value!r}. Expected one of: {valid}.") from exc


def _confirmation(value: TavernChestConfirmation | str) -> TavernChestConfirmation:
    if isinstance(value, TavernChestConfirmation):
        return value
    try:
        return TavernChestConfirmation(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in TavernChestConfirmation)
        raise ValueError(f"Invalid tavern chest confirmation: {value!r}. Expected one of: {valid}.") from exc


def _require_confidence(value: float, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    return numeric


@dataclass(frozen=True)
class TavernChestPolicy:
    allow_silver_free_chest: bool = True
    allow_gold_free_chest: bool = True
    allow_key_spending: bool = False
    allow_gem_spending: bool = False
    block_when_only_paid_or_key_options: bool = False
    minimum_detector_confidence: float = 0.85

    def normalized(self) -> TavernChestPolicy:
        if self.allow_key_spending:
            raise ValueError("Key spending is not supported by CITY-005.")
        if self.allow_gem_spending:
            raise ValueError("Gem spending is not supported by CITY-005.")
        return TavernChestPolicy(
            allow_silver_free_chest=bool(self.allow_silver_free_chest),
            allow_gold_free_chest=bool(self.allow_gold_free_chest),
            allow_key_spending=False,
            allow_gem_spending=False,
            block_when_only_paid_or_key_options=bool(self.block_when_only_paid_or_key_options),
            minimum_detector_confidence=_require_confidence(
                self.minimum_detector_confidence,
                "minimum_detector_confidence",
            ),
        )

    def allows(self, chest_type: TavernChestType) -> bool:
        if chest_type == TavernChestType.SILVER:
            return self.allow_silver_free_chest
        if chest_type == TavernChestType.GOLD:
            return self.allow_gold_free_chest
        return False

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "allow_silver_free_chest": normalized.allow_silver_free_chest,
            "allow_gold_free_chest": normalized.allow_gold_free_chest,
            "allow_key_spending": normalized.allow_key_spending,
            "allow_gem_spending": normalized.allow_gem_spending,
            "block_when_only_paid_or_key_options": normalized.block_when_only_paid_or_key_options,
            "minimum_detector_confidence": normalized.minimum_detector_confidence,
        }


@dataclass(frozen=True)
class TavernChestRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: TavernChestPolicy = field(default_factory=TavernChestPolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class TavernChestConfig:
    workflow_timeout_seconds: float = 120.0
    step_timeout_seconds: float = 15.0
    precondition_retry_limit: int = 1
    navigation_retry_limit: int = 1
    scan_retry_limit: int = 1
    action_retry_limit: int = 0
    retry_delay_seconds: float = 0.25


@dataclass(frozen=True)
class TavernChestObservation:
    chest_type: TavernChestType | str
    status: TavernChestStatus | str
    confidence: float = 1.0
    target: tuple[int, int] | None = None
    scene_verified: bool = True
    free_indicator_visible: bool = False
    cooldown_seconds: int | None = None
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_chest_type(self) -> TavernChestType:
        return _chest_type(self.chest_type)

    def normalized_status(self) -> TavernChestStatus:
        return _chest_status(self.status)

    def target_json(self) -> dict[str, int] | None:
        if self.target is None:
            return None
        return {"x": int(self.target[0]), "y": int(self.target[1])}

    def to_json(self) -> dict[str, object]:
        return {
            "chest_type": self.normalized_chest_type().value,
            "status": self.normalized_status().value,
            "confidence": self.confidence,
            "target": self.target_json(),
            "scene_verified": self.scene_verified,
            "free_indicator_visible": self.free_indicator_visible,
            "cooldown_seconds": self.cooldown_seconds,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class TavernChestScan:
    observations: tuple[TavernChestObservation, ...] = ()
    scene_verified: bool = True
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "observations": [item.to_json() for item in self.observations],
            "scene_verified": self.scene_verified,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class TavernChestOpenResult:
    success: bool
    changed: bool = False
    reward_ui_present: bool = True
    confirmation: TavernChestConfirmation | str = TavernChestConfirmation.NONE
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_confirmation(self) -> TavernChestConfirmation:
        return _confirmation(self.confirmation)

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "changed": self.changed,
            "reward_ui_present": self.reward_ui_present,
            "confirmation": self.normalized_confirmation().value,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class TavernRewardCloseResult:
    success: bool
    closed: bool = False
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "closed": self.closed,
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


class TavernChestAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: TavernChestRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class TavernChestCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: TavernChestRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class TavernChestDriver(Protocol):
    def open_tavern(
        self,
        request: TavernChestRequest,
        character: Character,
        policy: TavernChestPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def scan_chests(
        self,
        request: TavernChestRequest,
        character: Character,
        policy: TavernChestPolicy,
    ) -> TavernChestScan:
        ...

    def open_free_chest(
        self,
        request: TavernChestRequest,
        character: Character,
        chest: TavernChestObservation,
        policy: TavernChestPolicy,
    ) -> TavernChestOpenResult:
        ...

    def close_reward_ui(
        self,
        request: TavernChestRequest,
        character: Character,
        chest: TavernChestObservation,
        open_result: TavernChestOpenResult,
        policy: TavernChestPolicy,
    ) -> TavernRewardCloseResult:
        ...

    def verify_chest_state(
        self,
        request: TavernChestRequest,
        character: Character,
        chest: TavernChestObservation,
        open_result: TavernChestOpenResult,
        policy: TavernChestPolicy,
    ) -> TavernChestObservation:
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
class _TavernChestState:
    request: TavernChestRequest
    character: Character | None = None
    policy: TavernChestPolicy | None = None
    scan: TavernChestScan | None = None
    selected_chests: list[TavernChestObservation] = field(default_factory=list)
    opened_chests: list[tuple[TavernChestObservation, TavernChestOpenResult]] = field(default_factory=list)
    scan_attempts: list[dict[str, object]] = field(default_factory=list)
    ignored_chests: list[dict[str, object]] = field(default_factory=list)
    open_attempts: list[dict[str, object]] = field(default_factory=list)
    reward_close_attempts: list[dict[str, object]] = field(default_factory=list)
    verification_attempts: list[dict[str, object]] = field(default_factory=list)
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


class TavernChestWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: TavernChestDriver,
        account_precondition: TavernChestAccountPrecondition | None = None,
        character_precondition: TavernChestCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: TavernChestConfig | None = None,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.account_precondition = account_precondition
        self.character_precondition = character_precondition
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.incidents = incidents
        self.config = config or TavernChestConfig()
        self._states: dict[str, _TavernChestState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return CITY_TAVERN_CHEST_STATES

    def execute(
        self,
        request: TavernChestRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _TavernChestState(request=request)
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
                budget=StepBudget(max_steps=len(CITY_TAVERN_CHEST_STATES) + 12),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"city-tavern-chest:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"city_tavern_chest_run_id": token},
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
        for state in CITY_TAVERN_CHEST_STATES:
            registry.register(f"city_tavern_chest.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.precondition_retry_limit,
            "ensure_character": self.config.precondition_retry_limit,
            "ensure_game_running": self.config.precondition_retry_limit,
            "open_tavern": self.config.navigation_retry_limit,
            "scan_chests": self.config.scan_retry_limit,
            "open_free_chest": self.config.action_retry_limit,
            "close_reward_ui": self.config.action_retry_limit,
            "verify_chest_state": self.config.scan_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=CITY_TAVERN_CHEST_WORKFLOW_KEY,
            name="Open Free Tavern Chest",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"city_tavern_chest.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in CITY_TAVERN_CHEST_STATES
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
        state: _TavernChestState,
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
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "policy": state.policy.to_json(),
                "template_keys": list(CITY_TAVERN_CHEST_TEMPLATE_KEYS),
            },
        )

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _TavernChestState,
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
        state: _TavernChestState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.account_precondition is None:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"account_precondition": "not_configured"})
        return self._action_to_step(step, state, self.account_precondition.ensure_account(state.request, _require_character(state)))

    def _ensure_character(
        self,
        step: WorkflowStepSpec,
        state: _TavernChestState,
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
        state: _TavernChestState,
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

    def _open_tavern(
        self,
        step: WorkflowStepSpec,
        state: _TavernChestState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.open_tavern(state.request, _require_character(state), _require_policy(state)),
            hard_stop_message="Tavern scene could not be verified before opening free chests.",
        )

    def _scan_chests(
        self,
        step: WorkflowStepSpec,
        state: _TavernChestState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        scan = self.driver.scan_chests(state.request, _require_character(state), _require_policy(state))
        state.scan = scan
        if scan.screenshot_path:
            state.screenshot_path = scan.screenshot_path
        state.scan_attempts.append(scan.to_json())
        if not scan.scene_verified:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                scan.message or "Tavern scene could not be verified.",
                screenshot_path=scan.screenshot_path,
                data={"scan": scan.to_json()},
            )
        for observation in scan.observations:
            if observation.normalized_status() == TavernChestStatus.VERIFICATION_REQUIRED:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.FATAL_FAILURE,
                    observation.message or "Verification screen requires manual intervention.",
                    screenshot_path=observation.screenshot_path or scan.screenshot_path,
                    data={"scan": scan.to_json(), "chest": observation.to_json()},
                )
            if observation.normalized_status() == TavernChestStatus.UNKNOWN:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    observation.message or "Tavern chest state could not be determined safely.",
                    screenshot_path=observation.screenshot_path or scan.screenshot_path,
                    data={"scan": scan.to_json(), "chest": observation.to_json()},
                )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"scan": scan.to_json()}, screenshot_path=scan.screenshot_path)

    def _select_free_chest(
        self,
        step: WorkflowStepSpec,
        state: _TavernChestState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        selected, ignored = _select_free_chests(_require_scan(state).observations, _require_policy(state))
        state.selected_chests = selected
        state.ignored_chests = ignored
        if not selected:
            paid_or_key_only = bool(ignored) and all(
                item.get("ignored_reason") in {
                    "paid_not_allowed",
                    "key_spending_not_allowed",
                    "gem_spending_not_allowed",
                }
                for item in ignored
            )
            outcome = (
                WorkflowOutcome.BLOCKED
                if paid_or_key_only and _require_policy(state).block_when_only_paid_or_key_options
                else WorkflowOutcome.SKIPPED
            )
            reason = "Only paid, key, or gem tavern chest options are present." if paid_or_key_only else "No free tavern chest is available."
            return state.stop(
                step.step_key,
                outcome,
                reason,
                screenshot_path=state.screenshot_path,
                data={"ignored_chests": ignored, "scan": _require_scan(state).to_json()},
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "selected_chests": [item.to_json() for item in selected],
                "ignored_chests": ignored,
            },
            screenshot_path=state.screenshot_path,
        )

    def _open_free_chest(
        self,
        step: WorkflowStepSpec,
        state: _TavernChestState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        for chest in state.selected_chests:
            context.cancellation_token.throw_if_cancelled()
            result = self.driver.open_free_chest(state.request, _require_character(state), chest, _require_policy(state))
            if result.screenshot_path:
                state.screenshot_path = result.screenshot_path
            attempt = {"chest": chest.to_json(), **result.to_json()}
            state.open_attempts.append(attempt)
            confirmation = result.normalized_confirmation()
            if confirmation in {TavernChestConfirmation.PAID, TavernChestConfirmation.KEY, TavernChestConfirmation.GEM, TavernChestConfirmation.UNKNOWN}:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    f"Unsafe tavern chest confirmation cannot be handled safely: {confirmation.value}.",
                    screenshot_path=result.screenshot_path,
                    data={"attempt": attempt},
                )
            if not result.success:
                return self._open_failure(step, state, result, "Free tavern chest could not be opened.")
            state.opened_chests.append((chest, result))
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"open_attempts": state.open_attempts},
            screenshot_path=state.screenshot_path,
        )

    def _close_reward_ui(
        self,
        step: WorkflowStepSpec,
        state: _TavernChestState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        for chest, open_result in state.opened_chests:
            context.cancellation_token.throw_if_cancelled()
            if not open_result.reward_ui_present:
                state.reward_close_attempts.append(
                    {"chest": chest.to_json(), "skipped": True, "reason": "reward_ui_not_present"}
                )
                continue
            close = self.driver.close_reward_ui(
                state.request,
                _require_character(state),
                chest,
                open_result,
                _require_policy(state),
            )
            if close.screenshot_path:
                state.screenshot_path = close.screenshot_path
            attempt = {"chest": chest.to_json(), **close.to_json()}
            state.reward_close_attempts.append(attempt)
            if not close.success or not close.closed:
                return self._reward_close_failure(step, state, close, "Tavern reward UI could not be closed safely.")
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"reward_ui_handling": state.reward_close_attempts},
            screenshot_path=state.screenshot_path,
        )

    def _verify_chest_state(
        self,
        step: WorkflowStepSpec,
        state: _TavernChestState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        for chest, open_result in state.opened_chests:
            context.cancellation_token.throw_if_cancelled()
            verification = self.driver.verify_chest_state(
                state.request,
                _require_character(state),
                chest,
                open_result,
                _require_policy(state),
            )
            if verification.screenshot_path:
                state.screenshot_path = verification.screenshot_path
            state.verification_attempts.append({"before": chest.to_json(), "after": verification.to_json()})
            if not _chest_postcondition_verified(chest, verification):
                failure = replace(
                    verification,
                    status=TavernChestStatus.UNKNOWN,
                    message=("Tavern chest free indicator did not change to cooldown or unavailable. " f"{verification.message}").strip(),
                )
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    failure.message,
                    screenshot_path=failure.screenshot_path,
                    data={"verification": failure.to_json()},
                )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"verification_attempts": state.verification_attempts},
            screenshot_path=state.screenshot_path,
        )

    def _complete(self, step: WorkflowStepSpec, state: _TavernChestState) -> WorkflowStepResult:
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
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=self._tavern_payload(state))

    def _skipped(self, step: WorkflowStepSpec, state: _TavernChestState) -> WorkflowStepResult:
        if state.terminal_outcome == WorkflowOutcome.SKIPPED:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data={"skipped_reason": state.terminal_reason, **self._tavern_payload(state)},
                screenshot_path=state.screenshot_path,
            )
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED)

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _TavernChestState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if _is_manual_stop(state):
            state.recovery_outcome = {"attempted": False, "reason": "manual_intervention_required"}
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
        state.recovery_outcome = self._monitor_recovery(state, _job_run_id(context))
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _TavernChestState) -> WorkflowStepResult:
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
                **self._tavern_payload(state),
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _TavernChestState,
        action: ResourceGatheringActionResult,
        *,
        hard_stop_message: str = "Tavern chest action failed.",
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

    def _open_failure(
        self,
        step: WorkflowStepSpec,
        state: _TavernChestState,
        result: TavernChestOpenResult,
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

    def _reward_close_failure(
        self,
        step: WorkflowStepSpec,
        state: _TavernChestState,
        result: TavernRewardCloseResult,
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

    def _tavern_payload(self, state: _TavernChestState) -> dict[str, object]:
        return {
            "scan": state.scan.to_json() if state.scan is not None else {},
            "scanned_chest_states": state.scan_attempts,
            "selected_chests": [item.to_json() for item in state.selected_chests],
            "selected_chest_types": [item.normalized_chest_type().value for item in state.selected_chests],
            "ignored_chests": state.ignored_chests,
            "open_attempts": state.open_attempts,
            "reward_ui_handling": state.reward_close_attempts,
            "verification_attempts": state.verification_attempts,
            "verification_result": state.verification_attempts[-1] if state.verification_attempts else {},
            "failure_evidence": {
                "screenshot_path": state.screenshot_path,
                "terminal_state": state.terminal_state,
                "terminal_reason": state.terminal_reason,
            },
        }

    def _state_from_context(self, context: WorkflowExecutionContext) -> _TavernChestState:
        token = str(context.metadata.get("city_tavern_chest_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("City tavern chest runtime state is missing.") from exc

    def _open_incident(self, state: _TavernChestState) -> None:
        if self.incidents is None or not state.failed or state.incident_opened:
            return
        self.incidents.save(
            Incident(
                incident_key=f"city-tavern-chest:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="City tavern chest workflow blocked",
                details=state.terminal_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )
        state.incident_opened = True

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _TavernChestState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "City tavern chest workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(
        self,
        result: WorkflowExecutionResult,
        state: _TavernChestState,
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
        state: _TavernChestState,
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
        state: _TavernChestState,
    ) -> None:
        if state.terminal_outcome in {WorkflowOutcome.SKIPPED, WorkflowOutcome.BLOCKED}:
            result.outcome = state.terminal_outcome
            result.message = state.terminal_reason
        result.result = {
            **dict(result.result),
            "policy": state.policy.to_json() if state.policy is not None else {},
            **self._tavern_payload(state),
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
        state: _TavernChestState,
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


def _select_free_chests(
    observations: tuple[TavernChestObservation, ...],
    policy: TavernChestPolicy,
) -> tuple[list[TavernChestObservation], list[dict[str, object]]]:
    selected: list[TavernChestObservation] = []
    ignored: list[dict[str, object]] = []
    by_type = {item.normalized_chest_type(): item for item in observations}
    for chest_type in (TavernChestType.SILVER, TavernChestType.GOLD):
        observation = by_type.get(chest_type)
        if observation is None:
            continue
        reason = _chest_skip_reason(observation, policy)
        if reason:
            ignored.append({**observation.to_json(), "ignored_reason": reason})
            continue
        selected.append(observation)
    return selected, ignored


def _chest_skip_reason(observation: TavernChestObservation, policy: TavernChestPolicy) -> str:
    status = observation.normalized_status()
    if not policy.allows(observation.normalized_chest_type()):
        return "chest_type_not_allowed_by_policy"
    if not observation.scene_verified:
        return "scene_not_verified"
    if observation.confidence < policy.minimum_detector_confidence:
        return "below_confidence_threshold"
    if status == TavernChestStatus.FREE and observation.free_indicator_visible:
        return ""
    if status == TavernChestStatus.KEY_REQUIRED:
        return "key_spending_not_allowed"
    if status == TavernChestStatus.GEM_REQUIRED:
        return "gem_spending_not_allowed"
    if status == TavernChestStatus.PAID:
        return "paid_not_allowed"
    return "not_free"


def _chest_postcondition_verified(
    before: TavernChestObservation,
    verification: TavernChestObservation,
) -> bool:
    if before.normalized_chest_type() != verification.normalized_chest_type():
        return False
    if not verification.scene_verified:
        return False
    status = verification.normalized_status()
    if status in {TavernChestStatus.COOLDOWN, TavernChestStatus.UNAVAILABLE}:
        return True
    return status != TavernChestStatus.FREE and not verification.free_indicator_visible


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
        action_type=f"city_tavern_chest.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _TavernChestState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _TavernChestState) -> TavernChestPolicy:
    if state.policy is None:
        raise RuntimeError("Tavern chest policy has not been validated.")
    return state.policy


def _require_scan(state: _TavernChestState) -> TavernChestScan:
    if state.scan is None:
        raise RuntimeError("Tavern chests have not been scanned.")
    return state.scan


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_manual_stop(state: _TavernChestState) -> bool:
    text = state.terminal_reason.lower()
    return "verification" in text or "confirmation" in text or "manual" in text
