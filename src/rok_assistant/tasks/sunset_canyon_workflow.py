from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
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


SUNSET_CANYON_WORKFLOW_KEY = "sunset-canyon"
SUNSET_CANYON_TEMPLATE_KEYS = (
    "city.campaign.button",
    "campaign.sunset_canyon.entry",
    "sunset_canyon.scene",
    "sunset_canyon.remaining_attempts",
    "sunset_canyon.opponent",
    "sunset_canyon.challenge",
    "sunset_canyon.skip_battle",
    "sunset_canyon.result.victory",
    "sunset_canyon.result.defeat",
    "sunset_canyon.result.close",
)
SUNSET_CANYON_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "open_campaign",
    "open_sunset_canyon",
    "inspect_attempts",
    "fight_battles",
    "complete",
    "skipped",
    "recover",
    "failed",
    "cancelled",
)


class SunsetCanyonOpponentRule(StrEnum):
    FIRST_AVAILABLE = "FIRST_AVAILABLE"
    LOWEST_POWER = "LOWEST_POWER"
    HIGHEST_CONFIDENCE = "HIGHEST_CONFIDENCE"


class SunsetCanyonScanStatus(StrEnum):
    READY = "READY"
    NO_ATTEMPTS = "NO_ATTEMPTS"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


class SunsetCanyonBattleOutcome(StrEnum):
    VICTORY = "VICTORY"
    DEFEAT = "DEFEAT"
    UNKNOWN = "UNKNOWN"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"


def _opponent_rule(value: SunsetCanyonOpponentRule | str) -> SunsetCanyonOpponentRule:
    if isinstance(value, SunsetCanyonOpponentRule):
        return value
    try:
        return SunsetCanyonOpponentRule(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in SunsetCanyonOpponentRule)
        raise ValueError(f"Invalid Sunset Canyon opponent rule: {value!r}. Expected one of: {valid}.") from exc


def _scan_status(value: SunsetCanyonScanStatus | str) -> SunsetCanyonScanStatus:
    if isinstance(value, SunsetCanyonScanStatus):
        return value
    try:
        return SunsetCanyonScanStatus(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in SunsetCanyonScanStatus)
        raise ValueError(f"Invalid Sunset Canyon scan status: {value!r}. Expected one of: {valid}.") from exc


def _battle_outcome(value: SunsetCanyonBattleOutcome | str) -> SunsetCanyonBattleOutcome:
    if isinstance(value, SunsetCanyonBattleOutcome):
        return value
    try:
        return SunsetCanyonBattleOutcome(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in SunsetCanyonBattleOutcome)
        raise ValueError(f"Invalid Sunset Canyon battle outcome: {value!r}. Expected one of: {valid}.") from exc


def _require_confidence(value: float, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    return numeric


@dataclass(frozen=True)
class SunsetCanyonPolicy:
    max_battles_per_run: int = 5
    opponent_rule: SunsetCanyonOpponentRule | str = SunsetCanyonOpponentRule.LOWEST_POWER
    allow_skip_battle: bool = False
    allow_formation_changes: bool = False
    minimum_opponent_confidence: float = 0.80

    def normalized(self) -> SunsetCanyonPolicy:
        if self.max_battles_per_run < 0:
            raise ValueError("max_battles_per_run must be zero or greater.")
        if self.allow_formation_changes:
            raise ValueError("Formation mutation is outside PVP-001 conservative Sunset Canyon scope.")
        return SunsetCanyonPolicy(
            max_battles_per_run=int(self.max_battles_per_run),
            opponent_rule=_opponent_rule(self.opponent_rule),
            allow_skip_battle=bool(self.allow_skip_battle),
            allow_formation_changes=False,
            minimum_opponent_confidence=_require_confidence(
                self.minimum_opponent_confidence,
                "minimum_opponent_confidence",
            ),
        )

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "max_battles_per_run": normalized.max_battles_per_run,
            "opponent_rule": normalized.opponent_rule.value,
            "allow_skip_battle": normalized.allow_skip_battle,
            "allow_formation_changes": normalized.allow_formation_changes,
            "minimum_opponent_confidence": normalized.minimum_opponent_confidence,
        }


@dataclass(frozen=True)
class SunsetCanyonRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: SunsetCanyonPolicy = field(default_factory=SunsetCanyonPolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class SunsetCanyonConfig:
    workflow_timeout_seconds: float = 180.0
    step_timeout_seconds: float = 20.0
    retry_delay_seconds: float = 0.0
    precondition_retry_limit: int = 1
    navigation_retry_limit: int = 1
    battle_retry_limit: int = 0


@dataclass(frozen=True)
class SunsetCanyonOpponent:
    slot: int
    name: str = ""
    power: int | None = None
    rank: int | None = None
    confidence: float = 1.0
    target: tuple[int, int] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "name": self.name,
            "power": self.power,
            "rank": self.rank,
            "confidence": self.confidence,
            "target": list(self.target) if self.target is not None else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SunsetCanyonStateScan:
    status: SunsetCanyonScanStatus | str
    remaining_attempts: int | None = None
    opponents: tuple[SunsetCanyonOpponent, ...] = ()
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_status(self) -> SunsetCanyonScanStatus:
        return _scan_status(self.status)

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.normalized_status().value,
            "remaining_attempts": self.remaining_attempts,
            "opponents": [opponent.to_json() for opponent in self.opponents],
            "message": self.message,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class SunsetCanyonBattleStartResult:
    success: bool
    changed: bool = False
    retryable: bool = False
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "changed": self.changed,
            "retryable": self.retryable,
            "message": self.message,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class SunsetCanyonBattleReport:
    outcome: SunsetCanyonBattleOutcome | str
    handled: bool = True
    remaining_attempts: int | None = None
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_outcome(self) -> SunsetCanyonBattleOutcome:
        return _battle_outcome(self.outcome)

    def to_json(self) -> dict[str, object]:
        return {
            "outcome": self.normalized_outcome().value,
            "handled": self.handled,
            "remaining_attempts": self.remaining_attempts,
            "message": self.message,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


class SunsetCanyonDriver(Protocol):
    def open_campaign(
        self,
        request: SunsetCanyonRequest,
        character: Character,
        policy: SunsetCanyonPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def open_sunset_canyon(
        self,
        request: SunsetCanyonRequest,
        character: Character,
        policy: SunsetCanyonPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def inspect_sunset_canyon(
        self,
        request: SunsetCanyonRequest,
        character: Character,
        policy: SunsetCanyonPolicy,
    ) -> SunsetCanyonStateScan:
        ...

    def select_opponent(
        self,
        request: SunsetCanyonRequest,
        character: Character,
        opponent: SunsetCanyonOpponent,
        policy: SunsetCanyonPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def start_battle(
        self,
        request: SunsetCanyonRequest,
        character: Character,
        opponent: SunsetCanyonOpponent,
        policy: SunsetCanyonPolicy,
    ) -> SunsetCanyonBattleStartResult:
        ...

    def skip_battle(
        self,
        request: SunsetCanyonRequest,
        character: Character,
        opponent: SunsetCanyonOpponent,
        policy: SunsetCanyonPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def collect_battle_result(
        self,
        request: SunsetCanyonRequest,
        character: Character,
        opponent: SunsetCanyonOpponent,
        policy: SunsetCanyonPolicy,
    ) -> SunsetCanyonBattleReport:
        ...


class SunsetCanyonAccountPrecondition(Protocol):
    def ensure_account(self, request: SunsetCanyonRequest, character: Character) -> ResourceGatheringActionResult:
        ...


class SunsetCanyonCharacterPrecondition(Protocol):
    def ensure_character(self, request: SunsetCanyonRequest, character: Character) -> ResourceGatheringActionResult:
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
class _SunsetCanyonRuntimeState:
    request: SunsetCanyonRequest
    character: Character | None = None
    policy: SunsetCanyonPolicy | None = None
    initial_scan: SunsetCanyonStateScan | None = None
    latest_remaining_attempts: int | None = None
    selected_opponents: list[dict[str, object]] = field(default_factory=list)
    battle_results: list[dict[str, object]] = field(default_factory=list)
    popup_handling: list[dict[str, object]] = field(default_factory=list)
    skipped_skip_battle: list[dict[str, object]] = field(default_factory=list)
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


class SunsetCanyonWorkflow:
    def __init__(
        self,
        *,
        characters: object,
        driver: SunsetCanyonDriver,
        account_precondition: SunsetCanyonAccountPrecondition | None = None,
        character_precondition: SunsetCanyonCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: object | None = None,
        step_runs: object | None = None,
        incidents: object | None = None,
        config: SunsetCanyonConfig | None = None,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.account_precondition = account_precondition
        self.character_precondition = character_precondition
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.incidents = incidents
        self.config = config or SunsetCanyonConfig()
        self._states: dict[str, _SunsetCanyonRuntimeState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return SUNSET_CANYON_STATES

    def execute(
        self,
        request: SunsetCanyonRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _SunsetCanyonRuntimeState(request=request)
        self._states[token] = state
        persistence = None
        if self.job_runs is not None and self.step_runs is not None and request.job_id is not None:
            persistence = WorkflowRunRepositoryRecorder(self.job_runs, self.step_runs)
        try:
            policy_limit = max(0, int(request.policy.max_battles_per_run))
            context = WorkflowExecutionContext(
                cancellation_token=cancellation_token or CancellationToken(),
                deadline=WorkflowDeadline.from_timeout(
                    self.config.workflow_timeout_seconds,
                    time.monotonic,
                ),
                budget=StepBudget(max_steps=len(SUNSET_CANYON_STATES) + (policy_limit * 5) + 10),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"sunset-canyon:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"sunset_canyon_run_id": token},
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
        for state in SUNSET_CANYON_STATES:
            registry.register(f"sunset_canyon.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.precondition_retry_limit,
            "ensure_character": self.config.precondition_retry_limit,
            "ensure_game_running": self.config.precondition_retry_limit,
            "open_campaign": self.config.navigation_retry_limit,
            "open_sunset_canyon": self.config.navigation_retry_limit,
            "fight_battles": self.config.battle_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=SUNSET_CANYON_WORKFLOW_KEY,
            name="Fight Sunset Canyon",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"sunset_canyon.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in SUNSET_CANYON_STATES
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
            if state_name == "skipped":
                return self._skipped(step)
            if state_name == "cancelled":
                return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
            if state.stopped and state_name not in {"recover", "complete"}:
                return _step_result(
                    step.step_key,
                    WorkflowOutcome.SKIPPED,
                    data={"skipped_after_terminal_state": state.terminal_state},
                )
            if state_name == "complete":
                return self._complete(step, state)
            if state_name == "recover":
                return self._recover(step, state, context)
            method = getattr(self, f"_{state_name}")
            return method(step, state, context)

        return handler

    def _validate_input(
        self,
        step: WorkflowStepSpec,
        state: _SunsetCanyonRuntimeState,
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
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"policy": state.policy.to_json()})

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _SunsetCanyonRuntimeState,
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
        state: _SunsetCanyonRuntimeState,
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
        state: _SunsetCanyonRuntimeState,
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
        state: _SunsetCanyonRuntimeState,
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

    def _open_campaign(
        self,
        step: WorkflowStepSpec,
        state: _SunsetCanyonRuntimeState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.open_campaign(state.request, _require_character(state), _require_policy(state)),
            fallback_message="Campaign could not be opened.",
        )

    def _open_sunset_canyon(
        self,
        step: WorkflowStepSpec,
        state: _SunsetCanyonRuntimeState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.open_sunset_canyon(state.request, _require_character(state), _require_policy(state)),
            fallback_message="Sunset Canyon could not be opened.",
        )

    def _inspect_attempts(
        self,
        step: WorkflowStepSpec,
        state: _SunsetCanyonRuntimeState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        scan = self.driver.inspect_sunset_canyon(state.request, _require_character(state), _require_policy(state))
        state.initial_scan = scan
        if scan.screenshot_path:
            state.screenshot_path = scan.screenshot_path
        state.latest_remaining_attempts = scan.remaining_attempts
        status = scan.normalized_status()
        if status == SunsetCanyonScanStatus.NO_ATTEMPTS or (scan.remaining_attempts is not None and scan.remaining_attempts <= 0):
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                scan.message or "No Sunset Canyon attempts remain.",
                screenshot_path=scan.screenshot_path,
                data={"scan": scan.to_json()},
            )
        if status == SunsetCanyonScanStatus.VERIFICATION_REQUIRED:
            return state.stop(
                step.step_key,
                WorkflowOutcome.FATAL_FAILURE,
                scan.message or "Verification screen requires manual intervention.",
                screenshot_path=scan.screenshot_path,
                data={"scan": scan.to_json()},
            )
        if status != SunsetCanyonScanStatus.READY:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                scan.message or "Sunset Canyon attempts could not be verified.",
                screenshot_path=scan.screenshot_path,
                data={"scan": scan.to_json()},
            )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"scan": scan.to_json()}, screenshot_path=scan.screenshot_path)

    def _fight_battles(
        self,
        step: WorkflowStepSpec,
        state: _SunsetCanyonRuntimeState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        character = _require_character(state)
        policy = _require_policy(state)
        scan = _require_scan(state)
        remaining = scan.remaining_attempts if scan.remaining_attempts is not None else policy.max_battles_per_run
        battle_limit = min(max(0, remaining), policy.max_battles_per_run)
        if battle_limit <= 0:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                "Sunset Canyon policy limit is zero.",
                data={"policy_limit": policy.max_battles_per_run, "remaining_attempts": remaining},
            )
        available = _eligible_opponents(scan.opponents, policy)
        if not available:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                "No selectable Sunset Canyon opponent met policy requirements.",
                screenshot_path=scan.screenshot_path,
                data={"scan": scan.to_json()},
            )
        for battle_number in range(1, battle_limit + 1):
            context.cancellation_token.throw_if_cancelled()
            opponent = _select_opponent(available, policy)
            state.selected_opponents.append({"battle_number": battle_number, **opponent.to_json()})
            select_result = self.driver.select_opponent(state.request, character, opponent, policy)
            if select_result.screenshot_path:
                state.screenshot_path = select_result.screenshot_path
            if not select_result.success:
                return self._battle_failure(
                    step,
                    state,
                    select_result.message or "Sunset Canyon opponent could not be selected.",
                    screenshot_path=select_result.screenshot_path,
                    data={"battle_number": battle_number, "opponent": opponent.to_json(), **select_result.data},
                    retryable=select_result.retryable,
                )
            start_result = self.driver.start_battle(state.request, character, opponent, policy)
            if start_result.screenshot_path:
                state.screenshot_path = start_result.screenshot_path
            if not start_result.success or not start_result.changed:
                return self._battle_failure(
                    step,
                    state,
                    start_result.message or "Sunset Canyon battle start was not verified.",
                    screenshot_path=start_result.screenshot_path,
                    data={"battle_number": battle_number, "opponent": opponent.to_json(), "start": start_result.to_json()},
                    retryable=start_result.retryable,
                )
            if policy.allow_skip_battle:
                skip_result = self.driver.skip_battle(state.request, character, opponent, policy)
                state.skipped_skip_battle.append(
                    {
                        "battle_number": battle_number,
                        "attempted": True,
                        "success": skip_result.success,
                        "message": skip_result.message,
                        "screenshot_path": skip_result.screenshot_path,
                        **skip_result.data,
                    }
                )
                if skip_result.screenshot_path:
                    state.screenshot_path = skip_result.screenshot_path
                if not skip_result.success:
                    return self._battle_failure(
                        step,
                        state,
                        skip_result.message or "Sunset Canyon skip battle action failed.",
                        screenshot_path=skip_result.screenshot_path,
                        data={"battle_number": battle_number, "opponent": opponent.to_json(), **skip_result.data},
                        retryable=skip_result.retryable,
                    )
            else:
                state.skipped_skip_battle.append({"battle_number": battle_number, "attempted": False, "reason": "policy_disabled"})
            report = self.driver.collect_battle_result(state.request, character, opponent, policy)
            if report.screenshot_path:
                state.screenshot_path = report.screenshot_path
            state.popup_handling.append(
                {
                    "battle_number": battle_number,
                    "handled": report.handled,
                    "screenshot_path": report.screenshot_path,
                    "message": report.message,
                }
            )
            outcome = report.normalized_outcome()
            if outcome == SunsetCanyonBattleOutcome.VERIFICATION_REQUIRED:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.FATAL_FAILURE,
                    report.message or "Verification screen requires manual intervention.",
                    screenshot_path=report.screenshot_path,
                    data={"battle_number": battle_number, "result": report.to_json()},
                )
            if outcome == SunsetCanyonBattleOutcome.UNKNOWN or not report.handled:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    report.message or "Sunset Canyon result popup could not be handled.",
                    screenshot_path=report.screenshot_path,
                    data={"battle_number": battle_number, "result": report.to_json()},
                )
            if report.remaining_attempts is not None:
                state.latest_remaining_attempts = report.remaining_attempts
            else:
                state.latest_remaining_attempts = max(0, (state.latest_remaining_attempts or remaining) - 1)
            state.battle_results.append(
                {
                    "battle_number": battle_number,
                    "opponent": opponent.to_json(),
                    "start": start_result.to_json(),
                    "result": report.to_json(),
                    "remaining_attempts": state.latest_remaining_attempts,
                }
            )
            if state.latest_remaining_attempts <= 0:
                break
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=self._sunset_payload(state), screenshot_path=state.screenshot_path)

    def _complete(self, step: WorkflowStepSpec, state: _SunsetCanyonRuntimeState) -> WorkflowStepResult:
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
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=self._sunset_payload(state))

    def _skipped(self, step: WorkflowStepSpec) -> WorkflowStepResult:
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED)

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _SunsetCanyonRuntimeState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if _is_manual_stop(state):
            state.recovery_outcome = {"attempted": False, "reason": "manual_intervention_required"}
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
        state.recovery_outcome = self._monitor_recovery(state, _job_run_id(context))
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _SunsetCanyonRuntimeState) -> WorkflowStepResult:
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
                **self._sunset_payload(state),
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _SunsetCanyonRuntimeState,
        action: ResourceGatheringActionResult,
        *,
        fallback_message: str = "Sunset Canyon action failed.",
    ) -> WorkflowStepResult:
        if action.screenshot_path:
            state.screenshot_path = action.screenshot_path
        if action.success:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=action.data, screenshot_path=action.screenshot_path)
        if action.retryable:
            return _step_result(
                step.step_key,
                WorkflowOutcome.RETRYABLE_FAILURE,
                action.message or fallback_message,
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            action.message or fallback_message,
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _battle_failure(
        self,
        step: WorkflowStepSpec,
        state: _SunsetCanyonRuntimeState,
        message: str,
        *,
        screenshot_path: str = "",
        data: dict[str, object] | None = None,
        retryable: bool = False,
    ) -> WorkflowStepResult:
        if retryable:
            return _step_result(step.step_key, WorkflowOutcome.RETRYABLE_FAILURE, message, data=data, screenshot_path=screenshot_path)
        return state.stop(step.step_key, WorkflowOutcome.BLOCKED, message, screenshot_path=screenshot_path, data=data)

    def _sunset_payload(self, state: _SunsetCanyonRuntimeState) -> dict[str, object]:
        return {
            "initial_scan": state.initial_scan.to_json() if state.initial_scan is not None else {},
            "selected_opponents": state.selected_opponents,
            "battle_results": state.battle_results,
            "popup_handling": state.popup_handling,
            "skip_battle": state.skipped_skip_battle,
            "battles_started": len(state.battle_results),
            "latest_remaining_attempts": state.latest_remaining_attempts,
            "failure_evidence": {
                "screenshot_path": state.screenshot_path,
                "terminal_state": state.terminal_state,
                "terminal_reason": state.terminal_reason,
            },
        }

    def _state_from_context(self, context: WorkflowExecutionContext) -> _SunsetCanyonRuntimeState:
        token = str(context.metadata.get("sunset_canyon_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("Sunset Canyon runtime state is missing.") from exc

    def _open_incident(self, state: _SunsetCanyonRuntimeState) -> None:
        if self.incidents is None or not state.failed or state.incident_opened:
            return
        self.incidents.save(
            Incident(
                incident_key=f"sunset-canyon:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="Sunset Canyon workflow blocked",
                details=state.terminal_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )
        state.incident_opened = True

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _SunsetCanyonRuntimeState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "Sunset Canyon workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(
        self,
        result: WorkflowExecutionResult,
        state: _SunsetCanyonRuntimeState,
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
        state: _SunsetCanyonRuntimeState,
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
        state: _SunsetCanyonRuntimeState,
    ) -> None:
        if result.outcome == WorkflowOutcome.RETRYABLE_FAILURE and state.terminal_state:
            state.terminal_outcome = WorkflowOutcome.BLOCKED
        if state.terminal_outcome in {WorkflowOutcome.SKIPPED, WorkflowOutcome.BLOCKED}:
            result.outcome = state.terminal_outcome
            result.message = state.terminal_reason
        result.result = {
            **dict(result.result),
            "policy": state.policy.to_json() if state.policy is not None else {},
            **self._sunset_payload(state),
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
        state: _SunsetCanyonRuntimeState,
    ) -> None:
        if self.job_runs is None or result.job_run_id is None:
            return
        run: JobRun | None = self.job_runs.get(result.job_run_id)
        if run is None:
            return
        run.status = "completed" if result.outcome == WorkflowOutcome.SKIPPED else ("failed" if result.outcome.is_failure else "completed")
        run.result_json = json.dumps(result.to_json_dict(), sort_keys=True)
        run.error_message = state.terminal_reason if result.outcome.is_failure else ""
        run.screenshot_path = state.screenshot_path
        self.job_runs.save(run)


def _eligible_opponents(
    opponents: tuple[SunsetCanyonOpponent, ...],
    policy: SunsetCanyonPolicy,
) -> list[SunsetCanyonOpponent]:
    return [
        opponent
        for opponent in opponents
        if opponent.confidence >= policy.minimum_opponent_confidence
    ]


def _select_opponent(
    opponents: list[SunsetCanyonOpponent],
    policy: SunsetCanyonPolicy,
) -> SunsetCanyonOpponent:
    rule = _opponent_rule(policy.opponent_rule)
    if rule == SunsetCanyonOpponentRule.LOWEST_POWER:
        return min(opponents, key=lambda opponent: (opponent.power is None, opponent.power or 0, opponent.slot))
    if rule == SunsetCanyonOpponentRule.HIGHEST_CONFIDENCE:
        return max(opponents, key=lambda opponent: (opponent.confidence, -opponent.slot))
    return sorted(opponents, key=lambda opponent: opponent.slot)[0]


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
        action_type=f"sunset_canyon.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _SunsetCanyonRuntimeState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _SunsetCanyonRuntimeState) -> SunsetCanyonPolicy:
    if state.policy is None:
        raise RuntimeError("Sunset Canyon policy has not been validated.")
    return state.policy


def _require_scan(state: _SunsetCanyonRuntimeState) -> SunsetCanyonStateScan:
    if state.initial_scan is None:
        raise RuntimeError("Sunset Canyon attempts have not been inspected.")
    return state.initial_scan


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_manual_stop(state: _SunsetCanyonRuntimeState) -> bool:
    text = state.terminal_reason.lower()
    return "verification" in text or "manual" in text
