from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from rok_assistant.db.models import Character, JobRun, March
from rok_assistant.tasks.resource_search_workflow import (
    MarchAvailability,
    MarchDispatchResult,
    ResourceGatheringActionResult,
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


MAP_FOG_EXPLORATION_WORKFLOW_KEY = "map-fog-exploration"
MAP_FOG_EXPLORATION_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "validate_idle_scout",
    "scan_for_fog_target",
    "investigate_discovery",
    "dispatch_scout",
    "verify_scout_busy",
    "complete",
    "cancelled",
)


class FogTargetStatus(StrEnum):
    FOG = "FOG"
    CAVE = "CAVE"
    VILLAGE = "VILLAGE"
    NOT_FOUND = "NOT_FOUND"
    INVALID = "INVALID"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"


@dataclass(frozen=True)
class FogScoutPolicy:
    max_scans: int = 8
    total_deadline_seconds: float = 120.0
    scan_radius: int | None = None
    scout_preset: str = "default"
    minimum_confidence: float = 0.80
    investigate_caves: bool = False
    investigate_villages: bool = False

    def normalized(self) -> FogScoutPolicy:
        _require_positive_int(self.max_scans, "FogScoutPolicy.max_scans")
        if self.scan_radius is not None:
            _require_positive_int(self.scan_radius, "FogScoutPolicy.scan_radius")
        if (
            not isinstance(self.total_deadline_seconds, int | float)
            or isinstance(self.total_deadline_seconds, bool)
            or not math.isfinite(float(self.total_deadline_seconds))
            or self.total_deadline_seconds <= 0
        ):
            raise ValueError("FogScoutPolicy.total_deadline_seconds must be positive.")
        _require_confidence(self.minimum_confidence, "FogScoutPolicy.minimum_confidence")
        preset = str(self.scout_preset).strip()
        if not preset:
            raise ValueError("FogScoutPolicy.scout_preset must be configured.")
        return FogScoutPolicy(
            max_scans=int(self.max_scans),
            total_deadline_seconds=float(self.total_deadline_seconds),
            scan_radius=self.scan_radius,
            scout_preset=preset,
            minimum_confidence=float(self.minimum_confidence),
            investigate_caves=bool(self.investigate_caves),
            investigate_villages=bool(self.investigate_villages),
        )

    def allows(self, target: FogTargetStatus) -> bool:
        if target == FogTargetStatus.FOG:
            return True
        if target == FogTargetStatus.CAVE:
            return self.investigate_caves
        if target == FogTargetStatus.VILLAGE:
            return self.investigate_villages
        return False

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "max_scans": normalized.max_scans,
            "total_deadline_seconds": normalized.total_deadline_seconds,
            "scan_radius": normalized.scan_radius,
            "scout_preset": normalized.scout_preset,
            "minimum_confidence": normalized.minimum_confidence,
            "investigate_caves": normalized.investigate_caves,
            "investigate_villages": normalized.investigate_villages,
        }


@dataclass(frozen=True)
class FogExplorationRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: FogScoutPolicy = field(default_factory=FogScoutPolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class FogExplorationConfig:
    workflow_timeout_seconds: float = 180.0
    step_timeout_seconds: float = 20.0
    precondition_retry_limit: int = 1
    dispatch_retry_limit: int = 1
    retry_delay_seconds: float = 0.25


@dataclass(frozen=True)
class FogTargetObservation:
    status: FogTargetStatus | str
    target_id: str = ""
    confidence: float = 0.0
    x: int | None = None
    y: int | None = None
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_status(self) -> FogTargetStatus:
        if isinstance(self.status, FogTargetStatus):
            return self.status
        try:
            return FogTargetStatus(str(self.status).strip().upper())
        except ValueError as exc:
            valid = ", ".join(item.value for item in FogTargetStatus)
            raise ValueError(f"Invalid fog target status: {self.status!r}. Expected one of: {valid}.") from exc

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.normalized_status().value,
            "target_id": self.target_id,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            **self.data,
        }


class CharacterRepository(Protocol):
    def get(self, character_id: int) -> Character | None:
        ...


class MarchRepository(Protocol):
    def list_for_character(self, character_id: int) -> list[March]:
        ...

    def save(self, march: March) -> int:
        ...


class JobRunRepository(Protocol):
    def get(self, run_id: int) -> JobRun | None:
        ...

    def save(self, run: JobRun) -> int:
        ...


class StepRunRepository(Protocol):
    ...


class FogAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: FogExplorationRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class FogCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: FogExplorationRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class FogExplorationDriver(Protocol):
    def validate_idle_scout(
        self,
        request: FogExplorationRequest,
        policy: FogScoutPolicy,
    ) -> MarchAvailability:
        ...

    def scan_for_fog_target(
        self,
        request: FogExplorationRequest,
        policy: FogScoutPolicy,
        scan_index: int,
    ) -> FogTargetObservation:
        ...

    def investigate_discovery(
        self,
        request: FogExplorationRequest,
        target: FogTargetObservation,
        policy: FogScoutPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def dispatch_scout(
        self,
        request: FogExplorationRequest,
        target: FogTargetObservation,
        scout: MarchAvailability,
        policy: FogScoutPolicy,
    ) -> MarchDispatchResult:
        ...

    def verify_scout_busy(
        self,
        request: FogExplorationRequest,
        target: FogTargetObservation,
        dispatch: MarchDispatchResult,
    ) -> ResourceGatheringActionResult:
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
class _FogState:
    request: FogExplorationRequest
    character: Character | None = None
    policy: FogScoutPolicy | None = None
    scout: MarchAvailability | None = None
    target: FogTargetObservation | None = None
    dispatch: MarchDispatchResult | None = None
    scan_attempts: list[dict[str, object]] = field(default_factory=list)
    terminal_outcome: WorkflowOutcome | None = None
    terminal_state: str = ""
    terminal_reason: str = ""
    recovery_outcome: dict[str, object] = field(default_factory=dict)
    screenshot_path: str = ""

    @property
    def stopped(self) -> bool:
        return self.terminal_outcome is not None

    @property
    def failed(self) -> bool:
        return self.terminal_outcome is not None and self.terminal_outcome.is_failure

    def stop(
        self,
        step_key: str,
        outcome: WorkflowOutcome,
        reason: str,
        *,
        screenshot_path: str = "",
        data: dict[str, object] | None = None,
    ) -> WorkflowStepResult:
        self.terminal_outcome = outcome
        self.terminal_state = step_key
        self.terminal_reason = reason
        if screenshot_path:
            self.screenshot_path = screenshot_path
        return _step_result(
            step_key,
            outcome,
            reason,
            data={
                "terminal_outcome": outcome.value,
                "terminal_state": step_key,
                "terminal_reason": reason,
                **(data or {}),
            },
            screenshot_path=screenshot_path,
        )


class FogExplorationWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: FogExplorationDriver,
        account_precondition: FogAccountPrecondition | None = None,
        character_precondition: FogCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        marches: MarchRepository | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        config: FogExplorationConfig | None = None,
        clock: object = time.monotonic,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.account_precondition = account_precondition
        self.character_precondition = character_precondition
        self.recovery_watchdog = recovery_watchdog
        self.marches = marches
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.config = config or FogExplorationConfig()
        self.clock = clock
        self._states: dict[str, _FogState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return MAP_FOG_EXPLORATION_STATES

    def execute(
        self,
        request: FogExplorationRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _FogState(request=request)
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
                budget=StepBudget(max_steps=len(MAP_FOG_EXPLORATION_STATES) + request.policy.max_scans + 6),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"map-fog:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"map_fog_exploration_run_id": token},
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
        for state in MAP_FOG_EXPLORATION_STATES:
            registry.register(f"map_fog.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.precondition_retry_limit,
            "ensure_character": self.config.precondition_retry_limit,
            "ensure_game_running": self.config.precondition_retry_limit,
            "dispatch_scout": self.config.dispatch_retry_limit,
            "verify_scout_busy": self.config.dispatch_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=MAP_FOG_EXPLORATION_WORKFLOW_KEY,
            name="Explore Fog",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"map_fog.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in MAP_FOG_EXPLORATION_STATES
            ],
        )

    def _handler_for(self, state_name: str):
        def handler(
            context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            state = self._state_from_context(context)
            if state_name == "cancelled":
                return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
            if state.stopped and state_name != "complete":
                return _step_result(
                    step.step_key,
                    WorkflowOutcome.SKIPPED,
                    data={"skipped_after_terminal_state": state.terminal_state},
                )
            if state_name == "complete":
                return self._complete(step, state)
            method = getattr(self, f"_{state_name}")
            return method(step, state, context)

        return handler

    def _validate_input(
        self,
        step: WorkflowStepSpec,
        state: _FogState,
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
        state: _FogState,
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
        state: _FogState,
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
        state: _FogState,
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
        state: _FogState,
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

    def _validate_idle_scout(
        self,
        step: WorkflowStepSpec,
        state: _FogState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        policy = _require_policy(state)
        scout = self.driver.validate_idle_scout(state.request, policy)
        state.scout = scout
        if scout.screenshot_path:
            state.screenshot_path = scout.screenshot_path
        if not scout.available:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                scout.message or "No idle scout is available for fog exploration.",
                screenshot_path=scout.screenshot_path,
                data={
                    "available_count": scout.available_count,
                    "scout_slot": scout.march_slot,
                    **scout.data,
                },
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "available_count": scout.available_count,
                "scout_slot": scout.march_slot,
                **scout.data,
            },
            screenshot_path=scout.screenshot_path,
        )

    def _scan_for_fog_target(
        self,
        step: WorkflowStepSpec,
        state: _FogState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        policy = _require_policy(state)
        started_at = self.clock()
        for scan_index in range(1, policy.max_scans + 1):
            context.cancellation_token.throw_if_cancelled()
            if self.clock() - started_at >= policy.total_deadline_seconds:
                break
            target = self.driver.scan_for_fog_target(state.request, policy, scan_index)
            self._record_scan_attempt(state, scan_index, target)
            if target.screenshot_path:
                state.screenshot_path = target.screenshot_path
            status = target.normalized_status()
            if status == FogTargetStatus.VERIFICATION_REQUIRED:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    target.message or "Verification screen requires manual intervention.",
                    screenshot_path=target.screenshot_path,
                    data={"scan_attempts": state.scan_attempts},
                )
            if _target_is_valid(target, policy):
                state.target = target
                return _step_result(
                    step.step_key,
                    WorkflowOutcome.SUCCESS,
                    data={"target": target.to_json(), "scan_attempts": state.scan_attempts},
                    screenshot_path=target.screenshot_path,
                )
        return state.stop(
            step.step_key,
            WorkflowOutcome.RETRYABLE_FAILURE,
            "No valid fog target was found before the scan or deadline bound.",
            screenshot_path=state.screenshot_path,
            data={"scan_attempts": state.scan_attempts},
        )

    def _investigate_discovery(
        self,
        step: WorkflowStepSpec,
        state: _FogState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        target = _require_target(state)
        status = target.normalized_status()
        if status not in {FogTargetStatus.CAVE, FogTargetStatus.VILLAGE}:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"reason": "not_discovery_target"})
        return self._action_to_step(
            step,
            state,
            self.driver.investigate_discovery(state.request, target, _require_policy(state)),
        )

    def _dispatch_scout(
        self,
        step: WorkflowStepSpec,
        state: _FogState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        target = _require_target(state)
        scout = _require_scout(state)
        dispatch = self.driver.dispatch_scout(state.request, target, scout, _require_policy(state))
        state.dispatch = dispatch
        if dispatch.screenshot_path:
            state.screenshot_path = dispatch.screenshot_path
        if not dispatch.success:
            return (
                _step_result(
                    step.step_key,
                    WorkflowOutcome.RETRYABLE_FAILURE,
                    dispatch.message or "Fog exploration scout dispatch failed.",
                    data=dispatch.data,
                    screenshot_path=dispatch.screenshot_path,
                )
                if dispatch.retryable
                else state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    dispatch.message or "Fog exploration scout dispatch failed.",
                    screenshot_path=dispatch.screenshot_path,
                    data=dispatch.data,
                )
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"dispatch": _dispatch_json(dispatch), "scout_preset": _require_policy(state).scout_preset},
            screenshot_path=dispatch.screenshot_path,
        )

    def _verify_scout_busy(
        self,
        step: WorkflowStepSpec,
        state: _FogState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        result = self.driver.verify_scout_busy(state.request, _require_target(state), _require_dispatch(state))
        step_result = self._action_to_step(step, state, result)
        if step_result.outcome == WorkflowOutcome.SUCCESS:
            self._persist_scout_dispatch(state)
        return step_result

    def _complete(self, step: WorkflowStepSpec, state: _FogState) -> WorkflowStepResult:
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
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"target": _target_json(state.target), "dispatch": _dispatch_json(state.dispatch)},
        )

    def _record_scan_attempt(
        self,
        state: _FogState,
        scan_index: int,
        target: FogTargetObservation,
    ) -> None:
        state.scan_attempts.append(
            {
                "scan_index": scan_index,
                "status": target.normalized_status().value,
                "target_id": target.target_id,
                "confidence": target.confidence,
                "x": target.x,
                "y": target.y,
                "message": target.message,
                **target.data,
            }
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _FogState,
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
                action.message or "Fog exploration action failed.",
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            action.message or "Fog exploration action failed.",
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _persist_scout_dispatch(self, state: _FogState) -> None:
        dispatch = _require_dispatch(state)
        if self.marches is None or dispatch.march_slot is None:
            return
        character = _require_character(state)
        marches = self.marches.list_for_character(int(character.id or 0))
        existing = next((item for item in marches if item.march_slot == dispatch.march_slot), None)
        march = existing or March(character_id=character.id, march_slot=dispatch.march_slot)
        march.status = "fog_exploration"
        march.expected_return_time = dispatch.expected_return_time
        march.next_action_time = dispatch.expected_return_time
        self.marches.save(march)

    def _state_from_context(self, context: WorkflowExecutionContext) -> _FogState:
        token = str(context.metadata.get("map_fog_exploration_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("Fog exploration runtime state is missing.") from exc

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _FogState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "Fog exploration workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(
        self,
        result: WorkflowExecutionResult,
        state: _FogState,
    ) -> None:
        if state.terminal_outcome != WorkflowOutcome.BLOCKED or state.recovery_outcome:
            return
        if self.recovery_watchdog is None:
            state.recovery_outcome = {"attempted": False, "reason": "watchdog_not_configured"}
            return
        recovery = self.recovery_watchdog.monitor(
            instance_id=state.request.instance_id,
            instance_index=state.request.instance_index,
            instance_name=state.request.instance_name,
            job_run_id=result.job_run_id,
        )
        state.recovery_outcome = {
            "attempted": bool(getattr(recovery, "recovery_attempted", False)),
            "healthy": bool(getattr(recovery, "healthy", False)),
            "circuit_opened": bool(getattr(recovery, "circuit_opened", False)),
        }

    def _augment_result(
        self,
        result: WorkflowExecutionResult,
        state: _FogState,
    ) -> None:
        result.result = {
            **dict(result.result),
            "policy": state.policy.to_json() if state.policy is not None else {},
            "scan_attempts": state.scan_attempts,
            "target": _target_json(state.target),
            "scout": _scout_json(state.scout),
            "scout_dispatch": _dispatch_json(state.dispatch),
            "terminal_state": state.terminal_state,
            "terminal_reason": state.terminal_reason,
            "failure_state": state.terminal_state if result.outcome.is_failure else "",
            "failure_reason": state.terminal_reason if result.outcome.is_failure else "",
            "recovery_outcome": state.recovery_outcome,
        }

    def _update_persisted_run(
        self,
        result: WorkflowExecutionResult,
        state: _FogState,
    ) -> None:
        if self.job_runs is None or result.job_run_id is None:
            return
        run = self.job_runs.get(result.job_run_id)
        if run is None:
            return
        run.status = "failed" if result.outcome.is_failure else "completed"
        run.result_json = json.dumps(result.to_json_dict(), sort_keys=True)
        run.error_message = state.terminal_reason if result.outcome.is_failure else ""
        run.screenshot_path = state.screenshot_path
        self.job_runs.save(run)


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
        action_type=f"map_fog.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _target_is_valid(target: FogTargetObservation, policy: FogScoutPolicy) -> bool:
    status = target.normalized_status()
    return policy.allows(status) and target.confidence >= policy.minimum_confidence


def _require_character(state: _FogState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _FogState) -> FogScoutPolicy:
    if state.policy is None:
        raise RuntimeError("Fog scout policy has not been validated.")
    return state.policy


def _require_scout(state: _FogState) -> MarchAvailability:
    if state.scout is None:
        raise RuntimeError("Idle scout has not been validated.")
    return state.scout


def _require_target(state: _FogState) -> FogTargetObservation:
    if state.target is None:
        raise RuntimeError("Fog target has not been selected.")
    return state.target


def _require_dispatch(state: _FogState) -> MarchDispatchResult:
    if state.dispatch is None:
        raise RuntimeError("Fog scout has not been dispatched.")
    return state.dispatch


def _target_json(target: FogTargetObservation | None) -> dict[str, object]:
    return target.to_json() if target is not None else {}


def _scout_json(scout: MarchAvailability | None) -> dict[str, object]:
    if scout is None:
        return {}
    return {
        "available": scout.available,
        "available_count": scout.available_count,
        "scout_slot": scout.march_slot,
        **scout.data,
    }


def _dispatch_json(dispatch: MarchDispatchResult | None) -> dict[str, object]:
    if dispatch is None:
        return {}
    return {
        "scout_slot": dispatch.march_slot,
        "dispatch_id": dispatch.dispatch_id,
        "expected_return_time": dispatch.expected_return_time,
        **dispatch.data,
    }


def _require_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


def _require_confidence(value: float, field_name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
