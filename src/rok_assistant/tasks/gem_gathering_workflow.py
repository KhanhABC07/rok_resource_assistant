from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from rok_assistant.db.models import Character, Incident, JobRun, March
from rok_assistant.tasks.resource_search_workflow import (
    MAX_RESOURCE_LEVEL,
    MIN_RESOURCE_LEVEL,
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

GEM_GATHERING_WORKFLOW_KEY = "gem-gathering"
GEM_GATHERING_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "search_gem",
    "validate_node",
    "validate_march",
    "dispatch_march",
    "verify_dispatch",
    "complete",
    "recover",
    "failed",
    "cancelled",
)


class GemDepositStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    OCCUPIED = "OCCUPIED"
    NOT_FOUND = "NOT_FOUND"
    BELOW_CONFIDENCE = "BELOW_CONFIDENCE"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class GemSearchPolicy:
    allowed_levels: tuple[int, ...] = (1, 2, 3, 4, 5)
    attempt_limit: int = 5
    search_radius: int | None = None
    total_deadline_seconds: float = 120.0
    march_preset: str = "balanced"
    minimum_detector_confidence: float = 0.85

    def __post_init__(self) -> None:
        levels = tuple(sorted(dict.fromkeys(_validate_level(level) for level in self.allowed_levels)))
        if not levels:
            raise ValueError("GemSearchPolicy.allowed_levels must contain at least one level.")
        _require_positive_int(self.attempt_limit, "GemSearchPolicy.attempt_limit")
        if self.search_radius is not None:
            _require_positive_int(self.search_radius, "GemSearchPolicy.search_radius")
        if (
            not isinstance(self.total_deadline_seconds, int | float)
            or isinstance(self.total_deadline_seconds, bool)
            or not math.isfinite(float(self.total_deadline_seconds))
            or self.total_deadline_seconds <= 0
        ):
            raise ValueError("GemSearchPolicy.total_deadline_seconds must be positive.")
        if not isinstance(self.march_preset, str) or not self.march_preset.strip():
            raise ValueError("GemSearchPolicy.march_preset must be a non-empty string.")
        _require_confidence(self.minimum_detector_confidence, "GemSearchPolicy.minimum_detector_confidence")
        object.__setattr__(self, "allowed_levels", levels)
        object.__setattr__(self, "march_preset", self.march_preset.strip())
        object.__setattr__(self, "total_deadline_seconds", float(self.total_deadline_seconds))
        object.__setattr__(self, "minimum_detector_confidence", float(self.minimum_detector_confidence))

    def to_json(self) -> dict[str, object]:
        return {
            "allowed_levels": list(self.allowed_levels),
            "attempt_limit": self.attempt_limit,
            "search_radius": self.search_radius,
            "total_deadline_seconds": self.total_deadline_seconds,
            "march_preset": self.march_preset,
            "minimum_detector_confidence": self.minimum_detector_confidence,
        }


@dataclass(frozen=True)
class GemGatheringRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    policy: GemSearchPolicy = field(default_factory=GemSearchPolicy)
    target_account_id: int | None = None
    session_key: str = ""
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class GemGatheringConfig:
    workflow_timeout_seconds: float = 180.0
    step_timeout_seconds: float = 20.0
    precondition_retry_limit: int = 1
    dispatch_retry_limit: int = 1
    retry_delay_seconds: float = 0.25


@dataclass(frozen=True)
class GemDepositObservation:
    status: GemDepositStatus | str
    level: int | None = None
    confidence: float = 0.0
    x: int | None = None
    y: int | None = None
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_status(self) -> GemDepositStatus:
        if isinstance(self.status, GemDepositStatus):
            return self.status
        try:
            return GemDepositStatus(str(self.status).strip().upper())
        except ValueError as exc:
            valid = ", ".join(item.value for item in GemDepositStatus)
            raise ValueError(f"Invalid gem deposit status: {self.status!r}. Expected one of: {valid}.") from exc

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.normalized_status().value,
            "level": self.level,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            **self.data,
        }


@dataclass(frozen=True)
class GemDetectionDatasetCase:
    case_id: str
    expected_positive: bool
    observation: GemDepositObservation
    label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("GemDetectionDatasetCase.case_id must be a non-empty string.")
        if not isinstance(self.expected_positive, bool):
            raise ValueError("GemDetectionDatasetCase.expected_positive must be boolean.")
        object.__setattr__(self, "case_id", self.case_id.strip())
        object.__setattr__(self, "label", str(self.label))


@dataclass(frozen=True)
class GemDetectionMetrics:
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float

    def meets(self, *, minimum_precision: float, minimum_recall: float) -> bool:
        _require_confidence(minimum_precision, "minimum_precision")
        _require_confidence(minimum_recall, "minimum_recall")
        return self.precision >= minimum_precision and self.recall >= minimum_recall


def evaluate_gem_detection_dataset(
    cases: tuple[GemDetectionDatasetCase, ...],
    policy: GemSearchPolicy,
) -> GemDetectionMetrics:
    if not cases:
        raise ValueError("Gem detection evaluation requires at least one labeled case.")
    tp = tn = fp = fn = 0
    for case in sorted(cases, key=lambda item: item.case_id):
        actual = _observation_is_clickable(case.observation, policy)
        if case.expected_positive and actual:
            tp += 1
        elif case.expected_positive and not actual:
            fn += 1
        elif not case.expected_positive and actual:
            fp += 1
        else:
            tn += 1
    return GemDetectionMetrics(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
    )


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


class IncidentRepository(Protocol):
    def save(self, incident: Incident) -> int:
        ...


class GemAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: GemGatheringRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class GemCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: GemGatheringRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class GemGatheringDriver(Protocol):
    def search_for_gem(
        self,
        request: GemGatheringRequest,
        policy: GemSearchPolicy,
        attempt: int,
    ) -> GemDepositObservation:
        ...

    def validate_gem_node_available(
        self,
        request: GemGatheringRequest,
        deposit: GemDepositObservation,
        policy: GemSearchPolicy,
    ) -> GemDepositObservation:
        ...

    def validate_march_availability(
        self,
        request: GemGatheringRequest,
        deposit: GemDepositObservation,
        policy: GemSearchPolicy,
    ) -> MarchAvailability:
        ...

    def dispatch_gem_gather_march(
        self,
        request: GemGatheringRequest,
        deposit: GemDepositObservation,
        availability: MarchAvailability,
        policy: GemSearchPolicy,
    ) -> MarchDispatchResult:
        ...

    def verify_dispatch(
        self,
        request: GemGatheringRequest,
        deposit: GemDepositObservation,
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
class _GemState:
    request: GemGatheringRequest
    character: Character | None = None
    selected_deposit: GemDepositObservation | None = None
    validated_deposit: GemDepositObservation | None = None
    march_availability: MarchAvailability | None = None
    dispatch: MarchDispatchResult | None = None
    search_attempts: list[dict[str, object]] = field(default_factory=list)
    failure_reason: str = ""
    failure_state: str = ""
    recovery_outcome: dict[str, object] = field(default_factory=dict)
    screenshot_path: str = ""

    @property
    def failed(self) -> bool:
        return bool(self.failure_reason)

    def fail(
        self,
        state: str,
        reason: str,
        *,
        screenshot_path: str = "",
        data: dict[str, object] | None = None,
    ) -> WorkflowStepResult:
        self.failure_state = state
        self.failure_reason = reason
        if screenshot_path:
            self.screenshot_path = screenshot_path
        return _gem_step_result(
            state,
            WorkflowOutcome.SUCCESS,
            message=reason,
            data={
                "state_failed": True,
                "failure_state": state,
                "failure_reason": reason,
                **(data or {}),
            },
            screenshot_path=screenshot_path,
        )


class GemGatheringWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: GemGatheringDriver,
        account_precondition: GemAccountPrecondition | None = None,
        character_precondition: GemCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        marches: MarchRepository | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: GemGatheringConfig | None = None,
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
        self.incidents = incidents
        self.config = config or GemGatheringConfig()
        self.clock = clock
        self._states: dict[str, _GemState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return GEM_GATHERING_STATES

    def execute(
        self,
        request: GemGatheringRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _GemState(request=request)
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
                budget=StepBudget(max_steps=len(GEM_GATHERING_STATES) + request.policy.attempt_limit + 8),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"gem-gathering:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"gem_gathering_run_id": token},
            )
            result = self._engine().execute(self._definition(), context)
            self._record_engine_failure(result, state)
            self._augment_result(result, state)
            self._update_persisted_run(result, state)
            return result
        finally:
            self._states.pop(token, None)

    def _engine(self) -> WorkflowEngine:
        registry = ActionRegistry()
        for state in GEM_GATHERING_STATES:
            registry.register(f"gem_gathering.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.precondition_retry_limit,
            "ensure_character": self.config.precondition_retry_limit,
            "ensure_game_running": self.config.precondition_retry_limit,
            "dispatch_march": self.config.dispatch_retry_limit,
            "verify_dispatch": self.config.dispatch_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=GEM_GATHERING_WORKFLOW_KEY,
            name="Gather Gems",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"gem_gathering.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in GEM_GATHERING_STATES
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
                return _gem_step_result(step.step_key, WorkflowOutcome.SKIPPED)
            if state.failed and state_name not in {"recover", "complete"}:
                return _gem_step_result(
                    step.step_key,
                    WorkflowOutcome.SKIPPED,
                    data={"skipped_after_failure": state.failure_state},
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
        state: _GemState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        request = state.request
        if request.instance_id <= 0:
            return state.fail(step.step_key, "instance_id must be positive.")
        if request.instance_index < 0:
            return state.fail(step.step_key, "instance_index must be zero or greater.")
        if request.character_id <= 0:
            return state.fail(step.step_key, "character_id must be positive.")
        try:
            GemSearchPolicy(**request.policy.to_json())
        except ValueError as exc:
            return state.fail(step.step_key, str(exc))
        return _gem_step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"policy": request.policy.to_json()},
        )

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _GemState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        character = self.characters.get(state.request.character_id)
        if character is None:
            return state.fail(step.step_key, "Target character was not found.")
        if not character.enabled:
            return state.fail(step.step_key, "Target character is disabled.")
        if character.instance_id is not None and character.instance_id != state.request.instance_id:
            return state.fail(
                step.step_key,
                "Target character is not assigned to the requested instance.",
                data={
                    "character_instance_id": character.instance_id,
                    "request_instance_id": state.request.instance_id,
                },
            )
        state.character = character
        return _gem_step_result(
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
        state: _GemState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.account_precondition is None:
            return _gem_step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"account_precondition": "not_configured"})
        return self._action_to_step(
            step,
            state,
            self.account_precondition.ensure_account(state.request, _require_character(state)),
        )

    def _ensure_character(
        self,
        step: WorkflowStepSpec,
        state: _GemState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.character_precondition is None:
            return _gem_step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"character_precondition": "not_configured"})
        return self._action_to_step(
            step,
            state,
            self.character_precondition.ensure_character(state.request, _require_character(state)),
        )

    def _ensure_game_running(
        self,
        step: WorkflowStepSpec,
        state: _GemState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.recovery_watchdog is None:
            return _gem_step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"watchdog": "not_configured"})
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
            return state.fail(
                step.step_key,
                message,
                screenshot_path=screenshot_path,
                data={"watchdog_healthy": False},
            )
        return _gem_step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"watchdog_healthy": True})

    def _search_gem(
        self,
        step: WorkflowStepSpec,
        state: _GemState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        started_at = self.clock()
        policy = state.request.policy
        for attempt in range(1, policy.attempt_limit + 1):
            context.cancellation_token.throw_if_cancelled()
            if self.clock() - started_at >= policy.total_deadline_seconds:
                break
            observation = self.driver.search_for_gem(state.request, policy, attempt)
            self._record_search_attempt(state, attempt, observation)
            if observation.screenshot_path:
                state.screenshot_path = observation.screenshot_path
            status = observation.normalized_status()
            if status == GemDepositStatus.VERIFICATION_REQUIRED:
                return state.fail(
                    step.step_key,
                    observation.message or "Verification screen requires manual intervention.",
                    screenshot_path=observation.screenshot_path,
                    data={"search_attempts": state.search_attempts},
                )
            if _observation_is_clickable(observation, policy):
                state.selected_deposit = observation
                return _gem_step_result(
                    step.step_key,
                    WorkflowOutcome.SUCCESS,
                    data={
                        "selected_gem": observation.to_json(),
                        "search_attempts": state.search_attempts,
                    },
                    screenshot_path=observation.screenshot_path,
                )
        return state.fail(
            step.step_key,
            "No available gem node was found before the attempt or deadline bound.",
            screenshot_path=state.screenshot_path,
            data={"search_attempts": state.search_attempts},
        )

    def _validate_node(
        self,
        step: WorkflowStepSpec,
        state: _GemState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        deposit = _require_deposit(state)
        result = self.driver.validate_gem_node_available(state.request, deposit, state.request.policy)
        state.validated_deposit = result
        if result.screenshot_path:
            state.screenshot_path = result.screenshot_path
        status = result.normalized_status()
        if status == GemDepositStatus.VERIFICATION_REQUIRED:
            return state.fail(
                step.step_key,
                result.message or "Verification screen requires manual intervention.",
                screenshot_path=result.screenshot_path,
                data={"selected_gem": deposit.to_json(), "validated_gem": result.to_json()},
            )
        if status != GemDepositStatus.AVAILABLE:
            return state.fail(
                step.step_key,
                result.message or "Gem node is not available.",
                screenshot_path=result.screenshot_path,
                data={"selected_gem": deposit.to_json(), "validated_gem": result.to_json()},
            )
        return _gem_step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"validated_gem": result.to_json()},
            screenshot_path=result.screenshot_path,
        )

    def _validate_march(
        self,
        step: WorkflowStepSpec,
        state: _GemState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        deposit = _require_validated_deposit(state)
        availability = self.driver.validate_march_availability(state.request, deposit, state.request.policy)
        state.march_availability = availability
        if availability.screenshot_path:
            state.screenshot_path = availability.screenshot_path
        if not availability.available:
            return state.fail(
                step.step_key,
                availability.message or "No march is available for gem gathering.",
                screenshot_path=availability.screenshot_path,
                data={
                    "available_count": availability.available_count,
                    "march_slot": availability.march_slot,
                    **availability.data,
                },
            )
        return _gem_step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "available_count": availability.available_count,
                "march_slot": availability.march_slot,
                "march_preset": state.request.policy.march_preset,
                **availability.data,
            },
            screenshot_path=availability.screenshot_path,
        )

    def _dispatch_march(
        self,
        step: WorkflowStepSpec,
        state: _GemState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        deposit = _require_validated_deposit(state)
        availability = _require_availability(state)
        dispatch = self.driver.dispatch_gem_gather_march(state.request, deposit, availability, state.request.policy)
        state.dispatch = dispatch
        if dispatch.screenshot_path:
            state.screenshot_path = dispatch.screenshot_path
        if not dispatch.success:
            return (
                _gem_step_result(
                    step.step_key,
                    WorkflowOutcome.RETRYABLE_FAILURE,
                    dispatch.message or "Gem gather march dispatch failed.",
                    data=dispatch.data,
                    screenshot_path=dispatch.screenshot_path,
                )
                if dispatch.retryable
                else state.fail(
                    step.step_key,
                    dispatch.message or "Gem gather march dispatch failed.",
                    screenshot_path=dispatch.screenshot_path,
                    data=dispatch.data,
                )
            )
        self._persist_march_dispatch(state, dispatch)
        return _gem_step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"dispatch": _dispatch_json(dispatch)},
            screenshot_path=dispatch.screenshot_path,
        )

    def _verify_dispatch(
        self,
        step: WorkflowStepSpec,
        state: _GemState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.verify_dispatch(
                state.request,
                _require_validated_deposit(state),
                _require_dispatch(state),
            ),
        )

    def _complete(self, step: WorkflowStepSpec, state: _GemState) -> WorkflowStepResult:
        if state.failed:
            return _gem_step_result(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                data={"failure_state": state.failure_state},
            )
        return _gem_step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "selected_gem": _deposit_json(state.selected_deposit),
                "validated_gem": _deposit_json(state.validated_deposit),
                "dispatch": _dispatch_json(state.dispatch),
            },
        )

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _GemState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _gem_step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if _is_verification_stop(state):
            state.recovery_outcome = {"attempted": False, "reason": "verification_screen"}
            return _gem_step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
        if self.recovery_watchdog is None:
            state.recovery_outcome = {"attempted": False, "reason": "watchdog_not_configured"}
            return _gem_step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
        result = self.recovery_watchdog.monitor(
            instance_id=state.request.instance_id,
            instance_index=state.request.instance_index,
            instance_name=state.request.instance_name,
            job_run_id=_job_run_id(context),
        )
        state.recovery_outcome = {
            "attempted": bool(getattr(result, "recovery_attempted", False)),
            "healthy": bool(getattr(result, "healthy", False)),
            "circuit_opened": bool(getattr(result, "circuit_opened", False)),
        }
        return _gem_step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _GemState) -> WorkflowStepResult:
        if not state.failed:
            return _gem_step_result(step.step_key, WorkflowOutcome.SKIPPED)
        self._open_incident(state)
        return _gem_step_result(
            step.step_key,
            WorkflowOutcome.FATAL_FAILURE,
            state.failure_reason,
            data={
                "failure_state": state.failure_state,
                "failure_reason": state.failure_reason,
                "selected_gem": _deposit_json(state.selected_deposit),
                "validated_gem": _deposit_json(state.validated_deposit),
                "dispatch": _dispatch_json(state.dispatch),
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _record_search_attempt(
        self,
        state: _GemState,
        attempt: int,
        observation: GemDepositObservation,
    ) -> None:
        state.search_attempts.append(
            {
                "attempt": attempt,
                "status": observation.normalized_status().value,
                "level": observation.level,
                "confidence": observation.confidence,
                "x": observation.x,
                "y": observation.y,
                "message": observation.message,
                **observation.data,
            }
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _GemState,
        action: ResourceGatheringActionResult,
    ) -> WorkflowStepResult:
        if action.screenshot_path:
            state.screenshot_path = action.screenshot_path
        if action.success:
            return _gem_step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        if action.retryable:
            return _gem_step_result(
                step.step_key,
                WorkflowOutcome.RETRYABLE_FAILURE,
                action.message or "Gem gathering action failed.",
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.fail(
            step.step_key,
            action.message or "Gem gathering action failed.",
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _persist_march_dispatch(
        self,
        state: _GemState,
        dispatch: MarchDispatchResult,
    ) -> None:
        if self.marches is None or dispatch.march_slot is None:
            return
        character = _require_character(state)
        marches = self.marches.list_for_character(int(character.id or 0))
        existing = next((item for item in marches if item.march_slot == dispatch.march_slot), None)
        march = existing or March(character_id=character.id, march_slot=dispatch.march_slot)
        march.status = "gem_gathering"
        march.expected_return_time = dispatch.expected_return_time
        march.next_action_time = dispatch.expected_return_time
        self.marches.save(march)

    def _state_from_context(self, context: WorkflowExecutionContext) -> _GemState:
        token = str(context.metadata.get("gem_gathering_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("Gem gathering runtime state is missing.") from exc

    def _open_incident(self, state: _GemState) -> None:
        if self.incidents is None or not state.failed:
            return
        self.incidents.save(
            Incident(
                incident_key=f"gem-gathering:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="Gem gathering failed",
                details=state.failure_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _GemState,
    ) -> None:
        if not result.outcome.is_failure or state.failed or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.failure_state = last_step.step_key if last_step is not None else ""
        state.failure_reason = result.message or "Gem gathering workflow failed."
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""
        if _is_verification_stop(state):
            state.recovery_outcome = {"attempted": False, "reason": "verification_screen"}
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
        state: _GemState,
    ) -> None:
        result.result = {
            **dict(result.result),
            "policy": state.request.policy.to_json(),
            "search_attempts": state.search_attempts,
            "selected_gem": _deposit_json(state.selected_deposit),
            "validated_gem": _deposit_json(state.validated_deposit),
            "march_dispatch": _dispatch_json(state.dispatch),
            "failure_state": state.failure_state,
            "failure_reason": state.failure_reason,
            "recovery_outcome": state.recovery_outcome,
        }

    def _update_persisted_run(
        self,
        result: WorkflowExecutionResult,
        state: _GemState,
    ) -> None:
        if self.job_runs is None or result.job_run_id is None:
            return
        run = self.job_runs.get(result.job_run_id)
        if run is None:
            return
        run.result_json = json.dumps(result.to_json_dict(), sort_keys=True)
        run.error_message = state.failure_reason if result.outcome.is_failure else ""
        run.screenshot_path = state.screenshot_path
        self.job_runs.save(run)


def _gem_step_result(
    step_key: str,
    outcome: WorkflowOutcome,
    message: str = "",
    *,
    data: dict[str, object] | None = None,
    screenshot_path: str = "",
) -> WorkflowStepResult:
    return WorkflowStepResult(
        step_key=step_key,
        action_type=f"gem_gathering.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _observation_is_clickable(
    observation: GemDepositObservation,
    policy: GemSearchPolicy,
) -> bool:
    if observation.normalized_status() != GemDepositStatus.AVAILABLE:
        return False
    if observation.level not in policy.allowed_levels:
        return False
    return observation.confidence >= policy.minimum_detector_confidence


def _require_character(state: _GemState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_deposit(state: _GemState) -> GemDepositObservation:
    if state.selected_deposit is None:
        raise RuntimeError("Gem deposit has not been selected.")
    return state.selected_deposit


def _require_validated_deposit(state: _GemState) -> GemDepositObservation:
    if state.validated_deposit is None:
        raise RuntimeError("Gem deposit has not been validated.")
    return state.validated_deposit


def _require_availability(state: _GemState) -> MarchAvailability:
    if state.march_availability is None:
        raise RuntimeError("March availability has not been validated.")
    return state.march_availability


def _require_dispatch(state: _GemState) -> MarchDispatchResult:
    if state.dispatch is None:
        raise RuntimeError("Gem gather march has not been dispatched.")
    return state.dispatch


def _deposit_json(deposit: GemDepositObservation | None) -> dict[str, object]:
    return deposit.to_json() if deposit is not None else {}


def _dispatch_json(dispatch: MarchDispatchResult | None) -> dict[str, object]:
    if dispatch is None:
        return {}
    return {
        "march_slot": dispatch.march_slot,
        "dispatch_id": dispatch.dispatch_id,
        "expected_return_time": dispatch.expected_return_time,
        **dispatch.data,
    }


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_verification_stop(state: _GemState) -> bool:
    return "verification" in state.failure_reason.lower()


def _validate_level(level: int) -> int:
    if isinstance(level, bool):
        raise ValueError("Gem levels must be integers.")
    try:
        value = int(level)
    except (TypeError, ValueError) as exc:
        raise ValueError("Gem levels must be integers.") from exc
    if value < MIN_RESOURCE_LEVEL or value > MAX_RESOURCE_LEVEL:
        raise ValueError(f"Gem levels must be between {MIN_RESOURCE_LEVEL} and {MAX_RESOURCE_LEVEL}.")
    return value


def _require_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


def _require_confidence(value: float, field_name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
