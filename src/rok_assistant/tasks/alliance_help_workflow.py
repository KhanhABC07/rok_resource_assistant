from __future__ import annotations

import json
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


ALLIANCE_HELP_WORKFLOW_KEY = "alliance-help"
ALLIANCE_HELP_TEMPLATE_KEYS = ("alliance.help.ready",)
ALLIANCE_HELP_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "normalize_help_scene",
    "detect_help_button",
    "press_help",
    "verify_help_state",
    "complete",
    "skipped",
    "recover",
    "failed",
    "cancelled",
)


class AllianceHelpStatus(StrEnum):
    READY = "READY"
    NOT_READY = "NOT_READY"
    OVERLAY_BLOCKED = "OVERLAY_BLOCKED"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class AllianceHelpPolicy:
    minimum_detector_confidence: float = 0.85
    allow_overlay_dismissal: bool = True

    def normalized(self) -> AllianceHelpPolicy:
        confidence = float(self.minimum_detector_confidence)
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("minimum_detector_confidence must be between 0.0 and 1.0.")
        return AllianceHelpPolicy(
            minimum_detector_confidence=confidence,
            allow_overlay_dismissal=bool(self.allow_overlay_dismissal),
        )

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "minimum_detector_confidence": normalized.minimum_detector_confidence,
            "allow_overlay_dismissal": normalized.allow_overlay_dismissal,
        }


@dataclass(frozen=True)
class AllianceHelpRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: AllianceHelpPolicy = field(default_factory=AllianceHelpPolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class AllianceHelpConfig:
    workflow_timeout_seconds: float = 90.0
    step_timeout_seconds: float = 10.0
    precondition_retry_limit: int = 1
    navigation_retry_limit: int = 1
    detection_retry_limit: int = 1
    retry_delay_seconds: float = 0.25


@dataclass(frozen=True)
class AllianceHelpObservation:
    status: AllianceHelpStatus | str
    template_key: str = "alliance.help.ready"
    confidence: float = 0.0
    target: tuple[int, int] | None = None
    scene_verified: bool = True
    button_active: bool = False
    badge_visible: bool = False
    overlay_blocked: bool = False
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_status(self) -> AllianceHelpStatus:
        if isinstance(self.status, AllianceHelpStatus):
            return self.status
        try:
            return AllianceHelpStatus(str(self.status).strip().upper())
        except ValueError as exc:
            valid = ", ".join(item.value for item in AllianceHelpStatus)
            raise ValueError(f"Invalid alliance help status: {self.status!r}. Expected one of: {valid}.") from exc

    def target_json(self) -> dict[str, int] | None:
        if self.target is None:
            return None
        return {"x": int(self.target[0]), "y": int(self.target[1])}

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.normalized_status().value,
            "template_key": self.template_key,
            "confidence": self.confidence,
            "target": self.target_json(),
            "scene_verified": self.scene_verified,
            "button_active": self.button_active,
            "badge_visible": self.badge_visible,
            "overlay_blocked": self.overlay_blocked,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class AllianceHelpActionResult:
    success: bool
    changed: bool = False
    message: str = ""
    retryable: bool = True
    overlay_blocked: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "changed": self.changed,
            "overlay_blocked": self.overlay_blocked,
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


class AllianceHelpAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: AllianceHelpRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class AllianceHelpCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: AllianceHelpRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class AllianceHelpDriver(Protocol):
    def normalize_help_scene(
        self,
        request: AllianceHelpRequest,
        character: Character,
        policy: AllianceHelpPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def detect_help_button(
        self,
        request: AllianceHelpRequest,
        character: Character,
        policy: AllianceHelpPolicy,
    ) -> AllianceHelpObservation:
        ...

    def press_help(
        self,
        request: AllianceHelpRequest,
        character: Character,
        observation: AllianceHelpObservation,
        policy: AllianceHelpPolicy,
    ) -> AllianceHelpActionResult:
        ...

    def verify_help_state(
        self,
        request: AllianceHelpRequest,
        character: Character,
        before: AllianceHelpObservation,
        press_result: AllianceHelpActionResult,
        policy: AllianceHelpPolicy,
    ) -> AllianceHelpObservation:
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
class _AllianceHelpState:
    request: AllianceHelpRequest
    character: Character | None = None
    policy: AllianceHelpPolicy | None = None
    detection: AllianceHelpObservation | None = None
    press_result: AllianceHelpActionResult | None = None
    verification: AllianceHelpObservation | None = None
    detection_attempts: list[dict[str, object]] = field(default_factory=list)
    click_attempts: list[dict[str, object]] = field(default_factory=list)
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


class AllianceHelpWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: AllianceHelpDriver,
        account_precondition: AllianceHelpAccountPrecondition | None = None,
        character_precondition: AllianceHelpCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: AllianceHelpConfig | None = None,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.account_precondition = account_precondition
        self.character_precondition = character_precondition
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.incidents = incidents
        self.config = config or AllianceHelpConfig()
        self._states: dict[str, _AllianceHelpState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return ALLIANCE_HELP_STATES

    def execute(
        self,
        request: AllianceHelpRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _AllianceHelpState(request=request)
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
                budget=StepBudget(max_steps=len(ALLIANCE_HELP_STATES) + 10),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"alliance-help:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"alliance_help_run_id": token},
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
        for state in ALLIANCE_HELP_STATES:
            registry.register(f"alliance_help.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.precondition_retry_limit,
            "ensure_character": self.config.precondition_retry_limit,
            "ensure_game_running": self.config.precondition_retry_limit,
            "normalize_help_scene": self.config.navigation_retry_limit,
            "detect_help_button": self.config.detection_retry_limit,
            "verify_help_state": self.config.detection_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=ALLIANCE_HELP_WORKFLOW_KEY,
            name="Press Alliance Help",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"alliance_help.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in ALLIANCE_HELP_STATES
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
        state: _AllianceHelpState,
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
                "template_keys": list(ALLIANCE_HELP_TEMPLATE_KEYS),
            },
        )

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _AllianceHelpState,
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
        state: _AllianceHelpState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.account_precondition is None:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"account_precondition": "not_configured"})
        return self._action_to_step(step, state, self.account_precondition.ensure_account(state.request, _require_character(state)))

    def _ensure_character(
        self,
        step: WorkflowStepSpec,
        state: _AllianceHelpState,
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
        state: _AllianceHelpState,
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

    def _normalize_help_scene(
        self,
        step: WorkflowStepSpec,
        state: _AllianceHelpState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.normalize_help_scene(state.request, _require_character(state), _require_policy(state)),
        )

    def _detect_help_button(
        self,
        step: WorkflowStepSpec,
        state: _AllianceHelpState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        detection = self.driver.detect_help_button(state.request, _require_character(state), _require_policy(state))
        state.detection = detection
        if detection.screenshot_path:
            state.screenshot_path = detection.screenshot_path
        state.detection_attempts.append(detection.to_json())
        status = detection.normalized_status()
        if status == AllianceHelpStatus.NOT_READY:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                detection.message or "No alliance help is available.",
                screenshot_path=detection.screenshot_path,
                data={"help_detection": detection.to_json(), "skipped_reason": "no_help_available"},
            )
        if status == AllianceHelpStatus.VERIFICATION_REQUIRED:
            return state.stop(
                step.step_key,
                WorkflowOutcome.FATAL_FAILURE,
                detection.message or "Verification screen requires manual intervention.",
                screenshot_path=detection.screenshot_path,
                data={"help_detection": detection.to_json()},
            )
        if status == AllianceHelpStatus.OVERLAY_BLOCKED or detection.overlay_blocked:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                detection.message or "Alliance help button is obstructed by an overlay.",
                screenshot_path=detection.screenshot_path,
                data={"help_detection": detection.to_json()},
            )
        if status != AllianceHelpStatus.READY:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                detection.message or "Alliance help button state could not be determined.",
                screenshot_path=detection.screenshot_path,
                data={"help_detection": detection.to_json()},
            )
        if detection.template_key != "alliance.help.ready" or detection.confidence < _require_policy(state).minimum_detector_confidence:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                "Alliance help ready target was not verified above the confidence threshold.",
                screenshot_path=detection.screenshot_path,
                data={"help_detection": detection.to_json()},
            )
        if not detection.scene_verified or detection.target is None:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                detection.message or "Alliance help target scene and coordinates were not verified.",
                screenshot_path=detection.screenshot_path,
                data={"help_detection": detection.to_json()},
            )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"help_detection": detection.to_json()}, screenshot_path=detection.screenshot_path)

    def _press_help(
        self,
        step: WorkflowStepSpec,
        state: _AllianceHelpState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if state.click_attempts:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                "Alliance help button click was already attempted in this run.",
                screenshot_path=state.screenshot_path,
            )
        result = self.driver.press_help(
            state.request,
            _require_character(state),
            _require_detection(state),
            _require_policy(state),
        )
        state.press_result = result
        if result.screenshot_path:
            state.screenshot_path = result.screenshot_path
        state.click_attempts.append({"help_detection": _require_detection(state).to_json(), **result.to_json()})
        if result.overlay_blocked:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                result.message or "Alliance help button is obstructed by an overlay.",
                screenshot_path=result.screenshot_path,
                data=result.to_json(),
            )
        if not result.success:
            return self._help_action_failure(step, state, result, "Alliance help button could not be pressed.")
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=result.to_json(), screenshot_path=result.screenshot_path)

    def _verify_help_state(
        self,
        step: WorkflowStepSpec,
        state: _AllianceHelpState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        verification = self.driver.verify_help_state(
            state.request,
            _require_character(state),
            _require_detection(state),
            _require_press_result(state),
            _require_policy(state),
        )
        state.verification = verification
        if verification.screenshot_path:
            state.screenshot_path = verification.screenshot_path
        state.verification_attempts.append(verification.to_json())
        if not _help_postcondition_verified(verification):
            failure = replace(
                verification,
                status=AllianceHelpStatus.UNKNOWN,
                message=("Alliance help badge did not disappear and the button did not become inactive. " f"{verification.message}").strip(),
            )
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                failure.message,
                screenshot_path=failure.screenshot_path,
                data={"verification": failure.to_json()},
            )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"verification": verification.to_json()}, screenshot_path=verification.screenshot_path)

    def _complete(self, step: WorkflowStepSpec, state: _AllianceHelpState) -> WorkflowStepResult:
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
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=self._help_payload(state))

    def _skipped(self, step: WorkflowStepSpec, state: _AllianceHelpState) -> WorkflowStepResult:
        if state.terminal_outcome == WorkflowOutcome.SKIPPED:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data={"skipped_reason": state.terminal_reason, **self._help_payload(state)},
                screenshot_path=state.screenshot_path,
            )
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED)

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _AllianceHelpState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if _is_manual_stop(state):
            state.recovery_outcome = {"attempted": False, "reason": "manual_intervention_required"}
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
        state.recovery_outcome = self._monitor_recovery(state, _job_run_id(context))
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _AllianceHelpState) -> WorkflowStepResult:
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
                **self._help_payload(state),
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _AllianceHelpState,
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
                action.message or "Alliance help action failed.",
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            action.message or "Alliance help action failed.",
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _help_action_failure(
        self,
        step: WorkflowStepSpec,
        state: _AllianceHelpState,
        result: AllianceHelpActionResult,
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

    def _help_payload(self, state: _AllianceHelpState) -> dict[str, object]:
        return {
            "help_ready_detection": state.detection.to_json() if state.detection is not None else {},
            "click_attempted": bool(state.click_attempts),
            "click_attempts": state.click_attempts,
            "verification_result": state.verification.to_json() if state.verification is not None else {},
            "detection_attempts": state.detection_attempts,
            "verification_attempts": state.verification_attempts,
            "failure_evidence": {
                "screenshot_path": state.screenshot_path,
                "terminal_state": state.terminal_state,
                "terminal_reason": state.terminal_reason,
            },
        }

    def _state_from_context(self, context: WorkflowExecutionContext) -> _AllianceHelpState:
        token = str(context.metadata.get("alliance_help_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("Alliance help runtime state is missing.") from exc

    def _open_incident(self, state: _AllianceHelpState) -> None:
        if self.incidents is None or not state.failed or state.incident_opened:
            return
        self.incidents.save(
            Incident(
                incident_key=f"alliance-help:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="Alliance help workflow blocked",
                details=state.terminal_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )
        state.incident_opened = True

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _AllianceHelpState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "Alliance help workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(
        self,
        result: WorkflowExecutionResult,
        state: _AllianceHelpState,
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
        state: _AllianceHelpState,
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
        state: _AllianceHelpState,
    ) -> None:
        if state.terminal_outcome in {WorkflowOutcome.SKIPPED, WorkflowOutcome.BLOCKED}:
            result.outcome = state.terminal_outcome
            result.message = state.terminal_reason
        result.result = {
            **dict(result.result),
            "policy": state.policy.to_json() if state.policy is not None else {},
            **self._help_payload(state),
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
        state: _AllianceHelpState,
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


def _help_postcondition_verified(verification: AllianceHelpObservation) -> bool:
    if verification.overlay_blocked:
        return False
    status = verification.normalized_status()
    if status == AllianceHelpStatus.NOT_READY:
        return True
    return not verification.badge_visible or not verification.button_active


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
        action_type=f"alliance_help.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _AllianceHelpState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _AllianceHelpState) -> AllianceHelpPolicy:
    if state.policy is None:
        raise RuntimeError("Alliance help policy has not been validated.")
    return state.policy


def _require_detection(state: _AllianceHelpState) -> AllianceHelpObservation:
    if state.detection is None:
        raise RuntimeError("Alliance help button has not been detected.")
    return state.detection


def _require_press_result(state: _AllianceHelpState) -> AllianceHelpActionResult:
    if state.press_result is None:
        raise RuntimeError("Alliance help button has not been pressed.")
    return state.press_result


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_manual_stop(state: _AllianceHelpState) -> bool:
    text = state.terminal_reason.lower()
    return "verification" in text or "manual" in text
