from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from rok_assistant.db.models import Character, Incident, JobRun, March
from rok_assistant.tasks.resource_search_workflow import (
    MarchAvailability,
    MarchDispatchResult,
    ResourceGatheringActionResult,
    ResourceType,
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


ALLIANCE_PIT_WORKFLOW_KEY = "alliance-pit-gathering"
ALLIANCE_PIT_TEMPLATE_KEYS = (
    "city.alliance.button",
    "alliance.menu.territory",
    "alliance.menu.resource_center",
    "alliance.resource_center.panel",
    "alliance.resource_center.reward_popup",
    "alliance.resource_center.joinable_pit",
    "alliance.resource_center.already_participating",
    "alliance.resource_center.no_active_pit",
    "alliance.resource_center.not_joinable",
    "march.free_slot",
    "march.preset.selector",
    "march.dispatch.button",
    "march.dispatch.started_indicator",
)
ALLIANCE_PIT_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "navigate_to_resource_center",
    "clear_overlays",
    "detect_pit",
    "validate_march",
    "dispatch_march",
    "verify_dispatch",
    "complete",
    "recover",
    "failed",
    "cancelled",
)


class AlliancePitStatus(StrEnum):
    JOINABLE = "JOINABLE"
    NO_ACTIVE_PIT = "NO_ACTIVE_PIT"
    ALREADY_PARTICIPATING = "ALREADY_PARTICIPATING"
    NOT_JOINABLE = "NOT_JOINABLE"


@dataclass(frozen=True)
class AlliancePitPolicy:
    enabled_resource_types: tuple[ResourceType | str, ...] = (
        ResourceType.FOOD,
        ResourceType.WOOD,
        ResourceType.STONE,
        ResourceType.GOLD,
    )
    march_preset: str = "default"

    def normalized(self) -> AlliancePitPolicy:
        enabled = tuple(_resource_type(item) for item in self.enabled_resource_types)
        if not enabled:
            raise ValueError("At least one alliance pit resource type must be enabled.")
        preset = str(self.march_preset).strip()
        if not preset:
            raise ValueError("Alliance pit march preset must be configured.")
        return AlliancePitPolicy(enabled_resource_types=enabled, march_preset=preset)

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "enabled_resource_types": [item.value for item in normalized.enabled_resource_types],
            "march_preset": normalized.march_preset,
        }


@dataclass(frozen=True)
class AlliancePitRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: AlliancePitPolicy = field(default_factory=AlliancePitPolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class AlliancePitConfig:
    workflow_timeout_seconds: float = 180.0
    step_timeout_seconds: float = 20.0
    navigation_retry_limit: int = 1
    detection_retry_limit: int = 1
    dispatch_retry_limit: int = 1
    retry_delay_seconds: float = 0.25


@dataclass(frozen=True)
class AlliancePitObservation:
    status: AlliancePitStatus | str
    resource_type: ResourceType | str | None = None
    pit_id: str = ""
    pit_level: int | None = None
    confidence: float = 0.0
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_status(self) -> AlliancePitStatus:
        if isinstance(self.status, AlliancePitStatus):
            return self.status
        try:
            return AlliancePitStatus(str(self.status).strip().upper())
        except ValueError as exc:
            valid = ", ".join(item.value for item in AlliancePitStatus)
            raise ValueError(f"Invalid alliance pit status: {self.status!r}. Expected one of: {valid}.") from exc

    def normalized_resource_type(self) -> ResourceType | None:
        if self.resource_type is None:
            return None
        return _resource_type(self.resource_type)

    def to_json(self) -> dict[str, object]:
        resource_type = self.normalized_resource_type()
        return {
            "status": self.normalized_status().value,
            "resource_type": resource_type.value if resource_type is not None else "",
            "pit_id": self.pit_id,
            "pit_level": self.pit_level,
            "confidence": self.confidence,
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


class IncidentRepository(Protocol):
    def save(self, incident: Incident) -> int:
        ...


class AlliancePitAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: AlliancePitRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class AlliancePitCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: AlliancePitRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class AlliancePitDriver(Protocol):
    def navigate_to_resource_center(
        self,
        request: AlliancePitRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...

    def clear_overlays(
        self,
        request: AlliancePitRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...

    def detect_pit(
        self,
        request: AlliancePitRequest,
        policy: AlliancePitPolicy,
    ) -> AlliancePitObservation:
        ...

    def validate_march_availability(
        self,
        request: AlliancePitRequest,
        pit: AlliancePitObservation,
        policy: AlliancePitPolicy,
    ) -> MarchAvailability:
        ...

    def dispatch_gather_march(
        self,
        request: AlliancePitRequest,
        pit: AlliancePitObservation,
        availability: MarchAvailability,
        policy: AlliancePitPolicy,
    ) -> MarchDispatchResult:
        ...

    def verify_dispatch(
        self,
        request: AlliancePitRequest,
        pit: AlliancePitObservation,
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
class _AlliancePitState:
    request: AlliancePitRequest
    character: Character | None = None
    policy: AlliancePitPolicy | None = None
    pit: AlliancePitObservation | None = None
    march_availability: MarchAvailability | None = None
    dispatch: MarchDispatchResult | None = None
    terminal_outcome: WorkflowOutcome | None = None
    terminal_reason: str = ""
    terminal_state: str = ""
    recovery_outcome: dict[str, object] = field(default_factory=dict)
    screenshot_path: str = ""

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


class AlliancePitGatheringWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: AlliancePitDriver,
        account_precondition: AlliancePitAccountPrecondition | None = None,
        character_precondition: AlliancePitCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        marches: MarchRepository | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: AlliancePitConfig | None = None,
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
        self.config = config or AlliancePitConfig()
        self._states: dict[str, _AlliancePitState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return ALLIANCE_PIT_STATES

    def execute(
        self,
        request: AlliancePitRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _AlliancePitState(request=request)
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
                budget=StepBudget(max_steps=len(ALLIANCE_PIT_STATES) + 6),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"alliance-pit:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"alliance_pit_run_id": token},
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
        for state in ALLIANCE_PIT_STATES:
            registry.register(f"alliance_pit.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.navigation_retry_limit,
            "ensure_character": self.config.navigation_retry_limit,
            "ensure_game_running": self.config.navigation_retry_limit,
            "navigate_to_resource_center": self.config.navigation_retry_limit,
            "clear_overlays": self.config.navigation_retry_limit,
            "detect_pit": self.config.detection_retry_limit,
            "dispatch_march": self.config.dispatch_retry_limit,
            "verify_dispatch": self.config.dispatch_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=ALLIANCE_PIT_WORKFLOW_KEY,
            name="Gather Alliance Pit Resources",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"alliance_pit.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in ALLIANCE_PIT_STATES
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
        state: _AlliancePitState,
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
                "template_keys": list(ALLIANCE_PIT_TEMPLATE_KEYS),
            },
        )

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _AlliancePitState,
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
        state: _AlliancePitState,
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
        state: _AlliancePitState,
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
        state: _AlliancePitState,
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

    def _navigate_to_resource_center(
        self,
        step: WorkflowStepSpec,
        state: _AlliancePitState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.navigate_to_resource_center(state.request, _require_character(state)),
        )

    def _clear_overlays(
        self,
        step: WorkflowStepSpec,
        state: _AlliancePitState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.clear_overlays(state.request, _require_character(state)),
        )

    def _detect_pit(
        self,
        step: WorkflowStepSpec,
        state: _AlliancePitState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        policy = _require_policy(state)
        pit = self.driver.detect_pit(state.request, policy)
        state.pit = pit
        if pit.screenshot_path:
            state.screenshot_path = pit.screenshot_path
        status = pit.normalized_status()
        if status == AlliancePitStatus.NO_ACTIVE_PIT:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                pit.message or "No active alliance resource pit is available.",
                screenshot_path=pit.screenshot_path,
                data={"pit": pit.to_json()},
            )
        if status == AlliancePitStatus.ALREADY_PARTICIPATING:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                pit.message or "Character is already participating in an alliance resource pit.",
                screenshot_path=pit.screenshot_path,
                data={"pit": pit.to_json()},
            )
        if status == AlliancePitStatus.NOT_JOINABLE:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                pit.message or "Alliance resource pit is not joinable.",
                screenshot_path=pit.screenshot_path,
                data={"pit": pit.to_json()},
            )
        resource_type = pit.normalized_resource_type()
        if resource_type is None:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                "Joinable alliance resource pit did not expose a resource type.",
                screenshot_path=pit.screenshot_path,
                data={"pit": pit.to_json()},
            )
        if resource_type not in policy.enabled_resource_types:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                f"Alliance pit resource type {resource_type.value} is disabled by policy.",
                screenshot_path=pit.screenshot_path,
                data={"pit": pit.to_json(), "policy": policy.to_json()},
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"pit": pit.to_json()},
            screenshot_path=pit.screenshot_path,
        )

    def _validate_march(
        self,
        step: WorkflowStepSpec,
        state: _AlliancePitState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        pit = _require_pit(state)
        availability = self.driver.validate_march_availability(state.request, pit, _require_policy(state))
        state.march_availability = availability
        if availability.screenshot_path:
            state.screenshot_path = availability.screenshot_path
        if not availability.available:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                availability.message or "No free march is available for alliance pit gathering.",
                screenshot_path=availability.screenshot_path,
                data={
                    "available_count": availability.available_count,
                    "march_slot": availability.march_slot,
                    **availability.data,
                },
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "available_count": availability.available_count,
                "march_slot": availability.march_slot,
                **availability.data,
            },
            screenshot_path=availability.screenshot_path,
        )

    def _dispatch_march(
        self,
        step: WorkflowStepSpec,
        state: _AlliancePitState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        pit = _require_pit(state)
        availability = _require_availability(state)
        dispatch = self.driver.dispatch_gather_march(state.request, pit, availability, _require_policy(state))
        state.dispatch = dispatch
        if dispatch.screenshot_path:
            state.screenshot_path = dispatch.screenshot_path
        if not dispatch.success:
            return (
                _step_result(
                    step.step_key,
                    WorkflowOutcome.RETRYABLE_FAILURE,
                    dispatch.message or "Alliance pit march dispatch failed.",
                    data=dispatch.data,
                    screenshot_path=dispatch.screenshot_path,
                )
                if dispatch.retryable
                else state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    dispatch.message or "Alliance pit march dispatch failed.",
                    screenshot_path=dispatch.screenshot_path,
                    data=dispatch.data,
                )
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"dispatch": _dispatch_json(dispatch), "march_preset": _require_policy(state).march_preset},
            screenshot_path=dispatch.screenshot_path,
        )

    def _verify_dispatch(
        self,
        step: WorkflowStepSpec,
        state: _AlliancePitState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        result = self.driver.verify_dispatch(state.request, _require_pit(state), _require_dispatch(state))
        step_result = self._action_to_step(step, state, result)
        if step_result.outcome == WorkflowOutcome.SUCCESS:
            self._persist_march_dispatch(state)
        return step_result

    def _complete(self, step: WorkflowStepSpec, state: _AlliancePitState) -> WorkflowStepResult:
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
            data={
                "selected_pit": _pit_json(state.pit),
                "dispatch": _dispatch_json(state.dispatch),
            },
        )

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _AlliancePitState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
            state.recovery_outcome = self._monitor_recovery(state, _job_run_id(context))
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _AlliancePitState) -> WorkflowStepResult:
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
                "selected_pit": _pit_json(state.pit),
                "dispatch": _dispatch_json(state.dispatch),
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _AlliancePitState,
        action: ResourceGatheringActionResult,
    ) -> WorkflowStepResult:
        if action.screenshot_path:
            state.screenshot_path = action.screenshot_path
        if action.success:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        if action.retryable:
            return _step_result(
                step.step_key,
                WorkflowOutcome.RETRYABLE_FAILURE,
                action.message or "Alliance pit gathering action failed.",
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            action.message or "Alliance pit gathering action failed.",
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _persist_march_dispatch(self, state: _AlliancePitState) -> None:
        dispatch = _require_dispatch(state)
        if self.marches is None or dispatch.march_slot is None:
            return
        character = _require_character(state)
        marches = self.marches.list_for_character(int(character.id or 0))
        existing = next((item for item in marches if item.march_slot == dispatch.march_slot), None)
        march = existing or March(character_id=character.id, march_slot=dispatch.march_slot)
        march.status = "alliance_pit_gathering"
        march.expected_return_time = dispatch.expected_return_time
        march.next_action_time = dispatch.expected_return_time
        self.marches.save(march)

    def _state_from_context(self, context: WorkflowExecutionContext) -> _AlliancePitState:
        token = str(context.metadata.get("alliance_pit_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("Alliance pit runtime state is missing.") from exc

    def _open_incident(self, state: _AlliancePitState) -> None:
        if self.incidents is None or not state.failed:
            return
        self.incidents.save(
            Incident(
                incident_key=f"alliance-pit:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="Alliance pit gathering blocked",
                details=state.terminal_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _AlliancePitState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "Alliance pit gathering workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(
        self,
        result: WorkflowExecutionResult,
        state: _AlliancePitState,
    ) -> None:
        if state.terminal_outcome == WorkflowOutcome.BLOCKED and not state.recovery_outcome:
            state.recovery_outcome = self._monitor_recovery(state, result.job_run_id)

    def _monitor_recovery(
        self,
        state: _AlliancePitState,
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
        state: _AlliancePitState,
    ) -> None:
        if state.terminal_outcome in {WorkflowOutcome.SKIPPED, WorkflowOutcome.BLOCKED}:
            result.outcome = state.terminal_outcome
            result.message = state.terminal_reason
        result.result = {
            **dict(result.result),
            "selected_pit": _pit_json(state.pit),
            "march_dispatch": _dispatch_json(state.dispatch),
            "march_preset": _require_policy(state).march_preset if state.policy is not None else "",
            "terminal_state": state.terminal_state,
            "terminal_reason": state.terminal_reason,
            "failure_state": state.terminal_state if result.outcome.is_failure else "",
            "failure_reason": state.terminal_reason if result.outcome.is_failure else "",
            "recovery_outcome": state.recovery_outcome,
        }

    def _update_persisted_run(
        self,
        result: WorkflowExecutionResult,
        state: _AlliancePitState,
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
        action_type=f"alliance_pit.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _resource_type(value: ResourceType | str) -> ResourceType:
    if isinstance(value, ResourceType):
        return value
    try:
        return ResourceType(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in ResourceType)
        raise ValueError(f"Invalid resource type: {value!r}. Expected one of: {valid}.") from exc


def _require_character(state: _AlliancePitState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _AlliancePitState) -> AlliancePitPolicy:
    if state.policy is None:
        raise RuntimeError("Alliance pit policy has not been validated.")
    return state.policy


def _require_pit(state: _AlliancePitState) -> AlliancePitObservation:
    if state.pit is None:
        raise RuntimeError("Alliance pit has not been detected.")
    return state.pit


def _require_availability(state: _AlliancePitState) -> MarchAvailability:
    if state.march_availability is None:
        raise RuntimeError("March availability has not been validated.")
    return state.march_availability


def _require_dispatch(state: _AlliancePitState) -> MarchDispatchResult:
    if state.dispatch is None:
        raise RuntimeError("Alliance pit march has not been dispatched.")
    return state.dispatch


def _pit_json(pit: AlliancePitObservation | None) -> dict[str, object]:
    return pit.to_json() if pit is not None else {}


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
