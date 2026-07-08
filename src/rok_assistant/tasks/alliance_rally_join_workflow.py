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


ALLIANCE_RALLY_JOIN_WORKFLOW_KEY = "alliance-rally-join"
ALLIANCE_RALLY_JOIN_TEMPLATE_KEYS = (
    "city.alliance.button",
    "alliance.menu.war",
    "alliance.war.rally_list",
    "alliance.war.rally_candidate",
    "alliance.war.rally_join_button",
    "alliance.war.rally_full",
    "march.free_slot",
    "march.preset.selector",
    "march.dispatch.button",
    "march.dispatch.started_indicator",
    "alliance.war.joined_rally_indicator",
)
ALLIANCE_RALLY_JOIN_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "open_alliance_war",
    "inspect_rallies",
    "select_rally",
    "verify_march",
    "verify_troops",
    "join_rally",
    "choose_march_preset",
    "dispatch_march",
    "verify_joined",
    "complete",
    "recover",
    "failed",
    "cancelled",
)


class RallyPolicyDecisionCode(StrEnum):
    ALLOWED = "ALLOWED"
    DENIED_TYPE = "DENIED_TYPE"
    DENIED_LEADER = "DENIED_LEADER"
    DENIED_TARGET = "DENIED_TARGET"
    DURATION_BELOW_MINIMUM = "DURATION_BELOW_MINIMUM"
    DURATION_ABOVE_MAXIMUM = "DURATION_ABOVE_MAXIMUM"
    NO_CAPACITY = "NO_CAPACITY"


@dataclass(frozen=True)
class AllianceRallyJoinPolicy:
    allowed_rally_types: tuple[str, ...] = ()
    allowed_leaders: tuple[str, ...] = ()
    allowed_targets: tuple[str, ...] = ()
    minimum_duration_seconds: int | None = None
    maximum_duration_seconds: int | None = None
    march_preset: str = "default"
    minimum_remaining_capacity: int = 1

    def normalized(self) -> AllianceRallyJoinPolicy:
        preset = self.march_preset.strip()
        if not preset:
            raise ValueError("Alliance rally march preset must be configured.")
        minimum = _optional_non_negative_int(self.minimum_duration_seconds, "minimum_duration_seconds")
        maximum = _optional_non_negative_int(self.maximum_duration_seconds, "maximum_duration_seconds")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("minimum_duration_seconds cannot be greater than maximum_duration_seconds.")
        capacity = int(self.minimum_remaining_capacity)
        if capacity < 0:
            raise ValueError("minimum_remaining_capacity must be zero or greater.")
        return AllianceRallyJoinPolicy(
            allowed_rally_types=_normalized_set(self.allowed_rally_types),
            allowed_leaders=_normalized_set(self.allowed_leaders),
            allowed_targets=_normalized_set(self.allowed_targets),
            minimum_duration_seconds=minimum,
            maximum_duration_seconds=maximum,
            march_preset=preset,
            minimum_remaining_capacity=capacity,
        )

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "allowed_rally_types": list(normalized.allowed_rally_types),
            "allowed_leaders": list(normalized.allowed_leaders),
            "allowed_targets": list(normalized.allowed_targets),
            "minimum_duration_seconds": normalized.minimum_duration_seconds,
            "maximum_duration_seconds": normalized.maximum_duration_seconds,
            "march_preset": normalized.march_preset,
            "minimum_remaining_capacity": normalized.minimum_remaining_capacity,
        }


@dataclass(frozen=True)
class AllianceRallyJoinRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: AllianceRallyJoinPolicy = field(default_factory=AllianceRallyJoinPolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class AllianceRallyJoinConfig:
    workflow_timeout_seconds: float = 180.0
    step_timeout_seconds: float = 20.0
    navigation_retry_limit: int = 1
    scan_retry_limit: int = 1
    dispatch_retry_limit: int = 1
    retry_delay_seconds: float = 0.25


@dataclass(frozen=True)
class AllianceRallyObservation:
    rally_id: str
    rally_type: str
    leader_name: str
    target_name: str
    duration_seconds: int
    capacity_available: bool = True
    remaining_capacity: int | None = None
    confidence: float = 0.0
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "rally_id": self.rally_id,
            "rally_type": self.rally_type,
            "leader_name": self.leader_name,
            "target_name": self.target_name,
            "duration_seconds": self.duration_seconds,
            "capacity_available": self.capacity_available,
            "remaining_capacity": self.remaining_capacity,
            "confidence": self.confidence,
            **self.data,
        }


@dataclass(frozen=True)
class AllianceRallyScan:
    rallies: tuple[AllianceRallyObservation, ...] = ()
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "rallies": [rally.to_json() for rally in self.rallies],
            "message": self.message,
            **self.data,
        }


@dataclass(frozen=True)
class AllianceRallyPolicyDecision:
    allowed: bool
    code: RallyPolicyDecisionCode
    reason: str
    rally_id: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "code": self.code.value,
            "reason": self.reason,
            "rally_id": self.rally_id,
        }


@dataclass(frozen=True)
class TroopAvailability:
    available: bool
    troop_count: int = 0
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "available": self.available,
            "troop_count": self.troop_count,
            "message": self.message,
            **self.data,
        }


@dataclass(frozen=True)
class RallyJoinedVerification:
    joined: bool
    rally_id: str = ""
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "joined": self.joined,
            "rally_id": self.rally_id,
            "message": self.message,
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


class AllianceRallyAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: AllianceRallyJoinRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class AllianceRallyCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: AllianceRallyJoinRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class AllianceRallyJoinDriver(Protocol):
    def open_alliance_war(
        self,
        request: AllianceRallyJoinRequest,
        character: Character,
        policy: AllianceRallyJoinPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def inspect_rallies(
        self,
        request: AllianceRallyJoinRequest,
        character: Character,
        policy: AllianceRallyJoinPolicy,
    ) -> AllianceRallyScan:
        ...

    def select_rally(
        self,
        request: AllianceRallyJoinRequest,
        character: Character,
        rally: AllianceRallyObservation,
        policy: AllianceRallyJoinPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def verify_march_availability(
        self,
        request: AllianceRallyJoinRequest,
        character: Character,
        rally: AllianceRallyObservation,
        policy: AllianceRallyJoinPolicy,
    ) -> MarchAvailability:
        ...

    def verify_troop_availability(
        self,
        request: AllianceRallyJoinRequest,
        character: Character,
        rally: AllianceRallyObservation,
        policy: AllianceRallyJoinPolicy,
    ) -> TroopAvailability:
        ...

    def join_rally(
        self,
        request: AllianceRallyJoinRequest,
        character: Character,
        rally: AllianceRallyObservation,
        policy: AllianceRallyJoinPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def choose_march_preset(
        self,
        request: AllianceRallyJoinRequest,
        character: Character,
        rally: AllianceRallyObservation,
        policy: AllianceRallyJoinPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def dispatch_march(
        self,
        request: AllianceRallyJoinRequest,
        character: Character,
        rally: AllianceRallyObservation,
        availability: MarchAvailability,
        policy: AllianceRallyJoinPolicy,
    ) -> MarchDispatchResult:
        ...

    def verify_joined(
        self,
        request: AllianceRallyJoinRequest,
        character: Character,
        rally: AllianceRallyObservation,
        dispatch: MarchDispatchResult,
        policy: AllianceRallyJoinPolicy,
    ) -> RallyJoinedVerification:
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
class _AllianceRallyJoinState:
    request: AllianceRallyJoinRequest
    character: Character | None = None
    policy: AllianceRallyJoinPolicy | None = None
    scan: AllianceRallyScan | None = None
    selected_rally: AllianceRallyObservation | None = None
    policy_decisions: list[AllianceRallyPolicyDecision] = field(default_factory=list)
    march_availability: MarchAvailability | None = None
    troop_availability: TroopAvailability | None = None
    join_result: ResourceGatheringActionResult | None = None
    preset_result: ResourceGatheringActionResult | None = None
    dispatch_result: MarchDispatchResult | None = None
    joined_verification: RallyJoinedVerification | None = None
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


class AllianceRallyJoinWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: AllianceRallyJoinDriver,
        account_precondition: AllianceRallyAccountPrecondition | None = None,
        character_precondition: AllianceRallyCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        marches: MarchRepository | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: AllianceRallyJoinConfig | None = None,
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
        self.config = config or AllianceRallyJoinConfig()
        self._states: dict[str, _AllianceRallyJoinState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return ALLIANCE_RALLY_JOIN_STATES

    def execute(
        self,
        request: AllianceRallyJoinRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _AllianceRallyJoinState(request=request)
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
                budget=StepBudget(max_steps=len(ALLIANCE_RALLY_JOIN_STATES) + 8),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"alliance-rally:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"alliance_rally_join_run_id": token},
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
        for state in ALLIANCE_RALLY_JOIN_STATES:
            registry.register(f"alliance_rally_join.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.navigation_retry_limit,
            "ensure_character": self.config.navigation_retry_limit,
            "ensure_game_running": self.config.navigation_retry_limit,
            "open_alliance_war": self.config.navigation_retry_limit,
            "inspect_rallies": self.config.scan_retry_limit,
            "dispatch_march": self.config.dispatch_retry_limit,
            "verify_joined": self.config.dispatch_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=ALLIANCE_RALLY_JOIN_WORKFLOW_KEY,
            name="Join Alliance Rally",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"alliance_rally_join.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in ALLIANCE_RALLY_JOIN_STATES
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
        state: _AllianceRallyJoinState,
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
                "template_keys": list(ALLIANCE_RALLY_JOIN_TEMPLATE_KEYS),
            },
        )

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _AllianceRallyJoinState,
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
        state: _AllianceRallyJoinState,
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
        state: _AllianceRallyJoinState,
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
        state: _AllianceRallyJoinState,
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

    def _open_alliance_war(
        self,
        step: WorkflowStepSpec,
        state: _AllianceRallyJoinState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.open_alliance_war(state.request, _require_character(state), _require_policy(state)),
        )

    def _inspect_rallies(
        self,
        step: WorkflowStepSpec,
        state: _AllianceRallyJoinState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        scan = self.driver.inspect_rallies(state.request, _require_character(state), _require_policy(state))
        state.scan = scan
        if scan.screenshot_path:
            state.screenshot_path = scan.screenshot_path
        if not scan.rallies:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                scan.message or "No alliance rallies are available.",
                screenshot_path=scan.screenshot_path,
                data={"scan": scan.to_json()},
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"scan": scan.to_json()},
            screenshot_path=scan.screenshot_path,
        )

    def _select_rally(
        self,
        step: WorkflowStepSpec,
        state: _AllianceRallyJoinState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        policy = _require_policy(state)
        scan = _require_scan(state)
        for rally in scan.rallies:
            decision = _decide_rally(policy, rally)
            state.policy_decisions.append(decision)
            if decision.allowed:
                state.selected_rally = rally
                if rally.screenshot_path:
                    state.screenshot_path = rally.screenshot_path
                action = self.driver.select_rally(state.request, _require_character(state), rally, policy)
                result = self._action_to_step(step, state, action)
                result.data.update(
                    {
                        "selected_rally": rally.to_json(),
                        "policy_decision": decision.to_json(),
                        "policy_decisions": _decision_json(state.policy_decisions),
                    }
                )
                return result
        return state.stop(
            step.step_key,
            WorkflowOutcome.SKIPPED,
            "No eligible alliance rally matched the configured whitelist policy.",
            screenshot_path=state.screenshot_path,
            data={
                "policy": policy.to_json(),
                "policy_decisions": _decision_json(state.policy_decisions),
                "scan": scan.to_json(),
            },
        )

    def _verify_march(
        self,
        step: WorkflowStepSpec,
        state: _AllianceRallyJoinState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        availability = self.driver.verify_march_availability(
            state.request,
            _require_character(state),
            _require_rally(state),
            _require_policy(state),
        )
        state.march_availability = availability
        if availability.screenshot_path:
            state.screenshot_path = availability.screenshot_path
        if not availability.available:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                availability.message or "No free march is available for alliance rally joining.",
                screenshot_path=availability.screenshot_path,
                data=_march_availability_json(availability),
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data=_march_availability_json(availability),
            screenshot_path=availability.screenshot_path,
        )

    def _verify_troops(
        self,
        step: WorkflowStepSpec,
        state: _AllianceRallyJoinState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        availability = self.driver.verify_troop_availability(
            state.request,
            _require_character(state),
            _require_rally(state),
            _require_policy(state),
        )
        state.troop_availability = availability
        if availability.screenshot_path:
            state.screenshot_path = availability.screenshot_path
        if not availability.available:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                availability.message or "No available troops can join the selected rally.",
                screenshot_path=availability.screenshot_path,
                data={"troop_availability": availability.to_json()},
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"troop_availability": availability.to_json()},
            screenshot_path=availability.screenshot_path,
        )

    def _join_rally(
        self,
        step: WorkflowStepSpec,
        state: _AllianceRallyJoinState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        action = self.driver.join_rally(state.request, _require_character(state), _require_rally(state), _require_policy(state))
        state.join_result = action
        return self._action_to_step(step, state, action)

    def _choose_march_preset(
        self,
        step: WorkflowStepSpec,
        state: _AllianceRallyJoinState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        action = self.driver.choose_march_preset(
            state.request,
            _require_character(state),
            _require_rally(state),
            _require_policy(state),
        )
        state.preset_result = action
        result = self._action_to_step(step, state, action)
        result.data.setdefault("march_preset", _require_policy(state).march_preset)
        return result

    def _dispatch_march(
        self,
        step: WorkflowStepSpec,
        state: _AllianceRallyJoinState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        dispatch = self.driver.dispatch_march(
            state.request,
            _require_character(state),
            _require_rally(state),
            _require_march_availability(state),
            _require_policy(state),
        )
        state.dispatch_result = dispatch
        if dispatch.screenshot_path:
            state.screenshot_path = dispatch.screenshot_path
        if not dispatch.success:
            return (
                _step_result(
                    step.step_key,
                    WorkflowOutcome.RETRYABLE_FAILURE,
                    dispatch.message or "Alliance rally march dispatch failed.",
                    data={"dispatch": _dispatch_json(dispatch)},
                    screenshot_path=dispatch.screenshot_path,
                )
                if dispatch.retryable
                else state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    dispatch.message or "Alliance rally march dispatch failed.",
                    screenshot_path=dispatch.screenshot_path,
                    data={"dispatch": _dispatch_json(dispatch)},
                )
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"dispatch": _dispatch_json(dispatch), "march_preset": _require_policy(state).march_preset},
            screenshot_path=dispatch.screenshot_path,
        )

    def _verify_joined(
        self,
        step: WorkflowStepSpec,
        state: _AllianceRallyJoinState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        verification = self.driver.verify_joined(
            state.request,
            _require_character(state),
            _require_rally(state),
            _require_dispatch(state),
            _require_policy(state),
        )
        state.joined_verification = verification
        if verification.screenshot_path:
            state.screenshot_path = verification.screenshot_path
        if not verification.joined:
            return (
                _step_result(
                    step.step_key,
                    WorkflowOutcome.RETRYABLE_FAILURE,
                    verification.message or "Alliance rally joined state was not verified.",
                    data={"joined_verification": verification.to_json()},
                    screenshot_path=verification.screenshot_path,
                )
                if verification.retryable
                else state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    verification.message or "Alliance rally joined state was not verified.",
                    screenshot_path=verification.screenshot_path,
                    data={"joined_verification": verification.to_json()},
                )
            )
        self._persist_march_dispatch(state)
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"joined_verification": verification.to_json()},
            screenshot_path=verification.screenshot_path,
        )

    def _complete(self, step: WorkflowStepSpec, state: _AllianceRallyJoinState) -> WorkflowStepResult:
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
                "selected_rally": _rally_json(state.selected_rally),
                "policy_decisions": _decision_json(state.policy_decisions),
                "march_dispatch": _dispatch_json(state.dispatch_result),
                "joined_verification": _verification_json(state.joined_verification),
            },
        )

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _AllianceRallyJoinState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        state.recovery_outcome = self._monitor_recovery(state, _job_run_id(context))
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _AllianceRallyJoinState) -> WorkflowStepResult:
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
                "selected_rally": _rally_json(state.selected_rally),
                "policy_decisions": _decision_json(state.policy_decisions),
                "dispatch": _dispatch_json(state.dispatch_result),
                "joined_verification": _verification_json(state.joined_verification),
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _AllianceRallyJoinState,
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
                action.message or "Alliance rally join action failed.",
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            action.message or "Alliance rally join action failed.",
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _persist_march_dispatch(self, state: _AllianceRallyJoinState) -> None:
        dispatch = _require_dispatch(state)
        if self.marches is None or dispatch.march_slot is None:
            return
        character = _require_character(state)
        marches = self.marches.list_for_character(int(character.id or 0))
        existing = next((item for item in marches if item.march_slot == dispatch.march_slot), None)
        march = existing or March(character_id=character.id, march_slot=dispatch.march_slot)
        march.status = "alliance_rally_joined"
        march.expected_return_time = dispatch.expected_return_time
        march.next_action_time = dispatch.expected_return_time
        self.marches.save(march)

    def _state_from_context(self, context: WorkflowExecutionContext) -> _AllianceRallyJoinState:
        token = str(context.metadata.get("alliance_rally_join_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("Alliance rally join runtime state is missing.") from exc

    def _open_incident(self, state: _AllianceRallyJoinState) -> None:
        if self.incidents is None or not state.failed:
            return
        self.incidents.save(
            Incident(
                incident_key=f"alliance-rally:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="Alliance rally join blocked",
                details=state.terminal_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _AllianceRallyJoinState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "Alliance rally join workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(
        self,
        result: WorkflowExecutionResult,
        state: _AllianceRallyJoinState,
    ) -> None:
        if state.terminal_outcome == WorkflowOutcome.BLOCKED and not state.recovery_outcome:
            state.recovery_outcome = self._monitor_recovery(state, result.job_run_id)

    def _monitor_recovery(
        self,
        state: _AllianceRallyJoinState,
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
        state: _AllianceRallyJoinState,
    ) -> None:
        if state.terminal_outcome in {WorkflowOutcome.SKIPPED, WorkflowOutcome.BLOCKED}:
            result.outcome = state.terminal_outcome
            result.message = state.terminal_reason
        failure_evidence = {}
        if result.outcome.is_failure:
            failure_evidence = {
                "terminal_state": state.terminal_state,
                "terminal_reason": state.terminal_reason,
                "screenshot_path": state.screenshot_path,
                "selected_rally": _rally_json(state.selected_rally),
                "policy_decisions": _decision_json(state.policy_decisions),
            }
        result.result = {
            **dict(result.result),
            "selected_rally": _rally_json(state.selected_rally),
            "policy_decisions": _decision_json(state.policy_decisions),
            "march_preset": _require_policy(state).march_preset if state.policy is not None else "",
            "march_availability": _march_availability_json(state.march_availability),
            "troop_availability": _troop_json(state.troop_availability),
            "join_result": _action_json(state.join_result),
            "preset_result": _action_json(state.preset_result),
            "dispatch_result": _dispatch_json(state.dispatch_result),
            "joined_verification": _verification_json(state.joined_verification),
            "terminal_state": state.terminal_state,
            "terminal_reason": state.terminal_reason,
            "failure_state": state.terminal_state if result.outcome.is_failure else "",
            "failure_reason": state.terminal_reason if result.outcome.is_failure else "",
            "failure_evidence": failure_evidence,
            "recovery_outcome": state.recovery_outcome,
        }

    def _update_persisted_run(
        self,
        result: WorkflowExecutionResult,
        state: _AllianceRallyJoinState,
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


def _decide_rally(
    policy: AllianceRallyJoinPolicy,
    rally: AllianceRallyObservation,
) -> AllianceRallyPolicyDecision:
    rally_id = rally.rally_id
    if policy.allowed_rally_types and _norm(rally.rally_type) not in policy.allowed_rally_types:
        return AllianceRallyPolicyDecision(False, RallyPolicyDecisionCode.DENIED_TYPE, "Rally type is not whitelisted.", rally_id)
    if policy.allowed_leaders and _norm(rally.leader_name) not in policy.allowed_leaders:
        return AllianceRallyPolicyDecision(False, RallyPolicyDecisionCode.DENIED_LEADER, "Rally leader is not whitelisted.", rally_id)
    if policy.allowed_targets and _norm(rally.target_name) not in policy.allowed_targets:
        return AllianceRallyPolicyDecision(False, RallyPolicyDecisionCode.DENIED_TARGET, "Rally target is not whitelisted.", rally_id)
    if policy.minimum_duration_seconds is not None and rally.duration_seconds < policy.minimum_duration_seconds:
        return AllianceRallyPolicyDecision(False, RallyPolicyDecisionCode.DURATION_BELOW_MINIMUM, "Rally duration is below policy minimum.", rally_id)
    if policy.maximum_duration_seconds is not None and rally.duration_seconds > policy.maximum_duration_seconds:
        return AllianceRallyPolicyDecision(False, RallyPolicyDecisionCode.DURATION_ABOVE_MAXIMUM, "Rally duration is above policy maximum.", rally_id)
    remaining = rally.remaining_capacity
    if not rally.capacity_available or (remaining is not None and remaining < policy.minimum_remaining_capacity):
        return AllianceRallyPolicyDecision(False, RallyPolicyDecisionCode.NO_CAPACITY, "Rally capacity is unavailable.", rally_id)
    return AllianceRallyPolicyDecision(True, RallyPolicyDecisionCode.ALLOWED, "Rally matches whitelist policy.", rally_id)


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
        action_type=f"alliance_rally_join.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _norm(value: str) -> str:
    return " ".join(str(value).strip().upper().split())


def _normalized_set(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_norm(value) for value in values if _norm(value)))


def _optional_non_negative_int(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{field_name} must be zero or greater.")
    return converted


def _require_character(state: _AllianceRallyJoinState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _AllianceRallyJoinState) -> AllianceRallyJoinPolicy:
    if state.policy is None:
        raise RuntimeError("Alliance rally join policy has not been validated.")
    return state.policy


def _require_scan(state: _AllianceRallyJoinState) -> AllianceRallyScan:
    if state.scan is None:
        raise RuntimeError("Alliance rallies have not been scanned.")
    return state.scan


def _require_rally(state: _AllianceRallyJoinState) -> AllianceRallyObservation:
    if state.selected_rally is None:
        raise RuntimeError("Alliance rally has not been selected.")
    return state.selected_rally


def _require_march_availability(state: _AllianceRallyJoinState) -> MarchAvailability:
    if state.march_availability is None:
        raise RuntimeError("March availability has not been verified.")
    return state.march_availability


def _require_dispatch(state: _AllianceRallyJoinState) -> MarchDispatchResult:
    if state.dispatch_result is None:
        raise RuntimeError("Alliance rally march has not been dispatched.")
    return state.dispatch_result


def _rally_json(rally: AllianceRallyObservation | None) -> dict[str, object]:
    return rally.to_json() if rally is not None else {}


def _decision_json(decisions: list[AllianceRallyPolicyDecision]) -> list[dict[str, object]]:
    return [decision.to_json() for decision in decisions]


def _march_availability_json(availability: MarchAvailability | None) -> dict[str, object]:
    if availability is None:
        return {}
    return {
        "available": availability.available,
        "march_slot": availability.march_slot,
        "available_count": availability.available_count,
        "message": availability.message,
        **availability.data,
    }


def _troop_json(availability: TroopAvailability | None) -> dict[str, object]:
    return availability.to_json() if availability is not None else {}


def _action_json(action: ResourceGatheringActionResult | None) -> dict[str, object]:
    if action is None:
        return {}
    return {
        "success": action.success,
        "message": action.message,
        **action.data,
    }


def _dispatch_json(dispatch: MarchDispatchResult | None) -> dict[str, object]:
    if dispatch is None:
        return {}
    return {
        "success": dispatch.success,
        "march_slot": dispatch.march_slot,
        "dispatch_id": dispatch.dispatch_id,
        "expected_return_time": dispatch.expected_return_time,
        "message": dispatch.message,
        **dispatch.data,
    }


def _verification_json(verification: RallyJoinedVerification | None) -> dict[str, object]:
    return verification.to_json() if verification is not None else {}


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
