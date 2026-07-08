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


CITY_MATERIAL_PRODUCTION_WORKFLOW_KEY = "city-material-production"
CITY_MATERIAL_PRODUCTION_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "open_material_production",
    "inspect_queue_state",
    "select_material",
    "start_production",
    "verify_production_state",
    "complete",
    "skipped",
    "recover",
    "failed",
    "cancelled",
)


class CityMaterial(StrEnum):
    LEATHER = "LEATHER"
    IRON = "IRON"
    EBONY = "EBONY"
    BONE = "BONE"


class MaterialQuality(StrEnum):
    NORMAL = "NORMAL"
    ADVANCED = "ADVANCED"
    ELITE = "ELITE"
    EPIC = "EPIC"
    LEGENDARY = "LEGENDARY"


class MaterialQueueStatus(StrEnum):
    IDLE = "IDLE"
    BUSY = "BUSY"
    COOLDOWN = "COOLDOWN"
    READY = "READY"
    INSUFFICIENT_RESOURCES = "INSUFFICIENT_RESOURCES"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


def _material(value: CityMaterial | str) -> CityMaterial:
    if isinstance(value, CityMaterial):
        return value
    try:
        return CityMaterial(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in CityMaterial)
        raise ValueError(f"Invalid material: {value!r}. Expected one of: {valid}.") from exc


def _quality(value: MaterialQuality | str) -> MaterialQuality:
    if isinstance(value, MaterialQuality):
        return value
    try:
        return MaterialQuality(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in MaterialQuality)
        raise ValueError(f"Invalid material quality: {value!r}. Expected one of: {valid}.") from exc


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
class MaterialProductionPolicy:
    material_priority: tuple[CityMaterial | str, ...] = (
        CityMaterial.LEATHER,
        CityMaterial.IRON,
        CityMaterial.EBONY,
        CityMaterial.BONE,
    )
    allowed_qualities: tuple[MaterialQuality | str, ...] = (MaterialQuality.NORMAL,)
    minimum_tier: int = 1
    maximum_tier: int = 1
    minimum_detector_confidence: float = 0.85
    allow_overwrite: bool = False
    allow_acceleration: bool = False

    def normalized(self) -> MaterialProductionPolicy:
        priority = tuple(dict.fromkeys(_material(item) for item in self.material_priority))
        if not priority:
            raise ValueError("At least one material must be configured in priority order.")
        qualities = tuple(dict.fromkeys(_quality(item) for item in self.allowed_qualities))
        if not qualities:
            raise ValueError("At least one material quality must be allowed.")
        minimum_tier = _require_positive_int(self.minimum_tier, "minimum_tier")
        maximum_tier = _require_positive_int(self.maximum_tier, "maximum_tier")
        if maximum_tier < minimum_tier:
            raise ValueError("maximum_tier must be greater than or equal to minimum_tier.")
        return MaterialProductionPolicy(
            material_priority=priority,
            allowed_qualities=qualities,
            minimum_tier=minimum_tier,
            maximum_tier=maximum_tier,
            minimum_detector_confidence=_require_confidence(
                self.minimum_detector_confidence,
                "minimum_detector_confidence",
            ),
            allow_overwrite=bool(self.allow_overwrite),
            allow_acceleration=bool(self.allow_acceleration),
        )

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "material_priority": [item.value for item in normalized.material_priority],
            "allowed_qualities": [item.value for item in normalized.allowed_qualities],
            "minimum_tier": normalized.minimum_tier,
            "maximum_tier": normalized.maximum_tier,
            "minimum_detector_confidence": normalized.minimum_detector_confidence,
            "allow_overwrite": normalized.allow_overwrite,
            "allow_acceleration": normalized.allow_acceleration,
        }


@dataclass(frozen=True)
class MaterialProductionRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: MaterialProductionPolicy = field(default_factory=MaterialProductionPolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class MaterialProductionConfig:
    workflow_timeout_seconds: float = 120.0
    step_timeout_seconds: float = 15.0
    precondition_retry_limit: int = 1
    navigation_retry_limit: int = 1
    inspect_retry_limit: int = 1
    action_retry_limit: int = 0
    retry_delay_seconds: float = 0.25


@dataclass(frozen=True)
class MaterialProductionOption:
    material: CityMaterial | str
    quality: MaterialQuality | str = MaterialQuality.NORMAL
    tier: int = 1
    enabled: bool = True
    resources_available: bool = True
    confidence: float = 1.0
    scene_verified: bool = True
    material_verified: bool = True
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_material(self) -> CityMaterial:
        return _material(self.material)

    def normalized_quality(self) -> MaterialQuality:
        return _quality(self.quality)

    def to_json(self) -> dict[str, object]:
        return {
            "material": self.normalized_material().value,
            "quality": self.normalized_quality().value,
            "tier": self.tier,
            "enabled": self.enabled,
            "resources_available": self.resources_available,
            "confidence": self.confidence,
            "scene_verified": self.scene_verified,
            "material_verified": self.material_verified,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class MaterialQueueState:
    status: MaterialQueueStatus | str
    available_options: tuple[MaterialProductionOption, ...] = ()
    active_material: CityMaterial | str | None = None
    active_quality: MaterialQuality | str | None = None
    active_tier: int | None = None
    queue_size: int = 0
    cooldown_seconds: int = 0
    scene_verified: bool = True
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_status(self) -> MaterialQueueStatus:
        if isinstance(self.status, MaterialQueueStatus):
            return self.status
        try:
            return MaterialQueueStatus(str(self.status).strip().upper())
        except ValueError as exc:
            valid = ", ".join(item.value for item in MaterialQueueStatus)
            raise ValueError(f"Invalid material queue status: {self.status!r}. Expected one of: {valid}.") from exc

    def to_json(self) -> dict[str, object]:
        active_material = _material(self.active_material).value if self.active_material is not None else ""
        active_quality = _quality(self.active_quality).value if self.active_quality is not None else ""
        return {
            "status": self.normalized_status().value,
            "available_options": [item.to_json() for item in self.available_options],
            "active_material": active_material,
            "active_quality": active_quality,
            "active_tier": self.active_tier,
            "queue_size": self.queue_size,
            "cooldown_seconds": self.cooldown_seconds,
            "scene_verified": self.scene_verified,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class MaterialProductionStartResult:
    success: bool
    changed: bool = False
    queue_size: int | None = None
    cooldown_seconds: int | None = None
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "changed": self.changed,
            "queue_size": self.queue_size,
            "cooldown_seconds": self.cooldown_seconds,
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


class MaterialProductionAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: MaterialProductionRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class MaterialProductionCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: MaterialProductionRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class MaterialProductionDriver(Protocol):
    def open_material_production(
        self,
        request: MaterialProductionRequest,
        character: Character,
        policy: MaterialProductionPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def inspect_material_queue(
        self,
        request: MaterialProductionRequest,
        character: Character,
        policy: MaterialProductionPolicy,
    ) -> MaterialQueueState:
        ...

    def select_material(
        self,
        request: MaterialProductionRequest,
        character: Character,
        option: MaterialProductionOption,
        policy: MaterialProductionPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def start_material_production(
        self,
        request: MaterialProductionRequest,
        character: Character,
        option: MaterialProductionOption,
        before: MaterialQueueState,
        policy: MaterialProductionPolicy,
    ) -> MaterialProductionStartResult:
        ...

    def verify_material_production_state(
        self,
        request: MaterialProductionRequest,
        character: Character,
        option: MaterialProductionOption,
        before: MaterialQueueState,
        start: MaterialProductionStartResult,
        policy: MaterialProductionPolicy,
    ) -> MaterialProductionStartResult:
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
class _MaterialProductionState:
    request: MaterialProductionRequest
    character: Character | None = None
    policy: MaterialProductionPolicy | None = None
    queue_state: MaterialQueueState | None = None
    selected_material: MaterialProductionOption | None = None
    start_result: MaterialProductionStartResult | None = None
    inspection_attempts: list[dict[str, object]] = field(default_factory=list)
    selection_attempts: list[dict[str, object]] = field(default_factory=list)
    production_attempts: list[dict[str, object]] = field(default_factory=list)
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


class MaterialProductionWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: MaterialProductionDriver,
        account_precondition: MaterialProductionAccountPrecondition | None = None,
        character_precondition: MaterialProductionCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: MaterialProductionConfig | None = None,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.account_precondition = account_precondition
        self.character_precondition = character_precondition
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.incidents = incidents
        self.config = config or MaterialProductionConfig()
        self._states: dict[str, _MaterialProductionState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return CITY_MATERIAL_PRODUCTION_STATES

    def execute(
        self,
        request: MaterialProductionRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _MaterialProductionState(request=request)
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
                budget=StepBudget(max_steps=len(CITY_MATERIAL_PRODUCTION_STATES) + 16),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"city-material-production:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"city_material_production_run_id": token},
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
        for state in CITY_MATERIAL_PRODUCTION_STATES:
            registry.register(f"city_material_production.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.precondition_retry_limit,
            "ensure_character": self.config.precondition_retry_limit,
            "ensure_game_running": self.config.precondition_retry_limit,
            "open_material_production": self.config.navigation_retry_limit,
            "inspect_queue_state": self.config.inspect_retry_limit,
            "select_material": self.config.inspect_retry_limit,
            "start_production": self.config.action_retry_limit,
            "verify_production_state": self.config.inspect_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=CITY_MATERIAL_PRODUCTION_WORKFLOW_KEY,
            name="Produce City Materials",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"city_material_production.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in CITY_MATERIAL_PRODUCTION_STATES
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
        state: _MaterialProductionState,
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
        state: _MaterialProductionState,
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
        state: _MaterialProductionState,
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
        state: _MaterialProductionState,
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
        state: _MaterialProductionState,
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

    def _open_material_production(
        self,
        step: WorkflowStepSpec,
        state: _MaterialProductionState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.open_material_production(
                state.request,
                _require_character(state),
                _require_policy(state),
            ),
        )

    def _inspect_queue_state(
        self,
        step: WorkflowStepSpec,
        state: _MaterialProductionState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        queue = self.driver.inspect_material_queue(state.request, _require_character(state), _require_policy(state))
        state.queue_state = queue
        if queue.screenshot_path:
            state.screenshot_path = queue.screenshot_path
        state.inspection_attempts.append(queue.to_json())
        status = queue.normalized_status()
        if not queue.scene_verified:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                queue.message or "Material production scene could not be verified.",
                screenshot_path=queue.screenshot_path,
                data={"queue_state": queue.to_json()},
            )
        if status == MaterialQueueStatus.VERIFICATION_REQUIRED:
            return state.stop(
                step.step_key,
                WorkflowOutcome.FATAL_FAILURE,
                queue.message or "Verification screen requires manual intervention.",
                screenshot_path=queue.screenshot_path,
                data={"queue_state": queue.to_json()},
            )
        if status == MaterialQueueStatus.UNKNOWN:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                queue.message or "Material production queue state could not be determined.",
                screenshot_path=queue.screenshot_path,
                data={"queue_state": queue.to_json()},
            )
        if status in {MaterialQueueStatus.BUSY, MaterialQueueStatus.COOLDOWN, MaterialQueueStatus.READY} and not (
            _require_policy(state).allow_overwrite or _require_policy(state).allow_acceleration
        ):
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                queue.message or "Material production queue is already busy.",
                screenshot_path=queue.screenshot_path,
                data={"queue_state": queue.to_json()},
            )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"queue_state": queue.to_json()})

    def _select_material(
        self,
        step: WorkflowStepSpec,
        state: _MaterialProductionState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        queue = _require_queue_state(state)
        policy = _require_policy(state)
        selected, ignored = _select_material_option(queue.available_options, policy)
        state.ignored_options = ignored
        if selected is None:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                "No allowed material can be produced.",
                screenshot_path=state.screenshot_path,
                data={"queue_state": queue.to_json(), "ignored_options": ignored},
            )
        if not selected.scene_verified or not selected.material_verified:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                selected.message or "Selected material and tier could not be verified before production.",
                screenshot_path=selected.screenshot_path or state.screenshot_path,
                data={"selected_material": selected.to_json(), "ignored_options": ignored},
            )
        action = self.driver.select_material(state.request, _require_character(state), selected, policy)
        state.selection_attempts.append({"selected_material": selected.to_json(), **action.data})
        if action.screenshot_path:
            state.screenshot_path = action.screenshot_path
        if not action.success:
            return self._action_to_step(step, state, action)
        state.selected_material = selected
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"selected_material": selected.to_json(), "ignored_options": ignored, **action.data},
            screenshot_path=action.screenshot_path,
        )

    def _start_production(
        self,
        step: WorkflowStepSpec,
        state: _MaterialProductionState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        result = self.driver.start_material_production(
            state.request,
            _require_character(state),
            _require_selected_material(state),
            _require_queue_state(state),
            _require_policy(state),
        )
        state.start_result = result
        if result.screenshot_path:
            state.screenshot_path = result.screenshot_path
        state.production_attempts.append(
            {
                "selected_material": _require_selected_material(state).to_json(),
                **result.to_json(),
            }
        )
        if not result.success:
            return self._production_failure(step, state, result, "Material production could not be started.")
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=result.to_json(), screenshot_path=result.screenshot_path)

    def _verify_production_state(
        self,
        step: WorkflowStepSpec,
        state: _MaterialProductionState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        verify = self.driver.verify_material_production_state(
            state.request,
            _require_character(state),
            _require_selected_material(state),
            _require_queue_state(state),
            _require_start_result(state),
            _require_policy(state),
        )
        if verify.screenshot_path:
            state.screenshot_path = verify.screenshot_path
        state.verification_attempts.append(
            {
                "selected_material": _require_selected_material(state).to_json(),
                **verify.to_json(),
            }
        )
        if not _production_postcondition_verified(_require_queue_state(state), _require_start_result(state), verify):
            failure = replace(
                verify,
                retryable=False,
                message=("Material production queue or cooldown did not change after starting production. " f"{verify.message}").strip(),
            )
            return self._production_failure(
                step,
                state,
                failure,
                "Material production postcondition was not verified.",
            )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=verify.to_json(), screenshot_path=verify.screenshot_path)

    def _complete(self, step: WorkflowStepSpec, state: _MaterialProductionState) -> WorkflowStepResult:
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
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=self._production_payload(state))

    def _skipped(self, step: WorkflowStepSpec, state: _MaterialProductionState) -> WorkflowStepResult:
        if state.terminal_outcome == WorkflowOutcome.SKIPPED:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data={"skipped_reason": state.terminal_reason, **self._production_payload(state)},
                screenshot_path=state.screenshot_path,
            )
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED)

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _MaterialProductionState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if _is_manual_stop(state):
            state.recovery_outcome = {"attempted": False, "reason": "manual_intervention_required"}
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
        state.recovery_outcome = self._monitor_recovery(state, _job_run_id(context))
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _MaterialProductionState) -> WorkflowStepResult:
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
                **self._production_payload(state),
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _MaterialProductionState,
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
                action.message or "Material production action failed.",
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            action.message or "Material production action failed.",
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _production_failure(
        self,
        step: WorkflowStepSpec,
        state: _MaterialProductionState,
        result: MaterialProductionStartResult,
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

    def _production_payload(self, state: _MaterialProductionState) -> dict[str, object]:
        return {
            "selected_material": state.selected_material.to_json() if state.selected_material is not None else {},
            "queue_state": state.queue_state.to_json() if state.queue_state is not None else {},
            "inspection_attempts": state.inspection_attempts,
            "selection_attempts": state.selection_attempts,
            "production_attempts": state.production_attempts,
            "verification_attempts": state.verification_attempts,
            "verification_result": state.verification_attempts[-1] if state.verification_attempts else {},
            "ignored_options": state.ignored_options,
            "failure_evidence": {
                "screenshot_path": state.screenshot_path,
                "terminal_state": state.terminal_state,
                "terminal_reason": state.terminal_reason,
            },
        }

    def _state_from_context(self, context: WorkflowExecutionContext) -> _MaterialProductionState:
        token = str(context.metadata.get("city_material_production_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("City material production runtime state is missing.") from exc

    def _open_incident(self, state: _MaterialProductionState) -> None:
        if self.incidents is None or not state.failed or state.incident_opened:
            return
        self.incidents.save(
            Incident(
                incident_key=f"city-material-production:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="City material production blocked",
                details=state.terminal_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )
        state.incident_opened = True

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _MaterialProductionState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "City material production workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(
        self,
        result: WorkflowExecutionResult,
        state: _MaterialProductionState,
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
        state: _MaterialProductionState,
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
        state: _MaterialProductionState,
    ) -> None:
        if state.terminal_outcome in {WorkflowOutcome.SKIPPED, WorkflowOutcome.BLOCKED}:
            result.outcome = state.terminal_outcome
            result.message = state.terminal_reason
        result.result = {
            **dict(result.result),
            "policy": state.policy.to_json() if state.policy is not None else {},
            **self._production_payload(state),
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
        state: _MaterialProductionState,
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


def _select_material_option(
    options: tuple[MaterialProductionOption, ...],
    policy: MaterialProductionPolicy,
) -> tuple[MaterialProductionOption | None, list[dict[str, object]]]:
    ignored: list[dict[str, object]] = []
    by_material: dict[CityMaterial, list[MaterialProductionOption]] = {}
    for option in options:
        reason = _option_skip_reason(option, policy)
        if reason:
            ignored.append({**option.to_json(), "ignored_reason": reason})
            continue
        by_material.setdefault(option.normalized_material(), []).append(option)
    for material in policy.material_priority:
        candidates = by_material.get(material, [])
        if candidates:
            return sorted(candidates, key=lambda item: (item.tier, item.normalized_quality().value))[0], ignored
    return None, ignored


def _option_skip_reason(option: MaterialProductionOption, policy: MaterialProductionPolicy) -> str:
    material = option.normalized_material()
    quality = option.normalized_quality()
    if material not in policy.material_priority:
        return "material_not_in_priority"
    if quality not in policy.allowed_qualities:
        return "quality_not_allowed"
    if option.tier < policy.minimum_tier or option.tier > policy.maximum_tier:
        return "tier_not_allowed"
    if not option.enabled:
        return "disabled"
    if not option.resources_available:
        return "insufficient_resources"
    if option.confidence < policy.minimum_detector_confidence:
        return "below_confidence_threshold"
    return ""


def _production_postcondition_verified(
    before: MaterialQueueState,
    start: MaterialProductionStartResult,
    verify: MaterialProductionStartResult,
) -> bool:
    if not verify.success or not verify.changed:
        return False
    if start.changed:
        return True
    if verify.queue_size is not None and verify.queue_size != before.queue_size:
        return True
    if verify.cooldown_seconds is not None and verify.cooldown_seconds != before.cooldown_seconds:
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
        action_type=f"city_material_production.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _MaterialProductionState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _MaterialProductionState) -> MaterialProductionPolicy:
    if state.policy is None:
        raise RuntimeError("Material production policy has not been validated.")
    return state.policy


def _require_queue_state(state: _MaterialProductionState) -> MaterialQueueState:
    if state.queue_state is None:
        raise RuntimeError("Material production queue has not been inspected.")
    return state.queue_state


def _require_selected_material(state: _MaterialProductionState) -> MaterialProductionOption:
    if state.selected_material is None:
        raise RuntimeError("Material production option has not been selected.")
    return state.selected_material


def _require_start_result(state: _MaterialProductionState) -> MaterialProductionStartResult:
    if state.start_result is None:
        raise RuntimeError("Material production has not been started.")
    return state.start_result


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_manual_stop(state: _MaterialProductionState) -> bool:
    text = state.terminal_reason.lower()
    return "verification" in text or "manual" in text
