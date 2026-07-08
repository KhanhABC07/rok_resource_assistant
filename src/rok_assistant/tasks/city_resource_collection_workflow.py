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


CITY_RESOURCE_COLLECTION_WORKFLOW_KEY = "city-resource-collection"
CITY_RESOURCE_COLLECTION_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "normalize_city_home",
    "collect_resources",
    "complete",
    "recover",
    "failed",
    "cancelled",
)


class CityResourceType(StrEnum):
    FOOD = "FOOD"
    WOOD = "WOOD"
    STONE = "STONE"
    GOLD = "GOLD"
    CRYSTAL = "CRYSTAL"


class CityResourceScanStatus(StrEnum):
    READY = "READY"
    NONE_READY = "NONE_READY"
    POPUP_OVERLAY = "POPUP_OVERLAY"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


def _resource_type(value: CityResourceType | str) -> CityResourceType:
    if isinstance(value, CityResourceType):
        return value
    try:
        return CityResourceType(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in CityResourceType)
        raise ValueError(f"Invalid city resource type: {value!r}. Expected one of: {valid}.") from exc


def _require_positive_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _require_non_negative_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be zero or greater.")
    return value


def _require_confidence(value: float, field_name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")


@dataclass(frozen=True)
class CityResourceRoi:
    resource_type: CityResourceType | str
    x: int
    y: int
    width: int
    height: int

    def normalized_resource_type(self) -> CityResourceType:
        return _resource_type(self.resource_type)

    def contains(self, x: int | None, y: int | None) -> bool:
        if x is None or y is None:
            return False
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def to_json(self) -> dict[str, object]:
        return {
            "resource_type": self.normalized_resource_type().value,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class CityLayoutProfile:
    profile_id: str
    screen_width: int
    screen_height: int
    rois: tuple[CityResourceRoi, ...]

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("CityLayoutProfile.profile_id must be configured.")
        _require_positive_int(self.screen_width, "CityLayoutProfile.screen_width")
        _require_positive_int(self.screen_height, "CityLayoutProfile.screen_height")
        normalized_rois = tuple(
            CityResourceRoi(
                _resource_type(roi.resource_type),
                _require_non_negative_int(roi.x, "CityResourceRoi.x"),
                _require_non_negative_int(roi.y, "CityResourceRoi.y"),
                _require_positive_int(roi.width, "CityResourceRoi.width"),
                _require_positive_int(roi.height, "CityResourceRoi.height"),
            )
            for roi in self.rois
        )
        if not normalized_rois:
            raise ValueError("CityLayoutProfile.rois must contain at least one semantic ROI.")
        resource_types = [roi.normalized_resource_type() for roi in normalized_rois]
        if len(resource_types) != len(set(resource_types)):
            raise ValueError("CityLayoutProfile.rois cannot define duplicate resource types.")
        object.__setattr__(self, "profile_id", self.profile_id.strip())
        object.__setattr__(self, "rois", normalized_rois)

    def roi_for(self, resource_type: CityResourceType | str) -> CityResourceRoi | None:
        normalized = _resource_type(resource_type)
        return next((roi for roi in self.rois if roi.normalized_resource_type() == normalized), None)

    def to_json(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "rois": [roi.to_json() for roi in self.rois],
        }


DEFAULT_CITY_LAYOUT_PROFILE = CityLayoutProfile(
    profile_id="default-16x9",
    screen_width=1280,
    screen_height=720,
    rois=(
        CityResourceRoi(CityResourceType.FOOD, 880, 74, 76, 52),
        CityResourceRoi(CityResourceType.WOOD, 958, 74, 76, 52),
        CityResourceRoi(CityResourceType.STONE, 1036, 74, 76, 52),
        CityResourceRoi(CityResourceType.GOLD, 1114, 74, 76, 52),
    ),
)


@dataclass(frozen=True)
class CityResourceCollectionPolicy:
    layout_profile: CityLayoutProfile = DEFAULT_CITY_LAYOUT_PROFILE
    enabled_resource_types: tuple[CityResourceType | str, ...] = (
        CityResourceType.FOOD,
        CityResourceType.WOOD,
        CityResourceType.STONE,
        CityResourceType.GOLD,
    )
    minimum_detector_confidence: float = 0.85
    overlap_distance_pixels: int = 18

    def normalized(self) -> CityResourceCollectionPolicy:
        enabled = tuple(dict.fromkeys(_resource_type(item) for item in self.enabled_resource_types))
        if not enabled:
            raise ValueError("At least one city resource type must be enabled.")
        missing = [item.value for item in enabled if self.layout_profile.roi_for(item) is None]
        if missing:
            raise ValueError(f"Layout profile is missing ROI definitions for: {', '.join(missing)}.")
        _require_confidence(self.minimum_detector_confidence, "minimum_detector_confidence")
        distance = _require_non_negative_int(self.overlap_distance_pixels, "overlap_distance_pixels")
        return CityResourceCollectionPolicy(
            layout_profile=self.layout_profile,
            enabled_resource_types=enabled,
            minimum_detector_confidence=float(self.minimum_detector_confidence),
            overlap_distance_pixels=distance,
        )

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "layout_profile": normalized.layout_profile.to_json(),
            "enabled_resource_types": [item.value for item in normalized.enabled_resource_types],
            "minimum_detector_confidence": normalized.minimum_detector_confidence,
            "overlap_distance_pixels": normalized.overlap_distance_pixels,
        }


@dataclass(frozen=True)
class CityResourceCollectionRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: CityResourceCollectionPolicy = field(default_factory=CityResourceCollectionPolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class CityResourceCollectionConfig:
    workflow_timeout_seconds: float = 90.0
    step_timeout_seconds: float = 15.0
    precondition_retry_limit: int = 1
    collection_retry_limit: int = 0
    retry_delay_seconds: float = 0.25
    max_passes: int = 3

    def normalized_max_passes(self) -> int:
        return _require_positive_int(self.max_passes, "max_passes")


@dataclass(frozen=True)
class CityResourceObservation:
    resource_type: CityResourceType | str
    confidence: float
    x: int | None = None
    y: int | None = None
    indicator_id: str = ""
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_resource_type(self) -> CityResourceType:
        return _resource_type(self.resource_type)

    def to_json(self) -> dict[str, object]:
        return {
            "resource_type": self.normalized_resource_type().value,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            "indicator_id": self.indicator_id,
            **self.data,
        }


@dataclass(frozen=True)
class CityResourceScan:
    status: CityResourceScanStatus | str
    observations: tuple[CityResourceObservation, ...] = ()
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_status(self) -> CityResourceScanStatus:
        if isinstance(self.status, CityResourceScanStatus):
            return self.status
        try:
            return CityResourceScanStatus(str(self.status).strip().upper())
        except ValueError as exc:
            valid = ", ".join(item.value for item in CityResourceScanStatus)
            raise ValueError(f"Invalid city resource scan status: {self.status!r}. Expected one of: {valid}.") from exc

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.normalized_status().value,
            "observations": [item.to_json() for item in self.observations],
            **self.data,
        }


@dataclass(frozen=True)
class CityResourceCollectionResult:
    success: bool
    changed: bool = False
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)


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


class CityResourceAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: CityResourceCollectionRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class CityResourceCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: CityResourceCollectionRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class CityResourceCollectionDriver(Protocol):
    def normalize_to_city_home(
        self,
        request: CityResourceCollectionRequest,
        character: Character,
        policy: CityResourceCollectionPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def scan_city_resources(
        self,
        request: CityResourceCollectionRequest,
        character: Character,
        policy: CityResourceCollectionPolicy,
        pass_number: int,
    ) -> CityResourceScan:
        ...

    def clear_overlays(
        self,
        request: CityResourceCollectionRequest,
        character: Character,
        policy: CityResourceCollectionPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def collect_city_resource(
        self,
        request: CityResourceCollectionRequest,
        character: Character,
        observation: CityResourceObservation,
        policy: CityResourceCollectionPolicy,
    ) -> CityResourceCollectionResult:
        ...

    def verify_city_resource_collected(
        self,
        request: CityResourceCollectionRequest,
        character: Character,
        observation: CityResourceObservation,
        policy: CityResourceCollectionPolicy,
    ) -> CityResourceCollectionResult:
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
class _CityResourceState:
    request: CityResourceCollectionRequest
    character: Character | None = None
    policy: CityResourceCollectionPolicy | None = None
    collected_resource_types: list[CityResourceType] = field(default_factory=list)
    collection_attempts: list[dict[str, object]] = field(default_factory=list)
    scan_attempts: list[dict[str, object]] = field(default_factory=list)
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


class CityResourceCollectionWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: CityResourceCollectionDriver,
        account_precondition: CityResourceAccountPrecondition | None = None,
        character_precondition: CityResourceCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: CityResourceCollectionConfig | None = None,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.account_precondition = account_precondition
        self.character_precondition = character_precondition
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.incidents = incidents
        self.config = config or CityResourceCollectionConfig()
        self._states: dict[str, _CityResourceState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return CITY_RESOURCE_COLLECTION_STATES

    def execute(
        self,
        request: CityResourceCollectionRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _CityResourceState(request=request)
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
                budget=StepBudget(max_steps=len(CITY_RESOURCE_COLLECTION_STATES) + self.config.normalized_max_passes() + 8),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"city-resource-collection:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"city_resource_collection_run_id": token},
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
        for state in CITY_RESOURCE_COLLECTION_STATES:
            registry.register(f"city_resource_collection.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.precondition_retry_limit,
            "ensure_character": self.config.precondition_retry_limit,
            "ensure_game_running": self.config.precondition_retry_limit,
            "normalize_city_home": self.config.precondition_retry_limit,
            "collect_resources": self.config.collection_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=CITY_RESOURCE_COLLECTION_WORKFLOW_KEY,
            name="Collect City Resources",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"city_resource_collection.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in CITY_RESOURCE_COLLECTION_STATES
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
        state: _CityResourceState,
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
            max_passes = self.config.normalized_max_passes()
        except ValueError as exc:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, str(exc))
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"policy": state.policy.to_json(), "max_passes": max_passes},
        )

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _CityResourceState,
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
        state: _CityResourceState,
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
        state: _CityResourceState,
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
        state: _CityResourceState,
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

    def _normalize_city_home(
        self,
        step: WorkflowStepSpec,
        state: _CityResourceState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.normalize_to_city_home(
                state.request,
                _require_character(state),
                _require_policy(state),
            ),
        )

    def _collect_resources(
        self,
        step: WorkflowStepSpec,
        state: _CityResourceState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        character = _require_character(state)
        policy = _require_policy(state)
        collected_this_run: set[CityResourceType] = set()
        saw_ready = False
        for pass_number in range(1, self.config.normalized_max_passes() + 1):
            context.cancellation_token.throw_if_cancelled()
            clicked_this_pass: set[CityResourceType] = set()
            scan = self.driver.scan_city_resources(state.request, character, policy, pass_number)
            self._record_scan(state, pass_number, scan)
            if scan.screenshot_path:
                state.screenshot_path = scan.screenshot_path
            status = scan.normalized_status()
            if status == CityResourceScanStatus.VERIFICATION_REQUIRED:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.FATAL_FAILURE,
                    scan.message or "Verification screen requires manual intervention.",
                    screenshot_path=scan.screenshot_path,
                    data={"scan": scan.to_json()},
                )
            if status == CityResourceScanStatus.POPUP_OVERLAY:
                cleared = self.driver.clear_overlays(state.request, character, policy)
                overlay_step = self._action_to_step(step, state, cleared)
                if overlay_step.outcome != WorkflowOutcome.SUCCESS:
                    return overlay_step
                continue
            if status == CityResourceScanStatus.UNKNOWN:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    scan.message or "City resource readiness could not be determined.",
                    screenshot_path=scan.screenshot_path,
                    data={"scan": scan.to_json()},
                )
            ready = _ready_observations(scan.observations, policy)
            if not ready:
                if not saw_ready and pass_number == 1 and status == CityResourceScanStatus.NONE_READY:
                    return state.stop(
                        step.step_key,
                        WorkflowOutcome.SKIPPED,
                        scan.message or "No city resources are ready to collect.",
                        screenshot_path=scan.screenshot_path,
                        data={"scan": scan.to_json()},
                    )
                break
            saw_ready = True
            for observation in ready:
                resource_type = observation.normalized_resource_type()
                if resource_type in clicked_this_pass:
                    continue
                clicked_this_pass.add(resource_type)
                click_result = self.driver.collect_city_resource(state.request, character, observation, policy)
                if click_result.screenshot_path:
                    state.screenshot_path = click_result.screenshot_path
                if not click_result.success:
                    return self._collection_failure(step, state, click_result)
                verify_result = self.driver.verify_city_resource_collected(state.request, character, observation, policy)
                if verify_result.screenshot_path:
                    state.screenshot_path = verify_result.screenshot_path
                self._record_collection(state, pass_number, observation, click_result, verify_result)
                if not verify_result.success or not verify_result.changed:
                    return self._collection_failure(
                        step,
                        state,
                        verify_result,
                        fallback_message=f"City {resource_type.value.lower()} collection was not verified.",
                    )
                if resource_type not in collected_this_run:
                    collected_this_run.add(resource_type)
                    state.collected_resource_types.append(resource_type)
        if not state.collected_resource_types:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                "No city resources were collected within the configured pass bound.",
                screenshot_path=state.screenshot_path,
                data={"scan_attempts": state.scan_attempts},
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "collected_resource_types": [item.value for item in state.collected_resource_types],
                "collection_attempts": state.collection_attempts,
                "scan_attempts": state.scan_attempts,
            },
            screenshot_path=state.screenshot_path,
        )

    def _complete(self, step: WorkflowStepSpec, state: _CityResourceState) -> WorkflowStepResult:
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
            data={"collected_resource_types": [item.value for item in state.collected_resource_types]},
        )

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _CityResourceState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if _is_verification_stop(state):
            state.recovery_outcome = {"attempted": False, "reason": "verification_screen"}
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
        state.recovery_outcome = self._monitor_recovery(state, _job_run_id(context))
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _CityResourceState) -> WorkflowStepResult:
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
                "collected_resource_types": [item.value for item in state.collected_resource_types],
                "collection_attempts": state.collection_attempts,
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _CityResourceState,
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
                action.message or "City resource collection action failed.",
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            action.message or "City resource collection action failed.",
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _collection_failure(
        self,
        step: WorkflowStepSpec,
        state: _CityResourceState,
        result: CityResourceCollectionResult,
        *,
        fallback_message: str = "City resource collection failed.",
    ) -> WorkflowStepResult:
        if result.retryable:
            return _step_result(
                step.step_key,
                WorkflowOutcome.RETRYABLE_FAILURE,
                result.message or fallback_message,
                data=result.data,
                screenshot_path=result.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            result.message or fallback_message,
            screenshot_path=result.screenshot_path,
            data=result.data,
        )

    def _record_scan(
        self,
        state: _CityResourceState,
        pass_number: int,
        scan: CityResourceScan,
    ) -> None:
        state.scan_attempts.append(
            {
                "pass_number": pass_number,
                "status": scan.normalized_status().value,
                "observation_count": len(scan.observations),
                "screenshot_path": scan.screenshot_path,
                **scan.data,
            }
        )

    def _record_collection(
        self,
        state: _CityResourceState,
        pass_number: int,
        observation: CityResourceObservation,
        click_result: CityResourceCollectionResult,
        verify_result: CityResourceCollectionResult,
    ) -> None:
        state.collection_attempts.append(
            {
                "pass_number": pass_number,
                "resource_type": observation.normalized_resource_type().value,
                "indicator_id": observation.indicator_id,
                "confidence": observation.confidence,
                "x": observation.x,
                "y": observation.y,
                "click_success": click_result.success,
                "verified_changed": verify_result.changed,
                **click_result.data,
                **verify_result.data,
            }
        )

    def _state_from_context(self, context: WorkflowExecutionContext) -> _CityResourceState:
        token = str(context.metadata.get("city_resource_collection_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("City resource collection runtime state is missing.") from exc

    def _open_incident(self, state: _CityResourceState) -> None:
        if self.incidents is None or not state.failed or state.incident_opened:
            return
        self.incidents.save(
            Incident(
                incident_key=f"city-resource-collection:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="City resource collection blocked",
                details=state.terminal_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )
        state.incident_opened = True

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _CityResourceState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "City resource collection workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(
        self,
        result: WorkflowExecutionResult,
        state: _CityResourceState,
    ) -> None:
        if state.failed and not state.recovery_outcome:
            if _is_verification_stop(state):
                state.recovery_outcome = {"attempted": False, "reason": "verification_screen"}
            else:
                state.recovery_outcome = self._monitor_recovery(state, result.job_run_id)
        if state.failed:
            self._open_incident(state)

    def _monitor_recovery(
        self,
        state: _CityResourceState,
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
        state: _CityResourceState,
    ) -> None:
        if state.terminal_outcome in {WorkflowOutcome.SKIPPED, WorkflowOutcome.BLOCKED}:
            result.outcome = state.terminal_outcome
            result.message = state.terminal_reason
        result.result = {
            **dict(result.result),
            "policy": state.policy.to_json() if state.policy is not None else {},
            "collected_resource_types": [item.value for item in state.collected_resource_types],
            "scan_attempts": state.scan_attempts,
            "collection_attempts": state.collection_attempts,
            "terminal_state": state.terminal_state,
            "terminal_reason": state.terminal_reason,
            "failure_state": state.terminal_state if result.outcome.is_failure else "",
            "failure_reason": state.terminal_reason if result.outcome.is_failure else "",
            "recovery_outcome": state.recovery_outcome,
        }

    def _update_persisted_run(
        self,
        result: WorkflowExecutionResult,
        state: _CityResourceState,
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


def _ready_observations(
    observations: tuple[CityResourceObservation, ...],
    policy: CityResourceCollectionPolicy,
) -> list[CityResourceObservation]:
    candidates = []
    for observation in observations:
        resource_type = observation.normalized_resource_type()
        if resource_type not in policy.enabled_resource_types:
            continue
        if observation.confidence < policy.minimum_detector_confidence:
            continue
        roi = policy.layout_profile.roi_for(resource_type)
        if roi is None or not roi.contains(observation.x, observation.y):
            continue
        candidates.append(observation)
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    selected: list[CityResourceObservation] = []
    selected_types: set[CityResourceType] = set()
    for observation in candidates:
        resource_type = observation.normalized_resource_type()
        if resource_type in selected_types:
            continue
        if any(_points_overlap(observation, existing, policy.overlap_distance_pixels) for existing in selected):
            continue
        selected.append(observation)
        selected_types.add(resource_type)
    return sorted(selected, key=lambda item: item.normalized_resource_type().value)


def _points_overlap(
    left: CityResourceObservation,
    right: CityResourceObservation,
    distance: int,
) -> bool:
    if left.x is None or left.y is None or right.x is None or right.y is None:
        return False
    return math.hypot(left.x - right.x, left.y - right.y) <= distance


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
        action_type=f"city_resource_collection.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _CityResourceState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _CityResourceState) -> CityResourceCollectionPolicy:
    if state.policy is None:
        raise RuntimeError("City resource collection policy has not been validated.")
    return state.policy


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_verification_stop(state: _CityResourceState) -> bool:
    return "verification" in state.terminal_reason.lower()
