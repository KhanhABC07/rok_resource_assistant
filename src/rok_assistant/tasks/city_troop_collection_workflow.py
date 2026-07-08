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


CITY_TROOP_COLLECTION_WORKFLOW_KEY = "city-troop-collection"
CITY_TROOP_COLLECTION_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "normalize_city_home",
    "collect_completed_troops",
    "handle_result_panel",
    "complete",
    "recover",
    "failed",
    "cancelled",
)


class TroopBuilding(StrEnum):
    BARRACKS = "BARRACKS"
    STABLE = "STABLE"
    ARCHERY_RANGE = "ARCHERY_RANGE"
    SIEGE_WORKSHOP = "SIEGE_WORKSHOP"


class TroopIndicatorType(StrEnum):
    COMPLETED_TRAINING = "COMPLETED_TRAINING"
    SPEED_UP = "SPEED_UP"
    UPGRADE = "UPGRADE"
    AMBIGUOUS = "AMBIGUOUS"


class CityTroopScanStatus(StrEnum):
    READY = "READY"
    NONE_READY = "NONE_READY"
    RESULT_PANEL = "RESULT_PANEL"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


def _building(value: TroopBuilding | str) -> TroopBuilding:
    if isinstance(value, TroopBuilding):
        return value
    try:
        return TroopBuilding(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in TroopBuilding)
        raise ValueError(f"Invalid troop building: {value!r}. Expected one of: {valid}.") from exc


def _indicator_type(value: TroopIndicatorType | str) -> TroopIndicatorType:
    if isinstance(value, TroopIndicatorType):
        return value
    try:
        return TroopIndicatorType(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in TroopIndicatorType)
        raise ValueError(f"Invalid troop indicator type: {value!r}. Expected one of: {valid}.") from exc


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
class TroopBuildingRoi:
    building: TroopBuilding | str
    x: int
    y: int
    width: int
    height: int

    def normalized_building(self) -> TroopBuilding:
        return _building(self.building)

    def contains(self, x: int | None, y: int | None) -> bool:
        if x is None or y is None:
            return False
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def to_json(self) -> dict[str, object]:
        return {
            "building": self.normalized_building().value,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class CityTroopLayoutProfile:
    profile_id: str
    screen_width: int
    screen_height: int
    rois: tuple[TroopBuildingRoi, ...]

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("CityTroopLayoutProfile.profile_id must be configured.")
        _require_positive_int(self.screen_width, "CityTroopLayoutProfile.screen_width")
        _require_positive_int(self.screen_height, "CityTroopLayoutProfile.screen_height")
        normalized_rois = tuple(
            TroopBuildingRoi(
                _building(roi.building),
                _require_non_negative_int(roi.x, "TroopBuildingRoi.x"),
                _require_non_negative_int(roi.y, "TroopBuildingRoi.y"),
                _require_positive_int(roi.width, "TroopBuildingRoi.width"),
                _require_positive_int(roi.height, "TroopBuildingRoi.height"),
            )
            for roi in self.rois
        )
        if not normalized_rois:
            raise ValueError("CityTroopLayoutProfile.rois must contain at least one troop building ROI.")
        buildings = [roi.normalized_building() for roi in normalized_rois]
        if len(buildings) != len(set(buildings)):
            raise ValueError("CityTroopLayoutProfile.rois cannot define duplicate troop buildings.")
        object.__setattr__(self, "profile_id", self.profile_id.strip())
        object.__setattr__(self, "rois", normalized_rois)

    def roi_for(self, building: TroopBuilding | str) -> TroopBuildingRoi | None:
        normalized = _building(building)
        return next((roi for roi in self.rois if roi.normalized_building() == normalized), None)

    def to_json(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "rois": [roi.to_json() for roi in self.rois],
        }


DEFAULT_CITY_TROOP_LAYOUT_PROFILE = CityTroopLayoutProfile(
    profile_id="default-16x9",
    screen_width=1280,
    screen_height=720,
    rois=(
        TroopBuildingRoi(TroopBuilding.BARRACKS, 320, 280, 180, 150),
        TroopBuildingRoi(TroopBuilding.STABLE, 500, 270, 180, 150),
        TroopBuildingRoi(TroopBuilding.ARCHERY_RANGE, 680, 270, 180, 150),
        TroopBuildingRoi(TroopBuilding.SIEGE_WORKSHOP, 860, 280, 180, 150),
    ),
)


@dataclass(frozen=True)
class CityTroopCollectionPolicy:
    layout_profile: CityTroopLayoutProfile = DEFAULT_CITY_TROOP_LAYOUT_PROFILE
    enabled_buildings: tuple[TroopBuilding | str, ...] = (
        TroopBuilding.BARRACKS,
        TroopBuilding.STABLE,
        TroopBuilding.ARCHERY_RANGE,
        TroopBuilding.SIEGE_WORKSHOP,
    )
    minimum_indicator_confidence: float = 0.85
    overlap_distance_pixels: int = 18

    def normalized(self) -> CityTroopCollectionPolicy:
        enabled = tuple(dict.fromkeys(_building(item) for item in self.enabled_buildings))
        if not enabled:
            raise ValueError("At least one troop building must be enabled.")
        missing = [item.value for item in enabled if self.layout_profile.roi_for(item) is None]
        if missing:
            raise ValueError(f"Layout profile is missing ROI definitions for: {', '.join(missing)}.")
        _require_confidence(self.minimum_indicator_confidence, "minimum_indicator_confidence")
        distance = _require_non_negative_int(self.overlap_distance_pixels, "overlap_distance_pixels")
        return CityTroopCollectionPolicy(
            layout_profile=self.layout_profile,
            enabled_buildings=enabled,
            minimum_indicator_confidence=float(self.minimum_indicator_confidence),
            overlap_distance_pixels=distance,
        )

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "layout_profile": normalized.layout_profile.to_json(),
            "enabled_buildings": [item.value for item in normalized.enabled_buildings],
            "minimum_indicator_confidence": normalized.minimum_indicator_confidence,
            "overlap_distance_pixels": normalized.overlap_distance_pixels,
        }


@dataclass(frozen=True)
class CityTroopCollectionRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: CityTroopCollectionPolicy = field(default_factory=CityTroopCollectionPolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class CityTroopCollectionConfig:
    workflow_timeout_seconds: float = 90.0
    step_timeout_seconds: float = 15.0
    precondition_retry_limit: int = 1
    collection_retry_limit: int = 0
    result_panel_retry_limit: int = 1
    retry_delay_seconds: float = 0.25
    max_scan_passes: int = 4

    def normalized_max_scan_passes(self) -> int:
        return _require_positive_int(self.max_scan_passes, "max_scan_passes")


@dataclass(frozen=True)
class CityTroopObservation:
    building: TroopBuilding | str
    indicator_type: TroopIndicatorType | str
    confidence: float
    x: int | None = None
    y: int | None = None
    indicator_id: str = ""
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_building(self) -> TroopBuilding:
        return _building(self.building)

    def normalized_indicator_type(self) -> TroopIndicatorType:
        return _indicator_type(self.indicator_type)

    def to_json(self) -> dict[str, object]:
        return {
            "building": self.normalized_building().value,
            "indicator_type": self.normalized_indicator_type().value,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            "indicator_id": self.indicator_id,
            **self.data,
        }


@dataclass(frozen=True)
class CityTroopScan:
    status: CityTroopScanStatus | str
    observations: tuple[CityTroopObservation, ...] = ()
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_status(self) -> CityTroopScanStatus:
        if isinstance(self.status, CityTroopScanStatus):
            return self.status
        try:
            return CityTroopScanStatus(str(self.status).strip().upper())
        except ValueError as exc:
            valid = ", ".join(item.value for item in CityTroopScanStatus)
            raise ValueError(f"Invalid city troop scan status: {self.status!r}. Expected one of: {valid}.") from exc

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.normalized_status().value,
            "observations": [item.to_json() for item in self.observations],
            **self.data,
        }


@dataclass(frozen=True)
class CityTroopCollectionResult:
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


class CityTroopAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: CityTroopCollectionRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class CityTroopCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: CityTroopCollectionRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class CityTroopCollectionDriver(Protocol):
    def normalize_to_city_home(
        self,
        request: CityTroopCollectionRequest,
        character: Character,
        policy: CityTroopCollectionPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def scan_completed_troops(
        self,
        request: CityTroopCollectionRequest,
        character: Character,
        policy: CityTroopCollectionPolicy,
        pass_number: int,
    ) -> CityTroopScan:
        ...

    def click_completed_troop_indicator(
        self,
        request: CityTroopCollectionRequest,
        character: Character,
        observation: CityTroopObservation,
        policy: CityTroopCollectionPolicy,
    ) -> CityTroopCollectionResult:
        ...

    def verify_completed_troop_collected(
        self,
        request: CityTroopCollectionRequest,
        character: Character,
        observation: CityTroopObservation,
        policy: CityTroopCollectionPolicy,
    ) -> CityTroopCollectionResult:
        ...

    def handle_troop_collection_result_panel(
        self,
        request: CityTroopCollectionRequest,
        character: Character,
        policy: CityTroopCollectionPolicy,
    ) -> CityTroopCollectionResult:
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
class _CityTroopState:
    request: CityTroopCollectionRequest
    character: Character | None = None
    policy: CityTroopCollectionPolicy | None = None
    collected_buildings: list[TroopBuilding] = field(default_factory=list)
    collection_attempts: list[dict[str, object]] = field(default_factory=list)
    scan_attempts: list[dict[str, object]] = field(default_factory=list)
    result_panel_attempts: list[dict[str, object]] = field(default_factory=list)
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


class CityTroopCollectionWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: CityTroopCollectionDriver,
        account_precondition: CityTroopAccountPrecondition | None = None,
        character_precondition: CityTroopCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: CityTroopCollectionConfig | None = None,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.account_precondition = account_precondition
        self.character_precondition = character_precondition
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.incidents = incidents
        self.config = config or CityTroopCollectionConfig()
        self._states: dict[str, _CityTroopState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return CITY_TROOP_COLLECTION_STATES

    def execute(
        self,
        request: CityTroopCollectionRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _CityTroopState(request=request)
        self._states[token] = state
        persistence = None
        if self.job_runs is not None and self.step_runs is not None and request.job_id is not None:
            persistence = WorkflowRunRepositoryRecorder(self.job_runs, self.step_runs)
        try:
            max_passes = self.config.normalized_max_scan_passes()
            context = WorkflowExecutionContext(
                cancellation_token=cancellation_token or CancellationToken(),
                deadline=WorkflowDeadline.from_timeout(
                    self.config.workflow_timeout_seconds,
                    time.monotonic,
                ),
                budget=StepBudget(max_steps=len(CITY_TROOP_COLLECTION_STATES) + max_passes + 10),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"city-troop-collection:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"city_troop_collection_run_id": token},
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
        for state in CITY_TROOP_COLLECTION_STATES:
            registry.register(f"city_troop_collection.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.precondition_retry_limit,
            "ensure_character": self.config.precondition_retry_limit,
            "ensure_game_running": self.config.precondition_retry_limit,
            "normalize_city_home": self.config.precondition_retry_limit,
            "collect_completed_troops": self.config.collection_retry_limit,
            "handle_result_panel": self.config.result_panel_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=CITY_TROOP_COLLECTION_WORKFLOW_KEY,
            name="Collect Completed Troops",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"city_troop_collection.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in CITY_TROOP_COLLECTION_STATES
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
        state: _CityTroopState,
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
            max_passes = self.config.normalized_max_scan_passes()
        except ValueError as exc:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, str(exc))
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"policy": state.policy.to_json(), "max_scan_passes": max_passes},
        )

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _CityTroopState,
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
        state: _CityTroopState,
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
        state: _CityTroopState,
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
        state: _CityTroopState,
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
        state: _CityTroopState,
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

    def _collect_completed_troops(
        self,
        step: WorkflowStepSpec,
        state: _CityTroopState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        character = _require_character(state)
        policy = _require_policy(state)
        saw_ready = False
        for pass_number in range(1, self.config.normalized_max_scan_passes() + 1):
            context.cancellation_token.throw_if_cancelled()
            scan = self.driver.scan_completed_troops(state.request, character, policy, pass_number)
            self._record_scan(state, pass_number, scan)
            if scan.screenshot_path:
                state.screenshot_path = scan.screenshot_path
            status = scan.normalized_status()
            if status == CityTroopScanStatus.VERIFICATION_REQUIRED:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.FATAL_FAILURE,
                    scan.message or "Verification screen requires manual intervention.",
                    screenshot_path=scan.screenshot_path,
                    data={"scan": scan.to_json()},
                )
            if status == CityTroopScanStatus.RESULT_PANEL:
                panel_step = self._handle_result_panel_action(step, state)
                if panel_step.outcome != WorkflowOutcome.SUCCESS:
                    return panel_step
                continue
            if status == CityTroopScanStatus.UNKNOWN:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    scan.message or "Completed troop readiness could not be determined.",
                    screenshot_path=scan.screenshot_path,
                    data={"scan": scan.to_json()},
                )
            ready = _ready_observations(scan.observations, policy)
            if not ready:
                if not saw_ready:
                    return state.stop(
                        step.step_key,
                        WorkflowOutcome.SKIPPED,
                        scan.message or "No completed troop training is ready to collect.",
                        screenshot_path=scan.screenshot_path,
                        data={"scan": scan.to_json()},
                    )
                return _step_result(
                    step.step_key,
                    WorkflowOutcome.SUCCESS,
                    data=self._collection_payload(state),
                    screenshot_path=state.screenshot_path,
                )
            saw_ready = True
            clicked_this_pass: set[TroopBuilding] = set()
            for observation in ready:
                building = observation.normalized_building()
                if building in clicked_this_pass:
                    continue
                clicked_this_pass.add(building)
                click_result = self.driver.click_completed_troop_indicator(
                    state.request,
                    character,
                    observation,
                    policy,
                )
                if click_result.screenshot_path:
                    state.screenshot_path = click_result.screenshot_path
                if not click_result.success:
                    return self._collection_failure(step, state, click_result)
                panel_step = self._handle_result_panel_action(step, state)
                if panel_step.outcome != WorkflowOutcome.SUCCESS:
                    return panel_step
                verify_result = self.driver.verify_completed_troop_collected(
                    state.request,
                    character,
                    observation,
                    policy,
                )
                if verify_result.screenshot_path:
                    state.screenshot_path = verify_result.screenshot_path
                self._record_collection(state, pass_number, observation, click_result, verify_result)
                if not verify_result.success or not verify_result.changed:
                    return self._collection_failure(
                        step,
                        state,
                        verify_result,
                        fallback_message=f"{building.value} troop collection was not verified.",
                    )
                state.collected_buildings.append(building)
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            "Completed troop collection loop budget was exhausted before readiness cleared.",
            screenshot_path=state.screenshot_path,
            data=self._collection_payload(state),
        )

    def _handle_result_panel(
        self,
        step: WorkflowStepSpec,
        state: _CityTroopState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.collected_buildings:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        return self._handle_result_panel_action(step, state)

    def _handle_result_panel_action(
        self,
        step: WorkflowStepSpec,
        state: _CityTroopState,
    ) -> WorkflowStepResult:
        result = self.driver.handle_troop_collection_result_panel(
            state.request,
            _require_character(state),
            _require_policy(state),
        )
        if result.screenshot_path:
            state.screenshot_path = result.screenshot_path
        state.result_panel_attempts.append(
            {
                "success": result.success,
                "changed": result.changed,
                "screenshot_path": result.screenshot_path,
                **result.data,
            }
        )
        if result.success:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data=result.data,
                screenshot_path=result.screenshot_path,
            )
        return self._collection_failure(
            step,
            state,
            result,
            fallback_message="Completed troop result panel could not be handled.",
        )

    def _complete(self, step: WorkflowStepSpec, state: _CityTroopState) -> WorkflowStepResult:
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
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=self._collection_payload(state))

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _CityTroopState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if _is_verification_stop(state):
            state.recovery_outcome = {"attempted": False, "reason": "verification_screen"}
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
        state.recovery_outcome = self._monitor_recovery(state, _job_run_id(context))
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _CityTroopState) -> WorkflowStepResult:
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
                **self._collection_payload(state),
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _CityTroopState,
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
                action.message or "City troop collection action failed.",
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            action.message or "City troop collection action failed.",
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _collection_failure(
        self,
        step: WorkflowStepSpec,
        state: _CityTroopState,
        result: CityTroopCollectionResult,
        *,
        fallback_message: str = "City troop collection failed.",
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
        state: _CityTroopState,
        pass_number: int,
        scan: CityTroopScan,
    ) -> None:
        ignored_count = len(scan.observations) - len(_ready_observations(scan.observations, _require_policy(state)))
        state.scan_attempts.append(
            {
                "pass_number": pass_number,
                "status": scan.normalized_status().value,
                "observation_count": len(scan.observations),
                "ignored_observation_count": ignored_count,
                "screenshot_path": scan.screenshot_path,
                **scan.data,
            }
        )

    def _record_collection(
        self,
        state: _CityTroopState,
        pass_number: int,
        observation: CityTroopObservation,
        click_result: CityTroopCollectionResult,
        verify_result: CityTroopCollectionResult,
    ) -> None:
        state.collection_attempts.append(
            {
                "pass_number": pass_number,
                "building": observation.normalized_building().value,
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

    def _collection_payload(self, state: _CityTroopState) -> dict[str, object]:
        return {
            "collected_building_count": len(state.collected_buildings),
            "collected_buildings": [item.value for item in state.collected_buildings],
            "scan_attempts": state.scan_attempts,
            "collection_attempts": state.collection_attempts,
            "result_panel_attempts": state.result_panel_attempts,
            "failure_evidence": {
                "screenshot_path": state.screenshot_path,
                "terminal_state": state.terminal_state,
                "terminal_reason": state.terminal_reason,
            },
        }

    def _state_from_context(self, context: WorkflowExecutionContext) -> _CityTroopState:
        token = str(context.metadata.get("city_troop_collection_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("City troop collection runtime state is missing.") from exc

    def _open_incident(self, state: _CityTroopState) -> None:
        if self.incidents is None or not state.failed or state.incident_opened:
            return
        self.incidents.save(
            Incident(
                incident_key=f"city-troop-collection:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="City troop collection blocked",
                details=state.terminal_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )
        state.incident_opened = True

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _CityTroopState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "City troop collection workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(
        self,
        result: WorkflowExecutionResult,
        state: _CityTroopState,
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
        state: _CityTroopState,
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
        state: _CityTroopState,
    ) -> None:
        if state.terminal_outcome in {WorkflowOutcome.SKIPPED, WorkflowOutcome.BLOCKED}:
            result.outcome = state.terminal_outcome
            result.message = state.terminal_reason
        result.result = {
            **dict(result.result),
            "policy": state.policy.to_json() if state.policy is not None else {},
            **self._collection_payload(state),
            "terminal_state": state.terminal_state,
            "terminal_reason": state.terminal_reason,
            "failure_state": state.terminal_state if result.outcome.is_failure else "",
            "failure_reason": state.terminal_reason if result.outcome.is_failure else "",
            "recovery_outcome": state.recovery_outcome,
        }

    def _update_persisted_run(
        self,
        result: WorkflowExecutionResult,
        state: _CityTroopState,
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
    observations: tuple[CityTroopObservation, ...],
    policy: CityTroopCollectionPolicy,
) -> list[CityTroopObservation]:
    candidates = []
    for observation in observations:
        building = observation.normalized_building()
        if building not in policy.enabled_buildings:
            continue
        if observation.normalized_indicator_type() != TroopIndicatorType.COMPLETED_TRAINING:
            continue
        if observation.confidence < policy.minimum_indicator_confidence:
            continue
        roi = policy.layout_profile.roi_for(building)
        if roi is None or not roi.contains(observation.x, observation.y):
            continue
        candidates.append(observation)
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    selected: list[CityTroopObservation] = []
    selected_buildings: set[TroopBuilding] = set()
    for observation in candidates:
        building = observation.normalized_building()
        if building in selected_buildings:
            continue
        if any(_points_overlap(observation, existing, policy.overlap_distance_pixels) for existing in selected):
            continue
        selected.append(observation)
        selected_buildings.add(building)
    return sorted(selected, key=lambda item: item.normalized_building().value)


def _points_overlap(
    left: CityTroopObservation,
    right: CityTroopObservation,
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
        action_type=f"city_troop_collection.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _CityTroopState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _CityTroopState) -> CityTroopCollectionPolicy:
    if state.policy is None:
        raise RuntimeError("City troop collection policy has not been validated.")
    return state.policy


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_verification_stop(state: _CityTroopState) -> bool:
    return "verification" in state.terminal_reason.lower()
