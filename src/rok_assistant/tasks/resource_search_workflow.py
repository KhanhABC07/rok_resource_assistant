from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from rok_assistant.db.models import (
    Character,
    Incident,
    JobRun,
    March,
    TaskStep,
)
from rok_assistant.paths import PROJECT_ROOT
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


class ResourceType(StrEnum):
    FOOD = "FOOD"
    WOOD = "WOOD"
    STONE = "STONE"
    GOLD = "GOLD"


RESOURCE_SEARCH_TEMPLATE_ROOT = "templates/resource_search"
MIN_RESOURCE_LEVEL = 1
MAX_RESOURCE_LEVEL = 8
RESOURCE_GATHERING_WORKFLOW_KEY = "resource-gathering"
RESOURCE_GATHERING_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "navigate_to_resource_search",
    "select_resource",
    "search_resource",
    "validate_march",
    "dispatch_march",
    "verify_dispatch",
    "complete",
    "recover",
    "failed",
    "cancelled",
)


@dataclass(frozen=True)
class TemplateReadiness:
    ready: bool
    missing_templates: list[str]


def check_template_readiness(
    steps: list[TaskStep],
    base_dir: Path = PROJECT_ROOT,
) -> TemplateReadiness:
    missing_templates: list[str] = []
    checked_templates: set[str] = set()
    for step in sorted(steps, key=lambda item: item.order):
        template_path = str((step.parameters or {}).get("template_path", "")).strip()
        if not template_path or template_path in checked_templates:
            continue
        checked_templates.add(template_path)
        path = Path(template_path)
        resolved_path = path if path.is_absolute() else base_dir / path
        if not resolved_path.is_file():
            missing_templates.append(template_path)
    return TemplateReadiness(not missing_templates, missing_templates)


@dataclass
class ResourceSearchWorkflow:
    resource_type: ResourceType | str
    target_level: int
    fallback_enabled: bool = False
    march_required: bool = True

    def __post_init__(self) -> None:
        self.resource_type = self._validate_resource_type(self.resource_type)
        self.target_level = self._validate_target_level(self.target_level)

    def to_task_steps(self) -> list[TaskStep]:
        steps: list[TaskStep] = []

        if self.march_required:
            steps.extend(
                [
                    self._step(
                        "IfTemplateExists",
                        {"template_path": self._no_free_march_template},
                    ),
                    self._step(
                        "AbortTask",
                        {"reason": "No free march"},
                    ),
                    self._step("EndIf", {}),
                ]
            )

        steps.extend(
            [
                self._step(
                    "ClickTemplate",
                    {"template_path": self._world_map_button_template},
                ),
                self._step(
                    "ClickTemplate",
                    {"template_path": self._open_resource_search_button_template},
                ),
                self._step(
                    "WaitTemplate",
                    {"template_path": self._resource_search_panel_template},
                ),
                self._step(
                    "ClickTemplate",
                    {"template_path": self._resource_icon_template},
                ),
                self._step(
                    "ClickTemplate",
                    {"template_path": self._level_button_template},
                ),
                self._step(
                    "ClickTemplate",
                    {"template_path": self._resource_search_submit_button_template},
                ),
                self._step(
                    "WaitTemplate",
                    {"template_path": self._resource_node_template},
                ),
                self._step(
                    "ClickTemplate",
                    {"template_path": self._resource_node_template},
                ),
                self._step(
                    "WaitTemplate",
                    {"template_path": self._gather_button_template},
                ),
                self._step(
                    "ClickTemplate",
                    {"template_path": self._gather_button_template},
                ),
                self._step(
                    "WaitTemplate",
                    {"template_path": self._new_troop_window_template},
                ),
                self._step(
                    "ClickTemplate",
                    {"template_path": self._new_troop_march_button_template},
                ),
                self._step(
                    "WaitTemplate",
                    {"template_path": self._march_started_indicator_template},
                ),
            ]
        )

        for order, step in enumerate(steps, start=1):
            step.order = order
        return steps

    @property
    def _no_free_march_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/no_free_march.png"

    @property
    def _world_map_button_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/world_map_button.png"

    @property
    def _open_resource_search_button_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/open_resource_search_button.png"

    @property
    def _resource_search_panel_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/resource_search_panel.png"

    @property
    def _resource_icon_template(self) -> str:
        resource_name = self.resource_type.value.lower()
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/{resource_name}_resource_icon.png"

    @property
    def _level_button_template(self) -> str:
        # Placeholder until a real level-selector crop is supplied for each level.
        return (
            f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/"
            f"resource_level_{self.target_level}_selector.png"
        )

    @property
    def _resource_search_submit_button_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/resource_search_submit_button.png"

    @property
    def _resource_node_template(self) -> str:
        resource_name = self.resource_type.value.lower()
        return (
            f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/{resource_name}_node_level_"
            f"{self.target_level}.png"
        )

    @property
    def _gather_button_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/gather_button.png"

    @property
    def _new_troop_window_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/new_troop_window.png"

    @property
    def _new_troop_march_button_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/new_troop_march_button.png"

    @property
    def _march_started_indicator_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/march_started_indicator.png"

    @staticmethod
    def _step(action_type: str, parameters: dict[str, object]) -> TaskStep:
        return TaskStep(action_type=action_type, parameters=parameters)

    @staticmethod
    def _validate_resource_type(resource_type: ResourceType | str) -> ResourceType:
        if isinstance(resource_type, ResourceType):
            return resource_type
        value = str(resource_type).strip().upper()
        try:
            return ResourceType(value)
        except ValueError as exc:
            valid = ", ".join(item.value for item in ResourceType)
            raise ValueError(
                f"Invalid resource_type: {resource_type!r}. Expected one of: {valid}."
            ) from exc

    @staticmethod
    def _validate_target_level(target_level: int) -> int:
        if isinstance(target_level, bool):
            raise ValueError("target_level must be an integer resource level.")
        try:
            value = int(target_level)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_level must be an integer resource level.") from exc
        if value < MIN_RESOURCE_LEVEL or value > MAX_RESOURCE_LEVEL:
            raise ValueError(
                f"target_level must be between {MIN_RESOURCE_LEVEL} and {MAX_RESOURCE_LEVEL}."
            )
        return value


@dataclass(frozen=True)
class ResourcePreference:
    resource_type: ResourceType | str
    target_level: int
    minimum_level: int | None = None
    fallback_allowed: bool = False

    def normalized(self) -> ResourcePreference:
        resource_type = ResourceSearchWorkflow._validate_resource_type(self.resource_type)
        target_level = ResourceSearchWorkflow._validate_target_level(self.target_level)
        minimum_level = self.minimum_level if self.minimum_level is not None else target_level
        minimum_level = ResourceSearchWorkflow._validate_target_level(minimum_level)
        if minimum_level > target_level:
            raise ValueError("minimum_level cannot be greater than target_level.")
        return ResourcePreference(
            resource_type=resource_type,
            target_level=target_level,
            minimum_level=minimum_level,
            fallback_allowed=bool(self.fallback_allowed),
        )

    def candidate_levels(self) -> tuple[int, ...]:
        preference = self.normalized()
        if not preference.fallback_allowed:
            return (preference.target_level,)
        assert preference.minimum_level is not None
        return tuple(range(preference.target_level, preference.minimum_level - 1, -1))


@dataclass(frozen=True)
class ResourceGatheringRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    resource_preferences: tuple[ResourcePreference, ...] = ()
    resource_type: ResourceType | str = ResourceType.GOLD
    target_level: int = MAX_RESOURCE_LEVEL
    minimum_level: int | None = None
    fallback_allowed: bool = False
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"

    def preferences(self) -> tuple[ResourcePreference, ...]:
        if self.resource_preferences:
            return tuple(item.normalized() for item in self.resource_preferences)
        return (
            ResourcePreference(
                self.resource_type,
                self.target_level,
                self.minimum_level,
                self.fallback_allowed,
            ).normalized(),
        )


@dataclass(frozen=True)
class ResourceGatheringConfig:
    workflow_timeout_seconds: float = 180.0
    step_timeout_seconds: float = 20.0
    navigation_retry_limit: int = 1
    search_retry_limit: int = 1
    dispatch_retry_limit: int = 1
    retry_delay_seconds: float = 0.25


@dataclass(frozen=True)
class ResourceSelection:
    resource_type: ResourceType
    level: int
    fallback_from_level: int | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "resource_type": self.resource_type.value,
            "level": self.level,
            "fallback_from_level": self.fallback_from_level,
        }


@dataclass(frozen=True)
class ResourceGatheringActionResult:
    success: bool
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ResourceNodeSearchResult:
    found: bool
    resource_type: ResourceType | str
    level: int
    confidence: float = 0.0
    x: int | None = None
    y: int | None = None
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def selection(self, fallback_from_level: int | None = None) -> ResourceSelection:
        return ResourceSelection(
            ResourceSearchWorkflow._validate_resource_type(self.resource_type),
            ResourceSearchWorkflow._validate_target_level(self.level),
            fallback_from_level,
        )


@dataclass(frozen=True)
class MarchAvailability:
    available: bool
    march_slot: int | None = None
    available_count: int = 0
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MarchDispatchResult:
    success: bool
    march_slot: int | None = None
    dispatch_id: str = ""
    expected_return_time: str | None = None
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)


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


class ResourceAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: ResourceGatheringRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class ResourceCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: ResourceGatheringRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class ResourceGatheringDriver(Protocol):
    def navigate_to_resource_search(
        self,
        request: ResourceGatheringRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...

    def select_resource(
        self,
        request: ResourceGatheringRequest,
        selection: ResourceSelection,
    ) -> ResourceGatheringActionResult:
        ...

    def search_resource(
        self,
        request: ResourceGatheringRequest,
        selection: ResourceSelection,
    ) -> ResourceNodeSearchResult:
        ...

    def validate_march_availability(
        self,
        request: ResourceGatheringRequest,
        selection: ResourceSelection,
    ) -> MarchAvailability:
        ...

    def dispatch_gather_march(
        self,
        request: ResourceGatheringRequest,
        selection: ResourceSelection,
        availability: MarchAvailability,
    ) -> MarchDispatchResult:
        ...

    def verify_dispatch(
        self,
        request: ResourceGatheringRequest,
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
class _GatheringState:
    request: ResourceGatheringRequest
    character: Character | None = None
    preferences: tuple[ResourcePreference, ...] = ()
    pending_selection: ResourceSelection | None = None
    selected_resource: ResourceSelection | None = None
    node: ResourceNodeSearchResult | None = None
    march_availability: MarchAvailability | None = None
    dispatch: MarchDispatchResult | None = None
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
        return _resource_step_result(
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


class ResourceGatheringWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: ResourceGatheringDriver,
        account_precondition: ResourceAccountPrecondition | None = None,
        character_precondition: ResourceCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        marches: MarchRepository | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: ResourceGatheringConfig | None = None,
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
        self.config = config or ResourceGatheringConfig()
        self._states: dict[str, _GatheringState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return RESOURCE_GATHERING_STATES

    def execute(
        self,
        request: ResourceGatheringRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _GatheringState(request=request)
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
                budget=StepBudget(max_steps=len(RESOURCE_GATHERING_STATES) + 8),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"resource-gathering:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"resource_gathering_run_id": token},
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
        for state in RESOURCE_GATHERING_STATES:
            registry.register(f"resource_gathering.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.navigation_retry_limit,
            "ensure_character": self.config.navigation_retry_limit,
            "ensure_game_running": self.config.navigation_retry_limit,
            "navigate_to_resource_search": self.config.navigation_retry_limit,
            "select_resource": self.config.navigation_retry_limit,
            "search_resource": self.config.search_retry_limit,
            "dispatch_march": self.config.dispatch_retry_limit,
            "verify_dispatch": self.config.dispatch_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=RESOURCE_GATHERING_WORKFLOW_KEY,
            name="Gather Resources",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"resource_gathering.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in RESOURCE_GATHERING_STATES
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
                return _resource_step_result(step.step_key, WorkflowOutcome.SKIPPED)
            if state.failed and state_name not in {"recover", "complete"}:
                return _resource_step_result(
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
        state: _GatheringState,
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
            state.preferences = request.preferences()
        except ValueError as exc:
            return state.fail(step.step_key, str(exc))
        return _resource_step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"resource_preferences": [_preference_json(item) for item in state.preferences]},
        )

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _GatheringState,
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
        return _resource_step_result(
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
        state: _GatheringState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.account_precondition is None:
            return _resource_step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"account_precondition": "not_configured"})
        return self._action_to_step(
            step,
            state,
            self.account_precondition.ensure_account(state.request, _require_character(state)),
        )

    def _ensure_character(
        self,
        step: WorkflowStepSpec,
        state: _GatheringState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.character_precondition is None:
            return _resource_step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"character_precondition": "not_configured"})
        return self._action_to_step(
            step,
            state,
            self.character_precondition.ensure_character(state.request, _require_character(state)),
        )

    def _ensure_game_running(
        self,
        step: WorkflowStepSpec,
        state: _GatheringState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.recovery_watchdog is None:
            return _resource_step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"watchdog": "not_configured"})
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
        return _resource_step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"watchdog_healthy": True})

    def _navigate_to_resource_search(
        self,
        step: WorkflowStepSpec,
        state: _GatheringState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.navigate_to_resource_search(state.request, _require_character(state)),
        )

    def _select_resource(
        self,
        step: WorkflowStepSpec,
        state: _GatheringState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        selection = self._next_pending_selection(state)
        if selection is None:
            return state.fail(step.step_key, "No configured resource preference is available.")
        return self._action_to_step(
            step,
            state,
            self.driver.select_resource(state.request, selection),
        )

    def _search_resource(
        self,
        step: WorkflowStepSpec,
        state: _GatheringState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        for preference in state.preferences:
            target_level = preference.target_level
            for level in preference.candidate_levels():
                selection = ResourceSelection(
                    ResourceSearchWorkflow._validate_resource_type(preference.resource_type),
                    level,
                    target_level if level != target_level else None,
                )
                if state.pending_selection != selection:
                    select_result = self.driver.select_resource(state.request, selection)
                    if not select_result.success:
                        return self._action_to_step(step, state, select_result)
                    state.pending_selection = selection
                result = self.driver.search_resource(state.request, selection)
                if result.screenshot_path:
                    state.screenshot_path = result.screenshot_path
                if result.found:
                    state.selected_resource = result.selection(selection.fallback_from_level)
                    state.node = result
                    return _resource_step_result(
                        step.step_key,
                        WorkflowOutcome.SUCCESS,
                        data={
                            "selected_resource": state.selected_resource.to_json(),
                            "node": _node_json(result),
                        },
                        screenshot_path=result.screenshot_path,
                    )
        return state.fail(
            step.step_key,
            "No configured resource node was found.",
            screenshot_path=state.screenshot_path,
            data={"searched_preferences": [_preference_json(item) for item in state.preferences]},
        )

    def _validate_march(
        self,
        step: WorkflowStepSpec,
        state: _GatheringState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        selection = _require_selection(state)
        availability = self.driver.validate_march_availability(state.request, selection)
        state.march_availability = availability
        if availability.screenshot_path:
            state.screenshot_path = availability.screenshot_path
        if not availability.available:
            return state.fail(
                step.step_key,
                availability.message or "No march is available for gathering.",
                screenshot_path=availability.screenshot_path,
                data={
                    "available_count": availability.available_count,
                    "march_slot": availability.march_slot,
                    **availability.data,
                },
            )
        return _resource_step_result(
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
        state: _GatheringState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        selection = _require_selection(state)
        availability = _require_availability(state)
        dispatch = self.driver.dispatch_gather_march(state.request, selection, availability)
        state.dispatch = dispatch
        if dispatch.screenshot_path:
            state.screenshot_path = dispatch.screenshot_path
        if not dispatch.success:
            return (
                _resource_step_result(
                    step.step_key,
                    WorkflowOutcome.RETRYABLE_FAILURE,
                    dispatch.message or "Gather march dispatch failed.",
                    data=dispatch.data,
                    screenshot_path=dispatch.screenshot_path,
                )
                if dispatch.retryable
                else state.fail(
                    step.step_key,
                    dispatch.message or "Gather march dispatch failed.",
                    screenshot_path=dispatch.screenshot_path,
                    data=dispatch.data,
                )
            )
        self._persist_march_dispatch(state, selection, dispatch)
        return _resource_step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"dispatch": _dispatch_json(dispatch)},
            screenshot_path=dispatch.screenshot_path,
        )

    def _verify_dispatch(
        self,
        step: WorkflowStepSpec,
        state: _GatheringState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        dispatch = _require_dispatch(state)
        return self._action_to_step(
            step,
            state,
            self.driver.verify_dispatch(state.request, dispatch),
        )

    def _complete(self, step: WorkflowStepSpec, state: _GatheringState) -> WorkflowStepResult:
        if state.failed:
            return _resource_step_result(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                data={"failure_state": state.failure_state},
            )
        return _resource_step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "selected_resource": _selection_json(state.selected_resource),
                "dispatch": _dispatch_json(state.dispatch),
            },
        )

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _GatheringState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _resource_step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if self.recovery_watchdog is None:
            state.recovery_outcome = {"attempted": False, "reason": "watchdog_not_configured"}
            return _resource_step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
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
        return _resource_step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _GatheringState) -> WorkflowStepResult:
        if not state.failed:
            return _resource_step_result(step.step_key, WorkflowOutcome.SKIPPED)
        self._open_incident(state)
        return _resource_step_result(
            step.step_key,
            WorkflowOutcome.FATAL_FAILURE,
            state.failure_reason,
            data={
                "failure_state": state.failure_state,
                "failure_reason": state.failure_reason,
                "selected_resource": _selection_json(state.selected_resource),
                "dispatch": _dispatch_json(state.dispatch),
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _next_pending_selection(self, state: _GatheringState) -> ResourceSelection | None:
        if state.pending_selection is not None:
            return state.pending_selection
        for preference in state.preferences:
            level = preference.target_level
            state.pending_selection = ResourceSelection(
                ResourceSearchWorkflow._validate_resource_type(preference.resource_type),
                level,
            )
            return state.pending_selection
        return None

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _GatheringState,
        action: ResourceGatheringActionResult,
    ) -> WorkflowStepResult:
        if action.screenshot_path:
            state.screenshot_path = action.screenshot_path
        if action.success:
            return _resource_step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        if action.retryable:
            return _resource_step_result(
                step.step_key,
                WorkflowOutcome.RETRYABLE_FAILURE,
                action.message or "Resource gathering action failed.",
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.fail(
            step.step_key,
            action.message or "Resource gathering action failed.",
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _persist_march_dispatch(
        self,
        state: _GatheringState,
        selection: ResourceSelection,
        dispatch: MarchDispatchResult,
    ) -> None:
        if self.marches is None or dispatch.march_slot is None:
            return
        character = _require_character(state)
        marches = self.marches.list_for_character(int(character.id or 0))
        existing = next((item for item in marches if item.march_slot == dispatch.march_slot), None)
        march = existing or March(character_id=character.id, march_slot=dispatch.march_slot)
        march.status = "gathering"
        march.expected_return_time = dispatch.expected_return_time
        march.next_action_time = dispatch.expected_return_time
        self.marches.save(march)

    def _state_from_context(self, context: WorkflowExecutionContext) -> _GatheringState:
        token = str(context.metadata.get("resource_gathering_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("Resource gathering runtime state is missing.") from exc

    def _open_incident(self, state: _GatheringState) -> None:
        if self.incidents is None or not state.failed:
            return
        self.incidents.save(
            Incident(
                incident_key=f"resource-gathering:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="Resource gathering failed",
                details=state.failure_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _GatheringState,
    ) -> None:
        if not result.outcome.is_failure or state.failed or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.failure_state = last_step.step_key if last_step is not None else ""
        state.failure_reason = result.message or "Resource gathering workflow failed."
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""
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
        state: _GatheringState,
    ) -> None:
        result.result = {
            **dict(result.result),
            "selected_resource": _selection_json(state.selected_resource),
            "march_dispatch": _dispatch_json(state.dispatch),
            "failure_state": state.failure_state,
            "failure_reason": state.failure_reason,
            "recovery_outcome": state.recovery_outcome,
        }

    def _update_persisted_run(
        self,
        result: WorkflowExecutionResult,
        state: _GatheringState,
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


def _resource_step_result(
    step_key: str,
    outcome: WorkflowOutcome,
    message: str = "",
    *,
    data: dict[str, object] | None = None,
    screenshot_path: str = "",
) -> WorkflowStepResult:
    return WorkflowStepResult(
        step_key=step_key,
        action_type=f"resource_gathering.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _GatheringState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_selection(state: _GatheringState) -> ResourceSelection:
    if state.selected_resource is None:
        raise RuntimeError("Resource node has not been selected.")
    return state.selected_resource


def _require_availability(state: _GatheringState) -> MarchAvailability:
    if state.march_availability is None:
        raise RuntimeError("March availability has not been validated.")
    return state.march_availability


def _require_dispatch(state: _GatheringState) -> MarchDispatchResult:
    if state.dispatch is None:
        raise RuntimeError("Gather march has not been dispatched.")
    return state.dispatch


def _preference_json(preference: ResourcePreference) -> dict[str, object]:
    normalized = preference.normalized()
    return {
        "resource_type": normalized.resource_type.value,
        "target_level": normalized.target_level,
        "minimum_level": normalized.minimum_level,
        "fallback_allowed": normalized.fallback_allowed,
    }


def _selection_json(selection: ResourceSelection | None) -> dict[str, object]:
    return selection.to_json() if selection is not None else {}


def _node_json(node: ResourceNodeSearchResult) -> dict[str, object]:
    return {
        "resource_type": ResourceSearchWorkflow._validate_resource_type(node.resource_type).value,
        "level": node.level,
        "confidence": node.confidence,
        "x": node.x,
        "y": node.y,
        **node.data,
    }


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
