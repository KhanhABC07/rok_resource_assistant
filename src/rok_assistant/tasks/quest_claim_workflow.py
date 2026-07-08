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


QUEST_CLAIM_WORKFLOW_KEY = "quest-claim"
QUEST_CLAIM_TEMPLATE_KEYS = (
    "quest.button",
    "quest.scene",
    "quest.tab.main",
    "quest.tab.side",
    "quest.action.claim",
    "quest.action.go",
    "quest.action.complete",
    "quest.daily.objectives",
    "quest.reward.close",
)
QUEST_CLAIM_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "open_quest_ui",
    "scan_quest_tabs",
    "process_quest_page",
    "claim_quest",
    "process_daily_objectives",
    "claim_daily_milestone",
    "close_reward_overlay",
    "verify_claim_state",
    "complete",
    "skipped",
    "recover",
    "failed",
    "cancelled",
)


class QuestCategory(StrEnum):
    MAIN = "MAIN"
    SIDE = "SIDE"


class QuestAction(StrEnum):
    CLAIM = "CLAIM"
    GO = "GO"
    COMPLETE = "COMPLETE"
    SPEND = "SPEND"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class QuestPageScanStatus(StrEnum):
    READY = "READY"
    NONE_CLAIMABLE = "NONE_CLAIMABLE"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


def _category(value: QuestCategory | str) -> QuestCategory:
    if isinstance(value, QuestCategory):
        return value
    try:
        return QuestCategory(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in QuestCategory)
        raise ValueError(f"Invalid quest category: {value!r}. Expected one of: {valid}.") from exc


def _action(value: QuestAction | str) -> QuestAction:
    if isinstance(value, QuestAction):
        return value
    try:
        return QuestAction(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in QuestAction)
        raise ValueError(f"Invalid quest action: {value!r}. Expected one of: {valid}.") from exc


def _scan_status(value: QuestPageScanStatus | str) -> QuestPageScanStatus:
    if isinstance(value, QuestPageScanStatus):
        return value
    try:
        return QuestPageScanStatus(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in QuestPageScanStatus)
        raise ValueError(f"Invalid quest page scan status: {value!r}. Expected one of: {valid}.") from exc


def _require_positive_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _require_confidence(value: float, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    return numeric


@dataclass(frozen=True)
class QuestClaimPolicy:
    enabled_categories: tuple[QuestCategory | str, ...] = (QuestCategory.MAIN, QuestCategory.SIDE)
    minimum_claim_confidence: float = 0.85
    max_claim_iterations: int = 12

    def normalized(self) -> QuestClaimPolicy:
        enabled = tuple(dict.fromkeys(_category(item) for item in self.enabled_categories))
        if not enabled:
            raise ValueError("At least one quest category must be enabled.")
        return QuestClaimPolicy(
            enabled_categories=enabled,
            minimum_claim_confidence=_require_confidence(
                self.minimum_claim_confidence,
                "minimum_claim_confidence",
            ),
            max_claim_iterations=_require_positive_int(self.max_claim_iterations, "max_claim_iterations"),
        )

    def category_enabled(self, category: QuestCategory | str) -> bool:
        return _category(category) in self.normalized().enabled_categories

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "enabled_categories": [item.value for item in normalized.enabled_categories],
            "minimum_claim_confidence": normalized.minimum_claim_confidence,
            "max_claim_iterations": normalized.max_claim_iterations,
        }


@dataclass(frozen=True)
class QuestClaimRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: QuestClaimPolicy = field(default_factory=QuestClaimPolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class QuestClaimConfig:
    workflow_timeout_seconds: float = 120.0
    step_timeout_seconds: float = 15.0
    precondition_retry_limit: int = 1
    navigation_retry_limit: int = 1
    scan_retry_limit: int = 1
    action_retry_limit: int = 0
    retry_delay_seconds: float = 0.25
    max_quest_pages_per_tab: int = 4
    max_daily_pages: int = 2

    def normalized_max_quest_pages_per_tab(self) -> int:
        return _require_positive_int(self.max_quest_pages_per_tab, "max_quest_pages_per_tab")

    def normalized_max_daily_pages(self) -> int:
        return _require_positive_int(self.max_daily_pages, "max_daily_pages")


@dataclass(frozen=True)
class QuestEntryObservation:
    category: QuestCategory | str
    action: QuestAction | str
    quest_id: str = ""
    title: str = ""
    completed: bool = False
    claimed: bool = False
    confidence: float = 1.0
    target: tuple[int, int] | None = None
    page_number: int = 1
    scene_verified: bool = True
    spend_detected: bool = False
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_category(self) -> QuestCategory:
        return _category(self.category)

    def normalized_action(self) -> QuestAction:
        return _action(self.action)

    def target_json(self) -> dict[str, int] | None:
        if self.target is None:
            return None
        return {"x": int(self.target[0]), "y": int(self.target[1])}

    def to_json(self) -> dict[str, object]:
        return {
            "category": self.normalized_category().value,
            "action": self.normalized_action().value,
            "quest_id": self.quest_id,
            "title": self.title,
            "completed": self.completed,
            "claimed": self.claimed,
            "confidence": self.confidence,
            "target": self.target_json(),
            "page_number": self.page_number,
            "scene_verified": self.scene_verified,
            "spend_detected": self.spend_detected,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class QuestPageScan:
    status: QuestPageScanStatus | str
    category: QuestCategory | str
    page_number: int = 1
    observations: tuple[QuestEntryObservation, ...] = ()
    has_next_page: bool = False
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_status(self) -> QuestPageScanStatus:
        return _scan_status(self.status)

    def normalized_category(self) -> QuestCategory:
        return _category(self.category)

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.normalized_status().value,
            "category": self.normalized_category().value,
            "page_number": self.page_number,
            "observations": [item.to_json() for item in self.observations],
            "has_next_page": self.has_next_page,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class DailyObjectiveMilestoneObservation:
    milestone_id: str
    action: QuestAction | str
    points_required: int = 0
    completed: bool = False
    claimed: bool = False
    confidence: float = 1.0
    target: tuple[int, int] | None = None
    page_number: int = 1
    scene_verified: bool = True
    spend_detected: bool = False
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_action(self) -> QuestAction:
        return _action(self.action)

    def target_json(self) -> dict[str, int] | None:
        if self.target is None:
            return None
        return {"x": int(self.target[0]), "y": int(self.target[1])}

    def to_json(self) -> dict[str, object]:
        return {
            "milestone_id": self.milestone_id,
            "action": self.normalized_action().value,
            "points_required": self.points_required,
            "completed": self.completed,
            "claimed": self.claimed,
            "confidence": self.confidence,
            "target": self.target_json(),
            "page_number": self.page_number,
            "scene_verified": self.scene_verified,
            "spend_detected": self.spend_detected,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class DailyObjectiveScan:
    status: QuestPageScanStatus | str
    page_number: int = 1
    milestones: tuple[DailyObjectiveMilestoneObservation, ...] = ()
    has_next_page: bool = False
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_status(self) -> QuestPageScanStatus:
        return _scan_status(self.status)

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.normalized_status().value,
            "page_number": self.page_number,
            "milestones": [item.to_json() for item in self.milestones],
            "has_next_page": self.has_next_page,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class QuestClaimActionResult:
    success: bool
    changed: bool = False
    reward_overlay_present: bool = True
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "changed": self.changed,
            "reward_overlay_present": self.reward_overlay_present,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class QuestRewardCloseResult:
    success: bool
    closed: bool = False
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "closed": self.closed,
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


class QuestClaimAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: QuestClaimRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class QuestClaimCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: QuestClaimRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class QuestClaimDriver(Protocol):
    def open_quest_ui(
        self,
        request: QuestClaimRequest,
        character: Character,
        policy: QuestClaimPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def select_quest_tab(
        self,
        request: QuestClaimRequest,
        character: Character,
        category: QuestCategory,
        policy: QuestClaimPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def scan_quest_page(
        self,
        request: QuestClaimRequest,
        character: Character,
        category: QuestCategory,
        page_number: int,
        policy: QuestClaimPolicy,
    ) -> QuestPageScan:
        ...

    def go_to_next_quest_page(
        self,
        request: QuestClaimRequest,
        character: Character,
        category: QuestCategory,
        page_number: int,
        policy: QuestClaimPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def claim_quest(
        self,
        request: QuestClaimRequest,
        character: Character,
        observation: QuestEntryObservation,
        policy: QuestClaimPolicy,
    ) -> QuestClaimActionResult:
        ...

    def close_reward_overlay_for_quest(
        self,
        request: QuestClaimRequest,
        character: Character,
        observation: QuestEntryObservation,
        claim_result: QuestClaimActionResult,
        policy: QuestClaimPolicy,
    ) -> QuestRewardCloseResult:
        ...

    def verify_quest_claim(
        self,
        request: QuestClaimRequest,
        character: Character,
        observation: QuestEntryObservation,
        claim_result: QuestClaimActionResult,
        policy: QuestClaimPolicy,
    ) -> QuestEntryObservation:
        ...

    def open_daily_objectives(
        self,
        request: QuestClaimRequest,
        character: Character,
        policy: QuestClaimPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def scan_daily_objectives(
        self,
        request: QuestClaimRequest,
        character: Character,
        page_number: int,
        policy: QuestClaimPolicy,
    ) -> DailyObjectiveScan:
        ...

    def go_to_next_daily_objectives_page(
        self,
        request: QuestClaimRequest,
        character: Character,
        page_number: int,
        policy: QuestClaimPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def claim_daily_milestone(
        self,
        request: QuestClaimRequest,
        character: Character,
        milestone: DailyObjectiveMilestoneObservation,
        policy: QuestClaimPolicy,
    ) -> QuestClaimActionResult:
        ...

    def close_reward_overlay_for_daily_milestone(
        self,
        request: QuestClaimRequest,
        character: Character,
        milestone: DailyObjectiveMilestoneObservation,
        claim_result: QuestClaimActionResult,
        policy: QuestClaimPolicy,
    ) -> QuestRewardCloseResult:
        ...

    def verify_daily_milestone_claim(
        self,
        request: QuestClaimRequest,
        character: Character,
        milestone: DailyObjectiveMilestoneObservation,
        claim_result: QuestClaimActionResult,
        policy: QuestClaimPolicy,
    ) -> DailyObjectiveMilestoneObservation:
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
class _QuestClaimState:
    request: QuestClaimRequest
    character: Character | None = None
    policy: QuestClaimPolicy | None = None
    scanned_quest_pages: list[dict[str, object]] = field(default_factory=list)
    scanned_daily_pages: list[dict[str, object]] = field(default_factory=list)
    ignored_actions: list[dict[str, object]] = field(default_factory=list)
    quest_claim_attempts: list[dict[str, object]] = field(default_factory=list)
    daily_milestone_attempts: list[dict[str, object]] = field(default_factory=list)
    reward_overlay_handling: list[dict[str, object]] = field(default_factory=list)
    verification_attempts: list[dict[str, object]] = field(default_factory=list)
    claimed_quests: list[dict[str, object]] = field(default_factory=list)
    claimed_daily_milestones: list[dict[str, object]] = field(default_factory=list)
    claim_iterations: int = 0
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


class QuestClaimWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: QuestClaimDriver,
        account_precondition: QuestClaimAccountPrecondition | None = None,
        character_precondition: QuestClaimCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: QuestClaimConfig | None = None,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.account_precondition = account_precondition
        self.character_precondition = character_precondition
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.incidents = incidents
        self.config = config or QuestClaimConfig()
        self._states: dict[str, _QuestClaimState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return QUEST_CLAIM_STATES

    def execute(
        self,
        request: QuestClaimRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _QuestClaimState(request=request)
        self._states[token] = state
        persistence = None
        if self.job_runs is not None and self.step_runs is not None and request.job_id is not None:
            persistence = WorkflowRunRepositoryRecorder(self.job_runs, self.step_runs)
        try:
            max_pages = self.config.normalized_max_quest_pages_per_tab()
            max_daily_pages = self.config.normalized_max_daily_pages()
            context = WorkflowExecutionContext(
                cancellation_token=cancellation_token or CancellationToken(),
                deadline=WorkflowDeadline.from_timeout(
                    self.config.workflow_timeout_seconds,
                    time.monotonic,
                ),
                budget=StepBudget(max_steps=len(QUEST_CLAIM_STATES) + max_pages * 2 + max_daily_pages + 16),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"quest-claim:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"quest_claim_run_id": token},
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
        for state in QUEST_CLAIM_STATES:
            registry.register(f"quest_claim.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.precondition_retry_limit,
            "ensure_character": self.config.precondition_retry_limit,
            "ensure_game_running": self.config.precondition_retry_limit,
            "open_quest_ui": self.config.navigation_retry_limit,
            "scan_quest_tabs": self.config.scan_retry_limit,
            "process_quest_page": self.config.action_retry_limit,
            "process_daily_objectives": self.config.action_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=QUEST_CLAIM_WORKFLOW_KEY,
            name="Claim Quests and Daily Objectives",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"quest_claim.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in QUEST_CLAIM_STATES
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
        state: _QuestClaimState,
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
            max_pages = self.config.normalized_max_quest_pages_per_tab()
            max_daily_pages = self.config.normalized_max_daily_pages()
        except ValueError as exc:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, str(exc))
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "policy": state.policy.to_json(),
                "max_quest_pages_per_tab": max_pages,
                "max_daily_pages": max_daily_pages,
                "template_keys": list(QUEST_CLAIM_TEMPLATE_KEYS),
            },
        )

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
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
        state: _QuestClaimState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.account_precondition is None:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"account_precondition": "not_configured"})
        return self._action_to_step(step, state, self.account_precondition.ensure_account(state.request, _require_character(state)))

    def _ensure_character(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
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
        state: _QuestClaimState,
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

    def _open_quest_ui(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.open_quest_ui(state.request, _require_character(state), _require_policy(state)),
        )

    def _scan_quest_tabs(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"categories": [item.value for item in _require_policy(state).enabled_categories]},
        )

    def _process_quest_page(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        character = _require_character(state)
        policy = _require_policy(state)
        for category in policy.enabled_categories:
            context.cancellation_token.throw_if_cancelled()
            selected = self.driver.select_quest_tab(state.request, character, category, policy)
            selected_step = self._action_to_step(step, state, selected)
            if selected_step.outcome != WorkflowOutcome.SUCCESS:
                return selected_step
            for page_number in range(1, self.config.normalized_max_quest_pages_per_tab() + 1):
                context.cancellation_token.throw_if_cancelled()
                scan = self.driver.scan_quest_page(state.request, character, category, page_number, policy)
                self._record_quest_scan(state, scan)
                scan_step = self._handle_quest_scan(step, state, scan)
                if scan_step is not None:
                    return scan_step
                for observation in scan.observations:
                    decision = _quest_claim_skip_reason(observation, policy)
                    if decision == "unsafe":
                        return state.stop(
                            step.step_key,
                            WorkflowOutcome.BLOCKED,
                            observation.message or "Unsafe or ambiguous quest action detected.",
                            screenshot_path=observation.screenshot_path,
                            data={"quest": observation.to_json()},
                        )
                    if decision:
                        state.ignored_actions.append({**observation.to_json(), "ignored_reason": decision})
                        continue
                    claim_step = self._claim_quest_observation(step, state, observation)
                    if claim_step.outcome != WorkflowOutcome.SUCCESS:
                        return claim_step
                if not scan.has_next_page:
                    break
                if page_number >= self.config.normalized_max_quest_pages_per_tab():
                    break
                next_page = self.driver.go_to_next_quest_page(state.request, character, category, page_number + 1, policy)
                next_step = self._action_to_step(step, state, next_page)
                if next_step.outcome != WorkflowOutcome.SUCCESS:
                    return next_step
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data=self._quest_payload(state),
            screenshot_path=state.screenshot_path,
        )

    def _claim_quest(
        self,
        step: WorkflowStepSpec,
        _state: _QuestClaimState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"handled_during": "process_quest_page"})

    def _process_daily_objectives(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        character = _require_character(state)
        policy = _require_policy(state)
        open_daily = self.driver.open_daily_objectives(state.request, character, policy)
        open_step = self._action_to_step(step, state, open_daily)
        if open_step.outcome != WorkflowOutcome.SUCCESS:
            return open_step
        for page_number in range(1, self.config.normalized_max_daily_pages() + 1):
            context.cancellation_token.throw_if_cancelled()
            scan = self.driver.scan_daily_objectives(state.request, character, page_number, policy)
            self._record_daily_scan(state, scan)
            scan_step = self._handle_daily_scan(step, state, scan)
            if scan_step is not None:
                return scan_step
            for milestone in scan.milestones:
                decision = _daily_claim_skip_reason(milestone, policy)
                if decision == "unsafe":
                    return state.stop(
                        step.step_key,
                        WorkflowOutcome.BLOCKED,
                        milestone.message or "Unsafe or ambiguous daily objective milestone action detected.",
                        screenshot_path=milestone.screenshot_path,
                        data={"milestone": milestone.to_json()},
                    )
                if decision:
                    state.ignored_actions.append({**milestone.to_json(), "ignored_reason": decision, "category": "DAILY_OBJECTIVE"})
                    continue
                claim_step = self._claim_daily_milestone_observation(step, state, milestone)
                if claim_step.outcome != WorkflowOutcome.SUCCESS:
                    return claim_step
            if not scan.has_next_page:
                break
            if page_number >= self.config.normalized_max_daily_pages():
                break
            next_page = self.driver.go_to_next_daily_objectives_page(state.request, character, page_number + 1, policy)
            next_step = self._action_to_step(step, state, next_page)
            if next_step.outcome != WorkflowOutcome.SUCCESS:
                return next_step
        if not state.claimed_quests and not state.claimed_daily_milestones:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                "No completed quest or daily objective milestone claim is available.",
                screenshot_path=state.screenshot_path,
                data=self._quest_payload(state),
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data=self._quest_payload(state),
            screenshot_path=state.screenshot_path,
        )

    def _claim_daily_milestone(
        self,
        step: WorkflowStepSpec,
        _state: _QuestClaimState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"handled_during": "process_daily_objectives"})

    def _close_reward_overlay(
        self,
        step: WorkflowStepSpec,
        _state: _QuestClaimState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"handled_during": "claim_steps"})

    def _verify_claim_state(
        self,
        step: WorkflowStepSpec,
        _state: _QuestClaimState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"handled_during": "claim_steps"})

    def _complete(self, step: WorkflowStepSpec, state: _QuestClaimState) -> WorkflowStepResult:
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
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=self._quest_payload(state))

    def _skipped(self, step: WorkflowStepSpec, state: _QuestClaimState) -> WorkflowStepResult:
        if state.terminal_outcome == WorkflowOutcome.SKIPPED:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data={"skipped_reason": state.terminal_reason, **self._quest_payload(state)},
                screenshot_path=state.screenshot_path,
            )
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED)

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if _is_manual_stop(state):
            state.recovery_outcome = {"attempted": False, "reason": "manual_or_unsafe_action"}
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
        state.recovery_outcome = self._monitor_recovery(state, _job_run_id(context))
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _QuestClaimState) -> WorkflowStepResult:
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
                **self._quest_payload(state),
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _claim_quest_observation(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
        observation: QuestEntryObservation,
    ) -> WorkflowStepResult:
        limit_step = self._reserve_claim_iteration(step, state)
        if limit_step is not None:
            return limit_step
        claim = self.driver.claim_quest(state.request, _require_character(state), observation, _require_policy(state))
        if claim.screenshot_path:
            state.screenshot_path = claim.screenshot_path
        attempt = {"quest": observation.to_json(), **claim.to_json()}
        state.quest_claim_attempts.append(attempt)
        if not claim.success:
            return self._claim_action_failure(step, state, claim, "Quest claim action failed.")
        close_step = self._close_quest_reward_overlay(step, state, observation, claim)
        if close_step.outcome != WorkflowOutcome.SUCCESS:
            return close_step
        verification = self.driver.verify_quest_claim(
            state.request,
            _require_character(state),
            observation,
            claim,
            _require_policy(state),
        )
        if verification.screenshot_path:
            state.screenshot_path = verification.screenshot_path
        state.verification_attempts.append({"before": observation.to_json(), "after": verification.to_json()})
        if not _quest_postcondition_verified(observation, verification):
            failure = replace(
                verification,
                action=QuestAction.UNKNOWN,
                message=("Quest claim postcondition was not verified. " f"{verification.message}").strip(),
            )
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                failure.message,
                screenshot_path=failure.screenshot_path,
                data={"verification": failure.to_json()},
            )
        state.claimed_quests.append(
            {
                "category": observation.normalized_category().value,
                "quest_id": observation.quest_id,
                "title": observation.title,
                "page_number": observation.page_number,
                "reward_claimed": True,
            }
        )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=verification.to_json())

    def _claim_daily_milestone_observation(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
        milestone: DailyObjectiveMilestoneObservation,
    ) -> WorkflowStepResult:
        limit_step = self._reserve_claim_iteration(step, state)
        if limit_step is not None:
            return limit_step
        claim = self.driver.claim_daily_milestone(
            state.request,
            _require_character(state),
            milestone,
            _require_policy(state),
        )
        if claim.screenshot_path:
            state.screenshot_path = claim.screenshot_path
        attempt = {"milestone": milestone.to_json(), **claim.to_json()}
        state.daily_milestone_attempts.append(attempt)
        if not claim.success:
            return self._claim_action_failure(step, state, claim, "Daily objective milestone claim action failed.")
        close_step = self._close_daily_reward_overlay(step, state, milestone, claim)
        if close_step.outcome != WorkflowOutcome.SUCCESS:
            return close_step
        verification = self.driver.verify_daily_milestone_claim(
            state.request,
            _require_character(state),
            milestone,
            claim,
            _require_policy(state),
        )
        if verification.screenshot_path:
            state.screenshot_path = verification.screenshot_path
        state.verification_attempts.append({"before": milestone.to_json(), "after": verification.to_json(), "category": "DAILY_OBJECTIVE"})
        if not _daily_postcondition_verified(milestone, verification):
            failure = replace(
                verification,
                action=QuestAction.UNKNOWN,
                message=("Daily objective milestone claim postcondition was not verified. " f"{verification.message}").strip(),
            )
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                failure.message,
                screenshot_path=failure.screenshot_path,
                data={"verification": failure.to_json()},
            )
        state.claimed_daily_milestones.append(
            {
                "milestone_id": milestone.milestone_id,
                "points_required": milestone.points_required,
                "page_number": milestone.page_number,
                "reward_claimed": True,
            }
        )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=verification.to_json())

    def _close_quest_reward_overlay(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
        observation: QuestEntryObservation,
        claim: QuestClaimActionResult,
    ) -> WorkflowStepResult:
        if not claim.reward_overlay_present:
            state.reward_overlay_handling.append(
                {"quest": observation.to_json(), "skipped": True, "reason": "reward_overlay_not_present"}
            )
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS)
        close = self.driver.close_reward_overlay_for_quest(
            state.request,
            _require_character(state),
            observation,
            claim,
            _require_policy(state),
        )
        if close.screenshot_path:
            state.screenshot_path = close.screenshot_path
        state.reward_overlay_handling.append({"quest": observation.to_json(), **close.to_json()})
        if not close.success or not close.closed:
            return self._reward_close_failure(step, state, close, "Quest reward overlay could not be closed safely.")
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=close.to_json())

    def _close_daily_reward_overlay(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
        milestone: DailyObjectiveMilestoneObservation,
        claim: QuestClaimActionResult,
    ) -> WorkflowStepResult:
        if not claim.reward_overlay_present:
            state.reward_overlay_handling.append(
                {"milestone": milestone.to_json(), "skipped": True, "reason": "reward_overlay_not_present"}
            )
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS)
        close = self.driver.close_reward_overlay_for_daily_milestone(
            state.request,
            _require_character(state),
            milestone,
            claim,
            _require_policy(state),
        )
        if close.screenshot_path:
            state.screenshot_path = close.screenshot_path
        state.reward_overlay_handling.append({"milestone": milestone.to_json(), **close.to_json()})
        if not close.success or not close.closed:
            return self._reward_close_failure(step, state, close, "Daily objective reward overlay could not be closed safely.")
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=close.to_json())

    def _reserve_claim_iteration(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
    ) -> WorkflowStepResult | None:
        policy = _require_policy(state)
        if state.claim_iterations >= policy.max_claim_iterations:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                "Quest claim iteration budget was exhausted before a stable no-claim state.",
                screenshot_path=state.screenshot_path,
                data={"claim_iterations": state.claim_iterations, "policy": policy.to_json()},
            )
        state.claim_iterations += 1
        return None

    def _handle_quest_scan(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
        scan: QuestPageScan,
    ) -> WorkflowStepResult | None:
        if scan.screenshot_path:
            state.screenshot_path = scan.screenshot_path
        status = scan.normalized_status()
        if status == QuestPageScanStatus.VERIFICATION_REQUIRED:
            return state.stop(
                step.step_key,
                WorkflowOutcome.FATAL_FAILURE,
                scan.message or "Quest UI requires manual verification.",
                screenshot_path=scan.screenshot_path,
                data={"scan": scan.to_json()},
            )
        if status == QuestPageScanStatus.UNKNOWN:
            outcome = WorkflowOutcome.RETRYABLE_FAILURE if scan.retryable else WorkflowOutcome.BLOCKED
            if scan.retryable:
                return _step_result(
                    step.step_key,
                    outcome,
                    scan.message or "Quest page state could not be determined.",
                    data={"scan": scan.to_json()},
                    screenshot_path=scan.screenshot_path,
                )
            return state.stop(
                step.step_key,
                outcome,
                scan.message or "Quest page state could not be determined.",
                screenshot_path=scan.screenshot_path,
                data={"scan": scan.to_json()},
            )
        return None

    def _handle_daily_scan(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
        scan: DailyObjectiveScan,
    ) -> WorkflowStepResult | None:
        if scan.screenshot_path:
            state.screenshot_path = scan.screenshot_path
        status = scan.normalized_status()
        if status == QuestPageScanStatus.VERIFICATION_REQUIRED:
            return state.stop(
                step.step_key,
                WorkflowOutcome.FATAL_FAILURE,
                scan.message or "Daily objective UI requires manual verification.",
                screenshot_path=scan.screenshot_path,
                data={"scan": scan.to_json()},
            )
        if status == QuestPageScanStatus.UNKNOWN:
            outcome = WorkflowOutcome.RETRYABLE_FAILURE if scan.retryable else WorkflowOutcome.BLOCKED
            if scan.retryable:
                return _step_result(
                    step.step_key,
                    outcome,
                    scan.message or "Daily objective state could not be determined.",
                    data={"scan": scan.to_json()},
                    screenshot_path=scan.screenshot_path,
                )
            return state.stop(
                step.step_key,
                outcome,
                scan.message or "Daily objective state could not be determined.",
                screenshot_path=scan.screenshot_path,
                data={"scan": scan.to_json()},
            )
        return None

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
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
                action.message or "Quest claim workflow action failed.",
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            action.message or "Quest claim workflow action failed.",
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _claim_action_failure(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
        result: QuestClaimActionResult,
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

    def _reward_close_failure(
        self,
        step: WorkflowStepSpec,
        state: _QuestClaimState,
        result: QuestRewardCloseResult,
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

    def _record_quest_scan(self, state: _QuestClaimState, scan: QuestPageScan) -> None:
        state.scanned_quest_pages.append(
            {
                "category": scan.normalized_category().value,
                "page_number": scan.page_number,
                "status": scan.normalized_status().value,
                "observation_count": len(scan.observations),
                "has_next_page": scan.has_next_page,
                "screenshot_path": scan.screenshot_path,
                **scan.data,
            }
        )

    def _record_daily_scan(self, state: _QuestClaimState, scan: DailyObjectiveScan) -> None:
        state.scanned_daily_pages.append(
            {
                "page_number": scan.page_number,
                "status": scan.normalized_status().value,
                "milestone_count": len(scan.milestones),
                "has_next_page": scan.has_next_page,
                "screenshot_path": scan.screenshot_path,
                **scan.data,
            }
        )

    def _quest_payload(self, state: _QuestClaimState) -> dict[str, object]:
        claimed_by_category: dict[str, list[dict[str, object]]] = {item.value: [] for item in QuestCategory}
        for claim in state.claimed_quests:
            claimed_by_category[str(claim["category"])].append(claim)
        return {
            "scanned_quest_pages": state.scanned_quest_pages,
            "scanned_daily_pages": state.scanned_daily_pages,
            "ignored_actions": state.ignored_actions,
            "quest_claim_attempts": state.quest_claim_attempts,
            "daily_milestone_attempts": state.daily_milestone_attempts,
            "reward_overlay_handling": state.reward_overlay_handling,
            "verification_attempts": state.verification_attempts,
            "claimed_quests": state.claimed_quests,
            "claimed_quests_by_category": claimed_by_category,
            "claimed_daily_milestones": state.claimed_daily_milestones,
            "claimed_daily_milestone_ids": [item["milestone_id"] for item in state.claimed_daily_milestones],
            "claim_iterations": state.claim_iterations,
            "verification_result": state.verification_attempts[-1] if state.verification_attempts else {},
            "failure_evidence": {
                "screenshot_path": state.screenshot_path,
                "terminal_state": state.terminal_state,
                "terminal_reason": state.terminal_reason,
            },
        }

    def _state_from_context(self, context: WorkflowExecutionContext) -> _QuestClaimState:
        token = str(context.metadata.get("quest_claim_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("Quest claim runtime state is missing.") from exc

    def _open_incident(self, state: _QuestClaimState) -> None:
        if self.incidents is None or not state.failed or state.incident_opened:
            return
        self.incidents.save(
            Incident(
                incident_key=f"quest-claim:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="Quest claim workflow blocked",
                details=state.terminal_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )
        state.incident_opened = True

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _QuestClaimState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "Quest claim workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(
        self,
        result: WorkflowExecutionResult,
        state: _QuestClaimState,
    ) -> None:
        if state.failed and not state.recovery_outcome:
            if _is_manual_stop(state):
                state.recovery_outcome = {"attempted": False, "reason": "manual_or_unsafe_action"}
            else:
                state.recovery_outcome = self._monitor_recovery(state, result.job_run_id)
        if state.failed:
            self._open_incident(state)

    def _monitor_recovery(
        self,
        state: _QuestClaimState,
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
        state: _QuestClaimState,
    ) -> None:
        if state.terminal_outcome in {WorkflowOutcome.SKIPPED, WorkflowOutcome.BLOCKED}:
            result.outcome = state.terminal_outcome
            result.message = state.terminal_reason
        result.result = {
            **dict(result.result),
            "policy": state.policy.to_json() if state.policy is not None else {},
            **self._quest_payload(state),
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
        state: _QuestClaimState,
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


def _quest_claim_skip_reason(observation: QuestEntryObservation, policy: QuestClaimPolicy) -> str:
    action = observation.normalized_action()
    if observation.spend_detected or action == QuestAction.SPEND:
        return "unsafe"
    if action == QuestAction.UNKNOWN:
        return "unsafe"
    if action in {QuestAction.GO, QuestAction.COMPLETE}:
        return f"{action.value.lower()}_action_not_allowed"
    if action != QuestAction.CLAIM:
        return "not_claim_action"
    if not policy.category_enabled(observation.normalized_category()):
        return "category_not_enabled"
    if observation.claimed:
        return "already_claimed"
    if not observation.completed:
        return "claim_action_not_verified_completed"
    if not observation.scene_verified or observation.target is None:
        return "unsafe"
    if observation.confidence < policy.minimum_claim_confidence:
        return "unsafe"
    return ""


def _daily_claim_skip_reason(
    milestone: DailyObjectiveMilestoneObservation,
    policy: QuestClaimPolicy,
) -> str:
    action = milestone.normalized_action()
    if milestone.spend_detected or action == QuestAction.SPEND:
        return "unsafe"
    if action == QuestAction.UNKNOWN:
        return "unsafe"
    if action in {QuestAction.GO, QuestAction.COMPLETE}:
        return f"{action.value.lower()}_action_not_allowed"
    if action != QuestAction.CLAIM:
        return "not_claim_action"
    if milestone.claimed:
        return "already_claimed"
    if not milestone.completed:
        return "claim_action_not_verified_completed"
    if not milestone.scene_verified or milestone.target is None:
        return "unsafe"
    if milestone.confidence < policy.minimum_claim_confidence:
        return "unsafe"
    return ""


def _quest_postcondition_verified(
    before: QuestEntryObservation,
    verification: QuestEntryObservation,
) -> bool:
    if before.normalized_category() != verification.normalized_category():
        return False
    if before.quest_id and verification.quest_id and before.quest_id != verification.quest_id:
        return False
    if not verification.scene_verified:
        return False
    return verification.claimed or verification.normalized_action() != QuestAction.CLAIM


def _daily_postcondition_verified(
    before: DailyObjectiveMilestoneObservation,
    verification: DailyObjectiveMilestoneObservation,
) -> bool:
    if before.milestone_id and verification.milestone_id and before.milestone_id != verification.milestone_id:
        return False
    if not verification.scene_verified:
        return False
    return verification.claimed or verification.normalized_action() != QuestAction.CLAIM


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
        action_type=f"quest_claim.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _QuestClaimState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _QuestClaimState) -> QuestClaimPolicy:
    if state.policy is None:
        raise RuntimeError("Quest claim policy has not been validated.")
    return state.policy


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_manual_stop(state: _QuestClaimState) -> bool:
    text = state.terminal_reason.lower()
    return "verification" in text or "unsafe" in text or "ambiguous" in text or "unknown" in text
