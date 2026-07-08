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


ALLIANCE_GIFT_COLLECTION_WORKFLOW_KEY = "alliance-gift-collection"
ALLIANCE_GIFT_COLLECTION_TEMPLATE_KEYS = (
    "city.alliance.button",
    "alliance.menu.gifts",
    "alliance.gifts.panel",
    "alliance.gifts.normal_tab",
    "alliance.gifts.rare_tab",
    "alliance.gifts.claim_all_button",
    "alliance.gifts.reward_popup",
    "alliance.gifts.connection_popup",
    "alliance.gifts.no_claimable",
    "alliance.gifts.verification_required",
)
ALLIANCE_GIFT_COLLECTION_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "open_alliance",
    "open_alliance_gifts",
    "scan_gift_tabs",
    "process_gift_tab",
    "claim_gifts",
    "close_reward_popup",
    "handle_connection_popup",
    "verify_claim_state",
    "complete",
    "skipped",
    "recover",
    "failed",
    "cancelled",
)


class AllianceGiftTab(StrEnum):
    NORMAL = "NORMAL"
    RARE = "RARE"


class AllianceGiftScanStatus(StrEnum):
    READY = "READY"
    NONE_CLAIMABLE = "NONE_CLAIMABLE"
    CONNECTION_POPUP = "CONNECTION_POPUP"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


def _tab(value: AllianceGiftTab | str) -> AllianceGiftTab:
    if isinstance(value, AllianceGiftTab):
        return value
    try:
        return AllianceGiftTab(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in AllianceGiftTab)
        raise ValueError(f"Invalid alliance gift tab: {value!r}. Expected one of: {valid}.") from exc


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
class AllianceGiftCollectionPolicy:
    enabled_tabs: tuple[AllianceGiftTab | str, ...] = (AllianceGiftTab.NORMAL, AllianceGiftTab.RARE)
    minimum_detector_confidence: float = 0.85
    max_pages_per_tab: int = 4
    max_claim_iterations: int = 12
    allow_claim_all: bool = True

    def normalized(self) -> AllianceGiftCollectionPolicy:
        enabled = tuple(dict.fromkeys(_tab(item) for item in self.enabled_tabs))
        if not enabled:
            raise ValueError("At least one alliance gift tab must be enabled.")
        return AllianceGiftCollectionPolicy(
            enabled_tabs=enabled,
            minimum_detector_confidence=_require_confidence(
                self.minimum_detector_confidence,
                "minimum_detector_confidence",
            ),
            max_pages_per_tab=_require_positive_int(self.max_pages_per_tab, "max_pages_per_tab"),
            max_claim_iterations=_require_positive_int(self.max_claim_iterations, "max_claim_iterations"),
            allow_claim_all=bool(self.allow_claim_all),
        )

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "enabled_tabs": [item.value for item in normalized.enabled_tabs],
            "minimum_detector_confidence": normalized.minimum_detector_confidence,
            "max_pages_per_tab": normalized.max_pages_per_tab,
            "max_claim_iterations": normalized.max_claim_iterations,
            "allow_claim_all": normalized.allow_claim_all,
        }


@dataclass(frozen=True)
class AllianceGiftCollectionRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: AllianceGiftCollectionPolicy = field(default_factory=AllianceGiftCollectionPolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class AllianceGiftCollectionConfig:
    workflow_timeout_seconds: float = 120.0
    step_timeout_seconds: float = 15.0
    precondition_retry_limit: int = 1
    navigation_retry_limit: int = 1
    scan_retry_limit: int = 1
    action_retry_limit: int = 0
    retry_delay_seconds: float = 0.25


@dataclass(frozen=True)
class AllianceGiftObservation:
    tab: AllianceGiftTab | str
    claimable_count: int = 1
    confidence: float = 1.0
    gift_id: str = ""
    page_number: int = 1
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_tab(self) -> AllianceGiftTab:
        return _tab(self.tab)

    def to_json(self) -> dict[str, object]:
        return {
            "tab": self.normalized_tab().value,
            "claimable_count": self.claimable_count,
            "confidence": self.confidence,
            "gift_id": self.gift_id,
            "page_number": self.page_number,
            **self.data,
        }


@dataclass(frozen=True)
class AllianceGiftTabScan:
    status: AllianceGiftScanStatus | str
    tab: AllianceGiftTab | str
    page_number: int = 1
    observations: tuple[AllianceGiftObservation, ...] = ()
    has_next_page: bool = False
    scene_verified: bool = True
    tab_verified: bool = True
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_status(self) -> AllianceGiftScanStatus:
        if isinstance(self.status, AllianceGiftScanStatus):
            return self.status
        try:
            return AllianceGiftScanStatus(str(self.status).strip().upper())
        except ValueError as exc:
            valid = ", ".join(item.value for item in AllianceGiftScanStatus)
            raise ValueError(f"Invalid alliance gift scan status: {self.status!r}. Expected one of: {valid}.") from exc

    def normalized_tab(self) -> AllianceGiftTab:
        return _tab(self.tab)

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.normalized_status().value,
            "tab": self.normalized_tab().value,
            "page_number": self.page_number,
            "observations": [item.to_json() for item in self.observations],
            "has_next_page": self.has_next_page,
            "scene_verified": self.scene_verified,
            "tab_verified": self.tab_verified,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class AllianceGiftActionResult:
    success: bool
    changed: bool = False
    claimed_count: int = 0
    reward_popup_present: bool = False
    connection_popup_present: bool = False
    claimable_remaining: bool | None = None
    scene_verified: bool | None = None
    tab_verified: bool | None = None
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "changed": self.changed,
            "claimed_count": self.claimed_count,
            "reward_popup_present": self.reward_popup_present,
            "connection_popup_present": self.connection_popup_present,
            "claimable_remaining": self.claimable_remaining,
            "scene_verified": self.scene_verified,
            "tab_verified": self.tab_verified,
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


class AllianceGiftAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: AllianceGiftCollectionRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class AllianceGiftCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: AllianceGiftCollectionRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class AllianceGiftCollectionDriver(Protocol):
    def open_alliance(
        self,
        request: AllianceGiftCollectionRequest,
        character: Character,
        policy: AllianceGiftCollectionPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def open_alliance_gifts(
        self,
        request: AllianceGiftCollectionRequest,
        character: Character,
        policy: AllianceGiftCollectionPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def scan_gift_tab(
        self,
        request: AllianceGiftCollectionRequest,
        character: Character,
        tab: AllianceGiftTab,
        page_number: int,
        policy: AllianceGiftCollectionPolicy,
    ) -> AllianceGiftTabScan:
        ...

    def go_to_next_gift_page(
        self,
        request: AllianceGiftCollectionRequest,
        character: Character,
        tab: AllianceGiftTab,
        page_number: int,
        policy: AllianceGiftCollectionPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def claim_alliance_gifts(
        self,
        request: AllianceGiftCollectionRequest,
        character: Character,
        scan: AllianceGiftTabScan,
        policy: AllianceGiftCollectionPolicy,
    ) -> AllianceGiftActionResult:
        ...

    def close_reward_popup(
        self,
        request: AllianceGiftCollectionRequest,
        character: Character,
        scan: AllianceGiftTabScan,
        claim_result: AllianceGiftActionResult,
        policy: AllianceGiftCollectionPolicy,
    ) -> AllianceGiftActionResult:
        ...

    def handle_connection_popup(
        self,
        request: AllianceGiftCollectionRequest,
        character: Character,
        scan: AllianceGiftTabScan,
        action_result: AllianceGiftActionResult | None,
        policy: AllianceGiftCollectionPolicy,
    ) -> AllianceGiftActionResult:
        ...

    def verify_gift_claim_state(
        self,
        request: AllianceGiftCollectionRequest,
        character: Character,
        scan: AllianceGiftTabScan,
        claim_result: AllianceGiftActionResult,
        policy: AllianceGiftCollectionPolicy,
    ) -> AllianceGiftActionResult:
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
class _AllianceGiftState:
    request: AllianceGiftCollectionRequest
    character: Character | None = None
    policy: AllianceGiftCollectionPolicy | None = None
    claim_queue: list[AllianceGiftTabScan] = field(default_factory=list)
    scan_attempts: list[dict[str, object]] = field(default_factory=list)
    claim_attempts: list[dict[str, object]] = field(default_factory=list)
    reward_popup_attempts: list[dict[str, object]] = field(default_factory=list)
    connection_popup_attempts: list[dict[str, object]] = field(default_factory=list)
    verification_attempts: list[dict[str, object]] = field(default_factory=list)
    ignored_gifts: list[dict[str, object]] = field(default_factory=list)
    claimed_count: int = 0
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


class AllianceGiftCollectionWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: AllianceGiftCollectionDriver,
        account_precondition: AllianceGiftAccountPrecondition | None = None,
        character_precondition: AllianceGiftCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: AllianceGiftCollectionConfig | None = None,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.account_precondition = account_precondition
        self.character_precondition = character_precondition
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.incidents = incidents
        self.config = config or AllianceGiftCollectionConfig()
        self._states: dict[str, _AllianceGiftState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return ALLIANCE_GIFT_COLLECTION_STATES

    def execute(
        self,
        request: AllianceGiftCollectionRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _AllianceGiftState(request=request)
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
                budget=StepBudget(max_steps=len(ALLIANCE_GIFT_COLLECTION_STATES) + 24),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"alliance-gift-collection:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"alliance_gift_collection_run_id": token},
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
        for state in ALLIANCE_GIFT_COLLECTION_STATES:
            registry.register(f"alliance_gift_collection.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.precondition_retry_limit,
            "ensure_character": self.config.precondition_retry_limit,
            "ensure_game_running": self.config.precondition_retry_limit,
            "open_alliance": self.config.navigation_retry_limit,
            "open_alliance_gifts": self.config.navigation_retry_limit,
            "scan_gift_tabs": self.config.scan_retry_limit,
            "claim_gifts": self.config.action_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=ALLIANCE_GIFT_COLLECTION_WORKFLOW_KEY,
            name="Collect Alliance Gifts",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"alliance_gift_collection.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in ALLIANCE_GIFT_COLLECTION_STATES
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
        state: _AllianceGiftState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        request = state.request
        if request.instance_id <= 0:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, "instance_id must be positive.")
        if request.instance_index < 0:
            return state.stop(
                step.step_key,
                WorkflowOutcome.VALIDATION_FAILURE,
                "instance_index must be zero or greater.",
            )
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
                "template_keys": list(ALLIANCE_GIFT_COLLECTION_TEMPLATE_KEYS),
            },
        )

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _AllianceGiftState,
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
        state: _AllianceGiftState,
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
        state: _AllianceGiftState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.character_precondition is None:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data={"character_precondition": "not_configured"},
            )
        return self._action_to_step(
            step,
            state,
            self.character_precondition.ensure_character(state.request, _require_character(state)),
        )

    def _ensure_game_running(
        self,
        step: WorkflowStepSpec,
        state: _AllianceGiftState,
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

    def _open_alliance(
        self,
        step: WorkflowStepSpec,
        state: _AllianceGiftState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.open_alliance(state.request, _require_character(state), _require_policy(state)),
        )

    def _open_alliance_gifts(
        self,
        step: WorkflowStepSpec,
        state: _AllianceGiftState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.open_alliance_gifts(state.request, _require_character(state), _require_policy(state)),
        )

    def _scan_gift_tabs(
        self,
        step: WorkflowStepSpec,
        state: _AllianceGiftState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        policy = _require_policy(state)
        character = _require_character(state)
        for tab in policy.enabled_tabs:
            for page_number in range(1, policy.max_pages_per_tab + 1):
                context.cancellation_token.throw_if_cancelled()
                scan = self.driver.scan_gift_tab(state.request, character, tab, page_number, policy)
                if scan.screenshot_path:
                    state.screenshot_path = scan.screenshot_path
                self._record_scan(state, scan)
                status = scan.normalized_status()
                if status == AllianceGiftScanStatus.VERIFICATION_REQUIRED:
                    return state.stop(
                        step.step_key,
                        WorkflowOutcome.FATAL_FAILURE,
                        scan.message or "Verification screen requires manual intervention.",
                        screenshot_path=scan.screenshot_path,
                        data={"scan": scan.to_json()},
                    )
                if status == AllianceGiftScanStatus.CONNECTION_POPUP:
                    popup_step = self._handle_connection_popup_action(step, state, scan, None)
                    if popup_step.outcome != WorkflowOutcome.SUCCESS:
                        return popup_step
                    continue
                if status == AllianceGiftScanStatus.UNKNOWN:
                    return state.stop(
                        step.step_key,
                        WorkflowOutcome.BLOCKED,
                        scan.message or "Alliance gift claimable state could not be determined.",
                        screenshot_path=scan.screenshot_path,
                        data={"scan": scan.to_json()},
                    )
                claimable = _claimable_observations(scan.observations, policy)
                if claimable:
                    state.claim_queue.append(replace(scan, observations=tuple(claimable)))
                if not scan.has_next_page:
                    break
                next_page = self.driver.go_to_next_gift_page(state.request, character, tab, page_number + 1, policy)
                next_step = self._action_to_step(step, state, next_page)
                if next_step.outcome != WorkflowOutcome.SUCCESS:
                    return next_step
        if not state.claim_queue:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                "No claimable alliance gifts are available.",
                screenshot_path=state.screenshot_path,
                data={"skipped_reason": "no_claimable_gifts"},
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "claimable_pages": [item.to_json() for item in state.claim_queue],
                "scan_attempts": state.scan_attempts,
                "ignored_gifts": state.ignored_gifts,
            },
            screenshot_path=state.screenshot_path,
        )

    def _process_gift_tab(
        self,
        step: WorkflowStepSpec,
        _state: _AllianceGiftState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"handled_during": "scan_gift_tabs"})

    def _claim_gifts(
        self,
        step: WorkflowStepSpec,
        state: _AllianceGiftState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        policy = _require_policy(state)
        if not policy.allow_claim_all:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                "Alliance gift claim-all is disabled by policy.",
                data={"skipped_reason": "claim_all_disabled"},
            )
        for index, scan in enumerate(state.claim_queue, start=1):
            context.cancellation_token.throw_if_cancelled()
            if index > policy.max_claim_iterations:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    "Alliance gift claim iteration budget was exhausted before claimable gifts cleared.",
                    screenshot_path=state.screenshot_path,
                    data=self._collection_payload(state),
                )
            if not scan.scene_verified or not scan.tab_verified:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    "Alliance Gifts scene and tab must be verified before claim-all.",
                    screenshot_path=scan.screenshot_path,
                    data={"scan": scan.to_json()},
                )
            claim_result = self.driver.claim_alliance_gifts(
                state.request,
                _require_character(state),
                scan,
                policy,
            )
            if claim_result.screenshot_path:
                state.screenshot_path = claim_result.screenshot_path
            self._record_claim(state, scan, claim_result)
            if claim_result.connection_popup_present:
                popup_step = self._handle_connection_popup_action(step, state, scan, claim_result)
                if popup_step.outcome != WorkflowOutcome.SUCCESS:
                    return popup_step
            if not claim_result.success:
                return self._gift_action_failure(step, state, claim_result, "Alliance gift claim-all failed.")
            popup_step = self._close_reward_popup_action(step, state, scan, claim_result)
            if popup_step.outcome != WorkflowOutcome.SUCCESS:
                return popup_step
            verify_step = self._verify_claim_state_action(step, state, scan, claim_result)
            if verify_step.outcome != WorkflowOutcome.SUCCESS:
                return verify_step
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data=self._collection_payload(state),
            screenshot_path=state.screenshot_path,
        )

    def _close_reward_popup(
        self,
        step: WorkflowStepSpec,
        _state: _AllianceGiftState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"handled_during": "claim_gifts"})

    def _handle_connection_popup(
        self,
        step: WorkflowStepSpec,
        _state: _AllianceGiftState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"handled_during": "scan_or_claim"})

    def _verify_claim_state(
        self,
        step: WorkflowStepSpec,
        _state: _AllianceGiftState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"handled_during": "claim_gifts"})

    def _close_reward_popup_action(
        self,
        step: WorkflowStepSpec,
        state: _AllianceGiftState,
        scan: AllianceGiftTabScan,
        claim_result: AllianceGiftActionResult,
    ) -> WorkflowStepResult:
        if not claim_result.reward_popup_present:
            state.reward_popup_attempts.append(
                {
                    "tab": scan.normalized_tab().value,
                    "page_number": scan.page_number,
                    "popup_present": False,
                    "closed": False,
                }
            )
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"reward_popup_present": False})
        close = self.driver.close_reward_popup(
            state.request,
            _require_character(state),
            scan,
            claim_result,
            _require_policy(state),
        )
        if close.screenshot_path:
            state.screenshot_path = close.screenshot_path
        state.reward_popup_attempts.append(
            {
                "tab": scan.normalized_tab().value,
                "page_number": scan.page_number,
                "popup_present": True,
                "closed": close.success,
                **close.to_json(),
            }
        )
        if not close.success:
            return self._gift_action_failure(step, state, close, "Alliance gift reward popup could not be closed.")
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=close.to_json(), screenshot_path=close.screenshot_path)

    def _handle_connection_popup_action(
        self,
        step: WorkflowStepSpec,
        state: _AllianceGiftState,
        scan: AllianceGiftTabScan,
        action_result: AllianceGiftActionResult | None,
    ) -> WorkflowStepResult:
        handled = self.driver.handle_connection_popup(
            state.request,
            _require_character(state),
            scan,
            action_result,
            _require_policy(state),
        )
        if handled.screenshot_path:
            state.screenshot_path = handled.screenshot_path
        state.connection_popup_attempts.append(
            {
                "tab": scan.normalized_tab().value,
                "page_number": scan.page_number,
                "handled": handled.success,
                **handled.to_json(),
            }
        )
        if not handled.success:
            return self._gift_action_failure(step, state, handled, "Alliance gift connection popup could not be handled.")
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=handled.to_json(), screenshot_path=handled.screenshot_path)

    def _verify_claim_state_action(
        self,
        step: WorkflowStepSpec,
        state: _AllianceGiftState,
        scan: AllianceGiftTabScan,
        claim_result: AllianceGiftActionResult,
    ) -> WorkflowStepResult:
        verify = self.driver.verify_gift_claim_state(
            state.request,
            _require_character(state),
            scan,
            claim_result,
            _require_policy(state),
        )
        if verify.screenshot_path:
            state.screenshot_path = verify.screenshot_path
        state.verification_attempts.append(
            {
                "tab": scan.normalized_tab().value,
                "page_number": scan.page_number,
                **verify.to_json(),
            }
        )
        if not _claim_postcondition_verified(verify):
            failure = replace(
                verify,
                message=("Alliance gift claimable state remained after claim. " f"{verify.message}").strip(),
            )
            return self._gift_action_failure(step, state, failure, "Alliance gift claim postcondition was not verified.")
        state.claimed_count += _claimed_count(scan, claim_result, verify)
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=verify.to_json(), screenshot_path=verify.screenshot_path)

    def _complete(self, step: WorkflowStepSpec, state: _AllianceGiftState) -> WorkflowStepResult:
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

    def _skipped(self, step: WorkflowStepSpec, state: _AllianceGiftState) -> WorkflowStepResult:
        if state.terminal_outcome == WorkflowOutcome.SKIPPED:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data={"skipped_reason": state.terminal_reason, **self._collection_payload(state)},
                screenshot_path=state.screenshot_path,
            )
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED)

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _AllianceGiftState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if _is_manual_stop(state):
            state.recovery_outcome = {"attempted": False, "reason": "manual_intervention_required"}
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
        state.recovery_outcome = self._monitor_recovery(state, _job_run_id(context))
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _AllianceGiftState) -> WorkflowStepResult:
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
        state: _AllianceGiftState,
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
                action.message or "Alliance gift collection action failed.",
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            action.message or "Alliance gift collection action failed.",
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _gift_action_failure(
        self,
        step: WorkflowStepSpec,
        state: _AllianceGiftState,
        result: AllianceGiftActionResult,
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

    def _record_scan(self, state: _AllianceGiftState, scan: AllianceGiftTabScan) -> None:
        policy = _require_policy(state)
        selected = _claimable_observations(scan.observations, policy)
        state.scan_attempts.append(
            {
                "tab": scan.normalized_tab().value,
                "page_number": scan.page_number,
                "status": scan.normalized_status().value,
                "observation_count": len(scan.observations),
                "claimable_count": sum(max(0, item.claimable_count) for item in selected),
                "has_next_page": scan.has_next_page,
                "scene_verified": scan.scene_verified,
                "tab_verified": scan.tab_verified,
                "screenshot_path": scan.screenshot_path,
                **scan.data,
            }
        )
        state.ignored_gifts.extend(
            {
                **item.to_json(),
                "ignored_reason": _ignore_reason(item, policy),
            }
            for item in scan.observations
            if _ignore_reason(item, policy)
        )

    def _record_claim(
        self,
        state: _AllianceGiftState,
        scan: AllianceGiftTabScan,
        result: AllianceGiftActionResult,
    ) -> None:
        state.claim_attempts.append(
            {
                "tab": scan.normalized_tab().value,
                "page_number": scan.page_number,
                "observation_count": len(scan.observations),
                **result.to_json(),
            }
        )

    def _collection_payload(self, state: _AllianceGiftState) -> dict[str, object]:
        return {
            "claimed_count": state.claimed_count,
            "scanned_tabs_pages": state.scan_attempts,
            "claim_attempts": state.claim_attempts,
            "reward_popup_attempts": state.reward_popup_attempts,
            "connection_popup_attempts": state.connection_popup_attempts,
            "verification_attempts": state.verification_attempts,
            "ignored_gifts": state.ignored_gifts,
            "failure_evidence": {
                "screenshot_path": state.screenshot_path,
                "terminal_state": state.terminal_state,
                "terminal_reason": state.terminal_reason,
            },
        }

    def _state_from_context(self, context: WorkflowExecutionContext) -> _AllianceGiftState:
        token = str(context.metadata.get("alliance_gift_collection_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("Alliance gift collection runtime state is missing.") from exc

    def _open_incident(self, state: _AllianceGiftState) -> None:
        if self.incidents is None or not state.failed or state.incident_opened:
            return
        self.incidents.save(
            Incident(
                incident_key=f"alliance-gift-collection:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="Alliance gift collection blocked",
                details=state.terminal_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )
        state.incident_opened = True

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _AllianceGiftState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "Alliance gift collection workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(
        self,
        result: WorkflowExecutionResult,
        state: _AllianceGiftState,
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
        state: _AllianceGiftState,
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
        state: _AllianceGiftState,
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
            "skipped_reason": state.terminal_reason if result.outcome == WorkflowOutcome.SKIPPED else "",
            "failure_state": state.terminal_state if result.outcome.is_failure else "",
            "failure_reason": state.terminal_reason if result.outcome.is_failure else "",
            "recovery_outcome": state.recovery_outcome,
        }

    def _update_persisted_run(
        self,
        result: WorkflowExecutionResult,
        state: _AllianceGiftState,
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


def _claimable_observations(
    observations: tuple[AllianceGiftObservation, ...],
    policy: AllianceGiftCollectionPolicy,
) -> list[AllianceGiftObservation]:
    selected: list[AllianceGiftObservation] = []
    seen: set[tuple[str, str, int]] = set()
    for observation in observations:
        if _ignore_reason(observation, policy):
            continue
        identity = (observation.normalized_tab().value, observation.gift_id, observation.page_number)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(observation)
    return selected


def _ignore_reason(
    observation: AllianceGiftObservation,
    policy: AllianceGiftCollectionPolicy,
) -> str:
    if observation.normalized_tab() not in policy.enabled_tabs:
        return "disabled_tab"
    if observation.claimable_count <= 0:
        return "not_claimable"
    if observation.confidence < policy.minimum_detector_confidence:
        return "below_confidence_threshold"
    return ""


def _claim_postcondition_verified(verify: AllianceGiftActionResult) -> bool:
    return verify.success and verify.changed and verify.claimable_remaining is False


def _claimed_count(
    scan: AllianceGiftTabScan,
    claim_result: AllianceGiftActionResult,
    verify: AllianceGiftActionResult,
) -> int:
    observed = sum(max(0, item.claimable_count) for item in scan.observations)
    return max(1, verify.claimed_count, claim_result.claimed_count, observed)


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
        action_type=f"alliance_gift_collection.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _AllianceGiftState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _AllianceGiftState) -> AllianceGiftCollectionPolicy:
    if state.policy is None:
        raise RuntimeError("Alliance gift collection policy has not been validated.")
    return state.policy


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_manual_stop(state: _AllianceGiftState) -> bool:
    text = state.terminal_reason.lower()
    return "verification" in text or "manual" in text
