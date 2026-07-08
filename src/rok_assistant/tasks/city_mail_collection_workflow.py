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


CITY_MAIL_COLLECTION_WORKFLOW_KEY = "city-mail-collection"
CITY_MAIL_COLLECTION_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "open_mail",
    "scan_categories",
    "process_category",
    "handle_confirmation",
    "verify_postcondition",
    "complete",
    "recover",
    "failed",
    "cancelled",
)


class MailCategory(StrEnum):
    REWARDS = "REWARDS"
    REPORTS = "REPORTS"
    SYSTEM = "SYSTEM"
    ALLIANCE = "ALLIANCE"
    PLAYER = "PLAYER"
    UNKNOWN = "UNKNOWN"


class MailAction(StrEnum):
    CLAIM_ALL = "CLAIM_ALL"
    READ_ALL = "READ_ALL"
    NONE = "NONE"


class MailCategoryScanStatus(StrEnum):
    READY = "READY"
    NO_MAIL = "NO_MAIL"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


class MailConfirmationKind(StrEnum):
    NONE = "NONE"
    SAFE_CLAIM = "SAFE_CLAIM"
    SAFE_READ = "SAFE_READ"
    UNKNOWN = "UNKNOWN"
    DESTRUCTIVE = "DESTRUCTIVE"


def _category(value: MailCategory | str) -> MailCategory:
    if isinstance(value, MailCategory):
        return value
    try:
        return MailCategory(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in MailCategory)
        raise ValueError(f"Invalid mail category: {value!r}. Expected one of: {valid}.") from exc


def _action(value: MailAction | str) -> MailAction:
    if isinstance(value, MailAction):
        return value
    try:
        return MailAction(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in MailAction)
        raise ValueError(f"Invalid mail action: {value!r}. Expected one of: {valid}.") from exc


def _require_positive_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _require_non_negative_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be zero or greater.")
    return value


@dataclass(frozen=True)
class CityMailCollectionPolicy:
    enabled_categories: tuple[MailCategory | str, ...] = (
        MailCategory.REWARDS,
        MailCategory.REPORTS,
    )
    claim_all_categories: tuple[MailCategory | str, ...] = (MailCategory.REWARDS,)
    read_all_categories: tuple[MailCategory | str, ...] = (MailCategory.REPORTS,)

    def normalized(self) -> CityMailCollectionPolicy:
        enabled = tuple(dict.fromkeys(_category(item) for item in self.enabled_categories))
        if not enabled:
            raise ValueError("At least one mail category must be enabled.")
        if MailCategory.UNKNOWN in enabled:
            raise ValueError("UNKNOWN mail category cannot be enabled.")
        claim = tuple(dict.fromkeys(_category(item) for item in self.claim_all_categories))
        read = tuple(dict.fromkeys(_category(item) for item in self.read_all_categories))
        disallowed = [item.value for item in (*claim, *read) if item not in enabled]
        if disallowed:
            raise ValueError(f"Mail actions configured for non-whitelisted categories: {', '.join(disallowed)}.")
        overlap = sorted({item.value for item in claim if item in read})
        if overlap:
            raise ValueError(f"Mail category cannot be both claim-all and read-all: {', '.join(overlap)}.")
        return CityMailCollectionPolicy(
            enabled_categories=enabled,
            claim_all_categories=claim,
            read_all_categories=read,
        )

    def action_for(self, category: MailCategory | str) -> MailAction:
        normalized = self.normalized()
        item = _category(category)
        if item not in normalized.enabled_categories:
            return MailAction.NONE
        if item in normalized.claim_all_categories:
            return MailAction.CLAIM_ALL
        if item in normalized.read_all_categories:
            return MailAction.READ_ALL
        return MailAction.NONE

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "enabled_categories": [item.value for item in normalized.enabled_categories],
            "claim_all_categories": [item.value for item in normalized.claim_all_categories],
            "read_all_categories": [item.value for item in normalized.read_all_categories],
        }


@dataclass(frozen=True)
class CityMailCollectionRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: CityMailCollectionPolicy = field(default_factory=CityMailCollectionPolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class CityMailCollectionConfig:
    workflow_timeout_seconds: float = 90.0
    step_timeout_seconds: float = 15.0
    precondition_retry_limit: int = 1
    navigation_retry_limit: int = 1
    action_retry_limit: int = 0
    retry_delay_seconds: float = 0.25
    max_category_pages: int = 4

    def normalized_max_category_pages(self) -> int:
        return _require_positive_int(self.max_category_pages, "max_category_pages")


@dataclass(frozen=True)
class MailCategoryObservation:
    category: MailCategory | str
    unread_badge_count: int = 0
    claimable_count: int = 0
    category_id: str = ""
    page_number: int = 1
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_category(self) -> MailCategory:
        return _category(self.category)

    def to_json(self) -> dict[str, object]:
        return {
            "category": self.normalized_category().value,
            "unread_badge_count": self.unread_badge_count,
            "claimable_count": self.claimable_count,
            "category_id": self.category_id,
            "page_number": self.page_number,
            **self.data,
        }


@dataclass(frozen=True)
class MailCategoryScan:
    status: MailCategoryScanStatus | str
    observations: tuple[MailCategoryObservation, ...] = ()
    has_next_page: bool = False
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_status(self) -> MailCategoryScanStatus:
        if isinstance(self.status, MailCategoryScanStatus):
            return self.status
        try:
            return MailCategoryScanStatus(str(self.status).strip().upper())
        except ValueError as exc:
            valid = ", ".join(item.value for item in MailCategoryScanStatus)
            raise ValueError(f"Invalid mail category scan status: {self.status!r}. Expected one of: {valid}.") from exc

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.normalized_status().value,
            "observations": [item.to_json() for item in self.observations],
            "has_next_page": self.has_next_page,
            **self.data,
        }


@dataclass(frozen=True)
class CityMailActionResult:
    success: bool
    changed: bool = False
    confirmation_kind: MailConfirmationKind | str = MailConfirmationKind.NONE
    category: MailCategory | str | None = None
    action: MailAction | str | None = None
    claim_count: int = 0
    unread_before: int | None = None
    unread_after: int | None = None
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_confirmation_kind(self) -> MailConfirmationKind:
        if isinstance(self.confirmation_kind, MailConfirmationKind):
            return self.confirmation_kind
        try:
            return MailConfirmationKind(str(self.confirmation_kind).strip().upper())
        except ValueError as exc:
            valid = ", ".join(item.value for item in MailConfirmationKind)
            raise ValueError(f"Invalid mail confirmation kind: {self.confirmation_kind!r}. Expected one of: {valid}.") from exc

    def normalized_action(self) -> MailAction:
        if self.action is None:
            return MailAction.NONE
        return _action(self.action)

    def to_json(self) -> dict[str, object]:
        payload = {
            "success": self.success,
            "changed": self.changed,
            "confirmation_kind": self.normalized_confirmation_kind().value,
            "claim_count": self.claim_count,
            "unread_before": self.unread_before,
            "unread_after": self.unread_after,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }
        if self.category is not None:
            payload["category"] = _category(self.category).value
        if self.action is not None:
            payload["action"] = self.normalized_action().value
        return payload


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


class CityMailAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: CityMailCollectionRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class CityMailCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: CityMailCollectionRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class CityMailCollectionDriver(Protocol):
    def open_mail(
        self,
        request: CityMailCollectionRequest,
        character: Character,
        policy: CityMailCollectionPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def scan_mail_categories(
        self,
        request: CityMailCollectionRequest,
        character: Character,
        policy: CityMailCollectionPolicy,
        page_number: int,
    ) -> MailCategoryScan:
        ...

    def go_to_next_mail_category_page(
        self,
        request: CityMailCollectionRequest,
        character: Character,
        policy: CityMailCollectionPolicy,
        page_number: int,
    ) -> ResourceGatheringActionResult:
        ...

    def open_mail_category(
        self,
        request: CityMailCollectionRequest,
        character: Character,
        observation: MailCategoryObservation,
        action: MailAction,
        policy: CityMailCollectionPolicy,
    ) -> CityMailActionResult:
        ...

    def claim_all_mail(
        self,
        request: CityMailCollectionRequest,
        character: Character,
        observation: MailCategoryObservation,
        policy: CityMailCollectionPolicy,
    ) -> CityMailActionResult:
        ...

    def read_all_mail(
        self,
        request: CityMailCollectionRequest,
        character: Character,
        observation: MailCategoryObservation,
        policy: CityMailCollectionPolicy,
    ) -> CityMailActionResult:
        ...

    def confirm_mail_action(
        self,
        request: CityMailCollectionRequest,
        character: Character,
        observation: MailCategoryObservation,
        action_result: CityMailActionResult,
        policy: CityMailCollectionPolicy,
    ) -> CityMailActionResult:
        ...

    def verify_mail_postcondition(
        self,
        request: CityMailCollectionRequest,
        character: Character,
        observation: MailCategoryObservation,
        action: MailAction,
        policy: CityMailCollectionPolicy,
    ) -> CityMailActionResult:
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
class _CityMailState:
    request: CityMailCollectionRequest
    character: Character | None = None
    policy: CityMailCollectionPolicy | None = None
    category_queue: list[MailCategoryObservation] = field(default_factory=list)
    ignored_categories: list[dict[str, object]] = field(default_factory=list)
    processed_categories: list[dict[str, object]] = field(default_factory=list)
    scan_attempts: list[dict[str, object]] = field(default_factory=list)
    confirmation_attempts: list[dict[str, object]] = field(default_factory=list)
    verification_attempts: list[dict[str, object]] = field(default_factory=list)
    total_claim_count: int = 0
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


class CityMailCollectionWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: CityMailCollectionDriver,
        account_precondition: CityMailAccountPrecondition | None = None,
        character_precondition: CityMailCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: CityMailCollectionConfig | None = None,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.account_precondition = account_precondition
        self.character_precondition = character_precondition
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.incidents = incidents
        self.config = config or CityMailCollectionConfig()
        self._states: dict[str, _CityMailState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return CITY_MAIL_COLLECTION_STATES

    def execute(
        self,
        request: CityMailCollectionRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _CityMailState(request=request)
        self._states[token] = state
        persistence = None
        if self.job_runs is not None and self.step_runs is not None and request.job_id is not None:
            persistence = WorkflowRunRepositoryRecorder(self.job_runs, self.step_runs)
        try:
            max_pages = self.config.normalized_max_category_pages()
            context = WorkflowExecutionContext(
                cancellation_token=cancellation_token or CancellationToken(),
                deadline=WorkflowDeadline.from_timeout(
                    self.config.workflow_timeout_seconds,
                    time.monotonic,
                ),
                budget=StepBudget(max_steps=len(CITY_MAIL_COLLECTION_STATES) + max_pages + 10),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"city-mail-collection:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"city_mail_collection_run_id": token},
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
        for state in CITY_MAIL_COLLECTION_STATES:
            registry.register(f"city_mail_collection.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.precondition_retry_limit,
            "ensure_character": self.config.precondition_retry_limit,
            "ensure_game_running": self.config.precondition_retry_limit,
            "open_mail": self.config.navigation_retry_limit,
            "scan_categories": self.config.navigation_retry_limit,
            "process_category": self.config.action_retry_limit,
            "handle_confirmation": self.config.action_retry_limit,
            "verify_postcondition": self.config.action_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=CITY_MAIL_COLLECTION_WORKFLOW_KEY,
            name="Collect Safe Mail",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"city_mail_collection.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in CITY_MAIL_COLLECTION_STATES
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
        state: _CityMailState,
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
            max_pages = self.config.normalized_max_category_pages()
        except ValueError as exc:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, str(exc))
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"policy": state.policy.to_json(), "max_category_pages": max_pages},
        )

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _CityMailState,
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
        state: _CityMailState,
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
        state: _CityMailState,
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
        state: _CityMailState,
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

    def _open_mail(
        self,
        step: WorkflowStepSpec,
        state: _CityMailState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.open_mail(
                state.request,
                _require_character(state),
                _require_policy(state),
            ),
        )

    def _scan_categories(
        self,
        step: WorkflowStepSpec,
        state: _CityMailState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        character = _require_character(state)
        policy = _require_policy(state)
        for page_number in range(1, self.config.normalized_max_category_pages() + 1):
            context.cancellation_token.throw_if_cancelled()
            scan = self.driver.scan_mail_categories(state.request, character, policy, page_number)
            self._record_scan(state, page_number, scan)
            if scan.screenshot_path:
                state.screenshot_path = scan.screenshot_path
            status = scan.normalized_status()
            if status == MailCategoryScanStatus.VERIFICATION_REQUIRED:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.FATAL_FAILURE,
                    scan.message or "Verification screen requires manual intervention.",
                    screenshot_path=scan.screenshot_path,
                    data={"scan": scan.to_json()},
                )
            if status == MailCategoryScanStatus.UNKNOWN:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    scan.message or "Mail categories could not be determined safely.",
                    screenshot_path=scan.screenshot_path,
                    data={"scan": scan.to_json()},
                )
            if status == MailCategoryScanStatus.NO_MAIL and not state.category_queue:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.SKIPPED,
                    scan.message or "No mail is available to process.",
                    screenshot_path=scan.screenshot_path,
                    data={"scan": scan.to_json()},
                )
            for observation in scan.observations:
                action = policy.action_for(observation.normalized_category())
                if action == MailAction.NONE:
                    state.ignored_categories.append(
                        {
                            "category": observation.normalized_category().value,
                            "category_id": observation.category_id,
                            "page_number": observation.page_number,
                            "unread_badge_before": observation.unread_badge_count,
                            "claimable_count": observation.claimable_count,
                            "reason": "not_whitelisted_or_no_allowed_action",
                        }
                    )
                    continue
                state.category_queue.append(observation)
            if not scan.has_next_page:
                break
            if page_number >= self.config.normalized_max_category_pages():
                break
            next_page = self.driver.go_to_next_mail_category_page(state.request, character, policy, page_number + 1)
            if next_page.screenshot_path:
                state.screenshot_path = next_page.screenshot_path
            if not next_page.success:
                return self._action_failure(
                    step,
                    state,
                    next_page,
                    fallback_message="Mail category pagination failed.",
                )
        if not state.category_queue:
            return state.stop(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                "No whitelisted mail categories had allowed actions.",
                screenshot_path=state.screenshot_path,
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"queued_category_count": len(state.category_queue), "ignored_categories": state.ignored_categories},
            screenshot_path=state.screenshot_path,
        )

    def _process_category(
        self,
        step: WorkflowStepSpec,
        state: _CityMailState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        character = _require_character(state)
        policy = _require_policy(state)
        for observation in state.category_queue:
            context.cancellation_token.throw_if_cancelled()
            action = policy.action_for(observation.normalized_category())
            open_result = self.driver.open_mail_category(state.request, character, observation, action, policy)
            if open_result.screenshot_path:
                state.screenshot_path = open_result.screenshot_path
            if not open_result.success:
                return self._mail_action_failure(step, state, open_result, "Mail category could not be opened.")
            action_result = (
                self.driver.claim_all_mail(state.request, character, observation, policy)
                if action == MailAction.CLAIM_ALL
                else self.driver.read_all_mail(state.request, character, observation, policy)
            )
            if action_result.screenshot_path:
                state.screenshot_path = action_result.screenshot_path
            unsafe = self._handle_confirmation_action(step, state, observation, action_result)
            if unsafe is not None:
                return unsafe
            if not action_result.success:
                return self._mail_action_failure(
                    step,
                    state,
                    action_result,
                    f"{observation.normalized_category().value} mail {action.value} did not complete.",
                )
            verify_step = self._verify_postcondition_action(step, state, observation, action)
            if verify_step.outcome != WorkflowOutcome.SUCCESS:
                return verify_step
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data=self._collection_payload(state),
            screenshot_path=state.screenshot_path,
        )

    def _handle_confirmation(
        self,
        step: WorkflowStepSpec,
        _state: _CityMailState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"handled_during": "process_category"})

    def _verify_postcondition(
        self,
        step: WorkflowStepSpec,
        _state: _CityMailState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"handled_during": "process_category"})

    def _handle_confirmation_action(
        self,
        step: WorkflowStepSpec,
        state: _CityMailState,
        observation: MailCategoryObservation,
        action_result: CityMailActionResult,
    ) -> WorkflowStepResult | None:
        kind = action_result.normalized_confirmation_kind()
        state.confirmation_attempts.append(
            {
                "category": observation.normalized_category().value,
                "category_id": observation.category_id,
                "confirmation_kind": kind.value,
                "screenshot_path": action_result.screenshot_path,
                **action_result.data,
            }
        )
        if kind == MailConfirmationKind.DESTRUCTIVE:
            return state.stop(
                step.step_key,
                WorkflowOutcome.FATAL_FAILURE,
                action_result.message or "Destructive mail confirmation detected; mail collection stopped.",
                screenshot_path=action_result.screenshot_path,
                data={"confirmation": action_result.to_json()},
            )
        if kind == MailConfirmationKind.UNKNOWN:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                action_result.message or "Unknown mail confirmation detected; mail collection stopped.",
                screenshot_path=action_result.screenshot_path,
                data={"confirmation": action_result.to_json()},
            )
        if kind == MailConfirmationKind.NONE:
            return None
        confirm = self.driver.confirm_mail_action(
            state.request,
            _require_character(state),
            observation,
            action_result,
            _require_policy(state),
        )
        if confirm.screenshot_path:
            state.screenshot_path = confirm.screenshot_path
        confirm_kind = confirm.normalized_confirmation_kind()
        state.confirmation_attempts.append(
            {
                "category": observation.normalized_category().value,
                "category_id": observation.category_id,
                "confirmation_kind": confirm_kind.value,
                "confirmed": confirm.success,
                "screenshot_path": confirm.screenshot_path,
                **confirm.data,
            }
        )
        if confirm_kind == MailConfirmationKind.DESTRUCTIVE:
            return state.stop(
                step.step_key,
                WorkflowOutcome.FATAL_FAILURE,
                confirm.message or "Destructive mail confirmation detected; mail collection stopped.",
                screenshot_path=confirm.screenshot_path,
                data={"confirmation": confirm.to_json()},
            )
        if confirm_kind == MailConfirmationKind.UNKNOWN:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                confirm.message or "Unknown mail confirmation detected; mail collection stopped.",
                screenshot_path=confirm.screenshot_path,
                data={"confirmation": confirm.to_json()},
            )
        if not confirm.success:
            return self._mail_action_failure(step, state, confirm, "Mail confirmation could not be handled.")
        return None

    def _verify_postcondition_action(
        self,
        step: WorkflowStepSpec,
        state: _CityMailState,
        observation: MailCategoryObservation,
        action: MailAction,
    ) -> WorkflowStepResult:
        verify = self.driver.verify_mail_postcondition(
            state.request,
            _require_character(state),
            observation,
            action,
            _require_policy(state),
        )
        if verify.screenshot_path:
            state.screenshot_path = verify.screenshot_path
        state.verification_attempts.append(
            {
                "category": observation.normalized_category().value,
                "category_id": observation.category_id,
                "action": action.value,
                **verify.to_json(),
            }
        )
        if not _postcondition_verified(observation, action, verify):
            fallback_message = (
                f"{observation.normalized_category().value} mail {action.value} postcondition was not verified."
            )
            failure = replace(
                verify,
                message=f"{fallback_message} {verify.message}".strip(),
            )
            return self._mail_action_failure(
                step,
                state,
                failure,
                fallback_message,
            )
        state.total_claim_count += max(0, verify.claim_count)
        state.processed_categories.append(
            {
                "category": observation.normalized_category().value,
                "category_id": observation.category_id,
                "action": action.value,
                "claim_count": max(0, verify.claim_count),
                "unread_badge_before": observation.unread_badge_count,
                "unread_badge_after": verify.unread_after,
                "postcondition_verified": True,
            }
        )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data=verify.to_json(),
            screenshot_path=verify.screenshot_path,
        )

    def _complete(self, step: WorkflowStepSpec, state: _CityMailState) -> WorkflowStepResult:
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
        state: _CityMailState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if _is_manual_stop(state):
            state.recovery_outcome = {"attempted": False, "reason": "manual_or_unsafe_dialog"}
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
        state.recovery_outcome = self._monitor_recovery(state, _job_run_id(context))
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _CityMailState) -> WorkflowStepResult:
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
        state: _CityMailState,
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
                action.message or "City mail collection action failed.",
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            action.message or "City mail collection action failed.",
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _action_failure(
        self,
        step: WorkflowStepSpec,
        state: _CityMailState,
        result: ResourceGatheringActionResult,
        *,
        fallback_message: str,
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

    def _mail_action_failure(
        self,
        step: WorkflowStepSpec,
        state: _CityMailState,
        result: CityMailActionResult,
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

    def _record_scan(
        self,
        state: _CityMailState,
        page_number: int,
        scan: MailCategoryScan,
    ) -> None:
        state.scan_attempts.append(
            {
                "page_number": page_number,
                "status": scan.normalized_status().value,
                "observation_count": len(scan.observations),
                "has_next_page": scan.has_next_page,
                "screenshot_path": scan.screenshot_path,
                **scan.data,
            }
        )

    def _collection_payload(self, state: _CityMailState) -> dict[str, object]:
        return {
            "processed_category_count": len(state.processed_categories),
            "processed_categories": state.processed_categories,
            "ignored_categories": state.ignored_categories,
            "total_claim_count": state.total_claim_count,
            "scan_attempts": state.scan_attempts,
            "confirmation_attempts": state.confirmation_attempts,
            "verification_attempts": state.verification_attempts,
            "failure_evidence": {
                "screenshot_path": state.screenshot_path,
                "terminal_state": state.terminal_state,
                "terminal_reason": state.terminal_reason,
            },
        }

    def _state_from_context(self, context: WorkflowExecutionContext) -> _CityMailState:
        token = str(context.metadata.get("city_mail_collection_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("City mail collection runtime state is missing.") from exc

    def _open_incident(self, state: _CityMailState) -> None:
        if self.incidents is None or not state.failed or state.incident_opened:
            return
        self.incidents.save(
            Incident(
                incident_key=f"city-mail-collection:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="City mail collection blocked",
                details=state.terminal_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )
        state.incident_opened = True

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _CityMailState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "City mail collection workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(
        self,
        result: WorkflowExecutionResult,
        state: _CityMailState,
    ) -> None:
        if state.failed and not state.recovery_outcome:
            if _is_manual_stop(state):
                state.recovery_outcome = {"attempted": False, "reason": "manual_or_unsafe_dialog"}
            else:
                state.recovery_outcome = self._monitor_recovery(state, result.job_run_id)
        if state.failed:
            self._open_incident(state)

    def _monitor_recovery(
        self,
        state: _CityMailState,
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
        state: _CityMailState,
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
        state: _CityMailState,
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


def _postcondition_verified(
    observation: MailCategoryObservation,
    action: MailAction,
    verify: CityMailActionResult,
) -> bool:
    if not verify.success or not verify.changed:
        return False
    before = verify.unread_before if verify.unread_before is not None else observation.unread_badge_count
    after = verify.unread_after
    if after is None:
        return False
    if action == MailAction.CLAIM_ALL:
        return verify.claim_count > 0 and after <= before
    if action == MailAction.READ_ALL:
        return before > 0 and after < before
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
        action_type=f"city_mail_collection.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _CityMailState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _CityMailState) -> CityMailCollectionPolicy:
    if state.policy is None:
        raise RuntimeError("City mail collection policy has not been validated.")
    return state.policy


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_manual_stop(state: _CityMailState) -> bool:
    text = state.terminal_reason.lower()
    return "verification" in text or "confirmation" in text or "destructive" in text or "unknown" in text
