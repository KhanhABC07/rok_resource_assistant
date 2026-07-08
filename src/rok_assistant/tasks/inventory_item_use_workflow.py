from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Mapping, Protocol
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


INVENTORY_ITEM_USE_WORKFLOW_KEY = "inventory-item-use"
INVENTORY_ITEM_USE_TEMPLATE_KEYS = (
    "city.inventory.button",
    "inventory.scene",
    "inventory.item.card",
    "inventory.item.quantity",
    "inventory.item.use_button",
    "inventory.item.amount_input",
    "inventory.item.confirm",
)
INVENTORY_ITEM_USE_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "open_inventory",
    "scan_inventory",
    "select_item",
    "verify_item_identity",
    "check_budget",
    "preview_item_use",
    "enter_quantity",
    "confirm_usage",
    "verify_inventory_delta",
    "complete",
    "skipped",
    "recover",
    "failed",
    "cancelled",
)


class InventoryItemType(StrEnum):
    RESOURCE_FOOD = "RESOURCE_FOOD"
    RESOURCE_WOOD = "RESOURCE_WOOD"
    RESOURCE_STONE = "RESOURCE_STONE"
    RESOURCE_GOLD = "RESOURCE_GOLD"
    SPEEDUP_BUILDING = "SPEEDUP_BUILDING"
    SPEEDUP_TRAINING = "SPEEDUP_TRAINING"
    ACTION_POINT = "ACTION_POINT"
    EXP_BOOK = "EXP_BOOK"
    CHEST_KNOWN = "CHEST_KNOWN"
    UNKNOWN = "UNKNOWN"


class InventoryItemRarity(StrEnum):
    COMMON = "COMMON"
    UNCOMMON = "UNCOMMON"
    RARE = "RARE"
    EPIC = "EPIC"
    LEGENDARY = "LEGENDARY"
    PREMIUM = "PREMIUM"
    UNKNOWN = "UNKNOWN"


def _item_type(value: InventoryItemType | str) -> InventoryItemType:
    if isinstance(value, InventoryItemType):
        return value
    try:
        return InventoryItemType(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in InventoryItemType)
        raise ValueError(f"Invalid inventory item type: {value!r}. Expected one of: {valid}.") from exc


def _rarity(value: InventoryItemRarity | str) -> InventoryItemRarity:
    if isinstance(value, InventoryItemRarity):
        return value
    try:
        return InventoryItemRarity(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in InventoryItemRarity)
        raise ValueError(f"Invalid inventory item rarity: {value!r}. Expected one of: {valid}.") from exc


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
class InventoryItemBudget:
    max_per_run: int
    max_per_day: int
    total_budget: int
    used_today: int = 0
    used_total: int = 0

    def normalized(self) -> InventoryItemBudget:
        max_per_run = _require_positive_int(self.max_per_run, "max_per_run")
        max_per_day = _require_positive_int(self.max_per_day, "max_per_day")
        total_budget = _require_positive_int(self.total_budget, "total_budget")
        used_today = _require_non_negative_int(self.used_today, "used_today")
        used_total = _require_non_negative_int(self.used_total, "used_total")
        return InventoryItemBudget(
            max_per_run=max_per_run,
            max_per_day=max_per_day,
            total_budget=total_budget,
            used_today=used_today,
            used_total=used_total,
        )

    def remaining_daily(self) -> int:
        normalized = self.normalized()
        return max(0, normalized.max_per_day - normalized.used_today)

    def remaining_total(self) -> int:
        normalized = self.normalized()
        return max(0, normalized.total_budget - normalized.used_total)

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "max_per_run": normalized.max_per_run,
            "max_per_day": normalized.max_per_day,
            "total_budget": normalized.total_budget,
            "used_today": normalized.used_today,
            "used_total": normalized.used_total,
            "remaining_daily": normalized.remaining_daily(),
            "remaining_total": normalized.remaining_total(),
        }


@dataclass(frozen=True)
class InventoryItemUsePolicy:
    whitelist: Mapping[InventoryItemType | str, InventoryItemBudget] = field(default_factory=dict)
    dry_run: bool = False
    allow_premium_items: bool = False
    allow_rare_items: bool = False
    minimum_detector_confidence: float = 0.85

    def normalized(self) -> InventoryItemUsePolicy:
        normalized_whitelist: dict[InventoryItemType, InventoryItemBudget] = {}
        for key, budget in self.whitelist.items():
            item_type = _item_type(key)
            if item_type == InventoryItemType.UNKNOWN:
                raise ValueError("UNKNOWN cannot be whitelisted for item use.")
            normalized_whitelist[item_type] = budget.normalized()
        if not normalized_whitelist:
            raise ValueError("At least one semantic item type must be whitelisted.")
        return InventoryItemUsePolicy(
            whitelist=normalized_whitelist,
            dry_run=bool(self.dry_run),
            allow_premium_items=bool(self.allow_premium_items),
            allow_rare_items=bool(self.allow_rare_items),
            minimum_detector_confidence=_require_confidence(
                self.minimum_detector_confidence,
                "minimum_detector_confidence",
            ),
        )

    def budget_for(self, item_type: InventoryItemType | str) -> InventoryItemBudget | None:
        return self.normalized().whitelist.get(_item_type(item_type))

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "whitelist": {key.value: value.to_json() for key, value in normalized.whitelist.items()},
            "dry_run": normalized.dry_run,
            "allow_premium_items": normalized.allow_premium_items,
            "allow_rare_items": normalized.allow_rare_items,
            "minimum_detector_confidence": normalized.minimum_detector_confidence,
        }


@dataclass(frozen=True)
class InventoryItemUseRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    requested_item_type: InventoryItemType | str
    requested_quantity: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: InventoryItemUsePolicy = field(default_factory=InventoryItemUsePolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class InventoryItemUseConfig:
    workflow_timeout_seconds: float = 120.0
    step_timeout_seconds: float = 15.0
    precondition_retry_limit: int = 1
    navigation_retry_limit: int = 1
    scan_retry_limit: int = 1
    action_retry_limit: int = 0
    retry_delay_seconds: float = 0.25


@dataclass(frozen=True)
class InventoryItemObservation:
    item_type: InventoryItemType | str
    item_id: str
    display_name: str
    available_quantity: int
    rarity: InventoryItemRarity | str = InventoryItemRarity.COMMON
    recognized: bool = True
    premium: bool = False
    confidence: float = 1.0
    target: tuple[int, int] | None = None
    scene_verified: bool = True
    identity_verified: bool = True
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_item_type(self) -> InventoryItemType:
        return _item_type(self.item_type)

    def normalized_rarity(self) -> InventoryItemRarity:
        return _rarity(self.rarity)

    def target_json(self) -> dict[str, int] | None:
        if self.target is None:
            return None
        return {"x": int(self.target[0]), "y": int(self.target[1])}

    def to_json(self) -> dict[str, object]:
        return {
            "item_type": self.normalized_item_type().value,
            "item_id": self.item_id,
            "display_name": self.display_name,
            "available_quantity": self.available_quantity,
            "rarity": self.normalized_rarity().value,
            "recognized": self.recognized,
            "premium": self.premium,
            "confidence": self.confidence,
            "target": self.target_json(),
            "scene_verified": self.scene_verified,
            "identity_verified": self.identity_verified,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class InventoryScan:
    observations: tuple[InventoryItemObservation, ...] = ()
    scene_verified: bool = True
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "observations": [item.to_json() for item in self.observations],
            "scene_verified": self.scene_verified,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class InventoryUseConfirmation:
    success: bool
    confirmed: bool = False
    premium_or_rare_prompt: bool = False
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "confirmed": self.confirmed,
            "premium_or_rare_prompt": self.premium_or_rare_prompt,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class InventoryDeltaVerification:
    success: bool
    verified: bool = False
    before_quantity: int | None = None
    after_quantity: int | None = None
    used_quantity: int | None = None
    message: str = ""
    retryable: bool = False
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def inventory_delta(self) -> int | None:
        if self.before_quantity is None or self.after_quantity is None:
            return None
        return self.after_quantity - self.before_quantity

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "verified": self.verified,
            "before_quantity": self.before_quantity,
            "after_quantity": self.after_quantity,
            "used_quantity": self.used_quantity,
            "inventory_delta": self.inventory_delta(),
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


class InventoryItemUseAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: InventoryItemUseRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class InventoryItemUseCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: InventoryItemUseRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class InventoryItemUseDriver(Protocol):
    def open_inventory(
        self,
        request: InventoryItemUseRequest,
        character: Character,
        policy: InventoryItemUsePolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def scan_inventory(
        self,
        request: InventoryItemUseRequest,
        character: Character,
        policy: InventoryItemUsePolicy,
    ) -> InventoryScan:
        ...

    def select_inventory_item(
        self,
        request: InventoryItemUseRequest,
        character: Character,
        item: InventoryItemObservation,
        policy: InventoryItemUsePolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def verify_inventory_item_identity(
        self,
        request: InventoryItemUseRequest,
        character: Character,
        item: InventoryItemObservation,
        policy: InventoryItemUsePolicy,
    ) -> InventoryItemObservation:
        ...

    def enter_item_quantity(
        self,
        request: InventoryItemUseRequest,
        character: Character,
        item: InventoryItemObservation,
        quantity: int,
        policy: InventoryItemUsePolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def confirm_item_use(
        self,
        request: InventoryItemUseRequest,
        character: Character,
        item: InventoryItemObservation,
        quantity: int,
        policy: InventoryItemUsePolicy,
    ) -> InventoryUseConfirmation:
        ...

    def verify_inventory_delta(
        self,
        request: InventoryItemUseRequest,
        character: Character,
        item: InventoryItemObservation,
        quantity: int,
        confirmation: InventoryUseConfirmation,
        policy: InventoryItemUsePolicy,
    ) -> InventoryDeltaVerification:
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
class _InventoryItemUseState:
    request: InventoryItemUseRequest
    character: Character | None = None
    policy: InventoryItemUsePolicy | None = None
    requested_item_type: InventoryItemType | None = None
    scan: InventoryScan | None = None
    selected_item: InventoryItemObservation | None = None
    verified_item: InventoryItemObservation | None = None
    selected_quantity: int = 0
    confirmation: InventoryUseConfirmation | None = None
    delta_verification: InventoryDeltaVerification | None = None
    ignored_items: list[dict[str, object]] = field(default_factory=list)
    scan_attempts: list[dict[str, object]] = field(default_factory=list)
    selection_attempts: list[dict[str, object]] = field(default_factory=list)
    identity_attempts: list[dict[str, object]] = field(default_factory=list)
    quantity_attempts: list[dict[str, object]] = field(default_factory=list)
    confirmation_attempts: list[dict[str, object]] = field(default_factory=list)
    delta_attempts: list[dict[str, object]] = field(default_factory=list)
    preview: dict[str, object] = field(default_factory=dict)
    budget_status: dict[str, object] = field(default_factory=dict)
    dry_run_complete: bool = False
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


class InventoryItemUseWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: InventoryItemUseDriver,
        account_precondition: InventoryItemUseAccountPrecondition | None = None,
        character_precondition: InventoryItemUseCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: InventoryItemUseConfig | None = None,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.account_precondition = account_precondition
        self.character_precondition = character_precondition
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.incidents = incidents
        self.config = config or InventoryItemUseConfig()
        self._states: dict[str, _InventoryItemUseState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return INVENTORY_ITEM_USE_STATES

    def execute(
        self,
        request: InventoryItemUseRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _InventoryItemUseState(request=request)
        self._states[token] = state
        persistence = None
        if self.job_runs is not None and self.step_runs is not None and request.job_id is not None:
            persistence = WorkflowRunRepositoryRecorder(self.job_runs, self.step_runs)
        try:
            context = WorkflowExecutionContext(
                cancellation_token=cancellation_token or CancellationToken(),
                deadline=WorkflowDeadline.from_timeout(self.config.workflow_timeout_seconds, time.monotonic),
                budget=StepBudget(max_steps=len(INVENTORY_ITEM_USE_STATES) + 12),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"inventory-item-use:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"inventory_item_use_run_id": token},
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
        for state in INVENTORY_ITEM_USE_STATES:
            registry.register(f"inventory_item_use.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.precondition_retry_limit,
            "ensure_character": self.config.precondition_retry_limit,
            "ensure_game_running": self.config.precondition_retry_limit,
            "open_inventory": self.config.navigation_retry_limit,
            "scan_inventory": self.config.scan_retry_limit,
            "select_item": self.config.action_retry_limit,
            "verify_item_identity": self.config.scan_retry_limit,
            "enter_quantity": self.config.action_retry_limit,
            "confirm_usage": self.config.action_retry_limit,
            "verify_inventory_delta": self.config.scan_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=INVENTORY_ITEM_USE_WORKFLOW_KEY,
            name="Use Inventory Item",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"inventory_item_use.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in INVENTORY_ITEM_USE_STATES
            ],
        )

    def _handler_for(self, state_name: str):
        def handler(context: WorkflowExecutionContext, step: WorkflowStepSpec) -> WorkflowStepResult:
            state = self._state_from_context(context)
            if state_name == "failed":
                return self._failed(step, state)
            if state_name == "cancelled":
                return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
            if state.dry_run_complete and state_name in {"enter_quantity", "confirm_usage", "verify_inventory_delta"}:
                return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"dry_run": True})
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
        state: _InventoryItemUseState,
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
            requested_item_type = _item_type(request.requested_item_type)
            requested_quantity = _require_positive_int(request.requested_quantity, "requested_quantity")
            policy = request.policy.normalized()
        except ValueError as exc:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, str(exc))
        if requested_item_type not in policy.whitelist:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                f"Requested item type is not whitelisted: {requested_item_type.value}.",
                data={"requested_item_type": requested_item_type.value},
            )
        state.requested_item_type = requested_item_type
        state.policy = policy
        state.selected_quantity = requested_quantity
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "requested_item_type": requested_item_type.value,
                "requested_quantity": requested_quantity,
                "policy": policy.to_json(),
            },
        )

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _InventoryItemUseState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        character = self.characters.get(state.request.character_id)
        if character is None:
            return state.stop(
                step.step_key,
                WorkflowOutcome.VALIDATION_FAILURE,
                f"Character {state.request.character_id} was not found.",
            )
        state.character = character
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"character_id": character.id})

    def _ensure_account(
        self,
        step: WorkflowStepSpec,
        state: _InventoryItemUseState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.account_precondition is None:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"skipped": True})
        return self._action_to_step(step, state, self.account_precondition.ensure_account(state.request, _require_character(state)))

    def _ensure_character(
        self,
        step: WorkflowStepSpec,
        state: _InventoryItemUseState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.character_precondition is None:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"skipped": True})
        return self._action_to_step(step, state, self.character_precondition.ensure_character(state.request, _require_character(state)))

    def _ensure_game_running(
        self,
        step: WorkflowStepSpec,
        _state: _InventoryItemUseState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"precondition": "delegated_to_driver"})

    def _open_inventory(
        self,
        step: WorkflowStepSpec,
        state: _InventoryItemUseState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        result = self.driver.open_inventory(state.request, _require_character(state), _require_policy(state))
        return self._action_to_step(step, state, result, hard_stop_message="Inventory scene could not be verified.")

    def _scan_inventory(
        self,
        step: WorkflowStepSpec,
        state: _InventoryItemUseState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        scan = self.driver.scan_inventory(state.request, _require_character(state), _require_policy(state))
        state.scan = scan
        if scan.screenshot_path:
            state.screenshot_path = scan.screenshot_path
        state.scan_attempts.append(scan.to_json())
        if not scan.scene_verified:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                scan.message or "Inventory scene could not be verified.",
                screenshot_path=scan.screenshot_path,
                data={"scan": scan.to_json()},
            )
        selected, ignored = _select_inventory_item(scan.observations, _require_requested_type(state), _require_policy(state))
        state.ignored_items = ignored
        if selected is None:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                "No whitelisted inventory item matching the requested semantic type was found.",
                screenshot_path=scan.screenshot_path,
                data={"scan": scan.to_json(), "ignored_items": ignored},
            )
        state.selected_item = selected
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"selected_item": selected.to_json(), "ignored_items": ignored},
            screenshot_path=scan.screenshot_path,
        )

    def _select_item(
        self,
        step: WorkflowStepSpec,
        state: _InventoryItemUseState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        action = self.driver.select_inventory_item(
            state.request,
            _require_character(state),
            _require_selected_item(state),
            _require_policy(state),
        )
        state.selection_attempts.append({"selected_item": _require_selected_item(state).to_json(), **action.data})
        return self._action_to_step(step, state, action, hard_stop_message="Inventory item could not be selected safely.")

    def _verify_item_identity(
        self,
        step: WorkflowStepSpec,
        state: _InventoryItemUseState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        verified = self.driver.verify_inventory_item_identity(
            state.request,
            _require_character(state),
            _require_selected_item(state),
            _require_policy(state),
        )
        state.verified_item = verified
        if verified.screenshot_path:
            state.screenshot_path = verified.screenshot_path
        state.identity_attempts.append(verified.to_json())
        reason = _item_skip_reason(verified, _require_requested_type(state), _require_policy(state))
        if reason:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                f"Selected inventory item failed safety verification: {reason}.",
                screenshot_path=verified.screenshot_path,
                data={"selected_item": verified.to_json(), "ignored_reason": reason},
            )
        if verified.available_quantity < state.selected_quantity:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                "Insufficient inventory quantity for requested item use.",
                screenshot_path=verified.screenshot_path,
                data={"selected_item": verified.to_json(), "requested_quantity": state.selected_quantity},
            )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"selected_item": verified.to_json()})

    def _check_budget(
        self,
        step: WorkflowStepSpec,
        state: _InventoryItemUseState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        budget = _require_policy(state).budget_for(_require_requested_type(state))
        if budget is None:
            return state.stop(step.step_key, WorkflowOutcome.BLOCKED, "Requested item type is not whitelisted.")
        budget = budget.normalized()
        quantity = state.selected_quantity
        state.budget_status = {
            "item_type": _require_requested_type(state).value,
            "requested_quantity": quantity,
            "allowed": True,
            "reason": "",
            **budget.to_json(),
        }
        if quantity > budget.max_per_run:
            state.budget_status.update({"allowed": False, "reason": "quantity_per_run_exceeded"})
        elif quantity > budget.remaining_daily():
            state.budget_status.update({"allowed": False, "reason": "daily_quantity_budget_exceeded"})
        elif quantity > budget.remaining_total():
            state.budget_status.update({"allowed": False, "reason": "total_budget_exceeded"})
        if not state.budget_status["allowed"]:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                f"Inventory item use budget blocked request: {state.budget_status['reason']}.",
                screenshot_path=state.screenshot_path,
                data={"budget_status": state.budget_status},
            )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"budget_status": state.budget_status})

    def _preview_item_use(
        self,
        step: WorkflowStepSpec,
        state: _InventoryItemUseState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        item = _require_verified_item(state)
        state.preview = {
            "dry_run": _require_policy(state).dry_run,
            "selected_item_metadata": item.to_json(),
            "requested_quantity": state.request.requested_quantity,
            "used_quantity": 0 if _require_policy(state).dry_run else state.selected_quantity,
            "budget_status": state.budget_status,
        }
        if _require_policy(state).dry_run:
            state.dry_run_complete = True
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"preview": state.preview})

    def _enter_quantity(
        self,
        step: WorkflowStepSpec,
        state: _InventoryItemUseState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        action = self.driver.enter_item_quantity(
            state.request,
            _require_character(state),
            _require_verified_item(state),
            state.selected_quantity,
            _require_policy(state),
        )
        state.quantity_attempts.append({"quantity": state.selected_quantity, **action.data})
        return self._action_to_step(step, state, action, hard_stop_message="Inventory item quantity could not be entered safely.")

    def _confirm_usage(
        self,
        step: WorkflowStepSpec,
        state: _InventoryItemUseState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        confirmation = self.driver.confirm_item_use(
            state.request,
            _require_character(state),
            _require_verified_item(state),
            state.selected_quantity,
            _require_policy(state),
        )
        state.confirmation = confirmation
        if confirmation.screenshot_path:
            state.screenshot_path = confirmation.screenshot_path
        state.confirmation_attempts.append(confirmation.to_json())
        if confirmation.premium_or_rare_prompt:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                "Premium or rare item confirmation cannot be handled safely.",
                screenshot_path=confirmation.screenshot_path,
                data={"confirmation": confirmation.to_json()},
            )
        if not confirmation.success or not confirmation.confirmed:
            return self._confirmation_failure(step, state, confirmation, "Inventory item use confirmation failed.")
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=confirmation.to_json(), screenshot_path=confirmation.screenshot_path)

    def _verify_inventory_delta(
        self,
        step: WorkflowStepSpec,
        state: _InventoryItemUseState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        delta = self.driver.verify_inventory_delta(
            state.request,
            _require_character(state),
            _require_verified_item(state),
            state.selected_quantity,
            _require_confirmation(state),
            _require_policy(state),
        )
        state.delta_verification = delta
        if delta.screenshot_path:
            state.screenshot_path = delta.screenshot_path
        state.delta_attempts.append(delta.to_json())
        if not _inventory_delta_verified(_require_verified_item(state), state.selected_quantity, delta):
            failure = replace(
                delta,
                retryable=False,
                message=("Inventory quantity delta did not match requested item use. " f"{delta.message}").strip(),
            )
            return self._delta_failure(step, state, failure, "Inventory delta postcondition was not verified.")
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=delta.to_json(), screenshot_path=delta.screenshot_path)

    def _complete(self, step: WorkflowStepSpec, state: _InventoryItemUseState) -> WorkflowStepResult:
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
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=self._inventory_payload(state))

    def _skipped(self, step: WorkflowStepSpec, state: _InventoryItemUseState) -> WorkflowStepResult:
        if state.terminal_outcome == WorkflowOutcome.SKIPPED:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data={"skipped_reason": state.terminal_reason, **self._inventory_payload(state)},
                screenshot_path=state.screenshot_path,
            )
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED)

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _InventoryItemUseState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if _is_manual_stop(state):
            state.recovery_outcome = {"attempted": False, "reason": "manual_intervention_required"}
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
        state.recovery_outcome = self._monitor_recovery(state, _job_run_id(context))
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _InventoryItemUseState) -> WorkflowStepResult:
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
                **self._inventory_payload(state),
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _InventoryItemUseState,
        action: ResourceGatheringActionResult,
        *,
        hard_stop_message: str = "Inventory item action failed.",
    ) -> WorkflowStepResult:
        if action.screenshot_path:
            state.screenshot_path = action.screenshot_path
        if action.success:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=action.data, screenshot_path=action.screenshot_path)
        if action.retryable:
            return _step_result(
                step.step_key,
                WorkflowOutcome.RETRYABLE_FAILURE,
                action.message or hard_stop_message,
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            action.message or hard_stop_message,
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _confirmation_failure(
        self,
        step: WorkflowStepSpec,
        state: _InventoryItemUseState,
        result: InventoryUseConfirmation,
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

    def _delta_failure(
        self,
        step: WorkflowStepSpec,
        state: _InventoryItemUseState,
        result: InventoryDeltaVerification,
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

    def _inventory_payload(self, state: _InventoryItemUseState) -> dict[str, object]:
        used_quantity = 0
        if state.delta_verification is not None and state.delta_verification.used_quantity is not None:
            used_quantity = state.delta_verification.used_quantity
        elif state.confirmation is not None and state.confirmation.confirmed:
            used_quantity = state.selected_quantity
        return {
            "preview": state.preview,
            "scan": state.scan.to_json() if state.scan is not None else {},
            "scan_attempts": state.scan_attempts,
            "selected_item_metadata": state.verified_item.to_json()
            if state.verified_item is not None
            else (state.selected_item.to_json() if state.selected_item is not None else {}),
            "requested_quantity": state.request.requested_quantity,
            "used_quantity": used_quantity,
            "inventory_delta": state.delta_verification.to_json() if state.delta_verification is not None else {},
            "budget_status": state.budget_status,
            "ignored_items": state.ignored_items,
            "selection_attempts": state.selection_attempts,
            "identity_attempts": state.identity_attempts,
            "quantity_attempts": state.quantity_attempts,
            "confirmation_attempts": state.confirmation_attempts,
            "delta_attempts": state.delta_attempts,
            "dry_run": _require_policy(state).dry_run if state.policy is not None else False,
            "failure_evidence": {
                "screenshot_path": state.screenshot_path,
                "terminal_state": state.terminal_state,
                "terminal_reason": state.terminal_reason,
            },
        }

    def _state_from_context(self, context: WorkflowExecutionContext) -> _InventoryItemUseState:
        token = str(context.metadata.get("inventory_item_use_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("Inventory item use runtime state is missing.") from exc

    def _open_incident(self, state: _InventoryItemUseState) -> None:
        if self.incidents is None or not state.failed or state.incident_opened:
            return
        self.incidents.save(
            Incident(
                incident_key=f"inventory-item-use:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="Inventory item use workflow blocked",
                details=state.terminal_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )
        state.incident_opened = True

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _InventoryItemUseState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "Inventory item use workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(
        self,
        result: WorkflowExecutionResult,
        state: _InventoryItemUseState,
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
        state: _InventoryItemUseState,
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
        state: _InventoryItemUseState,
    ) -> None:
        if state.terminal_outcome in {WorkflowOutcome.SKIPPED, WorkflowOutcome.BLOCKED}:
            result.outcome = state.terminal_outcome
            result.message = state.terminal_reason
        result.result = {
            **dict(result.result),
            "policy": state.policy.to_json() if state.policy is not None else {},
            **self._inventory_payload(state),
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
        state: _InventoryItemUseState,
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


def _select_inventory_item(
    observations: tuple[InventoryItemObservation, ...],
    requested_type: InventoryItemType,
    policy: InventoryItemUsePolicy,
) -> tuple[InventoryItemObservation | None, list[dict[str, object]]]:
    ignored: list[dict[str, object]] = []
    for item in observations:
        reason = _item_skip_reason(item, requested_type, policy)
        if reason:
            ignored.append({**item.to_json(), "ignored_reason": reason})
            continue
        return item, ignored
    return None, ignored


def _item_skip_reason(
    item: InventoryItemObservation,
    requested_type: InventoryItemType,
    policy: InventoryItemUsePolicy,
) -> str:
    item_type = item.normalized_item_type()
    rarity = item.normalized_rarity()
    if not item.recognized or item_type == InventoryItemType.UNKNOWN:
        return "unrecognized_item"
    if item_type != requested_type:
        return "semantic_type_mismatch"
    if item_type not in policy.whitelist:
        return "item_type_not_whitelisted"
    if item.premium or rarity == InventoryItemRarity.PREMIUM:
        return "" if policy.allow_premium_items else "premium_item_blocked"
    if rarity in {InventoryItemRarity.RARE, InventoryItemRarity.EPIC, InventoryItemRarity.LEGENDARY}:
        return "" if policy.allow_rare_items else "rare_item_blocked"
    if not item.scene_verified or not item.identity_verified:
        return "identity_not_verified"
    if item.confidence < policy.minimum_detector_confidence:
        return "below_confidence_threshold"
    if item.available_quantity <= 0:
        return "insufficient_quantity"
    return ""


def _inventory_delta_verified(
    before_item: InventoryItemObservation,
    quantity: int,
    delta: InventoryDeltaVerification,
) -> bool:
    if not delta.success or not delta.verified:
        return False
    if delta.used_quantity is not None and delta.used_quantity != quantity:
        return False
    if delta.before_quantity is not None and delta.before_quantity != before_item.available_quantity:
        return False
    if delta.after_quantity is not None and delta.before_quantity is not None:
        return delta.before_quantity - delta.after_quantity == quantity
    return True


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
        action_type=f"inventory_item_use.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _InventoryItemUseState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _InventoryItemUseState) -> InventoryItemUsePolicy:
    if state.policy is None:
        raise RuntimeError("Inventory item use policy has not been validated.")
    return state.policy


def _require_requested_type(state: _InventoryItemUseState) -> InventoryItemType:
    if state.requested_item_type is None:
        raise RuntimeError("Requested inventory item type has not been validated.")
    return state.requested_item_type


def _require_selected_item(state: _InventoryItemUseState) -> InventoryItemObservation:
    if state.selected_item is None:
        raise RuntimeError("Inventory item has not been selected.")
    return state.selected_item


def _require_verified_item(state: _InventoryItemUseState) -> InventoryItemObservation:
    if state.verified_item is None:
        raise RuntimeError("Inventory item identity has not been verified.")
    return state.verified_item


def _require_confirmation(state: _InventoryItemUseState) -> InventoryUseConfirmation:
    if state.confirmation is None:
        raise RuntimeError("Inventory item usage has not been confirmed.")
    return state.confirmation


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_manual_stop(state: _InventoryItemUseState) -> bool:
    text = state.terminal_reason.lower()
    return "verification" in text or "confirmation" in text or "premium" in text or "rare" in text or "manual" in text
