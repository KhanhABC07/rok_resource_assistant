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


VIP_CLAIM_WORKFLOW_KEY = "vip-claim"
VIP_CLAIM_TEMPLATE_KEYS = (
    "city.vip.button",
    "vip.scene",
    "vip.daily_reward.free",
    "vip.vip_chest.free",
    "vip.target.cooldown",
    "vip.target.key_required",
    "vip.target.gem_required",
    "vip.reward.close",
)
VIP_CLAIM_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "ensure_game_running",
    "open_vip_ui",
    "scan_vip_rewards",
    "claim_vip_reward",
    "claim_vip_chest",
    "close_reward_overlay",
    "verify_vip_state",
    "complete",
    "skipped",
    "recover",
    "failed",
    "cancelled",
)


class VipClaimType(StrEnum):
    DAILY_REWARD = "DAILY_REWARD"
    VIP_CHEST = "VIP_CHEST"


class VipClaimStatus(StrEnum):
    FREE = "FREE"
    CLAIMED = "CLAIMED"
    COOLDOWN = "COOLDOWN"
    UNAVAILABLE = "UNAVAILABLE"
    PAID = "PAID"
    KEY_REQUIRED = "KEY_REQUIRED"
    GEM_REQUIRED = "GEM_REQUIRED"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


class VipClaimConfirmation(StrEnum):
    NONE = "NONE"
    FREE = "FREE"
    PAID = "PAID"
    KEY = "KEY"
    GEM = "GEM"
    UNKNOWN = "UNKNOWN"


def _target_type(value: VipClaimType | str) -> VipClaimType:
    if isinstance(value, VipClaimType):
        return value
    try:
        return VipClaimType(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in VipClaimType)
        raise ValueError(f"Invalid VIP target type: {value!r}. Expected one of: {valid}.") from exc


def _target_status(value: VipClaimStatus | str) -> VipClaimStatus:
    if isinstance(value, VipClaimStatus):
        return value
    try:
        return VipClaimStatus(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in VipClaimStatus)
        raise ValueError(f"Invalid VIP target status: {value!r}. Expected one of: {valid}.") from exc


def _confirmation(value: VipClaimConfirmation | str) -> VipClaimConfirmation:
    if isinstance(value, VipClaimConfirmation):
        return value
    try:
        return VipClaimConfirmation(str(value).strip().upper())
    except ValueError as exc:
        valid = ", ".join(item.value for item in VipClaimConfirmation)
        raise ValueError(f"Invalid VIP target confirmation: {value!r}. Expected one of: {valid}.") from exc


def _require_confidence(value: float, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    return numeric


@dataclass(frozen=True)
class VipClaimPolicy:
    allow_daily_reward: bool = True
    allow_vip_chest: bool = True
    allow_gem_spending: bool = False
    block_when_only_paid_options: bool = False
    minimum_detector_confidence: float = 0.85

    def normalized(self) -> VipClaimPolicy:
        if self.allow_gem_spending:
            raise ValueError("Gem spending is not supported by VIP-001.")
        return VipClaimPolicy(
            allow_daily_reward=bool(self.allow_daily_reward),
            allow_vip_chest=bool(self.allow_vip_chest),
            allow_gem_spending=False,
            block_when_only_paid_options=bool(self.block_when_only_paid_options),
            minimum_detector_confidence=_require_confidence(
                self.minimum_detector_confidence,
                "minimum_detector_confidence",
            ),
        )

    def allows(self, target_type: VipClaimType) -> bool:
        if target_type == VipClaimType.DAILY_REWARD:
            return self.allow_daily_reward
        if target_type == VipClaimType.VIP_CHEST:
            return self.allow_vip_chest
        return False

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "allow_daily_reward": normalized.allow_daily_reward,
            "allow_vip_chest": normalized.allow_vip_chest,
            "allow_gem_spending": normalized.allow_gem_spending,
            "block_when_only_paid_options": normalized.block_when_only_paid_options,
            "minimum_detector_confidence": normalized.minimum_detector_confidence,
        }


@dataclass(frozen=True)
class VipClaimRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: VipClaimPolicy = field(default_factory=VipClaimPolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class VipClaimConfig:
    workflow_timeout_seconds: float = 120.0
    step_timeout_seconds: float = 15.0
    precondition_retry_limit: int = 1
    navigation_retry_limit: int = 1
    scan_retry_limit: int = 1
    action_retry_limit: int = 0
    retry_delay_seconds: float = 0.25


@dataclass(frozen=True)
class VipClaimObservation:
    target_type: VipClaimType | str
    status: VipClaimStatus | str
    confidence: float = 1.0
    target: tuple[int, int] | None = None
    scene_verified: bool = True
    free_indicator_visible: bool = False
    cooldown_seconds: int | None = None
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_target_type(self) -> VipClaimType:
        return _target_type(self.target_type)

    def normalized_status(self) -> VipClaimStatus:
        return _target_status(self.status)

    def target_json(self) -> dict[str, int] | None:
        if self.target is None:
            return None
        return {"x": int(self.target[0]), "y": int(self.target[1])}

    def to_json(self) -> dict[str, object]:
        return {
            "target_type": self.normalized_target_type().value,
            "status": self.normalized_status().value,
            "confidence": self.confidence,
            "target": self.target_json(),
            "scene_verified": self.scene_verified,
            "free_indicator_visible": self.free_indicator_visible,
            "cooldown_seconds": self.cooldown_seconds,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class VipClaimScan:
    observations: tuple[VipClaimObservation, ...] = ()
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
class VipClaimOpenResult:
    success: bool
    changed: bool = False
    reward_ui_present: bool = True
    confirmation: VipClaimConfirmation | str = VipClaimConfirmation.NONE
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def normalized_confirmation(self) -> VipClaimConfirmation:
        return _confirmation(self.confirmation)

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "changed": self.changed,
            "reward_ui_present": self.reward_ui_present,
            "confirmation": self.normalized_confirmation().value,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class VipRewardCloseResult:
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


class VipClaimAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: VipClaimRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class VipClaimCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: VipClaimRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class VipClaimDriver(Protocol):
    def open_vip_ui(
        self,
        request: VipClaimRequest,
        character: Character,
        policy: VipClaimPolicy,
    ) -> ResourceGatheringActionResult:
        ...

    def scan_vip_rewards(
        self,
        request: VipClaimRequest,
        character: Character,
        policy: VipClaimPolicy,
    ) -> VipClaimScan:
        ...

    def claim_vip_reward(
        self,
        request: VipClaimRequest,
        character: Character,
        target: VipClaimObservation,
        policy: VipClaimPolicy,
    ) -> VipClaimOpenResult:
        ...

    def claim_vip_chest(
        self,
        request: VipClaimRequest,
        character: Character,
        target: VipClaimObservation,
        policy: VipClaimPolicy,
    ) -> VipClaimOpenResult:
        ...

    def close_reward_overlay(
        self,
        request: VipClaimRequest,
        character: Character,
        target: VipClaimObservation,
        open_result: VipClaimOpenResult,
        policy: VipClaimPolicy,
    ) -> VipRewardCloseResult:
        ...

    def verify_vip_state(
        self,
        request: VipClaimRequest,
        character: Character,
        target: VipClaimObservation,
        open_result: VipClaimOpenResult,
        policy: VipClaimPolicy,
    ) -> VipClaimObservation:
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
class _VipClaimState:
    request: VipClaimRequest
    character: Character | None = None
    policy: VipClaimPolicy | None = None
    scan: VipClaimScan | None = None
    selected_targets: list[VipClaimObservation] = field(default_factory=list)
    opened_targets: list[tuple[VipClaimObservation, VipClaimOpenResult]] = field(default_factory=list)
    scan_attempts: list[dict[str, object]] = field(default_factory=list)
    ignored_targets: list[dict[str, object]] = field(default_factory=list)
    open_attempts: list[dict[str, object]] = field(default_factory=list)
    reward_close_attempts: list[dict[str, object]] = field(default_factory=list)
    verification_attempts: list[dict[str, object]] = field(default_factory=list)
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


class VipClaimWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: VipClaimDriver,
        account_precondition: VipClaimAccountPrecondition | None = None,
        character_precondition: VipClaimCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: VipClaimConfig | None = None,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.account_precondition = account_precondition
        self.character_precondition = character_precondition
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.incidents = incidents
        self.config = config or VipClaimConfig()
        self._states: dict[str, _VipClaimState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return VIP_CLAIM_STATES

    def execute(
        self,
        request: VipClaimRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _VipClaimState(request=request)
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
                budget=StepBudget(max_steps=len(VIP_CLAIM_STATES) + 12),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"vip-claim:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"vip_claim_run_id": token},
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
        for state in VIP_CLAIM_STATES:
            registry.register(f"vip_claim.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.precondition_retry_limit,
            "ensure_character": self.config.precondition_retry_limit,
            "ensure_game_running": self.config.precondition_retry_limit,
            "open_vip_ui": self.config.navigation_retry_limit,
            "scan_vip_rewards": self.config.scan_retry_limit,
            "claim_vip_reward": self.config.action_retry_limit,
            "claim_vip_chest": self.config.action_retry_limit,
            "close_reward_overlay": self.config.action_retry_limit,
            "verify_vip_state": self.config.scan_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=VIP_CLAIM_WORKFLOW_KEY,
            name="Claim Free VIP Rewards",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"vip_claim.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in VIP_CLAIM_STATES
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
        state: _VipClaimState,
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
                "template_keys": list(VIP_CLAIM_TEMPLATE_KEYS),
            },
        )

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _VipClaimState,
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
        state: _VipClaimState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.account_precondition is None:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"account_precondition": "not_configured"})
        return self._action_to_step(step, state, self.account_precondition.ensure_account(state.request, _require_character(state)))

    def _ensure_character(
        self,
        step: WorkflowStepSpec,
        state: _VipClaimState,
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
        state: _VipClaimState,
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

    def _open_vip_ui(
        self,
        step: WorkflowStepSpec,
        state: _VipClaimState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            self.driver.open_vip_ui(state.request, _require_character(state), _require_policy(state)),
            hard_stop_message="VIP scene could not be verified before claiming free rewards.",
        )

    def _scan_vip_rewards(
        self,
        step: WorkflowStepSpec,
        state: _VipClaimState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        scan = self.driver.scan_vip_rewards(state.request, _require_character(state), _require_policy(state))
        state.scan = scan
        if scan.screenshot_path:
            state.screenshot_path = scan.screenshot_path
        state.scan_attempts.append(scan.to_json())
        if not scan.scene_verified:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                scan.message or "VIP scene could not be verified.",
                screenshot_path=scan.screenshot_path,
                data={"scan": scan.to_json()},
            )
        for observation in scan.observations:
            if observation.normalized_status() == VipClaimStatus.VERIFICATION_REQUIRED:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.FATAL_FAILURE,
                    observation.message or "Verification screen requires manual intervention.",
                    screenshot_path=observation.screenshot_path or scan.screenshot_path,
                    data={"scan": scan.to_json(), "target": observation.to_json()},
                )
            if observation.normalized_status() == VipClaimStatus.UNKNOWN:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    observation.message or "VIP reward state could not be determined safely.",
                    screenshot_path=observation.screenshot_path or scan.screenshot_path,
                    data={"scan": scan.to_json(), "target": observation.to_json()},
                )
        selected, ignored = _select_free_targets(_require_scan(state).observations, _require_policy(state))
        state.selected_targets = selected
        state.ignored_targets = ignored
        if not selected:
            paid_only = bool(ignored) and all(
                item.get("ignored_reason") in {
                    "paid_not_allowed",
                    "gem_spending_not_allowed",
                }
                for item in ignored
            )
            outcome = (
                WorkflowOutcome.BLOCKED
                if paid_only and _require_policy(state).block_when_only_paid_options
                else WorkflowOutcome.SKIPPED
            )
            reason = "Only paid or gem VIP reward options are present." if paid_only else "No free VIP reward or chest is available."
            return state.stop(
                step.step_key,
                outcome,
                reason,
                screenshot_path=state.screenshot_path,
                data={"ignored_targets": ignored, "scan": _require_scan(state).to_json()},
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "selected_targets": [item.to_json() for item in selected],
                "ignored_targets": ignored,
            },
            screenshot_path=state.screenshot_path,
        )

    def _claim_vip_reward(
        self,
        step: WorkflowStepSpec,
        state: _VipClaimState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._claim_selected_targets(step, state, context, VipClaimType.DAILY_REWARD)

    def _claim_vip_chest(
        self,
        step: WorkflowStepSpec,
        state: _VipClaimState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        return self._claim_selected_targets(step, state, context, VipClaimType.VIP_CHEST)

    def _claim_selected_targets(
        self,
        step: WorkflowStepSpec,
        state: _VipClaimState,
        context: WorkflowExecutionContext,
        target_type: VipClaimType,
    ) -> WorkflowStepResult:
        matching_targets = [target for target in state.selected_targets if target.normalized_target_type() == target_type]
        if not matching_targets:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"target_type": target_type.value})
        for target in matching_targets:
            context.cancellation_token.throw_if_cancelled()
            if target_type == VipClaimType.DAILY_REWARD:
                result = self.driver.claim_vip_reward(state.request, _require_character(state), target, _require_policy(state))
            else:
                result = self.driver.claim_vip_chest(state.request, _require_character(state), target, _require_policy(state))
            if result.screenshot_path:
                state.screenshot_path = result.screenshot_path
            attempt = {"target": target.to_json(), **result.to_json()}
            state.open_attempts.append(attempt)
            confirmation = result.normalized_confirmation()
            if confirmation in {VipClaimConfirmation.PAID, VipClaimConfirmation.KEY, VipClaimConfirmation.GEM, VipClaimConfirmation.UNKNOWN}:
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    f"Unsafe VIP reward confirmation cannot be handled safely: {confirmation.value}.",
                    screenshot_path=result.screenshot_path,
                    data={"attempt": attempt},
                )
            if not result.success:
                return self._open_failure(step, state, result, "Free VIP reward could not be claimed.")
            state.opened_targets.append((target, result))
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"open_attempts": state.open_attempts},
            screenshot_path=state.screenshot_path,
        )

    def _close_reward_overlay(
        self,
        step: WorkflowStepSpec,
        state: _VipClaimState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        for target, open_result in state.opened_targets:
            context.cancellation_token.throw_if_cancelled()
            if not open_result.reward_ui_present:
                state.reward_close_attempts.append(
                    {"target": target.to_json(), "skipped": True, "reason": "reward_ui_not_present"}
                )
                continue
            close = self.driver.close_reward_overlay(
                state.request,
                _require_character(state),
                target,
                open_result,
                _require_policy(state),
            )
            if close.screenshot_path:
                state.screenshot_path = close.screenshot_path
            attempt = {"target": target.to_json(), **close.to_json()}
            state.reward_close_attempts.append(attempt)
            if not close.success or not close.closed:
                return self._reward_close_failure(step, state, close, "VIP reward overlay could not be closed safely.")
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"reward_ui_handling": state.reward_close_attempts},
            screenshot_path=state.screenshot_path,
        )

    def _verify_vip_state(
        self,
        step: WorkflowStepSpec,
        state: _VipClaimState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        for target, open_result in state.opened_targets:
            context.cancellation_token.throw_if_cancelled()
            verification = self.driver.verify_vip_state(
                state.request,
                _require_character(state),
                target,
                open_result,
                _require_policy(state),
            )
            if verification.screenshot_path:
                state.screenshot_path = verification.screenshot_path
            state.verification_attempts.append({"before": target.to_json(), "after": verification.to_json()})
            if not _target_postcondition_verified(target, verification):
                failure = replace(
                    verification,
                    status=VipClaimStatus.UNKNOWN,
                    message=("VIP free indicator did not change to claimed, cooldown, or unavailable. " f"{verification.message}").strip(),
                )
                return state.stop(
                    step.step_key,
                    WorkflowOutcome.BLOCKED,
                    failure.message,
                    screenshot_path=failure.screenshot_path,
                    data={"verification": failure.to_json()},
                )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"verification_attempts": state.verification_attempts},
            screenshot_path=state.screenshot_path,
        )

    def _complete(self, step: WorkflowStepSpec, state: _VipClaimState) -> WorkflowStepResult:
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
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=self._vip_payload(state))

    def _skipped(self, step: WorkflowStepSpec, state: _VipClaimState) -> WorkflowStepResult:
        if state.terminal_outcome == WorkflowOutcome.SKIPPED:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data={"skipped_reason": state.terminal_reason, **self._vip_payload(state)},
                screenshot_path=state.screenshot_path,
            )
        return _step_result(step.step_key, WorkflowOutcome.SKIPPED)

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _VipClaimState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if _is_manual_stop(state):
            state.recovery_outcome = {"attempted": False, "reason": "manual_intervention_required"}
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
        state.recovery_outcome = self._monitor_recovery(state, _job_run_id(context))
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _VipClaimState) -> WorkflowStepResult:
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
                **self._vip_payload(state),
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _VipClaimState,
        action: ResourceGatheringActionResult,
        *,
        hard_stop_message: str = "Vip target action failed.",
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

    def _open_failure(
        self,
        step: WorkflowStepSpec,
        state: _VipClaimState,
        result: VipClaimOpenResult,
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
        state: _VipClaimState,
        result: VipRewardCloseResult,
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

    def _vip_payload(self, state: _VipClaimState) -> dict[str, object]:
        return {
            "scan": state.scan.to_json() if state.scan is not None else {},
            "scanned_vip_states": state.scan_attempts,
            "scanned_target_states": state.scan_attempts,
            "selected_targets": [item.to_json() for item in state.selected_targets],
            "selected_target_types": [item.normalized_target_type().value for item in state.selected_targets],
            "ignored_targets": state.ignored_targets,
            "claim_attempts": state.open_attempts,
            "open_attempts": state.open_attempts,
            "reward_overlay_handling": state.reward_close_attempts,
            "reward_ui_handling": state.reward_close_attempts,
            "verification_attempts": state.verification_attempts,
            "verification_result": state.verification_attempts[-1] if state.verification_attempts else {},
            "failure_evidence": {
                "screenshot_path": state.screenshot_path,
                "terminal_state": state.terminal_state,
                "terminal_reason": state.terminal_reason,
            },
        }

    def _state_from_context(self, context: WorkflowExecutionContext) -> _VipClaimState:
        token = str(context.metadata.get("vip_claim_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("VIP claim runtime state is missing.") from exc

    def _open_incident(self, state: _VipClaimState) -> None:
        if self.incidents is None or not state.failed or state.incident_opened:
            return
        self.incidents.save(
            Incident(
                incident_key=f"vip-claim:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="VIP claim workflow blocked",
                details=state.terminal_reason,
                job_run_id=None,
                screenshot_path=state.screenshot_path,
            )
        )
        state.incident_opened = True

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _VipClaimState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "VIP claim workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else ""

    def _record_recovery_for_terminal(
        self,
        result: WorkflowExecutionResult,
        state: _VipClaimState,
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
        state: _VipClaimState,
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
        state: _VipClaimState,
    ) -> None:
        if state.terminal_outcome in {WorkflowOutcome.SKIPPED, WorkflowOutcome.BLOCKED}:
            result.outcome = state.terminal_outcome
            result.message = state.terminal_reason
        result.result = {
            **dict(result.result),
            "policy": state.policy.to_json() if state.policy is not None else {},
            **self._vip_payload(state),
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
        state: _VipClaimState,
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


def _select_free_targets(
    observations: tuple[VipClaimObservation, ...],
    policy: VipClaimPolicy,
) -> tuple[list[VipClaimObservation], list[dict[str, object]]]:
    selected: list[VipClaimObservation] = []
    ignored: list[dict[str, object]] = []
    by_type = {item.normalized_target_type(): item for item in observations}
    for target_type in (VipClaimType.DAILY_REWARD, VipClaimType.VIP_CHEST):
        observation = by_type.get(target_type)
        if observation is None:
            continue
        reason = _target_skip_reason(observation, policy)
        if reason:
            ignored.append({**observation.to_json(), "ignored_reason": reason})
            continue
        selected.append(observation)
    return selected, ignored


def _target_skip_reason(observation: VipClaimObservation, policy: VipClaimPolicy) -> str:
    status = observation.normalized_status()
    if not policy.allows(observation.normalized_target_type()):
        return "target_type_not_allowed_by_policy"
    if not observation.scene_verified:
        return "scene_not_verified"
    if observation.confidence < policy.minimum_detector_confidence:
        return "below_confidence_threshold"
    if status == VipClaimStatus.FREE and observation.free_indicator_visible:
        return ""
    if status == VipClaimStatus.KEY_REQUIRED:
        return "key_spending_not_allowed"
    if status == VipClaimStatus.GEM_REQUIRED:
        return "gem_spending_not_allowed"
    if status == VipClaimStatus.PAID:
        return "paid_not_allowed"
    return "not_free"


def _target_postcondition_verified(
    before: VipClaimObservation,
    verification: VipClaimObservation,
) -> bool:
    if before.normalized_target_type() != verification.normalized_target_type():
        return False
    if not verification.scene_verified:
        return False
    status = verification.normalized_status()
    if status in {VipClaimStatus.CLAIMED, VipClaimStatus.COOLDOWN, VipClaimStatus.UNAVAILABLE}:
        return True
    return status != VipClaimStatus.FREE and not verification.free_indicator_visible


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
        action_type=f"vip_claim.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _VipClaimState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _VipClaimState) -> VipClaimPolicy:
    if state.policy is None:
        raise RuntimeError("VIP claim policy has not been validated.")
    return state.policy


def _require_scan(state: _VipClaimState) -> VipClaimScan:
    if state.scan is None:
        raise RuntimeError("VIP rewards have not been scanned.")
    return state.scan


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_manual_stop(state: _VipClaimState) -> bool:
    text = state.terminal_reason.lower()
    return "verification" in text or "confirmation" in text or "manual" in text
