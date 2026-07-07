from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from rok_assistant.db.models import (
    AuditLog,
    Character,
    GameAccount,
    Incident,
    InstanceSession,
    JobRun,
    utc_now_iso,
)
from rok_assistant.tasks.account_switch_workflow import AccountVerification
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


CHARACTER_SWITCH_WORKFLOW_KEY = "character-switch"
CHARACTER_SWITCH_STATES = (
    "validate_input",
    "load_character",
    "ensure_game_running",
    "verify_account",
    "verify_current_character",
    "open_character_management",
    "find_character",
    "select_character",
    "confirm_switch",
    "wait_for_reload",
    "verify_character",
    "complete",
    "recover",
    "failed",
    "cancelled",
)


@dataclass(frozen=True)
class CharacterSwitchRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    target_character_id: int | None = None
    target_character_name: str = ""
    target_account_id: int | None = None
    session_key: str = ""
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class CharacterSwitchConfig:
    max_characters_per_account: int = 12
    max_character_pages: int = 4
    workflow_timeout_seconds: float = 180.0
    step_timeout_seconds: float = 20.0
    navigation_retry_limit: int = 1
    retry_delay_seconds: float = 0.25


@dataclass(frozen=True)
class CharacterSwitchActionResult:
    success: bool
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CharacterSlotObservation:
    name: str
    character_slot: int | None = None
    display_fingerprint: str = ""
    kingdom_id: int | None = None
    page_index: int = 0
    slot_index: int = 0
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CharacterPageScan:
    success: bool
    observations: tuple[CharacterSlotObservation, ...] = ()
    has_next_page: bool = False
    message: str = ""
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CharacterVerification:
    character_id: int | None = None
    character_name: str = ""
    character_slot: int | None = None
    display_fingerprint: str = ""
    kingdom_id: int | None = None
    matched: bool = False
    verification_required: bool = False
    captcha_detected: bool = False
    screenshot_path: str = ""
    message: str = ""
    data: dict[str, object] = field(default_factory=dict)


class CharacterRepository(Protocol):
    def get(self, character_id: int) -> Character | None:
        ...

    def list_all(self, include_disabled: bool = True) -> list[Character]:
        ...

    def save(self, character: Character) -> int:
        ...

    def mark_switched(self, character_id: int) -> None:
        ...


class GameAccountRepository(Protocol):
    def get(self, account_id: int) -> GameAccount | None:
        ...


class InstanceSessionRepository(Protocol):
    def save(self, session: InstanceSession) -> int:
        ...

    def get_by_key(self, session_key: str) -> InstanceSession | None:
        ...

    def list_for_instance(self, instance_id: int) -> list[InstanceSession]:
        ...


class AuditLogRepository(Protocol):
    def append(self, log: AuditLog) -> int:
        ...


class IncidentRepository(Protocol):
    def save(self, incident: Incident) -> int:
        ...


class JobRunRepository(Protocol):
    def get(self, run_id: int) -> JobRun | None:
        ...

    def save(self, run: JobRun) -> int:
        ...


class StepRunRepository(Protocol):
    ...


class CharacterSwitchDriver(Protocol):
    def verify_account(
        self,
        request: CharacterSwitchRequest,
        account: GameAccount,
    ) -> AccountVerification:
        ...

    def verify_character(
        self,
        request: CharacterSwitchRequest,
        character: Character,
    ) -> CharacterVerification:
        ...

    def open_character_management(
        self,
        request: CharacterSwitchRequest,
        character: Character,
    ) -> CharacterSwitchActionResult:
        ...

    def scan_character_page(
        self,
        request: CharacterSwitchRequest,
        character: Character,
        page_index: int,
    ) -> CharacterPageScan:
        ...

    def go_to_next_character_page(
        self,
        request: CharacterSwitchRequest,
        character: Character,
        page_index: int,
    ) -> CharacterSwitchActionResult:
        ...

    def select_character(
        self,
        request: CharacterSwitchRequest,
        character: Character,
        observation: CharacterSlotObservation,
    ) -> CharacterSwitchActionResult:
        ...

    def confirm_switch(
        self,
        request: CharacterSwitchRequest,
        character: Character,
    ) -> CharacterSwitchActionResult:
        ...

    def wait_for_reload(
        self,
        request: CharacterSwitchRequest,
        character: Character,
    ) -> CharacterSwitchActionResult:
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
class _SwitchState:
    request: CharacterSwitchRequest
    target_character: Character | None = None
    target_account: GameAccount | None = None
    session: InstanceSession | None = None
    before_account_id: int | None = None
    before_character_id: int | None = None
    selected_character_id: int | None = None
    selected_character_name: str = ""
    selected_observation: CharacterSlotObservation | None = None
    verification: CharacterVerification | None = None
    failure_reason: str = ""
    failure_state: str = ""
    recovery_outcome: dict[str, object] = field(default_factory=dict)
    screenshot_path: str = ""
    already_active: bool = False

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
        return _step_result(
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


class CharacterSwitchWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        accounts: GameAccountRepository,
        sessions: InstanceSessionRepository,
        driver: CharacterSwitchDriver,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        audit_logs: AuditLogRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: CharacterSwitchConfig | None = None,
    ) -> None:
        self.characters = characters
        self.accounts = accounts
        self.sessions = sessions
        self.driver = driver
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.audit_logs = audit_logs
        self.incidents = incidents
        self.config = config or CharacterSwitchConfig()
        self._states: dict[str, _SwitchState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return CHARACTER_SWITCH_STATES

    def execute(
        self,
        request: CharacterSwitchRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _SwitchState(request=request)
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
                budget=StepBudget(max_steps=len(CHARACTER_SWITCH_STATES) + self.config.max_character_pages + 4),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"character-switch:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"character_switch_run_id": token},
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
        for state in CHARACTER_SWITCH_STATES:
            registry.register(f"character_switch.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retryable = {
            "ensure_game_running",
            "verify_account",
            "verify_current_character",
            "open_character_management",
            "find_character",
            "select_character",
            "confirm_switch",
            "wait_for_reload",
            "verify_character",
        }
        return WorkflowDefinitionSpec(
            workflow_key=CHARACTER_SWITCH_WORKFLOW_KEY,
            name="Switch Character",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"character_switch.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=(
                        self.config.navigation_retry_limit if state in retryable else 0
                    ),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in CHARACTER_SWITCH_STATES
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
            if state.failed and state_name not in {"recover", "complete"}:
                return _step_result(
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
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        request = state.request
        if request.instance_id <= 0:
            return state.fail(step.step_key, "instance_id must be positive.")
        if request.instance_index < 0:
            return state.fail(step.step_key, "instance_index must be zero or greater.")
        if request.target_character_id is None and not request.target_character_name.strip():
            return state.fail(step.step_key, "target character id or name is required.")
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS)

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        request = state.request
        character = (
            self.characters.get(request.target_character_id)
            if request.target_character_id is not None
            else self._character_by_name(request.target_character_name, request.target_account_id)
        )
        if character is None:
            return state.fail(step.step_key, "Target character was not found.")
        if not character.enabled:
            return state.fail(step.step_key, "Target character is disabled.")
        if character.instance_id != request.instance_id:
            return state.fail(step.step_key, "Target character belongs to a different instance.")
        if character.game_account_id is None:
            return state.fail(step.step_key, "Target character is not linked to a game account.")
        account = self.accounts.get(character.game_account_id)
        if account is None:
            return state.fail(step.step_key, "Target character account was not found.")
        if not account.enabled:
            return state.fail(step.step_key, "Target character account is disabled.")
        account_characters = [
            item
            for item in self.characters.list_all(include_disabled=False)
            if item.game_account_id == account.id
        ]
        if len(account_characters) > self.config.max_characters_per_account:
            return state.fail(
                step.step_key,
                "More than twelve enabled characters are configured for the account.",
                data={"enabled_character_count": len(account_characters)},
            )
        session = self._current_session(request)
        state.target_character = character
        state.target_account = account
        state.session = session
        metadata = _json_object(session.metadata_json if session else "{}")
        state.before_account_id = _int_or_none(metadata.get("current_account_id"))
        state.before_character_id = _int_or_none(metadata.get("current_character_id"))
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "target_character_id": character.id,
                "target_character_name": character.name,
                "target_account_id": account.id,
                "before_account_id": state.before_account_id,
                "before_character_id": state.before_character_id,
            },
        )

    def _ensure_game_running(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
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
            return state.fail(
                step.step_key,
                message,
                screenshot_path=screenshot_path,
                data={"watchdog_healthy": False},
            )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data={"watchdog_healthy": True})

    def _verify_account(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        account = _require_account(state)
        verification = self.driver.verify_account(state.request, account)
        if verification.screenshot_path:
            state.screenshot_path = verification.screenshot_path
        if verification.captcha_detected or verification.verification_required:
            self._open_incident(state, "Character switch stopped for manual account verification.", verification.screenshot_path)
            return state.fail(
                step.step_key,
                "Manual verification or CAPTCHA screen detected.",
                screenshot_path=verification.screenshot_path,
                data={"manual_verification_required": True},
            )
        matched_id = verification.account_id == account.id if verification.account_id is not None else False
        matched_name = (
            verification.account_name.strip().casefold() == account.account_name.strip().casefold()
            if verification.account_name.strip()
            else False
        )
        if not verification.matched and not matched_id and not matched_name:
            self._open_incident(state, "Character switch account precondition mismatch.", verification.screenshot_path)
            return state.fail(
                step.step_key,
                verification.message or "Expected account was not active before character switching.",
                screenshot_path=verification.screenshot_path,
                data={
                    "expected_account_id": account.id,
                    "observed_account_id": verification.account_id,
                    "observed_account_name": verification.account_name,
                },
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"verified_account_id": account.id, "account_fingerprint": verification.fingerprint},
            screenshot_path=verification.screenshot_path,
        )

    def _verify_current_character(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        character = _require_character(state)
        verification = self.driver.verify_character(state.request, character)
        if verification.screenshot_path:
            state.screenshot_path = verification.screenshot_path
        if verification.captcha_detected or verification.verification_required:
            self._open_incident(state, "Character switch stopped for manual character verification.", verification.screenshot_path)
            return state.fail(
                step.step_key,
                "Manual verification or CAPTCHA screen detected.",
                screenshot_path=verification.screenshot_path,
                data={"manual_verification_required": True},
            )
        if _verification_matches(character, verification):
            state.already_active = True
            state.verification = verification
            state.selected_character_id = character.id
            state.selected_character_name = character.name
            self._persist_verified_session(state)
            self._audit_verified_switch(state)
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data={"already_active": True, **_verification_data(verification)},
                screenshot_path=verification.screenshot_path,
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"already_active": False, **_verification_data(verification)},
            screenshot_path=verification.screenshot_path,
        )

    def _open_character_management(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if state.already_active:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"already_active": True})
        return self._action_to_step(
            step,
            state,
            self.driver.open_character_management(state.request, _require_character(state)),
        )

    def _find_character(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if state.already_active:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"already_active": True})
        character = _require_character(state)
        candidates: list[CharacterSlotObservation] = []
        pages_scanned = 0
        for page_index in range(self.config.max_character_pages):
            scan = self.driver.scan_character_page(state.request, character, page_index)
            pages_scanned += 1
            if scan.screenshot_path:
                state.screenshot_path = scan.screenshot_path
            if not scan.success:
                return _step_result(
                    step.step_key,
                    WorkflowOutcome.RETRYABLE_FAILURE,
                    scan.message or "Character page scan failed.",
                    data=scan.data,
                    screenshot_path=scan.screenshot_path,
                )
            candidates.extend(_matching_observations(character, scan.observations))
            if len(candidates) == 1:
                state.selected_observation = candidates[0]
                return _step_result(
                    step.step_key,
                    WorkflowOutcome.SUCCESS,
                    data={
                        "pages_scanned": pages_scanned,
                        "matched_page_index": candidates[0].page_index,
                        "matched_slot_index": candidates[0].slot_index,
                        "matched_character_slot": candidates[0].character_slot,
                    },
                    screenshot_path=candidates[0].screenshot_path or scan.screenshot_path,
                )
            if len(candidates) > 1:
                return state.fail(
                    step.step_key,
                    "Multiple visible characters matched the target; add a slot, kingdom ID, or display fingerprint.",
                    screenshot_path=scan.screenshot_path,
                    data={"candidate_count": len(candidates), "pages_scanned": pages_scanned},
                )
            if not scan.has_next_page:
                break
            action = self.driver.go_to_next_character_page(state.request, character, page_index)
            if action.screenshot_path:
                state.screenshot_path = action.screenshot_path
            if not action.success:
                return self._action_to_step(step, state, action)
        return state.fail(
            step.step_key,
            "Target character was not found in character management.",
            screenshot_path=state.screenshot_path,
            data={"pages_scanned": pages_scanned},
        )

    def _select_character(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if state.already_active:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"already_active": True})
        observation = _require_observation(state)
        return self._action_to_step(
            step,
            state,
            self.driver.select_character(state.request, _require_character(state), observation),
        )

    def _confirm_switch(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if state.already_active:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"already_active": True})
        return self._action_to_step(
            step,
            state,
            self.driver.confirm_switch(state.request, _require_character(state)),
        )

    def _wait_for_reload(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if state.already_active:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"already_active": True})
        return self._action_to_step(
            step,
            state,
            self.driver.wait_for_reload(state.request, _require_character(state)),
        )

    def _verify_character(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if state.already_active:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"already_active": True})
        character = _require_character(state)
        verification = self.driver.verify_character(state.request, character)
        if verification.screenshot_path:
            state.screenshot_path = verification.screenshot_path
        if verification.captcha_detected or verification.verification_required:
            self._open_incident(state, "Character switch stopped for manual post-switch verification.", verification.screenshot_path)
            return state.fail(
                step.step_key,
                "Manual verification or CAPTCHA screen detected.",
                screenshot_path=verification.screenshot_path,
                data={"manual_verification_required": True},
            )
        if not _verification_matches(character, verification):
            self._open_incident(state, "Character switch verification mismatch.", verification.screenshot_path)
            return state.fail(
                step.step_key,
                verification.message or "Expected character was not active after switching.",
                screenshot_path=verification.screenshot_path,
                data={
                    "expected_character_id": character.id,
                    **_verification_data(verification),
                },
            )
        state.verification = verification
        state.selected_character_id = character.id
        state.selected_character_name = character.name
        self._persist_verified_session(state)
        self._audit_verified_switch(state)
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"selected_character_id": character.id, **_verification_data(verification)},
            screenshot_path=verification.screenshot_path,
        )

    def _complete(self, step: WorkflowStepSpec, state: _SwitchState) -> WorkflowStepResult:
        if state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"failure_state": state.failure_state})
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "selected_character_id": state.selected_character_id,
                "selected_character_name": state.selected_character_name,
                "before_character_id": state.before_character_id,
                "already_active": state.already_active,
            },
        )

    def _recover(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if self.recovery_watchdog is None:
            state.recovery_outcome = {"attempted": False, "reason": "watchdog_not_configured"}
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)
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
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=state.recovery_outcome)

    def _failed(self, step: WorkflowStepSpec, state: _SwitchState) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        return _step_result(
            step.step_key,
            WorkflowOutcome.FATAL_FAILURE,
            state.failure_reason,
            data={
                "failure_state": state.failure_state,
                "failure_reason": state.failure_reason,
                "recovery_outcome": state.recovery_outcome,
            },
            screenshot_path=state.screenshot_path,
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        action: CharacterSwitchActionResult,
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
                action.message or "Character switch action failed.",
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.fail(
            step.step_key,
            action.message or "Character switch action failed.",
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _state_from_context(self, context: WorkflowExecutionContext) -> _SwitchState:
        token = str(context.metadata.get("character_switch_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("Character switch runtime state is missing.") from exc

    def _character_by_name(self, name: str, account_id: int | None) -> Character | None:
        matches = [
            item
            for item in self.characters.list_all(include_disabled=True)
            if item.name.strip().casefold() == name.strip().casefold()
            and (account_id is None or item.game_account_id == account_id)
        ]
        return matches[0] if len(matches) == 1 else None

    def _current_session(self, request: CharacterSwitchRequest) -> InstanceSession:
        if request.session_key.strip():
            session = self.sessions.get_by_key(request.session_key)
            if session is not None:
                return session
        sessions = self.sessions.list_for_instance(request.instance_id)
        for session in sessions:
            if session.status == "running":
                return session
        session_key = request.session_key.strip() or f"instance:{request.instance_id}:active"
        session = InstanceSession(
            instance_id=request.instance_id,
            session_key=session_key,
            status="running",
            started_at=utc_now_iso(),
            adb_serial="",
            metadata_json="{}",
        )
        session.id = self.sessions.save(session)
        return session

    def _persist_verified_session(self, state: _SwitchState) -> None:
        session = state.session
        character = _require_character(state)
        account = _require_account(state)
        if session is None:
            return
        verification = state.verification
        metadata = _json_object(session.metadata_json)
        metadata.update(
            {
                "current_account_id": account.id,
                "current_character_id": character.id,
                "selected_character_id": character.id,
                "selected_character_name": character.name,
                "character_slot": character.character_slot,
                "character_display_fingerprint": (
                    verification.display_fingerprint if verification else character.display_fingerprint
                ),
                "character_kingdom_id": verification.kingdom_id if verification else character.kingdom_id,
                "character_switched_at": utc_now_iso(),
            }
        )
        if verification is not None:
            metadata["character_verification"] = _verification_data(verification)
        session.metadata_json = json.dumps(metadata, sort_keys=True)
        session.status = "running"
        self.sessions.save(session)
        if character.id is not None:
            self.characters.mark_switched(character.id)

    def _audit_verified_switch(self, state: _SwitchState) -> None:
        if self.audit_logs is None:
            return
        character = _require_character(state)
        account = _require_account(state)
        self.audit_logs.append(
            AuditLog(
                audit_key=f"character-switch:{state.request.instance_id}:{uuid4().hex}",
                actor=state.request.actor or "system",
                action="character_switch_verified",
                entity_type="instance_session",
                entity_id=state.session.id if state.session is not None else None,
                occurred_at=utc_now_iso(),
                details_json=json.dumps(
                    {
                        "account_id": account.id,
                        "before_character_id": state.before_character_id,
                        "after_character_id": character.id,
                        "target_character_id": character.id,
                        "already_active": state.already_active,
                    },
                    sort_keys=True,
                ),
            )
        )

    def _open_incident(
        self,
        state: _SwitchState,
        details: str,
        screenshot_path: str,
    ) -> None:
        if self.incidents is None:
            return
        self.incidents.save(
            Incident(
                incident_key=f"character-switch:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="Character switch requires manual intervention",
                details=details,
                screenshot_path=screenshot_path,
            )
        )

    def _augment_result(self, result: WorkflowExecutionResult, state: _SwitchState) -> None:
        result.result = {
            **dict(result.result),
            "selected_character_id": state.selected_character_id,
            "selected_character_name": state.selected_character_name,
            "before_account_id": state.before_account_id,
            "before_character_id": state.before_character_id,
            "failure_state": state.failure_state,
            "failure_reason": state.failure_reason,
            "recovery_outcome": state.recovery_outcome,
        }

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _SwitchState,
    ) -> None:
        if not result.outcome.is_failure or state.failed or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.failure_state = last_step.step_key if last_step is not None else ""
        state.failure_reason = result.message or "Character switch workflow failed."
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

    def _update_persisted_run(self, result: WorkflowExecutionResult, state: _SwitchState) -> None:
        if self.job_runs is None or result.job_run_id is None:
            return
        run = self.job_runs.get(result.job_run_id)
        if run is None:
            return
        run.result_json = json.dumps(result.to_json_dict(), sort_keys=True)
        run.error_message = state.failure_reason if result.outcome.is_failure else ""
        run.screenshot_path = state.screenshot_path
        self.job_runs.save(run)


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
        action_type=f"character_switch.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _SwitchState) -> Character:
    if state.target_character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.target_character


def _require_account(state: _SwitchState) -> GameAccount:
    if state.target_account is None:
        raise RuntimeError("Target account has not been loaded.")
    return state.target_account


def _require_observation(state: _SwitchState) -> CharacterSlotObservation:
    if state.selected_observation is None:
        raise RuntimeError("Target character has not been located.")
    return state.selected_observation


def _matching_observations(
    character: Character,
    observations: tuple[CharacterSlotObservation, ...],
) -> list[CharacterSlotObservation]:
    return [item for item in observations if _observation_matches(character, item)]


def _observation_matches(character: Character, observation: CharacterSlotObservation) -> bool:
    if observation.name.strip().casefold() != character.name.strip().casefold():
        return False
    if character.character_slot is not None and observation.character_slot != character.character_slot:
        return False
    if character.kingdom_id is not None and observation.kingdom_id != character.kingdom_id:
        return False
    if character.display_fingerprint.strip():
        return (
            observation.display_fingerprint.strip().casefold()
            == character.display_fingerprint.strip().casefold()
        )
    return True


def _verification_matches(character: Character, verification: CharacterVerification) -> bool:
    if verification.matched:
        return True
    if verification.character_id is not None and verification.character_id == character.id:
        return True
    if verification.character_name.strip().casefold() != character.name.strip().casefold():
        return False
    if character.character_slot is not None and verification.character_slot != character.character_slot:
        return False
    if character.kingdom_id is not None and verification.kingdom_id != character.kingdom_id:
        return False
    if character.display_fingerprint.strip():
        return (
            verification.display_fingerprint.strip().casefold()
            == character.display_fingerprint.strip().casefold()
        )
    return bool(verification.character_name.strip())


def _verification_data(verification: CharacterVerification) -> dict[str, object]:
    return {
        "observed_character_id": verification.character_id,
        "observed_character_name": verification.character_name,
        "observed_character_slot": verification.character_slot,
        "observed_display_fingerprint": verification.display_fingerprint,
        "observed_kingdom_id": verification.kingdom_id,
        "matched": verification.matched,
        **verification.data,
    }


def _json_object(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    return _int_or_none(value)
