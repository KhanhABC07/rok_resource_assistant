from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from rok_assistant.db.models import (
    AuditLog,
    GameAccount,
    Incident,
    InstanceSession,
    JobRun,
    utc_now_iso,
)
from rok_assistant.security import SecretStore, validate_account_credentials
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


ACCOUNT_SWITCH_WORKFLOW_KEY = "account-switch"
ACCOUNT_SWITCH_STATES = (
    "validate_input",
    "load_account",
    "validate_credentials",
    "ensure_game_running",
    "open_settings",
    "open_account_menu",
    "select_account",
    "wait_for_loading",
    "verify_account",
    "complete",
    "recover",
    "failed",
    "cancelled",
)


@dataclass(frozen=True)
class AccountSwitchRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    target_account_id: int | None = None
    target_account_name: str = ""
    session_key: str = ""
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class AccountSwitchConfig:
    max_configured_accounts: int = 6
    workflow_timeout_seconds: float = 120.0
    step_timeout_seconds: float = 15.0
    navigation_retry_limit: int = 1
    retry_delay_seconds: float = 0.25


@dataclass(frozen=True)
class AccountSwitchActionResult:
    success: bool
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AccountVerification:
    account_id: int | None = None
    account_name: str = ""
    fingerprint: str = ""
    matched: bool = False
    verification_required: bool = False
    captcha_detected: bool = False
    screenshot_path: str = ""
    message: str = ""
    data: dict[str, object] = field(default_factory=dict)


class GameAccountRepository(Protocol):
    def list_all(self, include_disabled: bool = True) -> list[GameAccount]:
        ...

    def get(self, account_id: int) -> GameAccount | None:
        ...

    def get_by_name(self, account_name: str) -> GameAccount | None:
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


class AccountSwitchDriver(Protocol):
    def open_settings(
        self,
        request: AccountSwitchRequest,
        account: GameAccount,
    ) -> AccountSwitchActionResult:
        ...

    def open_account_menu(
        self,
        request: AccountSwitchRequest,
        account: GameAccount,
    ) -> AccountSwitchActionResult:
        ...

    def select_account(
        self,
        request: AccountSwitchRequest,
        account: GameAccount,
        credential_ref: str,
    ) -> AccountSwitchActionResult:
        ...

    def wait_for_loading(
        self,
        request: AccountSwitchRequest,
        account: GameAccount,
    ) -> AccountSwitchActionResult:
        ...

    def verify_account(
        self,
        request: AccountSwitchRequest,
        account: GameAccount,
    ) -> AccountVerification:
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
    request: AccountSwitchRequest
    target_account: GameAccount | None = None
    session: InstanceSession | None = None
    before_account_id: int | None = None
    selected_account_id: int | None = None
    selected_account_name: str = ""
    fingerprint: str = ""
    failure_reason: str = ""
    failure_state: str = ""
    recovery_outcome: dict[str, object] = field(default_factory=dict)
    screenshot_path: str = ""
    credential_checked: bool = False
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


class AccountSwitchWorkflow:
    def __init__(
        self,
        *,
        accounts: GameAccountRepository,
        sessions: InstanceSessionRepository,
        secret_store: SecretStore,
        driver: AccountSwitchDriver,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        audit_logs: AuditLogRepository | None = None,
        incidents: IncidentRepository | None = None,
        config: AccountSwitchConfig | None = None,
    ) -> None:
        self.accounts = accounts
        self.sessions = sessions
        self.secret_store = secret_store
        self.driver = driver
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.audit_logs = audit_logs
        self.incidents = incidents
        self.config = config or AccountSwitchConfig()
        self._states: dict[str, _SwitchState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return ACCOUNT_SWITCH_STATES

    def execute(
        self,
        request: AccountSwitchRequest,
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
                budget=StepBudget(max_steps=len(ACCOUNT_SWITCH_STATES) + 4),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"account-switch:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"account_switch_run_id": token},
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
        for state in ACCOUNT_SWITCH_STATES:
            registry.register(f"account_switch.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retryable = {
            "ensure_game_running",
            "open_settings",
            "open_account_menu",
            "select_account",
            "wait_for_loading",
            "verify_account",
        }
        return WorkflowDefinitionSpec(
            workflow_key=ACCOUNT_SWITCH_WORKFLOW_KEY,
            name="Switch Account",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"account_switch.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=(
                        self.config.navigation_retry_limit if state in retryable else 0
                    ),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in ACCOUNT_SWITCH_STATES
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
        if request.target_account_id is None and not request.target_account_name.strip():
            return state.fail(step.step_key, "target account id or name is required.")
        enabled_accounts = self.accounts.list_all(include_disabled=False)
        if len(enabled_accounts) > self.config.max_configured_accounts:
            return state.fail(
                step.step_key,
                "More than six enabled accounts are configured.",
                data={"enabled_account_count": len(enabled_accounts)},
            )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS)

    def _load_account(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        request = state.request
        account = (
            self.accounts.get(request.target_account_id)
            if request.target_account_id is not None
            else self.accounts.get_by_name(request.target_account_name)
        )
        if account is None:
            return state.fail(step.step_key, "Target account was not found.")
        if not account.enabled:
            return state.fail(step.step_key, "Target account is disabled.")
        session = self._current_session(request)
        state.target_account = account
        state.session = session
        state.before_account_id = _metadata_account_id(session.metadata_json if session else "{}")
        state.already_active = state.before_account_id == account.id
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "target_account_id": account.id,
                "target_account_name": account.account_name,
                "before_account_id": state.before_account_id,
                "already_active": state.already_active,
            },
        )

    def _validate_credentials(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        account = _require_account(state)
        if state.already_active:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data={"credential_check": "skipped_already_active"},
            )
        validation = validate_account_credentials(account.secret_ref, self.secret_store)
        state.credential_checked = True
        if not validation.ok:
            return state.fail(
                step.step_key,
                validation.message or "Account credentials are not available.",
                data={"credential_failure_reason": validation.reason.value if validation.reason else ""},
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={"credential_ref": account.secret_ref, "credential_check": "ok"},
        )

    def _ensure_game_running(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.recovery_watchdog is None:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data={"watchdog": "not_configured"},
            )
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

    def _open_settings(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if state.already_active:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"already_active": True})
        return self._driver_result(step, state, self.driver.open_settings)

    def _open_account_menu(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if state.already_active:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"already_active": True})
        return self._driver_result(step, state, self.driver.open_account_menu)

    def _select_account(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if state.already_active:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"already_active": True})
        account = _require_account(state)
        action = self.driver.select_account(
            state.request,
            account,
            account.secret_ref,
        )
        return self._action_to_step(step, state, action)

    def _wait_for_loading(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if state.already_active:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED, data={"already_active": True})
        return self._driver_result(step, state, self.driver.wait_for_loading)

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
            self._open_incident(
                state,
                "Account switch stopped for manual verification.",
                verification.screenshot_path,
            )
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
            self._open_incident(
                state,
                "Account switch verification mismatch.",
                verification.screenshot_path,
            )
            return state.fail(
                step.step_key,
                verification.message or "Expected account was not active after switching.",
                screenshot_path=verification.screenshot_path,
                data={
                    "expected_account_id": account.id,
                    "observed_account_id": verification.account_id,
                    "observed_account_name": verification.account_name,
                },
            )
        state.selected_account_id = account.id
        state.selected_account_name = account.account_name
        state.fingerprint = verification.fingerprint
        self._persist_verified_session(state)
        self._audit_verified_switch(state)
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "selected_account_id": account.id,
                "selected_account_name": account.account_name,
                "fingerprint": verification.fingerprint,
                "already_active": state.already_active,
            },
            screenshot_path=verification.screenshot_path,
        )

    def _complete(self, step: WorkflowStepSpec, state: _SwitchState) -> WorkflowStepResult:
        if state.failed:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                data={"failure_state": state.failure_state},
            )
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "selected_account_id": state.selected_account_id,
                "selected_account_name": state.selected_account_name,
                "before_account_id": state.before_account_id,
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

    def _driver_result(self, step: WorkflowStepSpec, state: _SwitchState, action: Any) -> WorkflowStepResult:
        return self._action_to_step(
            step,
            state,
            action(state.request, _require_account(state)),
        )

    def _action_to_step(
        self,
        step: WorkflowStepSpec,
        state: _SwitchState,
        action: AccountSwitchActionResult,
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
        return _step_result(
            step.step_key,
            WorkflowOutcome.RETRYABLE_FAILURE if action.retryable else WorkflowOutcome.SUCCESS,
            action.message or "Account switch action failed.",
            data=action.data,
            screenshot_path=action.screenshot_path,
        ) if action.retryable else state.fail(
            step.step_key,
            action.message or "Account switch action failed.",
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _state_from_context(self, context: WorkflowExecutionContext) -> _SwitchState:
        token = str(context.metadata.get("account_switch_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("Account switch runtime state is missing.") from exc

    def _current_session(self, request: AccountSwitchRequest) -> InstanceSession:
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
        account = _require_account(state)
        if session is None:
            return
        metadata = _json_object(session.metadata_json)
        metadata.update(
            {
                "current_account_id": account.id,
                "selected_account_id": account.id,
                "selected_account_name": account.account_name,
                "account_fingerprint": state.fingerprint,
                "account_switched_at": utc_now_iso(),
            }
        )
        session.metadata_json = json.dumps(metadata, sort_keys=True)
        session.status = "running"
        self.sessions.save(session)

    def _audit_verified_switch(self, state: _SwitchState) -> None:
        if self.audit_logs is None:
            return
        account = _require_account(state)
        self.audit_logs.append(
            AuditLog(
                audit_key=f"account-switch:{state.request.instance_id}:{uuid4().hex}",
                actor=state.request.actor or "system",
                action="account_switch_verified",
                entity_type="instance_session",
                entity_id=state.session.id if state.session is not None else None,
                occurred_at=utc_now_iso(),
                details_json=json.dumps(
                    {
                        "before_account_id": state.before_account_id,
                        "after_account_id": account.id,
                        "target_account_id": account.id,
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
                incident_key=f"account-switch:{state.request.instance_id}:{uuid4().hex}",
                severity="error",
                status="open",
                title="Account switch requires manual intervention",
                details=details,
                screenshot_path=screenshot_path,
            )
        )

    def _augment_result(
        self,
        result: WorkflowExecutionResult,
        state: _SwitchState,
    ) -> None:
        result.result = {
            **dict(result.result),
            "selected_account_id": state.selected_account_id,
            "selected_account_name": state.selected_account_name,
            "before_account_id": state.before_account_id,
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
        state.failure_reason = result.message or "Account switch workflow failed."
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

    def _update_persisted_run(
        self,
        result: WorkflowExecutionResult,
        state: _SwitchState,
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
        action_type=f"account_switch.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_account(state: _SwitchState) -> GameAccount:
    if state.target_account is None:
        raise RuntimeError("Target account has not been loaded.")
    return state.target_account


def _json_object(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _metadata_account_id(metadata_json: str) -> int | None:
    value = _json_object(metadata_json).get("current_account_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
