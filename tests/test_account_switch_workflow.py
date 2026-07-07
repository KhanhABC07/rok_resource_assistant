from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests.db_helpers import SRC_ROOT  # noqa: F401

from rok_assistant.db.database import Database
from rok_assistant.db.models import GameAccount, Instance, InstanceSession, Job
from rok_assistant.db.repositories import (
    AuditLogRepository,
    GameAccountRepository,
    IncidentRepository,
    InstanceRepository,
    InstanceSessionRepository,
    JobRepository,
    JobRunRepository,
    StepRunRepository,
)
from rok_assistant.security import InMemorySecretStore, SecretMaterial
from rok_assistant.tasks.account_switch_workflow import (
    ACCOUNT_SWITCH_STATES,
    AccountSwitchActionResult,
    AccountSwitchConfig,
    AccountSwitchRequest,
    AccountSwitchWorkflow,
    AccountVerification,
)
from rok_assistant.workflow_engine import CancellationToken, WorkflowOutcome


class FakeDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_open_settings_once = False
        self.verification = AccountVerification(matched=True, account_id=1, fingerprint="fp-1")
        self.secret_refs: list[str] = []

    def open_settings(self, _request: AccountSwitchRequest, _account: GameAccount) -> AccountSwitchActionResult:
        self.calls.append("open_settings")
        if self.fail_open_settings_once:
            self.fail_open_settings_once = False
            return AccountSwitchActionResult(False, "settings not ready", retryable=True)
        return AccountSwitchActionResult(True)

    def open_account_menu(self, _request: AccountSwitchRequest, _account: GameAccount) -> AccountSwitchActionResult:
        self.calls.append("open_account_menu")
        return AccountSwitchActionResult(True)

    def select_account(
        self,
        _request: AccountSwitchRequest,
        _account: GameAccount,
        credential_ref: str,
    ) -> AccountSwitchActionResult:
        self.calls.append("select_account")
        self.secret_refs.append(credential_ref)
        return AccountSwitchActionResult(True)

    def wait_for_loading(self, _request: AccountSwitchRequest, _account: GameAccount) -> AccountSwitchActionResult:
        self.calls.append("wait_for_loading")
        return AccountSwitchActionResult(True)

    def verify_account(self, _request: AccountSwitchRequest, _account: GameAccount) -> AccountVerification:
        self.calls.append("verify_account")
        return self.verification


class FakeWatchdog:
    def __init__(self, *, healthy: bool = True) -> None:
        self.healthy = healthy
        self.calls: list[int | None] = []

    def monitor(
        self,
        *,
        instance_id: int,
        instance_index: int,
        instance_name: str,
        job_run_id: int | None = None,
    ) -> object:
        del instance_id, instance_index, instance_name
        self.calls.append(job_run_id)
        return SimpleNamespace(
            healthy=self.healthy,
            recovery_attempted=not self.healthy,
            circuit_opened=not self.healthy,
            observation=SimpleNamespace(
                message="unhealthy" if not self.healthy else "",
                screenshot_path="runtime/screens/unhealthy.png" if not self.healthy else "",
            ),
        )


class AccountSwitchWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "acc.sqlite3")
        self.db.initialize()
        self.instances = InstanceRepository(self.db)
        self.accounts = GameAccountRepository(self.db)
        self.sessions = InstanceSessionRepository(self.db)
        self.jobs = JobRepository(self.db)
        self.job_runs = JobRunRepository(self.db)
        self.step_runs = StepRunRepository(self.db)
        self.audit_logs = AuditLogRepository(self.db)
        self.incidents = IncidentRepository(self.db)
        self.instance_id = self.instances.save(
            Instance(name="MEmu 1", instance_index=0, instance_name="MEmu 1")
        )
        self.secret_store = InMemorySecretStore()
        self.driver = FakeDriver()
        self.watchdog = FakeWatchdog()

    def tearDown(self) -> None:
        self.db.close()
        self.temp_dir.cleanup()

    def _account(self, name: str = "Account A", *, secret: bool = True) -> int:
        secret_ref = ""
        if secret:
            secret_ref = self.secret_store.put(
                SecretMaterial(username=name, password="super-secret-password")
            )
        return self.accounts.save(
            GameAccount(
                account_name=name,
                display_name=name,
                provider="email",
                external_id=name.lower().replace(" ", "-"),
                secret_ref=secret_ref,
            )
        )

    def _session(self, metadata: dict[str, object] | None = None) -> int:
        return self.sessions.save(
            InstanceSession(
                instance_id=self.instance_id,
                session_key="session-1",
                status="running",
                started_at="2026-07-07T00:00:00",
                metadata_json=json.dumps(metadata or {}, sort_keys=True),
            )
        )

    def _job(self) -> int:
        return self.jobs.save(
            Job(
                idempotency_key="account-switch-job",
                job_type="workflow",
                scheduled_for="2026-07-07T00:00:00",
            )
        )

    def _workflow(self) -> AccountSwitchWorkflow:
        return AccountSwitchWorkflow(
            accounts=self.accounts,
            sessions=self.sessions,
            secret_store=self.secret_store,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            audit_logs=self.audit_logs,
            incidents=self.incidents,
            config=AccountSwitchConfig(step_timeout_seconds=2, workflow_timeout_seconds=30),
        )

    def _request(self, account_id: int, *, job_id: int | None = None) -> AccountSwitchRequest:
        return AccountSwitchRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            target_account_id=account_id,
            session_key="session-1",
            job_id=job_id,
            run_key="run-account-switch",
        )

    def test_workflow_exposes_required_states(self) -> None:
        self.assertEqual(ACCOUNT_SWITCH_STATES, self._workflow().workflow_states)

    def test_already_active_account_skips_secret_retrieval_and_navigation(self) -> None:
        account_id = self._account(secret=False)
        self._session({"current_account_id": account_id})
        self.driver.verification = AccountVerification(
            matched=True,
            account_id=account_id,
            account_name="Account A",
            fingerprint="fp-active",
        )

        result = self._workflow().execute(self._request(account_id))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["verify_account"], self.driver.calls)
        session = self.sessions.get_by_key("session-1")
        metadata = json.loads(session.metadata_json)  # type: ignore[union-attr]
        self.assertEqual(account_id, metadata["current_account_id"])
        self.assertEqual("fp-active", metadata["account_fingerprint"])

    def test_successful_switch_persists_verified_session_audit_and_job_result(self) -> None:
        account_id = self._account()
        self._session({"current_account_id": 99})
        job_id = self._job()
        self.driver.fail_open_settings_once = True
        self.driver.verification = AccountVerification(
            matched=True,
            account_id=account_id,
            account_name="Account A",
            fingerprint="fp-switched",
            screenshot_path="runtime/screens/verified.png",
        )

        result = self._workflow().execute(self._request(account_id, job_id=job_id))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(
            [
                "open_settings",
                "open_settings",
                "open_account_menu",
                "select_account",
                "wait_for_loading",
                "verify_account",
            ],
            self.driver.calls,
        )
        session = self.sessions.get_by_key("session-1")
        metadata = json.loads(session.metadata_json)  # type: ignore[union-attr]
        self.assertEqual(account_id, metadata["current_account_id"])
        self.assertEqual("Account A", metadata["selected_account_name"])
        self.assertEqual("fp-switched", metadata["account_fingerprint"])
        self.assertEqual(1, len(self.audit_logs.list_recent()))
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual(account_id, payload["result"]["selected_account_id"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_missing_credentials_fail_before_navigation_and_do_not_update_session(self) -> None:
        account_id = self._account(secret=False)
        self._session({"current_account_id": 7})

        result = self._workflow().execute(self._request(account_id))

        self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
        self.assertEqual([], self.driver.calls)
        self.assertEqual("validate_credentials", result.result["failure_state"])
        session = self.sessions.get_by_key("session-1")
        metadata = json.loads(session.metadata_json)  # type: ignore[union-attr]
        self.assertEqual(7, metadata["current_account_id"])

    def test_wrong_account_opens_incident_runs_recovery_and_keeps_previous_session(self) -> None:
        account_id = self._account()
        self._session({"current_account_id": 7})
        self.driver.verification = AccountVerification(
            matched=False,
            account_id=123,
            account_name="Wrong",
            fingerprint="wrong-fp",
            screenshot_path="runtime/screens/wrong.png",
        )

        result = self._workflow().execute(self._request(account_id))

        self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
        self.assertEqual("verify_account", result.result["failure_state"])
        self.assertEqual({"attempted": False, "healthy": True, "circuit_opened": False}, result.result["recovery_outcome"])
        self.assertEqual(1, len(self.incidents.list_open()))
        session = self.sessions.get_by_key("session-1")
        metadata = json.loads(session.metadata_json)  # type: ignore[union-attr]
        self.assertEqual(7, metadata["current_account_id"])

    def test_verification_screen_stops_for_manual_intervention(self) -> None:
        account_id = self._account()
        self._session({"current_account_id": 7})
        self.driver.verification = AccountVerification(
            verification_required=True,
            screenshot_path="runtime/screens/verify.png",
        )

        result = self._workflow().execute(self._request(account_id))

        self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
        self.assertIn("Manual verification", result.message)
        self.assertEqual(1, len(self.incidents.list_open()))

    def test_cancellation_before_execution_persists_cancelled_run(self) -> None:
        account_id = self._account()
        self._session({"current_account_id": 7})
        job_id = self._job()
        token = CancellationToken()
        token.cancel("operator cancelled")

        result = self._workflow().execute(self._request(account_id, job_id=job_id), cancellation_token=token)

        self.assertEqual(WorkflowOutcome.CANCELLED, result.outcome)
        self.assertEqual("operator cancelled", result.message)
        self.assertEqual([], self.driver.calls)
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertIsNone(run)

    def test_no_plaintext_secret_is_written_to_database(self) -> None:
        account_id = self._account()
        self._session({"current_account_id": 7})
        job_id = self._job()
        self.driver.verification = AccountVerification(matched=True, account_id=account_id, fingerprint="fp")

        self._workflow().execute(self._request(account_id, job_id=job_id))

        rows = self.db.fetch_all(
            """
            SELECT result_json AS text FROM job_runs
            UNION ALL SELECT result_json AS text FROM step_runs
            UNION ALL SELECT metadata_json AS text FROM instance_sessions
            UNION ALL SELECT details_json AS text FROM audit_logs
            """
        )
        persisted = "\n".join(row["text"] for row in rows)
        self.assertNotIn("super-secret-password", persisted)

    def test_more_than_six_enabled_accounts_is_rejected(self) -> None:
        for index in range(7):
            self._account(f"Account {index}")
        self._session()

        result = self._workflow().execute(self._request(1))

        self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
        self.assertEqual("validate_input", result.result["failure_state"])
        self.assertEqual([], self.driver.calls)


if __name__ == "__main__":
    unittest.main()
