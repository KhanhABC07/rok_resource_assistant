from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests.db_helpers import SRC_ROOT  # noqa: F401

from rok_assistant.db.database import Database
from rok_assistant.db.models import Character, GameAccount, Instance, InstanceSession, Job
from rok_assistant.db.repositories import (
    AuditLogRepository,
    CharacterRepository,
    GameAccountRepository,
    IncidentRepository,
    InstanceRepository,
    InstanceSessionRepository,
    JobRepository,
    JobRunRepository,
    StepRunRepository,
)
from rok_assistant.tasks.account_switch_workflow import AccountVerification
from rok_assistant.tasks.character_switch_workflow import (
    CHARACTER_SWITCH_STATES,
    CharacterPageScan,
    CharacterSlotObservation,
    CharacterSwitchActionResult,
    CharacterSwitchConfig,
    CharacterSwitchRequest,
    CharacterSwitchWorkflow,
    CharacterVerification,
)
from rok_assistant.workflow_engine import WorkflowOutcome


class FakeCharacterDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.account_verification = AccountVerification(matched=True, account_id=1, fingerprint="account-fp")
        self.current_verification = CharacterVerification(character_name="Other")
        self.final_verification = CharacterVerification(
            matched=True,
            character_id=1,
            character_name="Farm01",
            character_slot=1,
            display_fingerprint="fp-1",
            kingdom_id=1001,
        )
        self.page_scans: dict[int, CharacterPageScan] = {}
        self.wait_result = CharacterSwitchActionResult(True)

    def verify_account(
        self,
        _request: CharacterSwitchRequest,
        _account: GameAccount,
    ) -> AccountVerification:
        self.calls.append("verify_account")
        return self.account_verification

    def verify_character(
        self,
        _request: CharacterSwitchRequest,
        _character: Character,
    ) -> CharacterVerification:
        self.calls.append("verify_character")
        if self.calls.count("verify_character") == 1:
            return self.current_verification
        return self.final_verification

    def open_character_management(
        self,
        _request: CharacterSwitchRequest,
        _character: Character,
    ) -> CharacterSwitchActionResult:
        self.calls.append("open_character_management")
        return CharacterSwitchActionResult(True)

    def scan_character_page(
        self,
        _request: CharacterSwitchRequest,
        _character: Character,
        page_index: int,
    ) -> CharacterPageScan:
        self.calls.append(f"scan_character_page:{page_index}")
        return self.page_scans.get(page_index, CharacterPageScan(True))

    def go_to_next_character_page(
        self,
        _request: CharacterSwitchRequest,
        _character: Character,
        page_index: int,
    ) -> CharacterSwitchActionResult:
        self.calls.append(f"go_to_next_character_page:{page_index}")
        return CharacterSwitchActionResult(True)

    def select_character(
        self,
        _request: CharacterSwitchRequest,
        _character: Character,
        observation: CharacterSlotObservation,
    ) -> CharacterSwitchActionResult:
        self.calls.append(f"select_character:{observation.page_index}:{observation.slot_index}")
        return CharacterSwitchActionResult(True)

    def confirm_switch(
        self,
        _request: CharacterSwitchRequest,
        _character: Character,
    ) -> CharacterSwitchActionResult:
        self.calls.append("confirm_switch")
        return CharacterSwitchActionResult(True)

    def wait_for_reload(
        self,
        _request: CharacterSwitchRequest,
        _character: Character,
    ) -> CharacterSwitchActionResult:
        self.calls.append("wait_for_reload")
        return self.wait_result


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


class CharacterSwitchWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "chr.sqlite3")
        self.db.initialize()
        self.instances = InstanceRepository(self.db)
        self.accounts = GameAccountRepository(self.db)
        self.characters = CharacterRepository(self.db)
        self.sessions = InstanceSessionRepository(self.db)
        self.jobs = JobRepository(self.db)
        self.job_runs = JobRunRepository(self.db)
        self.step_runs = StepRunRepository(self.db)
        self.audit_logs = AuditLogRepository(self.db)
        self.incidents = IncidentRepository(self.db)
        self.instance_id = self.instances.save(
            Instance(name="MEmu 1", instance_index=0, instance_name="MEmu 1")
        )
        self.account_id = self.accounts.save(GameAccount(account_name="Account A"))
        self.driver = FakeCharacterDriver()
        self.driver.account_verification = AccountVerification(
            matched=True,
            account_id=self.account_id,
            account_name="Account A",
            fingerprint="account-fp",
        )
        self.watchdog = FakeWatchdog()

    def tearDown(self) -> None:
        self.db.close()
        self.temp_dir.cleanup()

    def _character(
        self,
        name: str = "Farm01",
        *,
        slot: int = 1,
        kingdom_id: int = 1001,
        fingerprint: str = "fp-1",
    ) -> int:
        return self.characters.save(
            Character(
                name=name,
                instance_id=self.instance_id,
                account_name="Account A",
                game_account_id=self.account_id,
                character_slot=slot,
                kingdom_id=kingdom_id,
                display_fingerprint=fingerprint,
                verification_metadata_json='{"source": "test"}',
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
                idempotency_key="character-switch-job",
                job_type="workflow",
                scheduled_for="2026-07-07T00:00:00",
            )
        )

    def _workflow(self) -> CharacterSwitchWorkflow:
        return CharacterSwitchWorkflow(
            characters=self.characters,
            accounts=self.accounts,
            sessions=self.sessions,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            audit_logs=self.audit_logs,
            incidents=self.incidents,
            config=CharacterSwitchConfig(step_timeout_seconds=2, workflow_timeout_seconds=30),
        )

    def _request(self, character_id: int, *, job_id: int | None = None) -> CharacterSwitchRequest:
        return CharacterSwitchRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            target_character_id=character_id,
            session_key="session-1",
            job_id=job_id,
            run_key="run-character-switch",
        )

    def test_workflow_exposes_required_states(self) -> None:
        self.assertEqual(CHARACTER_SWITCH_STATES, self._workflow().workflow_states)

    def test_already_active_character_skips_navigation_and_persists_verification(self) -> None:
        character_id = self._character()
        self._session({"current_account_id": self.account_id, "current_character_id": character_id})
        self.driver.current_verification = CharacterVerification(
            matched=True,
            character_id=character_id,
            character_name="Farm01",
            character_slot=1,
            display_fingerprint="fp-active",
            kingdom_id=1001,
        )

        result = self._workflow().execute(self._request(character_id))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["verify_account", "verify_character"], self.driver.calls)
        session = self.sessions.get_by_key("session-1")
        metadata = json.loads(session.metadata_json)  # type: ignore[union-attr]
        self.assertEqual(character_id, metadata["current_character_id"])
        self.assertEqual("fp-active", metadata["character_display_fingerprint"])

    def test_target_on_later_page_is_selected_verified_and_persisted(self) -> None:
        character_id = self._character()
        self._session({"current_account_id": self.account_id, "current_character_id": 99})
        job_id = self._job()
        self.driver.page_scans = {
            0: CharacterPageScan(True, has_next_page=True),
            1: CharacterPageScan(
                True,
                observations=(
                    CharacterSlotObservation(
                        name="Farm01",
                        character_slot=1,
                        display_fingerprint="fp-1",
                        kingdom_id=1001,
                        page_index=1,
                        slot_index=2,
                    ),
                ),
            ),
        }
        self.driver.final_verification = CharacterVerification(
            matched=True,
            character_id=character_id,
            character_name="Farm01",
            character_slot=1,
            display_fingerprint="fp-1",
            kingdom_id=1001,
            screenshot_path="runtime/screens/character-verified.png",
        )

        result = self._workflow().execute(self._request(character_id, job_id=job_id))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("go_to_next_character_page:0", self.driver.calls)
        self.assertIn("select_character:1:2", self.driver.calls)
        session = self.sessions.get_by_key("session-1")
        metadata = json.loads(session.metadata_json)  # type: ignore[union-attr]
        self.assertEqual(character_id, metadata["current_character_id"])
        self.assertEqual(self.account_id, metadata["current_account_id"])
        self.assertEqual(1, len(self.audit_logs.list_recent()))
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual(character_id, payload["result"]["selected_character_id"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_duplicate_names_are_disambiguated_by_kingdom_and_fingerprint(self) -> None:
        other_id = self._character("Twin", slot=1, kingdom_id=1001, fingerprint="fp-a")
        del other_id
        character_id = self._character("Twin", slot=2, kingdom_id=1002, fingerprint="fp-b")
        self._session({"current_account_id": self.account_id, "current_character_id": 99})
        self.driver.page_scans = {
            0: CharacterPageScan(
                True,
                observations=(
                    CharacterSlotObservation(
                        name="Twin",
                        character_slot=1,
                        display_fingerprint="fp-a",
                        kingdom_id=1001,
                        slot_index=0,
                    ),
                    CharacterSlotObservation(
                        name="Twin",
                        character_slot=2,
                        display_fingerprint="fp-b",
                        kingdom_id=1002,
                        slot_index=1,
                    ),
                ),
            )
        }
        self.driver.final_verification = CharacterVerification(
            matched=True,
            character_id=character_id,
            character_name="Twin",
            character_slot=2,
            display_fingerprint="fp-b",
            kingdom_id=1002,
        )

        result = self._workflow().execute(self._request(character_id))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("select_character:0:1", self.driver.calls)

    def test_missing_target_fails_without_updating_session(self) -> None:
        character_id = self._character()
        self._session({"current_account_id": self.account_id, "current_character_id": 7})
        self.driver.page_scans = {
            0: CharacterPageScan(True, has_next_page=True),
            1: CharacterPageScan(True, has_next_page=False),
        }

        result = self._workflow().execute(self._request(character_id))

        self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
        self.assertEqual("find_character", result.result["failure_state"])
        session = self.sessions.get_by_key("session-1")
        metadata = json.loads(session.metadata_json)  # type: ignore[union-attr]
        self.assertEqual(7, metadata["current_character_id"])

    def test_reload_timeout_fails_and_records_recovery_outcome(self) -> None:
        character_id = self._character()
        self._session({"current_account_id": self.account_id, "current_character_id": 7})
        self.driver.page_scans = {
            0: CharacterPageScan(
                True,
                observations=(
                    CharacterSlotObservation(
                        name="Farm01",
                        character_slot=1,
                        display_fingerprint="fp-1",
                        kingdom_id=1001,
                    ),
                ),
            )
        }
        self.driver.wait_result = CharacterSwitchActionResult(
            False,
            "Reload timed out.",
            retryable=False,
            screenshot_path="runtime/screens/reload-timeout.png",
        )

        result = self._workflow().execute(self._request(character_id))

        self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
        self.assertEqual("wait_for_reload", result.result["failure_state"])
        self.assertEqual(
            {"attempted": False, "healthy": True, "circuit_opened": False},
            result.result["recovery_outcome"],
        )
        session = self.sessions.get_by_key("session-1")
        metadata = json.loads(session.metadata_json)  # type: ignore[union-attr]
        self.assertEqual(7, metadata["current_character_id"])


if __name__ == "__main__":
    unittest.main()
