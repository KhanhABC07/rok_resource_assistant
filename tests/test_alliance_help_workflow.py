from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.db.database import Database  # noqa: E402
from rok_assistant.db.models import Character, Instance, Job  # noqa: E402
from rok_assistant.db.repositories import (  # noqa: E402
    CharacterRepository,
    IncidentRepository,
    InstanceRepository,
    JobRepository,
    JobRunRepository,
    StepRunRepository,
)
from rok_assistant.tasks.alliance_help_workflow import (  # noqa: E402
    ALLIANCE_HELP_STATES,
    ALLIANCE_HELP_TEMPLATE_KEYS,
    AllianceHelpActionResult,
    AllianceHelpConfig,
    AllianceHelpObservation,
    AllianceHelpPolicy,
    AllianceHelpRequest,
    AllianceHelpStatus,
    AllianceHelpWorkflow,
)
from rok_assistant.tasks.resource_search_workflow import ResourceGatheringActionResult  # noqa: E402
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


class FakeAllianceHelpDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.normalize = ResourceGatheringActionResult(True, data={"scene": "ALLIANCE_HELP_OBSERVABLE"})
        self.detect = AllianceHelpObservation(
            AllianceHelpStatus.READY,
            confidence=0.94,
            target=(810, 210),
            button_active=True,
            badge_visible=True,
            screenshot_path="runtime/screens/help-ready.png",
        )
        self.press = AllianceHelpActionResult(True, changed=True, screenshot_path="runtime/screens/help-click.png")
        self.verify = AllianceHelpObservation(
            AllianceHelpStatus.NOT_READY,
            confidence=0.91,
            target=(810, 210),
            button_active=False,
            badge_visible=False,
            screenshot_path="runtime/screens/help-done.png",
        )

    def normalize_help_scene(self, _request, _character, _policy):
        self.calls.append("normalize_help_scene")
        return self.normalize

    def detect_help_button(self, _request, _character, _policy):
        self.calls.append("detect_help_button")
        return self.detect

    def press_help(self, _request, _character, _observation, _policy):
        self.calls.append("press_help")
        return self.press

    def verify_help_state(self, _request, _character, _before, _press_result, _policy):
        self.calls.append("verify_help_state")
        return self.verify


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


class AllianceHelpWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "alliance-help.sqlite3")
        self.db.initialize()
        self.instances = InstanceRepository(self.db)
        self.characters = CharacterRepository(self.db)
        self.jobs = JobRepository(self.db)
        self.job_runs = JobRunRepository(self.db)
        self.step_runs = StepRunRepository(self.db)
        self.incidents = IncidentRepository(self.db)
        self.instance_id = self.instances.save(
            Instance(name="MEmu 1", instance_index=0, instance_name="MEmu 1")
        )
        self.character_id = self.characters.save(
            Character(id=None, name="Farm01", instance_id=self.instance_id)
        )
        self.driver = FakeAllianceHelpDriver()
        self.watchdog = FakeWatchdog()

    def tearDown(self) -> None:
        self.db.close()
        self.temp_dir.cleanup()

    def _job(self, key: str) -> int:
        return self.jobs.save(
            Job(
                idempotency_key=key,
                job_type="workflow",
                scheduled_for="2026-07-08T00:00:00",
            )
        )

    def _workflow(self) -> AllianceHelpWorkflow:
        return AllianceHelpWorkflow(
            characters=self.characters,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=AllianceHelpConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "alliance-help-run",
        policy: AllianceHelpPolicy | None = None,
    ) -> AllianceHelpRequest:
        return AllianceHelpRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or AllianceHelpPolicy(),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states_and_template_key(self) -> None:
        self.assertEqual(ALLIANCE_HELP_STATES, self._workflow().workflow_states)
        self.assertEqual(("alliance.help.ready",), ALLIANCE_HELP_TEMPLATE_KEYS)

    def test_help_ready_clicks_once_and_verifies_inactive_state(self) -> None:
        job_id = self._job("alliance-help-ready")

        result = self._workflow().execute(
            self._request(job_id=job_id, run_key="alliance-help-ready")
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(
            ["normalize_help_scene", "detect_help_button", "press_help", "verify_help_state"],
            self.driver.calls,
        )
        self.assertEqual(1, self.driver.calls.count("press_help"))
        self.assertTrue(result.result["click_attempted"])
        self.assertFalse(result.result["verification_result"]["badge_visible"])
        self.assertFalse(result.result["verification_result"]["button_active"])
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertTrue(payload["result"]["click_attempted"])
        self.assertEqual("READY", payload["result"]["help_ready_detection"]["status"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_help_not_ready_returns_skipped_without_clicking(self) -> None:
        self.driver.detect = AllianceHelpObservation(
            AllianceHelpStatus.NOT_READY,
            confidence=0.88,
            button_active=False,
            badge_visible=False,
            message="No help badge.",
            screenshot_path="runtime/screens/help-empty.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("alliance-help-empty"), run_key="alliance-help-empty")
        )

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("detect_help_button", result.result["terminal_state"])
        self.assertEqual("No help badge.", result.result["skipped_reason"])
        self.assertFalse(result.result["click_attempted"])
        self.assertNotIn("press_help", self.driver.calls)
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("completed", run.status)  # type: ignore[union-attr]
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_overlay_obstruction_records_failure_evidence(self) -> None:
        self.driver.detect = AllianceHelpObservation(
            AllianceHelpStatus.OVERLAY_BLOCKED,
            confidence=0.92,
            target=(810, 210),
            button_active=True,
            badge_visible=True,
            overlay_blocked=True,
            message="Overlay blocks help button.",
            screenshot_path="runtime/screens/help-overlay.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("alliance-help-overlay"), run_key="alliance-help-overlay")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("detect_help_button", result.result["terminal_state"])
        self.assertIn("Overlay blocks", result.result["terminal_reason"])
        self.assertFalse(result.result["click_attempted"])
        self.assertNotIn("press_help", self.driver.calls)
        self.assertEqual(
            "runtime/screens/help-overlay.png",
            result.result["failure_evidence"]["screenshot_path"],
        )
        self.assertEqual(1, len(self.incidents.list_open()))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("failed", run.status)  # type: ignore[union-attr]

    def test_postcondition_failure_records_failure_evidence(self) -> None:
        self.driver.verify = AllianceHelpObservation(
            AllianceHelpStatus.READY,
            confidence=0.93,
            target=(810, 210),
            button_active=True,
            badge_visible=True,
            message="Badge remained visible.",
            screenshot_path="runtime/screens/help-still-ready.png",
        )

        result = self._workflow().execute(
            self._request(
                job_id=self._job("alliance-help-postcondition"),
                run_key="alliance-help-postcondition",
            )
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("verify_help_state", result.result["terminal_state"])
        self.assertIn("badge did not disappear", result.result["terminal_reason"])
        self.assertEqual(1, self.driver.calls.count("press_help"))
        self.assertEqual(
            "runtime/screens/help-still-ready.png",
            result.result["failure_evidence"]["screenshot_path"],
        )
        self.assertEqual(
            {"attempted": False, "healthy": True, "circuit_opened": False},
            result.result["recovery_outcome"],
        )
        self.assertEqual(1, len(self.incidents.list_open()))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("failed", run.status)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
