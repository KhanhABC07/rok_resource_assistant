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
    MarchRepository,
    StepRunRepository,
)
from rok_assistant.tasks.alliance_rally_join_workflow import (  # noqa: E402
    ALLIANCE_RALLY_JOIN_STATES,
    ALLIANCE_RALLY_JOIN_TEMPLATE_KEYS,
    AllianceRallyJoinConfig,
    AllianceRallyJoinPolicy,
    AllianceRallyJoinRequest,
    AllianceRallyJoinWorkflow,
    AllianceRallyObservation,
    AllianceRallyScan,
    RallyJoinedVerification,
    TroopAvailability,
)
from rok_assistant.tasks.resource_search_workflow import (  # noqa: E402
    MarchAvailability,
    MarchDispatchResult,
    ResourceGatheringActionResult,
)
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


class FakeAllianceRallyDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.open_result = ResourceGatheringActionResult(True)
        self.rallies = (
            AllianceRallyObservation(
                rally_id="rally-1",
                rally_type="barbarian_fort",
                leader_name="Alice",
                target_name="Fort 12",
                duration_seconds=600,
                remaining_capacity=5,
                confidence=0.95,
                data={"row": 1},
            ),
        )
        self.scan_message = ""
        self.scan_screenshot = ""
        self.select_result = ResourceGatheringActionResult(True, data={"selected": True})
        self.march = MarchAvailability(True, march_slot=3, available_count=1)
        self.troops = TroopAvailability(True, troop_count=120000)
        self.join = ResourceGatheringActionResult(True, data={"join_clicked": True})
        self.preset = ResourceGatheringActionResult(True, data={"preset_selected": True})
        self.dispatch = MarchDispatchResult(
            True,
            march_slot=3,
            dispatch_id="rally-dispatch-1",
            expected_return_time="2026-07-09T03:00:00",
            data={"travel_seconds": 45},
        )
        self.verify = RallyJoinedVerification(True, rally_id="rally-1", data={"joined_badge": True})

    def open_alliance_war(self, _request, _character, _policy):
        self.calls.append("open_alliance_war")
        return self.open_result

    def inspect_rallies(self, _request, _character, _policy):
        self.calls.append("inspect_rallies")
        return AllianceRallyScan(
            rallies=self.rallies,
            message=self.scan_message,
            screenshot_path=self.scan_screenshot,
        )

    def select_rally(self, _request, _character, rally, _policy):
        self.calls.append(f"select_rally:{rally.rally_id}")
        return self.select_result

    def verify_march_availability(self, _request, _character, rally, _policy):
        self.calls.append(f"verify_march:{rally.rally_id}")
        return self.march

    def verify_troop_availability(self, _request, _character, rally, _policy):
        self.calls.append(f"verify_troops:{rally.rally_id}")
        return self.troops

    def join_rally(self, _request, _character, rally, _policy):
        self.calls.append(f"join_rally:{rally.rally_id}")
        return self.join

    def choose_march_preset(self, _request, _character, rally, policy):
        self.calls.append(f"choose_preset:{rally.rally_id}:{policy.march_preset}")
        return self.preset

    def dispatch_march(self, _request, _character, rally, availability, policy):
        self.calls.append(f"dispatch_march:{rally.rally_id}:{availability.march_slot}:{policy.march_preset}")
        return self.dispatch

    def verify_joined(self, _request, _character, rally, dispatch, _policy):
        self.calls.append(f"verify_joined:{rally.rally_id}:{dispatch.dispatch_id}")
        return self.verify


class FakeWatchdog:
    def __init__(self, *, healthy: bool = True) -> None:
        self.healthy = healthy

    def monitor(
        self,
        *,
        instance_id: int,
        instance_index: int,
        instance_name: str,
        job_run_id: int | None = None,
    ) -> object:
        return SimpleNamespace(
            healthy=self.healthy,
            recovery_attempted=not self.healthy,
            circuit_opened=not self.healthy,
            observation=SimpleNamespace(
                message="unhealthy" if not self.healthy else "",
                screenshot_path="runtime/screens/unhealthy.png" if not self.healthy else "",
            ),
        )


class AllianceRallyJoinWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "alliance-rally.sqlite3")
        self.db.initialize()
        self.instances = InstanceRepository(self.db)
        self.characters = CharacterRepository(self.db)
        self.marches = MarchRepository(self.db)
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
        self.driver = FakeAllianceRallyDriver()
        self.watchdog = FakeWatchdog()

    def tearDown(self) -> None:
        self.db.close()
        self.temp_dir.cleanup()

    def _job(self, key: str) -> int:
        return self.jobs.save(
            Job(
                idempotency_key=key,
                job_type="workflow",
                scheduled_for="2026-07-09T00:00:00",
            )
        )

    def _workflow(self) -> AllianceRallyJoinWorkflow:
        return AllianceRallyJoinWorkflow(
            characters=self.characters,
            marches=self.marches,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=AllianceRallyJoinConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
            ),
        )

    def _policy(self) -> AllianceRallyJoinPolicy:
        return AllianceRallyJoinPolicy(
            allowed_rally_types=("barbarian_fort",),
            allowed_leaders=("Alice",),
            allowed_targets=("Fort 12",),
            minimum_duration_seconds=300,
            maximum_duration_seconds=900,
            march_preset="rally-cavalry",
            minimum_remaining_capacity=1,
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "alliance-rally-run",
        policy: AllianceRallyJoinPolicy | None = None,
    ) -> AllianceRallyJoinRequest:
        return AllianceRallyJoinRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or self._policy(),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states_and_template_keys(self) -> None:
        self.assertEqual(ALLIANCE_RALLY_JOIN_STATES, self._workflow().workflow_states)
        self.assertIn("alliance.war.rally_list", ALLIANCE_RALLY_JOIN_TEMPLATE_KEYS)
        self.assertIn("alliance.war.joined_rally_indicator", ALLIANCE_RALLY_JOIN_TEMPLATE_KEYS)

    def test_allowed_rally_join_persists_metadata_and_march(self) -> None:
        result = self._workflow().execute(
            self._request(job_id=self._job("rally-success"), run_key="rally-success")
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual("rally-1", result.result["selected_rally"]["rally_id"])
        self.assertEqual("ALLOWED", result.result["policy_decisions"][0]["code"])
        self.assertEqual("rally-cavalry", result.result["march_preset"])
        self.assertEqual("rally-dispatch-1", result.result["dispatch_result"]["dispatch_id"])
        self.assertTrue(result.result["joined_verification"]["joined"])
        self.assertIn("verify_joined:rally-1:rally-dispatch-1", self.driver.calls)
        marches = self.marches.list_for_character(self.character_id)
        dispatched = next(item for item in marches if item.march_slot == 3)
        self.assertEqual("alliance_rally_joined", dispatched.status)
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual("rally-1", payload["result"]["selected_rally"]["rally_id"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_denied_rally_type_returns_skipped_without_selecting(self) -> None:
        policy = AllianceRallyJoinPolicy(allowed_rally_types=("ark",), march_preset="rally-cavalry")

        result = self._workflow().execute(self._request(run_key="denied-type", policy=policy))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("DENIED_TYPE", result.result["policy_decisions"][0]["code"])
        self.assertNotIn("select_rally:rally-1", self.driver.calls)

    def test_denied_rally_leader_returns_skipped_without_selecting(self) -> None:
        policy = AllianceRallyJoinPolicy(allowed_leaders=("Bob",), march_preset="rally-cavalry")

        result = self._workflow().execute(self._request(run_key="denied-leader", policy=policy))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("DENIED_LEADER", result.result["policy_decisions"][0]["code"])
        self.assertNotIn("join_rally:rally-1", self.driver.calls)

    def test_denied_rally_target_returns_skipped_without_selecting(self) -> None:
        policy = AllianceRallyJoinPolicy(allowed_targets=("Fort 99",), march_preset="rally-cavalry")

        result = self._workflow().execute(self._request(run_key="denied-target", policy=policy))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("DENIED_TARGET", result.result["policy_decisions"][0]["code"])
        self.assertNotIn("select_rally:rally-1", self.driver.calls)

    def test_rally_duration_below_policy_returns_skipped(self) -> None:
        policy = AllianceRallyJoinPolicy(minimum_duration_seconds=700, march_preset="rally-cavalry")

        result = self._workflow().execute(self._request(run_key="duration-low", policy=policy))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("DURATION_BELOW_MINIMUM", result.result["policy_decisions"][0]["code"])
        self.assertNotIn("select_rally:rally-1", self.driver.calls)

    def test_rally_duration_above_policy_returns_skipped(self) -> None:
        policy = AllianceRallyJoinPolicy(maximum_duration_seconds=500, march_preset="rally-cavalry")

        result = self._workflow().execute(self._request(run_key="duration-high", policy=policy))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("DURATION_ABOVE_MAXIMUM", result.result["policy_decisions"][0]["code"])
        self.assertNotIn("select_rally:rally-1", self.driver.calls)

    def test_full_rally_returns_skipped_as_no_eligible_rally(self) -> None:
        self.driver.rallies = (
            AllianceRallyObservation(
                rally_id="full-rally",
                rally_type="barbarian_fort",
                leader_name="Alice",
                target_name="Fort 12",
                duration_seconds=600,
                capacity_available=False,
                remaining_capacity=0,
                screenshot_path="runtime/screens/full-rally.png",
            ),
        )

        result = self._workflow().execute(self._request(run_key="full-rally"))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("NO_CAPACITY", result.result["policy_decisions"][0]["code"])
        self.assertNotIn("select_rally:full-rally", self.driver.calls)

    def test_capacity_unavailable_after_selection_returns_blocked(self) -> None:
        self.driver.join = ResourceGatheringActionResult(
            False,
            message="Rally capacity is unavailable.",
            retryable=False,
            screenshot_path="runtime/screens/rally-full.png",
            data={"capacity_available": False},
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("capacity-blocked"), run_key="capacity-blocked")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("join_rally", result.result["terminal_state"])
        self.assertEqual("Rally capacity is unavailable.", result.result["failure_evidence"]["terminal_reason"])
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("failed", run.status)  # type: ignore[union-attr]
        self.assertEqual("runtime/screens/rally-full.png", run.screenshot_path)  # type: ignore[union-attr]

    def test_no_free_march_returns_blocked_before_join(self) -> None:
        self.driver.march = MarchAvailability(
            False,
            available_count=0,
            message="No free march.",
            screenshot_path="runtime/screens/no-march.png",
        )

        result = self._workflow().execute(self._request(run_key="no-march"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("verify_march", result.result["terminal_state"])
        self.assertNotIn("join_rally:rally-1", self.driver.calls)

    def test_no_troops_available_returns_blocked_before_join(self) -> None:
        self.driver.troops = TroopAvailability(
            False,
            troop_count=0,
            message="No available troops.",
            screenshot_path="runtime/screens/no-troops.png",
        )

        result = self._workflow().execute(self._request(run_key="no-troops"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("verify_troops", result.result["terminal_state"])
        self.assertNotIn("join_rally:rally-1", self.driver.calls)

    def test_no_eligible_rally_returns_skipped(self) -> None:
        self.driver.rallies = ()
        self.driver.scan_message = "No rallies are visible."

        result = self._workflow().execute(self._request(run_key="no-rally"))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("inspect_rallies", result.result["terminal_state"])
        self.assertNotIn("select_rally:rally-1", self.driver.calls)

    def test_successful_join_requires_joined_verification(self) -> None:
        result = self._workflow().execute(self._request(run_key="verify-success"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertTrue(result.result["joined_verification"]["joined"])
        self.assertIn("verify_joined:rally-1:rally-dispatch-1", self.driver.calls)

    def test_joined_verification_failure_persists_failure_evidence(self) -> None:
        self.driver.verify = RallyJoinedVerification(
            False,
            rally_id="rally-1",
            message="Joined indicator not visible.",
            retryable=False,
            screenshot_path="runtime/screens/join-verify-failed.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("verify-failure"), run_key="verify-failure")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("verify_joined", result.result["terminal_state"])
        self.assertEqual("Joined indicator not visible.", result.result["failure_evidence"]["terminal_reason"])
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual("verify_joined", payload["result"]["failure_evidence"]["terminal_state"])
        self.assertEqual("runtime/screens/join-verify-failed.png", run.screenshot_path)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
