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
from rok_assistant.tasks.alliance_pit_workflow import (  # noqa: E402
    ALLIANCE_PIT_STATES,
    ALLIANCE_PIT_TEMPLATE_KEYS,
    AlliancePitConfig,
    AlliancePitGatheringWorkflow,
    AlliancePitObservation,
    AlliancePitPolicy,
    AlliancePitRequest,
    AlliancePitStatus,
)
from rok_assistant.tasks.resource_search_workflow import (  # noqa: E402
    MarchAvailability,
    MarchDispatchResult,
    ResourceGatheringActionResult,
    ResourceType,
)
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


class FakeAlliancePitDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.navigation = ResourceGatheringActionResult(True)
        self.clear = ResourceGatheringActionResult(True, data={"overlays_dismissed": ["reward"]})
        self.pit = AlliancePitObservation(
            AlliancePitStatus.JOINABLE,
            resource_type=ResourceType.GOLD,
            pit_id="pit-1",
            pit_level=5,
            confidence=0.94,
            data={"template_key": "alliance.resource_center.joinable_pit"},
        )
        self.available = MarchAvailability(True, march_slot=2, available_count=1)
        self.dispatch = MarchDispatchResult(
            True,
            march_slot=2,
            dispatch_id="alliance-dispatch-1",
            expected_return_time="2026-07-07T02:00:00",
            data={"distance_seconds": 90},
        )
        self.verify = ResourceGatheringActionResult(True, data={"verified": True})

    def navigate_to_resource_center(self, _request, _character):
        self.calls.append("navigate_to_resource_center")
        return self.navigation

    def clear_overlays(self, _request, _character):
        self.calls.append("clear_overlays")
        return self.clear

    def detect_pit(self, _request, policy):
        self.calls.append(f"detect_pit:{','.join(item.value for item in policy.enabled_resource_types)}")
        return self.pit

    def validate_march_availability(self, _request, _pit, policy):
        self.calls.append(f"validate_march:{policy.march_preset}")
        return self.available

    def dispatch_gather_march(self, _request, _pit, availability, policy):
        self.calls.append(f"dispatch_march:{availability.march_slot}:{policy.march_preset}")
        return self.dispatch

    def verify_dispatch(self, _request, _pit, dispatch):
        self.calls.append(f"verify_dispatch:{dispatch.dispatch_id}")
        return self.verify


class FakeWatchdog:
    def __init__(self, *, healthy: bool = True) -> None:
        self.healthy = healthy
        self.calls: list[dict[str, object]] = []

    def monitor(
        self,
        *,
        instance_id: int,
        instance_index: int,
        instance_name: str,
        job_run_id: int | None = None,
    ) -> object:
        self.calls.append(
            {
                "instance_id": instance_id,
                "instance_index": instance_index,
                "instance_name": instance_name,
                "job_run_id": job_run_id,
            }
        )
        return SimpleNamespace(
            healthy=self.healthy,
            recovery_attempted=not self.healthy,
            circuit_opened=not self.healthy,
            observation=SimpleNamespace(
                message="unhealthy" if not self.healthy else "",
                screenshot_path="runtime/screens/unhealthy.png" if not self.healthy else "",
            ),
        )


class AlliancePitGatheringWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "alliance-pit.sqlite3")
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
        self.driver = FakeAlliancePitDriver()
        self.watchdog = FakeWatchdog()

    def tearDown(self) -> None:
        self.db.close()
        self.temp_dir.cleanup()

    def _job(self, key: str) -> int:
        return self.jobs.save(
            Job(
                idempotency_key=key,
                job_type="workflow",
                scheduled_for="2026-07-07T00:00:00",
            )
        )

    def _workflow(self) -> AlliancePitGatheringWorkflow:
        return AlliancePitGatheringWorkflow(
            characters=self.characters,
            marches=self.marches,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=AlliancePitConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "alliance-pit-run",
        policy: AlliancePitPolicy | None = None,
    ) -> AlliancePitRequest:
        return AlliancePitRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or AlliancePitPolicy(march_preset="cavalry"),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states_and_template_keys(self) -> None:
        self.assertEqual(ALLIANCE_PIT_STATES, self._workflow().workflow_states)
        self.assertIn("alliance.resource_center.joinable_pit", ALLIANCE_PIT_TEMPLATE_KEYS)
        self.assertIn("alliance.resource_center.reward_popup", ALLIANCE_PIT_TEMPLATE_KEYS)

    def test_dispatch_success_persists_pit_and_march_metadata(self) -> None:
        result = self._workflow().execute(
            self._request(job_id=self._job("alliance-pit-success"), run_key="alliance-pit-success")
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("clear_overlays", self.driver.calls)
        self.assertEqual("pit-1", result.result["selected_pit"]["pit_id"])
        self.assertEqual("alliance-dispatch-1", result.result["march_dispatch"]["dispatch_id"])
        self.assertEqual("cavalry", result.result["march_preset"])
        marches = self.marches.list_for_character(self.character_id)
        dispatched = next(item for item in marches if item.march_slot == 2)
        self.assertEqual("alliance_pit_gathering", dispatched.status)
        self.assertEqual("2026-07-07T02:00:00", dispatched.expected_return_time)
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual("GOLD", payload["result"]["selected_pit"]["resource_type"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_no_active_pit_returns_skipped_and_does_not_dispatch(self) -> None:
        self.driver.pit = AlliancePitObservation(
            AlliancePitStatus.NO_ACTIVE_PIT,
            message="No alliance resource pit is active.",
            screenshot_path="runtime/screens/no-pit.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("alliance-pit-none"), run_key="alliance-pit-none")
        )

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("detect_pit", result.result["terminal_state"])
        self.assertNotIn("validate_march:cavalry", self.driver.calls)
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("completed", run.status)  # type: ignore[union-attr]
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_already_participating_returns_skipped(self) -> None:
        self.driver.pit = AlliancePitObservation(
            AlliancePitStatus.ALREADY_PARTICIPATING,
            resource_type=ResourceType.WOOD,
            pit_id="pit-existing",
            message="Already gathering in alliance pit.",
        )

        result = self._workflow().execute(self._request(run_key="alliance-pit-already"))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("Already gathering in alliance pit.", result.message)
        self.assertEqual("pit-existing", result.result["selected_pit"]["pit_id"])
        self.assertNotIn("dispatch_march:2:cavalry", self.driver.calls)

    def test_pit_not_joinable_returns_blocked_with_recovery_outcome(self) -> None:
        self.driver.pit = AlliancePitObservation(
            AlliancePitStatus.NOT_JOINABLE,
            resource_type=ResourceType.STONE,
            pit_id="pit-closed",
            message="Alliance pit is full.",
            screenshot_path="runtime/screens/pit-full.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("alliance-pit-full"), run_key="alliance-pit-full")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("Alliance pit is full.", result.result["terminal_reason"])
        self.assertEqual({"attempted": False, "healthy": True, "circuit_opened": False}, result.result["recovery_outcome"])
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("failed", run.status)  # type: ignore[union-attr]
        self.assertEqual("Alliance pit is full.", run.error_message)  # type: ignore[union-attr]

    def test_no_free_march_returns_blocked_before_dispatch(self) -> None:
        self.driver.available = MarchAvailability(
            False,
            available_count=0,
            message="No free march.",
            screenshot_path="runtime/screens/no-free-march.png",
        )

        result = self._workflow().execute(self._request(run_key="alliance-pit-no-march"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("validate_march", result.result["terminal_state"])
        self.assertNotIn("dispatch_march:2:cavalry", self.driver.calls)

    def test_policy_disabled_resource_returns_skipped(self) -> None:
        self.driver.pit = AlliancePitObservation(
            AlliancePitStatus.JOINABLE,
            resource_type=ResourceType.GOLD,
            pit_id="gold-pit",
        )
        policy = AlliancePitPolicy(
            enabled_resource_types=(ResourceType.FOOD, ResourceType.WOOD),
            march_preset="infantry",
        )

        result = self._workflow().execute(
            self._request(run_key="alliance-pit-disabled-resource", policy=policy)
        )

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("detect_pit", result.result["terminal_state"])
        self.assertIn("disabled by policy", result.result["terminal_reason"])
        self.assertNotIn("validate_march:infantry", self.driver.calls)

    def test_dispatch_failure_returns_blocked_and_persists_failure_reason(self) -> None:
        self.driver.dispatch = MarchDispatchResult(
            False,
            march_slot=2,
            message="Dispatch button was disabled.",
            retryable=False,
            screenshot_path="runtime/screens/dispatch-disabled.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("alliance-pit-dispatch-blocked"), run_key="alliance-pit-dispatch-blocked")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("dispatch_march", result.result["terminal_state"])
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual("Dispatch button was disabled.", payload["message"])
        self.assertEqual("Dispatch button was disabled.", payload["result"]["terminal_reason"])


if __name__ == "__main__":
    unittest.main()
