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
from rok_assistant.tasks.gem_gathering_workflow import (  # noqa: E402
    GEM_GATHERING_STATES,
    GemDepositObservation,
    GemDepositStatus,
    GemDetectionDatasetCase,
    GemGatheringConfig,
    GemGatheringRequest,
    GemGatheringWorkflow,
    GemSearchPolicy,
    evaluate_gem_detection_dataset,
)
from rok_assistant.tasks.resource_search_workflow import (  # noqa: E402
    MarchAvailability,
    MarchDispatchResult,
    ResourceGatheringActionResult,
)
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


class FakeClock:
    def __init__(self, values: tuple[float, ...] = (0.0,)) -> None:
        self.values = list(values)
        self.last = values[-1] if values else 0.0

    def __call__(self) -> float:
        if self.values:
            self.last = self.values.pop(0)
        return self.last


class FakeGemDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.observations: list[GemDepositObservation] = [
            GemDepositObservation(GemDepositStatus.AVAILABLE, level=3, confidence=0.91, x=100, y=200)
        ]
        self.validated = GemDepositObservation(GemDepositStatus.AVAILABLE, level=3, confidence=0.91, x=100, y=200)
        self.available = MarchAvailability(True, march_slot=1, available_count=1)
        self.dispatch = MarchDispatchResult(
            True,
            march_slot=1,
            dispatch_id="gem-dispatch-1",
            expected_return_time="2026-07-07T03:00:00",
        )
        self.verify = ResourceGatheringActionResult(True, data={"marching": True})

    def search_for_gem(self, _request, policy, attempt):
        self.calls.append(f"search_for_gem:{attempt}:{policy.march_preset}")
        if self.observations:
            return self.observations.pop(0)
        return GemDepositObservation(GemDepositStatus.NOT_FOUND, message="not found")

    def validate_gem_node_available(self, _request, deposit, policy):
        self.calls.append(f"validate_node:{deposit.level}:{policy.minimum_detector_confidence}")
        return self.validated

    def validate_march_availability(self, _request, deposit, policy):
        self.calls.append(f"validate_march:{deposit.level}:{policy.march_preset}")
        return self.available

    def dispatch_gem_gather_march(self, _request, deposit, availability, policy):
        self.calls.append(f"dispatch_march:{deposit.level}:{availability.march_slot}:{policy.march_preset}")
        return self.dispatch

    def verify_dispatch(self, _request, deposit, dispatch):
        self.calls.append(f"verify_dispatch:{deposit.level}:{dispatch.dispatch_id}")
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


class GemGatheringWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "gem.sqlite3")
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
        self.driver = FakeGemDriver()
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

    def _workflow(self, *, clock: FakeClock | None = None) -> GemGatheringWorkflow:
        return GemGatheringWorkflow(
            characters=self.characters,
            marches=self.marches,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=GemGatheringConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
            ),
            clock=clock or FakeClock(),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "gem-run",
        policy: GemSearchPolicy | None = None,
    ) -> GemGatheringRequest:
        return GemGatheringRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or GemSearchPolicy(
                allowed_levels=(2, 3, 4),
                attempt_limit=3,
                total_deadline_seconds=20,
                march_preset="cavalry",
                minimum_detector_confidence=0.85,
            ),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states(self) -> None:
        self.assertEqual(GEM_GATHERING_STATES, self._workflow().workflow_states)

    def test_precision_recall_thresholds_from_labeled_dataset(self) -> None:
        policy = GemSearchPolicy(allowed_levels=(1, 2, 3), minimum_detector_confidence=0.8)
        cases = (
            GemDetectionDatasetCase("positive-1", True, GemDepositObservation(GemDepositStatus.AVAILABLE, level=2, confidence=0.91)),
            GemDetectionDatasetCase("positive-2", True, GemDepositObservation(GemDepositStatus.AVAILABLE, level=3, confidence=0.85)),
            GemDetectionDatasetCase("negative-low", False, GemDepositObservation(GemDepositStatus.AVAILABLE, level=2, confidence=0.79)),
            GemDetectionDatasetCase("negative-occupied", False, GemDepositObservation(GemDepositStatus.OCCUPIED, level=2, confidence=0.95)),
        )

        metrics = evaluate_gem_detection_dataset(cases, policy)

        self.assertEqual(2, metrics.true_positives)
        self.assertEqual(2, metrics.true_negatives)
        self.assertTrue(metrics.meets(minimum_precision=1.0, minimum_recall=1.0))

    def test_successful_dispatch_persists_gem_march_and_attempt_metadata(self) -> None:
        job_id = self._job("gem-success")

        result = self._workflow().execute(
            self._request(job_id=job_id, run_key="gem-success")
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual("gem-dispatch-1", result.result["march_dispatch"]["dispatch_id"])
        self.assertEqual(0.91, result.result["search_attempts"][0]["confidence"])
        self.assertIn("validate_march:3:cavalry", self.driver.calls)
        marches = self.marches.list_for_character(self.character_id)
        self.assertEqual("gem_gathering", marches[0].status)
        self.assertEqual("2026-07-07T03:00:00", marches[0].expected_return_time)
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual("AVAILABLE", payload["result"]["selected_gem"]["status"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_below_threshold_detection_does_not_validate_or_click(self) -> None:
        self.driver.observations = [
            GemDepositObservation(GemDepositStatus.AVAILABLE, level=3, confidence=0.60, x=100, y=200),
        ]
        policy = GemSearchPolicy(
            allowed_levels=(3,),
            attempt_limit=1,
            minimum_detector_confidence=0.85,
            total_deadline_seconds=20,
        )

        result = self._workflow().execute(
            self._request(run_key="gem-below-threshold", policy=policy)
        )

        self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
        self.assertEqual("search_gem", result.result["failure_state"])
        self.assertEqual(0.60, result.result["search_attempts"][0]["confidence"])
        self.assertFalse(any(call.startswith("validate_node") for call in self.driver.calls))
        self.assertFalse(any(call.startswith("dispatch_march") for call in self.driver.calls))

    def test_no_node_timeout_stops_after_deadline_bound(self) -> None:
        self.driver.observations = [
            GemDepositObservation(GemDepositStatus.NOT_FOUND, message="scan empty"),
            GemDepositObservation(GemDepositStatus.AVAILABLE, level=3, confidence=0.95),
        ]
        policy = GemSearchPolicy(
            allowed_levels=(3,),
            attempt_limit=5,
            minimum_detector_confidence=0.85,
            total_deadline_seconds=0.1,
        )
        clock = FakeClock((0.0, 0.0, 0.2, 0.2))

        result = self._workflow(clock=clock).execute(
            self._request(run_key="gem-timeout", policy=policy)
        )

        self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
        self.assertEqual("search_gem", result.result["failure_state"])
        self.assertEqual(1, len(result.result["search_attempts"]))
        self.assertFalse(any(call.startswith("validate_node") for call in self.driver.calls))

    def test_occupied_node_blocks_before_march_dispatch(self) -> None:
        self.driver.validated = GemDepositObservation(
            GemDepositStatus.OCCUPIED,
            level=3,
            confidence=0.92,
            message="Gem node is already occupied.",
            screenshot_path="runtime/screens/gem-occupied.png",
        )

        result = self._workflow().execute(self._request(run_key="gem-occupied"))

        self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
        self.assertEqual("validate_node", result.result["failure_state"])
        self.assertEqual("OCCUPIED", result.result["validated_gem"]["status"])
        self.assertFalse(any(call.startswith("dispatch_march") for call in self.driver.calls))

    def test_verification_screen_hard_stop_skips_recovery_and_dispatch(self) -> None:
        self.driver.observations = [
            GemDepositObservation(
                GemDepositStatus.VERIFICATION_REQUIRED,
                message="Verification screen requires manual intervention.",
                screenshot_path="runtime/screens/verification.png",
            )
        ]

        result = self._workflow().execute(self._request(run_key="gem-verification"))

        self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
        self.assertEqual("search_gem", result.result["failure_state"])
        self.assertEqual(
            {"attempted": False, "reason": "verification_screen"},
            result.result["recovery_outcome"],
        )
        self.assertFalse(any(call.startswith("validate_node") for call in self.driver.calls))
        self.assertFalse(any(call.startswith("dispatch_march") for call in self.driver.calls))


if __name__ == "__main__":
    unittest.main()
