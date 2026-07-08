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
    InstanceRepository,
    JobRepository,
    JobRunRepository,
    MarchRepository,
    StepRunRepository,
)
from rok_assistant.tasks.map_fog_exploration_workflow import (  # noqa: E402
    MAP_FOG_EXPLORATION_STATES,
    FogExplorationConfig,
    FogExplorationRequest,
    FogExplorationWorkflow,
    FogScoutPolicy,
    FogTargetObservation,
    FogTargetStatus,
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


class FakeFogDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.scout = MarchAvailability(True, march_slot=3, available_count=1)
        self.targets = [
            FogTargetObservation(
                FogTargetStatus.FOG,
                target_id="fog-1",
                confidence=0.91,
                x=100,
                y=200,
                data={"sector": "north"},
            )
        ]
        self.investigation = ResourceGatheringActionResult(True, data={"investigated": True})
        self.dispatch = MarchDispatchResult(
            True,
            march_slot=3,
            dispatch_id="fog-dispatch-1",
            expected_return_time="2026-07-09T02:00:00",
            data={"eta_seconds": 180},
        )
        self.verify = ResourceGatheringActionResult(True, data={"scout_busy": True})

    def validate_idle_scout(self, _request, policy):
        self.calls.append(f"validate_idle_scout:{policy.scout_preset}")
        return self.scout

    def scan_for_fog_target(self, _request, policy, scan_index):
        self.calls.append(f"scan_for_fog_target:{scan_index}:{policy.scan_radius}")
        if self.targets:
            return self.targets.pop(0)
        return FogTargetObservation(FogTargetStatus.NOT_FOUND, message="empty scan")

    def investigate_discovery(self, _request, target, policy):
        self.calls.append(f"investigate_discovery:{target.normalized_status().value}:{policy.scout_preset}")
        return self.investigation

    def dispatch_scout(self, _request, target, scout, policy):
        self.calls.append(f"dispatch_scout:{target.target_id}:{scout.march_slot}:{policy.scout_preset}")
        return self.dispatch

    def verify_scout_busy(self, _request, target, dispatch):
        self.calls.append(f"verify_scout_busy:{target.target_id}:{dispatch.dispatch_id}")
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


class FogExplorationWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "fog.sqlite3")
        self.db.initialize()
        self.instances = InstanceRepository(self.db)
        self.characters = CharacterRepository(self.db)
        self.marches = MarchRepository(self.db)
        self.jobs = JobRepository(self.db)
        self.job_runs = JobRunRepository(self.db)
        self.step_runs = StepRunRepository(self.db)
        self.instance_id = self.instances.save(
            Instance(name="MEmu 1", instance_index=0, instance_name="MEmu 1")
        )
        self.character_id = self.characters.save(
            Character(id=None, name="Scout01", instance_id=self.instance_id)
        )
        self.driver = FakeFogDriver()
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

    def _workflow(self, *, clock: FakeClock | None = None) -> FogExplorationWorkflow:
        return FogExplorationWorkflow(
            characters=self.characters,
            marches=self.marches,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            config=FogExplorationConfig(
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
        run_key: str = "fog-run",
        policy: FogScoutPolicy | None = None,
    ) -> FogExplorationRequest:
        return FogExplorationRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or FogScoutPolicy(
                max_scans=3,
                total_deadline_seconds=20,
                scan_radius=4,
                scout_preset="fast",
                minimum_confidence=0.85,
            ),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states(self) -> None:
        self.assertEqual(MAP_FOG_EXPLORATION_STATES, self._workflow().workflow_states)

    def test_successful_fog_dispatch_persists_busy_scout_eta(self) -> None:
        result = self._workflow().execute(
            self._request(job_id=self._job("fog-success"), run_key="fog-success")
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual("fog-1", result.result["target"]["target_id"])
        self.assertEqual("fog-dispatch-1", result.result["scout_dispatch"]["dispatch_id"])
        self.assertEqual(0.91, result.result["scan_attempts"][0]["confidence"])
        self.assertIn("verify_scout_busy:fog-1:fog-dispatch-1", self.driver.calls)
        marches = self.marches.list_for_character(self.character_id)
        dispatched = next(item for item in marches if item.march_slot == 3)
        self.assertEqual("fog_exploration", dispatched.status)
        self.assertEqual("2026-07-09T02:00:00", dispatched.expected_return_time)
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual("FOG", payload["result"]["target"]["status"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_no_idle_scout_returns_blocked_before_scan(self) -> None:
        self.driver.scout = MarchAvailability(
            False,
            available_count=0,
            message="No idle scout.",
            screenshot_path="runtime/screens/no-scout.png",
        )

        result = self._workflow().execute(self._request(run_key="fog-no-scout"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("validate_idle_scout", result.result["terminal_state"])
        self.assertEqual("No idle scout.", result.result["terminal_reason"])
        self.assertFalse(any(call.startswith("scan_for_fog_target") for call in self.driver.calls))
        self.assertFalse(any(call.startswith("dispatch_scout") for call in self.driver.calls))

    def test_no_valid_target_returns_retryable_failure_with_scan_attempts(self) -> None:
        self.driver.targets = [
            FogTargetObservation(FogTargetStatus.NOT_FOUND, message="empty scan"),
            FogTargetObservation(FogTargetStatus.INVALID, confidence=0.95, message="not fog"),
        ]
        policy = FogScoutPolicy(max_scans=2, total_deadline_seconds=20, minimum_confidence=0.85)

        result = self._workflow().execute(
            self._request(run_key="fog-no-target", policy=policy)
        )

        self.assertEqual(WorkflowOutcome.RETRYABLE_FAILURE, result.outcome)
        self.assertEqual("scan_for_fog_target", result.result["terminal_state"])
        self.assertEqual(2, len(result.result["scan_attempts"]))
        self.assertFalse(any(call.startswith("dispatch_scout") for call in self.driver.calls))

    def test_whitelisted_cave_is_investigated_and_dispatched(self) -> None:
        self.driver.targets = [
            FogTargetObservation(FogTargetStatus.CAVE, target_id="cave-1", confidence=0.93)
        ]
        policy = FogScoutPolicy(
            max_scans=2,
            total_deadline_seconds=20,
            scout_preset="fast",
            minimum_confidence=0.85,
            investigate_caves=True,
        )

        result = self._workflow().execute(
            self._request(run_key="fog-cave", policy=policy)
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("investigate_discovery:CAVE:fast", self.driver.calls)
        self.assertIn("dispatch_scout:cave-1:3:fast", self.driver.calls)

    def test_non_whitelisted_village_exhausts_budget_without_dispatch(self) -> None:
        self.driver.targets = [
            FogTargetObservation(FogTargetStatus.VILLAGE, target_id="village-1", confidence=0.93)
        ]
        policy = FogScoutPolicy(
            max_scans=1,
            total_deadline_seconds=20,
            minimum_confidence=0.85,
            investigate_villages=False,
        )

        result = self._workflow().execute(
            self._request(run_key="fog-village-not-whitelisted", policy=policy)
        )

        self.assertEqual(WorkflowOutcome.RETRYABLE_FAILURE, result.outcome)
        self.assertEqual("VILLAGE", result.result["scan_attempts"][0]["status"])
        self.assertFalse(any(call.startswith("investigate_discovery") for call in self.driver.calls))
        self.assertFalse(any(call.startswith("dispatch_scout") for call in self.driver.calls))

    def test_deadline_exhaustion_stops_scan_loop(self) -> None:
        self.driver.targets = [
            FogTargetObservation(FogTargetStatus.NOT_FOUND, message="empty scan"),
            FogTargetObservation(FogTargetStatus.FOG, target_id="late-fog", confidence=0.95),
        ]
        policy = FogScoutPolicy(max_scans=5, total_deadline_seconds=0.1, minimum_confidence=0.85)
        clock = FakeClock((0.0, 0.0, 0.2, 0.2))

        result = self._workflow(clock=clock).execute(
            self._request(run_key="fog-deadline", policy=policy)
        )

        self.assertEqual(WorkflowOutcome.RETRYABLE_FAILURE, result.outcome)
        self.assertEqual(1, len(result.result["scan_attempts"]))
        self.assertFalse(any(call.startswith("dispatch_scout") for call in self.driver.calls))


if __name__ == "__main__":
    unittest.main()
