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
from rok_assistant.tasks.alliance_resource_collection_workflow import (  # noqa: E402
    ALLIANCE_RESOURCE_COLLECTION_STATES,
    ALLIANCE_RESOURCE_COLLECTION_TEMPLATE_KEYS,
    AllianceResourceClaimResult,
    AllianceResourceCollectionConfig,
    AllianceResourceCollectionPolicy,
    AllianceResourceCollectionRequest,
    AllianceResourceCollectionWorkflow,
    AllianceResourceObservation,
    AllianceResourceScan,
    AllianceResourceScanStatus,
)
from rok_assistant.tasks.resource_search_workflow import ResourceGatheringActionResult  # noqa: E402
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


def _resource(
    key: str,
    *,
    count: int = 1,
    resource_id: str = "",
    confidence: float = 0.94,
) -> AllianceResourceObservation:
    return AllianceResourceObservation(
        key,
        claimable_count=count,
        confidence=confidence,
        resource_id=resource_id or key.lower(),
    )


class FakeAllianceResourceDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.open_alliance_result = ResourceGatheringActionResult(True, data={"scene": "ALLIANCE_HOME"})
        self.open_resources = ResourceGatheringActionResult(True, data={"scene": "ALLIANCE_RESOURCES"})
        self.scan = AllianceResourceScan(
            AllianceResourceScanStatus.READY,
            observations=(_resource("FOOD", count=1, resource_id="food-1"),),
        )
        self.claim = AllianceResourceClaimResult(True, changed=True, claimed_count=1)
        self.close = AllianceResourceClaimResult(True, changed=True, data={"overlay_closed": True})
        self.verify = AllianceResourceClaimResult(
            True,
            changed=True,
            claimed_count=1,
            claimable_remaining=False,
            data={"badge_cleared": True},
        )

    def open_alliance(self, _request, _character, _policy):
        self.calls.append("open_alliance")
        return self.open_alliance_result

    def open_alliance_resources(self, _request, _character, _policy):
        self.calls.append("open_alliance_resources")
        return self.open_resources

    def scan_claimable_resources(self, _request, _character, _policy):
        self.calls.append("scan_claimable_resources")
        return self.scan

    def claim_alliance_resource(self, _request, _character, observation, _policy):
        self.calls.append(f"claim:{observation.normalized_resource_key()}:{observation.resource_id}")
        return self.claim

    def close_reward_overlay(self, _request, _character, observation, _claim_result, _policy):
        self.calls.append(f"close_overlay:{observation.normalized_resource_key()}:{observation.resource_id}")
        return self.close

    def verify_alliance_resource_claimed(self, _request, _character, observation, _policy):
        self.calls.append(f"verify:{observation.normalized_resource_key()}:{observation.resource_id}")
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


class AllianceResourceCollectionWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "alliance-resource.sqlite3")
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
        self.driver = FakeAllianceResourceDriver()
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

    def _workflow(self) -> AllianceResourceCollectionWorkflow:
        return AllianceResourceCollectionWorkflow(
            characters=self.characters,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=AllianceResourceCollectionConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "alliance-resource-run",
        policy: AllianceResourceCollectionPolicy | None = None,
    ) -> AllianceResourceCollectionRequest:
        return AllianceResourceCollectionRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or AllianceResourceCollectionPolicy(),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states_and_template_keys(self) -> None:
        self.assertEqual(ALLIANCE_RESOURCE_COLLECTION_STATES, self._workflow().workflow_states)
        self.assertIn("alliance.resources.claimable_badge", ALLIANCE_RESOURCE_COLLECTION_TEMPLATE_KEYS)
        self.assertIn("alliance.resources.reward_overlay", ALLIANCE_RESOURCE_COLLECTION_TEMPLATE_KEYS)

    def test_no_claimable_alliance_resources_returns_skipped(self) -> None:
        job_id = self._job("alliance-resource-none")
        self.driver.scan = AllianceResourceScan(
            AllianceResourceScanStatus.NONE_CLAIMABLE,
            message="No alliance resources to claim.",
            screenshot_path="runtime/screens/alliance-none.png",
        )

        result = self._workflow().execute(self._request(job_id=job_id, run_key="alliance-resource-none"))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("scan_claimable_resources", result.result["terminal_state"])
        self.assertEqual("No alliance resources to claim.", result.result["skipped_reason"])
        self.assertFalse(any(call.startswith("claim:") for call in self.driver.calls))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("completed", run.status)  # type: ignore[union-attr]
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_one_claimable_resource_is_claimed_and_persisted(self) -> None:
        job_id = self._job("alliance-resource-one")

        result = self._workflow().execute(self._request(job_id=job_id, run_key="alliance-resource-one"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(1, result.result["claimed_count"])
        self.assertEqual(
            [
                "open_alliance",
                "open_alliance_resources",
                "scan_claimable_resources",
                "claim:FOOD:food-1",
                "verify:FOOD:food-1",
            ],
            self.driver.calls,
        )
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual(1, payload["result"]["claimed_count"])
        self.assertEqual("FOOD", payload["result"]["claim_attempts"][0]["resource_key"])

    def test_multiple_claimable_resources_are_claimed(self) -> None:
        self.driver.scan = AllianceResourceScan(
            AllianceResourceScanStatus.READY,
            observations=(
                _resource("FOOD", count=1, resource_id="food-1"),
                _resource("WOOD", count=2, resource_id="wood-1"),
                _resource("GOLD", count=0, resource_id="gold-empty"),
            ),
        )
        self.driver.claim = AllianceResourceClaimResult(True, changed=True, claimed_count=0)
        self.driver.verify = AllianceResourceClaimResult(True, changed=True, claimed_count=0, claimable_remaining=False)

        result = self._workflow().execute(self._request(run_key="alliance-resource-many"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(3, result.result["claimed_count"])
        self.assertIn("claim:FOOD:food-1", self.driver.calls)
        self.assertIn("claim:WOOD:wood-1", self.driver.calls)
        self.assertNotIn("claim:GOLD:gold-empty", self.driver.calls)
        self.assertEqual("not_claimable", result.result["ignored_resources"][0]["ignored_reason"])

    def test_reward_overlay_is_closed_after_claim(self) -> None:
        self.driver.claim = AllianceResourceClaimResult(
            True,
            changed=True,
            claimed_count=1,
            reward_overlay_present=True,
        )

        result = self._workflow().execute(self._request(run_key="alliance-resource-overlay"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("close_overlay:FOOD:food-1", self.driver.calls)
        self.assertTrue(result.result["overlay_attempts"][0]["overlay_present"])
        self.assertTrue(result.result["overlay_attempts"][0]["closed"])

    def test_postcondition_requires_claimable_state_gone(self) -> None:
        self.driver.verify = AllianceResourceClaimResult(
            True,
            changed=False,
            claimed_count=0,
            claimable_remaining=True,
            message="Badge stayed visible.",
            retryable=False,
            screenshot_path="runtime/screens/alliance-badge-stuck.png",
        )

        result = self._workflow().execute(self._request(run_key="alliance-resource-stuck"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("claim_resource", result.result["terminal_state"])
        self.assertIn("claimable state remained", result.result["terminal_reason"])
        self.assertEqual(0, result.result["claimed_count"])

    def test_navigation_failure_triggers_bounded_recovery(self) -> None:
        self.driver.open_resources = ResourceGatheringActionResult(
            False,
            message="Alliance resources tab did not open.",
            retryable=False,
            screenshot_path="runtime/screens/alliance-nav-failed.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("alliance-resource-nav-failed"), run_key="alliance-resource-nav-failed")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("open_alliance_resources", result.result["terminal_state"])
        self.assertEqual(
            {"attempted": False, "healthy": True, "circuit_opened": False},
            result.result["recovery_outcome"],
        )
        self.assertEqual([None, result.job_run_id], self.watchdog.calls)

    def test_blocked_verification_records_failure_evidence(self) -> None:
        self.driver.verify = AllianceResourceClaimResult(
            False,
            changed=False,
            claimable_remaining=True,
            message="Claim button remained active.",
            retryable=False,
            screenshot_path="runtime/screens/alliance-claim-still-active.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("alliance-resource-evidence"), run_key="alliance-resource-evidence")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual(
            "runtime/screens/alliance-claim-still-active.png",
            result.result["failure_evidence"]["screenshot_path"],
        )
        self.assertEqual(1, len(self.incidents.list_open()))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("failed", run.status)  # type: ignore[union-attr]
        self.assertIn("Claim button remained active.", run.error_message)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
