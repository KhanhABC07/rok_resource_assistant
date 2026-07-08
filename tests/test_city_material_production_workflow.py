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
from rok_assistant.tasks.city_material_production_workflow import (  # noqa: E402
    CITY_MATERIAL_PRODUCTION_STATES,
    CityMaterial,
    MaterialProductionConfig,
    MaterialProductionOption,
    MaterialProductionPolicy,
    MaterialProductionRequest,
    MaterialProductionStartResult,
    MaterialProductionWorkflow,
    MaterialQuality,
    MaterialQueueState,
    MaterialQueueStatus,
)
from rok_assistant.tasks.resource_search_workflow import ResourceGatheringActionResult  # noqa: E402
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


def _option(
    material: CityMaterial,
    *,
    quality: MaterialQuality = MaterialQuality.NORMAL,
    tier: int = 1,
    resources_available: bool = True,
    confidence: float = 0.94,
    material_verified: bool = True,
) -> MaterialProductionOption:
    return MaterialProductionOption(
        material,
        quality=quality,
        tier=tier,
        resources_available=resources_available,
        confidence=confidence,
        material_verified=material_verified,
    )


class FakeMaterialProductionDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.open_result = ResourceGatheringActionResult(True, data={"scene": "MATERIAL_PRODUCTION"})
        self.queue = MaterialQueueState(
            MaterialQueueStatus.IDLE,
            available_options=(
                _option(CityMaterial.LEATHER),
                _option(CityMaterial.IRON),
                _option(CityMaterial.EBONY),
                _option(CityMaterial.BONE),
            ),
            queue_size=0,
            cooldown_seconds=0,
            screenshot_path="runtime/screens/material-queue.png",
        )
        self.select_result = ResourceGatheringActionResult(True, data={"selected": True})
        self.start_result = MaterialProductionStartResult(True, changed=False, queue_size=0, cooldown_seconds=0)
        self.verify_result = MaterialProductionStartResult(
            True,
            changed=True,
            queue_size=1,
            cooldown_seconds=3600,
            screenshot_path="runtime/screens/material-started.png",
        )

    def open_material_production(self, _request, _character, _policy):
        self.calls.append("open_material_production")
        return self.open_result

    def inspect_material_queue(self, _request, _character, _policy):
        self.calls.append("inspect_material_queue")
        return self.queue

    def select_material(self, _request, _character, option, _policy):
        self.calls.append(
            f"select:{option.normalized_material().value}:{option.normalized_quality().value}:{option.tier}"
        )
        return self.select_result

    def start_material_production(self, _request, _character, option, before, _policy):
        del before
        self.calls.append(
            f"start:{option.normalized_material().value}:{option.normalized_quality().value}:{option.tier}"
        )
        return self.start_result

    def verify_material_production_state(self, _request, _character, option, before, start, _policy):
        del before, start
        self.calls.append(f"verify:{option.normalized_material().value}")
        return self.verify_result


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


class MaterialProductionWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "city-material.sqlite3")
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
        self.driver = FakeMaterialProductionDriver()
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

    def _workflow(self) -> MaterialProductionWorkflow:
        return MaterialProductionWorkflow(
            characters=self.characters,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=MaterialProductionConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "city-material-run",
        policy: MaterialProductionPolicy | None = None,
    ) -> MaterialProductionRequest:
        return MaterialProductionRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or MaterialProductionPolicy(),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states(self) -> None:
        self.assertEqual(CITY_MATERIAL_PRODUCTION_STATES, self._workflow().workflow_states)

    def test_free_queue_starts_production_and_persists_metadata(self) -> None:
        job_id = self._job("city-material-success")

        result = self._workflow().execute(
            self._request(job_id=job_id, run_key="city-material-success")
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual("LEATHER", result.result["selected_material"]["material"])
        self.assertEqual("NORMAL", result.result["selected_material"]["quality"])
        self.assertEqual(1, result.result["selected_material"]["tier"])
        self.assertEqual(
            [
                "open_material_production",
                "inspect_material_queue",
                "select:LEATHER:NORMAL:1",
                "start:LEATHER:NORMAL:1",
                "verify:LEATHER",
            ],
            self.driver.calls,
        )
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual("LEATHER", payload["result"]["selected_material"]["material"])
        self.assertEqual("IDLE", payload["result"]["queue_state"]["status"])
        self.assertTrue(payload["result"]["verification_result"]["changed"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_busy_queue_returns_skipped_when_overwrite_is_not_allowed(self) -> None:
        self.driver.queue = MaterialQueueState(
            MaterialQueueStatus.BUSY,
            active_material=CityMaterial.IRON,
            active_quality=MaterialQuality.NORMAL,
            active_tier=1,
            queue_size=1,
            cooldown_seconds=120,
            message="Queue already producing.",
            screenshot_path="runtime/screens/material-busy.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("city-material-busy"), run_key="city-material-busy")
        )

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("inspect_queue_state", result.result["terminal_state"])
        self.assertEqual("Queue already producing.", result.result["skipped_reason"])
        self.assertFalse(any(call.startswith("select:") for call in self.driver.calls))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("completed", run.status)  # type: ignore[union-attr]

    def test_insufficient_resources_returns_skipped(self) -> None:
        self.driver.queue = MaterialQueueState(
            MaterialQueueStatus.INSUFFICIENT_RESOURCES,
            available_options=(
                _option(CityMaterial.LEATHER, resources_available=False),
                _option(CityMaterial.IRON, resources_available=False),
            ),
            screenshot_path="runtime/screens/material-insufficient.png",
        )

        result = self._workflow().execute(self._request(run_key="city-material-insufficient"))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("select_material", result.result["terminal_state"])
        self.assertEqual("No allowed material can be produced.", result.result["skipped_reason"])
        self.assertEqual("insufficient_resources", result.result["ignored_options"][0]["ignored_reason"])

    def test_leather_selection(self) -> None:
        result = self._workflow().execute(self._request(run_key="city-material-leather"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("select:LEATHER:NORMAL:1", self.driver.calls)

    def test_iron_selection(self) -> None:
        policy = MaterialProductionPolicy(material_priority=(CityMaterial.IRON,))

        result = self._workflow().execute(self._request(run_key="city-material-iron", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("select:IRON:NORMAL:1", self.driver.calls)
        self.assertEqual("IRON", result.result["selected_material"]["material"])

    def test_ebony_selection(self) -> None:
        policy = MaterialProductionPolicy(material_priority=(CityMaterial.EBONY,))

        result = self._workflow().execute(self._request(run_key="city-material-ebony", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("select:EBONY:NORMAL:1", self.driver.calls)

    def test_bone_selection(self) -> None:
        policy = MaterialProductionPolicy(material_priority=(CityMaterial.BONE,))

        result = self._workflow().execute(self._request(run_key="city-material-bone", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("select:BONE:NORMAL:1", self.driver.calls)

    def test_priority_chooses_first_allowed_material(self) -> None:
        policy = MaterialProductionPolicy(
            material_priority=(CityMaterial.EBONY, CityMaterial.BONE, CityMaterial.LEATHER)
        )

        result = self._workflow().execute(self._request(run_key="city-material-priority", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual("EBONY", result.result["selected_material"]["material"])
        self.assertIn("select:EBONY:NORMAL:1", self.driver.calls)

    def test_quality_and_tier_rule_is_respected(self) -> None:
        self.driver.queue = MaterialQueueState(
            MaterialQueueStatus.IDLE,
            available_options=(
                _option(CityMaterial.LEATHER, quality=MaterialQuality.NORMAL, tier=1),
                _option(CityMaterial.LEATHER, quality=MaterialQuality.ELITE, tier=3),
            ),
        )
        policy = MaterialProductionPolicy(
            allowed_qualities=(MaterialQuality.ELITE,),
            minimum_tier=3,
            maximum_tier=3,
        )

        result = self._workflow().execute(self._request(run_key="city-material-tier", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual("ELITE", result.result["selected_material"]["quality"])
        self.assertEqual(3, result.result["selected_material"]["tier"])
        self.assertIn("quality_not_allowed", [item["ignored_reason"] for item in result.result["ignored_options"]])

    def test_unverified_selected_material_blocks_before_start(self) -> None:
        self.driver.queue = MaterialQueueState(
            MaterialQueueStatus.IDLE,
            available_options=(
                _option(CityMaterial.LEATHER, material_verified=False),
            ),
        )

        result = self._workflow().execute(self._request(run_key="city-material-unverified"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("select_material", result.result["terminal_state"])
        self.assertFalse(any(call.startswith("start:") for call in self.driver.calls))

    def test_postcondition_failure_records_failure_evidence(self) -> None:
        self.driver.verify_result = MaterialProductionStartResult(
            False,
            changed=False,
            queue_size=0,
            cooldown_seconds=0,
            message="Queue did not change.",
            retryable=False,
            screenshot_path="runtime/screens/material-not-started.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("city-material-postcondition"), run_key="city-material-postcondition")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("verify_production_state", result.result["terminal_state"])
        self.assertIn("did not change", result.result["terminal_reason"])
        self.assertEqual(
            "runtime/screens/material-not-started.png",
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
