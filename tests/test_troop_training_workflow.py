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
from rok_assistant.tasks.resource_search_workflow import ResourceGatheringActionResult  # noqa: E402
from rok_assistant.tasks.troop_training_workflow import (  # noqa: E402
    TROOP_TRAINING_STATES,
    TroopTrainingBuilding,
    TroopTrainingConfig,
    TroopTrainingConfirmation,
    TroopTrainingPolicy,
    TroopTrainingQueueState,
    TroopTrainingQueueStatus,
    TroopTrainingRequest,
    TroopTrainingStartResult,
    TroopTrainingTierOption,
    TroopTrainingWorkflow,
    TroopType,
    troop_type_for_building,
)
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


def _tier(building: TroopTrainingBuilding, tier: int, *, resources_available: bool = True) -> TroopTrainingTierOption:
    return TroopTrainingTierOption(
        building=building,
        troop_type=troop_type_for_building(building),
        tier=tier,
        resources_available=resources_available,
        confidence=0.94,
    )


class FakeTroopTrainingDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.unsafe_calls: list[str] = []
        self.open_result = ResourceGatheringActionResult(True, data={"scene": "TRAINING_BUILDING"})
        self.select_result = ResourceGatheringActionResult(True, data={"selected": True})
        self.start_result = TroopTrainingStartResult(
            True,
            changed=False,
            confirmation=TroopTrainingConfirmation.FREE,
            queue_size=0,
            timer_seconds=0,
        )
        self.verify_result = TroopTrainingStartResult(
            True,
            changed=True,
            confirmation=TroopTrainingConfirmation.NONE,
            queue_size=1,
            timer_seconds=1800,
            screenshot_path="runtime/screens/training-started.png",
        )
        self.queues: dict[TroopTrainingBuilding, TroopTrainingQueueState] = {
            building: TroopTrainingQueueState(
                building=building,
                troop_type=troop_type_for_building(building),
                status=TroopTrainingQueueStatus.IDLE,
                available_tiers=tuple(_tier(building, tier) for tier in range(1, 6)),
                queue_size=0,
                timer_seconds=0,
                screenshot_path=f"runtime/screens/{building.value.lower()}-queue.png",
            )
            for building in TroopTrainingBuilding
        }

    def normalize_city_view(self, _request, _character, _policy, building):
        self.calls.append(f"normalize:{building.value}")
        return ResourceGatheringActionResult(True, data={"scene": "CITY_HOME"})

    def open_training_building(self, _request, _character, _policy, building):
        self.calls.append(f"open:{building.value}")
        return self.open_result

    def inspect_training_queue(self, _request, _character, _policy, building):
        self.calls.append(f"inspect:{building.value}")
        return self.queues[building]

    def select_training_tier(self, _request, _character, option, _policy):
        self.calls.append(f"select:{option.normalized_building().value}:T{option.tier}")
        return self.select_result

    def start_training(self, _request, _character, option, _before, _policy):
        self.calls.append(f"start:{option.normalized_building().value}:T{option.tier}")
        return self.start_result

    def verify_training_state(self, _request, _character, option, _before, _start, _policy):
        self.calls.append(f"verify:{option.normalized_building().value}:T{option.tier}")
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
        )


class TroopTrainingWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "troop-training.sqlite3")
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
        self.driver = FakeTroopTrainingDriver()
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

    def _workflow(self) -> TroopTrainingWorkflow:
        return TroopTrainingWorkflow(
            characters=self.characters,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=TroopTrainingConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "troop-training",
        policy: TroopTrainingPolicy | None = None,
    ) -> TroopTrainingRequest:
        return TroopTrainingRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or TroopTrainingPolicy(),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states(self) -> None:
        self.assertEqual(TROOP_TRAINING_STATES, self._workflow().workflow_states)

    def _assert_single_building_starts(self, building: TroopTrainingBuilding, troop_type: TroopType) -> None:
        policy = TroopTrainingPolicy(enabled_buildings=(building,), desired_tier=2)
        result = self._workflow().execute(
            self._request(run_key=f"train-{building.value.lower()}", policy=policy)
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(building.value, result.result["selected_building"])
        self.assertEqual(troop_type.value, result.result["selected_troop_type"])
        self.assertEqual(2, result.result["selected_tier"]["tier"])
        self.assertIn(f"start:{building.value}:T2", self.driver.calls)
        self.assertIn(f"verify:{building.value}:T2", self.driver.calls)

    def test_infantry_training_starts_when_queue_is_free(self) -> None:
        self._assert_single_building_starts(TroopTrainingBuilding.BARRACKS, TroopType.INFANTRY)

    def test_archer_training_starts_when_queue_is_free(self) -> None:
        self._assert_single_building_starts(TroopTrainingBuilding.ARCHERY_RANGE, TroopType.ARCHER)

    def test_cavalry_training_starts_when_queue_is_free(self) -> None:
        self._assert_single_building_starts(TroopTrainingBuilding.STABLE, TroopType.CAVALRY)

    def test_siege_training_starts_when_queue_is_free(self) -> None:
        self._assert_single_building_starts(TroopTrainingBuilding.SIEGE_WORKSHOP, TroopType.SIEGE)

    def test_all_enabled_buildings_are_processed(self) -> None:
        result = self._workflow().execute(
            self._request(job_id=self._job("troop-all"), run_key="troop-all")
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(4, result.result["trained_building_count"])
        for building in TroopTrainingBuilding:
            self.assertIn(f"normalize:{building.value}", self.driver.calls)
            self.assertIn(f"open:{building.value}", self.driver.calls)
            self.assertIn(f"inspect:{building.value}", self.driver.calls)
            self.assertIn(f"select:{building.value}:T1", self.driver.calls)
            self.assertIn(f"start:{building.value}:T1", self.driver.calls)
            self.assertIn(f"verify:{building.value}:T1", self.driver.calls)
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual(4, payload["result"]["trained_building_count"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_configured_tier_is_selected(self) -> None:
        policy = TroopTrainingPolicy(
            enabled_buildings=(TroopTrainingBuilding.BARRACKS,),
            desired_tier=4,
        )

        result = self._workflow().execute(self._request(run_key="troop-tier-4", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(4, result.result["selected_tier"]["tier"])
        self.assertIn("select:BARRACKS:T4", self.driver.calls)

    def test_disabled_building_is_skipped(self) -> None:
        policy = TroopTrainingPolicy(
            enabled_buildings=(TroopTrainingBuilding.STABLE,),
            desired_tier=1,
        )

        result = self._workflow().execute(self._request(run_key="troop-disabled", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["STABLE"], result.result["enabled_buildings"])
        self.assertIn("start:STABLE:T1", self.driver.calls)
        self.assertFalse(any("BARRACKS" in call for call in self.driver.calls))

    def test_busy_queue_returns_skipped_without_starting(self) -> None:
        building = TroopTrainingBuilding.BARRACKS
        self.driver.queues[building] = TroopTrainingQueueState(
            building=building,
            troop_type=troop_type_for_building(building),
            status=TroopTrainingQueueStatus.BUSY,
            active_tier=1,
            queue_size=1,
            timer_seconds=1200,
            message="Queue already training.",
            screenshot_path="runtime/screens/troop-busy.png",
        )
        policy = TroopTrainingPolicy(enabled_buildings=(building,))

        result = self._workflow().execute(
            self._request(job_id=self._job("troop-busy"), run_key="troop-busy", policy=policy)
        )

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("inspect_training_queue", result.result["terminal_state"])
        self.assertEqual("No enabled troop training queue is available.", result.result["skipped_reason"])
        self.assertFalse(any(call.startswith("select:") for call in self.driver.calls))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("completed", run.status)  # type: ignore[union-attr]

    def test_insufficient_resources_returns_skipped_by_policy(self) -> None:
        building = TroopTrainingBuilding.BARRACKS
        self.driver.queues[building] = TroopTrainingQueueState(
            building=building,
            troop_type=troop_type_for_building(building),
            status=TroopTrainingQueueStatus.IDLE,
            available_tiers=(_tier(building, 3, resources_available=False),),
            screenshot_path="runtime/screens/troop-insufficient.png",
        )
        policy = TroopTrainingPolicy(enabled_buildings=(building,), desired_tier=3)

        result = self._workflow().execute(self._request(run_key="troop-insufficient", policy=policy))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("select_training_tier", result.result["terminal_state"])
        self.assertEqual("insufficient_resources", result.result["ignored_tiers"][0]["ignored_reason"])
        self.assertFalse(any(call.startswith("start:") for call in self.driver.calls))

    def test_insufficient_resources_returns_blocked_when_policy_requires(self) -> None:
        building = TroopTrainingBuilding.BARRACKS
        self.driver.queues[building] = TroopTrainingQueueState(
            building=building,
            troop_type=troop_type_for_building(building),
            status=TroopTrainingQueueStatus.IDLE,
            available_tiers=(_tier(building, 3, resources_available=False),),
            screenshot_path="runtime/screens/troop-insufficient.png",
        )
        policy = TroopTrainingPolicy(
            enabled_buildings=(building,),
            desired_tier=3,
            skip_insufficient_resources=False,
        )

        result = self._workflow().execute(self._request(run_key="troop-insufficient-block", policy=policy))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("select_training_tier", result.result["terminal_state"])
        self.assertFalse(any(call.startswith("start:") for call in self.driver.calls))

    def test_speedup_gem_and_premium_controls_are_not_clicked(self) -> None:
        self.driver.start_result = TroopTrainingStartResult(
            True,
            changed=False,
            confirmation=TroopTrainingConfirmation.GEM,
            screenshot_path="runtime/screens/gem-confirm.png",
        )
        policy = TroopTrainingPolicy(enabled_buildings=(TroopTrainingBuilding.BARRACKS,))

        result = self._workflow().execute(self._request(run_key="troop-gem-confirm", policy=policy))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertIn("GEM", result.result["terminal_reason"])
        self.assertEqual([], self.driver.unsafe_calls)
        self.assertNotIn("verify:BARRACKS:T1", self.driver.calls)

    def test_postcondition_failure_records_failure_evidence(self) -> None:
        self.driver.verify_result = TroopTrainingStartResult(
            False,
            changed=False,
            confirmation=TroopTrainingConfirmation.NONE,
            queue_size=0,
            timer_seconds=0,
            message="Timer did not appear.",
            retryable=False,
            screenshot_path="runtime/screens/troop-not-started.png",
        )
        policy = TroopTrainingPolicy(enabled_buildings=(TroopTrainingBuilding.BARRACKS,))

        result = self._workflow().execute(
            self._request(job_id=self._job("troop-postcondition"), run_key="troop-postcondition", policy=policy)
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("verify_training_state", result.result["terminal_state"])
        self.assertIn("timer/state did not change", result.result["terminal_reason"])
        self.assertEqual(
            "runtime/screens/troop-not-started.png",
            result.result["failure_evidence"]["screenshot_path"],
        )
        self.assertEqual(
            {"attempted": False, "healthy": True, "circuit_opened": False},
            result.result["recovery_outcome"],
        )
        self.assertEqual(1, len(self.incidents.list_open()))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("failed", run.status)  # type: ignore[union-attr]

    def test_policy_validation_rejects_invalid_tier_and_building_config(self) -> None:
        with self.assertRaises(ValueError):
            TroopTrainingPolicy(desired_tier=0).normalized()
        with self.assertRaises(ValueError):
            TroopTrainingPolicy(desired_tier=6).normalized()
        with self.assertRaises(ValueError):
            TroopTrainingPolicy(enabled_buildings=()).normalized()
        with self.assertRaises(ValueError):
            TroopTrainingPolicy(enabled_buildings=("watchtower",)).normalized()
        with self.assertRaises(ValueError):
            TroopTrainingPolicy(allow_speedups=True).normalized()
        with self.assertRaises(ValueError):
            TroopTrainingPolicy(allow_gem_spending=True).normalized()
        with self.assertRaises(ValueError):
            TroopTrainingPolicy(allow_premium_spending=True).normalized()


if __name__ == "__main__":
    unittest.main()
