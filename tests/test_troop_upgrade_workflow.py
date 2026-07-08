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
    TroopTrainingBuilding,
    TroopType,
    troop_type_for_building,
)
from rok_assistant.tasks.troop_upgrade_workflow import (  # noqa: E402
    TROOP_UPGRADE_STATES,
    TroopUpgradeConfig,
    TroopUpgradeConfirmation,
    TroopUpgradeOption,
    TroopUpgradePolicy,
    TroopUpgradeQueueStatus,
    TroopUpgradeRequest,
    TroopUpgradeStartResult,
    TroopUpgradeState,
    TroopUpgradeWorkflow,
)
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


def _option(
    building: TroopTrainingBuilding,
    *,
    source_tier: int = 1,
    target_tier: int = 2,
    eligible_count: int = 25,
    resources_available: bool = True,
    resource_cost: int = 0,
) -> TroopUpgradeOption:
    return TroopUpgradeOption(
        building=building,
        troop_type=troop_type_for_building(building),
        source_tier=source_tier,
        target_tier=target_tier,
        eligible_count=eligible_count,
        resources_available=resources_available,
        resource_cost=resource_cost,
        confidence=0.94,
    )


class FakeTroopUpgradeDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.unsafe_calls: list[str] = []
        self.open_result = ResourceGatheringActionResult(True, data={"scene": "TRAINING_BUILDING"})
        self.tab_result = ResourceGatheringActionResult(True, data={"tab": "UPGRADE"})
        self.select_result = ResourceGatheringActionResult(True, data={"selected": True})
        self.amount_result = ResourceGatheringActionResult(True, data={"amount_set": True})
        self.start_result = TroopUpgradeStartResult(
            True,
            changed=False,
            confirmation=TroopUpgradeConfirmation.FREE,
            queue_size=0,
            timer_seconds=0,
        )
        self.verify_result = TroopUpgradeStartResult(
            True,
            changed=True,
            confirmation=TroopUpgradeConfirmation.NONE,
            queue_size=1,
            timer_seconds=900,
            screenshot_path="runtime/screens/upgrade-started.png",
        )
        self.states: dict[TroopTrainingBuilding, TroopUpgradeState] = {
            building: TroopUpgradeState(
                building=building,
                troop_type=troop_type_for_building(building),
                status=TroopUpgradeQueueStatus.IDLE,
                upgrade_tab_active=True,
                eligible_options=(_option(building),),
                queue_size=0,
                timer_seconds=0,
                screenshot_path=f"runtime/screens/{building.value.lower()}-upgrade.png",
            )
            for building in TroopTrainingBuilding
        }

    def normalize_city_view(self, _request, _character, _policy, building):
        self.calls.append(f"normalize:{building.value}")
        return ResourceGatheringActionResult(True, data={"scene": "CITY_HOME"})

    def open_training_building(self, _request, _character, _policy, building):
        self.calls.append(f"open:{building.value}")
        return self.open_result

    def open_upgrade_tab(self, _request, _character, _policy, building):
        self.calls.append(f"open_upgrade_tab:{building.value}")
        return self.tab_result

    def inspect_upgrade_state(self, _request, _character, _policy, building):
        self.calls.append(f"inspect:{building.value}")
        return self.states[building]

    def select_upgrade_tiers(self, _request, _character, option, _policy):
        self.calls.append(
            f"select_upgrade:{option.normalized_building().value}:T{option.source_tier}->T{option.target_tier}"
        )
        return self.select_result

    def set_upgrade_amount(self, _request, _character, option, amount, _policy):
        self.calls.append(f"amount:{option.normalized_building().value}:{amount.selected_amount}")
        return self.amount_result

    def start_upgrade(self, _request, _character, option, amount, _before, _policy):
        self.calls.append(f"start_upgrade:{option.normalized_building().value}:{amount.selected_amount}")
        return self.start_result

    def verify_upgrade_state(self, _request, _character, option, amount, _before, _start, _policy):
        self.calls.append(f"verify:{option.normalized_building().value}:{amount.selected_amount}")
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


class TroopUpgradeWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "troop-upgrade.sqlite3")
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
        self.driver = FakeTroopUpgradeDriver()
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

    def _workflow(self) -> TroopUpgradeWorkflow:
        return TroopUpgradeWorkflow(
            characters=self.characters,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=TroopUpgradeConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "troop-upgrade",
        policy: TroopUpgradePolicy | None = None,
    ) -> TroopUpgradeRequest:
        return TroopUpgradeRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or TroopUpgradePolicy(),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states(self) -> None:
        self.assertEqual(TROOP_UPGRADE_STATES, self._workflow().workflow_states)

    def _assert_single_building_upgrades(
        self,
        building: TroopTrainingBuilding,
        troop_type: TroopType,
    ) -> None:
        policy = TroopUpgradePolicy(enabled_buildings=(building,), source_tier=2, target_tier=3)
        self.driver.states[building] = TroopUpgradeState(
            building=building,
            troop_type=troop_type,
            status=TroopUpgradeQueueStatus.IDLE,
            upgrade_tab_active=True,
            eligible_options=(_option(building, source_tier=2, target_tier=3),),
        )

        result = self._workflow().execute(
            self._request(run_key=f"upgrade-{building.value.lower()}", policy=policy)
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(building.value, result.result["selected_building"])
        self.assertEqual(troop_type.value, result.result["selected_troop_type"])
        self.assertEqual(2, result.result["source_tier"])
        self.assertEqual(3, result.result["target_tier"])
        self.assertIn(f"open_upgrade_tab:{building.value}", self.driver.calls)
        self.assertIn(f"start_upgrade:{building.value}:1", self.driver.calls)

    def test_eligible_infantry_upgrade_starts_successfully(self) -> None:
        self._assert_single_building_upgrades(TroopTrainingBuilding.BARRACKS, TroopType.INFANTRY)

    def test_eligible_archer_upgrade_starts_successfully(self) -> None:
        self._assert_single_building_upgrades(TroopTrainingBuilding.ARCHERY_RANGE, TroopType.ARCHER)

    def test_eligible_cavalry_upgrade_starts_successfully(self) -> None:
        self._assert_single_building_upgrades(TroopTrainingBuilding.STABLE, TroopType.CAVALRY)

    def test_eligible_siege_upgrade_starts_successfully(self) -> None:
        self._assert_single_building_upgrades(TroopTrainingBuilding.SIEGE_WORKSHOP, TroopType.SIEGE)

    def test_no_eligible_units_returns_skipped(self) -> None:
        building = TroopTrainingBuilding.BARRACKS
        self.driver.states[building] = TroopUpgradeState(
            building=building,
            troop_type=troop_type_for_building(building),
            status=TroopUpgradeQueueStatus.IDLE,
            upgrade_tab_active=True,
            eligible_options=(),
            screenshot_path="runtime/screens/no-eligible.png",
        )
        policy = TroopUpgradePolicy(enabled_buildings=(building,))

        result = self._workflow().execute(self._request(run_key="upgrade-none", policy=policy))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("select_upgrade_tiers", result.result["terminal_state"])
        self.assertEqual("no_eligible_units", result.result["skipped_buildings"][0]["reason"])
        self.assertFalse(any(call.startswith("start_upgrade:") for call in self.driver.calls))

    def test_wrong_tab_blocks_and_records_failure_evidence(self) -> None:
        building = TroopTrainingBuilding.BARRACKS
        self.driver.states[building] = TroopUpgradeState(
            building=building,
            troop_type=troop_type_for_building(building),
            status=TroopUpgradeQueueStatus.IDLE,
            upgrade_tab_active=False,
            eligible_options=(_option(building),),
            message="Training tab is still active.",
            screenshot_path="runtime/screens/wrong-tab.png",
        )
        policy = TroopUpgradePolicy(enabled_buildings=(building,))

        result = self._workflow().execute(self._request(run_key="upgrade-wrong-tab", policy=policy))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("inspect_upgrade_state", result.result["terminal_state"])
        self.assertEqual(
            "runtime/screens/wrong-tab.png",
            result.result["failure_evidence"]["screenshot_path"],
        )
        self.assertFalse(any(call.startswith("select_upgrade:") for call in self.driver.calls))

    def test_insufficient_resources_returns_skipped_by_policy(self) -> None:
        building = TroopTrainingBuilding.BARRACKS
        self.driver.states[building] = TroopUpgradeState(
            building=building,
            troop_type=troop_type_for_building(building),
            status=TroopUpgradeQueueStatus.IDLE,
            upgrade_tab_active=True,
            eligible_options=(_option(building, resources_available=False),),
            screenshot_path="runtime/screens/upgrade-insufficient.png",
        )
        policy = TroopUpgradePolicy(enabled_buildings=(building,))

        result = self._workflow().execute(self._request(run_key="upgrade-insufficient", policy=policy))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("insufficient_resources", result.result["skipped_buildings"][0]["reason"])
        self.assertFalse(any(call.startswith("start_upgrade:") for call in self.driver.calls))

    def test_insufficient_resources_returns_blocked_when_policy_requires(self) -> None:
        building = TroopTrainingBuilding.BARRACKS
        self.driver.states[building] = TroopUpgradeState(
            building=building,
            troop_type=troop_type_for_building(building),
            status=TroopUpgradeQueueStatus.IDLE,
            upgrade_tab_active=True,
            eligible_options=(_option(building, resources_available=False),),
            screenshot_path="runtime/screens/upgrade-insufficient.png",
        )
        policy = TroopUpgradePolicy(enabled_buildings=(building,), skip_insufficient_resources=False)

        result = self._workflow().execute(self._request(run_key="upgrade-insufficient-block", policy=policy))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("select_upgrade_tiers", result.result["terminal_state"])
        self.assertFalse(any(call.startswith("start_upgrade:") for call in self.driver.calls))

    def test_configured_source_and_target_tiers_are_respected(self) -> None:
        building = TroopTrainingBuilding.BARRACKS
        self.driver.states[building] = TroopUpgradeState(
            building=building,
            troop_type=troop_type_for_building(building),
            status=TroopUpgradeQueueStatus.IDLE,
            upgrade_tab_active=True,
            eligible_options=(
                _option(building, source_tier=1, target_tier=2),
                _option(building, source_tier=3, target_tier=4),
            ),
        )
        policy = TroopUpgradePolicy(enabled_buildings=(building,), source_tier=3, target_tier=4)

        result = self._workflow().execute(self._request(run_key="upgrade-tier-policy", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(3, result.result["source_tier"])
        self.assertEqual(4, result.result["target_tier"])
        self.assertIn("select_upgrade:BARRACKS:T3->T4", self.driver.calls)

    def test_all_eligible_option_respects_budget(self) -> None:
        building = TroopTrainingBuilding.BARRACKS
        self.driver.states[building] = TroopUpgradeState(
            building=building,
            troop_type=troop_type_for_building(building),
            status=TroopUpgradeQueueStatus.IDLE,
            upgrade_tab_active=True,
            eligible_options=(_option(building, eligible_count=80),),
        )
        policy = TroopUpgradePolicy(
            enabled_buildings=(building,),
            allow_all_eligible=True,
            max_upgrade_amount=35,
        )

        result = self._workflow().execute(self._request(run_key="upgrade-all-eligible", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(35, result.result["selected_upgrade_amount"])
        self.assertIn("amount:BARRACKS:35", self.driver.calls)

    def test_queue_resource_budget_blocks_expensive_upgrade(self) -> None:
        building = TroopTrainingBuilding.BARRACKS
        self.driver.states[building] = TroopUpgradeState(
            building=building,
            troop_type=troop_type_for_building(building),
            status=TroopUpgradeQueueStatus.IDLE,
            upgrade_tab_active=True,
            eligible_options=(_option(building, resource_cost=5000),),
        )
        policy = TroopUpgradePolicy(enabled_buildings=(building,), max_resource_cost=1000)

        result = self._workflow().execute(self._request(run_key="upgrade-budget", policy=policy))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("resource_budget_exceeded", result.result["ignored_options"][0]["ignored_reason"])
        self.assertFalse(any(call.startswith("start_upgrade:") for call in self.driver.calls))

    def test_accidental_training_controls_are_not_clicked(self) -> None:
        self.driver.start_result = TroopUpgradeStartResult(
            True,
            changed=False,
            confirmation=TroopUpgradeConfirmation.TRAINING,
            screenshot_path="runtime/screens/training-confirm.png",
        )
        policy = TroopUpgradePolicy(enabled_buildings=(TroopTrainingBuilding.BARRACKS,))

        result = self._workflow().execute(self._request(run_key="upgrade-training-confirm", policy=policy))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertIn("TRAINING", result.result["terminal_reason"])
        self.assertEqual([], self.driver.unsafe_calls)
        self.assertNotIn("verify:BARRACKS:1", self.driver.calls)

    def test_speedup_gem_and_premium_controls_are_not_clicked(self) -> None:
        self.driver.start_result = TroopUpgradeStartResult(
            True,
            changed=False,
            confirmation=TroopUpgradeConfirmation.GEM,
            screenshot_path="runtime/screens/gem-confirm.png",
        )
        policy = TroopUpgradePolicy(enabled_buildings=(TroopTrainingBuilding.BARRACKS,))

        result = self._workflow().execute(self._request(run_key="upgrade-gem-confirm", policy=policy))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertIn("GEM", result.result["terminal_reason"])
        self.assertEqual([], self.driver.unsafe_calls)
        self.assertNotIn("verify:BARRACKS:1", self.driver.calls)

    def test_postcondition_mismatch_records_failure_evidence(self) -> None:
        self.driver.verify_result = TroopUpgradeStartResult(
            False,
            changed=False,
            confirmation=TroopUpgradeConfirmation.NONE,
            queue_size=0,
            timer_seconds=0,
            message="Timer did not appear.",
            retryable=False,
            screenshot_path="runtime/screens/upgrade-not-started.png",
        )
        policy = TroopUpgradePolicy(enabled_buildings=(TroopTrainingBuilding.BARRACKS,))

        result = self._workflow().execute(
            self._request(job_id=self._job("upgrade-postcondition"), run_key="upgrade-postcondition", policy=policy)
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("verify_upgrade_state", result.result["terminal_state"])
        self.assertIn("timer/state did not change", result.result["terminal_reason"])
        self.assertEqual(
            "runtime/screens/upgrade-not-started.png",
            result.result["failure_evidence"]["screenshot_path"],
        )
        self.assertEqual(
            {"attempted": False, "healthy": True, "circuit_opened": False},
            result.result["recovery_outcome"],
        )
        self.assertEqual(1, len(self.incidents.list_open()))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("failed", run.status)  # type: ignore[union-attr]

    def test_persists_upgrade_result_json_without_schema_change(self) -> None:
        result = self._workflow().execute(
            self._request(job_id=self._job("upgrade-persist"), run_key="upgrade-persist")
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual(4, payload["result"]["upgraded_building_count"])
        self.assertEqual(1, payload["result"]["source_tier"])
        self.assertEqual(2, payload["result"]["target_tier"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_policy_validation_rejects_invalid_tier_and_building_config(self) -> None:
        with self.assertRaises(ValueError):
            TroopUpgradePolicy(source_tier=0).normalized()
        with self.assertRaises(ValueError):
            TroopUpgradePolicy(target_tier=6).normalized()
        with self.assertRaises(ValueError):
            TroopUpgradePolicy(source_tier=3, target_tier=3).normalized()
        with self.assertRaises(ValueError):
            TroopUpgradePolicy(enabled_buildings=()).normalized()
        with self.assertRaises(ValueError):
            TroopUpgradePolicy(enabled_buildings=("watchtower",)).normalized()
        with self.assertRaises(ValueError):
            TroopUpgradePolicy(upgrade_amount=5, max_upgrade_amount=4).normalized()
        with self.assertRaises(ValueError):
            TroopUpgradePolicy(allow_speedups=True).normalized()
        with self.assertRaises(ValueError):
            TroopUpgradePolicy(allow_gem_spending=True).normalized()
        with self.assertRaises(ValueError):
            TroopUpgradePolicy(allow_premium_spending=True).normalized()
        with self.assertRaises(ValueError):
            TroopUpgradePolicy(allow_resource_items=True).normalized()


if __name__ == "__main__":
    unittest.main()
