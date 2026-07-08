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
from rok_assistant.tasks.lost_canyon_workflow import (  # noqa: E402
    LOST_CANYON_STATES,
    LOST_CANYON_TEMPLATE_KEYS,
    LostCanyonBattleOutcome,
    LostCanyonBattleReport,
    LostCanyonBattleStartResult,
    LostCanyonConfig,
    LostCanyonOpponent,
    LostCanyonPolicy,
    LostCanyonRequest,
    LostCanyonScanStatus,
    LostCanyonStateScan,
    LostCanyonWorkflow,
)
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


def _opponent(slot: int, power: int, *, name: str = "", confidence: float = 0.94) -> LostCanyonOpponent:
    return LostCanyonOpponent(
        slot=slot,
        name=name or f"Opponent {slot}",
        power=power,
        rank=100 + slot,
        confidence=confidence,
        target=(500 + (slot * 20), 300),
        metadata={"alliance": f"A{slot}"},
    )


class FakeLostCanyonDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.open_campaign_result = ResourceGatheringActionResult(True, data={"scene": "CAMPAIGN"})
        self.open_lost_result = ResourceGatheringActionResult(True, data={"scene": "LOST_CANYON"})
        self.scan = LostCanyonStateScan(
            LostCanyonScanStatus.READY,
            remaining_attempts=1,
            opponents=(
                _opponent(1, 2_000_000, name="High Power"),
                _opponent(2, 1_000_000, name="Low Power"),
            ),
            screenshot_path="runtime/screens/lost-scan.png",
        )
        self.start_result = LostCanyonBattleStartResult(
            True,
            changed=True,
            screenshot_path="runtime/screens/battle-started.png",
        )
        self.skip_result = ResourceGatheringActionResult(True, data={"skipped": True})
        self.reports: list[LostCanyonBattleReport] = [
            LostCanyonBattleReport(
                LostCanyonBattleOutcome.VICTORY,
                handled=True,
                remaining_attempts=0,
                screenshot_path="runtime/screens/victory.png",
            )
        ]

    def open_campaign(self, _request, _character, _policy):
        self.calls.append("open_campaign")
        return self.open_campaign_result

    def open_lost_canyon(self, _request, _character, _policy):
        self.calls.append("open_lost_canyon")
        return self.open_lost_result

    def inspect_lost_canyon(self, _request, _character, _policy):
        self.calls.append("inspect_lost_canyon")
        return self.scan

    def select_opponent(self, _request, _character, opponent, _policy):
        self.calls.append(f"select:{opponent.slot}:{opponent.name}")
        return ResourceGatheringActionResult(True, data={"selected_slot": opponent.slot})

    def start_battle(self, _request, _character, opponent, _policy):
        self.calls.append(f"start:{opponent.slot}")
        return self.start_result

    def skip_battle(self, _request, _character, opponent, _policy):
        self.calls.append(f"skip:{opponent.slot}")
        return self.skip_result

    def collect_battle_result(self, _request, _character, opponent, _policy):
        self.calls.append(f"collect_result:{opponent.slot}")
        if self.reports:
            return self.reports.pop(0)
        return LostCanyonBattleReport(
            LostCanyonBattleOutcome.VICTORY,
            handled=True,
            remaining_attempts=0,
        )


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


class LostCanyonWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "lost-canyon.sqlite3")
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
        self.driver = FakeLostCanyonDriver()
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

    def _workflow(self) -> LostCanyonWorkflow:
        return LostCanyonWorkflow(
            characters=self.characters,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=LostCanyonConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "lost-run",
        policy: LostCanyonPolicy | None = None,
    ) -> LostCanyonRequest:
        return LostCanyonRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or LostCanyonPolicy(),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states_and_template_keys(self) -> None:
        self.assertEqual(LOST_CANYON_STATES, self._workflow().workflow_states)
        self.assertIn("campaign.lost_canyon.entry", LOST_CANYON_TEMPLATE_KEYS)
        self.assertIn("lost_canyon.unavailable", LOST_CANYON_TEMPLATE_KEYS)
        self.assertIn("lost_canyon.result.victory", LOST_CANYON_TEMPLATE_KEYS)

    def test_lost_canyon_unavailable_skips_without_starting_battle(self) -> None:
        self.driver.scan = LostCanyonStateScan(
            LostCanyonScanStatus.FEATURE_UNAVAILABLE,
            remaining_attempts=None,
            message="Lost Canyon has not unlocked.",
            screenshot_path="runtime/screens/lost-unavailable.png",
        )

        result = self._workflow().execute(self._request(run_key="lost-unavailable"))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("inspect_attempts", result.result["terminal_state"])
        self.assertEqual("Lost Canyon has not unlocked.", result.result["skipped_reason"])
        self.assertFalse(any(call.startswith("start:") for call in self.driver.calls))

    def test_no_attempts_remaining_skips_without_starting_battle(self) -> None:
        self.driver.scan = LostCanyonStateScan(
            LostCanyonScanStatus.NO_ATTEMPTS,
            remaining_attempts=0,
            message="Daily attempts exhausted.",
            screenshot_path="runtime/screens/no-attempts.png",
        )

        result = self._workflow().execute(self._request(run_key="lost-none"))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("inspect_attempts", result.result["terminal_state"])
        self.assertEqual(0, result.result["latest_remaining_attempts"])
        self.assertNotIn("select:2:Low Power", self.driver.calls)
        self.assertFalse(any(call.startswith("start:") for call in self.driver.calls))

    def test_ui_variation_scan_metadata_is_preserved(self) -> None:
        self.driver.scan = LostCanyonStateScan(
            LostCanyonScanStatus.READY,
            remaining_attempts=1,
            opponents=(_opponent(1, 900_000, name="Variant Opponent"),),
            screenshot_path="runtime/screens/lost-variant.png",
            data={"ui_variant": "seasonal_banner", "attempts_label": "challenge chances"},
        )

        result = self._workflow().execute(self._request(run_key="lost-variant"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual("seasonal_banner", result.result["initial_scan"]["ui_variant"])
        self.assertEqual("challenge chances", result.result["initial_scan"]["attempts_label"])

    def test_victory_result_persists_opponent_result_and_remaining_attempts(self) -> None:
        job_id = self._job("lost-victory")

        result = self._workflow().execute(
            self._request(job_id=job_id, run_key="lost-victory")
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(1, result.result["battles_started"])
        self.assertEqual(0, result.result["latest_remaining_attempts"])
        self.assertEqual("Low Power", result.result["selected_opponents"][0]["name"])
        self.assertEqual("VICTORY", result.result["battle_results"][0]["result"]["outcome"])
        self.assertEqual(
            [
                "open_campaign",
                "open_lost_canyon",
                "inspect_lost_canyon",
                "select:2:Low Power",
                "start:2",
                "collect_result:2",
            ],
            self.driver.calls,
        )
        self.assertEqual({"battle_number": 1, "attempted": False, "reason": "policy_disabled"}, result.result["skip_battle"][0])
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual("Low Power", payload["result"]["battle_results"][0]["opponent"]["name"])
        self.assertEqual("VICTORY", payload["result"]["battle_results"][0]["result"]["outcome"])
        self.assertEqual(0, payload["result"]["latest_remaining_attempts"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_defeat_result_is_successful_completed_battle(self) -> None:
        self.driver.reports = [
            LostCanyonBattleReport(
                LostCanyonBattleOutcome.DEFEAT,
                handled=True,
                remaining_attempts=0,
                screenshot_path="runtime/screens/defeat.png",
            )
        ]

        result = self._workflow().execute(self._request(run_key="lost-defeat"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual("DEFEAT", result.result["battle_results"][0]["result"]["outcome"])
        self.assertEqual(0, result.result["latest_remaining_attempts"])

    def test_result_popup_handling_failure_blocks_workflow(self) -> None:
        self.driver.reports = [
            LostCanyonBattleReport(
                LostCanyonBattleOutcome.UNKNOWN,
                handled=False,
                remaining_attempts=1,
                message="Result popup was not recognized.",
                screenshot_path="runtime/screens/result-unknown.png",
            )
        ]

        result = self._workflow().execute(self._request(run_key="lost-popup-blocked"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("fight_battles", result.result["terminal_state"])
        self.assertEqual("Result popup was not recognized.", result.result["failure_reason"])
        self.assertFalse(result.result["popup_handling"][0]["handled"])

    def test_navigation_failure_blocks_before_attempt_scan(self) -> None:
        self.driver.open_lost_result = ResourceGatheringActionResult(
            False,
            message="Campaign entry was not visible.",
            screenshot_path="runtime/screens/no-lost.png",
        )

        result = self._workflow().execute(self._request(run_key="lost-navigation"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("open_lost_canyon", result.result["terminal_state"])
        self.assertEqual("Campaign entry was not visible.", result.result["failure_reason"])
        self.assertEqual(["open_campaign", "open_lost_canyon", "open_lost_canyon"], self.driver.calls)
        self.assertNotIn("inspect_lost_canyon", self.driver.calls)
        self.assertFalse(any(call.startswith("start:") for call in self.driver.calls))

    def test_skip_battle_is_disabled_by_default(self) -> None:
        result = self._workflow().execute(self._request(run_key="lost-no-skip"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertFalse(any(call.startswith("skip:") for call in self.driver.calls))
        self.assertEqual("policy_disabled", result.result["skip_battle"][0]["reason"])

    def test_skip_battle_runs_only_when_policy_enables_it(self) -> None:
        policy = LostCanyonPolicy(allow_skip_battle=True)

        result = self._workflow().execute(self._request(run_key="lost-skip", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("skip:2", self.driver.calls)
        self.assertTrue(result.result["skip_battle"][0]["attempted"])

    def test_formation_mutation_is_disabled_by_default_and_rejected_when_enabled(self) -> None:
        self.assertFalse(LostCanyonPolicy().normalized().allow_formation_changes)

        result = self._workflow().execute(
            self._request(
                run_key="lost-formation-disabled",
                policy=LostCanyonPolicy(allow_formation_changes=True),
            )
        )

        self.assertEqual(WorkflowOutcome.VALIDATION_FAILURE, result.outcome)
        self.assertEqual("validate_input", result.result["terminal_state"])
        self.assertIn("Formation mutation", result.result["failure_reason"])
        self.assertFalse(any(call.startswith("start:") for call in self.driver.calls))

    def test_policy_limit_is_respected_before_remaining_attempts(self) -> None:
        self.driver.scan = LostCanyonStateScan(
            LostCanyonScanStatus.READY,
            remaining_attempts=3,
            opponents=(_opponent(1, 1_000_000),),
        )
        self.driver.reports = [
            LostCanyonBattleReport(LostCanyonBattleOutcome.VICTORY, handled=True, remaining_attempts=2),
            LostCanyonBattleReport(LostCanyonBattleOutcome.VICTORY, handled=True, remaining_attempts=1),
            LostCanyonBattleReport(LostCanyonBattleOutcome.VICTORY, handled=True, remaining_attempts=0),
        ]
        policy = LostCanyonPolicy(max_battles_per_run=2)

        result = self._workflow().execute(self._request(run_key="lost-limit", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(2, result.result["battles_started"])
        self.assertEqual(1, result.result["latest_remaining_attempts"])
        self.assertEqual(2, len([call for call in self.driver.calls if call.startswith("start:")]))


if __name__ == "__main__":
    unittest.main()
