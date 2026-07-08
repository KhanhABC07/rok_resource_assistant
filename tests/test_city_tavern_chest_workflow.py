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
from rok_assistant.tasks.city_tavern_chest_workflow import (  # noqa: E402
    CITY_TAVERN_CHEST_STATES,
    CITY_TAVERN_CHEST_TEMPLATE_KEYS,
    TavernChestConfig,
    TavernChestConfirmation,
    TavernChestObservation,
    TavernChestOpenResult,
    TavernChestPolicy,
    TavernChestRequest,
    TavernChestScan,
    TavernChestStatus,
    TavernChestType,
    TavernChestWorkflow,
    TavernRewardCloseResult,
)
from rok_assistant.tasks.resource_search_workflow import ResourceGatheringActionResult  # noqa: E402
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


def _chest(
    chest_type: TavernChestType,
    status: TavernChestStatus,
    *,
    free_indicator_visible: bool | None = None,
    screenshot_path: str = "",
) -> TavernChestObservation:
    return TavernChestObservation(
        chest_type,
        status,
        confidence=0.94,
        target=(500, 300) if chest_type == TavernChestType.SILVER else (760, 300),
        free_indicator_visible=status == TavernChestStatus.FREE if free_indicator_visible is None else free_indicator_visible,
        screenshot_path=screenshot_path,
    )


class FakeTavernChestDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.open_tavern_result = ResourceGatheringActionResult(
            True,
            data={"scene": "TAVERN"},
            screenshot_path="runtime/screens/tavern.png",
        )
        self.scan = TavernChestScan(
            (
                _chest(TavernChestType.SILVER, TavernChestStatus.FREE),
                _chest(TavernChestType.GOLD, TavernChestStatus.COOLDOWN),
            ),
            screenshot_path="runtime/screens/tavern-scan.png",
        )
        self.open_results: dict[TavernChestType, TavernChestOpenResult] = {
            TavernChestType.SILVER: TavernChestOpenResult(
                True,
                changed=True,
                confirmation=TavernChestConfirmation.FREE,
                screenshot_path="runtime/screens/silver-opened.png",
            ),
            TavernChestType.GOLD: TavernChestOpenResult(
                True,
                changed=True,
                confirmation=TavernChestConfirmation.FREE,
                screenshot_path="runtime/screens/gold-opened.png",
            ),
        }
        self.close_result = TavernRewardCloseResult(
            True,
            closed=True,
            screenshot_path="runtime/screens/reward-closed.png",
        )
        self.verify_results: dict[TavernChestType, TavernChestObservation] = {
            TavernChestType.SILVER: _chest(
                TavernChestType.SILVER,
                TavernChestStatus.COOLDOWN,
                free_indicator_visible=False,
                screenshot_path="runtime/screens/silver-cooldown.png",
            ),
            TavernChestType.GOLD: _chest(
                TavernChestType.GOLD,
                TavernChestStatus.COOLDOWN,
                free_indicator_visible=False,
                screenshot_path="runtime/screens/gold-cooldown.png",
            ),
        }

    def open_tavern(self, _request, _character, _policy):
        self.calls.append("open_tavern")
        return self.open_tavern_result

    def scan_chests(self, _request, _character, _policy):
        self.calls.append("scan_chests")
        return self.scan

    def open_free_chest(self, _request, _character, chest, _policy):
        chest_type = chest.normalized_chest_type()
        self.calls.append(f"open:{chest_type.value}")
        return self.open_results[chest_type]

    def close_reward_ui(self, _request, _character, chest, _open_result, _policy):
        self.calls.append(f"close_reward:{chest.normalized_chest_type().value}")
        return self.close_result

    def verify_chest_state(self, _request, _character, chest, _open_result, _policy):
        chest_type = chest.normalized_chest_type()
        self.calls.append(f"verify:{chest_type.value}")
        return self.verify_results[chest_type]


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


class TavernChestWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "city-tavern.sqlite3")
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
        self.driver = FakeTavernChestDriver()
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

    def _workflow(self) -> TavernChestWorkflow:
        return TavernChestWorkflow(
            characters=self.characters,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=TavernChestConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "city-tavern-run",
        policy: TavernChestPolicy | None = None,
    ) -> TavernChestRequest:
        return TavernChestRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or TavernChestPolicy(),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states_and_template_keys(self) -> None:
        self.assertEqual(CITY_TAVERN_CHEST_STATES, self._workflow().workflow_states)
        self.assertIn("tavern.silver.free", CITY_TAVERN_CHEST_TEMPLATE_KEYS)
        self.assertIn("tavern.gold.free", CITY_TAVERN_CHEST_TEMPLATE_KEYS)

    def test_free_silver_chest_opens_and_persists_metadata(self) -> None:
        job_id = self._job("city-tavern-silver")

        result = self._workflow().execute(
            self._request(job_id=job_id, run_key="city-tavern-silver")
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["SILVER"], result.result["selected_chest_types"])
        self.assertEqual(
            [
                "open_tavern",
                "scan_chests",
                "open:SILVER",
                "close_reward:SILVER",
                "verify:SILVER",
            ],
            self.driver.calls,
        )
        self.assertTrue(result.result["reward_ui_handling"][0]["closed"])
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual("SILVER", payload["result"]["selected_chest_types"][0])
        self.assertEqual("COOLDOWN", payload["result"]["verification_result"]["after"]["status"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_free_gold_chest_opens_when_silver_is_not_free(self) -> None:
        self.driver.scan = TavernChestScan(
            (
                _chest(TavernChestType.SILVER, TavernChestStatus.COOLDOWN, free_indicator_visible=False),
                _chest(TavernChestType.GOLD, TavernChestStatus.FREE),
            )
        )

        result = self._workflow().execute(self._request(run_key="city-tavern-gold"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["GOLD"], result.result["selected_chest_types"])
        self.assertIn("open:GOLD", self.driver.calls)
        self.assertNotIn("open:SILVER", self.driver.calls)

    def test_both_free_chests_open_according_to_policy(self) -> None:
        self.driver.scan = TavernChestScan(
            (
                _chest(TavernChestType.SILVER, TavernChestStatus.FREE),
                _chest(TavernChestType.GOLD, TavernChestStatus.FREE),
            )
        )

        result = self._workflow().execute(self._request(run_key="city-tavern-both"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["SILVER", "GOLD"], result.result["selected_chest_types"])
        self.assertIn("open:SILVER", self.driver.calls)
        self.assertIn("open:GOLD", self.driver.calls)
        self.assertIn("close_reward:SILVER", self.driver.calls)
        self.assertIn("close_reward:GOLD", self.driver.calls)

    def test_both_free_chests_respect_gold_only_policy(self) -> None:
        self.driver.scan = TavernChestScan(
            (
                _chest(TavernChestType.SILVER, TavernChestStatus.FREE),
                _chest(TavernChestType.GOLD, TavernChestStatus.FREE),
            )
        )
        policy = TavernChestPolicy(allow_silver_free_chest=False, allow_gold_free_chest=True)

        result = self._workflow().execute(self._request(run_key="city-tavern-gold-only", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["GOLD"], result.result["selected_chest_types"])
        self.assertEqual("chest_type_not_allowed_by_policy", result.result["ignored_chests"][0]["ignored_reason"])
        self.assertIn("open:GOLD", self.driver.calls)
        self.assertNotIn("open:SILVER", self.driver.calls)

    def test_no_free_chest_returns_skipped(self) -> None:
        self.driver.scan = TavernChestScan(
            (
                _chest(TavernChestType.SILVER, TavernChestStatus.COOLDOWN, free_indicator_visible=False),
                _chest(TavernChestType.GOLD, TavernChestStatus.UNAVAILABLE, free_indicator_visible=False),
            ),
            screenshot_path="runtime/screens/no-free.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("city-tavern-none"), run_key="city-tavern-none")
        )

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("select_free_chest", result.result["terminal_state"])
        self.assertEqual("No free tavern chest is available.", result.result["skipped_reason"])
        self.assertFalse(any(call.startswith("open:") for call in self.driver.calls))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("completed", run.status)  # type: ignore[union-attr]

    def test_paid_or_key_only_options_are_not_clicked(self) -> None:
        self.driver.scan = TavernChestScan(
            (
                _chest(TavernChestType.SILVER, TavernChestStatus.KEY_REQUIRED, free_indicator_visible=False),
                _chest(TavernChestType.GOLD, TavernChestStatus.GEM_REQUIRED, free_indicator_visible=False),
            ),
            screenshot_path="runtime/screens/paid-only.png",
        )
        policy = TavernChestPolicy(block_when_only_paid_or_key_options=True)

        result = self._workflow().execute(self._request(run_key="city-tavern-paid-only", policy=policy))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("select_free_chest", result.result["terminal_state"])
        self.assertIn("Only paid", result.result["terminal_reason"])
        self.assertFalse(any(call.startswith("open:") for call in self.driver.calls))
        self.assertEqual("key_spending_not_allowed", result.result["ignored_chests"][0]["ignored_reason"])
        self.assertEqual("gem_spending_not_allowed", result.result["ignored_chests"][1]["ignored_reason"])

    def test_ambiguous_paid_confirmation_hard_stops_safely(self) -> None:
        self.driver.open_results[TavernChestType.SILVER] = TavernChestOpenResult(
            True,
            changed=False,
            confirmation=TavernChestConfirmation.UNKNOWN,
            screenshot_path="runtime/screens/ambiguous-confirm.png",
        )

        result = self._workflow().execute(self._request(run_key="city-tavern-ambiguous"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("open_free_chest", result.result["terminal_state"])
        self.assertIn("UNKNOWN", result.result["terminal_reason"])
        self.assertIn("open:SILVER", self.driver.calls)
        self.assertNotIn("close_reward:SILVER", self.driver.calls)
        self.assertNotIn("verify:SILVER", self.driver.calls)
        self.assertEqual(
            "runtime/screens/ambiguous-confirm.png",
            result.result["failure_evidence"]["screenshot_path"],
        )

    def test_reward_ui_is_closed_after_opening(self) -> None:
        result = self._workflow().execute(self._request(run_key="city-tavern-reward-close"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["close_reward:SILVER"], [call for call in self.driver.calls if call.startswith("close_reward")])
        self.assertEqual("runtime/screens/reward-closed.png", result.result["reward_ui_handling"][0]["screenshot_path"])

    def test_postcondition_failure_records_failure_evidence(self) -> None:
        self.driver.verify_results[TavernChestType.SILVER] = _chest(
            TavernChestType.SILVER,
            TavernChestStatus.FREE,
            free_indicator_visible=True,
            screenshot_path="runtime/screens/silver-still-free.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("city-tavern-postcondition"), run_key="city-tavern-postcondition")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("verify_chest_state", result.result["terminal_state"])
        self.assertIn("free indicator did not change", result.result["terminal_reason"])
        self.assertEqual(
            "runtime/screens/silver-still-free.png",
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
