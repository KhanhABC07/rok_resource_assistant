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
from rok_assistant.tasks.alliance_gift_collection_workflow import (  # noqa: E402
    ALLIANCE_GIFT_COLLECTION_STATES,
    ALLIANCE_GIFT_COLLECTION_TEMPLATE_KEYS,
    AllianceGiftActionResult,
    AllianceGiftCollectionConfig,
    AllianceGiftCollectionPolicy,
    AllianceGiftCollectionRequest,
    AllianceGiftCollectionWorkflow,
    AllianceGiftObservation,
    AllianceGiftScanStatus,
    AllianceGiftTab,
    AllianceGiftTabScan,
)
from rok_assistant.tasks.resource_search_workflow import ResourceGatheringActionResult  # noqa: E402
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


def _gift(
    tab: AllianceGiftTab | str,
    *,
    count: int = 1,
    gift_id: str = "",
    page_number: int = 1,
    confidence: float = 0.94,
) -> AllianceGiftObservation:
    normalized = AllianceGiftTab(str(tab).strip().upper()) if not isinstance(tab, AllianceGiftTab) else tab
    return AllianceGiftObservation(
        normalized,
        claimable_count=count,
        confidence=confidence,
        gift_id=gift_id or f"{normalized.value.lower()}-{page_number}",
        page_number=page_number,
    )


class FakeAllianceGiftDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.open_alliance_result = ResourceGatheringActionResult(True, data={"scene": "ALLIANCE_HOME"})
        self.open_gifts = ResourceGatheringActionResult(True, data={"scene": "ALLIANCE_GIFTS"})
        self.scans: dict[tuple[AllianceGiftTab, int], AllianceGiftTabScan] = {
            (AllianceGiftTab.NORMAL, 1): AllianceGiftTabScan(
                AllianceGiftScanStatus.READY,
                AllianceGiftTab.NORMAL,
                page_number=1,
                observations=(_gift(AllianceGiftTab.NORMAL, count=1, gift_id="normal-1"),),
            ),
            (AllianceGiftTab.RARE, 1): AllianceGiftTabScan(
                AllianceGiftScanStatus.NONE_CLAIMABLE,
                AllianceGiftTab.RARE,
                page_number=1,
            ),
        }
        self.next_page = ResourceGatheringActionResult(True, data={"page_changed": True})
        self.claim = AllianceGiftActionResult(True, changed=True, claimed_count=1)
        self.close = AllianceGiftActionResult(True, changed=True, data={"popup_closed": True})
        self.connection = AllianceGiftActionResult(True, changed=True, data={"connection_popup_closed": True})
        self.verify = AllianceGiftActionResult(
            True,
            changed=True,
            claimed_count=1,
            claimable_remaining=False,
            data={"claim_button_visible": False},
        )

    def open_alliance(self, _request, _character, _policy):
        self.calls.append("open_alliance")
        return self.open_alliance_result

    def open_alliance_gifts(self, _request, _character, _policy):
        self.calls.append("open_alliance_gifts")
        return self.open_gifts

    def scan_gift_tab(self, _request, _character, tab, page_number, _policy):
        self.calls.append(f"scan:{tab.value}:{page_number}")
        return self.scans.get(
            (tab, page_number),
            AllianceGiftTabScan(AllianceGiftScanStatus.NONE_CLAIMABLE, tab, page_number=page_number),
        )

    def go_to_next_gift_page(self, _request, _character, tab, page_number, _policy):
        self.calls.append(f"next_page:{tab.value}:{page_number}")
        return self.next_page

    def claim_alliance_gifts(self, _request, _character, scan, _policy):
        self.calls.append(f"claim_all:{scan.normalized_tab().value}:{scan.page_number}")
        return self.claim

    def close_reward_popup(self, _request, _character, scan, _claim_result, _policy):
        self.calls.append(f"close_reward:{scan.normalized_tab().value}:{scan.page_number}")
        return self.close

    def handle_connection_popup(self, _request, _character, scan, _action_result, _policy):
        self.calls.append(f"connection:{scan.normalized_tab().value}:{scan.page_number}")
        return self.connection

    def verify_gift_claim_state(self, _request, _character, scan, _claim_result, _policy):
        self.calls.append(f"verify:{scan.normalized_tab().value}:{scan.page_number}")
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


class AllianceGiftCollectionWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "alliance-gift.sqlite3")
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
        self.driver = FakeAllianceGiftDriver()
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

    def _workflow(self) -> AllianceGiftCollectionWorkflow:
        return AllianceGiftCollectionWorkflow(
            characters=self.characters,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=AllianceGiftCollectionConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "alliance-gift-run",
        policy: AllianceGiftCollectionPolicy | None = None,
    ) -> AllianceGiftCollectionRequest:
        return AllianceGiftCollectionRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or AllianceGiftCollectionPolicy(),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states_and_template_keys(self) -> None:
        self.assertEqual(ALLIANCE_GIFT_COLLECTION_STATES, self._workflow().workflow_states)
        self.assertIn("alliance.gifts.claim_all_button", ALLIANCE_GIFT_COLLECTION_TEMPLATE_KEYS)
        self.assertIn("alliance.gifts.reward_popup", ALLIANCE_GIFT_COLLECTION_TEMPLATE_KEYS)

    def test_empty_gift_state_returns_skipped(self) -> None:
        job_id = self._job("alliance-gift-empty")
        self.driver.scans = {
            (AllianceGiftTab.NORMAL, 1): AllianceGiftTabScan(
                AllianceGiftScanStatus.NONE_CLAIMABLE,
                AllianceGiftTab.NORMAL,
                page_number=1,
                message="No alliance gifts to claim.",
                screenshot_path="runtime/screens/gift-empty.png",
            ),
            (AllianceGiftTab.RARE, 1): AllianceGiftTabScan(
                AllianceGiftScanStatus.NONE_CLAIMABLE,
                AllianceGiftTab.RARE,
                page_number=1,
            ),
        }

        result = self._workflow().execute(self._request(job_id=job_id, run_key="alliance-gift-empty"))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("scan_gift_tabs", result.result["terminal_state"])
        self.assertEqual("No claimable alliance gifts are available.", result.result["skipped_reason"])
        self.assertFalse(any(call.startswith("claim_all:") for call in self.driver.calls))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("completed", run.status)  # type: ignore[union-attr]
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_normal_gift_claim_is_persisted(self) -> None:
        job_id = self._job("alliance-gift-normal")

        result = self._workflow().execute(self._request(job_id=job_id, run_key="alliance-gift-normal"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(1, result.result["claimed_count"])
        self.assertEqual(
            [
                "open_alliance",
                "open_alliance_gifts",
                "scan:NORMAL:1",
                "scan:RARE:1",
                "claim_all:NORMAL:1",
                "verify:NORMAL:1",
            ],
            self.driver.calls,
        )
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual(1, payload["result"]["claimed_count"])
        self.assertEqual("NORMAL", payload["result"]["claim_attempts"][0]["tab"])

    def test_rare_gift_claim_is_supported(self) -> None:
        self.driver.scans = {
            (AllianceGiftTab.NORMAL, 1): AllianceGiftTabScan(
                AllianceGiftScanStatus.NONE_CLAIMABLE,
                AllianceGiftTab.NORMAL,
                page_number=1,
            ),
            (AllianceGiftTab.RARE, 1): AllianceGiftTabScan(
                AllianceGiftScanStatus.READY,
                AllianceGiftTab.RARE,
                page_number=1,
                observations=(_gift(AllianceGiftTab.RARE, count=2, gift_id="rare-1"),),
            ),
        }
        self.driver.claim = AllianceGiftActionResult(True, changed=True, claimed_count=2)
        self.driver.verify = AllianceGiftActionResult(True, changed=True, claimed_count=2, claimable_remaining=False)

        result = self._workflow().execute(self._request(run_key="alliance-gift-rare"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(2, result.result["claimed_count"])
        self.assertIn("claim_all:RARE:1", self.driver.calls)

    def test_multiple_pages_and_tabs_with_claimable_gifts_are_processed(self) -> None:
        self.driver.scans = {
            (AllianceGiftTab.NORMAL, 1): AllianceGiftTabScan(
                AllianceGiftScanStatus.READY,
                AllianceGiftTab.NORMAL,
                page_number=1,
                observations=(_gift(AllianceGiftTab.NORMAL, count=1, gift_id="normal-1"),),
                has_next_page=True,
            ),
            (AllianceGiftTab.NORMAL, 2): AllianceGiftTabScan(
                AllianceGiftScanStatus.READY,
                AllianceGiftTab.NORMAL,
                page_number=2,
                observations=(_gift(AllianceGiftTab.NORMAL, count=1, gift_id="normal-2", page_number=2),),
            ),
            (AllianceGiftTab.RARE, 1): AllianceGiftTabScan(
                AllianceGiftScanStatus.READY,
                AllianceGiftTab.RARE,
                page_number=1,
                observations=(_gift(AllianceGiftTab.RARE, count=3, gift_id="rare-1"),),
            ),
        }
        self.driver.claim = AllianceGiftActionResult(True, changed=True, claimed_count=0)
        self.driver.verify = AllianceGiftActionResult(True, changed=True, claimed_count=0, claimable_remaining=False)

        result = self._workflow().execute(self._request(run_key="alliance-gift-pages"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("next_page:NORMAL:2", self.driver.calls)
        self.assertEqual(5, result.result["claimed_count"])
        self.assertEqual(3, len(result.result["claim_attempts"]))

    def test_reward_popup_is_closed_after_claim(self) -> None:
        self.driver.claim = AllianceGiftActionResult(
            True,
            changed=True,
            claimed_count=1,
            reward_popup_present=True,
        )

        result = self._workflow().execute(self._request(run_key="alliance-gift-reward-popup"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("close_reward:NORMAL:1", self.driver.calls)
        self.assertTrue(result.result["reward_popup_attempts"][0]["popup_present"])
        self.assertTrue(result.result["reward_popup_attempts"][0]["closed"])

    def test_connection_popup_is_handled_safely(self) -> None:
        self.driver.claim = AllianceGiftActionResult(
            True,
            changed=True,
            claimed_count=1,
            connection_popup_present=True,
        )

        result = self._workflow().execute(self._request(run_key="alliance-gift-connection-popup"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("connection:NORMAL:1", self.driver.calls)
        self.assertTrue(result.result["connection_popup_attempts"][0]["handled"])

    def test_claim_button_disappearing_after_success_verifies_postcondition(self) -> None:
        self.driver.verify = AllianceGiftActionResult(
            True,
            changed=True,
            claimed_count=1,
            claimable_remaining=False,
            data={"claim_button_visible": False},
        )

        result = self._workflow().execute(self._request(run_key="alliance-gift-button-gone"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertFalse(result.result["verification_attempts"][0]["claimable_remaining"])
        self.assertFalse(result.result["verification_attempts"][0]["claim_button_visible"])

    def test_claim_all_requires_verified_gifts_scene_and_tab(self) -> None:
        self.driver.scans[(AllianceGiftTab.NORMAL, 1)] = AllianceGiftTabScan(
            AllianceGiftScanStatus.READY,
            AllianceGiftTab.NORMAL,
            page_number=1,
            observations=(_gift(AllianceGiftTab.NORMAL, count=1, gift_id="normal-unverified"),),
            scene_verified=False,
            tab_verified=True,
            screenshot_path="runtime/screens/gift-unverified.png",
        )

        result = self._workflow().execute(self._request(run_key="alliance-gift-unverified"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("claim_gifts", result.result["terminal_state"])
        self.assertIn("must be verified", result.result["terminal_reason"])
        self.assertFalse(any(call.startswith("claim_all:") for call in self.driver.calls))
        self.assertEqual(
            "runtime/screens/gift-unverified.png",
            result.result["failure_evidence"]["screenshot_path"],
        )

    def test_iteration_budget_exhaustion_records_failure_evidence(self) -> None:
        self.driver.scans = {
            (AllianceGiftTab.NORMAL, 1): AllianceGiftTabScan(
                AllianceGiftScanStatus.READY,
                AllianceGiftTab.NORMAL,
                page_number=1,
                observations=(_gift(AllianceGiftTab.NORMAL, count=1, gift_id="normal-1"),),
            ),
            (AllianceGiftTab.RARE, 1): AllianceGiftTabScan(
                AllianceGiftScanStatus.READY,
                AllianceGiftTab.RARE,
                page_number=1,
                observations=(_gift(AllianceGiftTab.RARE, count=1, gift_id="rare-1"),),
                screenshot_path="runtime/screens/gift-budget.png",
            ),
        }
        policy = AllianceGiftCollectionPolicy(max_claim_iterations=1)

        result = self._workflow().execute(
            self._request(
                job_id=self._job("alliance-gift-budget"),
                run_key="alliance-gift-budget",
                policy=policy,
            )
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("claim_gifts", result.result["terminal_state"])
        self.assertIn("iteration budget", result.result["terminal_reason"])
        self.assertEqual(
            "runtime/screens/gift-budget.png",
            result.result["failure_evidence"]["screenshot_path"],
        )
        self.assertEqual(1, len(self.incidents.list_open()))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("failed", run.status)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
