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
from rok_assistant.tasks.inventory_item_use_workflow import (  # noqa: E402
    INVENTORY_ITEM_USE_STATES,
    INVENTORY_ITEM_USE_TEMPLATE_KEYS,
    InventoryDeltaVerification,
    InventoryItemBudget,
    InventoryItemObservation,
    InventoryItemRarity,
    InventoryItemType,
    InventoryItemUseConfig,
    InventoryItemUsePolicy,
    InventoryItemUseRequest,
    InventoryItemUseWorkflow,
    InventoryScan,
    InventoryUseConfirmation,
)
from rok_assistant.tasks.resource_search_workflow import ResourceGatheringActionResult  # noqa: E402
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


def _item(
    item_type: InventoryItemType = InventoryItemType.RESOURCE_FOOD,
    *,
    quantity: int = 100,
    rarity: InventoryItemRarity = InventoryItemRarity.COMMON,
    recognized: bool = True,
    premium: bool = False,
    confidence: float = 0.94,
    identity_verified: bool = True,
) -> InventoryItemObservation:
    return InventoryItemObservation(
        item_type=item_type,
        item_id=f"{item_type.value.lower()}-1",
        display_name=item_type.value.replace("_", " ").title(),
        available_quantity=quantity,
        rarity=rarity,
        recognized=recognized,
        premium=premium,
        confidence=confidence,
        target=(420, 320),
        identity_verified=identity_verified,
        screenshot_path=f"runtime/screens/{item_type.value.lower()}.png",
    )


def _policy(
    *,
    budget: InventoryItemBudget | None = None,
    dry_run: bool = False,
    whitelist_type: InventoryItemType = InventoryItemType.RESOURCE_FOOD,
) -> InventoryItemUsePolicy:
    return InventoryItemUsePolicy(
        whitelist={
            whitelist_type: budget
            or InventoryItemBudget(max_per_run=10, max_per_day=30, total_budget=100)
        },
        dry_run=dry_run,
    )


class FakeInventoryDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.open_result = ResourceGatheringActionResult(
            True,
            data={"scene": "INVENTORY"},
            screenshot_path="runtime/screens/inventory.png",
        )
        self.scan = InventoryScan(
            (_item(),),
            screenshot_path="runtime/screens/inventory-scan.png",
        )
        self.verified_item = _item()
        self.select_result = ResourceGatheringActionResult(True, data={"selected": True})
        self.quantity_result = ResourceGatheringActionResult(True, data={"quantity_entered": True})
        self.confirmation = InventoryUseConfirmation(
            True,
            confirmed=True,
            screenshot_path="runtime/screens/confirmed.png",
        )
        self.delta = InventoryDeltaVerification(
            True,
            verified=True,
            before_quantity=100,
            after_quantity=95,
            used_quantity=5,
            screenshot_path="runtime/screens/delta.png",
        )

    def open_inventory(self, _request, _character, _policy):
        self.calls.append("open_inventory")
        return self.open_result

    def scan_inventory(self, _request, _character, _policy):
        self.calls.append("scan_inventory")
        return self.scan

    def select_inventory_item(self, _request, _character, item, _policy):
        self.calls.append(f"select:{item.normalized_item_type().value}")
        return self.select_result

    def verify_inventory_item_identity(self, _request, _character, item, _policy):
        self.calls.append(f"verify_identity:{item.normalized_item_type().value}")
        return self.verified_item

    def enter_item_quantity(self, _request, _character, item, quantity, _policy):
        self.calls.append(f"enter_quantity:{item.normalized_item_type().value}:{quantity}")
        return self.quantity_result

    def confirm_item_use(self, _request, _character, item, quantity, _policy):
        self.calls.append(f"confirm:{item.normalized_item_type().value}:{quantity}")
        return self.confirmation

    def verify_inventory_delta(self, _request, _character, item, quantity, _confirmation, _policy):
        self.calls.append(f"verify_delta:{item.normalized_item_type().value}:{quantity}")
        return self.delta


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


class InventoryItemUseWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "inventory.sqlite3")
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
        self.driver = FakeInventoryDriver()
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

    def _workflow(self) -> InventoryItemUseWorkflow:
        return InventoryItemUseWorkflow(
            characters=self.characters,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=InventoryItemUseConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "inventory-run",
        policy: InventoryItemUsePolicy | None = None,
        requested_quantity: int = 5,
        requested_type: InventoryItemType = InventoryItemType.RESOURCE_FOOD,
    ) -> InventoryItemUseRequest:
        return InventoryItemUseRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            requested_item_type=requested_type,
            requested_quantity=requested_quantity,
            policy=policy or _policy(),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states_and_template_keys(self) -> None:
        self.assertEqual(INVENTORY_ITEM_USE_STATES, self._workflow().workflow_states)
        self.assertIn("inventory.item.confirm", INVENTORY_ITEM_USE_TEMPLATE_KEYS)

    def test_whitelisted_item_found_and_persisted(self) -> None:
        job_id = self._job("inventory-found")

        result = self._workflow().execute(
            self._request(job_id=job_id, run_key="inventory-found")
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual("RESOURCE_FOOD", result.result["selected_item_metadata"]["item_type"])
        self.assertEqual(5, result.result["requested_quantity"])
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual("RESOURCE_FOOD", payload["result"]["selected_item_metadata"]["item_type"])
        self.assertEqual(5, payload["result"]["used_quantity"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_non_whitelisted_item_blocked(self) -> None:
        result = self._workflow().execute(
            self._request(
                run_key="inventory-not-whitelisted",
                requested_type=InventoryItemType.RESOURCE_FOOD,
                policy=_policy(whitelist_type=InventoryItemType.RESOURCE_WOOD),
            )
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("validate_input", result.result["terminal_state"])
        self.assertIn("not whitelisted", result.result["terminal_reason"])
        self.assertEqual([], self.driver.calls)

    def test_premium_or_rare_item_blocked(self) -> None:
        self.driver.scan = InventoryScan((_item(rarity=InventoryItemRarity.RARE),))
        self.driver.verified_item = _item(rarity=InventoryItemRarity.RARE)

        result = self._workflow().execute(self._request(run_key="inventory-rare"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("scan_inventory", result.result["terminal_state"])
        self.assertEqual("rare_item_blocked", result.result["ignored_items"][0]["ignored_reason"])
        self.assertNotIn("select:RESOURCE_FOOD", self.driver.calls)

        self.driver.calls.clear()
        self.driver.scan = InventoryScan((_item(rarity=InventoryItemRarity.PREMIUM, premium=True),))
        self.driver.verified_item = _item(rarity=InventoryItemRarity.PREMIUM, premium=True)

        premium_result = self._workflow().execute(self._request(run_key="inventory-premium"))

        self.assertEqual(WorkflowOutcome.BLOCKED, premium_result.outcome)
        self.assertEqual("premium_item_blocked", premium_result.result["ignored_items"][0]["ignored_reason"])
        self.assertNotIn("select:RESOURCE_FOOD", self.driver.calls)

    def test_unrecognized_item_blocked(self) -> None:
        self.driver.scan = InventoryScan((_item(recognized=False),))

        result = self._workflow().execute(self._request(run_key="inventory-unrecognized"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("unrecognized_item", result.result["ignored_items"][0]["ignored_reason"])
        self.assertFalse(any(call.startswith("select:") for call in self.driver.calls))

    def test_insufficient_quantity_blocked(self) -> None:
        self.driver.verified_item = _item(quantity=3)

        result = self._workflow().execute(
            self._request(run_key="inventory-insufficient", requested_quantity=5)
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("verify_item_identity", result.result["terminal_state"])
        self.assertIn("Insufficient", result.result["terminal_reason"])
        self.assertNotIn("enter_quantity:RESOURCE_FOOD:5", self.driver.calls)

    def test_quantity_per_run_exceeded(self) -> None:
        policy = _policy(budget=InventoryItemBudget(max_per_run=4, max_per_day=30, total_budget=100))

        result = self._workflow().execute(
            self._request(run_key="inventory-run-budget", policy=policy, requested_quantity=5)
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("quantity_per_run_exceeded", result.result["budget_status"]["reason"])

    def test_daily_quantity_budget_exceeded(self) -> None:
        policy = _policy(
            budget=InventoryItemBudget(
                max_per_run=10,
                max_per_day=30,
                total_budget=100,
                used_today=28,
            )
        )

        result = self._workflow().execute(
            self._request(run_key="inventory-daily-budget", policy=policy, requested_quantity=5)
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("daily_quantity_budget_exceeded", result.result["budget_status"]["reason"])

    def test_total_budget_exceeded(self) -> None:
        policy = _policy(
            budget=InventoryItemBudget(
                max_per_run=10,
                max_per_day=30,
                total_budget=100,
                used_total=98,
            )
        )

        result = self._workflow().execute(
            self._request(run_key="inventory-total-budget", policy=policy, requested_quantity=5)
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("total_budget_exceeded", result.result["budget_status"]["reason"])

    def test_dry_run_preview_does_not_use_item(self) -> None:
        result = self._workflow().execute(
            self._request(run_key="inventory-dry-run", policy=_policy(dry_run=True))
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertTrue(result.result["preview"]["dry_run"])
        self.assertEqual(0, result.result["used_quantity"])
        self.assertNotIn("enter_quantity:RESOURCE_FOOD:5", self.driver.calls)
        self.assertFalse(any(call.startswith("confirm:") for call in self.driver.calls))
        self.assertFalse(any(call.startswith("verify_delta:") for call in self.driver.calls))

    def test_confirmation_failure_blocks_with_evidence(self) -> None:
        self.driver.confirmation = InventoryUseConfirmation(
            False,
            confirmed=False,
            message="Confirm button disappeared.",
            retryable=False,
            screenshot_path="runtime/screens/confirm-failed.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("inventory-confirm-fail"), run_key="inventory-confirm-fail")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("confirm_usage", result.result["terminal_state"])
        self.assertEqual(
            "runtime/screens/confirm-failed.png",
            result.result["failure_evidence"]["screenshot_path"],
        )
        self.assertEqual(1, len(self.incidents.list_open()))

    def test_inventory_delta_failure_blocks(self) -> None:
        self.driver.delta = InventoryDeltaVerification(
            True,
            verified=True,
            before_quantity=100,
            after_quantity=99,
            used_quantity=1,
            screenshot_path="runtime/screens/delta-failed.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("inventory-delta-fail"), run_key="inventory-delta-fail")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("verify_inventory_delta", result.result["terminal_state"])
        self.assertIn("delta", result.result["terminal_reason"])
        self.assertEqual("failed", self.job_runs.get(result.job_run_id or 0).status)  # type: ignore[union-attr]

    def test_successful_item_usage(self) -> None:
        result = self._workflow().execute(self._request(run_key="inventory-success"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(
            [
                "open_inventory",
                "scan_inventory",
                "select:RESOURCE_FOOD",
                "verify_identity:RESOURCE_FOOD",
                "enter_quantity:RESOURCE_FOOD:5",
                "confirm:RESOURCE_FOOD:5",
                "verify_delta:RESOURCE_FOOD:5",
            ],
            self.driver.calls,
        )
        self.assertEqual(-5, result.result["inventory_delta"]["inventory_delta"])
        self.assertEqual(5, result.result["used_quantity"])


if __name__ == "__main__":
    unittest.main()
