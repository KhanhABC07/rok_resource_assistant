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
from rok_assistant.tasks.vip_claim_workflow import (  # noqa: E402
    VIP_CLAIM_STATES,
    VIP_CLAIM_TEMPLATE_KEYS,
    VipClaimConfig,
    VipClaimConfirmation,
    VipClaimObservation,
    VipClaimOpenResult,
    VipClaimPolicy,
    VipClaimRequest,
    VipClaimScan,
    VipClaimStatus,
    VipClaimType,
    VipClaimWorkflow,
    VipRewardCloseResult,
)
from rok_assistant.tasks.resource_search_workflow import ResourceGatheringActionResult  # noqa: E402
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


def _target(
    target_type: VipClaimType,
    status: VipClaimStatus,
    *,
    free_indicator_visible: bool | None = None,
    screenshot_path: str = "",
) -> VipClaimObservation:
    return VipClaimObservation(
        target_type,
        status,
        confidence=0.94,
        target=(500, 300) if target_type == VipClaimType.DAILY_REWARD else (760, 300),
        free_indicator_visible=status == VipClaimStatus.FREE if free_indicator_visible is None else free_indicator_visible,
        screenshot_path=screenshot_path,
    )


class FakeVipClaimDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.open_vip_result = ResourceGatheringActionResult(
            True,
            data={"scene": "VIP"},
            screenshot_path="runtime/screens/vip.png",
        )
        self.scan = VipClaimScan(
            (
                _target(VipClaimType.DAILY_REWARD, VipClaimStatus.FREE),
                _target(VipClaimType.VIP_CHEST, VipClaimStatus.COOLDOWN),
            ),
            screenshot_path="runtime/screens/vip-scan.png",
        )
        self.open_results: dict[VipClaimType, VipClaimOpenResult] = {
            VipClaimType.DAILY_REWARD: VipClaimOpenResult(
                True,
                changed=True,
                confirmation=VipClaimConfirmation.FREE,
                screenshot_path="runtime/screens/daily_reward-opened.png",
            ),
            VipClaimType.VIP_CHEST: VipClaimOpenResult(
                True,
                changed=True,
                confirmation=VipClaimConfirmation.FREE,
                screenshot_path="runtime/screens/vip_chest-opened.png",
            ),
        }
        self.close_result = VipRewardCloseResult(
            True,
            closed=True,
            screenshot_path="runtime/screens/reward-closed.png",
        )
        self.verify_results: dict[VipClaimType, VipClaimObservation] = {
            VipClaimType.DAILY_REWARD: _target(
                VipClaimType.DAILY_REWARD,
                VipClaimStatus.COOLDOWN,
                free_indicator_visible=False,
                screenshot_path="runtime/screens/daily_reward-cooldown.png",
            ),
            VipClaimType.VIP_CHEST: _target(
                VipClaimType.VIP_CHEST,
                VipClaimStatus.COOLDOWN,
                free_indicator_visible=False,
                screenshot_path="runtime/screens/vip_chest-cooldown.png",
            ),
        }

    def open_vip_ui(self, _request, _character, _policy):
        self.calls.append("open_vip_ui")
        return self.open_vip_result

    def scan_vip_rewards(self, _request, _character, _policy):
        self.calls.append("scan_vip_rewards")
        return self.scan

    def claim_vip_reward(self, _request, _character, target, _policy):
        target_type = target.normalized_target_type()
        self.calls.append(f"claim_reward:{target_type.value}")
        return self.open_results[target_type]

    def claim_vip_chest(self, _request, _character, target, _policy):
        target_type = target.normalized_target_type()
        self.calls.append(f"claim_chest:{target_type.value}")
        return self.open_results[target_type]

    def close_reward_overlay(self, _request, _character, target, _open_result, _policy):
        self.calls.append(f"close_reward:{target.normalized_target_type().value}")
        return self.close_result

    def verify_vip_state(self, _request, _character, target, _open_result, _policy):
        target_type = target.normalized_target_type()
        self.calls.append(f"verify:{target_type.value}")
        return self.verify_results[target_type]


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


class VipClaimWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "vip.sqlite3")
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
        self.driver = FakeVipClaimDriver()
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

    def _workflow(self) -> VipClaimWorkflow:
        return VipClaimWorkflow(
            characters=self.characters,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=VipClaimConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "vip-run",
        policy: VipClaimPolicy | None = None,
    ) -> VipClaimRequest:
        return VipClaimRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or VipClaimPolicy(),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states_and_template_keys(self) -> None:
        self.assertEqual(VIP_CLAIM_STATES, self._workflow().workflow_states)
        self.assertIn("vip.daily_reward.free", VIP_CLAIM_TEMPLATE_KEYS)
        self.assertIn("vip.vip_chest.free", VIP_CLAIM_TEMPLATE_KEYS)

    def test_free_daily_reward_target_opens_and_persists_metadata(self) -> None:
        job_id = self._job("vip-daily_reward")

        result = self._workflow().execute(
            self._request(job_id=job_id, run_key="vip-daily_reward")
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["DAILY_REWARD"], result.result["selected_target_types"])
        self.assertEqual(
            [
                "open_vip_ui",
                "scan_vip_rewards",
                "claim_reward:DAILY_REWARD",
                "close_reward:DAILY_REWARD",
                "verify:DAILY_REWARD",
            ],
            self.driver.calls,
        )
        self.assertTrue(result.result["reward_overlay_handling"][0]["closed"])
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual("DAILY_REWARD", payload["result"]["selected_target_types"][0])
        self.assertEqual("COOLDOWN", payload["result"]["verification_result"]["after"]["status"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_free_vip_chest_target_opens_when_daily_reward_is_not_free(self) -> None:
        self.driver.scan = VipClaimScan(
            (
                _target(VipClaimType.DAILY_REWARD, VipClaimStatus.COOLDOWN, free_indicator_visible=False),
                _target(VipClaimType.VIP_CHEST, VipClaimStatus.FREE),
            )
        )

        result = self._workflow().execute(self._request(run_key="vip-vip_chest"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["VIP_CHEST"], result.result["selected_target_types"])
        self.assertIn("claim_chest:VIP_CHEST", self.driver.calls)
        self.assertNotIn("claim_reward:DAILY_REWARD", self.driver.calls)

    def test_both_free_targets_open_according_to_policy(self) -> None:
        self.driver.scan = VipClaimScan(
            (
                _target(VipClaimType.DAILY_REWARD, VipClaimStatus.FREE),
                _target(VipClaimType.VIP_CHEST, VipClaimStatus.FREE),
            )
        )

        result = self._workflow().execute(self._request(run_key="vip-both"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["DAILY_REWARD", "VIP_CHEST"], result.result["selected_target_types"])
        self.assertIn("claim_reward:DAILY_REWARD", self.driver.calls)
        self.assertIn("claim_chest:VIP_CHEST", self.driver.calls)
        self.assertIn("close_reward:DAILY_REWARD", self.driver.calls)
        self.assertIn("close_reward:VIP_CHEST", self.driver.calls)

    def test_both_free_targets_respect_vip_chest_only_policy(self) -> None:
        self.driver.scan = VipClaimScan(
            (
                _target(VipClaimType.DAILY_REWARD, VipClaimStatus.FREE),
                _target(VipClaimType.VIP_CHEST, VipClaimStatus.FREE),
            )
        )
        policy = VipClaimPolicy(allow_daily_reward=False, allow_vip_chest=True)

        result = self._workflow().execute(self._request(run_key="vip-vip_chest-only", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["VIP_CHEST"], result.result["selected_target_types"])
        self.assertEqual("target_type_not_allowed_by_policy", result.result["ignored_targets"][0]["ignored_reason"])
        self.assertIn("claim_chest:VIP_CHEST", self.driver.calls)
        self.assertNotIn("claim_reward:DAILY_REWARD", self.driver.calls)

    def test_no_free_target_returns_skipped(self) -> None:
        self.driver.scan = VipClaimScan(
            (
                _target(VipClaimType.DAILY_REWARD, VipClaimStatus.COOLDOWN, free_indicator_visible=False),
                _target(VipClaimType.VIP_CHEST, VipClaimStatus.UNAVAILABLE, free_indicator_visible=False),
            ),
            screenshot_path="runtime/screens/no-free.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("vip-none"), run_key="vip-none")
        )

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("scan_vip_rewards", result.result["terminal_state"])
        self.assertEqual("No free VIP reward or chest is available.", result.result["skipped_reason"])
        self.assertFalse(any(call.startswith("claim_") for call in self.driver.calls))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("completed", run.status)  # type: ignore[union-attr]

    def test_paid_or_gem_only_options_are_not_clicked(self) -> None:
        self.driver.scan = VipClaimScan(
            (
                _target(VipClaimType.DAILY_REWARD, VipClaimStatus.PAID, free_indicator_visible=False),
                _target(VipClaimType.VIP_CHEST, VipClaimStatus.GEM_REQUIRED, free_indicator_visible=False),
            ),
            screenshot_path="runtime/screens/paid-only.png",
        )
        policy = VipClaimPolicy(block_when_only_paid_options=True)

        result = self._workflow().execute(self._request(run_key="vip-paid-only", policy=policy))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("scan_vip_rewards", result.result["terminal_state"])
        self.assertIn("Only paid", result.result["terminal_reason"])
        self.assertFalse(any(call.startswith("claim_") for call in self.driver.calls))
        self.assertEqual("paid_not_allowed", result.result["ignored_targets"][0]["ignored_reason"])
        self.assertEqual("gem_spending_not_allowed", result.result["ignored_targets"][1]["ignored_reason"])

    def test_ambiguous_paid_confirmation_hard_stops_safely(self) -> None:
        self.driver.open_results[VipClaimType.DAILY_REWARD] = VipClaimOpenResult(
            True,
            changed=False,
            confirmation=VipClaimConfirmation.UNKNOWN,
            screenshot_path="runtime/screens/ambiguous-confirm.png",
        )

        result = self._workflow().execute(self._request(run_key="vip-ambiguous"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("claim_vip_reward", result.result["terminal_state"])
        self.assertIn("UNKNOWN", result.result["terminal_reason"])
        self.assertIn("claim_reward:DAILY_REWARD", self.driver.calls)
        self.assertNotIn("close_reward:DAILY_REWARD", self.driver.calls)
        self.assertNotIn("verify:DAILY_REWARD", self.driver.calls)
        self.assertEqual(
            "runtime/screens/ambiguous-confirm.png",
            result.result["failure_evidence"]["screenshot_path"],
        )

    def test_navigation_failure_blocks_before_claiming(self) -> None:
        self.driver.open_vip_result = ResourceGatheringActionResult(
            False,
            message="VIP scene not verified.",
            retryable=False,
            screenshot_path="runtime/screens/vip-navigation-failed.png",
        )

        result = self._workflow().execute(self._request(run_key="vip-navigation-failed"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("open_vip_ui", result.result["terminal_state"])
        self.assertEqual(
            "runtime/screens/vip-navigation-failed.png",
            result.result["failure_evidence"]["screenshot_path"],
        )
        self.assertEqual(["open_vip_ui"], self.driver.calls)

    def test_reward_overlay_is_closed_after_claiming(self) -> None:
        result = self._workflow().execute(self._request(run_key="vip-reward-close"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["close_reward:DAILY_REWARD"], [call for call in self.driver.calls if call.startswith("close_reward")])
        self.assertEqual("runtime/screens/reward-closed.png", result.result["reward_overlay_handling"][0]["screenshot_path"])

    def test_postcondition_failure_records_failure_evidence(self) -> None:
        self.driver.verify_results[VipClaimType.DAILY_REWARD] = _target(
            VipClaimType.DAILY_REWARD,
            VipClaimStatus.FREE,
            free_indicator_visible=True,
            screenshot_path="runtime/screens/daily_reward-still-free.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("vip-postcondition"), run_key="vip-postcondition")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("verify_vip_state", result.result["terminal_state"])
        self.assertIn("free indicator did not change", result.result["terminal_reason"])
        self.assertEqual(
            "runtime/screens/daily_reward-still-free.png",
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
