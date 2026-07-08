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
from rok_assistant.tasks.quest_claim_workflow import (  # noqa: E402
    QUEST_CLAIM_STATES,
    QUEST_CLAIM_TEMPLATE_KEYS,
    DailyObjectiveMilestoneObservation,
    DailyObjectiveScan,
    QuestAction,
    QuestCategory,
    QuestClaimActionResult,
    QuestClaimConfig,
    QuestClaimPolicy,
    QuestClaimRequest,
    QuestClaimWorkflow,
    QuestEntryObservation,
    QuestPageScan,
    QuestPageScanStatus,
    QuestRewardCloseResult,
)
from rok_assistant.tasks.resource_search_workflow import ResourceGatheringActionResult  # noqa: E402
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


def _quest(
    category: QuestCategory,
    action: QuestAction,
    *,
    quest_id: str = "",
    title: str = "",
    completed: bool | None = None,
    page_number: int = 1,
    spend_detected: bool = False,
    screenshot_path: str = "",
) -> QuestEntryObservation:
    return QuestEntryObservation(
        category=category,
        action=action,
        quest_id=quest_id or f"{category.value.lower()}-{action.value.lower()}-{page_number}",
        title=title or f"{category.value.title()} quest",
        completed=action == QuestAction.CLAIM if completed is None else completed,
        confidence=0.94,
        target=(500, 320),
        page_number=page_number,
        spend_detected=spend_detected,
        screenshot_path=screenshot_path,
    )


def _milestone(
    action: QuestAction,
    *,
    milestone_id: str = "daily-40",
    points_required: int = 40,
    completed: bool | None = None,
    page_number: int = 1,
    spend_detected: bool = False,
    screenshot_path: str = "",
) -> DailyObjectiveMilestoneObservation:
    return DailyObjectiveMilestoneObservation(
        milestone_id=milestone_id,
        action=action,
        points_required=points_required,
        completed=action == QuestAction.CLAIM if completed is None else completed,
        confidence=0.95,
        target=(700, 220),
        page_number=page_number,
        spend_detected=spend_detected,
        screenshot_path=screenshot_path,
    )


class FakeQuestClaimDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.quest_scans: dict[tuple[QuestCategory, int], QuestPageScan] = {}
        self.daily_scans: dict[int, DailyObjectiveScan] = {}
        self.claim_result = QuestClaimActionResult(
            True,
            changed=True,
            reward_overlay_present=True,
            screenshot_path="runtime/screens/claim.png",
        )
        self.daily_claim_result = QuestClaimActionResult(
            True,
            changed=True,
            reward_overlay_present=True,
            screenshot_path="runtime/screens/daily-claim.png",
        )
        self.close_result = QuestRewardCloseResult(
            True,
            closed=True,
            screenshot_path="runtime/screens/reward-closed.png",
        )
        self.next_page_result = ResourceGatheringActionResult(True, data={"page_changed": True})

    def open_quest_ui(self, _request, _character, _policy):
        self.calls.append("open_quest_ui")
        return ResourceGatheringActionResult(True, data={"scene": "QUEST"}, screenshot_path="runtime/screens/quest.png")

    def select_quest_tab(self, _request, _character, category, _policy):
        self.calls.append(f"select_tab:{category.value}")
        return ResourceGatheringActionResult(True, data={"category": category.value})

    def scan_quest_page(self, _request, _character, category, page_number, _policy):
        self.calls.append(f"scan_quest:{category.value}:{page_number}")
        return self.quest_scans.get(
            (category, page_number),
            QuestPageScan(QuestPageScanStatus.NONE_CLAIMABLE, category, page_number=page_number),
        )

    def go_to_next_quest_page(self, _request, _character, category, page_number, _policy):
        self.calls.append(f"next_quest_page:{category.value}:{page_number}")
        return self.next_page_result

    def claim_quest(self, _request, _character, observation, _policy):
        self.calls.append(f"claim_quest:{observation.quest_id}")
        return self.claim_result

    def close_reward_overlay_for_quest(self, _request, _character, observation, _claim_result, _policy):
        self.calls.append(f"close_quest_reward:{observation.quest_id}")
        return self.close_result

    def verify_quest_claim(self, _request, _character, observation, _claim_result, _policy):
        self.calls.append(f"verify_quest:{observation.quest_id}")
        return QuestEntryObservation(
            category=observation.normalized_category(),
            action=QuestAction.NONE,
            quest_id=observation.quest_id,
            title=observation.title,
            completed=True,
            claimed=True,
            page_number=observation.page_number,
            scene_verified=True,
            screenshot_path=f"runtime/screens/{observation.quest_id}-verified.png",
        )

    def open_daily_objectives(self, _request, _character, _policy):
        self.calls.append("open_daily_objectives")
        return ResourceGatheringActionResult(True, data={"scene": "DAILY_OBJECTIVES"})

    def scan_daily_objectives(self, _request, _character, page_number, _policy):
        self.calls.append(f"scan_daily:{page_number}")
        return self.daily_scans.get(
            page_number,
            DailyObjectiveScan(QuestPageScanStatus.NONE_CLAIMABLE, page_number=page_number),
        )

    def go_to_next_daily_objectives_page(self, _request, _character, page_number, _policy):
        self.calls.append(f"next_daily_page:{page_number}")
        return self.next_page_result

    def claim_daily_milestone(self, _request, _character, milestone, _policy):
        self.calls.append(f"claim_daily:{milestone.milestone_id}")
        return self.daily_claim_result

    def close_reward_overlay_for_daily_milestone(self, _request, _character, milestone, _claim_result, _policy):
        self.calls.append(f"close_daily_reward:{milestone.milestone_id}")
        return self.close_result

    def verify_daily_milestone_claim(self, _request, _character, milestone, _claim_result, _policy):
        self.calls.append(f"verify_daily:{milestone.milestone_id}")
        return DailyObjectiveMilestoneObservation(
            milestone_id=milestone.milestone_id,
            action=QuestAction.NONE,
            points_required=milestone.points_required,
            completed=True,
            claimed=True,
            page_number=milestone.page_number,
            scene_verified=True,
            screenshot_path=f"runtime/screens/{milestone.milestone_id}-verified.png",
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


class QuestClaimWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "quest.sqlite3")
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
        self.driver = FakeQuestClaimDriver()
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

    def _workflow(
        self,
        *,
        max_quest_pages_per_tab: int = 4,
        max_daily_pages: int = 2,
    ) -> QuestClaimWorkflow:
        return QuestClaimWorkflow(
            characters=self.characters,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=QuestClaimConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
                max_quest_pages_per_tab=max_quest_pages_per_tab,
                max_daily_pages=max_daily_pages,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "quest-run",
        policy: QuestClaimPolicy | None = None,
    ) -> QuestClaimRequest:
        return QuestClaimRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or QuestClaimPolicy(),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states_and_template_keys(self) -> None:
        self.assertEqual(QUEST_CLAIM_STATES, self._workflow().workflow_states)
        self.assertIn("quest.action.claim", QUEST_CLAIM_TEMPLATE_KEYS)
        self.assertIn("quest.daily.objectives", QUEST_CLAIM_TEMPLATE_KEYS)

    def test_no_claimable_quests_returns_skipped(self) -> None:
        job_id = self._job("quest-none")

        result = self._workflow().execute(self._request(job_id=job_id, run_key="quest-none"))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("process_daily_objectives", result.result["terminal_state"])
        self.assertIn("No completed quest", result.result["skipped_reason"])
        self.assertFalse(any(call.startswith("claim_") for call in self.driver.calls))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("completed", run.status)  # type: ignore[union-attr]

    def test_main_quest_claim_is_reported_and_persisted(self) -> None:
        job_id = self._job("quest-main")
        self.driver.quest_scans[(QuestCategory.MAIN, 1)] = QuestPageScan(
            QuestPageScanStatus.READY,
            QuestCategory.MAIN,
            observations=(_quest(QuestCategory.MAIN, QuestAction.CLAIM, quest_id="main-1"),),
        )

        result = self._workflow().execute(self._request(job_id=job_id, run_key="quest-main"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["main-1"], [item["quest_id"] for item in result.result["claimed_quests_by_category"]["MAIN"]])
        self.assertIn("claim_quest:main-1", self.driver.calls)
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual("main-1", payload["result"]["claimed_quests"][0]["quest_id"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_side_quest_claim_is_reported_by_category(self) -> None:
        self.driver.quest_scans[(QuestCategory.SIDE, 1)] = QuestPageScan(
            QuestPageScanStatus.READY,
            QuestCategory.SIDE,
            observations=(_quest(QuestCategory.SIDE, QuestAction.CLAIM, quest_id="side-1"),),
        )

        result = self._workflow().execute(self._request(run_key="quest-side"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual([], result.result["claimed_quests_by_category"]["MAIN"])
        self.assertEqual("side-1", result.result["claimed_quests_by_category"]["SIDE"][0]["quest_id"])
        self.assertIn("claim_quest:side-1", self.driver.calls)

    def test_daily_objective_milestone_claim_is_reported(self) -> None:
        self.driver.daily_scans[1] = DailyObjectiveScan(
            QuestPageScanStatus.READY,
            milestones=(_milestone(QuestAction.CLAIM, milestone_id="daily-100", points_required=100),),
        )

        result = self._workflow().execute(self._request(run_key="quest-daily"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["daily-100"], result.result["claimed_daily_milestone_ids"])
        self.assertIn("claim_daily:daily-100", self.driver.calls)
        self.assertIn("verify_daily:daily-100", self.driver.calls)

    def test_mixed_claim_and_go_buttons_only_clicks_claim(self) -> None:
        self.driver.quest_scans[(QuestCategory.MAIN, 1)] = QuestPageScan(
            QuestPageScanStatus.READY,
            QuestCategory.MAIN,
            observations=(
                _quest(QuestCategory.MAIN, QuestAction.GO, quest_id="main-go", completed=False),
                _quest(QuestCategory.MAIN, QuestAction.CLAIM, quest_id="main-claim"),
            ),
        )

        result = self._workflow().execute(self._request(run_key="quest-mixed"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("claim_quest:main-claim", self.driver.calls)
        self.assertNotIn("claim_quest:main-go", self.driver.calls)
        self.assertEqual("go_action_not_allowed", result.result["ignored_actions"][0]["ignored_reason"])

    def test_multiple_pages_are_scanned_with_bounded_pagination(self) -> None:
        self.driver.quest_scans[(QuestCategory.MAIN, 1)] = QuestPageScan(
            QuestPageScanStatus.NONE_CLAIMABLE,
            QuestCategory.MAIN,
            page_number=1,
            has_next_page=True,
        )
        self.driver.quest_scans[(QuestCategory.MAIN, 2)] = QuestPageScan(
            QuestPageScanStatus.READY,
            QuestCategory.MAIN,
            page_number=2,
            observations=(_quest(QuestCategory.MAIN, QuestAction.CLAIM, quest_id="main-2", page_number=2),),
        )

        result = self._workflow(max_quest_pages_per_tab=2).execute(self._request(run_key="quest-pages"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("next_quest_page:MAIN:2", self.driver.calls)
        self.assertIn("scan_quest:MAIN:2", self.driver.calls)
        self.assertIn("claim_quest:main-2", self.driver.calls)

    def test_reward_popup_is_closed_after_claim(self) -> None:
        self.driver.quest_scans[(QuestCategory.MAIN, 1)] = QuestPageScan(
            QuestPageScanStatus.READY,
            QuestCategory.MAIN,
            observations=(_quest(QuestCategory.MAIN, QuestAction.CLAIM, quest_id="main-reward"),),
        )

        result = self._workflow().execute(self._request(run_key="quest-reward"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual("runtime/screens/reward-closed.png", result.result["reward_overlay_handling"][0]["screenshot_path"])
        self.assertIn("close_quest_reward:main-reward", self.driver.calls)

    def test_iteration_limit_records_failure_evidence(self) -> None:
        self.driver.quest_scans[(QuestCategory.MAIN, 1)] = QuestPageScan(
            QuestPageScanStatus.READY,
            QuestCategory.MAIN,
            observations=(
                _quest(QuestCategory.MAIN, QuestAction.CLAIM, quest_id="main-one"),
                _quest(
                    QuestCategory.MAIN,
                    QuestAction.CLAIM,
                    quest_id="main-two",
                    screenshot_path="runtime/screens/quest-budget.png",
                ),
            ),
        )
        policy = QuestClaimPolicy(max_claim_iterations=1)

        result = self._workflow().execute(
            self._request(
                job_id=self._job("quest-budget"),
                run_key="quest-budget",
                policy=policy,
            )
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("process_quest_page", result.result["terminal_state"])
        self.assertIn("iteration budget", result.result["terminal_reason"])
        self.assertEqual(1, result.result["claim_iterations"])
        self.assertEqual(1, len(result.result["claimed_quests"]))
        self.assertEqual(1, len(self.incidents.list_open()))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("failed", run.status)  # type: ignore[union-attr]
        self.assertIn("iteration budget", run.error_message)  # type: ignore[union-attr]

    def test_unknown_or_spend_action_hard_stops_safely(self) -> None:
        self.driver.quest_scans[(QuestCategory.MAIN, 1)] = QuestPageScan(
            QuestPageScanStatus.READY,
            QuestCategory.MAIN,
            observations=(
                _quest(
                    QuestCategory.MAIN,
                    QuestAction.SPEND,
                    quest_id="main-spend",
                    completed=False,
                    spend_detected=True,
                    screenshot_path="runtime/screens/spend.png",
                ),
            ),
        )

        result = self._workflow().execute(self._request(run_key="quest-spend"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("process_quest_page", result.result["terminal_state"])
        self.assertIn("Unsafe", result.result["terminal_reason"])
        self.assertEqual("runtime/screens/spend.png", result.result["failure_evidence"]["screenshot_path"])
        self.assertNotIn("claim_quest:main-spend", self.driver.calls)


if __name__ == "__main__":
    unittest.main()
