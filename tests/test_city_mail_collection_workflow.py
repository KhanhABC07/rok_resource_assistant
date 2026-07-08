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
from rok_assistant.tasks.city_mail_collection_workflow import (  # noqa: E402
    CITY_MAIL_COLLECTION_STATES,
    CityMailActionResult,
    CityMailCollectionConfig,
    CityMailCollectionPolicy,
    CityMailCollectionRequest,
    CityMailCollectionWorkflow,
    MailAction,
    MailCategory,
    MailCategoryObservation,
    MailCategoryScan,
    MailCategoryScanStatus,
    MailConfirmationKind,
)
from rok_assistant.tasks.resource_search_workflow import ResourceGatheringActionResult  # noqa: E402
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


def _mail(
    category: MailCategory,
    *,
    unread: int,
    claimable: int = 0,
    category_id: str = "",
    page: int = 1,
) -> MailCategoryObservation:
    return MailCategoryObservation(
        category,
        unread_badge_count=unread,
        claimable_count=claimable,
        category_id=category_id or category.value.lower(),
        page_number=page,
    )


class FakeMailDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.scans: list[MailCategoryScan] = [
            MailCategoryScan(MailCategoryScanStatus.NO_MAIL),
        ]
        self.open = ResourceGatheringActionResult(True, data={"scene": "MAIL"})
        self.next_page = ResourceGatheringActionResult(True, data={"page_changed": True})
        self.open_category = CityMailActionResult(True)
        self.claim = CityMailActionResult(
            True,
            changed=True,
            confirmation_kind=MailConfirmationKind.SAFE_CLAIM,
            action=MailAction.CLAIM_ALL,
            claim_count=2,
            unread_before=3,
            unread_after=0,
        )
        self.read = CityMailActionResult(
            True,
            changed=True,
            confirmation_kind=MailConfirmationKind.SAFE_READ,
            action=MailAction.READ_ALL,
            unread_before=4,
            unread_after=0,
        )
        self.confirm = CityMailActionResult(
            True,
            changed=True,
            confirmation_kind=MailConfirmationKind.NONE,
            data={"confirmed": True},
        )
        self.verify_by_category: dict[MailCategory, CityMailActionResult] = {}

    def open_mail(self, _request, _character, _policy):
        self.calls.append("open_mail")
        return self.open

    def scan_mail_categories(self, _request, _character, _policy, page_number):
        self.calls.append(f"scan:{page_number}")
        if self.scans:
            return self.scans.pop(0)
        return MailCategoryScan(MailCategoryScanStatus.NO_MAIL)

    def go_to_next_mail_category_page(self, _request, _character, _policy, page_number):
        self.calls.append(f"next_page:{page_number}")
        return self.next_page

    def open_mail_category(self, _request, _character, observation, action, _policy):
        self.calls.append(f"open_category:{observation.normalized_category().value}:{action.value}")
        return self.open_category

    def claim_all_mail(self, _request, _character, observation, _policy):
        self.calls.append(f"claim_all:{observation.normalized_category().value}:{observation.category_id}")
        return self.claim

    def read_all_mail(self, _request, _character, observation, _policy):
        self.calls.append(f"read_all:{observation.normalized_category().value}:{observation.category_id}")
        return self.read

    def confirm_mail_action(self, _request, _character, observation, action_result, _policy):
        self.calls.append(
            "confirm:"
            f"{observation.normalized_category().value}:"
            f"{action_result.normalized_confirmation_kind().value}"
        )
        return self.confirm

    def verify_mail_postcondition(self, _request, _character, observation, action, _policy):
        category = observation.normalized_category()
        self.calls.append(f"verify:{category.value}:{action.value}")
        if category in self.verify_by_category:
            return self.verify_by_category[category]
        if action == MailAction.CLAIM_ALL:
            return CityMailActionResult(
                True,
                changed=True,
                category=category,
                action=action,
                claim_count=observation.claimable_count,
                unread_before=observation.unread_badge_count,
                unread_after=0,
            )
        return CityMailActionResult(
            True,
            changed=True,
            category=category,
            action=action,
            unread_before=observation.unread_badge_count,
            unread_after=0,
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


class CityMailCollectionWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "city-mail.sqlite3")
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
        self.driver = FakeMailDriver()
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

    def _workflow(self, *, max_category_pages: int = 4) -> CityMailCollectionWorkflow:
        return CityMailCollectionWorkflow(
            characters=self.characters,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=CityMailCollectionConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
                max_category_pages=max_category_pages,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "mail-run",
        policy: CityMailCollectionPolicy | None = None,
    ) -> CityMailCollectionRequest:
        return CityMailCollectionRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or CityMailCollectionPolicy(),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states(self) -> None:
        self.assertEqual(CITY_MAIL_COLLECTION_STATES, self._workflow().workflow_states)

    def test_no_mail_returns_skipped_without_processing_categories(self) -> None:
        job_id = self._job("mail-none")

        result = self._workflow().execute(self._request(job_id=job_id, run_key="mail-none"))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual(0, result.result["processed_category_count"])
        self.assertEqual("scan_categories", result.result["terminal_state"])
        self.assertFalse(any(call.startswith("open_category:") for call in self.driver.calls))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("completed", run.status)  # type: ignore[union-attr]
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_reward_mail_claims_all_when_postcondition_is_verified(self) -> None:
        job_id = self._job("mail-rewards")
        self.driver.scans = [
            MailCategoryScan(
                MailCategoryScanStatus.READY,
                observations=(_mail(MailCategory.REWARDS, unread=3, claimable=2, category_id="rewards-1"),),
            )
        ]

        result = self._workflow().execute(self._request(job_id=job_id, run_key="mail-rewards"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(1, result.result["processed_category_count"])
        self.assertEqual(2, result.result["total_claim_count"])
        self.assertEqual(
            ["open_mail", "scan:1", "open_category:REWARDS:CLAIM_ALL", "claim_all:REWARDS:rewards-1"],
            self.driver.calls[:4],
        )
        self.assertIn("confirm:REWARDS:SAFE_CLAIM", self.driver.calls)
        self.assertIn("verify:REWARDS:CLAIM_ALL", self.driver.calls)
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual(2, payload["result"]["total_claim_count"])

    def test_reports_mail_is_marked_read_without_claiming_rewards(self) -> None:
        self.driver.scans = [
            MailCategoryScan(
                MailCategoryScanStatus.READY,
                observations=(_mail(MailCategory.REPORTS, unread=4, category_id="reports-1"),),
            )
        ]

        result = self._workflow().execute(self._request(run_key="mail-reports"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(0, result.result["total_claim_count"])
        self.assertEqual("READ_ALL", result.result["processed_categories"][0]["action"])
        self.assertIn("read_all:REPORTS:reports-1", self.driver.calls)
        self.assertFalse(any(call.startswith("claim_all:") for call in self.driver.calls))

    def test_pagination_scans_following_pages_and_processes_whitelisted_mail(self) -> None:
        self.driver.scans = [
            MailCategoryScan(
                MailCategoryScanStatus.READY,
                observations=(_mail(MailCategory.PLAYER, unread=9, category_id="player-1", page=1),),
                has_next_page=True,
            ),
            MailCategoryScan(
                MailCategoryScanStatus.READY,
                observations=(_mail(MailCategory.REWARDS, unread=1, claimable=1, category_id="rewards-2", page=2),),
            ),
        ]

        result = self._workflow().execute(self._request(run_key="mail-pages"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("next_page:2", self.driver.calls)
        self.assertEqual(1, result.result["processed_category_count"])
        self.assertEqual("PLAYER", result.result["ignored_categories"][0]["category"])

    def test_unknown_confirmation_dialog_stops_safely_with_evidence(self) -> None:
        self.driver.scans = [
            MailCategoryScan(
                MailCategoryScanStatus.READY,
                observations=(_mail(MailCategory.REWARDS, unread=3, claimable=2, category_id="rewards-unsafe"),),
            )
        ]
        self.driver.claim = CityMailActionResult(
            True,
            confirmation_kind=MailConfirmationKind.UNKNOWN,
            message="Unknown confirmation dialog.",
            retryable=False,
            screenshot_path="runtime/screens/mail-unknown.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("mail-unknown"), run_key="mail-unknown")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("process_category", result.result["terminal_state"])
        self.assertEqual("runtime/screens/mail-unknown.png", result.result["failure_evidence"]["screenshot_path"])
        self.assertEqual(1, len(self.incidents.list_open()))
        self.assertFalse(any(call.startswith("verify:") for call in self.driver.calls))

    def test_destructive_confirmation_dialog_stops_safely_with_evidence(self) -> None:
        self.driver.scans = [
            MailCategoryScan(
                MailCategoryScanStatus.READY,
                observations=(_mail(MailCategory.REPORTS, unread=2, category_id="reports-delete"),),
            )
        ]
        self.driver.read = CityMailActionResult(
            True,
            confirmation_kind=MailConfirmationKind.DESTRUCTIVE,
            message="Delete mail confirmation detected.",
            retryable=False,
            screenshot_path="runtime/screens/mail-delete.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("mail-delete"), run_key="mail-delete")
        )

        self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
        self.assertEqual("process_category", result.result["terminal_state"])
        self.assertIn("Delete mail confirmation", result.message)
        self.assertEqual("runtime/screens/mail-delete.png", result.result["failure_evidence"]["screenshot_path"])
        self.assertEqual(1, len(self.incidents.list_open()))
        self.assertFalse(any(call.startswith("verify:") for call in self.driver.calls))

    def test_unread_badge_must_clear_for_read_all_postcondition(self) -> None:
        self.driver.scans = [
            MailCategoryScan(
                MailCategoryScanStatus.READY,
                observations=(_mail(MailCategory.REPORTS, unread=4, category_id="reports-stuck"),),
            )
        ]
        self.driver.verify_by_category[MailCategory.REPORTS] = CityMailActionResult(
            True,
            changed=False,
            category=MailCategory.REPORTS,
            action=MailAction.READ_ALL,
            unread_before=4,
            unread_after=4,
            message="Unread badge did not clear.",
            retryable=False,
            screenshot_path="runtime/screens/mail-unread-stuck.png",
        )

        result = self._workflow().execute(self._request(run_key="mail-unread-stuck"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("process_category", result.result["terminal_state"])
        self.assertIn("postcondition", result.result["terminal_reason"])
        self.assertEqual(0, result.result["processed_category_count"])

    def test_non_whitelisted_category_is_ignored(self) -> None:
        self.driver.scans = [
            MailCategoryScan(
                MailCategoryScanStatus.READY,
                observations=(_mail(MailCategory.ALLIANCE, unread=7, claimable=3, category_id="alliance-1"),),
            )
        ]

        result = self._workflow().execute(self._request(run_key="mail-ignored"))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual(0, result.result["processed_category_count"])
        self.assertEqual("ALLIANCE", result.result["ignored_categories"][0]["category"])
        self.assertFalse(any(call.startswith("claim_all:") for call in self.driver.calls))
        self.assertFalse(any(call.startswith("read_all:") for call in self.driver.calls))


if __name__ == "__main__":
    unittest.main()
