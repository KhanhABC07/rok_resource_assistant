from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
)
from rok_assistant.scheduler.clock import utc_datetime_to_text  # noqa: E402
from rok_assistant.tasks.peace_shield_workflow import (  # noqa: E402
    EMERGENCY_PEACE_SHIELD_PRIORITY,
    PEACE_SHIELD_STATES,
    PEACE_SHIELD_TEMPLATE_KEYS,
    AttackMonitorConfig,
    AttackMonitorDecision,
    AttackSignal,
    AttackSignalStatus,
    CityVerification,
    IncomingAttackMonitor,
    PeaceShieldPolicy,
    PeaceShieldRequest,
    PeaceShieldWorkflow,
    ShieldActionResult,
    ShieldInventoryScan,
    ShieldOption,
    ShieldSource,
    ShieldSpendLimit,
)
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


NOW = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)


class MutableClock:
    def __init__(self, value: datetime = NOW) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class FakeAttackDetector:
    def __init__(self, signal: AttackSignal) -> None:
        self.signal = signal

    def detect(self) -> AttackSignal:
        return self.signal


class FakeScheduler:
    def __init__(self) -> None:
        self.wake_count = 0

    def wake(self) -> None:
        self.wake_count += 1


class FakeNotifier:
    def __init__(self) -> None:
        self.notifications: list[dict[str, object]] = []

    def notify_critical(self, *, title: str, message: str, data: dict[str, object]) -> None:
        self.notifications.append({"title": title, "message": message, "data": data})


class FakePeaceShieldDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.city = CityVerification(
            verified=True,
            city_scene=True,
            active_character_id=None,
            screenshot_path="runtime/screens/city.png",
        )
        self.scan = ShieldInventoryScan(
            (
                ShieldOption(duration_hours=8, source=ShieldSource.ITEM, quantity=1),
            ),
            screenshot_path="runtime/screens/shields.png",
        )
        self.selection = ShieldActionResult(True, verified=True)
        self.application = ShieldActionResult(
            True,
            verified=True,
            screenshot_path="runtime/screens/applied.png",
        )
        self.postcondition = CityVerification(
            verified=True,
            city_scene=True,
            shield_active=True,
            screenshot_path="runtime/screens/shield-active.png",
        )

    def verify_active_city(self, _request, _character):
        self.calls.append("verify_active_city")
        return self.city

    def open_shield_menu(self, _request, _character, _policy):
        self.calls.append("open_shield_menu")
        return self.scan

    def select_shield(self, _request, _character, option, _policy):
        self.calls.append(f"select_shield:{option.duration_hours}:{option.normalized_source().value}")
        return self.selection

    def apply_shield(self, _request, _character, option, _policy):
        self.calls.append(f"apply_shield:{option.duration_hours}:{option.normalized_source().value}")
        return self.application

    def verify_shield_active(self, _request, _character, option, _policy):
        self.calls.append(f"verify_shield_active:{option.duration_hours}:{option.normalized_source().value}")
        return self.postcondition


class PeaceShieldWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "defense.sqlite3")
        self.db.initialize()
        self.instances = InstanceRepository(self.db)
        self.characters = CharacterRepository(self.db)
        self.jobs = JobRepository(self.db)
        self.job_runs = JobRunRepository(self.db)
        self.incidents = IncidentRepository(self.db)
        self.instance_id = self.instances.save(
            Instance(name="MEmu 1", instance_index=0, instance_name="MEmu 1")
        )
        self.character_id = self.characters.save(
            Character(name="Farm01", instance_id=self.instance_id)
        )
        self.driver = FakePeaceShieldDriver()
        self.notifier = FakeNotifier()

    def tearDown(self) -> None:
        self.db.close()
        self.temp_dir.cleanup()

    def _job(self, key: str) -> int:
        return self.jobs.save(
            Job(
                idempotency_key=key,
                job_type="workflow",
                scheduled_for="2026-07-09T12:00:00",
            )
        )

    def _workflow(self) -> PeaceShieldWorkflow:
        return PeaceShieldWorkflow(
            characters=self.characters,
            driver=self.driver,
            job_runs=self.job_runs,
            incidents=self.incidents,
            notifier=self.notifier,
        )

    def _request(
        self,
        *,
        policy: PeaceShieldPolicy | None = None,
        job_id: int | None = None,
        run_key: str = "shield-run",
    ) -> PeaceShieldRequest:
        return PeaceShieldRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or PeaceShieldPolicy(),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states_and_template_keys(self) -> None:
        workflow = self._workflow()

        self.assertEqual(PEACE_SHIELD_STATES, workflow.workflow_states)
        self.assertIn("city.peace_shield.active", PEACE_SHIELD_TEMPLATE_KEYS)

    def test_successful_shield_activation_persists_result(self) -> None:
        job_id = self._job("shield-success")

        result = self._workflow().execute(
            self._request(job_id=job_id, run_key="shield-success")
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(
            [
                "verify_active_city",
                "open_shield_menu",
                "select_shield:8:ITEM",
                "apply_shield:8:ITEM",
                "verify_shield_active:8:ITEM",
            ],
            self.driver.calls,
        )
        self.assertTrue(result.result["postcondition"]["shield_active"])
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("completed", run.status)  # type: ignore[union-attr]
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual("SUCCESS", payload["outcome"])

    def test_no_shield_item_available_opens_critical_incident_and_notifies(self) -> None:
        self.driver.scan = ShieldInventoryScan(())

        result = self._workflow().execute(
            self._request(job_id=self._job("shield-no-item"), run_key="shield-no-item")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("select_shield", result.result["terminal_state"])
        incidents = self.incidents.list_open()
        self.assertEqual(1, len(incidents))
        self.assertEqual("critical", incidents[0].severity)
        self.assertEqual(1, len(self.notifier.notifications))
        self.assertIn("No allowed peace shield", self.notifier.notifications[0]["message"])

    def test_policy_denial_blocks_before_navigation(self) -> None:
        policy = PeaceShieldPolicy(
            allow_inventory_items=False,
            allow_buff_activation=False,
            allow_gem_spend=False,
        )

        result = self._workflow().execute(self._request(policy=policy))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("evaluate_policy", result.result["terminal_state"])
        self.assertEqual(["verify_active_city"], self.driver.calls)

    def test_spend_limit_denial_blocks_gem_purchase(self) -> None:
        self.driver.scan = ShieldInventoryScan(
            (
                ShieldOption(
                    duration_hours=8,
                    source=ShieldSource.GEM_PURCHASE,
                    gem_cost=1000,
                ),
            )
        )
        policy = PeaceShieldPolicy(
            allow_inventory_items=False,
            allow_buff_activation=False,
            allow_gem_spend=True,
            spend_limit=ShieldSpendLimit(max_gems_per_activation=500, max_gems_per_day=500),
        )

        result = self._workflow().execute(self._request(policy=policy))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("spend_limit_denied", result.result["ignored_options"][0]["ignored_reason"])
        self.assertNotIn("apply_shield:8:GEM_PURCHASE", self.driver.calls)

    def test_manual_override_allows_gem_purchase_over_limit(self) -> None:
        self.driver.scan = ShieldInventoryScan(
            (
                ShieldOption(
                    duration_hours=8,
                    source=ShieldSource.GEM_PURCHASE,
                    gem_cost=1000,
                ),
            )
        )
        policy = PeaceShieldPolicy(
            allow_inventory_items=False,
            allow_buff_activation=False,
            allow_gem_spend=False,
            manual_override=True,
        )

        result = self._workflow().execute(self._request(policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("apply_shield:8:GEM_PURCHASE", self.driver.calls)

    def test_shield_active_postcondition_failure_notifies_operator(self) -> None:
        self.driver.postcondition = CityVerification(
            verified=True,
            city_scene=True,
            shield_active=False,
            message="Buff icon absent.",
            screenshot_path="runtime/screens/no-shield.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("shield-postcondition"), run_key="shield-postcondition")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("verify_shield_active", result.result["terminal_state"])
        self.assertEqual("failed", self.job_runs.get(result.job_run_id or 0).status)  # type: ignore[union-attr]
        self.assertEqual(1, len(self.incidents.list_open()))
        self.assertEqual(1, len(self.notifier.notifications))


class IncomingAttackMonitorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "monitor.sqlite3")
        self.db.initialize()
        self.instances = InstanceRepository(self.db)
        self.characters = CharacterRepository(self.db)
        self.jobs = JobRepository(self.db)
        self.instance_id = self.instances.save(
            Instance(name="MEmu 10", instance_index=10, instance_name="MEmu 10")
        )
        self.character_id = self.characters.save(
            Character(name="Farm20", instance_id=self.instance_id)
        )
        self.clock = MutableClock()
        self.scheduler = FakeScheduler()

    def tearDown(self) -> None:
        self.db.close()
        self.temp_dir.cleanup()

    def _signal(
        self,
        signal_id: str = "attack-1",
        *,
        status: AttackSignalStatus = AttackSignalStatus.DETECTED,
    ) -> AttackSignal:
        return AttackSignal(
            status=status,
            signal_id=signal_id,
            instance_id=self.instance_id,
            character_id=self.character_id,
            observed_at=self.clock(),
        )

    def _monitor(self, signal: AttackSignal) -> IncomingAttackMonitor:
        return IncomingAttackMonitor(
            detector=FakeAttackDetector(signal),
            jobs=self.jobs,
            scheduler=self.scheduler,
            config=AttackMonitorConfig(debounce_seconds=30, cooldown_seconds=300),
            clock=self.clock,
        )

    def test_emergency_attack_signal_enqueues_high_priority_job_and_wakes_scheduler(self) -> None:
        monitor = self._monitor(self._signal())

        result = monitor.poll()

        self.assertEqual(AttackMonitorDecision.ENQUEUED, result.decision)
        self.assertEqual(EMERGENCY_PEACE_SHIELD_PRIORITY, result.job.priority)  # type: ignore[union-attr]
        self.assertEqual("pending", result.job.status)  # type: ignore[union-attr]
        self.assertEqual(1, self.scheduler.wake_count)
        due = self.jobs.list_due_for_claim(utc_datetime_to_text(self.clock()), 10)
        self.assertEqual([result.job.id], [job.id for job in due])
        payload = json.loads(result.job.payload_json)  # type: ignore[union-attr]
        self.assertEqual("peace-shield", payload["workflow_key"])

    def test_duplicate_attack_signal_is_debounced(self) -> None:
        monitor = self._monitor(self._signal("attack-dup"))

        first = monitor.poll()
        second = monitor.poll()

        self.assertEqual(AttackMonitorDecision.ENQUEUED, first.decision)
        self.assertEqual(AttackMonitorDecision.DEBOUNCED, second.decision)
        self.assertEqual(1, len(self.jobs.list_by_status("pending")))

    def test_cooldown_blocks_new_attack_signal_for_same_target(self) -> None:
        monitor = self._monitor(self._signal("attack-1"))

        first = monitor.poll()
        monitor.detector.signal = self._signal("attack-2")  # type: ignore[attr-defined]
        self.clock.advance(31)
        second = monitor.poll()

        self.assertEqual(AttackMonitorDecision.ENQUEUED, first.decision)
        self.assertEqual(AttackMonitorDecision.COOLDOWN, second.decision)
        self.assertEqual(1, len(self.jobs.list_by_status("pending")))

    def test_false_positive_attack_detection_does_not_enqueue(self) -> None:
        monitor = self._monitor(self._signal(status=AttackSignalStatus.NOT_DETECTED))

        result = monitor.poll()

        self.assertEqual(AttackMonitorDecision.FALSE_POSITIVE, result.decision)
        self.assertEqual([], self.jobs.list_by_status("pending"))
        self.assertEqual(0, self.scheduler.wake_count)


if __name__ == "__main__":
    unittest.main()
