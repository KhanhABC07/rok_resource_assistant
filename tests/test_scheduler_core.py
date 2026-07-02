from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tests.db_helpers import SRC_ROOT  # noqa: F401

from rok_assistant.db.database import Database
from rok_assistant.db.models import (
    AutomationProfile,
    Character,
    Instance,
    Job,
    ScheduleDefinition,
    ScheduledTask,
    WorkflowDefinition,
    WorkflowStep,
)
from rok_assistant.db.repositories import (
    AutomationProfileRepository,
    CharacterRepository,
    InstanceRepository,
    JobRepository,
    ScheduleDefinitionRepository,
    TaskRepository,
    WorkflowDefinitionRepository,
    WorkflowStepRepository,
)
from rok_assistant.scheduler import (
    Scheduler,
    SchedulerConfig,
    SchedulerService,
    SchedulerStartupReconciliationError,
    SchedulerStartupReconciler,
    WorkerPool,
)
from rok_assistant.scheduler.claiming import JobClaimer
from rok_assistant.scheduler.clock import utc_datetime_to_text
from rok_assistant.scheduler.dispatcher import DispatchResult, WorkflowDispatchRequest
from rok_assistant.scheduler.models import (
    ClaimState,
    DispatchState,
    OccurrenceState,
    StartupReconciliationResult,
)
from rok_assistant.scheduler.planner import SchedulePlanner, build_occurrence_key


NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


class FakeClock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now
        self.monotonic_value = 0.0

    def utc_now(self) -> datetime:
        return self.now

    def monotonic(self) -> float:
        return self.monotonic_value


class RecordingDispatcher:
    def __init__(
        self,
        result: DispatchResult | None = None,
        *,
        db: Database | None = None,
    ) -> None:
        self.result = result or DispatchResult.accepted_result()
        self.db = db
        self.requests: list[WorkflowDispatchRequest] = []
        self.transaction_depths: list[int] = []

    def submit(self, request: WorkflowDispatchRequest) -> DispatchResult:
        self.requests.append(request)
        if self.db is not None:
            self.transaction_depths.append(self.db._transaction_depth)  # type: ignore[attr-defined]
        return self.result


class MutatingAcceptedDispatcher(RecordingDispatcher):
    def __init__(self, db: Database) -> None:
        super().__init__(DispatchResult.accepted_result())
        self.db = db

    def submit(self, request: WorkflowDispatchRequest) -> DispatchResult:
        self.requests.append(request)
        self.db.execute(
            """
            UPDATE jobs
            SET payload_json = ?
            WHERE id = ?
            """,
            (json.dumps({"changed": True}, sort_keys=True), request.job_id),
        )
        return self.result


class CountingJobs:
    def __init__(
        self,
        *,
        fail_first: bool = False,
        order: list[str] | None = None,
    ) -> None:
        self.fail_first = fail_first
        self.order = order
        self.calls = 0
        self.first_call = threading.Event()
        self.second_call = threading.Event()
        self.third_call = threading.Event()
        self.thread_ids: set[int] = set()

    def list_due_for_claim(self, _scheduled_at_or_before: str, _limit: int) -> list[Job]:
        if self.order is not None:
            self.order.append("dispatch")
        self.calls += 1
        self.thread_ids.add(threading.get_ident())
        if self.calls == 1:
            self.first_call.set()
            if self.fail_first:
                raise RuntimeError("injected iteration failure")
        if self.calls >= 2:
            self.second_call.set()
        if self.calls >= 3:
            self.third_call.set()
        return []


class NoOpReconciler:
    def __init__(self, order: list[str] | None = None) -> None:
        self.order = order
        self.calls = 0

    def reconcile(self) -> StartupReconciliationResult:
        self.calls += 1
        if self.order is not None:
            self.order.append("reconcile")
        return StartupReconciliationResult()


class FailingReconciler:
    def reconcile(self) -> StartupReconciliationResult:
        raise RuntimeError("reconciliation failed")


class SchedulerCoreTest(unittest.TestCase):
    def _open_db(self, temp_dir: str) -> tuple[Database, dict[str, object]]:
        db = Database(Path(temp_dir) / "scheduler.sqlite3")
        db.initialize()
        repositories: dict[str, object] = {
            "profiles": AutomationProfileRepository(db),
            "schedules": ScheduleDefinitionRepository(db),
            "workflows": WorkflowDefinitionRepository(db),
            "steps": WorkflowStepRepository(db),
            "jobs": JobRepository(db),
            "instances": InstanceRepository(db),
            "characters": CharacterRepository(db),
            "tasks": TaskRepository(db),
        }
        return db, repositories

    def _valid_workflow(
        self,
        repositories: dict[str, object],
        *,
        workflow_key: str = "help-flow",
        enabled: bool = True,
        action_type: str = "delay",
        parameters_json: str = '{"seconds": 0.0}',
    ) -> tuple[int, int]:
        profiles = repositories["profiles"]
        workflows = repositories["workflows"]
        steps = repositories["steps"]
        profile_id = profiles.save(AutomationProfile(name=f"Profile {workflow_key}"))  # type: ignore[attr-defined]
        workflow_id = workflows.save(  # type: ignore[attr-defined]
            WorkflowDefinition(
                profile_id=profile_id,
                workflow_key=workflow_key,
                name=workflow_key,
                enabled=enabled,
                config_json='{"schema_version": 2}',
            )
        )
        steps.save(  # type: ignore[attr-defined]
            WorkflowStep(
                workflow_id=workflow_id,
                step_order=1,
                step_key="wait",
                action_type=action_type,
                parameters_json=parameters_json,
            )
        )
        return profile_id, workflow_id

    def _schedule(
        self,
        repositories: dict[str, object],
        profile_id: int,
        *,
        workflow_key: str = "help-flow",
        enabled: bool = True,
        interval_seconds: int = 3600,
        start_at: datetime = NOW,
        config: dict[str, object] | None = None,
    ) -> int:
        schedules = repositories["schedules"]
        payload = {
            "workflow_key": workflow_key,
            "workflow_version": 1,
            "start_at": utc_datetime_to_text(start_at),
            "priority": 25,
        }
        if config:
            payload.update(config)
        return schedules.save(  # type: ignore[attr-defined]
            ScheduleDefinition(
                profile_id=profile_id,
                schedule_key=f"schedule-{workflow_key}",
                name=f"Schedule {workflow_key}",
                enabled=enabled,
                interval_seconds=interval_seconds,
                config_json=json.dumps(payload, sort_keys=True),
            )
        )

    def _planner(self, repositories: dict[str, object]) -> SchedulePlanner:
        return SchedulePlanner(
            repositories["schedules"],  # type: ignore[arg-type]
            repositories["workflows"],  # type: ignore[arg-type]
            repositories["steps"],  # type: ignore[arg-type]
            repositories["jobs"],  # type: ignore[arg-type]
        )

    def _scheduler_state(
        self,
        repositories: dict[str, object],
        schedule_id: int,
    ) -> dict[str, object]:
        schedules = repositories["schedules"]
        schedule = schedules.get(schedule_id)  # type: ignore[attr-defined]
        config = json.loads(schedule.config_json)  # type: ignore[union-attr]
        state = config.get("scheduler")
        self.assertIsInstance(state, dict)
        return state

    def _job(
        self,
        jobs: JobRepository,
        key: str,
        scheduled_for: datetime,
        *,
        status: str = "pending",
        priority: int = 100,
        payload_json: str = "{}",
    ) -> int:
        job, _created = jobs.create_if_absent(
            Job(
                idempotency_key=key,
                job_type="workflow",
                status=status,
                priority=priority,
                scheduled_for=utc_datetime_to_text(scheduled_for),
                payload_json=payload_json,
            )
        )
        return int(job.id or 0)

    def _claim_payload(
        self,
        claimed_at: datetime,
        *,
        status: str,
        dispatch: dict[str, object] | None = None,
        extra: dict[str, object] | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "source": "test",
            "_scheduler": {
                "claim": {
                    "claimed_at": utc_datetime_to_text(claimed_at),
                    "status": status,
                }
            },
        }
        if dispatch is not None:
            payload["_scheduler"]["dispatch"] = dispatch  # type: ignore[index]
        if extra:
            payload.update(extra)
        return json.dumps(payload, sort_keys=True)

    def _dispatching_payload(self, claimed_at: datetime) -> str:
        return self._claim_payload(
            claimed_at,
            status="queued",
            dispatch={
                "phase": "dispatching",
                "dispatch_started_at": utc_datetime_to_text(claimed_at),
                "dispatch_key": "scheduler-dispatch:test",
            },
        )

    def _startup_reconciler(
        self,
        jobs: JobRepository,
        *,
        now: datetime = NOW,
        stale_claim_timeout_seconds: float = 900.0,
    ) -> SchedulerStartupReconciler:
        return SchedulerStartupReconciler(
            jobs,
            clock=FakeClock(now),
            config=SchedulerConfig(
                stale_claim_timeout_seconds=stale_claim_timeout_seconds
            ),
        )

    def test_aware_utc_due_boundary_and_naive_rejection(self) -> None:
        self.assertEqual("2026-01-01T12:00:00+00:00", utc_datetime_to_text(NOW))
        with self.assertRaises(ValueError):
            utc_datetime_to_text(datetime(2026, 1, 1, 12, 0))

    def test_disabled_schedule_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            profile_id, _workflow_id = self._valid_workflow(repositories)
            self._schedule(repositories, profile_id, enabled=False)

            result = self._planner(repositories).evaluate(NOW)

            self.assertEqual(0, result.created_count)
            self.assertEqual([], repositories["jobs"].list_by_status("pending"))  # type: ignore[attr-defined]
            db.close()

    def test_missing_workflow_is_rejected_before_job_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            profiles = repositories["profiles"]
            profile_id = profiles.save(AutomationProfile(name="Default"))  # type: ignore[attr-defined]
            schedule_id = self._schedule(repositories, profile_id, workflow_key="missing")
            planner = self._planner(repositories)

            result = planner.evaluate(NOW)
            again = planner.evaluate(NOW)
            state = self._scheduler_state(repositories, schedule_id)

            self.assertEqual(["missing_workflow"], [item.category for item in result.diagnostics])
            self.assertEqual([], list(again.diagnostics))
            self.assertEqual("2026-01-01T13:00:00+00:00", state["next_occurrence_at"])
            self.assertEqual("2026-01-01T12:00:00+00:00", state["last_occurrence_at"])
            self.assertEqual("2026-01-01T12:00:00+00:00", state["last_evaluated_at"])
            self.assertEqual([], repositories["jobs"].list_by_status("pending"))  # type: ignore[attr-defined]
            db.close()

    def test_disabled_workflow_is_rejected_before_job_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            profile_id, _workflow_id = self._valid_workflow(
                repositories,
                workflow_key="disabled-flow",
                enabled=False,
            )
            schedule_id = self._schedule(repositories, profile_id, workflow_key="disabled-flow")
            planner = self._planner(repositories)

            result = planner.evaluate(NOW)
            again = planner.evaluate(NOW)
            state = self._scheduler_state(repositories, schedule_id)

            self.assertEqual(["disabled_workflow"], [item.category for item in result.diagnostics])
            self.assertEqual([], list(again.diagnostics))
            self.assertEqual("2026-01-01T13:00:00+00:00", state["next_occurrence_at"])
            self.assertEqual("2026-01-01T12:00:00+00:00", state["last_occurrence_at"])
            self.assertEqual([], repositories["jobs"].list_by_status("pending"))  # type: ignore[attr-defined]
            db.close()

    def test_invalid_workflow_is_rejected_before_job_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            profile_id, _workflow_id = self._valid_workflow(
                repositories,
                workflow_key="invalid-flow",
                action_type="missing_action",
            )
            schedule_id = self._schedule(repositories, profile_id, workflow_key="invalid-flow")
            planner = self._planner(repositories)

            result = planner.evaluate(NOW)
            again = planner.evaluate(NOW)
            state = self._scheduler_state(repositories, schedule_id)

            self.assertEqual(["invalid_workflow"], [item.category for item in result.diagnostics])
            self.assertEqual([], list(again.diagnostics))
            self.assertEqual("2026-01-01T13:00:00+00:00", state["next_occurrence_at"])
            self.assertEqual("2026-01-01T12:00:00+00:00", state["last_occurrence_at"])
            self.assertEqual([], repositories["jobs"].list_by_status("pending"))  # type: ignore[attr-defined]
            db.close()

    def test_deterministic_occurrence_key_and_repeated_polling_no_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            profile_id, workflow_id = self._valid_workflow(repositories)
            schedule_id = self._schedule(repositories, profile_id)
            schedules = repositories["schedules"]
            workflows = repositories["workflows"]
            schedule = schedules.get(schedule_id)  # type: ignore[attr-defined]
            workflow = workflows.get(workflow_id)  # type: ignore[attr-defined]

            first_key = build_occurrence_key(schedule, workflow, "none", NOW)  # type: ignore[arg-type]
            second_key = build_occurrence_key(schedule, workflow, "none", NOW)  # type: ignore[arg-type]
            result = self._planner(repositories).evaluate(NOW)
            again = self._planner(repositories).evaluate(NOW)

            self.assertEqual(first_key, second_key)
            self.assertEqual(1, result.created_count)
            self.assertEqual(0, again.created_count)
            self.assertEqual(1, len(repositories["jobs"].list_by_status("pending")))  # type: ignore[attr-defined]
            db.close()

    def test_duplicate_insert_race_converges_on_one_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            ready = threading.Barrier(2)

            def create() -> tuple[int | None, bool]:
                ready.wait(timeout=5)
                job, created = jobs.create_if_absent(
                    Job(
                        idempotency_key="same-occurrence",
                        job_type="workflow",
                        scheduled_for=utc_datetime_to_text(NOW),
                        payload_json='{"source": "test"}',
                    )
                )
                return job.id, created

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _index: create(), range(2)))

            self.assertEqual({results[0][0]}, {item[0] for item in results})
            self.assertEqual([False, True], sorted(item[1] for item in results))
            self.assertEqual(1, len(jobs.list_by_status("pending")))
            db.close()

    def test_due_job_selection_filters_and_orders_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            due_now = self._job(jobs, "due-now", NOW, priority=20)
            future = self._job(jobs, "future", NOW + timedelta(seconds=1), priority=1)
            terminal = self._job(jobs, "terminal", NOW, status="completed", priority=1)
            queued = self._job(jobs, "queued", NOW, status="queued", priority=1)
            running = self._job(jobs, "running", NOW, status="running", priority=1)
            older_high_priority = self._job(jobs, "older-high", NOW - timedelta(seconds=1), priority=1)
            same_priority_later = self._job(jobs, "later", NOW, priority=1)

            due = jobs.list_due_for_claim(utc_datetime_to_text(NOW), limit=10)
            limited = jobs.list_due_for_claim(utc_datetime_to_text(NOW), limit=2)

            self.assertEqual([older_high_priority, same_priority_later, due_now], [job.id for job in due])
            self.assertEqual([older_high_priority, same_priority_later], [job.id for job in limited])
            self.assertNotIn(future, [job.id for job in due])
            self.assertNotIn(terminal, [job.id for job in due])
            self.assertNotIn(queued, [job.id for job in due])
            self.assertNotIn(running, [job.id for job in due])
            db.close()

    def test_atomic_claim_succeeds_once_and_conflicts_normally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            job_id = self._job(jobs, "claim", NOW)
            claimer = JobClaimer(jobs, clock=FakeClock())

            first = claimer.claim(job_id)
            second = claimer.claim(job_id)

            self.assertEqual(ClaimState.CLAIMED, first.state)
            self.assertEqual(ClaimState.CONFLICT, second.state)
            self.assertEqual("queued", jobs.get(job_id).status)  # type: ignore[union-attr]
            db.close()

    def test_second_concurrent_claim_loses_normally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            job_id = self._job(jobs, "concurrent-claim", NOW)
            ready = threading.Barrier(2)

            def claim() -> ClaimState:
                ready.wait(timeout=5)
                return JobClaimer(jobs, clock=FakeClock()).claim(job_id).state

            with ThreadPoolExecutor(max_workers=2) as executor:
                states = list(executor.map(lambda _index: claim(), range(2)))

            self.assertEqual([ClaimState.CLAIMED, ClaimState.CONFLICT], sorted(states, key=str))
            self.assertEqual("queued", jobs.get(job_id).status)  # type: ignore[union-attr]
            db.close()

    def test_failed_claim_transaction_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            job_id = self._job(jobs, "rollback-claim", NOW)

            with self.assertRaisesRegex(RuntimeError, "claim failure"):
                with db.transaction():
                    claimed = jobs.transition_status_if_current(
                        job_id,
                        "pending",
                        "queued",
                        claimed_at=utc_datetime_to_text(NOW),
                    )
                    self.assertIsNotNone(claimed)
                    raise RuntimeError("claim failure")

            self.assertEqual("pending", jobs.get(job_id).status)  # type: ignore[union-attr]
            db.close()

    def test_stale_queued_claim_is_recovered_to_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            job_id = self._job(
                jobs,
                "stale-queued",
                NOW,
                status="queued",
                payload_json=self._claim_payload(
                    NOW - timedelta(seconds=901),
                    status="queued",
                ),
            )

            result = self._startup_reconciler(jobs).reconcile()
            job = jobs.get(job_id)
            payload = json.loads(job.payload_json)  # type: ignore[union-attr]

            self.assertEqual(1, result.recovered_count)
            self.assertEqual("pending", job.status)  # type: ignore[union-attr]
            self.assertNotIn("claim", payload.get("_scheduler", {}))
            self.assertEqual("test", payload["source"])
            db.close()

    def test_exact_stale_threshold_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            job_id = self._job(
                jobs,
                "exact-threshold",
                NOW,
                status="queued",
                payload_json=self._claim_payload(
                    NOW - timedelta(seconds=900),
                    status="queued",
                ),
            )

            result = self._startup_reconciler(jobs).reconcile()

            self.assertEqual(1, result.recovered_count)
            self.assertEqual("pending", jobs.get(job_id).status)  # type: ignore[union-attr]
            db.close()

    def test_stale_running_claim_fails_conservatively(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            job_id = self._job(
                jobs,
                "stale-running",
                NOW,
                status="running",
                payload_json=self._claim_payload(
                    NOW - timedelta(seconds=901),
                    status="running",
                ),
            )

            result = self._startup_reconciler(jobs).reconcile()

            self.assertEqual(1, result.recovered_count)
            self.assertEqual("failed", jobs.get(job_id).status)  # type: ignore[union-attr]
            self.assertIn("replay safety is unknown", result.records[0].reason)
            db.close()

    def test_stale_dispatching_claim_fails_without_resubmission(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            job_id = self._job(
                jobs,
                "stale-dispatching",
                NOW,
                status="queued",
                payload_json=self._dispatching_payload(NOW - timedelta(seconds=901)),
            )
            dispatcher = RecordingDispatcher()

            result = self._startup_reconciler(jobs).reconcile()
            SchedulerService(
                jobs=jobs,
                dispatcher=dispatcher,
                clock=FakeClock(),
                config=SchedulerConfig(batch_size=5),
                reconciler=NoOpReconciler(),
            ).run_once()

            self.assertEqual(1, result.recovered_count)
            self.assertEqual("failed", jobs.get(job_id).status)  # type: ignore[union-attr]
            self.assertEqual([], dispatcher.requests)
            self.assertIn("ambiguous dispatch metadata", result.records[0].reason)
            db.close()

    def test_malformed_claim_or_dispatch_metadata_fails_conservatively(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            malformed_claim_id = self._job(
                jobs,
                "malformed-claim",
                NOW,
                status="queued",
                payload_json=json.dumps(
                    {
                        "source": "test",
                        "_scheduler": {"claim": {"claimed_at": "not-a-time"}},
                    },
                    sort_keys=True,
                ),
            )
            malformed_dispatch_id = self._job(
                jobs,
                "malformed-dispatch",
                NOW,
                status="queued",
                payload_json=self._claim_payload(
                    NOW - timedelta(seconds=901),
                    status="queued",
                    dispatch={"phase": "dispatching"},
                ),
            )

            result = self._startup_reconciler(jobs).reconcile()

            self.assertEqual(2, result.recovered_count)
            self.assertEqual("failed", jobs.get(malformed_claim_id).status)  # type: ignore[union-attr]
            self.assertEqual("failed", jobs.get(malformed_dispatch_id).status)  # type: ignore[union-attr]
            db.close()

    def test_fresh_active_claims_are_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            queued_id = self._job(
                jobs,
                "fresh-queued",
                NOW,
                status="queued",
                payload_json=self._claim_payload(
                    NOW - timedelta(seconds=899),
                    status="queued",
                ),
            )
            running_id = self._job(
                jobs,
                "fresh-running",
                NOW,
                status="running",
                payload_json=self._claim_payload(
                    NOW - timedelta(seconds=899),
                    status="running",
                ),
            )

            result = self._startup_reconciler(jobs).reconcile()

            self.assertEqual((), result.records)
            self.assertEqual("queued", jobs.get(queued_id).status)  # type: ignore[union-attr]
            self.assertEqual("running", jobs.get(running_id).status)  # type: ignore[union-attr]
            db.close()

    def test_terminal_jobs_are_never_recovered_or_redispatched(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            job_ids = {
                status: self._job(
                    jobs,
                    f"{status}-stale",
                    NOW,
                    status=status,
                    payload_json=self._claim_payload(
                        NOW - timedelta(seconds=3600),
                        status="running",
                    ),
                )
                for status in ("completed", "failed", "aborted", "cancelled")
            }
            dispatcher = RecordingDispatcher()

            result = self._startup_reconciler(jobs).reconcile()
            SchedulerService(
                jobs=jobs,
                dispatcher=dispatcher,
                clock=FakeClock(),
                config=SchedulerConfig(batch_size=5),
                reconciler=NoOpReconciler(),
            ).run_once()

            self.assertEqual((), result.records)
            for status, job_id in job_ids.items():
                self.assertEqual(status, jobs.get(job_id).status)  # type: ignore[union-attr]
            self.assertEqual([], dispatcher.requests)
            db.close()

    def test_concurrent_stale_recovery_succeeds_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            job_id = self._job(
                jobs,
                "concurrent-stale",
                NOW,
                status="queued",
                payload_json=self._claim_payload(
                    NOW - timedelta(seconds=901),
                    status="queued",
                ),
            )
            job = jobs.get(job_id)
            ready = threading.Barrier(2)

            def recover() -> str:
                ready.wait(timeout=5)
                return jobs.recover_stale_active_job(
                    job_id,
                    "queued",
                    job.payload_json,  # type: ignore[union-attr]
                    utc_datetime_to_text(NOW - timedelta(seconds=900)),
                    utc_datetime_to_text(NOW),
                ).state

            with ThreadPoolExecutor(max_workers=2) as executor:
                states = list(executor.map(lambda _index: recover(), range(2)))

            self.assertEqual(["conflict", "recovered"], sorted(states))
            self.assertEqual("pending", jobs.get(job_id).status)  # type: ignore[union-attr]
            db.close()

    def test_startup_reconciliation_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            job_id = self._job(
                jobs,
                "idempotent-stale",
                NOW,
                status="queued",
                payload_json=self._claim_payload(
                    NOW - timedelta(seconds=901),
                    status="queued",
                ),
            )
            reconciler = self._startup_reconciler(jobs)

            first = reconciler.reconcile()
            second = reconciler.reconcile()

            self.assertEqual(1, first.recovered_count)
            self.assertEqual(0, second.recovered_count)
            self.assertEqual((), second.records)
            self.assertEqual("pending", jobs.get(job_id).status)  # type: ignore[union-attr]
            db.close()

    def test_startup_reconciliation_runs_before_dispatch(self) -> None:
        order: list[str] = []
        jobs = CountingJobs(order=order)
        service = SchedulerService(
            jobs=jobs,  # type: ignore[arg-type]
            dispatcher=RecordingDispatcher(),
            clock=FakeClock(),
            config=SchedulerConfig(poll_interval_seconds=60.0, batch_size=1),
            reconciler=NoOpReconciler(order),
        )

        service.start()
        self.assertTrue(jobs.first_call.wait(timeout=1))
        service.stop()

        self.assertEqual(["reconcile", "dispatch"], order[:2])

    def test_reconciliation_failure_prevents_scheduler_startup(self) -> None:
        jobs = CountingJobs()
        service = SchedulerService(
            jobs=jobs,  # type: ignore[arg-type]
            dispatcher=RecordingDispatcher(),
            clock=FakeClock(),
            config=SchedulerConfig(poll_interval_seconds=60.0, batch_size=1),
            reconciler=FailingReconciler(),
        )

        with self.assertRaisesRegex(RuntimeError, "reconciliation failed"):
            service.start()

        self.assertFalse(service.is_running)
        self.assertEqual(0, jobs.calls)

    def test_restart_after_stale_running_job_does_not_duplicate_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            job_id = self._job(
                jobs,
                "restart-running",
                NOW,
                status="running",
                payload_json=self._claim_payload(
                    NOW - timedelta(seconds=901),
                    status="running",
                ),
            )
            dispatcher = RecordingDispatcher()
            service = SchedulerService(
                jobs=jobs,
                dispatcher=dispatcher,
                clock=FakeClock(),
                config=SchedulerConfig(poll_interval_seconds=60.0, batch_size=5),
            )

            service.start()
            service.stop()

            self.assertEqual("failed", jobs.get(job_id).status)  # type: ignore[union-attr]
            self.assertEqual([], dispatcher.requests)
            db.close()

    def test_illegal_repository_status_transitions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            pending_id = self._job(jobs, "illegal-pending", NOW)
            completed_id = self._job(jobs, "illegal-completed", NOW, status="completed")

            with self.assertRaises(ValueError):
                jobs.transition_status_if_current(pending_id, "pending", "completed")
            with self.assertRaises(ValueError):
                jobs.transition_status_if_current(completed_id, "completed", "pending")

            self.assertEqual("pending", jobs.get(pending_id).status)  # type: ignore[union-attr]
            self.assertEqual("completed", jobs.get(completed_id).status)  # type: ignore[union-attr]
            db.close()

    def test_dispatcher_called_outside_transaction_and_rejection_returns_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            job_id = self._job(
                jobs,
                "dispatch-rejected",
                NOW,
                payload_json=json.dumps({"source": "test"}, sort_keys=True),
            )
            dispatcher = RecordingDispatcher(DispatchResult.rejected("queue full"), db=db)
            service = SchedulerService(
                jobs=jobs,
                dispatcher=dispatcher,
                clock=FakeClock(),
                config=SchedulerConfig(batch_size=5),
            )

            result = service.run_once()

            self.assertEqual([0], dispatcher.transaction_depths)
            self.assertEqual([DispatchState.REJECTED], [item.state for item in result.dispatches])
            job = jobs.get(job_id)
            payload = json.loads(job.payload_json)  # type: ignore[union-attr]
            self.assertEqual("pending", job.status)  # type: ignore[union-attr]
            self.assertEqual({"source": "test"}, payload)
            db.close()

    def test_dispatch_acceptance_conflict_is_not_reported_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            job_id = self._job(jobs, "dispatch-accept-conflict", NOW)
            dispatcher = MutatingAcceptedDispatcher(db)
            service = SchedulerService(
                jobs=jobs,
                dispatcher=dispatcher,
                clock=FakeClock(),
                config=SchedulerConfig(batch_size=5),
            )

            result = service.run_once()

            self.assertEqual([DispatchState.CONFLICT], [item.state for item in result.dispatches])
            self.assertEqual("queued", jobs.get(job_id).status)  # type: ignore[union-attr]
            db.close()

    def test_run_once_creates_claims_and_dispatches_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            profile_id, _workflow_id = self._valid_workflow(repositories)
            self._schedule(repositories, profile_id)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            dispatcher = RecordingDispatcher()
            service = SchedulerService(
                jobs=jobs,
                planner=self._planner(repositories),
                dispatcher=dispatcher,
                clock=FakeClock(),
                config=SchedulerConfig(batch_size=5),
            )

            first = service.run_once()
            second = service.run_once()

            self.assertEqual(1, first.evaluation.created_count)
            self.assertEqual(1, first.due_jobs_found)
            self.assertEqual([DispatchState.ACCEPTED], [item.state for item in first.dispatches])
            self.assertEqual(0, second.evaluation.created_count)
            self.assertEqual(0, second.due_jobs_found)
            self.assertEqual(1, len(dispatcher.requests))
            running_jobs = jobs.list_by_status("running")
            self.assertEqual(1, len(running_jobs))
            payload = json.loads(running_jobs[0].payload_json)
            self.assertEqual("running", payload["_scheduler"]["claim"]["status"])
            self.assertEqual("accepted", payload["_scheduler"]["dispatch"]["phase"])
            self.assertEqual(utc_datetime_to_text(NOW), payload["_scheduler"]["dispatch"]["accepted_at"])
            db.close()

    def test_accepted_job_later_stale_is_not_returned_to_pending_or_redispatched(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            jobs: JobRepository = repositories["jobs"]  # type: ignore[assignment]
            profile_id, _workflow_id = self._valid_workflow(repositories)
            self._schedule(repositories, profile_id)
            first_dispatcher = RecordingDispatcher()
            SchedulerService(
                jobs=jobs,
                planner=self._planner(repositories),
                dispatcher=first_dispatcher,
                clock=FakeClock(),
                config=SchedulerConfig(batch_size=5),
            ).run_once()
            restart_dispatcher = RecordingDispatcher()
            restart = SchedulerService(
                jobs=jobs,
                dispatcher=restart_dispatcher,
                clock=FakeClock(NOW + timedelta(seconds=901)),
                config=SchedulerConfig(batch_size=5),
            )

            restart.start()
            restart.stop()

            self.assertEqual(1, len(first_dispatcher.requests))
            self.assertEqual([], restart_dispatcher.requests)
            self.assertEqual(0, len(jobs.list_by_status("pending")))
            self.assertEqual(1, len(jobs.list_by_status("failed")))
            db.close()

    def test_no_historical_explosion_and_restart_does_not_duplicate_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            profile_id, _workflow_id = self._valid_workflow(repositories)
            old_start = NOW - timedelta(days=30)
            schedule_id = self._schedule(repositories, profile_id, start_at=old_start)
            planner = self._planner(repositories)

            first = planner.evaluate(NOW)
            schedules = repositories["schedules"]
            config = json.loads(schedules.get(schedule_id).config_json)  # type: ignore[attr-defined, union-attr]
            config["scheduler"]["next_occurrence_at"] = first.occurrences[0].scheduled_for
            schedules.update_config_json(schedule_id, json.dumps(config, sort_keys=True))  # type: ignore[attr-defined]
            restart = planner.evaluate(NOW)

            self.assertEqual(1, first.created_count)
            self.assertEqual(0, restart.created_count)
            self.assertEqual([OccurrenceState.ALREADY_EXISTS], [item.state for item in restart.occurrences])
            self.assertEqual(1, len(repositories["jobs"].list_by_status("pending")))  # type: ignore[attr-defined]
            db.close()

    def test_scheduler_lifecycle_start_stop_wake_and_iteration_failure(self) -> None:
        jobs = CountingJobs(fail_first=True)
        service = SchedulerService(
            jobs=jobs,  # type: ignore[arg-type]
            dispatcher=RecordingDispatcher(),
            clock=FakeClock(),
            config=SchedulerConfig(poll_interval_seconds=60.0, batch_size=1),
            reconciler=NoOpReconciler(),
        )

        service.start()
        first_thread = service._thread  # type: ignore[attr-defined]
        service.start()
        self.assertIs(first_thread, service._thread)  # type: ignore[attr-defined]
        self.assertTrue(jobs.first_call.wait(timeout=1))
        service.wake()
        self.assertTrue(jobs.second_call.wait(timeout=1))
        service.stop()
        service.stop()

        self.assertFalse(service.is_running)
        self.assertEqual(1, len(jobs.thread_ids))

    def test_scheduler_does_not_busy_loop_when_no_jobs_are_due(self) -> None:
        jobs = CountingJobs()
        service = SchedulerService(
            jobs=jobs,  # type: ignore[arg-type]
            dispatcher=RecordingDispatcher(),
            clock=FakeClock(),
            config=SchedulerConfig(poll_interval_seconds=0.2, batch_size=1),
            reconciler=NoOpReconciler(),
        )

        service.start()
        self.assertTrue(jobs.first_call.wait(timeout=1))
        self.assertFalse(jobs.third_call.wait(timeout=0.05))
        service.stop()

        self.assertEqual(1, jobs.calls)

    def test_config_validation_rejects_invalid_values(self) -> None:
        for kwargs in (
            {"poll_interval_seconds": 0},
            {"poll_interval_seconds": True},
            {"poll_interval_seconds": float("nan")},
            {"poll_interval_seconds": float("inf")},
            {"poll_interval_seconds": float("-inf")},
            {"batch_size": 0},
            {"batch_size": True},
            {"batch_size": 1001},
            {"stale_claim_timeout_seconds": True},
            {"stale_claim_timeout_seconds": 0},
            {"stale_claim_timeout_seconds": -1},
            {"stale_claim_timeout_seconds": float("nan")},
            {"stale_claim_timeout_seconds": float("inf")},
            {"stale_claim_timeout_seconds": float("-inf")},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SchedulerConfig(**kwargs)

    def test_legacy_scheduler_imports_and_dispatch_remain_compatible(self) -> None:
        from rok_assistant import scheduler as public

        self.assertIs(public.Scheduler, Scheduler)
        self.assertIs(public.WorkerPool, WorkerPool)
        with tempfile.TemporaryDirectory() as temp_dir:
            db, repositories = self._open_db(temp_dir)
            instances: InstanceRepository = repositories["instances"]  # type: ignore[assignment]
            characters: CharacterRepository = repositories["characters"]  # type: ignore[assignment]
            tasks: TaskRepository = repositories["tasks"]  # type: ignore[assignment]
            instance_id = instances.save(Instance(name="MEmu_0", instance_name="MEmu_0"))
            character_id = characters.save(Character(name="Farm01", instance_id=instance_id))
            task_id = tasks.enqueue(
                ScheduledTask(
                    character_id=character_id,
                    task_type="alliance_help",
                    priority=1,
                    scheduled_for="2000-01-01T00:00:00",
                )
            )

            class FakeWorkerPool:
                max_workers = 1

                def __init__(self) -> None:
                    self.submitted: list[ScheduledTask] = []

                @property
                def active_count(self) -> int:
                    return 0

                def submit(self, task: ScheduledTask) -> None:
                    self.submitted.append(task)

            worker = FakeWorkerPool()
            Scheduler(tasks, worker).dispatch_due_tasks()  # type: ignore[arg-type]

            self.assertEqual([task_id], [task.id for task in worker.submitted])
            db.close()

    def test_no_lock_lease_heartbeat_or_distributed_schema_added(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db, _repositories = self._open_db(temp_dir)
            rows = db.fetch_all(
                "SELECT name FROM sqlite_schema WHERE type IN ('table', 'index')"
            )
            names = " ".join(row["name"].lower() for row in rows)
            job_columns = {
                row["name"].lower()
                for row in db.fetch_all("PRAGMA table_info(jobs)")
            }

            for forbidden in ("lease", "heartbeat", "worker_owner", "distributed_lock"):
                self.assertNotIn(forbidden, names)
                self.assertNotIn(forbidden, job_columns)
            db.close()


if __name__ == "__main__":
    unittest.main()
