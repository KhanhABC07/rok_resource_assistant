from __future__ import annotations

import json
import logging
import threading
from datetime import timedelta
from typing import Protocol

from rok_assistant.db.models import Job
from rok_assistant.db.repositories import JobRepository
from rok_assistant.scheduler.claiming import JobClaimer
from rok_assistant.scheduler.clock import (
    SchedulerClock,
    SystemSchedulerClock,
    utc_datetime_to_text,
)
from rok_assistant.scheduler.dispatcher import (
    DispatchResult,
    WorkflowDispatcher,
    WorkflowDispatchRequest,
)
from rok_assistant.scheduler.models import (
    ClaimState,
    DispatchState,
    JobClaimResult,
    JobDispatchRecord,
    ScheduleEvaluationResult,
    SchedulerConfig,
    SchedulerRunResult,
    StartupReconciliationResult,
    StartupRecoveryRecord,
    StartupRecoveryState,
)
from rok_assistant.scheduler.planner import SchedulePlanner


class StartupReconciler(Protocol):
    def reconcile(self) -> StartupReconciliationResult:
        ...


class SchedulerStartupReconciliationError(RuntimeError):
    def __init__(self, result: StartupReconciliationResult) -> None:
        self.result = result
        super().__init__("Scheduler startup reconciliation failed.")


class SchedulerStartupReconciler:
    def __init__(
        self,
        jobs: JobRepository,
        *,
        clock: SchedulerClock,
        config: SchedulerConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.jobs = jobs
        self.clock = clock
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def reconcile(self) -> StartupReconciliationResult:
        now = self.clock.utc_now()
        stale_before = now - timedelta(seconds=self.config.stale_claim_timeout_seconds)
        stale_before_text = utc_datetime_to_text(stale_before)
        recovered_at = utc_datetime_to_text(now)
        records: list[StartupRecoveryRecord] = []
        stale_jobs = self.jobs.list_stale_active_for_recovery(stale_before_text)
        self.logger.info(
            "scheduler_startup_reconciliation_started stale_before=%s candidates=%s",
            stale_before_text,
            len(stale_jobs),
        )
        for job in stale_jobs:
            if job.id is None:
                records.append(
                    StartupRecoveryRecord(
                        StartupRecoveryState.SKIPPED,
                        job_id=0,
                        previous_status=job.status,
                        reason="Unsaved job.",
                    )
                )
                continue
            try:
                result = self.jobs.recover_stale_active_job(
                    job.id,
                    job.status,
                    job.payload_json,
                    stale_before_text,
                    recovered_at,
                )
            except Exception as exc:
                record = StartupRecoveryRecord(
                    StartupRecoveryState.FAILED,
                    job_id=job.id,
                    previous_status=job.status,
                    reason=_safe_message(exc),
                )
                records.append(record)
                self.logger.error(
                    "scheduler_startup_recovery_failed job_id=%s status=%s error_type=%s message=%s",
                    job.id,
                    job.status,
                    exc.__class__.__name__,
                    record.reason,
                )
                continue
            record = StartupRecoveryRecord(
                _startup_state(result.state),
                job_id=result.job_id,
                previous_status=result.previous_status,
                new_status=result.new_status,
                reason=result.reason,
            )
            records.append(record)
            self.logger.info(
                "scheduler_startup_recovery_record state=%s job_id=%s previous_status=%s new_status=%s reason=%s",
                record.state.value,
                record.job_id,
                record.previous_status,
                record.new_status,
                record.reason,
            )
        reconciliation = StartupReconciliationResult(tuple(records))
        self.logger.info(
            "scheduler_startup_reconciliation_finished recovered=%s failed=%s",
            reconciliation.recovered_count,
            reconciliation.failed_count,
        )
        if reconciliation.failed_count:
            raise SchedulerStartupReconciliationError(reconciliation)
        return reconciliation


class SchedulerService:
    def __init__(
        self,
        *,
        jobs: JobRepository,
        dispatcher: WorkflowDispatcher,
        planner: SchedulePlanner | None = None,
        claimer: JobClaimer | None = None,
        clock: SchedulerClock | None = None,
        config: SchedulerConfig | None = None,
        reconciler: StartupReconciler | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.jobs = jobs
        self.dispatcher = dispatcher
        self.planner = planner
        self.clock = clock or SystemSchedulerClock()
        self.config = config or SchedulerConfig()
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.claimer = claimer or JobClaimer(jobs, clock=self.clock, logger=self.logger)
        self.reconciler = reconciler or SchedulerStartupReconciler(
            jobs,
            clock=self.clock,
            config=self.config,
            logger=self.logger,
        )
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._state_lock = threading.RLock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                self.logger.info("scheduler_start_ignored already_running=true")
                return
            self._stop_event.clear()
            self._wake_event.clear()
            self.reconciler.reconcile()
            thread = threading.Thread(
                target=self._run_loop,
                name="rok-scheduler-v2",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        self.logger.info(
            "scheduler_started poll_interval_seconds=%s batch_size=%s",
            self.config.poll_interval_seconds,
            self.config.batch_size,
        )

    def stop(self) -> None:
        """Request shutdown and wait at most config.stop_timeout_seconds."""

        with self._state_lock:
            thread = self._thread
            self._stop_event.set()
            self._wake_event.set()
        if thread is not None:
            thread.join(timeout=self.config.stop_timeout_seconds)
        with self._state_lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None
        self.logger.info(
            "scheduler_stopped max_wait_seconds=%s",
            self.config.stop_timeout_seconds,
        )

    def wake(self) -> None:
        self._wake_event.set()
        self.logger.info("scheduler_wake_requested")

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def dispatch_due_tasks(self) -> SchedulerRunResult:
        return self.run_once()

    def run_once(self) -> SchedulerRunResult:
        now = self.clock.utc_now()
        self.logger.info("scheduler_iteration_started now=%s", utc_datetime_to_text(now))
        evaluation = (
            self.planner.evaluate(now)
            if self.planner is not None
            else ScheduleEvaluationResult()
        )
        due_jobs = self.jobs.list_due_for_claim(
            utc_datetime_to_text(now),
            self.config.batch_size,
        )
        self.logger.info("scheduler_due_jobs_found count=%s", len(due_jobs))
        claims: list[JobClaimResult] = []
        dispatches: list[JobDispatchRecord] = []
        for job in due_jobs:
            if self._stop_event.is_set():
                break
            if job.id is None:
                continue
            claim = self.claimer.claim(job.id)
            claims.append(claim)
            if claim.state != ClaimState.CLAIMED or claim.job is None:
                continue
            dispatch = self._dispatch_claimed_job(claim.job)
            dispatches.append(dispatch)
        self.logger.info(
            "scheduler_iteration_finished created=%s due=%s claimed=%s dispatched=%s",
            evaluation.created_count,
            len(due_jobs),
            sum(1 for claim in claims if claim.state == ClaimState.CLAIMED),
            len(dispatches),
        )
        return SchedulerRunResult(
            evaluation=evaluation,
            due_jobs_found=len(due_jobs),
            claims=tuple(claims),
            dispatches=tuple(dispatches),
        )

    def _dispatch_claimed_job(self, job: Job) -> JobDispatchRecord:
        if job.id is None:
            return JobDispatchRecord(DispatchState.REJECTED, job_id=0, reason="Unsaved job.")
        dispatch_started_at = utc_datetime_to_text(self.clock.utc_now())
        dispatching = self.jobs.mark_dispatching_if_current(
            job.id,
            job.payload_json,
            dispatch_started_at=dispatch_started_at,
            dispatch_key=_dispatch_key(job),
        )
        if dispatching is None:
            self.logger.info("scheduler_dispatch_conflict job_id=%s phase=dispatching", job.id)
            return JobDispatchRecord(
                DispatchState.CONFLICT,
                job_id=job.id,
                reason="Job changed before dispatch.",
            )
        request = _dispatch_request(dispatching)
        try:
            result = self.dispatcher.submit(request)
        except Exception as exc:
            result = DispatchResult.rejected(_safe_message(exc))
        if result.accepted:
            accepted_at = utc_datetime_to_text(self.clock.utc_now())
            accepted = self.jobs.mark_dispatch_accepted_if_current(
                job.id,
                dispatching.payload_json,
                accepted_at=accepted_at,
            )
            if accepted is None:
                self.logger.info(
                    "scheduler_dispatch_conflict job_id=%s phase=accepted",
                    job.id,
                )
                return JobDispatchRecord(
                    DispatchState.CONFLICT,
                    job_id=job.id,
                    reason="Job changed before dispatch acceptance was persisted.",
                )
            self.logger.info("scheduler_dispatch_accepted job_id=%s", job.id)
            return JobDispatchRecord(DispatchState.ACCEPTED, job_id=job.id)
        returned = self.jobs.return_dispatch_rejected_to_pending_if_current(
            job.id,
            dispatching.payload_json,
        )
        self.logger.warning(
            "scheduler_dispatch_rejected job_id=%s returned_to_pending=%s",
            job.id,
            returned is not None,
        )
        return JobDispatchRecord(
            DispatchState.REJECTED,
            job_id=job.id,
            reason=result.reason,
        )

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                self.logger.error(
                    "scheduler_iteration_failed error_type=%s message=%s",
                    exc.__class__.__name__,
                    _safe_message(exc),
                )
            if self._stop_event.is_set():
                break
            self._wake_event.wait(self.config.poll_interval_seconds)
            self._wake_event.clear()
        self.logger.info("scheduler_shutdown")


def _dispatch_request(job: Job) -> WorkflowDispatchRequest:
    payload = _payload(job.payload_json)
    payload.pop("_scheduler", None)
    workflow_key = str(payload.get("workflow_key") or "")
    workflow_version = _int_payload(payload.get("workflow_version"), default=1)
    return WorkflowDispatchRequest(
        job_id=int(job.id or 0),
        workflow_id=job.workflow_id,
        workflow_key=workflow_key,
        workflow_version=workflow_version,
        schedule_id=job.schedule_id,
        idempotency_key=job.idempotency_key,
        scheduled_for=job.scheduled_for,
        payload=payload,
    )


def _dispatch_key(job: Job) -> str:
    return f"scheduler-dispatch:{job.idempotency_key}"


def _payload(payload_json: str) -> dict[str, object]:
    try:
        parsed = json.loads(payload_json or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _int_payload(value: object, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _startup_state(value: str) -> StartupRecoveryState:
    try:
        return StartupRecoveryState(value)
    except ValueError:
        return StartupRecoveryState.FAILED


def _safe_message(exc: Exception) -> str:
    message = str(exc).strip()
    if len(message) > 300:
        return message[:297] + "..."
    return message or exc.__class__.__name__
