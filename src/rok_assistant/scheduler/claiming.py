from __future__ import annotations

import logging

from rok_assistant.db.repositories import JobRepository
from rok_assistant.scheduler.clock import (
    SchedulerClock,
    SystemSchedulerClock,
    utc_datetime_to_text,
)
from rok_assistant.scheduler.models import ClaimState, JobClaimResult


class JobClaimer:
    def __init__(
        self,
        jobs: JobRepository,
        *,
        clock: SchedulerClock | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.jobs = jobs
        self.clock = clock or SystemSchedulerClock()
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def claim(self, job_id: int) -> JobClaimResult:
        claimed = self.jobs.transition_status_if_current(
            job_id,
            "pending",
            "queued",
            claimed_at=utc_datetime_to_text(self.clock.utc_now()),
        )
        if claimed is None:
            self.logger.info("scheduler_claim_conflict job_id=%s", job_id)
            return JobClaimResult(ClaimState.CONFLICT, job_id=job_id)
        self.logger.info("scheduler_claimed job_id=%s", job_id)
        return JobClaimResult(ClaimState.CLAIMED, job_id=job_id, job=claimed)

    def return_to_pending(self, job_id: int) -> bool:
        returned = self.jobs.transition_status_if_current(job_id, "queued", "pending")
        return returned is not None
