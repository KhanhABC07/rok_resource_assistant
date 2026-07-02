from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from rok_assistant.db.models import Job


MAX_SCHEDULER_BATCH_SIZE = 1000


class OccurrenceState(str, Enum):
    CREATED = "created"
    ALREADY_EXISTS = "already_exists"
    SKIPPED = "skipped"


class ClaimState(str, Enum):
    CLAIMED = "claimed"
    CONFLICT = "conflict"


class DispatchState(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CONFLICT = "conflict"


class StartupRecoveryState(str, Enum):
    RECOVERED = "recovered"
    SKIPPED = "skipped"
    CONFLICT = "conflict"
    FAILED = "failed"


@dataclass(frozen=True)
class SchedulerConfig:
    poll_interval_seconds: float = 5.0
    batch_size: int = 10
    stop_timeout_seconds: float = 3.0
    max_workers: int = 5
    max_active_instances: int = 5
    retry_delay_seconds: float = 600.0
    pre_launch_seconds: float = 120.0
    # Seconds before a queued/running scheduler claim is considered stale.
    stale_claim_timeout_seconds: float = 900.0

    def __post_init__(self) -> None:
        _require_positive_finite(self.poll_interval_seconds, "poll_interval_seconds")
        _require_positive_finite(self.stop_timeout_seconds, "stop_timeout_seconds")
        _require_positive_finite(
            self.stale_claim_timeout_seconds,
            "stale_claim_timeout_seconds",
        )
        _require_bounded_positive_int(self.batch_size, "batch_size")
        _require_bounded_positive_int(self.max_workers, "max_workers")
        _require_bounded_positive_int(self.max_active_instances, "max_active_instances")
        _require_nonnegative_finite(self.retry_delay_seconds, "retry_delay_seconds")
        _require_nonnegative_finite(self.pre_launch_seconds, "pre_launch_seconds")


@dataclass(frozen=True)
class ScheduleDiagnostic:
    category: str
    schedule_id: int | None
    schedule_key: str
    message: str
    workflow_key: str = ""


@dataclass(frozen=True)
class OccurrenceResult:
    state: OccurrenceState
    schedule_id: int
    schedule_key: str
    occurrence_key: str
    scheduled_for: str
    job: Job | None = None


@dataclass(frozen=True)
class ScheduleEvaluationResult:
    occurrences: tuple[OccurrenceResult, ...] = ()
    diagnostics: tuple[ScheduleDiagnostic, ...] = ()

    @property
    def created_count(self) -> int:
        return sum(
            1 for occurrence in self.occurrences if occurrence.state == OccurrenceState.CREATED
        )


@dataclass(frozen=True)
class JobClaimResult:
    state: ClaimState
    job_id: int
    job: Job | None = None


@dataclass(frozen=True)
class JobDispatchRecord:
    state: DispatchState
    job_id: int
    reason: str = ""


@dataclass(frozen=True)
class SchedulerRunResult:
    evaluation: ScheduleEvaluationResult = field(default_factory=ScheduleEvaluationResult)
    due_jobs_found: int = 0
    claims: tuple[JobClaimResult, ...] = ()
    dispatches: tuple[JobDispatchRecord, ...] = ()
    iteration_failed: bool = False


@dataclass(frozen=True)
class StartupRecoveryRecord:
    state: StartupRecoveryState
    job_id: int
    previous_status: str
    new_status: str = ""
    reason: str = ""


@dataclass(frozen=True)
class StartupReconciliationResult:
    records: tuple[StartupRecoveryRecord, ...] = ()

    @property
    def failed_count(self) -> int:
        return sum(1 for record in self.records if record.state == StartupRecoveryState.FAILED)

    @property
    def recovered_count(self) -> int:
        return sum(
            1 for record in self.records if record.state == StartupRecoveryState.RECOVERED
        )


def _require_positive_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite positive number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field_name} must be a finite positive number.")


def _require_nonnegative_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite nonnegative number.")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field_name} must be a finite nonnegative number.")


def _require_bounded_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value <= 0 or value > MAX_SCHEDULER_BATCH_SIZE:
        raise ValueError(
            f"{field_name} must be between 1 and {MAX_SCHEDULER_BATCH_SIZE}."
        )
