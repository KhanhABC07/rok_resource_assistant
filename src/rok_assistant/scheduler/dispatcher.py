from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rok_assistant.scheduler.models import DispatchState


@dataclass(frozen=True)
class WorkflowDispatchRequest:
    job_id: int
    workflow_id: int | None
    workflow_key: str
    workflow_version: int
    schedule_id: int | None
    idempotency_key: str
    scheduled_for: str
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DispatchResult:
    state: DispatchState
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.state == DispatchState.ACCEPTED

    @classmethod
    def accepted_result(cls) -> "DispatchResult":
        return cls(DispatchState.ACCEPTED)

    @classmethod
    def rejected(cls, reason: str) -> "DispatchResult":
        return cls(DispatchState.REJECTED, reason=reason.strip())


class WorkflowDispatcher(Protocol):
    def submit(self, request: WorkflowDispatchRequest) -> DispatchResult:
        ...


class RejectingWorkflowDispatcher:
    def __init__(self, reason: str = "No workflow dispatcher configured.") -> None:
        self.reason = reason

    def submit(self, request: WorkflowDispatchRequest) -> DispatchResult:
        del request
        return DispatchResult.rejected(self.reason)
