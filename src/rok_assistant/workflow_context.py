from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from rok_assistant.workflow_recovery import RecoveryPhase, StepRecoveryDecision
from rok_assistant.workflow_serialization import safe_serialize_metadata, freeze_safe_metadata
from rok_assistant.workflow_types import (
    MAX_EXECUTED_STEPS,
    MAX_REPEAT_ITERATIONS,
    MAX_SUB_WORKFLOW_DEPTH,
    SemanticTemplate,
    WorkflowCancelledError,
    WorkflowDefinitionSpec,
    WorkflowExecutionResult,
    WorkflowOutcome,
    WorkflowStepResult,
    WorkflowStepSpec,
)


MAX_RUNTIME_SLEEP_CHUNK_SECONDS = 0.25


class CancellationToken:
    def __init__(self) -> None:
        self._cancelled = False
        self.reason = ""

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self, reason: str = "") -> None:
        self._cancelled = True
        self.reason = reason.strip()

    def throw_if_cancelled(self) -> None:
        if self._cancelled:
            raise WorkflowCancelledError(self.reason or "Workflow cancelled.")


@dataclass
class WorkflowDeadline:
    expires_at: float | None = None

    @classmethod
    def from_timeout(
        cls,
        timeout_seconds: float | None,
        clock: Callable[[], float],
    ) -> WorkflowDeadline:
        if timeout_seconds is None:
            return cls()
        return cls(clock() + max(0.0, float(timeout_seconds)))

    def is_expired(self, clock: Callable[[], float]) -> bool:
        return self.expires_at is not None and clock() >= self.expires_at

    def remaining(self, clock: Callable[[], float]) -> float | None:
        if self.expires_at is None:
            return None
        return max(0.0, self.expires_at - clock())

    def child(self, timeout_seconds: float | None, clock: Callable[[], float]) -> WorkflowDeadline:
        if timeout_seconds is None:
            return WorkflowDeadline(self.expires_at)
        step_expires_at = clock() + max(0.0, float(timeout_seconds))
        if self.expires_at is None:
            return WorkflowDeadline(step_expires_at)
        return WorkflowDeadline(min(self.expires_at, step_expires_at))


@dataclass
class StepBudget:
    max_steps: int = MAX_EXECUTED_STEPS
    max_depth: int = MAX_SUB_WORKFLOW_DEPTH
    max_repeat_iterations: int = MAX_REPEAT_ITERATIONS
    steps_used: int = 0

    def __post_init__(self) -> None:
        if self.max_steps < 0:
            raise ValueError("max_steps must be zero or greater.")
        if self.max_steps > MAX_EXECUTED_STEPS:
            raise ValueError(f"max_steps cannot exceed {MAX_EXECUTED_STEPS}.")
        if self.max_depth < 0:
            raise ValueError("max_depth must be zero or greater.")
        if self.max_depth > MAX_SUB_WORKFLOW_DEPTH:
            raise ValueError(f"max_depth cannot exceed {MAX_SUB_WORKFLOW_DEPTH}.")
        if self.max_repeat_iterations < 0:
            raise ValueError("max_repeat_iterations must be zero or greater.")
        if self.max_repeat_iterations > MAX_REPEAT_ITERATIONS:
            raise ValueError(
                f"max_repeat_iterations cannot exceed {MAX_REPEAT_ITERATIONS}."
            )

    def consume_step(self) -> bool:
        if self.steps_used >= self.max_steps:
            return False
        self.steps_used += 1
        return True


class TemplateResolver(Protocol):
    def __call__(self, template_key: str) -> SemanticTemplate | None:
        ...


class SceneNormalizer(Protocol):
    def __call__(self, context: WorkflowExecutionContext, step: WorkflowStepSpec) -> WorkflowStepResult:
        ...


class WorkflowResolver(Protocol):
    def __call__(self, workflow_key: str) -> WorkflowDefinitionSpec | None:
        ...


class WorkflowRunRecorder(Protocol):
    def start_workflow_run(
        self,
        workflow: WorkflowDefinitionSpec,
        context: WorkflowExecutionContext,
        started_at: str,
    ) -> int | WorkflowExecutionResult | None:
        ...

    def finish_workflow_run(
        self,
        job_run_id: int | None,
        result: WorkflowExecutionResult,
    ) -> None:
        ...

    def get_completed_step(
        self,
        job_run_id: int | None,
        step: WorkflowStepSpec,
        attempt: int,
    ) -> WorkflowStepResult | None:
        ...

    def get_step_recovery(
        self,
        job_run_id: int | None,
        step: WorkflowStepSpec,
        attempt: int,
    ) -> StepRecoveryDecision | None:
        ...

    def start_step_run(
        self,
        job_run_id: int | None,
        step: WorkflowStepSpec,
        attempt: int,
        started_at: str,
    ) -> int | None:
        ...

    def mark_step_phase(
        self,
        step_run_id: int | None,
        step: WorkflowStepSpec,
        attempt: int,
        phase: RecoveryPhase,
    ) -> None:
        ...

    def finish_step_run(
        self,
        step_run_id: int | None,
        result: WorkflowStepResult,
    ) -> None:
        ...


@dataclass
class WorkflowExecutionContext:
    action_engine: object | None = None
    sleeper: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    cancellation_token: CancellationToken = field(default_factory=CancellationToken)
    deadline: WorkflowDeadline = field(default_factory=WorkflowDeadline)
    budget: StepBudget = field(default_factory=StepBudget)
    persistence: WorkflowRunRecorder | None = None
    template_resolver: TemplateResolver | None = None
    scene_normalizer: SceneNormalizer | None = None
    workflow_resolver: WorkflowResolver | None = None
    normalizer_registry: object | None = None
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    resume_completed_steps: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)
    _result_metadata: dict[str, object] = field(default_factory=dict, repr=False)

    @property
    def result_metadata(self) -> Mapping[str, object]:
        return freeze_safe_metadata(self._result_metadata)  # type: ignore[return-value]

    def add_result_metadata(self, key: str, value: object) -> None:
        key = key.strip()
        if not key:
            raise ValueError("metadata key is required.")
        safe_value = safe_serialize_metadata({key: value}, source=f"result_metadata.{key}")
        if not safe_value.ok or not isinstance(safe_value.value, dict):
            raise ValueError("result metadata value is not safely serializable.")
        self._result_metadata[key] = safe_value.value[key]

    def with_runtime_metadata(self, metadata: Mapping[str, object]) -> WorkflowExecutionContext:
        return WorkflowExecutionContext(
            action_engine=self.action_engine,
            sleeper=self.sleeper,
            clock=self.clock,
            cancellation_token=self.cancellation_token,
            deadline=self.deadline,
            budget=self.budget,
            persistence=self.persistence,
            template_resolver=self.template_resolver,
            scene_normalizer=self.scene_normalizer,
            workflow_resolver=self.workflow_resolver,
            normalizer_registry=self.normalizer_registry,
            job_id=self.job_id,
            run_key=self.run_key,
            run_attempt=self.run_attempt,
            resume_completed_steps=self.resume_completed_steps,
            metadata=dict(metadata),
        )

    def for_step(self, deadline: WorkflowDeadline) -> WorkflowExecutionContext:
        return WorkflowExecutionContext(
            action_engine=self.action_engine,
            sleeper=self.sleeper,
            clock=self.clock,
            cancellation_token=self.cancellation_token,
            deadline=deadline,
            budget=self.budget,
            persistence=self.persistence,
            template_resolver=self.template_resolver,
            scene_normalizer=self.scene_normalizer,
            workflow_resolver=self.workflow_resolver,
            normalizer_registry=self.normalizer_registry,
            job_id=self.job_id,
            run_key=self.run_key,
            run_attempt=self.run_attempt,
            resume_completed_steps=self.resume_completed_steps,
            metadata=freeze_safe_metadata(dict(self.metadata)),  # type: ignore[arg-type]
        )

    def check_cancelled_or_expired(self) -> WorkflowOutcome | None:
        if self.cancellation_token.is_cancelled:
            return WorkflowOutcome.CANCELLED
        if self.deadline.is_expired(self.clock):
            return WorkflowOutcome.TIMEOUT
        return None

    def sleep(self, seconds: float) -> None:
        remaining = max(0.0, seconds)
        while remaining > 0:
            self.cancellation_token.throw_if_cancelled()
            deadline_remaining = self.deadline.remaining(self.clock)
            if deadline_remaining is not None and deadline_remaining <= 0:
                raise TimeoutError("Workflow deadline exceeded.")
            chunk = remaining
            if deadline_remaining is not None:
                chunk = min(chunk, deadline_remaining)
            chunk = min(chunk, MAX_RUNTIME_SLEEP_CHUNK_SECONDS)
            self.sleeper(max(0.0, chunk))
            remaining = max(0.0, remaining - chunk)
        self.cancellation_token.throw_if_cancelled()
        if self.deadline.is_expired(self.clock):
            raise TimeoutError("Workflow deadline exceeded.")
