from __future__ import annotations

import json
from contextlib import nullcontext
from typing import Any
from uuid import uuid4

from rok_assistant.db.models import JobRun, StepRun, utc_now_iso
from rok_assistant.workflow_context import WorkflowExecutionContext
from rok_assistant.workflow_recovery import (
    RecoveryPhase,
    SAFE_RESUME_PHASES,
    TERMINAL_RUN_STATUSES,
    UNCERTAIN_SIDE_EFFECT_PHASES,
    StepRecoveryDecision,
    outcome_from_payload,
    outcome_from_status,
    parse_persisted_payload,
    payload_with_recovery,
    recovery_phase_from_payload,
)
from rok_assistant.workflow_serialization import (
    safe_json_payload,
    sanitize_diagnostic_message,
)
from rok_assistant.workflow_types import (
    WorkflowDefinitionSpec,
    WorkflowExecutionResult,
    WorkflowOutcome,
    WorkflowStepResult,
    WorkflowStepSpec,
)


class WorkflowRunRepositoryRecorder:
    """Repository-backed run recorder.

    Reusing a terminal run_key returns the stored terminal result. Reusing an
    active run_key resumes only phases that are explicitly safe or recoverable.
    """

    def __init__(self, job_runs: object, step_runs: object) -> None:
        self.job_runs = job_runs
        self.step_runs = step_runs
        job_db = getattr(job_runs, "db", None)
        step_db = getattr(step_runs, "db", None)
        if job_db is not None and step_db is not None and job_db is not step_db:
            raise ValueError("job_runs and step_runs must use the same Database instance.")
        self.db = job_db if job_db is not None else step_db

    def start_workflow_run(
        self,
        workflow: WorkflowDefinitionSpec,
        context: WorkflowExecutionContext,
        started_at: str,
    ) -> int | WorkflowExecutionResult | None:
        if context.job_id is None:
            return None
        run_key = context.run_key.strip() or f"{workflow.workflow_key}-{uuid4()}"
        context.run_key = run_key
        with self._transaction():
            existing = self.job_runs.get_by_key(run_key)
            if existing is not None:
                return self._existing_workflow_start(workflow, existing, started_at)
            run_id = self.job_runs.save(
                JobRun(
                    job_id=context.job_id,
                    run_key=run_key,
                    status="running",
                    attempt=context.run_attempt,
                    started_at=started_at,
                    result_json=json.dumps(
                        self._workflow_running_payload(
                            workflow,
                            RecoveryPhase.NOT_STARTED,
                        ),
                        sort_keys=True,
                    ),
                )
            )
            self._after_workflow_start_saved(run_id)
            return run_id

    def finish_workflow_run(
        self,
        job_run_id: int | None,
        result: WorkflowExecutionResult,
    ) -> None:
        if job_run_id is None:
            return
        with self._transaction():
            existing = _require_existing_job_run(self.job_runs, job_run_id)
            self.job_runs.save(
                JobRun(
                    id=job_run_id,
                    job_id=int(existing.job_id or 0),
                    run_key=str(existing.run_key),
                    status=_status_for_outcome(result.outcome),
                    attempt=int(existing.attempt),
                    started_at=result.started_at or existing.started_at,
                    finished_at=result.finished_at or utc_now_iso(),
                    result_json=json.dumps(
                        self._workflow_result_payload(result),
                        sort_keys=True,
                    ),
                    error_message="" if result.success else result.message,
                    screenshot_path=_first_screenshot_path(result.steps),
                )
            )
            self._after_workflow_finish_saved(job_run_id)

    def get_completed_step(
        self,
        job_run_id: int | None,
        step: WorkflowStepSpec,
        attempt: int,
    ) -> WorkflowStepResult | None:
        if job_run_id is None:
            return None
        existing = self.step_runs.get_by_key(job_run_id, step.step_key, attempt)
        if existing is None:
            return None
        if existing.status != "completed":
            return None
        parsed = parse_persisted_payload(
            existing.result_json,
            source=f"step_run[{existing.id}].result_json",
        )
        if not parsed.ok:
            return _recovery_failure_step_result(step, attempt, parsed.message)
        return WorkflowStepResult(
            step_key=step.step_key,
            action_type=step.action_type,
            outcome=WorkflowOutcome.SKIPPED,
            message="Step already completed in existing run.",
            data={"resumed_from_step_run": existing.id, "previous_result": parsed.value},
            attempt=attempt,
            started_at=existing.started_at,
            finished_at=existing.finished_at or utc_now_iso(),
            screenshot_path=existing.screenshot_path,
            workflow_step_id=step.workflow_step_id,
        )

    def get_step_recovery(
        self,
        job_run_id: int | None,
        step: WorkflowStepSpec,
        attempt: int,
    ) -> StepRecoveryDecision | None:
        if job_run_id is None:
            return None
        existing = self.step_runs.get_by_key(job_run_id, step.step_key, attempt)
        if existing is None:
            return None
        parsed = parse_persisted_payload(
            existing.result_json,
            source=f"step_run[{existing.id}].result_json",
        )
        if not parsed.ok:
            return StepRecoveryDecision(
                step_run_id=int(existing.id or 0),
                phase=RecoveryPhase.SIDE_EFFECT_UNCERTAIN,
                result=_recovery_failure_step_result(step, attempt, parsed.message),
            )
        default_phase = (
            RecoveryPhase.COMPLETED
            if existing.status == "completed"
            else RecoveryPhase.SIDE_EFFECT_UNCERTAIN
        )
        phase = recovery_phase_from_payload(parsed.value, default=default_phase)
        if phase is None:
            return StepRecoveryDecision(
                step_run_id=int(existing.id or 0),
                phase=RecoveryPhase.SIDE_EFFECT_UNCERTAIN,
                result=_recovery_failure_step_result(
                    step,
                    attempt,
                    f"step_run[{existing.id}].result_json contains an invalid recovery phase.",
                ),
                payload=parsed.value,
            )
        if existing.status == "completed":
            return StepRecoveryDecision(
                step_run_id=int(existing.id or 0),
                phase=RecoveryPhase.COMPLETED,
                result=self.get_completed_step(job_run_id, step, attempt),
                payload=parsed.value,
            )
        if existing.status in TERMINAL_RUN_STATUSES:
            return StepRecoveryDecision(
                step_run_id=int(existing.id or 0),
                phase=phase,
                result=_step_result_from_payload(step, existing, parsed.value),
                payload=parsed.value,
            )
        if phase in SAFE_RESUME_PHASES:
            return StepRecoveryDecision(
                step_run_id=int(existing.id or 0),
                phase=phase,
                can_resume=True,
                payload=parsed.value,
            )
        if phase == RecoveryPhase.POSTCONDITION_VERIFIED:
            return StepRecoveryDecision(
                step_run_id=int(existing.id or 0),
                phase=phase,
                result=WorkflowStepResult(
                    step_key=step.step_key,
                    action_type=step.action_type,
                    outcome=WorkflowOutcome.SUCCESS,
                    message="Step recovered from verified postcondition.",
                    data={
                        "recovery": {
                            "phase": phase.value,
                            "resumed_from_step_run": existing.id,
                        }
                    },
                    attempt=attempt,
                    started_at=existing.started_at,
                    finished_at=utc_now_iso(),
                    screenshot_path=existing.screenshot_path,
                    workflow_step_id=step.workflow_step_id,
                ),
                payload=parsed.value,
            )
        if phase in UNCERTAIN_SIDE_EFFECT_PHASES:
            return StepRecoveryDecision(
                step_run_id=int(existing.id or 0),
                phase=phase,
                requires_postcondition=step.postcondition is not None,
                payload=parsed.value,
            )
        return StepRecoveryDecision(
            step_run_id=int(existing.id or 0),
            phase=phase,
            result=_recovery_failure_step_result(
                step,
                attempt,
                f"step_run[{existing.id}] cannot be resumed from phase {phase.value}.",
            ),
            payload=parsed.value,
        )

    def start_step_run(
        self,
        job_run_id: int | None,
        step: WorkflowStepSpec,
        attempt: int,
        started_at: str,
    ) -> int | None:
        if job_run_id is None:
            return None
        with self._transaction():
            step_run_id = self.step_runs.save(
                StepRun(
                    job_run_id=job_run_id,
                    workflow_step_id=step.workflow_step_id,
                    step_key=step.step_key,
                    status="running",
                    attempt=attempt,
                    started_at=started_at,
                    result_json=json.dumps(
                        self._step_running_payload(
                            step,
                            attempt,
                            RecoveryPhase.PRECONDITION_VERIFIED,
                        ),
                        sort_keys=True,
                    ),
                )
            )
            self._after_step_start_saved(step_run_id)
            self._update_job_running_phase(
                job_run_id,
                RecoveryPhase.PRECONDITION_VERIFIED,
                step=step,
                attempt=attempt,
            )
            return step_run_id

    def mark_step_phase(
        self,
        step_run_id: int | None,
        step: WorkflowStepSpec,
        attempt: int,
        phase: RecoveryPhase,
    ) -> None:
        if step_run_id is None:
            return
        with self._transaction():
            existing = _require_existing_step_run(self.step_runs, step_run_id)
            parsed = parse_persisted_payload(
                existing.result_json,
                source=f"step_run[{step_run_id}].result_json",
            )
            payload = parsed.value if parsed.ok else {}
            payload.update(
                {
                    "step_key": step.step_key,
                    "action_type": step.action_type,
                    "outcome": "RUNNING",
                    "attempt": attempt,
                }
            )
            self.step_runs.save(
                StepRun(
                    id=step_run_id,
                    job_run_id=existing.job_run_id,
                    workflow_step_id=step.workflow_step_id,
                    step_key=step.step_key,
                    status="running",
                    attempt=attempt,
                    started_at=existing.started_at,
                    finished_at=None,
                    result_json=json.dumps(
                        payload_with_recovery(
                            payload,
                            phase,
                            source=f"step_run[{step_run_id}].phase",
                        ),
                        sort_keys=True,
                    ),
                    error_message="",
                    screenshot_path=existing.screenshot_path,
                )
            )
            self._update_job_running_phase(
                existing.job_run_id,
                phase,
                step=step,
                attempt=attempt,
            )

    def finish_step_run(
        self,
        step_run_id: int | None,
        result: WorkflowStepResult,
    ) -> None:
        if step_run_id is None:
            return
        with self._transaction():
            existing = _require_existing_step_run(self.step_runs, step_run_id)
            phase = (
                RecoveryPhase.COMPLETED
                if result.outcome in {WorkflowOutcome.SUCCESS, WorkflowOutcome.SKIPPED}
                else RecoveryPhase.SIDE_EFFECT_UNCERTAIN
            )
            self.step_runs.save(
                StepRun(
                    id=step_run_id,
                    job_run_id=existing.job_run_id,
                    workflow_step_id=result.workflow_step_id,
                    step_key=result.step_key,
                    status=_status_for_outcome(result.outcome),
                    attempt=result.attempt,
                    started_at=result.started_at or existing.started_at,
                    finished_at=result.finished_at or utc_now_iso(),
                    result_json=json.dumps(
                        self._step_result_payload(result, phase),
                        sort_keys=True,
                    ),
                    error_message="" if result.success else result.message,
                    screenshot_path=result.screenshot_path,
                )
            )
            self._after_step_finish_saved(step_run_id)
            self._update_job_running_phase(
                existing.job_run_id,
                phase,
                result=result,
            )

    def _existing_workflow_start(
        self,
        workflow: WorkflowDefinitionSpec,
        existing: JobRun,
        started_at: str,
    ) -> int | WorkflowExecutionResult:
        parsed = parse_persisted_payload(
            existing.result_json,
            source=f"job_run[{existing.id}].result_json",
        )
        if existing.status in TERMINAL_RUN_STATUSES:
            return self._terminal_workflow_result(workflow, existing, parsed)
        if not parsed.ok:
            result = WorkflowExecutionResult(
                workflow_key=workflow.workflow_key,
                schema_version=workflow.schema_version,
                outcome=WorkflowOutcome.FATAL_FAILURE,
                message=parsed.message,
                started_at=existing.started_at or started_at,
                finished_at=utc_now_iso(),
                job_run_id=existing.id,
                result={"recovery_failure": {"message": parsed.message}},
            )
            self.job_runs.save(
                JobRun(
                    id=existing.id,
                    job_id=existing.job_id,
                    run_key=existing.run_key,
                    status="failed",
                    attempt=existing.attempt,
                    started_at=existing.started_at,
                    finished_at=result.finished_at,
                    result_json=json.dumps(self._workflow_result_payload(result), sort_keys=True),
                    error_message=result.message,
                    screenshot_path=existing.screenshot_path,
                )
            )
            return result
        phase = recovery_phase_from_payload(parsed.value, default=RecoveryPhase.NOT_STARTED)
        if phase is None:
            message = f"job_run[{existing.id}].result_json contains an invalid recovery phase."
            result = WorkflowExecutionResult(
                workflow_key=workflow.workflow_key,
                schema_version=workflow.schema_version,
                outcome=WorkflowOutcome.FATAL_FAILURE,
                message=message,
                started_at=existing.started_at or started_at,
                finished_at=utc_now_iso(),
                job_run_id=existing.id,
                result={"recovery_failure": {"message": message}},
            )
            self.job_runs.save(
                JobRun(
                    id=existing.id,
                    job_id=existing.job_id,
                    run_key=existing.run_key,
                    status="failed",
                    attempt=existing.attempt,
                    started_at=existing.started_at,
                    finished_at=result.finished_at,
                    result_json=json.dumps(self._workflow_result_payload(result), sort_keys=True),
                    error_message=result.message,
                    screenshot_path=existing.screenshot_path,
                )
            )
            return result
        return int(existing.id or 0)

    def _terminal_workflow_result(
        self,
        workflow: WorkflowDefinitionSpec,
        existing: JobRun,
        parsed: Any,
    ) -> WorkflowExecutionResult:
        if not parsed.ok:
            return WorkflowExecutionResult(
                workflow_key=workflow.workflow_key,
                schema_version=workflow.schema_version,
                outcome=WorkflowOutcome.FATAL_FAILURE,
                message=parsed.message,
                started_at=existing.started_at,
                finished_at=existing.finished_at or utc_now_iso(),
                job_run_id=existing.id,
                result={
                    "recovery_failure": {
                        "terminal_run": True,
                        "message": parsed.message,
                    }
                },
            )
        outcome = outcome_from_payload(parsed.value, fallback_status=existing.status)
        steps = self._stored_step_results(existing.id or 0)
        message = str(parsed.value.get("message") or existing.error_message or "")
        return WorkflowExecutionResult(
            workflow_key=str(parsed.value.get("workflow_key") or workflow.workflow_key),
            schema_version=workflow.schema_version,
            outcome=outcome,
            message=message,
            steps=steps,
            started_at=existing.started_at,
            finished_at=existing.finished_at or utc_now_iso(),
            job_run_id=existing.id,
            result={
                "recovery": {
                    "terminal_run": True,
                    "resumed_from_job_run": existing.id,
                    "run_key": existing.run_key,
                }
            },
        )

    def _stored_step_results(self, job_run_id: int) -> list[WorkflowStepResult]:
        list_for_job_run = getattr(self.step_runs, "list_for_job_run", None)
        if list_for_job_run is None:
            return []
        results: list[WorkflowStepResult] = []
        for step_run in list_for_job_run(job_run_id):
            parsed = parse_persisted_payload(
                step_run.result_json,
                source=f"step_run[{step_run.id}].result_json",
            )
            if not parsed.ok:
                results.append(
                    WorkflowStepResult(
                        step_key=step_run.step_key,
                        action_type="",
                        outcome=WorkflowOutcome.FATAL_FAILURE,
                        message=parsed.message,
                        attempt=step_run.attempt,
                        started_at=step_run.started_at,
                        finished_at=step_run.finished_at or utc_now_iso(),
                    )
                )
                continue
            results.append(_step_result_from_payload(None, step_run, parsed.value))
        return results

    def _workflow_running_payload(
        self,
        workflow: WorkflowDefinitionSpec,
        phase: RecoveryPhase,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload = {
            "workflow_key": workflow.workflow_key,
            "schema_version": workflow.schema_version,
            "outcome": "RUNNING",
        }
        return payload_with_recovery(
            payload,
            phase,
            source="workflow_run.running",
            extra=extra,
        )

    def _workflow_result_payload(self, result: WorkflowExecutionResult) -> dict[str, object]:
        return payload_with_recovery(
            result.to_json_dict(),
            RecoveryPhase.COMPLETED,
            source="workflow_run.result",
            extra={"job_run_id": result.job_run_id},
        )

    def _step_running_payload(
        self,
        step: WorkflowStepSpec,
        attempt: int,
        phase: RecoveryPhase,
    ) -> dict[str, object]:
        payload = {
            "step_key": step.step_key,
            "action_type": step.action_type,
            "attempt": attempt,
            "outcome": "RUNNING",
        }
        return payload_with_recovery(payload, phase, source="step_run.running")

    def _step_result_payload(
        self,
        result: WorkflowStepResult,
        phase: RecoveryPhase,
    ) -> dict[str, object]:
        return payload_with_recovery(
            result.to_json_dict(),
            phase,
            source="step_run.result",
            extra={"step_run_completed": True},
        )

    def _update_job_running_phase(
        self,
        job_run_id: int | None,
        phase: RecoveryPhase,
        *,
        step: WorkflowStepSpec | None = None,
        attempt: int | None = None,
        result: WorkflowStepResult | None = None,
    ) -> None:
        if job_run_id is None:
            return
        existing = _require_existing_job_run(self.job_runs, job_run_id)
        if existing.status in TERMINAL_RUN_STATUSES:
            return
        parsed = parse_persisted_payload(
            existing.result_json,
            source=f"job_run[{job_run_id}].result_json",
        )
        payload = parsed.value if parsed.ok else {}
        payload["outcome"] = "RUNNING"
        extra: dict[str, object] = {}
        if step is not None:
            extra.update({"step_key": step.step_key, "attempt": attempt or 1})
        if result is not None:
            extra.update(
                {
                    "last_step_key": result.step_key,
                    "last_step_outcome": result.outcome.value,
                    "attempt": result.attempt,
                }
            )
        self.job_runs.save(
            JobRun(
                id=job_run_id,
                job_id=existing.job_id,
                run_key=existing.run_key,
                status="running",
                attempt=existing.attempt,
                started_at=existing.started_at,
                finished_at=None,
                result_json=json.dumps(
                    payload_with_recovery(
                        payload,
                        phase,
                        source=f"job_run[{job_run_id}].running",
                        extra=extra,
                    ),
                    sort_keys=True,
                ),
                error_message="",
                screenshot_path=existing.screenshot_path,
            )
        )

    def _transaction(self):
        transaction = getattr(self.db, "transaction", None)
        if transaction is None:
            return nullcontext()
        return transaction()

    def _after_workflow_start_saved(self, _job_run_id: int) -> None:
        return None

    def _after_workflow_finish_saved(self, _job_run_id: int) -> None:
        return None

    def _after_step_start_saved(self, _step_run_id: int) -> None:
        return None

    def _after_step_finish_saved(self, _step_run_id: int) -> None:
        return None


def _step_result_from_payload(
    step: WorkflowStepSpec | None,
    run: StepRun,
    payload: dict[str, object],
) -> WorkflowStepResult:
    outcome = outcome_from_payload(payload, fallback_status=run.status)
    data = payload.get("data")
    return WorkflowStepResult(
        step_key=step.step_key if step is not None else run.step_key,
        action_type=(
            step.action_type
            if step is not None
            else str(payload.get("action_type") or "")
        ),
        outcome=outcome,
        message=str(payload.get("message") or run.error_message or ""),
        data=dict(data) if isinstance(data, dict) else {},
        attempt=run.attempt,
        started_at=run.started_at,
        finished_at=run.finished_at or utc_now_iso(),
        screenshot_path=run.screenshot_path,
        workflow_step_id=step.workflow_step_id if step is not None else run.workflow_step_id,
    )


def _recovery_failure_step_result(
    step: WorkflowStepSpec,
    attempt: int,
    message: str,
) -> WorkflowStepResult:
    return WorkflowStepResult(
        step_key=step.step_key,
        action_type=step.action_type,
        outcome=WorkflowOutcome.FATAL_FAILURE,
        message=sanitize_diagnostic_message(message),
        data={"recovery_failure": {"message": sanitize_diagnostic_message(message)}},
        attempt=attempt,
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
        workflow_step_id=step.workflow_step_id,
    )


def _status_for_outcome(outcome: WorkflowOutcome) -> str:
    if outcome in {WorkflowOutcome.SUCCESS, WorkflowOutcome.SKIPPED}:
        return "completed"
    if outcome == WorkflowOutcome.CANCELLED:
        return "cancelled"
    return "failed"


def _first_screenshot_path(steps: list[WorkflowStepResult]) -> str:
    for step in steps:
        if step.screenshot_path:
            return step.screenshot_path
    return ""


def _require_existing_job_run(job_runs: object, job_run_id: int) -> JobRun:
    method = getattr(job_runs, "get", None)
    if method is not None:
        run = method(job_run_id)
        if run is not None:
            return run
    raise ValueError(f"Job run not found: {job_run_id}")


def _require_existing_step_run(step_runs: object, step_run_id: int) -> StepRun:
    method = getattr(step_runs, "get", None)
    if method is not None:
        run = method(step_run_id)
        if run is not None:
            return run
    raise ValueError(f"Step run not found: {step_run_id}")
