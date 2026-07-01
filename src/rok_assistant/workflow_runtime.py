from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from rok_assistant.db.models import utc_now_iso
from rok_assistant.workflow_actions import (
    default_action_registry,
    default_condition_registry,
    default_normalizer_registry,
)
from rok_assistant.workflow_context import WorkflowExecutionContext
from rok_assistant.workflow_invocation import (
    invoke_condition_handler,
    invoke_step_handler,
)
from rok_assistant.workflow_recovery import RecoveryPhase, StepRecoveryDecision
from rok_assistant.workflow_registry import (
    ActionRegistry,
    ConditionRegistry,
    NormalizerRegistry,
)
from rok_assistant.workflow_serialization import (
    safe_serialize_metadata,
    sanitize_diagnostic_message,
)
from rok_assistant.workflow_types import (
    ConditionEvaluation,
    MAX_CALCULATED_BACKOFF_SECONDS,
    MAX_RETRY_BACKOFF_MULTIPLIER,
    MAX_RETRY_DELAY_SECONDS,
    MAX_RETRY_LIMIT,
    WorkflowCancelledError,
    WorkflowDefinitionSpec,
    WorkflowExecutionResult,
    WorkflowOutcome,
    WorkflowStepResult,
    WorkflowStepSpec,
    WorkflowValidationError,
    _int_value,
)
from rok_assistant.workflow_validation import WorkflowValidationLimits, WorkflowValidator


class WorkflowEngine:
    CONTROL_ACTIONS = {"sequence", "bounded_repeat", "if_else", "sub_workflow"}
    SUCCESS_OUTCOMES = {WorkflowOutcome.SUCCESS, WorkflowOutcome.SKIPPED}

    def __init__(
        self,
        *,
        action_registry: ActionRegistry | None = None,
        condition_registry: ConditionRegistry | None = None,
        normalizer_registry: NormalizerRegistry | None = None,
        validation_limits: WorkflowValidationLimits | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.action_registry = action_registry or default_action_registry()
        self.condition_registry = condition_registry or default_condition_registry()
        self.normalizer_registry = normalizer_registry or default_normalizer_registry()
        self.validation_limits = validation_limits or WorkflowValidationLimits()
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def validate(
        self,
        workflow: WorkflowDefinitionSpec,
        context: WorkflowExecutionContext | None = None,
    ) -> None:
        errors = self.validation_errors(workflow, context)
        if errors:
            raise WorkflowValidationError(errors=errors)

    def validation_errors(
        self,
        workflow: WorkflowDefinitionSpec,
        context: WorkflowExecutionContext | None = None,
    ) -> list[str]:
        validator = WorkflowValidator(
            action_registry=self.action_registry,
            condition_registry=self.condition_registry,
            normalizer_registry=self.normalizer_registry,
            limits=self.validation_limits,
        )
        return validator.validation_errors(
            workflow,
            workflow_resolver=context.workflow_resolver if context is not None else None,
        )

    def execute(
        self,
        workflow: WorkflowDefinitionSpec,
        context: WorkflowExecutionContext | None = None,
    ) -> WorkflowExecutionResult:
        base_context = context or WorkflowExecutionContext()
        context = base_context.with_runtime_metadata(dict(base_context.metadata))
        if context.normalizer_registry is None:
            context.normalizer_registry = self.normalizer_registry
        started_at = utc_now_iso()
        try:
            self.validate(workflow, context)
        except WorkflowValidationError as exc:
            return WorkflowExecutionResult(
                workflow_key=workflow.workflow_key,
                schema_version=workflow.schema_version,
                outcome=WorkflowOutcome.FATAL_FAILURE,
                message=str(exc),
                started_at=started_at,
                finished_at=utc_now_iso(),
            )
        if not workflow.enabled:
            return WorkflowExecutionResult(
                workflow_key=workflow.workflow_key,
                schema_version=workflow.schema_version,
                outcome=WorkflowOutcome.SKIPPED,
                message="Workflow is disabled.",
                started_at=started_at,
                finished_at=utc_now_iso(),
            )

        job_run_id = None
        outcome = context.check_cancelled_or_expired()
        if outcome == WorkflowOutcome.CANCELLED:
            return WorkflowExecutionResult(
                workflow_key=workflow.workflow_key,
                schema_version=workflow.schema_version,
                outcome=WorkflowOutcome.CANCELLED,
                message=context.cancellation_token.reason or "Workflow cancelled.",
                started_at=started_at,
                finished_at=utc_now_iso(),
            )
        if outcome == WorkflowOutcome.TIMEOUT:
            return WorkflowExecutionResult(
                workflow_key=workflow.workflow_key,
                schema_version=workflow.schema_version,
                outcome=WorkflowOutcome.TIMEOUT,
                message="Workflow deadline exceeded.",
                started_at=started_at,
                finished_at=utc_now_iso(),
            )
        if context.persistence is not None:
            try:
                start_result = context.persistence.start_workflow_run(
                    workflow,
                    context,
                    started_at,
                )
            except Exception as exc:
                return WorkflowExecutionResult(
                    workflow_key=workflow.workflow_key,
                    schema_version=workflow.schema_version,
                    outcome=WorkflowOutcome.FATAL_FAILURE,
                    message="Workflow persistence failed before execution.",
                    started_at=started_at,
                    finished_at=utc_now_iso(),
                    result={
                        "persistence_failure": _persistence_failure_metadata(
                            exc,
                            "start_workflow_run",
                        )
                    },
                )
            if isinstance(start_result, WorkflowExecutionResult):
                return start_result
            job_run_id = start_result
        runtime_metadata = dict(context.metadata)
        runtime_metadata["_job_run_id"] = job_run_id
        context = context.with_runtime_metadata(runtime_metadata)
        steps: list[WorkflowStepResult] = []
        outcome = WorkflowOutcome.SUCCESS
        message = ""
        try:
            outcome, message = self._execute_sequence(
                workflow.steps,
                context,
                job_run_id=job_run_id,
                results=steps,
                depth=0,
            )
        except WorkflowCancelledError as exc:
            outcome = WorkflowOutcome.CANCELLED
            message = str(exc) or "Workflow cancelled."
        except TimeoutError as exc:
            outcome = WorkflowOutcome.TIMEOUT
            message = str(exc) or "Workflow deadline exceeded."
        terminal = context.check_cancelled_or_expired()
        if terminal == WorkflowOutcome.CANCELLED and outcome in self.SUCCESS_OUTCOMES:
            outcome = WorkflowOutcome.CANCELLED
            message = context.cancellation_token.reason or "Workflow cancelled."
        elif terminal == WorkflowOutcome.TIMEOUT and outcome in self.SUCCESS_OUTCOMES:
            outcome = WorkflowOutcome.TIMEOUT
            message = "Workflow deadline exceeded."
        finished_at = utc_now_iso()
        result = WorkflowExecutionResult(
            workflow_key=workflow.workflow_key,
            schema_version=workflow.schema_version,
            outcome=outcome,
            message=message,
            steps=steps,
            started_at=started_at,
            finished_at=finished_at,
            job_run_id=job_run_id,
        )
        if context.persistence is not None:
            terminal = context.check_cancelled_or_expired()
            if terminal == WorkflowOutcome.CANCELLED:
                result.outcome = WorkflowOutcome.CANCELLED
                result.message = context.cancellation_token.reason or "Workflow cancelled."
            elif terminal == WorkflowOutcome.TIMEOUT:
                result.outcome = WorkflowOutcome.TIMEOUT
                result.message = "Workflow deadline exceeded."
            try:
                context.persistence.finish_workflow_run(job_run_id, result)
            except Exception as exc:
                previous_outcome = result.outcome
                result.outcome = WorkflowOutcome.FATAL_FAILURE
                result.message = "Workflow persistence failed while finishing workflow run."
                result.result = {
                    **dict(result.result),
                    "previous_outcome": previous_outcome.value,
                    "persistence_failure": _persistence_failure_metadata(
                        exc,
                        "finish_workflow_run",
                    ),
                }
        return result

    def _execute_sequence(
        self,
        steps: Sequence[WorkflowStepSpec],
        context: WorkflowExecutionContext,
        *,
        job_run_id: int | None,
        results: list[WorkflowStepResult],
        depth: int,
    ) -> tuple[WorkflowOutcome, str]:
        if depth > context.budget.max_depth:
            return WorkflowOutcome.BLOCKED, "Workflow nesting depth exceeded."
        sequence_outcome = WorkflowOutcome.SKIPPED
        for step in steps:
            outcome = context.check_cancelled_or_expired()
            if outcome == WorkflowOutcome.CANCELLED:
                return outcome, context.cancellation_token.reason or "Workflow cancelled."
            if outcome == WorkflowOutcome.TIMEOUT:
                return outcome, "Workflow deadline exceeded."
            result = self._execute_step_with_retries(
                step,
                context,
                job_run_id=job_run_id,
                results=results,
                depth=depth,
            )
            if result.outcome not in self.SUCCESS_OUTCOMES:
                return result.outcome, result.message
            if result.outcome == WorkflowOutcome.SUCCESS:
                sequence_outcome = WorkflowOutcome.SUCCESS
        return sequence_outcome, ""

    def _execute_step_with_retries(
        self,
        step: WorkflowStepSpec,
        context: WorkflowExecutionContext,
        *,
        job_run_id: int | None,
        results: list[WorkflowStepResult],
        depth: int,
    ) -> WorkflowStepResult:
        if not step.enabled:
            result = self._step_result(
                step,
                WorkflowOutcome.SKIPPED,
                "Step is disabled.",
                attempt=1,
            )
            results.append(result)
            return result

        attempts = min(max(0, int(step.retry_limit)), MAX_RETRY_LIMIT) + 1
        last_result: WorkflowStepResult | None = None
        for attempt in range(1, attempts + 1):
            terminal = self._terminal_step_result(step, context, attempt=attempt)
            if terminal is not None:
                results.append(terminal)
                return terminal
            if context.resume_completed_steps and context.persistence is not None:
                recovered = self._recover_step_from_persistence(
                    step,
                    context,
                    job_run_id=job_run_id,
                    attempt=attempt,
                )
                if recovered is not None:
                    results.append(recovered)
                    return recovered

            started_at = utc_now_iso()
            step_run_id = None
            terminal = self._terminal_step_result(step, context, attempt=attempt, started_at=started_at)
            if terminal is not None:
                results.append(terminal)
                return terminal
            if context.persistence is not None:
                try:
                    step_run_id = context.persistence.start_step_run(
                        job_run_id,
                        step,
                        attempt,
                        started_at,
                    )
                    mark_step_phase = getattr(context.persistence, "mark_step_phase", None)
                    if mark_step_phase is not None:
                        mark_step_phase(
                            step_run_id,
                            step,
                            attempt,
                            RecoveryPhase.SIDE_EFFECT_STARTED,
                        )
                except Exception as exc:
                    result = self._step_result(
                        step,
                        WorkflowOutcome.FATAL_FAILURE,
                        "Workflow persistence failed before step execution.",
                        attempt=attempt,
                        started_at=started_at,
                        data={
                            "persistence_failure": _persistence_failure_metadata(
                                exc,
                                "start_step_run",
                            )
                        },
                    )
                    results.append(result)
                    return result
            result = self._execute_step_once(
                step,
                context,
                attempt=attempt,
                started_at=started_at,
                depth=depth,
            )
            last_result = result
            if result.outcome == WorkflowOutcome.RETRYABLE_FAILURE and attempt < attempts:
                requested_delay = _requested_retry_delay_seconds(step, attempt)
                retry_delay = _retry_delay_seconds(step, attempt)
                deadline_remaining = context.deadline.remaining(context.clock)
                applied_delay = retry_delay
                if deadline_remaining is not None:
                    applied_delay = min(applied_delay, deadline_remaining)
                retry_metadata = {
                    "attempt": attempt,
                    "next_attempt": attempt + 1,
                    "reason": result.message or "Retryable failure.",
                    "requested_delay_seconds": requested_delay,
                    "applied_delay_seconds": max(0.0, applied_delay),
                    "max_attempts": attempts,
                }
                result.data = {
                    **dict(result.data),
                    "retry": retry_metadata,
                }
                try:
                    if applied_delay > 0:
                        context.sleep(applied_delay)
                    terminal = self._terminal_step_result(
                        step,
                        context,
                        attempt=attempt,
                        started_at=started_at,
                        data={
                            **dict(result.data),
                            "retry": {
                                **retry_metadata,
                                "interrupted": True,
                            },
                            "previous_outcome": result.outcome.value,
                        },
                    )
                    if terminal is not None:
                        result = terminal
                except WorkflowCancelledError as exc:
                    result = self._step_result(
                        step,
                        WorkflowOutcome.CANCELLED,
                        str(exc) or "Workflow cancelled.",
                        attempt=attempt,
                        started_at=started_at,
                        data={
                            **dict(result.data),
                            "retry": {
                                **retry_metadata,
                                "interrupted": True,
                            },
                            "previous_outcome": WorkflowOutcome.RETRYABLE_FAILURE.value,
                        },
                    )
                except TimeoutError as exc:
                    result = self._step_result(
                        step,
                        WorkflowOutcome.TIMEOUT,
                        str(exc) or "Workflow deadline exceeded.",
                        attempt=attempt,
                        started_at=started_at,
                        data={
                            **dict(result.data),
                            "retry": {
                                **retry_metadata,
                                "interrupted": True,
                            },
                            "previous_outcome": WorkflowOutcome.RETRYABLE_FAILURE.value,
                        },
                    )
            elif result.outcome == WorkflowOutcome.RETRYABLE_FAILURE:
                result.data = {
                    **dict(result.data),
                    "retry_exhausted": {
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "reason": result.message or "Retryable failure.",
                    },
                }
            terminal = self._terminal_step_result(
                step,
                context,
                attempt=attempt,
                started_at=started_at,
                data={"previous_result": result.to_json_dict()},
            )
            if terminal is not None and result.outcome not in {
                WorkflowOutcome.CANCELLED,
                WorkflowOutcome.TIMEOUT,
            }:
                result = terminal
            if context.persistence is not None:
                try:
                    context.persistence.finish_step_run(step_run_id, result)
                except Exception as exc:
                    result = self._step_result(
                        step,
                        WorkflowOutcome.FATAL_FAILURE,
                        "Workflow persistence failed after step execution.",
                        attempt=attempt,
                        started_at=started_at,
                        data={
                            "previous_result": result.to_json_dict(),
                            "side_effect_state": "uncertain",
                            "persistence_failure": _persistence_failure_metadata(
                                exc,
                                "finish_step_run",
                            ),
                        },
                        screenshot_path=result.screenshot_path,
                    )
            results.append(result)
            if result.outcome != WorkflowOutcome.RETRYABLE_FAILURE:
                return result
        if last_result is None:
            return self._step_result(
                step,
                WorkflowOutcome.FATAL_FAILURE,
                "Step was not executed.",
                attempt=1,
            )
        return last_result

    def _recover_step_from_persistence(
        self,
        step: WorkflowStepSpec,
        context: WorkflowExecutionContext,
        *,
        job_run_id: int | None,
        attempt: int,
    ) -> WorkflowStepResult | None:
        if context.persistence is None:
            return None
        get_recovery = getattr(context.persistence, "get_step_recovery", None)
        if get_recovery is None:
            completed = context.persistence.get_completed_step(job_run_id, step, attempt)
            return completed
        try:
            decision = get_recovery(job_run_id, step, attempt)
        except Exception as exc:
            return self._step_result(
                step,
                WorkflowOutcome.FATAL_FAILURE,
                "Workflow recovery failed while reading persisted step state.",
                attempt=attempt,
                data={
                    "persistence_failure": _persistence_failure_metadata(
                        exc,
                        "get_step_recovery",
                    )
                },
            )
        if decision is None:
            return None
        if not isinstance(decision, StepRecoveryDecision):
            return self._step_result(
                step,
                WorkflowOutcome.FATAL_FAILURE,
                "Workflow recovery returned an invalid step decision.",
                attempt=attempt,
                data={"invalid_recovery_decision": {"type": type(decision).__name__}},
            )
        if decision.result is not None:
            if decision.phase == RecoveryPhase.COMPLETED:
                return decision.result
            return self._finish_recovered_step(step, context, decision, decision.result)
        if decision.can_resume:
            return None
        if decision.requires_postcondition:
            result = self._recover_uncertain_step_with_postcondition(
                step,
                context,
                decision,
                attempt,
            )
            return self._finish_recovered_step(step, context, decision, result)
        result = self._step_result(
            step,
            WorkflowOutcome.BLOCKED,
            "Step side effect is uncertain; manual recovery is required.",
            attempt=attempt,
            data={
                "recovery": {
                    "phase": decision.phase.value,
                    "resumed_from_step_run": decision.step_run_id,
                    "action_reexecuted": False,
                }
            },
        )
        return self._finish_recovered_step(step, context, decision, result)

    def _recover_uncertain_step_with_postcondition(
        self,
        step: WorkflowStepSpec,
        context: WorkflowExecutionContext,
        decision: StepRecoveryDecision,
        attempt: int,
    ) -> WorkflowStepResult:
        if step.postcondition is None:
            return self._step_result(
                step,
                WorkflowOutcome.BLOCKED,
                "Step side effect is uncertain; manual recovery is required.",
                attempt=attempt,
                data={
                    "recovery": {
                        "phase": decision.phase.value,
                        "resumed_from_step_run": decision.step_run_id,
                        "action_reexecuted": False,
                    }
                },
            )
        evaluation = self._evaluate_condition(
            step.postcondition,
            context,
            step,
            handler_kind="postcondition",
        )
        data = {
            "recovery": {
                "phase": decision.phase.value,
                "resumed_from_step_run": decision.step_run_id,
                "action_reexecuted": False,
            },
            "postcondition": evaluation.data,
        }
        if evaluation.outcome not in self.SUCCESS_OUTCOMES:
            return self._step_result(
                step,
                evaluation.outcome,
                evaluation.message,
                attempt=attempt,
                data=data,
                screenshot_path=evaluation.screenshot_path,
            )
        if not evaluation.matched:
            return self._step_result(
                step,
                WorkflowOutcome.BLOCKED,
                evaluation.message or "Step side effect remains uncertain after recovery postcondition.",
                attempt=attempt,
                data=data,
                screenshot_path=evaluation.screenshot_path,
            )
        mark_step_phase = getattr(context.persistence, "mark_step_phase", None)
        if mark_step_phase is not None:
            try:
                mark_step_phase(
                    decision.step_run_id,
                    step,
                    attempt,
                    RecoveryPhase.POSTCONDITION_VERIFIED,
                )
            except Exception as exc:
                return self._step_result(
                    step,
                    WorkflowOutcome.FATAL_FAILURE,
                    "Workflow persistence failed while recording recovery postcondition.",
                    attempt=attempt,
                    data={
                        **data,
                        "persistence_failure": _persistence_failure_metadata(
                            exc,
                            "mark_step_phase",
                        ),
                    },
                    screenshot_path=evaluation.screenshot_path,
                )
        return self._step_result(
            step,
            WorkflowOutcome.SUCCESS,
            "Step recovered from verified postcondition.",
            attempt=attempt,
            data={
                **data,
                "recovery": {
                    **dict(data["recovery"]),
                    "phase": RecoveryPhase.POSTCONDITION_VERIFIED.value,
                },
            },
            screenshot_path=evaluation.screenshot_path,
        )

    def _finish_recovered_step(
        self,
        step: WorkflowStepSpec,
        context: WorkflowExecutionContext,
        decision: StepRecoveryDecision,
        result: WorkflowStepResult,
    ) -> WorkflowStepResult:
        if context.persistence is None:
            return result
        try:
            context.persistence.finish_step_run(decision.step_run_id, result)
        except Exception as exc:
            return self._step_result(
                step,
                WorkflowOutcome.FATAL_FAILURE,
                "Workflow persistence failed while recording recovery decision.",
                attempt=result.attempt,
                started_at=result.started_at,
                data={
                    "previous_result": result.to_json_dict(),
                    "persistence_failure": _persistence_failure_metadata(
                        exc,
                        "finish_step_run",
                    ),
                },
                screenshot_path=result.screenshot_path,
            )
        return result

    def _execute_step_once(
        self,
        step: WorkflowStepSpec,
        context: WorkflowExecutionContext,
        *,
        attempt: int,
        started_at: str,
        depth: int,
    ) -> WorkflowStepResult:
        if not context.budget.consume_step():
            return self._step_result(
                step,
                WorkflowOutcome.BLOCKED,
                "Step budget exhausted.",
                attempt=attempt,
                started_at=started_at,
            )
        try:
            context.cancellation_token.throw_if_cancelled()
            step_deadline = context.deadline.child(step.timeout_seconds, context.clock)
            step_context = context.for_step(step_deadline)
            if step.action_type == "sequence":
                result = self._execute_sequence_step(step, step_context, attempt, depth)
            elif step.action_type == "bounded_repeat":
                result = self._execute_repeat_step(step, step_context, attempt, depth)
            elif step.action_type == "if_else":
                result = self._execute_if_else_step(step, step_context, attempt, depth)
            elif step.action_type == "sub_workflow":
                result = self._execute_sub_workflow_step(step, step_context, attempt, depth)
            else:
                result = self._execute_registered_action(step, step_context, attempt)
            result.attempt = attempt
            result.workflow_step_id = step.workflow_step_id
            terminal = self._terminal_step_result(
                step,
                step_context,
                attempt=attempt,
                started_at=started_at,
                data={
                    "side_effect_state": "uncertain",
                    "handler_result": result.to_json_dict(),
                },
            )
            if terminal is not None and result.outcome == WorkflowOutcome.SUCCESS:
                return terminal
            if result.outcome == WorkflowOutcome.SUCCESS:
                result = self._verify_postcondition(step, step_context, result)
            if step_deadline.is_expired(context.clock) and result.outcome == WorkflowOutcome.SUCCESS:
                return self._step_result(
                    step,
                    WorkflowOutcome.TIMEOUT,
                    "Step deadline exceeded.",
                    attempt=attempt,
                    started_at=started_at,
                    data=result.data,
                    screenshot_path=result.screenshot_path,
                )
            result.started_at = started_at
            result.finished_at = result.finished_at or utc_now_iso()
            return result
        except WorkflowCancelledError as exc:
            return self._step_result(
                step,
                WorkflowOutcome.CANCELLED,
                str(exc) or "Workflow cancelled.",
                attempt=attempt,
                started_at=started_at,
            )
        except TimeoutError as exc:
            return self._step_result(
                step,
                WorkflowOutcome.TIMEOUT,
                str(exc) or "Step deadline exceeded.",
                attempt=attempt,
                started_at=started_at,
            )

    def _terminal_step_result(
        self,
        step: WorkflowStepSpec,
        context: WorkflowExecutionContext,
        *,
        attempt: int,
        started_at: str = "",
        data: dict[str, object] | None = None,
    ) -> WorkflowStepResult | None:
        outcome = context.check_cancelled_or_expired()
        if outcome == WorkflowOutcome.CANCELLED:
            return self._step_result(
                step,
                WorkflowOutcome.CANCELLED,
                context.cancellation_token.reason or "Workflow cancelled.",
                attempt=attempt,
                started_at=started_at,
                data=data,
            )
        if outcome == WorkflowOutcome.TIMEOUT:
            return self._step_result(
                step,
                WorkflowOutcome.TIMEOUT,
                "Workflow deadline exceeded.",
                attempt=attempt,
                started_at=started_at,
                data=data,
            )
        return None

    def _execute_sequence_step(
        self,
        step: WorkflowStepSpec,
        context: WorkflowExecutionContext,
        attempt: int,
        depth: int,
    ) -> WorkflowStepResult:
        child_results: list[WorkflowStepResult] = []
        outcome, message = self._execute_sequence(
            step.steps,
            context,
            job_run_id=_job_run_id_from_context(context),
            results=child_results,
            depth=depth + 1,
        )
        return self._step_result(
            step,
            outcome,
            message,
            attempt=attempt,
            data={"child_results": [item.to_json_dict() for item in child_results]},
        )

    def _execute_repeat_step(
        self,
        step: WorkflowStepSpec,
        context: WorkflowExecutionContext,
        attempt: int,
        depth: int,
    ) -> WorkflowStepResult:
        count = _int_value(step.parameters.get("count"), 0)
        max_count = _int_value(step.parameters.get("max_count"), count)
        if count > context.budget.max_repeat_iterations or max_count > context.budget.max_repeat_iterations:
            return self._step_result(
                step,
                WorkflowOutcome.BLOCKED,
                "Repeat count exceeds step budget.",
                attempt=attempt,
            )
        iterations: list[dict[str, object]] = []
        outcome = WorkflowOutcome.SKIPPED
        message = ""
        for iteration in range(1, count + 1):
            child_results: list[WorkflowStepResult] = []
            outcome, message = self._execute_sequence(
                step.steps,
                context,
                job_run_id=_job_run_id_from_context(context),
                results=child_results,
                depth=depth + 1,
            )
            iterations.append(
                {
                    "iteration": iteration,
                    "outcome": outcome.value,
                    "steps": [item.to_json_dict() for item in child_results],
                }
            )
            if outcome not in self.SUCCESS_OUTCOMES:
                return self._step_result(
                    step,
                    outcome,
                    message,
                    attempt=attempt,
                    data={"iterations": iterations, "count": count},
                )
        return self._step_result(
            step,
            WorkflowOutcome.SUCCESS if count > 0 else WorkflowOutcome.SKIPPED,
            "",
            attempt=attempt,
            data={"iterations": iterations, "count": count},
        )

    def _execute_if_else_step(
        self,
        step: WorkflowStepSpec,
        context: WorkflowExecutionContext,
        attempt: int,
        depth: int,
    ) -> WorkflowStepResult:
        condition = self._evaluate_condition(step.parameters, context, step)
        if condition.outcome not in self.SUCCESS_OUTCOMES:
            return self._step_result(
                step,
                condition.outcome,
                condition.message,
                attempt=attempt,
                data=condition.data,
                screenshot_path=condition.screenshot_path,
            )
        branch_steps = step.then_steps if condition.matched else step.else_steps
        branch_name = "then" if condition.matched else "else"
        child_results: list[WorkflowStepResult] = []
        outcome, message = self._execute_sequence(
            branch_steps,
            context,
            job_run_id=_job_run_id_from_context(context),
            results=child_results,
            depth=depth + 1,
        )
        return self._step_result(
            step,
            outcome,
            message,
            attempt=attempt,
            data={
                "condition_result": condition.matched,
                "condition": condition.data,
                "branch": branch_name,
                "child_results": [item.to_json_dict() for item in child_results],
            },
            screenshot_path=condition.screenshot_path,
        )

    def _execute_sub_workflow_step(
        self,
        step: WorkflowStepSpec,
        context: WorkflowExecutionContext,
        attempt: int,
        depth: int,
    ) -> WorkflowStepResult:
        terminal = self._terminal_step_result(step, context, attempt=attempt)
        if terminal is not None:
            return terminal
        if context.workflow_resolver is None:
            return self._step_result(
                step,
                WorkflowOutcome.BLOCKED,
                "No workflow resolver configured.",
                attempt=attempt,
            )
        workflow_key = str(step.parameters.get("workflow_key", "")).strip()
        workflow = context.workflow_resolver(workflow_key)
        if workflow is None:
            return self._step_result(
                step,
                WorkflowOutcome.BLOCKED,
                f"Sub-workflow not found: {workflow_key}.",
                attempt=attempt,
            )
        child_results: list[WorkflowStepResult] = []
        terminal = self._terminal_step_result(step, context, attempt=attempt)
        if terminal is not None:
            return terminal
        outcome, message = self._execute_sequence(
            workflow.steps,
            context,
            job_run_id=_job_run_id_from_context(context),
            results=child_results,
            depth=depth + 1,
        )
        terminal = self._terminal_step_result(
            step,
            context,
            attempt=attempt,
            data={
                "sub_workflow_key": workflow_key,
                "child_results": [item.to_json_dict() for item in child_results],
            },
        )
        if terminal is not None:
            return terminal
        return self._step_result(
            step,
            outcome,
            message,
            attempt=attempt,
            data={
                "sub_workflow_key": workflow_key,
                "child_results": [item.to_json_dict() for item in child_results],
            },
        )

    def _execute_registered_action(
        self,
        step: WorkflowStepSpec,
        context: WorkflowExecutionContext,
        attempt: int,
    ) -> WorkflowStepResult:
        registration = self.action_registry.get(step.action_type)
        if registration is None:
            return self._step_result(
                step,
                WorkflowOutcome.FATAL_FAILURE,
                f"Unsupported action type: {step.action_type}",
                attempt=attempt,
            )
        return invoke_step_handler(
            registration.handler,
            context,
            step,
            handler_kind="action",
        )

    def _verify_postcondition(
        self,
        step: WorkflowStepSpec,
        context: WorkflowExecutionContext,
        result: WorkflowStepResult,
    ) -> WorkflowStepResult:
        if step.postcondition is None:
            return result
        terminal = self._terminal_step_result(
            step,
            context,
            attempt=result.attempt,
            data={
                **dict(result.data),
                "side_effect_state": "uncertain",
                "postcondition": {"status": "not_evaluated"},
            },
        )
        if terminal is not None:
            terminal.screenshot_path = result.screenshot_path
            return terminal
        evaluation = self._evaluate_condition(
            step.postcondition,
            context,
            step,
            handler_kind="postcondition",
        )
        data = dict(result.data)
        data["postcondition"] = evaluation.data
        if evaluation.outcome not in self.SUCCESS_OUTCOMES:
            return self._step_result(
                step,
                evaluation.outcome,
                evaluation.message,
                attempt=result.attempt,
                data=data,
                screenshot_path=evaluation.screenshot_path or result.screenshot_path,
            )
        if not evaluation.matched:
            return self._step_result(
                step,
                WorkflowOutcome.FATAL_FAILURE,
                evaluation.message or "Postcondition failed.",
                attempt=result.attempt,
                data=data,
                screenshot_path=evaluation.screenshot_path or result.screenshot_path,
            )
        result.data = data
        terminal = self._terminal_step_result(
            step,
            context,
            attempt=result.attempt,
            data={
                **dict(result.data),
                "side_effect_state": "uncertain",
            },
        )
        if terminal is not None:
            terminal.screenshot_path = result.screenshot_path
            return terminal
        return result

    def _evaluate_condition(
        self,
        parameters: Mapping[str, object],
        context: WorkflowExecutionContext,
        step: WorkflowStepSpec,
        *,
        handler_kind: str = "condition",
    ) -> ConditionEvaluation:
        terminal = context.check_cancelled_or_expired()
        if terminal == WorkflowOutcome.CANCELLED:
            return ConditionEvaluation(
                False,
                outcome=WorkflowOutcome.CANCELLED,
                message=context.cancellation_token.reason or "Workflow cancelled.",
                data={"handler_kind": handler_kind},
            )
        if terminal == WorkflowOutcome.TIMEOUT:
            return ConditionEvaluation(
                False,
                outcome=WorkflowOutcome.TIMEOUT,
                message="Workflow deadline exceeded.",
                data={"handler_kind": handler_kind},
            )
        condition_type = str(parameters.get("condition_type", "")).strip()
        registration = self.condition_registry.get(condition_type)
        if registration is None:
            return ConditionEvaluation(
                matched=False,
                outcome=WorkflowOutcome.FATAL_FAILURE,
                message=f"Unsupported condition type: {condition_type}",
            )
        condition_step = WorkflowStepSpec(
            step_key=step.step_key,
            action_type=condition_type,
            parameters=dict(parameters),
            workflow_step_id=step.workflow_step_id,
        )
        return invoke_condition_handler(
            registration.handler,
            context,
            condition_step,
            handler_kind=handler_kind,
        )

    @staticmethod
    def _step_result(
        step: WorkflowStepSpec,
        outcome: WorkflowOutcome,
        message: str = "",
        *,
        attempt: int,
        started_at: str = "",
        data: dict[str, object] | None = None,
        screenshot_path: str = "",
    ) -> WorkflowStepResult:
        safe_data = _safe_result_data(step, data or {}, source=f"step.{step.step_key}.data")
        if not safe_data.ok:
            outcome = WorkflowOutcome.VALIDATION_FAILURE
            message = "Step result metadata is not safely serializable."
            result_data = {
                "serialization_failure": safe_data.failure_metadata(source=f"step.{step.step_key}.data"),
                "sanitized_data": safe_data.value,
            }
        else:
            result_data = safe_data.value if isinstance(safe_data.value, dict) else {}
        return WorkflowStepResult(
            step_key=step.step_key,
            action_type=step.action_type,
            outcome=outcome,
            message=sanitize_diagnostic_message(message),
            data=result_data,
            attempt=attempt,
            started_at=started_at,
            finished_at=utc_now_iso(),
            screenshot_path=screenshot_path,
            workflow_step_id=step.workflow_step_id,
        )


def _retry_delay_seconds(step: WorkflowStepSpec, completed_attempt: int) -> float:
    if step.retry_delay_seconds <= 0:
        return 0.0
    base_delay = min(max(0.0, float(step.retry_delay_seconds)), MAX_RETRY_DELAY_SECONDS)
    multiplier = min(
        max(1.0, float(step.retry_backoff_multiplier)),
        MAX_RETRY_BACKOFF_MULTIPLIER,
    )
    exponent = max(0, completed_attempt - 1)
    delay = base_delay * (multiplier ** exponent)
    maximum = step.max_retry_delay_seconds
    if maximum is None:
        maximum = MAX_CALCULATED_BACKOFF_SECONDS
    maximum = min(max(0.0, float(maximum)), MAX_CALCULATED_BACKOFF_SECONDS)
    return min(delay, maximum, MAX_CALCULATED_BACKOFF_SECONDS)


def _requested_retry_delay_seconds(step: WorkflowStepSpec, completed_attempt: int) -> float:
    if step.retry_delay_seconds <= 0:
        return 0.0
    exponent = max(0, completed_attempt - 1)
    return float(step.retry_delay_seconds) * (float(step.retry_backoff_multiplier) ** exponent)


def _safe_result_data(
    step: WorkflowStepSpec,
    data: dict[str, object],
    *,
    source: str,
):
    return safe_serialize_metadata(_with_step_metadata(step, data), source=source)


def _with_step_metadata(
    step: WorkflowStepSpec,
    data: dict[str, object],
) -> dict[str, object]:
    output = dict(data)
    if step.legacy_order is not None:
        output.setdefault("legacy_order", step.legacy_order)
    if step.legacy_action_type is not None:
        output.setdefault("legacy_action_type", step.legacy_action_type)
    return output


def _job_run_id_from_context(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _persistence_failure_metadata(exc: Exception, operation: str) -> dict[str, object]:
    return {
        "operation": operation,
        "exception_class": type(exc).__name__,
        "message": sanitize_diagnostic_message(str(exc)),
    }
