"""Shared workflow handler invocation boundary.

Custom handlers are cooperative: they must observe the supplied
``WorkflowExecutionContext.cancellation_token`` and ``deadline``. The runtime
does not terminate arbitrary blocking Python code or third-party calls.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from rok_assistant.action_engine import DEFAULT_ABORT_REASON
from rok_assistant.workflow_context import WorkflowExecutionContext
from rok_assistant.workflow_serialization import (
    safe_exception_diagnostics,
    safe_json_payload,
    safe_serialize_metadata,
    sanitize_diagnostic_message,
)
from rok_assistant.workflow_types import (
    ConditionEvaluation,
    WorkflowCancelledError,
    WorkflowOutcome,
    WorkflowStepResult,
    WorkflowStepSpec,
)


StepHandler = Callable[[WorkflowExecutionContext, WorkflowStepSpec], object]
ConditionHandler = Callable[[WorkflowExecutionContext, WorkflowStepSpec], object]

_CONTROL_RESULT_KEYS = {
    "outcome",
    "message",
    "data",
    "screenshot_path",
    "success",
    "fatal",
    "aborted",
    "retryable",
}


def invoke_step_handler(
    handler: StepHandler,
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
    *,
    handler_kind: str,
) -> WorkflowStepResult:
    terminal = _terminal_step_result(context, step, handler_kind=handler_kind)
    if terminal is not None:
        return terminal
    try:
        raw_result = handler(context, step)
    except WorkflowCancelledError as exc:
        return _cancelled_step_result(context, step, str(exc), handler_kind=handler_kind)
    except TimeoutError as exc:
        return _timeout_step_result(step, str(exc), handler_kind=handler_kind)
    except Exception as exc:
        return _exception_step_result(step, exc, handler_kind=handler_kind)

    terminal = _terminal_step_result(
        context,
        step,
        handler_kind=handler_kind,
        after_handler=True,
        raw_result=raw_result,
    )
    if terminal is not None:
        return terminal

    return normalize_step_handler_result(
        raw_result,
        context,
        step,
        handler_kind=handler_kind,
    )


def invoke_condition_handler(
    handler: ConditionHandler,
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
    *,
    handler_kind: str,
) -> ConditionEvaluation:
    terminal = _terminal_condition_result(context, handler_kind=handler_kind)
    if terminal is not None:
        return terminal
    try:
        raw_result = handler(context, step)
    except WorkflowCancelledError as exc:
        return _cancelled_condition_result(context, str(exc), handler_kind=handler_kind)
    except TimeoutError as exc:
        return _timeout_condition_result(str(exc), handler_kind=handler_kind)
    except Exception as exc:
        return _exception_condition_result(exc, handler_kind=handler_kind)

    terminal = _terminal_condition_result(
        context,
        handler_kind=handler_kind,
        after_handler=True,
        raw_result=raw_result,
    )
    if terminal is not None:
        return terminal

    return normalize_condition_handler_result(
        raw_result,
        handler_kind=handler_kind,
    )


def normalize_step_handler_result(
    raw_result: object,
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
    *,
    handler_kind: str,
) -> WorkflowStepResult:
    if isinstance(raw_result, WorkflowStepResult):
        return _validated_step_result(raw_result, context, step, handler_kind=handler_kind)
    if isinstance(raw_result, Mapping):
        return _step_result_from_mapping(raw_result, context, step, handler_kind=handler_kind)
    return _invalid_step_result(
        step,
        handler_kind=handler_kind,
        message="Handler returned an unsupported action result shape.",
        raw_result=raw_result,
    )


def normalize_condition_handler_result(
    raw_result: object,
    *,
    handler_kind: str,
) -> ConditionEvaluation:
    if isinstance(raw_result, bool):
        return ConditionEvaluation(raw_result)
    if isinstance(raw_result, ConditionEvaluation):
        return _validated_condition_result(raw_result, handler_kind=handler_kind)
    if isinstance(raw_result, Mapping):
        return _condition_result_from_mapping(raw_result, handler_kind=handler_kind)
    return _invalid_condition_result(
        handler_kind=handler_kind,
        message="Handler returned an unsupported condition result shape.",
        raw_result=raw_result,
    )


def _validated_step_result(
    result: WorkflowStepResult,
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
    *,
    handler_kind: str,
) -> WorkflowStepResult:
    if result.step_key != step.step_key or result.action_type != step.action_type:
        return _invalid_step_result(
            step,
            handler_kind=handler_kind,
            message="Handler result step identity does not match the invoked step.",
            raw_result=result,
        )
    if not isinstance(result.outcome, WorkflowOutcome):
        return _invalid_step_result(
            step,
            handler_kind=handler_kind,
            message="Handler result outcome must be a WorkflowOutcome.",
            raw_result=result,
        )
    if not isinstance(result.message, str):
        return _invalid_step_result(
            step,
            handler_kind=handler_kind,
            message="Handler result message must be a string.",
            raw_result=result,
        )
    if not isinstance(result.data, Mapping):
        return _invalid_step_result(
            step,
            handler_kind=handler_kind,
            message="Handler result data must be a mapping.",
            raw_result=result,
        )
    safe_data = safe_serialize_metadata(dict(result.data), source=f"{handler_kind}.data")
    if not safe_data.ok:
        return _serialization_failure_step_result(step, handler_kind=handler_kind, safe_data=safe_data)
    result.data = _merge_result_metadata(safe_data.value, context)
    result.message = sanitize_diagnostic_message(result.message)
    result.screenshot_path = str(result.screenshot_path or "")
    return result


def _step_result_from_mapping(
    raw_result: Mapping[object, object],
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
    *,
    handler_kind: str,
) -> WorkflowStepResult:
    if "outcome" in raw_result:
        outcome = _parse_outcome(raw_result.get("outcome"))
        if outcome is None:
            return _invalid_step_result(
                step,
                handler_kind=handler_kind,
                message="Handler mapping outcome is not supported.",
                raw_result=raw_result,
            )
        data_value = raw_result.get("data", {})
        if not isinstance(data_value, Mapping):
            return _invalid_step_result(
                step,
                handler_kind=handler_kind,
                message="Handler mapping data must be a mapping.",
                raw_result=raw_result,
            )
        return _mapped_step_result(
            step,
            outcome,
            raw_result,
            data=dict(data_value),
            context=context,
            handler_kind=handler_kind,
        )

    if "success" not in raw_result or not isinstance(raw_result.get("success"), bool):
        return _invalid_step_result(
            step,
            handler_kind=handler_kind,
            message="Handler mapping must include an explicit boolean success field or outcome field.",
            raw_result=raw_result,
        )
    for key in ("fatal", "aborted", "retryable"):
        if key in raw_result and not isinstance(raw_result.get(key), bool):
            return _invalid_step_result(
                step,
                handler_kind=handler_kind,
                message=f"Handler mapping {key} field must be a boolean.",
                raw_result=raw_result,
            )

    if bool(raw_result.get("aborted", False)):
        outcome = WorkflowOutcome.CANCELLED
        message = str(raw_result.get("message") or DEFAULT_ABORT_REASON)
    elif bool(raw_result.get("fatal", False)):
        outcome = WorkflowOutcome.FATAL_FAILURE
        message = str(raw_result.get("message") or "Action failed.")
    elif bool(raw_result.get("success")):
        outcome = WorkflowOutcome.SUCCESS
        message = str(raw_result.get("message") or "")
    else:
        retryable = bool(raw_result.get("retryable", True))
        outcome = WorkflowOutcome.RETRYABLE_FAILURE if retryable else WorkflowOutcome.FATAL_FAILURE
        message = str(raw_result.get("message") or "Action failed.")

    return _mapped_step_result(
        step,
        outcome,
        raw_result,
        data=dict(raw_result),
        context=context,
        handler_kind=handler_kind,
        message=message,
    )


def _mapped_step_result(
    step: WorkflowStepSpec,
    outcome: WorkflowOutcome,
    raw_result: Mapping[object, object],
    *,
    data: Mapping[str, object],
    context: WorkflowExecutionContext,
    handler_kind: str,
    message: str | None = None,
) -> WorkflowStepResult:
    safe_data = safe_serialize_metadata(dict(data), source=f"{handler_kind}.data")
    if not safe_data.ok:
        return _serialization_failure_step_result(step, handler_kind=handler_kind, safe_data=safe_data)
    screenshot_path = str(raw_result.get("screenshot_path") or "")
    return WorkflowStepResult(
        step_key=step.step_key,
        action_type=step.action_type,
        outcome=outcome,
        message=sanitize_diagnostic_message(str(message if message is not None else raw_result.get("message") or "")),
        data=_merge_result_metadata(safe_data.value, context),
        screenshot_path=screenshot_path,
        workflow_step_id=step.workflow_step_id,
    )


def _validated_condition_result(
    result: ConditionEvaluation,
    *,
    handler_kind: str,
) -> ConditionEvaluation:
    if not isinstance(result.matched, bool):
        return _invalid_condition_result(
            handler_kind=handler_kind,
            message="Condition result matched field must be a boolean.",
            raw_result=result,
        )
    if not isinstance(result.outcome, WorkflowOutcome):
        return _invalid_condition_result(
            handler_kind=handler_kind,
            message="Condition result outcome must be a WorkflowOutcome.",
            raw_result=result,
        )
    if not isinstance(result.message, str):
        return _invalid_condition_result(
            handler_kind=handler_kind,
            message="Condition result message must be a string.",
            raw_result=result,
        )
    if not isinstance(result.data, Mapping):
        return _invalid_condition_result(
            handler_kind=handler_kind,
            message="Condition result data must be a mapping.",
            raw_result=result,
        )
    safe_data = safe_serialize_metadata(dict(result.data), source=f"{handler_kind}.data")
    if not safe_data.ok:
        return _serialization_failure_condition_result(handler_kind=handler_kind, safe_data=safe_data)
    result.data = safe_data.value if isinstance(safe_data.value, dict) else {}
    result.message = sanitize_diagnostic_message(result.message)
    result.screenshot_path = str(result.screenshot_path or "")
    return result


def _condition_result_from_mapping(
    raw_result: Mapping[object, object],
    *,
    handler_kind: str,
) -> ConditionEvaluation:
    if "matched" not in raw_result or not isinstance(raw_result.get("matched"), bool):
        return _invalid_condition_result(
            handler_kind=handler_kind,
            message="Condition mapping must include an explicit boolean matched field.",
            raw_result=raw_result,
        )
    outcome = WorkflowOutcome.SUCCESS
    if "outcome" in raw_result:
        parsed = _parse_outcome(raw_result.get("outcome"))
        if parsed is None:
            return _invalid_condition_result(
                handler_kind=handler_kind,
                message="Condition mapping outcome is not supported.",
                raw_result=raw_result,
            )
        outcome = parsed
    data_value = raw_result.get("data", {})
    if not isinstance(data_value, Mapping):
        return _invalid_condition_result(
            handler_kind=handler_kind,
            message="Condition mapping data must be a mapping.",
            raw_result=raw_result,
        )
    safe_data = safe_serialize_metadata(dict(data_value), source=f"{handler_kind}.data")
    if not safe_data.ok:
        return _serialization_failure_condition_result(handler_kind=handler_kind, safe_data=safe_data)
    return ConditionEvaluation(
        matched=bool(raw_result["matched"]),
        outcome=outcome,
        message=sanitize_diagnostic_message(str(raw_result.get("message") or "")),
        data=safe_data.value if isinstance(safe_data.value, dict) else {},
        screenshot_path=str(raw_result.get("screenshot_path") or ""),
    )


def _terminal_step_result(
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
    *,
    handler_kind: str,
    after_handler: bool = False,
    raw_result: object | None = None,
) -> WorkflowStepResult | None:
    outcome = context.check_cancelled_or_expired()
    if outcome == WorkflowOutcome.CANCELLED:
        return _cancelled_step_result(
            context,
            step,
            context.cancellation_token.reason or "Workflow cancelled.",
            handler_kind=handler_kind,
            side_effect_uncertain=after_handler,
            raw_result=raw_result,
        )
    if outcome == WorkflowOutcome.TIMEOUT:
        return _timeout_step_result(
            step,
            "Workflow deadline exceeded.",
            handler_kind=handler_kind,
            side_effect_uncertain=after_handler,
            raw_result=raw_result,
        )
    return None


def _terminal_condition_result(
    context: WorkflowExecutionContext,
    *,
    handler_kind: str,
    after_handler: bool = False,
    raw_result: object | None = None,
) -> ConditionEvaluation | None:
    outcome = context.check_cancelled_or_expired()
    if outcome == WorkflowOutcome.CANCELLED:
        data = {"handler_kind": handler_kind}
        if after_handler:
            data["side_effect_state"] = "uncertain"
            data["handler_return"] = safe_json_payload(raw_result, source=f"{handler_kind}.return")
        return ConditionEvaluation(
            False,
            outcome=WorkflowOutcome.CANCELLED,
            message=context.cancellation_token.reason or "Workflow cancelled.",
            data=data,
        )
    if outcome == WorkflowOutcome.TIMEOUT:
        data = {"handler_kind": handler_kind}
        if after_handler:
            data["side_effect_state"] = "uncertain"
            data["handler_return"] = safe_json_payload(raw_result, source=f"{handler_kind}.return")
        return ConditionEvaluation(
            False,
            outcome=WorkflowOutcome.TIMEOUT,
            message="Workflow deadline exceeded.",
            data=data,
        )
    return None


def _cancelled_step_result(
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
    message: str,
    *,
    handler_kind: str,
    side_effect_uncertain: bool = False,
    raw_result: object | None = None,
) -> WorkflowStepResult:
    data: dict[str, object] = {"handler_kind": handler_kind}
    if side_effect_uncertain:
        data["side_effect_state"] = "uncertain"
        data["handler_return"] = safe_json_payload(raw_result, source=f"{handler_kind}.return")
    result_metadata = dict(context.result_metadata)
    if result_metadata:
        data["result_metadata"] = result_metadata
    return WorkflowStepResult(
        step_key=step.step_key,
        action_type=step.action_type,
        outcome=WorkflowOutcome.CANCELLED,
        message=sanitize_diagnostic_message(message or "Workflow cancelled."),
        data=data,
        workflow_step_id=step.workflow_step_id,
    )


def _timeout_step_result(
    step: WorkflowStepSpec,
    message: str,
    *,
    handler_kind: str,
    side_effect_uncertain: bool = False,
    raw_result: object | None = None,
) -> WorkflowStepResult:
    data: dict[str, object] = {"handler_kind": handler_kind}
    if side_effect_uncertain:
        data["side_effect_state"] = "uncertain"
        data["handler_return"] = safe_json_payload(raw_result, source=f"{handler_kind}.return")
    return WorkflowStepResult(
        step_key=step.step_key,
        action_type=step.action_type,
        outcome=WorkflowOutcome.TIMEOUT,
        message=sanitize_diagnostic_message(message or "Workflow deadline exceeded."),
        data=data,
        workflow_step_id=step.workflow_step_id,
    )


def _exception_step_result(
    step: WorkflowStepSpec,
    exc: Exception,
    *,
    handler_kind: str,
) -> WorkflowStepResult:
    diagnostics = safe_exception_diagnostics(exc)
    return WorkflowStepResult(
        step_key=step.step_key,
        action_type=step.action_type,
        outcome=WorkflowOutcome.FATAL_FAILURE,
        message=(
            f"{handler_kind} handler raised {diagnostics['exception_class']}: "
            f"{diagnostics['message']}"
        ),
        data={"handler_kind": handler_kind, "exception": diagnostics},
        workflow_step_id=step.workflow_step_id,
    )


def _invalid_step_result(
    step: WorkflowStepSpec,
    *,
    handler_kind: str,
    message: str,
    raw_result: object,
) -> WorkflowStepResult:
    return WorkflowStepResult(
        step_key=step.step_key,
        action_type=step.action_type,
        outcome=WorkflowOutcome.VALIDATION_FAILURE,
        message=message,
        data={
            "handler_kind": handler_kind,
            "invalid_handler_result": {
                "type": type(raw_result).__name__,
                "message": message,
            },
        },
        workflow_step_id=step.workflow_step_id,
    )


def _serialization_failure_step_result(
    step: WorkflowStepSpec,
    *,
    handler_kind: str,
    safe_data: object,
) -> WorkflowStepResult:
    result = safe_data
    if not hasattr(result, "failure_metadata"):
        return _invalid_step_result(
            step,
            handler_kind=handler_kind,
            message="Handler result metadata is not safely serializable.",
            raw_result=safe_data,
        )
    return WorkflowStepResult(
        step_key=step.step_key,
        action_type=step.action_type,
        outcome=WorkflowOutcome.VALIDATION_FAILURE,
        message="Handler result metadata is not safely serializable.",
        data={
            "handler_kind": handler_kind,
            "serialization_failure": result.failure_metadata(source=f"{handler_kind}.data"),
            "sanitized_data": result.value,
        },
        workflow_step_id=step.workflow_step_id,
    )


def _cancelled_condition_result(
    context: WorkflowExecutionContext,
    message: str,
    *,
    handler_kind: str,
) -> ConditionEvaluation:
    return ConditionEvaluation(
        False,
        outcome=WorkflowOutcome.CANCELLED,
        message=sanitize_diagnostic_message(message or context.cancellation_token.reason or "Workflow cancelled."),
        data={"handler_kind": handler_kind},
    )


def _timeout_condition_result(
    message: str,
    *,
    handler_kind: str,
) -> ConditionEvaluation:
    return ConditionEvaluation(
        False,
        outcome=WorkflowOutcome.TIMEOUT,
        message=sanitize_diagnostic_message(message or "Workflow deadline exceeded."),
        data={"handler_kind": handler_kind},
    )


def _exception_condition_result(
    exc: Exception,
    *,
    handler_kind: str,
) -> ConditionEvaluation:
    diagnostics = safe_exception_diagnostics(exc)
    return ConditionEvaluation(
        False,
        outcome=WorkflowOutcome.FATAL_FAILURE,
        message=(
            f"{handler_kind} handler raised {diagnostics['exception_class']}: "
            f"{diagnostics['message']}"
        ),
        data={"handler_kind": handler_kind, "exception": diagnostics},
    )


def _invalid_condition_result(
    *,
    handler_kind: str,
    message: str,
    raw_result: object,
) -> ConditionEvaluation:
    return ConditionEvaluation(
        False,
        outcome=WorkflowOutcome.VALIDATION_FAILURE,
        message=message,
        data={
            "handler_kind": handler_kind,
            "invalid_handler_result": {
                "type": type(raw_result).__name__,
                "message": message,
            },
        },
    )


def _serialization_failure_condition_result(
    *,
    handler_kind: str,
    safe_data: object,
) -> ConditionEvaluation:
    result = safe_data
    if not hasattr(result, "failure_metadata"):
        return _invalid_condition_result(
            handler_kind=handler_kind,
            message="Condition result metadata is not safely serializable.",
            raw_result=safe_data,
        )
    return ConditionEvaluation(
        False,
        outcome=WorkflowOutcome.VALIDATION_FAILURE,
        message="Condition result metadata is not safely serializable.",
        data={
            "handler_kind": handler_kind,
            "serialization_failure": result.failure_metadata(source=f"{handler_kind}.data"),
            "sanitized_data": result.value,
        },
    )


def _merge_result_metadata(
    data: object,
    context: WorkflowExecutionContext,
) -> dict[str, object]:
    output = dict(data) if isinstance(data, Mapping) else {}
    result_metadata = dict(context.result_metadata)
    if result_metadata:
        output["result_metadata"] = result_metadata
    return output


def _parse_outcome(value: object) -> WorkflowOutcome | None:
    if isinstance(value, WorkflowOutcome):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return WorkflowOutcome[cleaned]
        except KeyError:
            try:
                return WorkflowOutcome(cleaned)
            except ValueError:
                return None
    return None
