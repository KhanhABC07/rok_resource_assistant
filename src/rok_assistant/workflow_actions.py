from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from rok_assistant.action_engine import DEFAULT_ABORT_REASON
from rok_assistant.workflow_context import WorkflowExecutionContext
from rok_assistant.workflow_invocation import invoke_condition_handler, invoke_step_handler
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
    WorkflowCancelledError,
    WorkflowOutcome,
    WorkflowStepResult,
    WorkflowStepSpec,
    WorkflowValidationError,
    _float_strict,
    _float_value,
    _int_value,
)


MAX_TEMPLATE_WAIT_CHUNK_SECONDS = 0.25


def default_action_registry() -> ActionRegistry:
    registry = ActionRegistry()
    registry.register("wait", _wait_action, _validate_wait)
    registry.register("click_semantic_template", _click_semantic_template_action, _validate_template_action)
    registry.register("tap", _tap_action, _validate_tap)
    registry.register("swipe", _swipe_action, _validate_swipe)
    registry.register("delay", _delay_action, _validate_delay)
    registry.register("normalize_scene", _normalize_scene_action)
    registry.register("cancel", _cancel_action)
    registry.freeze()
    return registry


def default_condition_registry() -> ConditionRegistry:
    registry = ConditionRegistry()
    registry.register("always", _always_condition)
    registry.register("template_exists", _template_exists_condition, _validate_template_condition)
    registry.freeze()
    return registry


def default_normalizer_registry() -> NormalizerRegistry:
    registry = NormalizerRegistry()
    registry.freeze()
    return registry


def _wait_action(
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
) -> WorkflowStepResult:
    condition_type = str(step.parameters.get("condition_type", "")).strip()
    if not condition_type:
        seconds = max(0.0, _float_value(step.parameters.get("seconds"), 0.0))
        context.sleep(seconds)
        return _action_result(step, WorkflowOutcome.SUCCESS, data={"elapsed_time": seconds})
    evaluation = _evaluate_builtin_wait_condition(context, step)
    outcome = WorkflowOutcome.SUCCESS if evaluation.matched else WorkflowOutcome.RETRYABLE_FAILURE
    message = "" if evaluation.matched else evaluation.message or "Wait condition was not met."
    if evaluation.outcome not in {WorkflowOutcome.SUCCESS, WorkflowOutcome.SKIPPED}:
        outcome = evaluation.outcome
        message = evaluation.message
    return _action_result(
        step,
        outcome,
        message=message,
        data=evaluation.data,
        screenshot_path=evaluation.screenshot_path,
    )


def _click_semantic_template_action(
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
) -> WorkflowStepResult:
    engine = _require_action_engine(context, step)
    if isinstance(engine, WorkflowStepResult):
        return engine
    template_path, threshold = _resolve_template(context, step)
    if template_path is None:
        return _action_result(
            step,
            WorkflowOutcome.BLOCKED,
            "Template could not be resolved.",
        )
    try:
        result = engine.click_template(str(template_path), threshold=threshold)
    except Exception as exc:
        return _action_result(step, WorkflowOutcome.FATAL_FAILURE, str(exc))
    return _result_from_action_dict(step, result)


def _tap_action(
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
) -> WorkflowStepResult:
    engine = _require_action_engine(context, step)
    if isinstance(engine, WorkflowStepResult):
        return engine
    x = _int_value(step.parameters.get("x"), 0)
    y = _int_value(step.parameters.get("y"), 0)
    try:
        result = engine.click_coordinates(x, y)
    except Exception as exc:
        return _action_result(step, WorkflowOutcome.FATAL_FAILURE, str(exc))
    return _result_from_action_dict(step, result)


def _swipe_action(
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
) -> WorkflowStepResult:
    engine = _require_action_engine(context, step)
    if isinstance(engine, WorkflowStepResult):
        return engine
    try:
        result = engine.swipe_coordinates(
            _int_value(step.parameters.get("x1"), 0),
            _int_value(step.parameters.get("y1"), 0),
            _int_value(step.parameters.get("x2"), 0),
            _int_value(step.parameters.get("y2"), 0),
            _int_value(step.parameters.get("duration_ms"), 500),
        )
    except Exception as exc:
        return _action_result(step, WorkflowOutcome.FATAL_FAILURE, str(exc))
    return _result_from_action_dict(step, result)


def _delay_action(
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
) -> WorkflowStepResult:
    seconds = max(0.0, _float_value(step.parameters.get("seconds"), 0.0))
    context.sleep(seconds)
    return _action_result(step, WorkflowOutcome.SUCCESS, data={"elapsed_time": seconds})


def _normalize_scene_action(
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
) -> WorkflowStepResult:
    normalizer_type = str(step.parameters.get("normalizer_type", "")).strip()
    registry = context.normalizer_registry
    if normalizer_type and isinstance(registry, NormalizerRegistry):
        registration = registry.get(normalizer_type)
        if registration is None:
            return _action_result(
                step,
                WorkflowOutcome.BLOCKED,
                f"Scene normalizer is not registered: {normalizer_type}.",
            )
        return invoke_step_handler(
            registration.handler,
            context,
            step,
            handler_kind="normalizer",
        )
    if context.scene_normalizer is None:
        return _action_result(
            step,
            WorkflowOutcome.SKIPPED,
            "No scene normalizer configured.",
        )
    return invoke_step_handler(
        context.scene_normalizer,
        context,
        step,
        handler_kind="normalizer",
    )


def _cancel_action(
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
) -> WorkflowStepResult:
    reason = str(step.parameters.get("reason") or "").strip()
    engine = context.action_engine
    if engine is not None and hasattr(engine, "abort_task"):
        try:
            if "reason" in step.parameters:
                result = engine.abort_task(reason)
            else:
                result = engine.abort_task()
        except Exception as exc:
            return _action_result(step, WorkflowOutcome.FATAL_FAILURE, str(exc))
        message = str(result.get("message") or reason or DEFAULT_ABORT_REASON)
        context.cancellation_token.cancel(message)
        return _result_from_action_dict(step, result)
    message = reason or DEFAULT_ABORT_REASON
    context.cancellation_token.cancel(message)
    return _action_result(step, WorkflowOutcome.CANCELLED, message)


def _always_condition(
    _context: WorkflowExecutionContext,
    _step: WorkflowStepSpec,
) -> ConditionEvaluation:
    return ConditionEvaluation(True, data={"condition_type": "always"})


def _template_exists_condition(
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
) -> ConditionEvaluation:
    engine = _require_action_engine(context, step)
    if isinstance(engine, WorkflowStepResult):
        return ConditionEvaluation(
            False,
            outcome=engine.outcome,
            message=engine.message,
            data=engine.data,
            screenshot_path=engine.screenshot_path,
        )
    template_path, threshold = _resolve_template(context, step)
    if template_path is None:
        return ConditionEvaluation(
            False,
            outcome=WorkflowOutcome.BLOCKED,
            message="Template could not be resolved.",
        )
    try:
        requested_timeout = max(0.0, _float_value(step.parameters.get("timeout_seconds"), 10.0))
        retry_interval = max(
            0.01,
            _float_value(
                step.parameters.get("retry_interval_seconds"),
                1.0,
            ),
        )
        remaining_timeout = requested_timeout
        result: Mapping[str, object] | None = None
        while result is None or (
            not bool(result.get("success")) and not bool(result.get("fatal"))
        ):
            context.cancellation_token.throw_if_cancelled()
            deadline_remaining = context.deadline.remaining(context.clock)
            if deadline_remaining is not None and deadline_remaining <= 0:
                raise TimeoutError("Workflow deadline exceeded.")
            wait_slice = remaining_timeout
            if deadline_remaining is not None:
                wait_slice = min(wait_slice, deadline_remaining)
            wait_slice = min(wait_slice, MAX_TEMPLATE_WAIT_CHUNK_SECONDS)
            if wait_slice <= 0:
                break
            result = engine.wait_for_template(
                str(template_path),
                threshold=threshold,
                timeout_seconds=wait_slice,
                retry_interval_seconds=min(retry_interval, wait_slice),
            )
            context.cancellation_token.throw_if_cancelled()
            if context.deadline.is_expired(context.clock):
                raise TimeoutError("Workflow deadline exceeded.")
            if result.get("success") or result.get("fatal"):
                break
            if "elapsed_time" not in result:
                break
            elapsed = max(0.0, _float_value(result.get("elapsed_time"), wait_slice))
            remaining_timeout = max(0.0, remaining_timeout - max(elapsed, wait_slice))
            if remaining_timeout <= 0:
                break
        if result is None:
            result = {
                "success": False,
                "message": "timeout",
                "elapsed_time": requested_timeout,
            }
    except (WorkflowCancelledError, TimeoutError):
        raise
    except Exception as exc:
        return ConditionEvaluation(
            False,
            outcome=WorkflowOutcome.FATAL_FAILURE,
            message=str(exc),
        )
    data = dict(result)
    screenshot_path = str(data.get("screenshot_path") or "")
    for key in ("success", "fatal"):
        if key in result and not isinstance(result.get(key), bool):
            return ConditionEvaluation(
                False,
                outcome=WorkflowOutcome.VALIDATION_FAILURE,
                message=f"Condition result {key} field must be a boolean.",
                data={
                    "invalid_handler_result": {
                        "field": key,
                        "type": type(result.get(key)).__name__,
                    }
                },
                screenshot_path=screenshot_path,
            )
    if "success" not in result and not bool(result.get("fatal", False)):
        return ConditionEvaluation(
            False,
            outcome=WorkflowOutcome.VALIDATION_FAILURE,
            message="Condition result must include an explicit boolean success field.",
            data={"invalid_handler_result": {"missing_field": "success"}},
            screenshot_path=screenshot_path,
        )
    if bool(result.get("fatal", False)):
        return ConditionEvaluation(
            False,
            outcome=WorkflowOutcome.FATAL_FAILURE,
            message=str(result.get("message") or "Template condition failed."),
            data=data,
            screenshot_path=screenshot_path,
        )
    matched = result.get("success") is True
    return ConditionEvaluation(
        matched,
        outcome=WorkflowOutcome.SUCCESS,
        message="" if matched else str(result.get("message") or "Template not found."),
        data=data,
        screenshot_path=screenshot_path,
    )


def _evaluate_builtin_wait_condition(
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
) -> ConditionEvaluation:
    condition_type = str(step.parameters.get("condition_type", "")).strip()
    if condition_type == "template_exists":
        return invoke_condition_handler(
            _template_exists_condition,
            context,
            step,
            handler_kind="condition",
        )
    if condition_type == "always":
        return invoke_condition_handler(
            _always_condition,
            context,
            step,
            handler_kind="condition",
        )
    return ConditionEvaluation(
        False,
        WorkflowOutcome.FATAL_FAILURE,
        f"Unsupported condition type: {condition_type}",
    )


def _validate_wait(step: WorkflowStepSpec) -> list[str]:
    condition_type = str(step.parameters.get("condition_type", "")).strip()
    if not condition_type:
        return _validate_delay(step)
    if condition_type == "template_exists":
        return _validate_template_condition(step.parameters)
    return []


def _validate_template_action(step: WorkflowStepSpec) -> list[str]:
    errors: list[str] = []
    if not str(step.parameters.get("template_key", "")).strip() and not str(
        step.parameters.get("template_path", "")
    ).strip():
        errors.append("template_key or template_path is required.")
    errors.extend(_validate_optional_float_parameters(step.parameters, ("threshold",)))
    return errors


def _validate_template_condition(parameters: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    if not str(parameters.get("template_key", "")).strip() and not str(
        parameters.get("template_path", "")
    ).strip():
        errors.append("template_key or template_path is required.")
    errors.extend(
        _validate_optional_float_parameters(
            parameters,
            ("threshold", "timeout_seconds", "retry_interval_seconds"),
        )
    )
    return errors


def _validate_tap(step: WorkflowStepSpec) -> list[str]:
    errors: list[str] = []
    for key in ("x", "y"):
        if key not in step.parameters:
            errors.append(f"{key} is required.")
    return errors


def _validate_swipe(step: WorkflowStepSpec) -> list[str]:
    errors: list[str] = []
    for key in ("x1", "y1", "x2", "y2"):
        if key not in step.parameters:
            errors.append(f"{key} is required.")
    return errors


def _validate_delay(step: WorkflowStepSpec) -> list[str]:
    errors = _validate_optional_float_parameters(step.parameters, ("seconds",))
    if errors:
        return errors
    if _float_value(step.parameters.get("seconds"), 0.0) < 0:
        return ["seconds must be zero or greater."]
    return []


def _validate_optional_float_parameters(
    parameters: Mapping[str, object],
    keys: Sequence[str],
) -> list[str]:
    errors: list[str] = []
    for key in keys:
        value = parameters.get(key)
        if value is None or value == "":
            continue
        try:
            _float_strict(value, key)
        except WorkflowValidationError as exc:
            errors.extend(exc.errors)
    return errors


def _require_action_engine(
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
) -> object | WorkflowStepResult:
    if context.action_engine is None:
        return _action_result(
            step,
            WorkflowOutcome.BLOCKED,
            "No action engine configured.",
        )
    return context.action_engine


def _resolve_template(
    context: WorkflowExecutionContext,
    step: WorkflowStepSpec,
) -> tuple[str | Path | None, float]:
    threshold = _float_value(step.parameters.get("threshold"), 0.8)
    template_key = str(step.parameters.get("template_key", "")).strip()
    if template_key and context.template_resolver is not None:
        template = context.template_resolver(template_key)
        if template is not None:
            return template.file_path, _float_value(step.parameters.get("threshold"), template.threshold)
    template_path = str(step.parameters.get("template_path", "")).strip()
    if template_path:
        return template_path, threshold
    return None, threshold


def _result_from_action_dict(
    step: WorkflowStepSpec,
    result: Mapping[str, object],
) -> WorkflowStepResult:
    data = dict(result)
    screenshot_path = str(data.get("screenshot_path") or "")
    for key in ("success", "fatal", "aborted", "retryable"):
        if key in result and not isinstance(result.get(key), bool):
            return _action_result(
                step,
                WorkflowOutcome.VALIDATION_FAILURE,
                f"Action result {key} field must be a boolean.",
                data={
                    "invalid_handler_result": {
                        "field": key,
                        "type": type(result.get(key)).__name__,
                    }
                },
                screenshot_path=screenshot_path,
            )
    if "success" not in result and not bool(result.get("fatal", False)) and not bool(result.get("aborted", False)):
        return _action_result(
            step,
            WorkflowOutcome.VALIDATION_FAILURE,
            "Action result must include an explicit boolean success field.",
            data={"invalid_handler_result": {"missing_field": "success"}},
            screenshot_path=screenshot_path,
        )
    if bool(result.get("aborted", False)):
        return _action_result(
            step,
            WorkflowOutcome.CANCELLED,
            str(result.get("message") or DEFAULT_ABORT_REASON),
            data=data,
            screenshot_path=screenshot_path,
        )
    if bool(result.get("fatal", False)):
        return _action_result(
            step,
            WorkflowOutcome.FATAL_FAILURE,
            str(result.get("message") or "Action failed."),
            data=data,
            screenshot_path=screenshot_path,
        )
    if result.get("success") is True:
        return _action_result(step, WorkflowOutcome.SUCCESS, data=data, screenshot_path=screenshot_path)
    retryable = bool(result.get("retryable", True))
    return _action_result(
        step,
        WorkflowOutcome.RETRYABLE_FAILURE if retryable else WorkflowOutcome.FATAL_FAILURE,
        str(result.get("message") or "Action failed."),
        data=data,
        screenshot_path=screenshot_path,
    )


def _action_result(
    step: WorkflowStepSpec,
    outcome: WorkflowOutcome,
    message: str = "",
    *,
    data: dict[str, object] | None = None,
    screenshot_path: str = "",
) -> WorkflowStepResult:
    safe_data = _safe_result_data(step, data or {}, source=f"action.{step.step_key}.data")
    if not safe_data.ok:
        outcome = WorkflowOutcome.VALIDATION_FAILURE
        message = "Step result metadata is not safely serializable."
        result_data = {
            "serialization_failure": safe_data.failure_metadata(source=f"action.{step.step_key}.data"),
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
        screenshot_path=screenshot_path,
        workflow_step_id=step.workflow_step_id,
    )


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
