from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from rok_assistant.db.models import (
    WorkflowDefinition as PersistedWorkflowDefinition,
    WorkflowStep as PersistedWorkflowStep,
)


SUPPORTED_WORKFLOW_SCHEMA_VERSION = 2
UNKNOWN_FIELD_POLICY = (
    "reject unknown workflow and step specification fields; config, parameters, "
    "and postcondition are explicit JSON object payloads"
)
MAX_RETRY_LIMIT = 10
MAX_RETRY_DELAY_SECONDS = 60.0
MAX_RETRY_BACKOFF_MULTIPLIER = 10.0
MAX_CALCULATED_BACKOFF_SECONDS = 300.0
MAX_REPEAT_ITERATIONS = 100
MAX_EXECUTED_STEPS = 1000
MAX_SUB_WORKFLOW_DEPTH = 20


@dataclass(frozen=True)
class WorkflowFieldError:
    field: str
    message: str

    def format(self) -> str:
        return f"{self.field}: {self.message}"


class WorkflowValidationError(ValueError):
    def __init__(
        self,
        message: str = "",
        *,
        errors: Sequence[str] | None = None,
        field_errors: Sequence[WorkflowFieldError] | None = None,
    ) -> None:
        self.field_errors = tuple(field_errors or ())
        formatted_field_errors = tuple(error.format() for error in self.field_errors)
        self.errors = tuple(errors or ()) + formatted_field_errors
        super().__init__(message or "; ".join(self.errors) or "Workflow validation failed.")


class WorkflowOutcome(Enum):
    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    VALIDATION_FAILURE = "VALIDATION_FAILURE"
    TIMEOUT = "TIMEOUT"
    FATAL_FAILURE = "FATAL_FAILURE"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal_success(self) -> bool:
        return self in {WorkflowOutcome.SUCCESS, WorkflowOutcome.SKIPPED}

    @property
    def is_failure(self) -> bool:
        return not self.is_terminal_success


class WorkflowCancelledError(RuntimeError):
    pass


@dataclass(frozen=True)
class SemanticTemplate:
    template_key: str
    file_path: str | Path
    threshold: float = 0.8


@dataclass
class ConditionEvaluation:
    matched: bool
    outcome: WorkflowOutcome = WorkflowOutcome.SUCCESS
    message: str = ""
    data: dict[str, object] = field(default_factory=dict)
    screenshot_path: str = ""


@dataclass
class WorkflowStepResult:
    step_key: str
    action_type: str
    outcome: WorkflowOutcome
    message: str = ""
    data: dict[str, object] = field(default_factory=dict)
    attempt: int = 1
    started_at: str = ""
    finished_at: str = ""
    screenshot_path: str = ""
    workflow_step_id: int | None = None

    @property
    def success(self) -> bool:
        return self.outcome.is_terminal_success

    def to_json_dict(self) -> dict[str, object]:
        from rok_assistant.workflow_serialization import safe_json_payload

        return {
            "step_key": self.step_key,
            "action_type": self.action_type,
            "outcome": self.outcome.value,
            "message": self.message,
            "attempt": self.attempt,
            "data": safe_json_payload(self.data, source=f"step.{self.step_key}.data"),
        }


@dataclass
class WorkflowExecutionResult:
    workflow_key: str
    schema_version: int
    outcome: WorkflowOutcome
    message: str = ""
    steps: list[WorkflowStepResult] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    job_run_id: int | None = None
    result: dict[str, object] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.outcome.is_terminal_success

    def to_json_dict(self) -> dict[str, object]:
        from rok_assistant.workflow_serialization import safe_json_payload

        return {
            "workflow_key": self.workflow_key,
            "schema_version": self.schema_version,
            "outcome": self.outcome.value,
            "message": self.message,
            "step_count": len(self.steps),
            "result": safe_json_payload(self.result, source=f"workflow.{self.workflow_key}.result"),
        }


_WORKFLOW_FIELDS = {
    "workflow_key",
    "name",
    "schema_version",
    "version",
    "enabled",
    "steps",
    "config",
}
_STEP_FIELDS = {
    "step_key",
    "action_type",
    "parameters",
    "steps",
    "then_steps",
    "else_steps",
    "timeout_seconds",
    "retry_limit",
    "retry_delay_seconds",
    "retry_backoff_multiplier",
    "max_retry_delay_seconds",
    "enabled",
    "postcondition",
}


@dataclass
class WorkflowStepSpec:
    step_key: str
    action_type: str
    parameters: dict[str, object] = field(default_factory=dict)
    steps: list[WorkflowStepSpec] = field(default_factory=list)
    then_steps: list[WorkflowStepSpec] = field(default_factory=list)
    else_steps: list[WorkflowStepSpec] = field(default_factory=list)
    timeout_seconds: float | None = None
    retry_limit: int = 0
    retry_delay_seconds: float = 0.0
    retry_backoff_multiplier: float = 1.0
    max_retry_delay_seconds: float | None = None
    enabled: bool = True
    postcondition: dict[str, object] | None = None
    workflow_step_id: int | None = None
    legacy_order: int | None = None
    legacy_action_type: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], *, path: str = "steps[]") -> WorkflowStepSpec:
        _reject_unknown_fields(value, _STEP_FIELDS, path)
        step_key = _required_string(value, "step_key", path)
        action_type = _required_string(value, "action_type", path)
        parameters = _dict_value(value.get("parameters"), f"{path}.parameters")
        children = [
            cls.from_mapping(item, path=f"{path}.steps[{index}]")
            for index, item in enumerate(_sequence_value(value.get("steps"), f"{path}.steps"))
        ]
        then_steps = [
            cls.from_mapping(item, path=f"{path}.then_steps[{index}]")
            for index, item in enumerate(
                _sequence_value(value.get("then_steps"), f"{path}.then_steps")
            )
        ]
        else_steps = [
            cls.from_mapping(item, path=f"{path}.else_steps[{index}]")
            for index, item in enumerate(
                _sequence_value(value.get("else_steps"), f"{path}.else_steps")
            )
        ]
        return cls(
            step_key=step_key,
            action_type=action_type,
            parameters=parameters,
            steps=children,
            then_steps=then_steps,
            else_steps=else_steps,
            timeout_seconds=_optional_float_strict(
                value.get("timeout_seconds"),
                f"{path}.timeout_seconds",
            ),
            retry_limit=_int_strict(value.get("retry_limit", 0), f"{path}.retry_limit"),
            retry_delay_seconds=_float_strict(
                value.get("retry_delay_seconds", 0.0),
                f"{path}.retry_delay_seconds",
            ),
            retry_backoff_multiplier=_float_strict(
                value.get("retry_backoff_multiplier", 1.0),
                f"{path}.retry_backoff_multiplier",
            ),
            max_retry_delay_seconds=_optional_float_strict(
                value.get("max_retry_delay_seconds"),
                f"{path}.max_retry_delay_seconds",
            ),
            enabled=_bool_strict(value.get("enabled", True), f"{path}.enabled"),
            postcondition=_optional_dict(value.get("postcondition"), f"{path}.postcondition"),
        )


@dataclass
class WorkflowDefinitionSpec:
    workflow_key: str
    name: str = ""
    schema_version: int = SUPPORTED_WORKFLOW_SCHEMA_VERSION
    version: int = 1
    enabled: bool = True
    steps: list[WorkflowStepSpec] = field(default_factory=list)
    config: dict[str, object] = field(default_factory=dict)
    workflow_id: int | None = None

    @classmethod
    def from_json(cls, value: str, *, field_name: str = "workflow_json") -> WorkflowDefinitionSpec:
        return cls.from_mapping(_json_object(value, field_name))

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        path: str = "workflow",
    ) -> WorkflowDefinitionSpec:
        _reject_unknown_fields(value, _WORKFLOW_FIELDS, path)
        workflow_key = _required_string(value, "workflow_key", path)
        if "schema_version" not in value:
            raise WorkflowValidationError(
                field_errors=[
                    WorkflowFieldError(f"{path}.schema_version", "is required."),
                ]
            )
        schema_version = _int_strict(value.get("schema_version"), f"{path}.schema_version")
        _validate_supported_schema_version(schema_version, f"{path}.schema_version")
        if "steps" not in value:
            raise WorkflowValidationError(
                field_errors=[WorkflowFieldError(f"{path}.steps", "is required.")]
            )
        steps = [
            WorkflowStepSpec.from_mapping(item, path=f"{path}.steps[{index}]")
            for index, item in enumerate(_sequence_value(value.get("steps"), f"{path}.steps"))
        ]
        return cls(
            workflow_key=workflow_key,
            name=_optional_string(value.get("name"), "", f"{path}.name"),
            schema_version=schema_version,
            version=_int_strict(value.get("version", 1), f"{path}.version"),
            enabled=_bool_strict(value.get("enabled", True), f"{path}.enabled"),
            steps=steps,
            config=_dict_value(value.get("config"), f"{path}.config"),
        )

    @classmethod
    def from_persisted(
        cls,
        workflow: PersistedWorkflowDefinition,
        steps: Sequence[PersistedWorkflowStep],
    ) -> WorkflowDefinitionSpec:
        config = _json_object(workflow.config_json, "config_json")
        schema_version = _persisted_schema_version(config)
        step_specs: list[WorkflowStepSpec] = []
        for step in sorted(steps, key=lambda item: item.step_order):
            parameters = _json_object(step.parameters_json, "parameters_json")
            step_specs.append(
                WorkflowStepSpec(
                    step_key=step.step_key,
                    action_type=step.action_type,
                    parameters=parameters,
                    steps=[
                        WorkflowStepSpec.from_mapping(item, path=f"{step.step_key}.steps[]")
                        for item in _sequence_value(parameters.get("steps"), "steps")
                    ],
                    then_steps=[
                        WorkflowStepSpec.from_mapping(item, path=f"{step.step_key}.then_steps[]")
                        for item in _sequence_value(parameters.get("then_steps"), "then_steps")
                    ],
                    else_steps=[
                        WorkflowStepSpec.from_mapping(item, path=f"{step.step_key}.else_steps[]")
                        for item in _sequence_value(parameters.get("else_steps"), "else_steps")
                    ],
                    timeout_seconds=step.timeout_seconds,
                    retry_limit=step.retry_limit,
                    retry_delay_seconds=_float_strict(
                        parameters.get("retry_delay_seconds", 0.0),
                        "parameters_json.retry_delay_seconds",
                    ),
                    retry_backoff_multiplier=_float_strict(
                        parameters.get("retry_backoff_multiplier", 1.0),
                        "parameters_json.retry_backoff_multiplier",
                    ),
                    max_retry_delay_seconds=_optional_float_strict(
                        parameters.get("max_retry_delay_seconds"),
                        "parameters_json.max_retry_delay_seconds",
                    ),
                    enabled=step.enabled,
                    postcondition=_optional_dict(
                        parameters.get("postcondition"),
                        "postcondition",
                    ),
                    workflow_step_id=step.id,
                )
            )
        return cls(
            workflow_key=workflow.workflow_key,
            name=workflow.name,
            schema_version=schema_version,
            version=workflow.version,
            enabled=workflow.enabled,
            steps=step_specs,
            config=config,
            workflow_id=workflow.id,
        )


def _persisted_schema_version(config: Mapping[str, object]) -> int:
    if "schema_version" not in config:
        return SUPPORTED_WORKFLOW_SCHEMA_VERSION
    schema_version = _int_strict(config.get("schema_version"), "config_json.schema_version")
    _validate_supported_schema_version(schema_version, "config_json.schema_version")
    return schema_version


def _validate_supported_schema_version(schema_version: int, field_name: str) -> None:
    if schema_version != SUPPORTED_WORKFLOW_SCHEMA_VERSION:
        raise WorkflowValidationError(
            field_errors=[
                WorkflowFieldError(
                    field_name,
                    f"unsupported schema_version {schema_version}.",
                )
            ]
        )


def _reject_unknown_fields(
    value: Mapping[str, object],
    allowed: set[str],
    path: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise WorkflowValidationError(
            field_errors=[
                WorkflowFieldError(
                    path,
                    f"unknown field(s): {', '.join(unknown)}.",
                )
            ]
        )


def _required_string(value: Mapping[str, object], key: str, path: str) -> str:
    if key not in value:
        raise WorkflowValidationError(
            field_errors=[WorkflowFieldError(f"{path}.{key}", "is required.")]
        )
    text = _optional_string(value.get(key), "", f"{path}.{key}")
    if not text:
        raise WorkflowValidationError(
            field_errors=[WorkflowFieldError(f"{path}.{key}", "is required.")]
        )
    return text


def _optional_string(value: object, default: str, field_name: str = "string") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    raise WorkflowValidationError(
        field_errors=[WorkflowFieldError(field_name, "must be a string.")]
    )


def _dict_value(value: object, field_name: str) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise WorkflowValidationError(
        field_errors=[WorkflowFieldError(field_name, "must be a JSON object.")]
    )


def _optional_dict(value: object, field_name: str) -> dict[str, object] | None:
    if value is None:
        return None
    return _dict_value(value, field_name)


def _sequence_value(value: object, field_name: str) -> list[Mapping[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise WorkflowValidationError(
            field_errors=[WorkflowFieldError(field_name, "must be a list.")]
        )
    output: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise WorkflowValidationError(
                field_errors=[
                    WorkflowFieldError(f"{field_name}[{index}]", "must be an object.")
                ]
            )
        output.append(item)
    return output


def _json_object(value: str, field_name: str) -> dict[str, object]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise WorkflowValidationError(
            field_errors=[WorkflowFieldError(field_name, "must be valid JSON.")]
        ) from exc
    if not isinstance(parsed, Mapping):
        raise WorkflowValidationError(
            field_errors=[WorkflowFieldError(field_name, "must be a JSON object.")]
        )
    return dict(parsed)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return _float_value(value, 0.0)


def _optional_float_strict(value: object, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    return _float_strict(value, field_name)


def _bool_strict(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise WorkflowValidationError(
        field_errors=[WorkflowFieldError(field_name, "must be a boolean.")]
    )


def _int_strict(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise WorkflowValidationError(
            field_errors=[WorkflowFieldError(field_name, "must be an integer.")]
        )
    if not isinstance(value, int):
        raise WorkflowValidationError(
            field_errors=[WorkflowFieldError(field_name, "must be an integer.")]
        )
    return value


def _float_strict(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise WorkflowValidationError(
            field_errors=[WorkflowFieldError(field_name, "must be a number.")]
        )
    if not isinstance(value, (int, float)):
        raise WorkflowValidationError(
            field_errors=[WorkflowFieldError(field_name, "must be a number.")]
        )
    parsed = float(value)
    if not math.isfinite(parsed):
        raise WorkflowValidationError(
            field_errors=[WorkflowFieldError(field_name, "must be a finite number.")]
        )
    return parsed


def _int_value(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
