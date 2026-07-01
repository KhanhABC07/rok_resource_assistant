from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from rok_assistant.workflow_context import WorkflowResolver
from rok_assistant.workflow_registry import (
    ActionRegistry,
    ConditionRegistry,
    NormalizerRegistry,
)
from rok_assistant.workflow_types import (
    MAX_CALCULATED_BACKOFF_SECONDS,
    MAX_EXECUTED_STEPS,
    MAX_REPEAT_ITERATIONS,
    MAX_RETRY_BACKOFF_MULTIPLIER,
    MAX_RETRY_DELAY_SECONDS,
    MAX_RETRY_LIMIT,
    MAX_SUB_WORKFLOW_DEPTH,
    SUPPORTED_WORKFLOW_SCHEMA_VERSION,
    WorkflowDefinitionSpec,
    WorkflowStepSpec,
    WorkflowValidationError,
    _float_strict,
    _int_strict,
    _optional_float_strict,
)


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


@dataclass(frozen=True)
class WorkflowValidationLimits:
    max_retry_limit: int = MAX_RETRY_LIMIT
    max_retry_delay_seconds: float = MAX_RETRY_DELAY_SECONDS
    max_retry_backoff_multiplier: float = MAX_RETRY_BACKOFF_MULTIPLIER
    max_calculated_backoff_seconds: float = MAX_CALCULATED_BACKOFF_SECONDS
    max_repeat_iterations: int = MAX_REPEAT_ITERATIONS
    max_executed_steps: int = MAX_EXECUTED_STEPS
    max_sub_workflow_depth: int = MAX_SUB_WORKFLOW_DEPTH


@dataclass(frozen=True)
class SubWorkflowReference:
    workflow_key: str
    path: str


class WorkflowValidator:
    CONTROL_ACTIONS = {"sequence", "bounded_repeat", "if_else", "sub_workflow"}

    def __init__(
        self,
        *,
        action_registry: ActionRegistry,
        condition_registry: ConditionRegistry,
        normalizer_registry: NormalizerRegistry | None = None,
        limits: WorkflowValidationLimits | None = None,
    ) -> None:
        self.action_registry = action_registry
        self.condition_registry = condition_registry
        self.normalizer_registry = normalizer_registry
        self.limits = limits or WorkflowValidationLimits()

    def validate(
        self,
        workflow: WorkflowDefinitionSpec,
        *,
        workflow_resolver: WorkflowResolver | None = None,
    ) -> None:
        errors = self.validation_errors(workflow, workflow_resolver=workflow_resolver)
        if errors:
            raise WorkflowValidationError(errors=errors)

    def validation_errors(
        self,
        workflow: WorkflowDefinitionSpec,
        *,
        workflow_resolver: WorkflowResolver | None = None,
    ) -> list[str]:
        errors: list[str] = []
        self._validate_graph(
            workflow,
            workflow_resolver=workflow_resolver,
            errors=errors,
            active_path=[],
            depth=0,
            visited=set(),
        )
        return errors

    def _validate_graph(
        self,
        workflow: WorkflowDefinitionSpec,
        *,
        workflow_resolver: WorkflowResolver | None,
        errors: list[str],
        active_path: list[str],
        depth: int,
        visited: set[str],
    ) -> None:
        workflow_key = workflow.workflow_key.strip() if isinstance(workflow.workflow_key, str) else "<invalid>"
        workflow_key = workflow_key or "<missing>"
        if workflow_key in active_path:
            cycle = self._cycle_path(active_path, workflow_key)
            errors.append(f"Sub-workflow cycle detected: {' -> '.join(cycle)}.")
            return
        if depth > self.limits.max_sub_workflow_depth:
            chain = " -> ".join([*active_path, workflow_key])
            errors.append(
                "Maximum sub-workflow depth exceeded "
                f"({self.limits.max_sub_workflow_depth}): {chain}."
            )
            return

        already_validated = workflow_key in visited
        if not already_validated:
            visited.add(workflow_key)
            references = self._validate_single_workflow(workflow, depth, errors)
        else:
            references = self._sub_workflow_references(workflow.steps, "steps")

        next_path = [*active_path, workflow_key]
        for reference in references:
            if reference.workflow_key in next_path:
                cycle = self._cycle_path(next_path, reference.workflow_key)
                errors.append(f"Sub-workflow cycle detected: {' -> '.join(cycle)}.")
                continue
            if workflow_resolver is None:
                errors.append(
                    f"{reference.path}: workflow_resolver is required for sub-workflow "
                    f"{reference.workflow_key}."
                )
                continue
            child = workflow_resolver(reference.workflow_key)
            if child is None:
                errors.append(
                    f"{reference.path}: sub-workflow not found: "
                    f"{reference.workflow_key}."
                )
                continue
            self._validate_graph(
                child,
                workflow_resolver=workflow_resolver,
                errors=errors,
                active_path=next_path,
                depth=depth + 1,
                visited=visited,
            )

    def _validate_single_workflow(
        self,
        workflow: WorkflowDefinitionSpec,
        depth: int,
        errors: list[str],
    ) -> list[SubWorkflowReference]:
        path = f"workflow[{workflow.workflow_key or '<missing>'}]"
        schema_version = self._int_field(
            workflow.schema_version,
            f"{path}.schema_version",
            errors,
        )
        if schema_version is not None and schema_version != SUPPORTED_WORKFLOW_SCHEMA_VERSION:
            errors.append(
                f"{path}.schema_version: unsupported schema_version "
                f"{schema_version}."
            )
        if not isinstance(workflow.workflow_key, str):
            errors.append(f"{path}.workflow_key: must be a string.")
        elif not workflow.workflow_key.strip():
            errors.append(f"{path}.workflow_key: workflow_key is required.")
        elif workflow.workflow_key != workflow.workflow_key.strip():
            errors.append(f"{path}.workflow_key: invalid workflow_key.")
        elif not self._valid_identifier(workflow.workflow_key):
            errors.append(f"{path}.workflow_key: invalid workflow_key.")
        version = self._int_field(workflow.version, f"{path}.version", errors)
        if version is not None and version <= 0:
            errors.append(f"{path}.version: version must be greater than zero.")
        if not workflow.steps:
            errors.append(f"{path}.steps: workflow must contain at least one step.")
        self._validate_workflow_config(workflow.config, path, errors)

        seen_keys: set[str] = set()
        references: list[SubWorkflowReference] = []
        for index, step in enumerate(workflow.steps):
            self._validate_step(
                step,
                f"{path}.steps[{index}]",
                seen_keys,
                references,
                errors,
                depth,
            )
        if len(seen_keys) > self.limits.max_executed_steps:
            errors.append(
                f"{path}.steps: static step count cannot exceed "
                f"{self.limits.max_executed_steps}."
            )
        return references

    def _validate_workflow_config(
        self,
        config: Mapping[str, object],
        path: str,
        errors: list[str],
    ) -> None:
        self._validate_optional_upper_bound(
            config,
            ("max_steps", "max_executed_steps"),
            self.limits.max_executed_steps,
            f"{path}.config",
            errors,
        )
        self._validate_optional_upper_bound(
            config,
            ("max_depth", "max_sub_workflow_depth"),
            self.limits.max_sub_workflow_depth,
            f"{path}.config",
            errors,
        )
        self._validate_optional_upper_bound(
            config,
            ("max_repeat_iterations",),
            self.limits.max_repeat_iterations,
            f"{path}.config",
            errors,
        )

    def _validate_optional_upper_bound(
        self,
        config: Mapping[str, object],
        keys: tuple[str, ...],
        maximum: int,
        path: str,
        errors: list[str],
    ) -> None:
        for key in keys:
            if key not in config:
                continue
            value = self._int_field(config.get(key), f"{path}.{key}", errors)
            if value is None:
                continue
            if value < 0:
                errors.append(f"{path}.{key}: must be zero or greater.")
            elif value > maximum:
                errors.append(f"{path}.{key}: cannot exceed {maximum}.")

    def _validate_step(
        self,
        step: WorkflowStepSpec,
        path: str,
        seen_keys: set[str],
        references: list[SubWorkflowReference],
        errors: list[str],
        depth: int,
    ) -> None:
        step_key = step.step_key.strip() if isinstance(step.step_key, str) else ""
        step_key_label = step_key or "<missing>"
        current_path = f"{path}.{step_key_label}"
        if not isinstance(step.step_key, str):
            errors.append(f"{path}.step_key: must be a string.")
        elif not step_key:
            errors.append(f"{path}.step_key: step_key is required.")
        elif step.step_key != step_key:
            errors.append(f"{current_path}: invalid step_key.")
        elif not self._valid_identifier(step_key):
            errors.append(f"{current_path}: invalid step_key.")
        elif step_key in seen_keys:
            errors.append(f"{current_path}: Duplicate step_key: {step_key}.")
        else:
            seen_keys.add(step_key)

        action_type = step.action_type.strip() if isinstance(step.action_type, str) else ""
        action_type_valid = False
        if not isinstance(step.action_type, str):
            errors.append(f"{current_path}.action_type: must be a string.")
        elif not action_type:
            errors.append(f"{current_path}.action_type: action_type is required.")
        elif step.action_type != action_type:
            errors.append(f"{current_path}.action_type: invalid action_type.")
        elif not self._valid_identifier(action_type):
            errors.append(f"{current_path}.action_type: invalid action_type.")
        else:
            action_type_valid = True

        self._validate_retry_policy(step, current_path, errors)

        timeout_seconds = self._optional_float_field(
            step.timeout_seconds,
            f"{current_path}.timeout_seconds",
            errors,
        )
        if timeout_seconds is not None and timeout_seconds <= 0:
            errors.append(f"{current_path}.timeout_seconds: must be greater than zero.")

        if not action_type_valid:
            pass
        elif action_type == "sequence":
            if not step.steps:
                errors.append(f"{current_path}: sequence requires steps.")
        elif action_type == "bounded_repeat":
            self._validate_repeat(step, current_path, errors)
        elif action_type == "if_else":
            self._validate_if_else(step, current_path, errors)
        elif action_type == "sub_workflow":
            self._validate_sub_workflow_reference(
                step,
                current_path,
                references,
                errors,
                depth,
            )
        elif not self.action_registry.contains(action_type):
            errors.append(f"{current_path}: unsupported action_type {action_type}.")
        else:
            self._validate_registered_action(step, current_path, errors)

        self._validate_postcondition(step, current_path, errors)

        for index, child in enumerate(step.steps):
            self._validate_step(
                child,
                f"{current_path}.steps[{index}]",
                seen_keys,
                references,
                errors,
                depth,
            )
        for index, child in enumerate(step.then_steps):
            self._validate_step(
                child,
                f"{current_path}.then_steps[{index}]",
                seen_keys,
                references,
                errors,
                depth,
            )
        for index, child in enumerate(step.else_steps):
            self._validate_step(
                child,
                f"{current_path}.else_steps[{index}]",
                seen_keys,
                references,
                errors,
                depth,
            )

    def _validate_retry_policy(
        self,
        step: WorkflowStepSpec,
        path: str,
        errors: list[str],
    ) -> None:
        retry_limit = self._int_field(step.retry_limit, f"{path}.retry_limit", errors)
        retry_delay_seconds = self._float_field(
            step.retry_delay_seconds,
            f"{path}.retry_delay_seconds",
            errors,
        )
        retry_backoff_multiplier = self._float_field(
            step.retry_backoff_multiplier,
            f"{path}.retry_backoff_multiplier",
            errors,
        )
        max_retry_delay_seconds = self._optional_float_field(
            step.max_retry_delay_seconds,
            f"{path}.max_retry_delay_seconds",
            errors,
        )
        if retry_limit is None or retry_delay_seconds is None or retry_backoff_multiplier is None:
            return

        if retry_limit < 0:
            errors.append(f"{path}.retry_limit: must be zero or greater.")
        elif retry_limit > self.limits.max_retry_limit:
            errors.append(
                f"{path}.retry_limit: cannot exceed {self.limits.max_retry_limit}."
            )
        if retry_delay_seconds < 0:
            errors.append(f"{path}.retry_delay_seconds: must be zero or greater.")
        elif retry_delay_seconds > self.limits.max_retry_delay_seconds:
            errors.append(
                f"{path}.retry_delay_seconds: cannot exceed "
                f"{self.limits.max_retry_delay_seconds}."
            )
        if retry_backoff_multiplier < 1.0:
            errors.append(
                f"{path}.retry_backoff_multiplier: must be greater than or equal to 1."
            )
        elif retry_backoff_multiplier > self.limits.max_retry_backoff_multiplier:
            errors.append(
                f"{path}.retry_backoff_multiplier: cannot exceed "
                f"{self.limits.max_retry_backoff_multiplier}."
            )
        if max_retry_delay_seconds is not None:
            if max_retry_delay_seconds <= 0:
                errors.append(
                    f"{path}.max_retry_delay_seconds: must be greater than zero."
                )
            elif max_retry_delay_seconds > self.limits.max_calculated_backoff_seconds:
                errors.append(
                    f"{path}.max_retry_delay_seconds: cannot exceed "
                    f"{self.limits.max_calculated_backoff_seconds}."
                )
        if retry_limit > 0 and retry_delay_seconds > 0:
            calculated = retry_delay_seconds * (
                retry_backoff_multiplier ** max(0, retry_limit - 1)
            )
            if max_retry_delay_seconds is not None:
                calculated = min(calculated, max_retry_delay_seconds)
            if calculated > self.limits.max_calculated_backoff_seconds:
                errors.append(
                    f"{path}.retry_delay_seconds: calculated backoff cannot exceed "
                    f"{self.limits.max_calculated_backoff_seconds}."
                )

    def _validate_repeat(
        self,
        step: WorkflowStepSpec,
        path: str,
        errors: list[str],
    ) -> None:
        count = self._required_int_parameter(step.parameters, "count", path, errors)
        max_count = self._optional_int_parameter(
            step.parameters,
            "max_count",
            path,
            errors,
            default=count,
        )
        if count is not None and count < 0:
            errors.append(f"{path}.parameters.count: must be zero or greater.")
        if max_count is not None and max_count < 0:
            errors.append(f"{path}.parameters.max_count: must be zero or greater.")
        if count is not None and max_count is not None and count > max_count:
            errors.append(f"{path}.parameters.count: cannot exceed max_count.")
        if count is not None and count > self.limits.max_repeat_iterations:
            errors.append(
                f"{path}.parameters.count: cannot exceed "
                f"{self.limits.max_repeat_iterations}."
            )
        if max_count is not None and max_count > self.limits.max_repeat_iterations:
            errors.append(
                f"{path}.parameters.max_count: cannot exceed "
                f"{self.limits.max_repeat_iterations}."
            )
        if not step.steps:
            errors.append(f"{path}: bounded_repeat requires steps.")

    def _validate_if_else(
        self,
        step: WorkflowStepSpec,
        path: str,
        errors: list[str],
    ) -> None:
        condition_type = self._required_string_parameter(
            step.parameters,
            "condition_type",
            path,
            errors,
        )
        if condition_type is None:
            pass
        elif not self._valid_identifier(condition_type):
            errors.append(f"{path}.parameters.condition_type: invalid condition_type.")
        elif not self.condition_registry.contains(condition_type):
            errors.append(f"{path}: unsupported condition_type {condition_type}.")
        if not step.then_steps and not step.else_steps:
            errors.append(f"{path}: if_else requires then_steps or else_steps.")
        if condition_type is not None:
            self._validate_condition_parameters(condition_type, step.parameters, path, errors)

    def _validate_sub_workflow_reference(
        self,
        step: WorkflowStepSpec,
        path: str,
        references: list[SubWorkflowReference],
        errors: list[str],
        depth: int,
    ) -> None:
        workflow_key = self._required_string_parameter(
            step.parameters,
            "workflow_key",
            path,
            errors,
        )
        if workflow_key is None:
            return
        if not self._valid_identifier(workflow_key):
            errors.append(f"{path}.parameters.workflow_key: invalid workflow_key.")
            return
        if depth + 1 > self.limits.max_sub_workflow_depth:
            errors.append(
                f"{path}.parameters.workflow_key: sub-workflow depth cannot exceed "
                f"{self.limits.max_sub_workflow_depth}."
            )
        references.append(SubWorkflowReference(workflow_key, path))

    def _validate_registered_action(
        self,
        step: WorkflowStepSpec,
        path: str,
        errors: list[str],
    ) -> None:
        registration = self.action_registry.get(step.action_type)
        if registration is not None and registration.validator is not None:
            try:
                errors.extend(f"{path}: {message}" for message in registration.validator(step))
            except Exception as exc:
                errors.append(f"{path}: action validator failed: {exc}.")
        if step.action_type == "normalize_scene":
            self._validate_normalizer_reference(step, path, errors)

    def _validate_normalizer_reference(
        self,
        step: WorkflowStepSpec,
        path: str,
        errors: list[str],
    ) -> None:
        normalizer_type = self._optional_string_parameter(
            step.parameters,
            "normalizer_type",
            f"{path}.parameters.normalizer_type",
            errors,
        )
        if normalizer_type is None:
            return
        if not normalizer_type:
            return
        if not self._valid_identifier(normalizer_type):
            errors.append(f"{path}.parameters.normalizer_type: invalid normalizer_type.")
            return
        if self.normalizer_registry is None:
            errors.append(f"{path}: no normalizer registry configured.")
            return
        registration = self.normalizer_registry.get(normalizer_type)
        if registration is None:
            errors.append(f"{path}: unsupported normalizer_type {normalizer_type}.")
            return
        if registration.validator is not None:
            try:
                errors.extend(
                    f"{path}: {message}" for message in registration.validator(step)
                )
            except Exception as exc:
                errors.append(f"{path}: normalizer validator failed: {exc}.")

    def _validate_postcondition(
        self,
        step: WorkflowStepSpec,
        path: str,
        errors: list[str],
    ) -> None:
        if step.postcondition is None:
            return
        condition_value = step.postcondition.get("condition_type")
        if condition_value is None:
            errors.append(f"{path}.postcondition.condition_type: condition_type is required.")
            return
        if not isinstance(condition_value, str):
            errors.append(f"{path}.postcondition.condition_type: must be a string.")
            return
        condition_type = condition_value.strip()
        if not condition_type:
            errors.append(f"{path}.postcondition.condition_type: condition_type is required.")
            return
        if not self._valid_identifier(condition_type):
            errors.append(f"{path}.postcondition.condition_type: invalid condition_type.")
        elif not self.condition_registry.contains(condition_type):
            errors.append(
                f"{path}: unsupported postcondition condition_type {condition_type}."
            )
        self._validate_condition_parameters(
            condition_type,
            step.postcondition,
            f"{path}.postcondition",
            errors,
        )

    def _validate_condition_parameters(
        self,
        condition_type: str,
        parameters: Mapping[str, object],
        path: str,
        errors: list[str],
    ) -> None:
        registration = self.condition_registry.get(condition_type)
        if registration is not None and registration.validator is not None:
            try:
                errors.extend(f"{path}: {message}" for message in registration.validator(parameters))
            except Exception as exc:
                errors.append(f"{path}: condition validator failed: {exc}.")

    def _sub_workflow_references(
        self,
        steps: list[WorkflowStepSpec],
        path: str,
    ) -> list[SubWorkflowReference]:
        references: list[SubWorkflowReference] = []
        self._collect_references(steps, path, references)
        return references

    def _collect_references(
        self,
        steps: list[WorkflowStepSpec],
        path: str,
        references: list[SubWorkflowReference],
    ) -> None:
        for index, step in enumerate(steps):
            current_path = f"{path}[{index}].{step.step_key or '<missing>'}"
            if step.action_type == "sub_workflow":
                workflow_key = step.parameters.get("workflow_key")
                if not isinstance(workflow_key, str):
                    workflow_key = ""
                workflow_key = workflow_key.strip()
                if workflow_key:
                    references.append(SubWorkflowReference(workflow_key, current_path))
            self._collect_references(step.steps, f"{current_path}.steps", references)
            self._collect_references(
                step.then_steps,
                f"{current_path}.then_steps",
                references,
            )
            self._collect_references(
                step.else_steps,
                f"{current_path}.else_steps",
                references,
            )

    @staticmethod
    def _cycle_path(active_path: list[str], repeated_key: str) -> list[str]:
        try:
            index = active_path.index(repeated_key)
        except ValueError:
            return [*active_path, repeated_key]
        return [*active_path[index:], repeated_key]

    @staticmethod
    def _valid_identifier(value: str) -> bool:
        return bool(_IDENTIFIER_RE.match(value))

    @staticmethod
    def _int_field(value: object, path: str, errors: list[str]) -> int | None:
        try:
            return _int_strict(value, path)
        except WorkflowValidationError as exc:
            errors.extend(exc.errors)
            return None

    @staticmethod
    def _float_field(value: object, path: str, errors: list[str]) -> float | None:
        try:
            return _float_strict(value, path)
        except WorkflowValidationError as exc:
            errors.extend(exc.errors)
            return None

    @staticmethod
    def _optional_float_field(value: object, path: str, errors: list[str]) -> float | None:
        try:
            return _optional_float_strict(value, path)
        except WorkflowValidationError as exc:
            errors.extend(exc.errors)
            return None

    def _required_int_parameter(
        self,
        parameters: Mapping[str, object],
        key: str,
        path: str,
        errors: list[str],
    ) -> int | None:
        if key not in parameters:
            errors.append(f"{path}.parameters.{key}: {key} is required.")
            return None
        return self._int_field(parameters.get(key), f"{path}.parameters.{key}", errors)

    def _optional_int_parameter(
        self,
        parameters: Mapping[str, object],
        key: str,
        path: str,
        errors: list[str],
        *,
        default: int | None,
    ) -> int | None:
        if key not in parameters:
            return default
        return self._int_field(parameters.get(key), f"{path}.parameters.{key}", errors)

    @staticmethod
    def _required_string_parameter(
        parameters: Mapping[str, object],
        key: str,
        path: str,
        errors: list[str],
    ) -> str | None:
        value = parameters.get(key)
        field_path = f"{path}.parameters.{key}"
        if value is None:
            errors.append(f"{field_path}: {key} is required.")
            return None
        if not isinstance(value, str):
            errors.append(f"{field_path}: must be a string.")
            return None
        text = value.strip()
        if not text:
            errors.append(f"{field_path}: {key} is required.")
            return None
        return text

    @staticmethod
    def _optional_string_parameter(
        parameters: Mapping[str, object],
        key: str,
        field_path: str,
        errors: list[str],
    ) -> str | None:
        value = parameters.get(key)
        if value is None:
            return ""
        if not isinstance(value, str):
            errors.append(f"{field_path}: must be a string.")
            return None
        return value.strip()
