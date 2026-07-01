from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import RLock

from rok_assistant.workflow_context import WorkflowExecutionContext
from rok_assistant.workflow_types import (
    ConditionEvaluation,
    WorkflowStepResult,
    WorkflowStepSpec,
)


ActionHandler = Callable[[WorkflowExecutionContext, WorkflowStepSpec], object]
ActionValidator = Callable[[WorkflowStepSpec], list[str]]
ConditionHandler = Callable[[WorkflowExecutionContext, WorkflowStepSpec], object]
ConditionValidator = Callable[[Mapping[str, object]], list[str]]
NormalizerHandler = Callable[[WorkflowExecutionContext, WorkflowStepSpec], object]
NormalizerValidator = Callable[[WorkflowStepSpec], list[str]]


class RegistryFrozenError(ValueError):
    pass


class DuplicateRegistrationError(ValueError):
    pass


class UnknownRegistrationError(KeyError):
    pass


@dataclass(frozen=True)
class ActionRegistration:
    action_type: str
    handler: ActionHandler
    validator: ActionValidator | None = None


@dataclass(frozen=True)
class ConditionRegistration:
    condition_type: str
    handler: ConditionHandler
    validator: ConditionValidator | None = None


@dataclass(frozen=True)
class NormalizerRegistration:
    normalizer_type: str
    handler: NormalizerHandler
    validator: NormalizerValidator | None = None


class _FrozenRegistry:
    _label = "handler"

    def __init__(self) -> None:
        self._lock = RLock()
        self._frozen = False

    @property
    def frozen(self) -> bool:
        with self._lock:
            return self._frozen

    def freeze(self) -> None:
        with self._lock:
            self._frozen = True

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise RegistryFrozenError(f"{self._label} registry is frozen.")

    @staticmethod
    def _clean_name(name: str, field_name: str) -> str:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError(f"{field_name} is required.")
        return cleaned


class ActionRegistry(_FrozenRegistry):
    _label = "action"

    def __init__(self) -> None:
        super().__init__()
        self._actions: dict[str, ActionRegistration] = {}

    def register(
        self,
        action_type: str,
        handler: ActionHandler,
        validator: ActionValidator | None = None,
    ) -> None:
        cleaned = self._clean_name(action_type, "action_type")
        with self._lock:
            self._ensure_mutable()
            if cleaned in self._actions:
                raise DuplicateRegistrationError(
                    f"action_type is already registered: {cleaned}."
                )
            self._actions[cleaned] = ActionRegistration(cleaned, handler, validator)

    def get(self, action_type: str) -> ActionRegistration | None:
        with self._lock:
            return self._actions.get(action_type)

    def require(self, action_type: str) -> ActionRegistration:
        registration = self.get(action_type)
        if registration is None:
            raise UnknownRegistrationError(f"Unknown action_type: {action_type}.")
        return registration

    def contains(self, action_type: str) -> bool:
        with self._lock:
            return action_type in self._actions

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._actions)

    def items(self) -> tuple[tuple[str, ActionRegistration], ...]:
        with self._lock:
            return tuple(self._actions.items())


class ConditionRegistry(_FrozenRegistry):
    _label = "condition"

    def __init__(self) -> None:
        super().__init__()
        self._conditions: dict[str, ConditionRegistration] = {}

    def register(
        self,
        condition_type: str,
        handler: ConditionHandler,
        validator: ConditionValidator | None = None,
    ) -> None:
        cleaned = self._clean_name(condition_type, "condition_type")
        with self._lock:
            self._ensure_mutable()
            if cleaned in self._conditions:
                raise DuplicateRegistrationError(
                    f"condition_type is already registered: {cleaned}."
                )
            self._conditions[cleaned] = ConditionRegistration(cleaned, handler, validator)

    def get(self, condition_type: str) -> ConditionRegistration | None:
        with self._lock:
            return self._conditions.get(condition_type)

    def require(self, condition_type: str) -> ConditionRegistration:
        registration = self.get(condition_type)
        if registration is None:
            raise UnknownRegistrationError(f"Unknown condition_type: {condition_type}.")
        return registration

    def contains(self, condition_type: str) -> bool:
        with self._lock:
            return condition_type in self._conditions

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._conditions)

    def items(self) -> tuple[tuple[str, ConditionRegistration], ...]:
        with self._lock:
            return tuple(self._conditions.items())


class NormalizerRegistry(_FrozenRegistry):
    _label = "normalizer"

    def __init__(self) -> None:
        super().__init__()
        self._normalizers: dict[str, NormalizerRegistration] = {}

    def register(
        self,
        normalizer_type: str,
        handler: NormalizerHandler,
        validator: NormalizerValidator | None = None,
    ) -> None:
        cleaned = self._clean_name(normalizer_type, "normalizer_type")
        with self._lock:
            self._ensure_mutable()
            if cleaned in self._normalizers:
                raise DuplicateRegistrationError(
                    f"normalizer_type is already registered: {cleaned}."
                )
            self._normalizers[cleaned] = NormalizerRegistration(cleaned, handler, validator)

    def get(self, normalizer_type: str) -> NormalizerRegistration | None:
        with self._lock:
            return self._normalizers.get(normalizer_type)

    def require(self, normalizer_type: str) -> NormalizerRegistration:
        registration = self.get(normalizer_type)
        if registration is None:
            raise UnknownRegistrationError(
                f"Unknown normalizer_type: {normalizer_type}."
            )
        return registration

    def contains(self, normalizer_type: str) -> bool:
        with self._lock:
            return normalizer_type in self._normalizers

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._normalizers)

    def items(self) -> tuple[tuple[str, NormalizerRegistration], ...]:
        with self._lock:
            return tuple(self._normalizers.items())
