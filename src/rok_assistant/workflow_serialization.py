from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


MAX_SAFE_METADATA_DEPTH = 32
MAX_SAFE_COLLECTION_SIZE = 100
MAX_SAFE_SERIALIZATION_ERRORS = 20
MAX_SAFE_DIAGNOSTIC_LENGTH = 300

_SECRET_KEY_TERMS = (
    "secret",
    "credential",
    "credentials",
    "token",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "cookie",
)
_SECRET_PATTERN = re.compile(
    r"(?i)\b(secret|credential|credentials|token|password|passwd|api[_-]?key|"
    r"authorization|auth|cookie)\b\s*[:=]\s*[^,;\s]+"
)


@dataclass(frozen=True)
class SafeSerializationError:
    path: str
    code: str
    type_name: str
    message: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "code": self.code,
            "type": self.type_name,
            "message": self.message,
        }


@dataclass(frozen=True)
class SafeSerializationResult:
    ok: bool
    value: object
    errors: tuple[SafeSerializationError, ...] = ()

    def failure_metadata(self, *, source: str) -> dict[str, object]:
        return {
            "source": source,
            "error_count": len(self.errors),
            "errors": [error.to_json_dict() for error in self.errors],
        }


def safe_exception_diagnostics(exc: BaseException) -> dict[str, object]:
    return {
        "exception_class": exc.__class__.__name__,
        "exception_module": exc.__class__.__module__,
        "message": sanitize_diagnostic_message(str(exc)),
    }


def sanitize_diagnostic_message(message: str) -> str:
    sanitized = _SECRET_PATTERN.sub(
        lambda match: f"{match.group(1)}=[REDACTED]",
        message,
    )
    if len(sanitized) > MAX_SAFE_DIAGNOSTIC_LENGTH:
        return sanitized[: MAX_SAFE_DIAGNOSTIC_LENGTH - 3] + "..."
    return sanitized


def safe_serialize_metadata(
    value: object,
    *,
    source: str = "metadata",
    max_depth: int = MAX_SAFE_METADATA_DEPTH,
    max_collection_size: int = MAX_SAFE_COLLECTION_SIZE,
) -> SafeSerializationResult:
    errors: list[SafeSerializationError] = []
    active_ids: set[int] = set()

    converted = _convert_value(
        value,
        path="$",
        depth=0,
        max_depth=max_depth,
        max_collection_size=max_collection_size,
        errors=errors,
        active_ids=active_ids,
    )
    limited_errors = tuple(errors[:MAX_SAFE_SERIALIZATION_ERRORS])
    if len(errors) > MAX_SAFE_SERIALIZATION_ERRORS:
        limited_errors = (
            *limited_errors,
            SafeSerializationError(
                path="$",
                code="too_many_errors",
                type_name="serialization",
                message="Additional serialization errors were omitted.",
            ),
        )
    return SafeSerializationResult(ok=not errors, value=converted, errors=limited_errors)


def safe_json_payload(value: object, *, source: str = "payload") -> object:
    result = safe_serialize_metadata(value, source=source)
    if result.ok:
        return result.value
    return {
        "serialization_failure": result.failure_metadata(source=source),
        "sanitized_value": result.value,
    }


def freeze_safe_metadata(value: object) -> object:
    result = safe_serialize_metadata(value, source="metadata")
    return _freeze_value(result.value)


def _convert_value(
    value: object,
    *,
    path: str,
    depth: int,
    max_depth: int,
    max_collection_size: int,
    errors: list[SafeSerializationError],
    active_ids: set[int],
) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return _error_value(
            errors,
            path,
            "non_finite_number",
            type(value).__name__,
            "Non-finite numbers are not JSON serializable.",
        )
    if isinstance(value, Mapping):
        return _convert_mapping(
            value,
            path=path,
            depth=depth,
            max_depth=max_depth,
            max_collection_size=max_collection_size,
            errors=errors,
            active_ids=active_ids,
        )
    if _is_supported_sequence(value):
        return _convert_sequence(
            value,
            path=path,
            depth=depth,
            max_depth=max_depth,
            max_collection_size=max_collection_size,
            errors=errors,
            active_ids=active_ids,
        )
    return _error_value(
        errors,
        path,
        "unsupported_type",
        type(value).__name__,
        f"Values of type {type(value).__name__} are not safely serializable.",
    )


def _convert_mapping(
    value: Mapping[object, object],
    *,
    path: str,
    depth: int,
    max_depth: int,
    max_collection_size: int,
    errors: list[SafeSerializationError],
    active_ids: set[int],
) -> object:
    if depth >= max_depth:
        return _error_value(
            errors,
            path,
            "max_depth_exceeded",
            type(value).__name__,
            f"Maximum serialization depth {max_depth} was exceeded.",
        )
    value_id = id(value)
    if value_id in active_ids:
        return _error_value(
            errors,
            path,
            "cycle_detected",
            type(value).__name__,
            "Cycle detected in mapping metadata.",
        )
    active_ids.add(value_id)
    try:
        if len(value) > max_collection_size:
            _append_error(
                errors,
                path,
                "max_collection_size_exceeded",
                type(value).__name__,
                f"Collection size {len(value)} exceeds limit {max_collection_size}.",
            )
        output: dict[str, object] = {}
        for index, (raw_key, raw_item) in enumerate(value.items()):
            if index >= max_collection_size:
                break
            if not isinstance(raw_key, str):
                output[f"<invalid-key-{index}>"] = _error_value(
                    errors,
                    f"{path}.<key[{index}]>",
                    "unsupported_key_type",
                    type(raw_key).__name__,
                    "Mapping metadata keys must be strings.",
                )
                continue
            if _is_secret_key(raw_key):
                output[raw_key] = "[REDACTED]"
                continue
            output[raw_key] = _convert_value(
                raw_item,
                path=f"{path}.{raw_key}",
                depth=depth + 1,
                max_depth=max_depth,
                max_collection_size=max_collection_size,
                errors=errors,
                active_ids=active_ids,
            )
        return output
    finally:
        active_ids.remove(value_id)


def _convert_sequence(
    value: Sequence[object],
    *,
    path: str,
    depth: int,
    max_depth: int,
    max_collection_size: int,
    errors: list[SafeSerializationError],
    active_ids: set[int],
) -> object:
    if depth >= max_depth:
        return _error_value(
            errors,
            path,
            "max_depth_exceeded",
            type(value).__name__,
            f"Maximum serialization depth {max_depth} was exceeded.",
        )
    value_id = id(value)
    if value_id in active_ids:
        return _error_value(
            errors,
            path,
            "cycle_detected",
            type(value).__name__,
            "Cycle detected in sequence metadata.",
        )
    active_ids.add(value_id)
    try:
        if len(value) > max_collection_size:
            _append_error(
                errors,
                path,
                "max_collection_size_exceeded",
                type(value).__name__,
                f"Collection size {len(value)} exceeds limit {max_collection_size}.",
            )
        output: list[object] = []
        for index, item in enumerate(value):
            if index >= max_collection_size:
                break
            output.append(
                _convert_value(
                    item,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_collection_size=max_collection_size,
                    errors=errors,
                    active_ids=active_ids,
                )
            )
        return output
    finally:
        active_ids.remove(value_id)


def _is_supported_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def _is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(term in lowered for term in _SECRET_KEY_TERMS)


def _append_error(
    errors: list[SafeSerializationError],
    path: str,
    code: str,
    type_name: str,
    message: str,
) -> None:
    errors.append(
        SafeSerializationError(
            path=path,
            code=code,
            type_name=type_name,
            message=message,
        )
    )


def _error_value(
    errors: list[SafeSerializationError],
    path: str,
    code: str,
    type_name: str,
    message: str,
) -> dict[str, object]:
    _append_error(errors, path, code, type_name, message)
    return {
        "serialization_error": {
            "path": path,
            "code": code,
            "type": type_name,
            "message": message,
        }
    }


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value
