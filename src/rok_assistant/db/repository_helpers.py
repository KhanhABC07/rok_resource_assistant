from __future__ import annotations

import json
from typing import Any


def require_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required.")
    return cleaned


def json_object_text(value: str, field_name: str) -> str:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return json.dumps(parsed, sort_keys=True)


def require_id(value: int | None, field_name: str) -> int:
    if value is None or value <= 0:
        raise ValueError(f"{field_name} is required.")
    return value


def validate_choice(value: str, choices: tuple[str, ...], field_name: str) -> str:
    cleaned = require_text(value, field_name)
    if cleaned not in choices:
        raise ValueError(f"Unsupported {field_name}: {cleaned}")
    return cleaned


def validate_positive(value: int, field_name: str) -> int:
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


def row_id(row: Any | None) -> int:
    return int(row.id if row is not None and row.id is not None else 0)
