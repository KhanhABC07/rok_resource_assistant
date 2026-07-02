from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol


class SchedulerClock(Protocol):
    def utc_now(self) -> datetime:
        ...

    def monotonic(self) -> float:
        ...


class SystemSchedulerClock:
    def utc_now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()


def require_aware_utc(value: datetime, field_name: str = "datetime") -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC).replace(microsecond=0)


def utc_datetime_to_text(value: datetime) -> str:
    return require_aware_utc(value).isoformat(timespec="seconds")


def parse_persisted_utc(value: str, field_name: str = "timestamp") -> datetime:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required.")
    normalized = cleaned.replace("Z", "+00:00")
    if " " in normalized and "T" not in normalized:
        normalized = normalized.replace(" ", "T", 1)
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        # Existing schema defaults and legacy rows are naive UTC strings. New
        # scheduler writes are aware; persisted legacy values are normalized
        # here for compatibility instead of accepted at public datetime inputs.
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(microsecond=0)
