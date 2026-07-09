from __future__ import annotations

import contextlib
import contextvars
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator

from .security import RedactingLogFilter, redact_value

CORRELATION_FIELDS = (
    "job_id",
    "run_id",
    "step_id",
    "instance_id",
    "account_id",
    "character_id",
    "feature_key",
    "workflow_version",
    "template_pack_version",
    "incident_id",
    "evidence_path",
)
_LOG_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "rok_log_context",
    default={},
)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = dict(_LOG_CONTEXT.get())
        for field in CORRELATION_FIELDS:
            value = getattr(record, field, context.get(field, None))
            payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(redact_value(payload), sort_keys=True, ensure_ascii=False)


@contextlib.contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    filtered = {key: value for key, value in fields.items() if key in CORRELATION_FIELDS}
    current = dict(_LOG_CONTEXT.get())
    current.update(filtered)
    token = _LOG_CONTEXT.set(current)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def configure_logging(log_file: Path, level_name: str = "INFO") -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, level_name.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonLogFormatter())
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    redacting_filter = RedactingLogFilter()
    file_handler.addFilter(redacting_filter)
    console_handler.addFilter(redacting_filter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)
