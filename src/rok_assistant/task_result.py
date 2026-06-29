from __future__ import annotations

from enum import Enum


class TaskResult(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
