from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass
class RecoveryDecision:
    should_retry: bool
    retry_at: str | None
    message: str


class ErrorRecoveryPolicy:
    def __init__(self, retry_delay_minutes: int = 10, max_attempts: int = 3):
        self.retry_delay_minutes = retry_delay_minutes
        self.max_attempts = max_attempts
        self.logger = logging.getLogger(self.__class__.__name__)

    def decide(self, attempts: int, error_message: str) -> RecoveryDecision:
        if attempts >= self.max_attempts:
            message = f"Task failed permanently after {attempts} attempts: {error_message}"
            self.logger.error(message)
            return RecoveryDecision(False, None, message)

        retry_at = (
            datetime.now(UTC) + timedelta(minutes=self.retry_delay_minutes)
        ).replace(tzinfo=None, microsecond=0).isoformat()
        message = f"Task will retry at {retry_at}: {error_message}"
        self.logger.warning(message)
        return RecoveryDecision(True, retry_at, message)
