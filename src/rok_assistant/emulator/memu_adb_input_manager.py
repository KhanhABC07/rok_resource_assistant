from __future__ import annotations

import logging

from rok_assistant.emulator.memu_adb_manager import MEmuAdbManager
from rok_assistant.emulator.provider import EmulatorCommandResult


class MEmuAdbInputManager:
    def __init__(
        self,
        adb_manager: MEmuAdbManager,
        instance_index: int,
        instance_name: str = "",
    ):
        self.adb_manager = adb_manager
        self.instance_index = instance_index
        self.instance_name = instance_name or f"index {instance_index}"
        self.logger = logging.getLogger(self.__class__.__name__)

    def tap(self, x: int, y: int) -> bool:
        return self._input(
            self.adb_manager.provider.tap(self.instance_index, x, y),
            f"tap {x} {y}",
        )

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> bool:
        return self._input(
            self.adb_manager.provider.swipe(
                self.instance_index,
                x1,
                y1,
                x2,
                y2,
                duration_ms,
            ),
            f"swipe {x1} {y1} {x2} {y2} {duration_ms}",
        )

    def keyevent(self, code: int) -> bool:
        return self._input(
            self.adb_manager.provider.keyevent(self.instance_index, code),
            f"keyevent {code}",
        )

    def _input(self, result: EmulatorCommandResult[None], display_command: str) -> bool:
        self.logger.info(
            "[MEmu][ADB][Input] %s on %s",
            display_command,
            self.instance_name,
        )
        success = result.succeeded
        if success:
            self.logger.info(
                "[MEmu][ADB][Input] %s on %s success",
                display_command,
                self.instance_name,
            )
            return True

        message = (
            result.error_message
            or result.stderr.strip()
            or result.stdout.strip()
            or result.error_category.value
        )
        self.logger.error(
            "[MEmu][ADB][Input] %s on %s failed: %s",
            display_command,
            self.instance_name,
            message,
        )
        return False
