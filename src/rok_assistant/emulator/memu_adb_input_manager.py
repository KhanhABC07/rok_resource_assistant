from __future__ import annotations

import logging

from rok_assistant.emulator.memu_adb_manager import MEmuAdbManager


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
        return self._input(["tap", str(x), str(y)], f"tap {x} {y}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> bool:
        return self._input(
            ["swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)],
            f"swipe {x1} {y1} {x2} {y2} {duration_ms}",
        )

    def keyevent(self, code: int) -> bool:
        return self._input(["keyevent", str(code)], f"keyevent {code}")

    def _input(self, input_args: list[str], display_command: str) -> bool:
        self.logger.info(
            "[MEmu][ADB][Input] %s on %s",
            display_command,
            self.instance_name,
        )
        result = self.adb_manager._run_adb_text(  # noqa: SLF001 - same-layer adapter.
            self.instance_index,
            ["shell", "input", *input_args],
        )
        success = self.adb_manager._command_succeeded(result)  # noqa: SLF001
        if success:
            self.logger.info(
                "[MEmu][ADB][Input] %s on %s success",
                display_command,
                self.instance_name,
            )
            return True

        if result is None:
            message = "command did not run"
        else:
            message = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        self.logger.error(
            "[MEmu][ADB][Input] %s on %s failed: %s",
            display_command,
            self.instance_name,
            message,
        )
        return False
