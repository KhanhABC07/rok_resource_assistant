from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rok_assistant.emulator.memu_manager import DEFAULT_MEMU_INSTALL_PATH
from rok_assistant.emulator.provider import (
    CommandErrorCategory,
    EmulatorCommandResult,
    MEmuEmulatorProvider,
    parse_adb_devices_output,
)
from rok_assistant.paths import SCREENSHOT_DIR


TextCommandRunner = Callable[[list[str], Path | None, int], subprocess.CompletedProcess[str]]


@dataclass
class AdbStatus:
    index: int
    serial: str = ""
    connected: bool = False


class MEmuAdbManager:
    def __init__(
        self,
        install_path: str | Path = DEFAULT_MEMU_INSTALL_PATH,
        *,
        timeout_seconds: int = 30,
        text_runner: TextCommandRunner | None = None,
    ):
        self.install_path = str(install_path)
        self.timeout_seconds = timeout_seconds
        self.text_runner = text_runner
        self.provider = MEmuEmulatorProvider(
            self.install_path,
            timeout_seconds=timeout_seconds,
            command_runner=text_runner,
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def memuc_path(self) -> Path:
        return Path(self.provider.memuc_path)

    def set_install_path(self, install_path: str | Path) -> None:
        self.install_path = str(install_path)
        self.provider.set_install_path(self.install_path)

    def connect_instance(self, index: int) -> bool:
        result = self.provider.adb_connect(index)
        success = result.succeeded
        if success:
            self.logger.info("[MEmu][ADB] Connect instance index %s success", index)
        else:
            self._log_adb_error("Connect", index, result)
        return success

    def disconnect_instance(self, index: int) -> bool:
        result = self.provider.adb_disconnect(index)
        success = result.succeeded
        if success:
            self.logger.info("[MEmu][ADB] Disconnect instance index %s success", index)
        else:
            self._log_adb_error("Disconnect", index, result)
        return success

    def refresh_adb_status(self, indexes: list[int]) -> dict[int, dict[str, object]]:
        statuses: dict[int, dict[str, object]] = {}
        for index in indexes:
            result = self.provider.adb_status(index)
            if result.error_category not in {
                CommandErrorCategory.NONE,
                CommandErrorCategory.ADB_OFFLINE,
            }:
                self._log_adb_error("Refresh status", index, result)
                statuses[index] = {"serial": "", "connected": False}
                continue
            status = result.payload or AdbStatus(index=index)
            statuses[index] = {"serial": status.serial, "connected": status.connected}
            self.logger.info(
                "[MEmu][ADB] Status index %s: connected=%s serial=%s",
                index,
                status.connected,
                status.serial or "-",
            )
        return statuses

    def capture_screenshot(
        self,
        index: int,
        instance_name: str,
        output_dir: Path = SCREENSHOT_DIR,
    ) -> Path | None:
        result = self.provider.screenshot(index, instance_name, output_dir)
        if not result.succeeded or result.payload is None:
            self._log_adb_error("Capture screenshot", index, result)
            return None

        file_size = result.payload.stat().st_size
        self.logger.info(
            "[MEmu][ADB] Screenshot captured for index %s: %s (%s bytes)",
            index,
            result.payload,
            file_size,
        )
        return result.payload

    def _run_adb_text(
        self,
        index: int,
        adb_args: list[str],
    ) -> subprocess.CompletedProcess[str] | None:
        result = self.provider.run_adb(index, adb_args)
        return result.to_completed_process()

    @staticmethod
    def _parse_devices_output(index: int, output: str) -> AdbStatus:
        status = parse_adb_devices_output(index, output)
        return AdbStatus(index=status.index, serial=status.serial, connected=status.connected)

    @staticmethod
    def _command_succeeded(
        result: subprocess.CompletedProcess[str] | EmulatorCommandResult[object] | None,
    ) -> bool:
        if isinstance(result, EmulatorCommandResult):
            return result.succeeded
        if result is None or result.returncode != 0:
            return False
        output = f"{result.stdout}\n{result.stderr}".lower()
        return "error" not in output and "failed" not in output

    def _log_adb_error(
        self,
        action: str,
        index: int,
        result: subprocess.CompletedProcess[str] | EmulatorCommandResult[object] | None,
    ) -> None:
        if result is None:
            self.logger.error("[MEmu][ADB] %s index %s failed: command did not run", action, index)
            return
        if isinstance(result, EmulatorCommandResult):
            message = (
                result.error_message
                or result.stderr.strip()
                or result.stdout.strip()
                or result.error_category.value
            )
            self.logger.error("[MEmu][ADB] %s index %s failed: %s", action, index, message)
            return
        message = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        self.logger.error("[MEmu][ADB] %s index %s failed: %s", action, index, message)

    def _remove_remote_capture(self, index: int, remote_path: str) -> None:
        result = self.provider.run_adb(index, ["shell", "rm", remote_path])
        if result.succeeded:
            return
        self._log_adb_error("Remove remote screenshot", index, result)
