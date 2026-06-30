from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Callable

from rok_assistant.emulator.provider import (
    CommandErrorCategory,
    EmulatorCommandResult,
    MEmuEmulatorProvider,
    parse_memu_listvms,
)


DEFAULT_MEMU_INSTALL_PATH = r"C:\MEmu\Microvirt\MEmu"

CommandRunner = Callable[[list[str], Path | None, int], subprocess.CompletedProcess[str]]


class MEmuManager:
    def __init__(
        self,
        install_path: str | Path = DEFAULT_MEMU_INSTALL_PATH,
        *,
        timeout_seconds: int = 30,
        command_runner: CommandRunner | None = None,
    ):
        self.install_path = str(install_path)
        self.timeout_seconds = timeout_seconds
        self.command_runner = command_runner
        self.provider = MEmuEmulatorProvider(
            self.install_path,
            timeout_seconds=timeout_seconds,
            command_runner=command_runner,
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def memuc_path(self) -> Path:
        return Path(self.provider.memuc_path)

    def set_install_path(self, install_path: str | Path) -> None:
        self.install_path = str(install_path)
        self.provider.set_install_path(self.install_path)

    def scan_instances(self) -> list[dict[str, object]]:
        result = self.provider.discover()
        if not result.succeeded and result.error_category != CommandErrorCategory.MALFORMED_OUTPUT:
            self._log_command_error("Scan instances", None, result)
            return []

        if result.error_category == CommandErrorCategory.MALFORMED_OUTPUT:
            self._log_command_error("Scan instances", None, result)
        instances = [
            instance.as_legacy_dict()
            for instance in (result.payload or [])
        ]
        self.logger.info("[MEmu] Scan instances success: %s detected", len(instances))
        return instances

    def launch_instance(self, index: int) -> bool:
        result = self.provider.start(index)
        success = result.succeeded
        if success:
            self.logger.info("[MEmu] Start instance index %s success", index)
        else:
            self._log_command_error("Start instance", index, result)
        return success

    def stop_instance(self, index: int) -> bool:
        result = self.provider.stop(index)
        success = result.succeeded
        if success:
            self.logger.info("[MEmu] Stop instance index %s success", index)
        else:
            self._log_command_error("Stop instance", index, result)
        return success

    def stop_all_instances(self) -> bool:
        result = self.provider.stop_all()
        success = result.succeeded
        if success:
            self.logger.info("[MEmu] Stop all instances success")
        else:
            self._log_command_error("Stop all instances", None, result)
        return success

    def is_running(self, index: int) -> bool:
        result = self.provider.is_running(index)
        if not result.succeeded:
            self._log_command_error("Running check", index, result)
            return False
        running = bool(result.payload)
        self.logger.info("[MEmu] Running check index %s: %s", index, running)
        return running

    def activate_window(self, index: int) -> bool:
        result = self.provider.activate(index)
        success = result.succeeded
        if success:
            self.logger.info("[MEmu] Activate index %s success", index)
        else:
            self._log_command_error("Activate", index, result)
        return success

    def _run_memuc(self, args: list[str]) -> subprocess.CompletedProcess[str] | None:
        result = self.provider.run_memuc(args)
        return result.to_completed_process()

    @staticmethod
    def _parse_listvms(output: str) -> list[dict[str, object]]:
        instances, malformed_count = parse_memu_listvms(output)
        if malformed_count:
            logging.getLogger(MEmuManager.__name__).warning(
                "[MEmu] Skipped %s malformed listvms line(s)",
                malformed_count,
            )
        return [instance.as_legacy_dict() for instance in instances]

    def _is_running_from_scan(self, index: int) -> bool:
        return any(
            item["index"] == index and bool(item["running"])
            for item in self.scan_instances()
        )

    def _log_command_error(
        self,
        action: str,
        index: int | None,
        result: subprocess.CompletedProcess[str] | EmulatorCommandResult[object] | None,
    ) -> None:
        suffix = "" if index is None else f" index {index}"
        if result is None:
            self.logger.error("[MEmu] %s%s failed: command did not run", action, suffix)
            return
        if isinstance(result, EmulatorCommandResult):
            message = (
                result.error_message
                or result.stderr.strip()
                or result.stdout.strip()
                or result.error_category.value
            )
            self.logger.error(
                "[MEmu] %s%s failed: %s",
                action,
                suffix,
                message,
            )
            return
        message = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        self.logger.error("[MEmu] %s%s failed: %s", action, suffix, message)
