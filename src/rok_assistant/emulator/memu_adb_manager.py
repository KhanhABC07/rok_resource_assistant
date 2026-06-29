from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from rok_assistant.emulator.memu_manager import DEFAULT_MEMU_INSTALL_PATH
from rok_assistant.paths import SCREENSHOT_DIR


TextCommandRunner = Callable[[list[str], Path, int], subprocess.CompletedProcess[str]]


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
        self.install_path = Path(install_path)
        self.timeout_seconds = timeout_seconds
        self.text_runner = text_runner
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def memuc_path(self) -> Path:
        if self.install_path.name.lower() == "memuc.exe":
            return self.install_path
        return self.install_path / "memuc.exe"

    def set_install_path(self, install_path: str | Path) -> None:
        self.install_path = Path(install_path)

    def connect_instance(self, index: int) -> bool:
        result = self._run_adb_text(index, ["connect"])
        success = self._command_succeeded(result)
        if success:
            self.logger.info("[MEmu][ADB] Connect instance index %s success", index)
        else:
            self._log_adb_error("Connect", index, result)
        return success

    def disconnect_instance(self, index: int) -> bool:
        result = self._run_adb_text(index, ["disconnect"])
        success = self._command_succeeded(result)
        if success:
            self.logger.info("[MEmu][ADB] Disconnect instance index %s success", index)
        else:
            self._log_adb_error("Disconnect", index, result)
        return success

    def refresh_adb_status(self, indexes: list[int]) -> dict[int, dict[str, object]]:
        statuses: dict[int, dict[str, object]] = {}
        for index in indexes:
            result = self._run_adb_text(index, ["devices"])
            if result is None or result.returncode != 0:
                self._log_adb_error("Refresh status", index, result)
                statuses[index] = {"serial": "", "connected": False}
                continue
            status = self._parse_devices_output(index, result.stdout)
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
        remote_path = "/sdcard/rok_capture.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in instance_name
        ).strip("_") or f"instance_{index}"
        destination = output_dir / f"{safe_name}_{index}_{timestamp}.png"

        capture_result = self._run_adb_text(
            index,
            ["shell", "screencap", "-p", remote_path],
        )
        if not self._command_succeeded(capture_result):
            self._log_adb_error("Capture screenshot", index, capture_result)
            return None

        pull_result = self._run_adb_text(
            index,
            ["pull", remote_path, str(destination)],
        )
        if not self._command_succeeded(pull_result):
            self._log_adb_error("Pull screenshot", index, pull_result)
            self._remove_remote_capture(index, remote_path)
            return None

        if not destination.exists():
            self.logger.error(
                "[MEmu][ADB] Screenshot pull reported success but file is missing: %s",
                destination,
            )
            self._remove_remote_capture(index, remote_path)
            return None

        file_size = destination.stat().st_size
        self.logger.info(
            "[MEmu][ADB] Screenshot captured for index %s: %s (%s bytes)",
            index,
            destination,
            file_size,
        )
        self._remove_remote_capture(index, remote_path)
        return destination

    def _run_adb_text(
        self,
        index: int,
        adb_args: list[str],
    ) -> subprocess.CompletedProcess[str] | None:
        command = [str(self.memuc_path), "adb", "-i", str(index), *adb_args]
        try:
            if self.text_runner is not None:
                return self.text_runner(command, self.memuc_path.parent, self.timeout_seconds)
            return subprocess.run(
                command,
                cwd=str(self.memuc_path.parent),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            self.logger.error("[MEmu][ADB] memuc.exe not found: %s", self.memuc_path)
        except subprocess.TimeoutExpired:
            self.logger.error("[MEmu][ADB] Command timed out: %s", " ".join(command))
        except OSError as exc:
            self.logger.error("[MEmu][ADB] Command failed: %s", exc)
        return None

    @staticmethod
    def _parse_devices_output(index: int, output: str) -> AdbStatus:
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line or line.lower().startswith("list of devices"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            serial, state = parts[0], parts[1].lower()
            if state == "device":
                return AdbStatus(index=index, serial=serial, connected=True)
            if state in {"offline", "unauthorized", "disconnect", "disconnected"}:
                return AdbStatus(index=index, serial=serial, connected=False)
        return AdbStatus(index=index)

    @staticmethod
    def _command_succeeded(result: subprocess.CompletedProcess[str] | None) -> bool:
        if result is None or result.returncode != 0:
            return False
        output = f"{result.stdout}\n{result.stderr}".lower()
        return "error" not in output and "failed" not in output

    def _log_adb_error(
        self,
        action: str,
        index: int,
        result: subprocess.CompletedProcess[str] | None,
    ) -> None:
        if result is None:
            self.logger.error("[MEmu][ADB] %s index %s failed: command did not run", action, index)
            return
        message = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        self.logger.error("[MEmu][ADB] %s index %s failed: %s", action, index, message)

    def _remove_remote_capture(self, index: int, remote_path: str) -> None:
        result = self._run_adb_text(index, ["shell", "rm", remote_path])
        if result is not None and result.returncode == 0:
            return
        self._log_adb_error("Remove remote screenshot", index, result)
