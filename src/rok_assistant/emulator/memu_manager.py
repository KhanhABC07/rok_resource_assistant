from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Callable


DEFAULT_MEMU_INSTALL_PATH = r"C:\MEmu\Microvirt\MEmu"

CommandRunner = Callable[[list[str], Path, int], subprocess.CompletedProcess[str]]


class MEmuManager:
    def __init__(
        self,
        install_path: str | Path = DEFAULT_MEMU_INSTALL_PATH,
        *,
        timeout_seconds: int = 30,
        command_runner: CommandRunner | None = None,
    ):
        self.install_path = Path(install_path)
        self.timeout_seconds = timeout_seconds
        self.command_runner = command_runner
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def memuc_path(self) -> Path:
        if self.install_path.name.lower() == "memuc.exe":
            return self.install_path
        return self.install_path / "memuc.exe"

    def set_install_path(self, install_path: str | Path) -> None:
        self.install_path = Path(install_path)

    def scan_instances(self) -> list[dict[str, object]]:
        result = self._run_memuc(["listvms"])
        if result is None:
            return []
        if result.returncode != 0:
            self.logger.error("[MEmu] Scan instances failed: %s", result.stderr.strip())
            return []

        instances = self._parse_listvms(result.stdout)
        self.logger.info("[MEmu] Scan instances success: %s detected", len(instances))
        return instances

    def launch_instance(self, index: int) -> bool:
        result = self._run_memuc(["start", "-i", str(index)])
        success = result is not None and result.returncode == 0
        if success:
            self.logger.info("[MEmu] Start instance index %s success", index)
        else:
            self._log_command_error("Start instance", index, result)
        return success

    def stop_instance(self, index: int) -> bool:
        result = self._run_memuc(["stop", "-i", str(index)])
        success = result is not None and result.returncode == 0
        if success:
            self.logger.info("[MEmu] Stop instance index %s success", index)
        else:
            self._log_command_error("Stop instance", index, result)
        return success

    def stop_all_instances(self) -> bool:
        result = self._run_memuc(["stopall"])
        success = result is not None and result.returncode == 0
        if success:
            self.logger.info("[MEmu] Stop all instances success")
        else:
            self._log_command_error("Stop all instances", None, result)
        return success

    def is_running(self, index: int) -> bool:
        result = self._run_memuc(["isvmrunning", "-i", str(index)])
        if result is None:
            return False
        output = result.stdout.strip().lower()
        running = result.returncode == 0 and output in {"1", "true", "yes", "running"}
        if not output and result.returncode == 0:
            running = self._is_running_from_scan(index)
        self.logger.info("[MEmu] Running check index %s: %s", index, running)
        return running

    def activate_window(self, index: int) -> bool:
        result = self._run_memuc(["activate", "-i", str(index)])
        success = result is not None and result.returncode == 0
        if success:
            self.logger.info("[MEmu] Activate index %s success", index)
        else:
            self._log_command_error("Activate", index, result)
        return success

    def _run_memuc(self, args: list[str]) -> subprocess.CompletedProcess[str] | None:
        command = [str(self.memuc_path), *args]
        try:
            if self.command_runner is not None:
                return self.command_runner(command, self.memuc_path.parent, self.timeout_seconds)
            return subprocess.run(
                command,
                cwd=str(self.memuc_path.parent),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            self.logger.error("[MEmu] memuc.exe not found: %s", self.memuc_path)
        except subprocess.TimeoutExpired:
            self.logger.error("[MEmu] Command timed out: %s", " ".join(command))
        except OSError as exc:
            self.logger.error("[MEmu] Command failed: %s", exc)
        return None

    @staticmethod
    def _parse_listvms(output: str) -> list[dict[str, object]]:
        instances: list[dict[str, object]] = []
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 5:
                logging.getLogger(MEmuManager.__name__).warning(
                    "[MEmu] Skipping malformed listvms line: %s",
                    line,
                )
                continue

            index_text, name, _memory_text, running_text, pid_text = parts
            try:
                index = int(index_text)
                running = int(running_text) == 1
                pid = int(pid_text)
            except ValueError:
                logging.getLogger(MEmuManager.__name__).warning(
                    "[MEmu] Skipping unparsable listvms line: %s",
                    line,
                )
                continue

            instances.append(
                {
                    "index": index,
                    "name": name,
                    "running": running,
                    "pid": pid if pid > 0 else None,
                }
            )
        return instances

    def _is_running_from_scan(self, index: int) -> bool:
        return any(
            item["index"] == index and bool(item["running"])
            for item in self.scan_instances()
        )

    def _log_command_error(
        self,
        action: str,
        index: int | None,
        result: subprocess.CompletedProcess[str] | None,
    ) -> None:
        suffix = "" if index is None else f" index {index}"
        if result is None:
            self.logger.error("[MEmu] %s%s failed: command did not run", action, suffix)
            return
        message = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        self.logger.error("[MEmu] %s%s failed: %s", action, suffix, message)
