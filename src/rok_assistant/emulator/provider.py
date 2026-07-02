from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Callable, Generic, Protocol, TypeVar
from uuid import uuid4

from rok_assistant.paths import SCREENSHOT_DIR


ANDROID_KEYCODE_MIN = 0
ANDROID_KEYCODE_MAX = 288
LEGACY_COMMAND_TIMEOUT_SECONDS = 30
_SHELL_OPERATOR_CHARACTERS = frozenset("&|<>;")
_ANDROID_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_ANDROID_PACKAGE_RE = re.compile(rf"^{_ANDROID_IDENTIFIER}(?:\.{_ANDROID_IDENTIFIER})+$")
_ANDROID_ACTIVITY_RE = re.compile(rf"^\.?{_ANDROID_IDENTIFIER}(?:\.{_ANDROID_IDENTIFIER})*$")


class CommandErrorCategory(str, Enum):
    NONE = "none"
    INVALID_ARGUMENT = "invalid_argument"
    NON_ZERO_EXIT = "non_zero_exit"
    COMMAND_FAILED = "command_failed"
    TIMEOUT = "timeout"
    MISSING_EXECUTABLE = "missing_executable"
    OS_ERROR = "os_error"
    MALFORMED_OUTPUT = "malformed_output"
    ADB_OFFLINE = "adb_offline"
    SCREENSHOT_MISSING = "screenshot_missing"


T = TypeVar("T")


@dataclass(frozen=True)
class BuiltCommand:
    command: tuple[str, ...]
    cwd: str | None


@dataclass(frozen=True)
class EmulatorCommandResult(Generic[T]):
    command: tuple[str, ...]
    cwd: str | None
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    error_category: CommandErrorCategory = CommandErrorCategory.NONE
    error_message: str = ""
    instance_index: int | None = None
    payload: T | None = None
    diagnostics: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and self.error_category == CommandErrorCategory.NONE

    def with_payload(self, payload: T | None) -> EmulatorCommandResult[T]:
        return replace(self, payload=payload)

    def with_error_category(
        self,
        error_category: CommandErrorCategory,
        *,
        error_message: str = "",
    ) -> EmulatorCommandResult[T]:
        return replace(
            self,
            error_category=error_category,
            error_message=error_message or self.error_message,
        )

    def with_diagnostic(self, message: str) -> EmulatorCommandResult[T]:
        if not message:
            return self
        return replace(self, diagnostics=(*self.diagnostics, message))

    def to_completed_process(self) -> subprocess.CompletedProcess[str] | None:
        if self.exit_code is None:
            return None
        return subprocess.CompletedProcess(
            list(self.command),
            self.exit_code,
            stdout=self.stdout,
            stderr=self.stderr,
        )


@dataclass(frozen=True)
class EmulatorInstanceInfo:
    index: int
    name: str
    running: bool
    pid: int | None = None

    def as_legacy_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "running": self.running,
            "pid": self.pid,
        }


@dataclass(frozen=True)
class AdbDeviceInfo:
    index: int
    serial: str = ""
    connected: bool = False


@dataclass(frozen=True)
class EmulatorHealth:
    index: int
    running: bool
    adb_connected: bool
    serial: str = ""


CommandRunner = Callable[[list[str], Path | None, int], subprocess.CompletedProcess[str]]


class EmulatorProvider(Protocol):
    def discover(self) -> EmulatorCommandResult[list[EmulatorInstanceInfo]]:
        ...

    def start(self, index: int) -> EmulatorCommandResult[None]:
        ...

    def stop(self, index: int) -> EmulatorCommandResult[None]:
        ...

    def force_stop_game_package(
        self,
        index: int,
        package_name: str,
    ) -> EmulatorCommandResult[None]:
        ...

    def launch_game_activity(
        self,
        index: int,
        component: str,
    ) -> EmulatorCommandResult[None]:
        ...

    def reboot(self, index: int) -> EmulatorCommandResult[None]:
        ...

    def adb_connect(self, index: int) -> EmulatorCommandResult[None]:
        ...

    def adb_disconnect(self, index: int) -> EmulatorCommandResult[None]:
        ...

    def screenshot(
        self,
        index: int,
        instance_name: str,
        output_dir: Path = SCREENSHOT_DIR,
    ) -> EmulatorCommandResult[Path]:
        ...

    def tap(self, index: int, x: int, y: int) -> EmulatorCommandResult[None]:
        ...

    def swipe(
        self,
        index: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int,
    ) -> EmulatorCommandResult[None]:
        ...

    def keyevent(self, index: int, code: int) -> EmulatorCommandResult[None]:
        ...

    def health_check(self, index: int) -> EmulatorCommandResult[EmulatorHealth]:
        ...


class MEmuCommandBuilder:
    def __init__(self, install_path: str | Path):
        self.set_install_path(install_path)

    def set_install_path(self, install_path: str | Path) -> None:
        self._install_path = validate_path_text(install_path, "install_path")

    @property
    def memuc_path(self) -> str:
        candidate = PureWindowsPath(self._install_path)
        if candidate.name.lower() == "memuc.exe":
            return str(candidate)
        return str(candidate / "memuc.exe")

    @property
    def cwd(self) -> str:
        return str(PureWindowsPath(self.memuc_path).parent)

    def memuc(self, *args: str) -> BuiltCommand:
        validate_command_arguments(args)
        return BuiltCommand((self.memuc_path, *args), self.cwd)

    def adb(self, index: int, *args: str) -> BuiltCommand:
        validate_instance_index(index)
        validate_command_arguments(args)
        return self.memuc("adb", "-i", str(index), *args)


class MEmuEmulatorProvider:
    def __init__(
        self,
        install_path: str | Path,
        *,
        timeout_seconds: int = 30,
        command_runner: CommandRunner | None = None,
    ):
        self.command_builder = MEmuCommandBuilder(install_path)
        self.timeout_seconds = timeout_seconds
        self.command_runner = command_runner

    def set_install_path(self, install_path: str | Path) -> None:
        self.command_builder.set_install_path(install_path)

    @property
    def memuc_path(self) -> str:
        return self.command_builder.memuc_path

    def run_memuc(
        self,
        args: list[str],
        *,
        instance_index: int | None = None,
    ) -> EmulatorCommandResult[None]:
        validation_error = self._validate_command_request(args, instance_index=instance_index)
        if validation_error is not None:
            return validation_error
        return self._execute(
            self.command_builder.memuc(*args),
            instance_index=instance_index,
        )

    def run_adb(
        self,
        index: int,
        args: list[str],
    ) -> EmulatorCommandResult[None]:
        validation_error = self._validate_command_request(args, instance_index=index)
        if validation_error is not None:
            return validation_error
        return self._execute(
            self.command_builder.adb(index, *args),
            instance_index=index,
        )

    def discover(self) -> EmulatorCommandResult[list[EmulatorInstanceInfo]]:
        result = self.run_memuc(["listvms"])
        if not result.succeeded:
            return result.with_payload([])

        instances, malformed_count = parse_memu_listvms(result.stdout)
        result_with_instances: EmulatorCommandResult[list[EmulatorInstanceInfo]] = result.with_payload(
            instances
        )
        if malformed_count and not instances:
            return result_with_instances.with_error_category(
                CommandErrorCategory.MALFORMED_OUTPUT,
                error_message="MEmu listvms output did not contain any valid instance rows.",
            )
        return result_with_instances

    def start(self, index: int) -> EmulatorCommandResult[None]:
        validation_error = invalid_result_for_index(index)
        if validation_error is not None:
            return validation_error
        return self.run_memuc(["start", "-i", str(index)], instance_index=index)

    def stop(self, index: int) -> EmulatorCommandResult[None]:
        validation_error = invalid_result_for_index(index)
        if validation_error is not None:
            return validation_error
        return self.run_memuc(["stop", "-i", str(index)], instance_index=index)

    def stop_all(self) -> EmulatorCommandResult[None]:
        return self.run_memuc(["stopall"])

    def activate(self, index: int) -> EmulatorCommandResult[None]:
        validation_error = invalid_result_for_index(index)
        if validation_error is not None:
            return validation_error
        return self.run_memuc(["activate", "-i", str(index)], instance_index=index)

    def is_running(self, index: int) -> EmulatorCommandResult[bool]:
        validation_error = invalid_result_for_index(index)
        if validation_error is not None:
            return validation_error.with_payload(False)
        result = self.run_memuc(["isvmrunning", "-i", str(index)], instance_index=index)
        output = result.stdout.strip().lower()
        running = result.succeeded and output in {"1", "true", "yes", "running"}
        if result.succeeded and not output:
            discovered = self.discover()
            running = any(
                instance.index == index and instance.running
                for instance in (discovered.payload or [])
            )
        return result.with_payload(running)

    def force_stop_game_package(
        self,
        index: int,
        package_name: str,
    ) -> EmulatorCommandResult[None]:
        validation_error = invalid_result_for_index(index) or validate_package_result(package_name)
        if validation_error is not None:
            return validation_error
        return self.run_adb(index, ["shell", "am", "force-stop", package_name])

    def launch_game_activity(
        self,
        index: int,
        component: str,
    ) -> EmulatorCommandResult[None]:
        validation_error = invalid_result_for_index(index) or validate_component_result(component)
        if validation_error is not None:
            return validation_error
        return self.run_adb(index, ["shell", "am", "start", "-n", component])

    def reboot(self, index: int) -> EmulatorCommandResult[None]:
        validation_error = invalid_result_for_index(index)
        if validation_error is not None:
            return validation_error
        return self.run_memuc(["reboot", "-i", str(index)], instance_index=index)

    def adb_connect(self, index: int) -> EmulatorCommandResult[None]:
        validation_error = invalid_result_for_index(index)
        if validation_error is not None:
            return validation_error
        return self.run_adb(index, ["connect"])

    def adb_disconnect(self, index: int) -> EmulatorCommandResult[None]:
        validation_error = invalid_result_for_index(index)
        if validation_error is not None:
            return validation_error
        return self.run_adb(index, ["disconnect"])

    def adb_status(self, index: int) -> EmulatorCommandResult[AdbDeviceInfo]:
        validation_error = invalid_result_for_index(index)
        if validation_error is not None:
            return validation_error.with_payload(AdbDeviceInfo(index=0))
        result = self.run_adb(index, ["devices"])
        if not result.succeeded:
            return result.with_payload(AdbDeviceInfo(index=index))
        status = parse_adb_devices_output(index, result.stdout)
        result_with_status: EmulatorCommandResult[AdbDeviceInfo] = result.with_payload(status)
        if status.serial and not status.connected:
            return result_with_status.with_error_category(
                CommandErrorCategory.ADB_OFFLINE,
                error_message=f"ADB device {status.serial} is not connected.",
            )
        return result_with_status

    def screenshot(
        self,
        index: int,
        instance_name: str,
        output_dir: Path = SCREENSHOT_DIR,
    ) -> EmulatorCommandResult[Path]:
        validation_error = invalid_result_for_index(index) or validate_string_result(
            instance_name,
            "instance_name",
            allow_empty=True,
        )
        if validation_error is not None:
            return validation_error
        if not isinstance(output_dir, Path):
            return invalid_argument_result("output_dir must be a pathlib.Path.", instance_index=index)

        remote_path = f"/sdcard/rok_capture_{index}_{uuid4().hex}.png"
        mkdir_start = time.perf_counter()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return EmulatorCommandResult(
                command=(),
                cwd=None,
                exit_code=None,
                stdout="",
                stderr=str(exc),
                duration_seconds=time.perf_counter() - mkdir_start,
                error_category=CommandErrorCategory.OS_ERROR,
                error_message=str(exc),
                instance_index=index,
            )

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        safe_name = sanitize_instance_name(instance_name, index)
        destination = output_dir / f"{safe_name}_{index}_{timestamp}.png"
        primary_result: EmulatorCommandResult[Path] | None = None
        cleanup_possible = False

        try:
            cleanup_possible = True
            capture_result = self.run_adb(index, ["shell", "screencap", "-p", remote_path])
            if not capture_result.succeeded:
                primary_result = capture_result.with_payload(None)
            else:
                pull_result = self.run_adb(index, ["pull", remote_path, str(destination)])
                if not pull_result.succeeded:
                    primary_result = pull_result.with_payload(None)
                elif not destination.exists():
                    primary_result = pull_result.with_payload(None).with_error_category(
                        CommandErrorCategory.SCREENSHOT_MISSING,
                        error_message=(
                            "Screenshot pull reported success but file is missing: "
                            f"{destination}"
                        ),
                    )
                else:
                    primary_result = pull_result.with_payload(destination)
        finally:
            if cleanup_possible:
                cleanup_result = self.run_adb(index, ["shell", "rm", remote_path])
                if not cleanup_result.succeeded:
                    cleanup_message = command_failure_message(
                        "Remote screenshot cleanup",
                        cleanup_result,
                    )
                    if primary_result is None:
                        primary_result = cleanup_result.with_payload(None)
                    elif primary_result.succeeded:
                        remove_local_file(destination)
                        primary_result = cleanup_result.with_payload(None).with_error_category(
                            cleanup_result.error_category,
                            error_message=cleanup_message,
                        )
                    else:
                        primary_result = primary_result.with_diagnostic(cleanup_message)
                if primary_result is not None and not primary_result.succeeded:
                    remove_local_file(destination)

        if primary_result is None:
            return invalid_argument_result("Screenshot capture did not produce a result.", instance_index=index)
        return primary_result

    def tap(self, index: int, x: int, y: int) -> EmulatorCommandResult[None]:
        validation_error = (
            invalid_result_for_index(index)
            or validate_non_negative_int_result(x, "x")
            or validate_non_negative_int_result(y, "y")
        )
        if validation_error is not None:
            return validation_error
        return self.run_adb(index, ["shell", "input", "tap", str(x), str(y)])

    def swipe(
        self,
        index: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int,
    ) -> EmulatorCommandResult[None]:
        validation_error = (
            invalid_result_for_index(index)
            or validate_non_negative_int_result(x1, "x1")
            or validate_non_negative_int_result(y1, "y1")
            or validate_non_negative_int_result(x2, "x2")
            or validate_non_negative_int_result(y2, "y2")
            or validate_positive_int_result(duration_ms, "duration_ms")
        )
        if validation_error is not None:
            return validation_error
        return self.run_adb(
            index,
            [
                "shell",
                "input",
                "swipe",
                str(x1),
                str(y1),
                str(x2),
                str(y2),
                str(duration_ms),
            ],
        )

    def keyevent(self, index: int, code: int) -> EmulatorCommandResult[None]:
        validation_error = invalid_result_for_index(index) or validate_keycode_result(code)
        if validation_error is not None:
            return validation_error
        return self.run_adb(index, ["shell", "input", "keyevent", str(code)])

    def health_check(self, index: int) -> EmulatorCommandResult[EmulatorHealth]:
        validation_error = invalid_result_for_index(index)
        if validation_error is not None:
            return validation_error.with_payload(
                EmulatorHealth(index=0, running=False, adb_connected=False)
            )
        running_result = self.is_running(index)
        adb_result = self.adb_status(index)
        status = adb_result.payload or AdbDeviceInfo(index=index)
        health = EmulatorHealth(
            index=index,
            running=bool(running_result.payload),
            adb_connected=status.connected,
            serial=status.serial,
        )
        if adb_result.error_category != CommandErrorCategory.NONE:
            return adb_result.with_payload(health)
        return running_result.with_payload(health)

    @staticmethod
    def _validate_command_request(
        args: Sequence[str],
        *,
        instance_index: int | None,
    ) -> EmulatorCommandResult[None] | None:
        if instance_index is not None:
            index_error = invalid_result_for_index(instance_index)
            if index_error is not None:
                return index_error
        try:
            validate_command_arguments(args)
        except ValueError as exc:
            return invalid_argument_result(str(exc), instance_index=instance_index)
        return None

    def _execute(
        self,
        built_command: BuiltCommand,
        *,
        instance_index: int | None = None,
    ) -> EmulatorCommandResult[None]:
        return execute_built_command(
            built_command,
            timeout_seconds=self.timeout_seconds,
            command_runner=self.command_runner,
            instance_index=instance_index,
        )


def classify_completed_process(
    exit_code: int,
    stdout: str,
    stderr: str,
) -> CommandErrorCategory:
    if exit_code != 0:
        return CommandErrorCategory.NON_ZERO_EXIT
    output = f"{stdout}\n{stderr}".lower()
    if "error" in output or "failed" in output:
        return CommandErrorCategory.COMMAND_FAILED
    return CommandErrorCategory.NONE


def execute_built_command(
    built_command: BuiltCommand,
    *,
    timeout_seconds: int,
    command_runner: CommandRunner | None = None,
    instance_index: int | None = None,
) -> EmulatorCommandResult[None]:
    started_at = time.perf_counter()
    command = list(built_command.command)
    try:
        if command_runner is not None:
            completed = command_runner(
                command,
                Path(built_command.cwd) if built_command.cwd is not None else None,
                timeout_seconds,
            )
        else:
            completed = subprocess.run(
                command,
                cwd=built_command.cwd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
    except FileNotFoundError as exc:
        return exception_result(
            built_command,
            started_at,
            CommandErrorCategory.MISSING_EXECUTABLE,
            str(exc),
            instance_index,
        )
    except subprocess.TimeoutExpired as exc:
        return EmulatorCommandResult(
            command=built_command.command,
            cwd=built_command.cwd,
            exit_code=None,
            stdout=normalize_output(exc.stdout),
            stderr=normalize_output(exc.stderr),
            duration_seconds=time.perf_counter() - started_at,
            error_category=CommandErrorCategory.TIMEOUT,
            error_message=str(exc),
            instance_index=instance_index,
        )
    except OSError as exc:
        return exception_result(
            built_command,
            started_at,
            CommandErrorCategory.OS_ERROR,
            str(exc),
            instance_index,
        )

    stdout = normalize_output(completed.stdout)
    stderr = normalize_output(completed.stderr)
    error_category = classify_completed_process(completed.returncode, stdout, stderr)
    return EmulatorCommandResult(
        command=built_command.command,
        cwd=built_command.cwd,
        exit_code=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=time.perf_counter() - started_at,
        error_category=error_category,
        instance_index=instance_index,
    )


def exception_result(
    built_command: BuiltCommand,
    started_at: float,
    error_category: CommandErrorCategory,
    error_message: str,
    instance_index: int | None,
) -> EmulatorCommandResult[None]:
    return EmulatorCommandResult(
        command=built_command.command,
        cwd=built_command.cwd,
        exit_code=None,
        stdout="",
        stderr=error_message,
        duration_seconds=time.perf_counter() - started_at,
        error_category=error_category,
        error_message=error_message,
        instance_index=instance_index,
    )


def execute_legacy_command(
    command: str | Sequence[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: int = LEGACY_COMMAND_TIMEOUT_SECONDS,
    command_runner: CommandRunner | None = None,
) -> EmulatorCommandResult[None]:
    try:
        command_args = normalize_legacy_command(command)
    except ValueError as exc:
        return invalid_argument_result(str(exc))
    return execute_built_command(
        BuiltCommand(command_args, str(cwd) if cwd is not None else None),
        timeout_seconds=timeout_seconds,
        command_runner=command_runner,
    )


def normalize_legacy_command(command: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(command, str):
        validate_clean_string(command, "command", allow_empty=False)
        return parse_legacy_command_string(command)
    if isinstance(command, bytes) or not isinstance(command, Sequence):
        raise ValueError("command must be a string or sequence of strings.")

    normalized: list[str] = []
    for index, item in enumerate(command):
        if not isinstance(item, str):
            raise ValueError(f"command argument {index} must be a string.")
        validate_clean_string(item, f"command argument {index}", allow_empty=False)
        reject_shell_syntax(item, f"command argument {index}")
        normalized.append(item)
    if not normalized:
        raise ValueError("command must not be empty.")
    return tuple(normalized)


def parse_legacy_command_string(command: str) -> tuple[str, ...]:
    tokens: list[str] = []
    current: list[str] = []
    in_quotes = False
    token_was_quoted = False

    for character in command.strip():
        if character in _SHELL_OPERATOR_CHARACTERS:
            raise ValueError("command contains unsupported shell syntax.")
        if character == '"':
            in_quotes = not in_quotes
            token_was_quoted = True
            continue
        if character.isspace() and not in_quotes:
            if current or token_was_quoted:
                token = "".join(current)
                validate_clean_string(token, "command argument", allow_empty=False)
                tokens.append(token)
                current = []
                token_was_quoted = False
            continue
        current.append(character)

    if in_quotes:
        raise ValueError("command contains an unterminated quoted argument.")
    if current or token_was_quoted:
        token = "".join(current)
        validate_clean_string(token, "command argument", allow_empty=False)
        tokens.append(token)
    if not tokens:
        raise ValueError("command must not be empty.")
    return tuple(tokens)


def validate_path_text(path: str | Path, field_name: str) -> str:
    if not isinstance(path, (str, Path)):
        raise TypeError(f"{field_name} must be a string or pathlib.Path.")
    text = str(path)
    validate_clean_string(text, field_name, allow_empty=False)
    return text


def validate_command_arguments(args: Sequence[str]) -> None:
    if isinstance(args, (str, bytes)):
        raise ValueError("command arguments must be a sequence of strings.")
    for index, arg in enumerate(args):
        if not isinstance(arg, str):
            raise ValueError(f"command argument {index} must be a string.")
        validate_clean_string(arg, f"command argument {index}", allow_empty=False)


def validate_clean_string(value: str, field_name: str, *, allow_empty: bool) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must not be empty.")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain null bytes.")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} must not contain control characters.")


def reject_shell_syntax(value: str, field_name: str) -> None:
    if any(character in _SHELL_OPERATOR_CHARACTERS for character in value):
        raise ValueError(f"{field_name} contains unsupported shell syntax.")


def validate_instance_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("instance index must be an integer.")
    if value < 0:
        raise ValueError("instance index must be non-negative.")
    return value


def validate_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return value


def validate_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return value


def invalid_argument_result(
    message: str,
    *,
    instance_index: int | None = None,
) -> EmulatorCommandResult[None]:
    return EmulatorCommandResult(
        command=(),
        cwd=None,
        exit_code=None,
        stdout="",
        stderr=message,
        duration_seconds=0.0,
        error_category=CommandErrorCategory.INVALID_ARGUMENT,
        error_message=message,
        instance_index=instance_index,
    )


def invalid_result_for_index(index: object) -> EmulatorCommandResult[None] | None:
    try:
        validate_instance_index(index)
    except ValueError as exc:
        return invalid_argument_result(str(exc))
    return None


def validate_non_negative_int_result(
    value: object,
    field_name: str,
) -> EmulatorCommandResult[None] | None:
    try:
        validate_non_negative_int(value, field_name)
    except ValueError as exc:
        return invalid_argument_result(str(exc))
    return None


def validate_positive_int_result(
    value: object,
    field_name: str,
) -> EmulatorCommandResult[None] | None:
    try:
        validate_positive_int(value, field_name)
    except ValueError as exc:
        return invalid_argument_result(str(exc))
    return None


def validate_keycode_result(code: object) -> EmulatorCommandResult[None] | None:
    try:
        keycode = validate_non_negative_int(code, "keycode")
    except ValueError as exc:
        return invalid_argument_result(str(exc))
    if not ANDROID_KEYCODE_MIN <= keycode <= ANDROID_KEYCODE_MAX:
        return invalid_argument_result(
            f"keycode must be between {ANDROID_KEYCODE_MIN} and {ANDROID_KEYCODE_MAX}."
        )
    return None


def validate_string_result(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> EmulatorCommandResult[None] | None:
    try:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string.")
        validate_clean_string(value, field_name, allow_empty=allow_empty)
    except ValueError as exc:
        return invalid_argument_result(str(exc))
    return None


def validate_package_result(package_name: object) -> EmulatorCommandResult[None] | None:
    string_error = validate_string_result(package_name, "package_name")
    if string_error is not None:
        return string_error
    assert isinstance(package_name, str)
    if not _ANDROID_PACKAGE_RE.fullmatch(package_name):
        return invalid_argument_result("package_name is not a valid Android package name.")
    return None


def validate_component_result(component: object) -> EmulatorCommandResult[None] | None:
    string_error = validate_string_result(component, "component")
    if string_error is not None:
        return string_error
    assert isinstance(component, str)
    if "/" not in component:
        return invalid_argument_result("component must be in package/activity form.")
    package_name, activity_name = component.split("/", 1)
    if validate_package_result(package_name) is not None:
        return invalid_argument_result("component package is not a valid Android package name.")
    if not activity_name or not _ANDROID_ACTIVITY_RE.fullmatch(activity_name):
        return invalid_argument_result("component activity is not a valid Android activity name.")
    return None


def command_failure_message(action: str, result: EmulatorCommandResult[object]) -> str:
    detail = (
        result.error_message
        or result.stderr.strip()
        or result.stdout.strip()
        or result.error_category.value
    )
    return f"{action} failed: {detail}"


def remove_local_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def normalize_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def parse_memu_listvms(output: str) -> tuple[list[EmulatorInstanceInfo], int]:
    instances: list[EmulatorInstanceInfo] = []
    malformed_count = 0
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            malformed_count += 1
            continue

        index_text, name, _memory_text, running_text, pid_text = parts
        try:
            index = int(index_text)
            running = int(running_text) == 1
            pid = int(pid_text)
        except ValueError:
            malformed_count += 1
            continue

        instances.append(
            EmulatorInstanceInfo(
                index=index,
                name=name,
                running=running,
                pid=pid if pid > 0 else None,
            )
        )
    return instances, malformed_count


def parse_adb_devices_output(index: int, output: str) -> AdbDeviceInfo:
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("list of devices"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        serial, state = parts[0], parts[1].lower()
        if state == "device":
            return AdbDeviceInfo(index=index, serial=serial, connected=True)
        if state in {"offline", "unauthorized", "disconnect", "disconnected"}:
            return AdbDeviceInfo(index=index, serial=serial, connected=False)
    return AdbDeviceInfo(index=index)


def sanitize_instance_name(instance_name: str, index: int) -> str:
    return (
        "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in instance_name
        ).strip("_")
        or f"instance_{index}"
    )
