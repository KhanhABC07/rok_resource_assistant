from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.emulator.provider import (  # noqa: E402
    ANDROID_KEYCODE_MAX,
    CommandErrorCategory,
    MEmuCommandBuilder,
    MEmuEmulatorProvider,
    execute_legacy_command,
    normalize_legacy_command,
)


class MEmuCommandBuilderTest(unittest.TestCase):
    def test_uses_windows_path_separator_for_directory_install_path(self) -> None:
        builder = MEmuCommandBuilder(r"C:\Program Files\Microvirt\MEmu")

        command = builder.memuc("start", "-i", "5")

        self.assertEqual(
            (
                r"C:\Program Files\Microvirt\MEmu\memuc.exe",
                "start",
                "-i",
                "5",
            ),
            command.command,
        )
        self.assertEqual(r"C:\Program Files\Microvirt\MEmu", command.cwd)

    def test_uses_windows_path_separator_for_executable_install_path(self) -> None:
        builder = MEmuCommandBuilder(r"C:\Program Files\Microvirt\MEmu\memuc.exe")

        command = builder.adb(3, "shell", "input", "keyevent", "4")

        self.assertEqual(
            (
                r"C:\Program Files\Microvirt\MEmu\memuc.exe",
                "adb",
                "-i",
                "3",
                "shell",
                "input",
                "keyevent",
                "4",
            ),
            command.command,
        )
        self.assertEqual(r"C:\Program Files\Microvirt\MEmu", command.cwd)


class MEmuEmulatorProviderTest(unittest.TestCase):
    def test_provider_records_missing_executable(self) -> None:
        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            raise FileNotFoundError(command[0])

        provider = MEmuEmulatorProvider(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        result = provider.start(2)

        self.assertEqual(CommandErrorCategory.MISSING_EXECUTABLE, result.error_category)
        self.assertIsNone(result.exit_code)
        self.assertEqual(
            (
                r"C:\MEmu\Microvirt\MEmu\memuc.exe",
                "start",
                "-i",
                "2",
            ),
            result.command,
        )
        self.assertGreaterEqual(result.duration_seconds, 0)

    def test_provider_records_timeout(self) -> None:
        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(
                cmd=command,
                timeout=timeout,
                output="partial stdout",
                stderr="partial stderr",
            )

        provider = MEmuEmulatorProvider(
            r"C:\MEmu\Microvirt\MEmu",
            timeout_seconds=7,
            command_runner=runner,
        )

        result = provider.stop(4)

        self.assertEqual(CommandErrorCategory.TIMEOUT, result.error_category)
        self.assertIsNone(result.exit_code)
        self.assertEqual("partial stdout", result.stdout)
        self.assertEqual("partial stderr", result.stderr)

    def test_production_subprocess_decodes_invalid_utf8_with_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "invalid_utf8.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import sys",
                        "sys.stdout.buffer.write(b'\\xffstdout')",
                        "sys.stderr.buffer.write(b'\\xfestderr')",
                    ]
                ),
                encoding="utf-8",
            )
            result = execute_legacy_command(
                [sys.executable, str(script_path)],
                timeout_seconds=5,
            )

        self.assertTrue(result.succeeded)
        self.assertIn("\ufffdstdout", result.stdout)
        self.assertIn("\ufffdstderr", result.stderr)

    def test_production_subprocess_timeout_returns_after_process_termination(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "sleep.py"
            script_path.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
            result = execute_legacy_command(
                [sys.executable, str(script_path)],
                timeout_seconds=0.2,
            )

        self.assertEqual(CommandErrorCategory.TIMEOUT, result.error_category)
        self.assertIsNone(result.exit_code)

    def test_discover_reports_malformed_list_output(self) -> None:
        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="\n  not,a,valid,row  \n0, MEmu , 0, 1, 123\nalso invalid\n",
                stderr="",
            )

        provider = MEmuEmulatorProvider(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        result = provider.discover()

        self.assertTrue(result.succeeded)
        self.assertEqual(1, len(result.payload or []))
        self.assertEqual("MEmu", (result.payload or [])[0].name)

    def test_discover_reports_fully_malformed_list_output(self) -> None:
        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="not,a,valid,row\nalso invalid\n",
                stderr="",
            )

        provider = MEmuEmulatorProvider(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        result = provider.discover()

        self.assertEqual(CommandErrorCategory.MALFORMED_OUTPUT, result.error_category)
        self.assertEqual([], result.payload)

    def test_adb_status_reports_offline_device(self) -> None:
        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="List of devices attached\n127.0.0.1:21513\toffline\n",
                stderr="",
            )

        provider = MEmuEmulatorProvider(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        result = provider.adb_status(1)

        self.assertEqual(CommandErrorCategory.ADB_OFFLINE, result.error_category)
        self.assertIsNotNone(result.payload)
        self.assertEqual("127.0.0.1:21513", result.payload.serial)  # type: ignore[union-attr]
        self.assertFalse(result.payload.connected)  # type: ignore[union-attr]

    def test_screenshot_success_keeps_local_file_and_removes_remote_capture(self) -> None:
        calls: list[list[str]] = []
        png_bytes = b"\x89PNG\r\n\x1a\nfake"

        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            if "pull" in command:
                Path(command[-1]).write_bytes(png_bytes)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        provider = MEmuEmulatorProvider(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = provider.screenshot(2, "MEmu2", Path(temp_dir))

            self.assertTrue(result.succeeded)
            self.assertIsNotNone(result.payload)
            self.assertEqual(png_bytes, result.payload.read_bytes())  # type: ignore[union-attr]
            remote_path = calls[0][-1]
            self.assertEqual(remote_path, calls[1][-2])
            self.assertEqual(["shell", "rm", remote_path], calls[2][-3:])

    def test_screenshot_cleans_up_remote_capture_when_screencap_fails(self) -> None:
        calls: list[list[str]] = []

        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            if "screencap" in command:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="capture failed")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        provider = MEmuEmulatorProvider(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = provider.screenshot(2, "MEmu2", Path(temp_dir))

        self.assertEqual(CommandErrorCategory.NON_ZERO_EXIT, result.error_category)
        self.assertEqual(2, len(calls))
        self.assertIn("screencap", calls[0])
        self.assertEqual(["shell", "rm", calls[0][-1]], calls[1][-3:])

    def test_screenshot_cleans_up_remote_capture_when_pull_fails(self) -> None:
        calls: list[list[str]] = []

        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            if "pull" in command:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="pull failed")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        provider = MEmuEmulatorProvider(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = provider.screenshot(2, "MEmu2", Path(temp_dir))

        self.assertEqual(CommandErrorCategory.NON_ZERO_EXIT, result.error_category)
        remote_path = calls[0][-1]
        self.assertEqual(remote_path, calls[1][-2])
        self.assertEqual(["shell", "rm", remote_path], calls[2][-3:])

    def test_screenshot_removes_partial_local_file_when_pull_fails(self) -> None:
        partial_path: Path | None = None

        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            nonlocal partial_path
            if "pull" in command:
                partial_path = Path(command[-1])
                partial_path.write_bytes(b"partial")
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="pull failed")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        provider = MEmuEmulatorProvider(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = provider.screenshot(2, "MEmu2", Path(temp_dir))
            self.assertEqual(CommandErrorCategory.NON_ZERO_EXIT, result.error_category)
            self.assertIsNotNone(partial_path)
            self.assertFalse(partial_path.exists())  # type: ignore[union-attr]

    def test_screenshot_cleanup_failure_fails_otherwise_successful_capture(self) -> None:
        local_path: Path | None = None

        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            nonlocal local_path
            if "pull" in command:
                local_path = Path(command[-1])
                local_path.write_bytes(b"png")
            if command[-2:] == ["rm", command[-1]]:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="rm failed")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        provider = MEmuEmulatorProvider(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = provider.screenshot(2, "MEmu2", Path(temp_dir))
            self.assertEqual(CommandErrorCategory.NON_ZERO_EXIT, result.error_category)
            self.assertIn("Remote screenshot cleanup failed", result.error_message)
            self.assertIsNotNone(local_path)
            self.assertFalse(local_path.exists())  # type: ignore[union-attr]

    def test_screenshot_preserves_primary_error_when_cleanup_also_fails(self) -> None:
        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            if "screencap" in command:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="capture failed")
            if "rm" in command:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="rm failed")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        provider = MEmuEmulatorProvider(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = provider.screenshot(2, "MEmu2", Path(temp_dir))

        self.assertEqual(CommandErrorCategory.NON_ZERO_EXIT, result.error_category)
        self.assertEqual("capture failed", result.stderr)
        self.assertTrue(result.diagnostics)
        self.assertIn("Remote screenshot cleanup failed", result.diagnostics[0])

    def test_health_check_combines_running_and_adb_status(self) -> None:
        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            if "isvmrunning" in command:
                return subprocess.CompletedProcess(command, 0, stdout="1\n", stderr="")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="List of devices attached\n127.0.0.1:21503\tdevice\n",
                stderr="",
            )

        provider = MEmuEmulatorProvider(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        result = provider.health_check(0)

        self.assertTrue(result.succeeded)
        self.assertIsNotNone(result.payload)
        self.assertTrue(result.payload.running)  # type: ignore[union-attr]
        self.assertTrue(result.payload.adb_connected)  # type: ignore[union-attr]
        self.assertEqual("127.0.0.1:21503", result.payload.serial)  # type: ignore[union-attr]

    def test_provider_builds_game_and_input_commands(self) -> None:
        calls: list[list[str]] = []

        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        provider = MEmuEmulatorProvider(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        provider.force_stop_game_package(6, "com.lilithgame.roc.gp")
        provider.launch_game_activity(6, "com.lilithgame.roc.gp/.UnityPlayerActivity")
        provider.reboot(6)
        provider.adb_connect(6)
        provider.adb_disconnect(6)
        provider.tap(6, 100, 200)
        provider.swipe(6, 10, 20, 300, 400, 500)
        provider.keyevent(6, 4)

        self.assertEqual(
            [
                [
                    r"C:\MEmu\Microvirt\MEmu\memuc.exe",
                    "adb",
                    "-i",
                    "6",
                    "shell",
                    "am",
                    "force-stop",
                    "com.lilithgame.roc.gp",
                ],
                [
                    r"C:\MEmu\Microvirt\MEmu\memuc.exe",
                    "adb",
                    "-i",
                    "6",
                    "shell",
                    "am",
                    "start",
                    "-n",
                    "com.lilithgame.roc.gp/.UnityPlayerActivity",
                ],
                [
                    r"C:\MEmu\Microvirt\MEmu\memuc.exe",
                    "reboot",
                    "-i",
                    "6",
                ],
                [r"C:\MEmu\Microvirt\MEmu\memuc.exe", "adb", "-i", "6", "connect"],
                [r"C:\MEmu\Microvirt\MEmu\memuc.exe", "adb", "-i", "6", "disconnect"],
                [
                    r"C:\MEmu\Microvirt\MEmu\memuc.exe",
                    "adb",
                    "-i",
                    "6",
                    "shell",
                    "input",
                    "tap",
                    "100",
                    "200",
                ],
                [
                    r"C:\MEmu\Microvirt\MEmu\memuc.exe",
                    "adb",
                    "-i",
                    "6",
                    "shell",
                    "input",
                    "swipe",
                    "10",
                    "20",
                    "300",
                    "400",
                    "500",
                ],
                [
                    r"C:\MEmu\Microvirt\MEmu\memuc.exe",
                    "adb",
                    "-i",
                    "6",
                    "shell",
                    "input",
                    "keyevent",
                    "4",
                ],
            ],
            calls,
        )

    def test_valid_boundary_values_execute(self) -> None:
        calls: list[list[str]] = []

        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        provider = MEmuEmulatorProvider(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        results = [
            provider.start(0),
            provider.tap(0, 0, 0),
            provider.swipe(0, 0, 0, 0, 0, 1),
            provider.keyevent(0, ANDROID_KEYCODE_MAX),
            provider.force_stop_game_package(0, "com.example.app"),
            provider.launch_game_activity(0, "com.example.app/.MainActivity"),
        ]

        self.assertTrue(all(result.succeeded for result in results))
        self.assertEqual(6, len(calls))

    def test_invalid_values_do_not_execute(self) -> None:
        calls: list[list[str]] = []

        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        provider = MEmuEmulatorProvider(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        results = [
            provider.start(-1),
            provider.start(True),  # type: ignore[arg-type]
            provider.tap(0, -1, 0),
            provider.tap(0, True, 0),  # type: ignore[arg-type]
            provider.swipe(0, 0, 0, 0, 0, 0),
            provider.keyevent(0, -1),
            provider.keyevent(0, ANDROID_KEYCODE_MAX + 1),
            provider.keyevent(0, False),  # type: ignore[arg-type]
            provider.force_stop_game_package(0, "bad"),
            provider.force_stop_game_package(0, "com.example.\x00bad"),
            provider.force_stop_game_package(0, "com.example.\x1fbad"),
            provider.launch_game_activity(0, "com.example.app/Main-Activity"),
            provider.launch_game_activity(0, "com.example.app\x00/.MainActivity"),
        ]

        self.assertTrue(
            all(result.error_category == CommandErrorCategory.INVALID_ARGUMENT for result in results)
        )
        self.assertEqual([], calls)


class LegacyCommandCompatibilityTest(unittest.TestCase):
    def test_legacy_string_command_supports_quoted_executable_path_with_spaces(self) -> None:
        self.assertEqual(
            (r"C:\Program Files\Tool\tool.exe", "--profile", "Farm 1"),
            normalize_legacy_command(r'"C:\Program Files\Tool\tool.exe" --profile "Farm 1"'),
        )

    def test_legacy_sequence_command_executes_with_timeout_and_cwd(self) -> None:
        calls: list[tuple[list[str], Path | None, int]] = []

        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            calls.append((command, cwd, timeout))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            cwd = Path(temp_dir)
            result = execute_legacy_command(
                [r"C:\Program Files\Tool\tool.exe", "--start"],
                cwd=cwd,
                timeout_seconds=9,
                command_runner=runner,
            )

        self.assertTrue(result.succeeded)
        self.assertEqual([([r"C:\Program Files\Tool\tool.exe", "--start"], cwd, 9)], calls)

    def test_legacy_string_command_rejects_shell_syntax_without_executing(self) -> None:
        calls: list[list[str]] = []

        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        result = execute_legacy_command(
            "tool.exe --start && del important.txt",
            command_runner=runner,
        )

        self.assertEqual(CommandErrorCategory.INVALID_ARGUMENT, result.error_category)
        self.assertEqual([], calls)


if __name__ == "__main__":
    unittest.main()
