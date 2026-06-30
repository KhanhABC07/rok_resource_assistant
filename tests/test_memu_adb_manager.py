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

from rok_assistant.emulator.memu_adb_manager import MEmuAdbManager


class MEmuAdbManagerTest(unittest.TestCase):
    def test_adb_output_parsing_detects_connected_device(self) -> None:
        status = MEmuAdbManager._parse_devices_output(
            0,
            "List of devices attached\n127.0.0.1:21503\tdevice\n",
        )

        self.assertEqual(0, status.index)
        self.assertEqual("127.0.0.1:21503", status.serial)
        self.assertTrue(status.connected)

    def test_connection_status_detection_handles_offline_device(self) -> None:
        status = MEmuAdbManager._parse_devices_output(
            1,
            "List of devices attached\n127.0.0.1:21513\toffline\n",
        )

        self.assertEqual("127.0.0.1:21513", status.serial)
        self.assertFalse(status.connected)

    def test_refresh_adb_status_maps_instances_by_index(self) -> None:
        outputs = {
            "0": "List of devices attached\n127.0.0.1:21503\tdevice\n",
            "1": "List of devices attached\n127.0.0.1:21513\toffline\n",
        }

        def runner(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
            index = command[command.index("-i") + 1]
            return subprocess.CompletedProcess(command, 0, stdout=outputs[index], stderr="")

        manager = MEmuAdbManager(r"C:\MEmu\Microvirt\MEmu", text_runner=runner)

        self.assertEqual(
            {
                0: {"serial": "127.0.0.1:21503", "connected": True},
                1: {"serial": "127.0.0.1:21513", "connected": False},
            },
            manager.refresh_adb_status([0, 1]),
        )

    def test_connect_and_disconnect_commands_use_memuc_adb(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        manager = MEmuAdbManager(r"C:\MEmu\Microvirt\MEmu", text_runner=runner)

        self.assertTrue(manager.connect_instance(5))
        self.assertTrue(manager.disconnect_instance(5))
        self.assertEqual(
            [
                [r"C:\MEmu\Microvirt\MEmu\memuc.exe", "adb", "-i", "5", "connect"],
                [r"C:\MEmu\Microvirt\MEmu\memuc.exe", "adb", "-i", "5", "disconnect"],
            ],
            calls,
        )

    def test_capture_screenshot_uses_shell_capture_pull_and_cleanup(self) -> None:
        calls: list[list[str]] = []
        png_bytes = b"\x89PNG\r\n\x1a\nfake"

        def runner(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            if "pull" in command:
                local_path = Path(command[-1])
                local_path.write_bytes(png_bytes)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        manager = MEmuAdbManager(r"C:\MEmu\Microvirt\MEmu", text_runner=runner)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = manager.capture_screenshot(2, "MEmu2", Path(temp_dir))

            self.assertIsNotNone(path)
            self.assertEqual(png_bytes, path.read_bytes())  # type: ignore[union-attr]
            remote_path = calls[0][-1]
            self.assertTrue(remote_path.startswith("/sdcard/rok_capture_2_"))
            self.assertTrue(remote_path.endswith(".png"))
            self.assertEqual(
                [
                    [
                        r"C:\MEmu\Microvirt\MEmu\memuc.exe",
                        "adb",
                        "-i",
                        "2",
                        "shell",
                        "screencap",
                        "-p",
                        remote_path,
                    ],
                    [
                        r"C:\MEmu\Microvirt\MEmu\memuc.exe",
                        "adb",
                        "-i",
                        "2",
                        "pull",
                        remote_path,
                        str(path),
                    ],
                    [
                        r"C:\MEmu\Microvirt\MEmu\memuc.exe",
                        "adb",
                        "-i",
                        "2",
                        "shell",
                        "rm",
                        remote_path,
                    ]
                ],
                calls,
            )


if __name__ == "__main__":
    unittest.main()
