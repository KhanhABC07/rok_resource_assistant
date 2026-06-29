from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.emulator.memu_manager import MEmuManager


class MEmuManagerTest(unittest.TestCase):
    def test_scan_instances_parses_listvms_output(self) -> None:
        def runner(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="0,MEmu,3932470,1,2092\n1,MEmu1,0,0,0\n",
                stderr="",
            )

        manager = MEmuManager(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        self.assertEqual(
            [
                {"index": 0, "name": "MEmu", "running": True, "pid": 2092},
                {"index": 1, "name": "MEmu1", "running": False, "pid": None},
            ],
            manager.scan_instances(),
        )

    def test_running_status_refresh_uses_latest_listvms_output(self) -> None:
        outputs = [
            "0,MEmu,0,0,0\n1,MEmu1,0,0,0\n",
            "0,MEmu,3932470,1,2092\n1,MEmu1,0,0,0\n",
        ]

        def runner(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(command, 0, stdout=outputs.pop(0), stderr="")

        manager = MEmuManager(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        first_scan = manager.scan_instances()
        second_scan = manager.scan_instances()

        self.assertFalse(first_scan[0]["running"])
        self.assertIsNone(first_scan[0]["pid"])
        self.assertTrue(second_scan[0]["running"])
        self.assertEqual(2092, second_scan[0]["pid"])

    def test_start_command_generation_uses_memuc_exe(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        manager = MEmuManager(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        self.assertTrue(manager.launch_instance(5))
        self.assertEqual(
            [[r"C:\MEmu\Microvirt\MEmu\memuc.exe", "start", "-i", "5"]],
            calls,
        )

    def test_stop_command_generation_uses_memuc_exe(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        manager = MEmuManager(r"C:\MEmu\Microvirt\MEmu", command_runner=runner)

        self.assertTrue(manager.stop_instance(7))
        self.assertEqual(
            [[r"C:\MEmu\Microvirt\MEmu\memuc.exe", "stop", "-i", "7"]],
            calls,
        )


if __name__ == "__main__":
    unittest.main()
