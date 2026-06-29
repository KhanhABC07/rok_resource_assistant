from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.emulator import MEmuAdbInputManager
from rok_assistant.emulator.memu_adb_manager import MEmuAdbManager


class MEmuAdbInputManagerTest(unittest.TestCase):
    def test_input_command_generation(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        adb_manager = MEmuAdbManager(r"C:\MEmu\Microvirt\MEmu", text_runner=runner)
        input_manager = MEmuAdbInputManager(adb_manager, 3, "MEmu3")

        self.assertTrue(input_manager.tap(100, 200))
        self.assertTrue(input_manager.swipe(10, 20, 300, 400, 500))
        self.assertTrue(input_manager.keyevent(4))

        self.assertEqual(
            [
                [
                    r"C:\MEmu\Microvirt\MEmu\memuc.exe",
                    "adb",
                    "-i",
                    "3",
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
                    "3",
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
                    "3",
                    "shell",
                    "input",
                    "keyevent",
                    "4",
                ],
            ],
            calls,
        )


if __name__ == "__main__":
    unittest.main()
