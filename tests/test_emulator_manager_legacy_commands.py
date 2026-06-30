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

from rok_assistant.db.models import Instance  # noqa: E402
from rok_assistant.emulator.manager import EmulatorManager, EmulatorState  # noqa: E402


class FakeInstanceRepository:
    def __init__(self, instances: list[Instance]):
        self._instances = {instance.id: instance for instance in instances}

    def get(self, instance_id: int) -> Instance | None:
        return self._instances.get(instance_id)

    def list_all(self) -> list[Instance]:
        return list(self._instances.values())


class EmulatorManagerLegacyCommandTest(unittest.TestCase):
    def test_launch_and_close_string_commands_use_argument_lists_with_spaces(self) -> None:
        calls: list[tuple[list[str], Path | None, int]] = []

        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            calls.append((command, cwd, timeout))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            launch_path = Path(temp_dir)
            instance = Instance(
                id=1,
                name="Generic",
                launch_path=str(launch_path),
                launch_command=r'"C:\Program Files\Tool\tool.exe" --profile "Farm 1"',
                close_command=r'"C:\Program Files\Tool\tool.exe" --stop',
            )
            manager = EmulatorManager(
                FakeInstanceRepository([instance]),  # type: ignore[arg-type]
                command_timeout_seconds=8,
                command_runner=runner,
            )

            self.assertTrue(manager.launch_instance(instance))
            self.assertEqual(EmulatorState.RUNNING, manager.state_for(1))
            self.assertTrue(manager.close_instance(instance))
            self.assertEqual(EmulatorState.STOPPED, manager.state_for(1))

        self.assertEqual(
            [
                (
                    [r"C:\Program Files\Tool\tool.exe", "--profile", "Farm 1"],
                    launch_path,
                    8,
                ),
                ([r"C:\Program Files\Tool\tool.exe", "--stop"], launch_path, 8),
            ],
            calls,
        )

    def test_launch_rejects_shell_syntax_without_executing(self) -> None:
        calls: list[list[str]] = []

        def runner(
            command: list[str],
            cwd: Path | None,
            timeout: int,
        ) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        instance = Instance(
            id=1,
            name="Generic",
            launch_command="tool.exe --start && del important.txt",
        )
        manager = EmulatorManager(
            FakeInstanceRepository([instance]),  # type: ignore[arg-type]
            command_runner=runner,
        )

        self.assertFalse(manager.launch_instance(instance))
        self.assertEqual(EmulatorState.FAILED, manager.state_for(1))
        self.assertEqual([], calls)


if __name__ == "__main__":
    unittest.main()
