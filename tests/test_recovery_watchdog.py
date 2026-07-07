from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tests.db_helpers import SRC_ROOT  # noqa: F401

from rok_assistant.db.database import Database
from rok_assistant.db.models import Instance
from rok_assistant.db.repositories import (
    IncidentRepository,
    InstanceCircuitBreakerRepository,
    InstanceRepository,
    RecoveryAttemptRepository,
)
from rok_assistant.emulator.provider import (
    CommandErrorCategory,
    EmulatorCommandResult,
    EmulatorHealth,
)
from rok_assistant.recovery_watchdog import (
    RecoveryPhase,
    RecoveryWatchdog,
    WatchdogConfig,
    WatchdogIssue,
)


class FakeEmulator:
    def __init__(self, screenshot_dir: Path) -> None:
        self.running = True
        self.adb_connected = True
        self.booted = True
        self.activity = "com.lilithgame.roc.gp/.UnityPlayerActivity"
        self.blank_screenshot = False
        self.calls: list[str] = []
        self.screenshot_dir = screenshot_dir

    def health_check(self, index: int) -> EmulatorCommandResult[EmulatorHealth]:
        self.calls.append("health_check")
        return self._ok(
            payload=EmulatorHealth(
                index=index,
                running=self.running,
                adb_connected=self.adb_connected,
            )
        )

    def adb_connect(self, index: int) -> EmulatorCommandResult[None]:
        self.calls.append("adb_connect")
        self.adb_connected = True
        return self._ok()

    def keyevent(self, index: int, code: int) -> EmulatorCommandResult[None]:
        self.calls.append(f"keyevent:{code}")
        return self._ok()

    def force_stop_game_package(
        self,
        index: int,
        package_name: str,
    ) -> EmulatorCommandResult[None]:
        self.calls.append("force_stop")
        self.activity = ""
        return self._ok()

    def launch_game_activity(
        self,
        index: int,
        component: str,
    ) -> EmulatorCommandResult[None]:
        self.calls.append("launch_game")
        self.activity = component
        return self._ok()

    def stop(self, index: int) -> EmulatorCommandResult[None]:
        self.calls.append("stop")
        self.running = False
        return self._ok()

    def start(self, index: int) -> EmulatorCommandResult[None]:
        self.calls.append("start")
        self.running = True
        self.adb_connected = True
        self.booted = True
        self.activity = "com.lilithgame.roc.gp/.UnityPlayerActivity"
        self.blank_screenshot = False
        return self._ok()

    def screenshot(
        self,
        index: int,
        instance_name: str,
        output_dir: Path,
    ) -> EmulatorCommandResult[Path]:
        self.calls.append("screenshot")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{len(self.calls)}.png"
        if self.blank_screenshot:
            image = np.zeros((40, 40, 3), dtype=np.uint8)
        else:
            image = np.zeros((40, 40, 3), dtype=np.uint8)
            image[10:30, 10:30] = (255, 255, 255)
        self._write(path, image)
        return self._ok(payload=path)

    def run_adb(self, index: int, args: list[str]) -> EmulatorCommandResult[None]:
        self.calls.append("adb:" + " ".join(args))
        if args[-1:] == ["sys.boot_completed"]:
            return self._ok(stdout="1\n" if self.booted else "0\n")
        if args[:3] == ["shell", "dumpsys", "window"]:
            if not self.activity:
                return self._ok(stdout="")
            return self._ok(stdout=f"mCurrentFocus=Window{{u0 {self.activity}}}\n")
        return self._ok()

    @staticmethod
    def _write(path: Path, image: np.ndarray) -> None:
        if not cv2.imwrite(str(path), image):
            raise AssertionError(f"Could not write screenshot {path}")

    @staticmethod
    def _ok(
        *,
        stdout: str = "",
        payload: object | None = None,
    ) -> EmulatorCommandResult[object]:
        return EmulatorCommandResult(
            command=(),
            cwd=None,
            exit_code=0,
            stdout=stdout,
            stderr="",
            duration_seconds=0.0,
            payload=payload,
        )


class RecoveryWatchdogTest(unittest.TestCase):
    def test_config_from_mapping_loads_phase_timeouts(self) -> None:
        config = WatchdogConfig.from_mapping(
            {
                "same_screen_timeout_seconds": 7,
                "same_screen_max_observations": 4,
                "phase_timeouts": {
                    "reconnect_adb": 3,
                    "restart_emulator": 90,
                },
            }
        )

        self.assertEqual(7.0, config.same_screen_timeout_seconds)
        self.assertEqual(4, config.same_screen_max_observations)
        self.assertEqual(3.0, config.phase_timeouts[RecoveryPhase.RECONNECT_ADB])
        self.assertEqual(90.0, config.phase_timeouts[RecoveryPhase.RESTART_EMULATOR])

    def test_adb_offline_recovers_by_reconnecting_adb(self) -> None:
        with self._fixture() as fixture:
            fixture.emulator.adb_connected = False

            result = fixture.watchdog.monitor(
                instance_id=fixture.instance_id,
                instance_index=0,
                instance_name="MEmu0",
            )

            self.assertTrue(result.healthy)
            self.assertEqual([RecoveryPhase.RECONNECT_ADB], self._phases(result))
            self.assertIn("adb_connect", fixture.emulator.calls)
            self.assertEqual(
                ["reconnect_adb"],
                [attempt.phase for attempt in fixture.attempts.list_for_instance(fixture.instance_id)],
            )

    def test_game_crash_recovers_by_relaunching_game(self) -> None:
        with self._fixture() as fixture:
            fixture.emulator.activity = ""

            result = fixture.watchdog.monitor(
                instance_id=fixture.instance_id,
                instance_index=0,
                instance_name="MEmu0",
            )

            self.assertTrue(result.healthy)
            self.assertIn(RecoveryPhase.RELAUNCH_GAME, self._phases(result))
            self.assertIn("launch_game", fixture.emulator.calls)

    def test_wrong_activity_recovers_by_relaunching_game(self) -> None:
        with self._fixture() as fixture:
            fixture.emulator.activity = "com.android.launcher/.Launcher"

            result = fixture.watchdog.monitor(
                instance_id=fixture.instance_id,
                instance_index=0,
                instance_name="MEmu0",
            )

            self.assertTrue(result.healthy)
            self.assertEqual(WatchdogIssue.NONE, result.observation.issue)
            self.assertIn(RecoveryPhase.RELAUNCH_GAME, self._phases(result))

    def test_blank_screenshot_restarts_emulator_before_recovering(self) -> None:
        with self._fixture() as fixture:
            fixture.emulator.blank_screenshot = True

            result = fixture.watchdog.monitor(
                instance_id=fixture.instance_id,
                instance_index=0,
                instance_name="MEmu0",
            )

            self.assertTrue(result.healthy)
            self.assertIn(RecoveryPhase.RESTART_EMULATOR, self._phases(result))
            self.assertIn("start", fixture.emulator.calls)

    def test_anr_like_popup_sends_back_and_recovers(self) -> None:
        with self._fixture(scene_key="android.anr") as fixture:
            def scene_resolver(_path: Path) -> str:
                if "keyevent:4" in fixture.emulator.calls:
                    return "city"
                return "android.anr"

            fixture.watchdog.scene_resolver = scene_resolver

            result = fixture.watchdog.monitor(
                instance_id=fixture.instance_id,
                instance_index=0,
                instance_name="MEmu0",
            )

            self.assertTrue(result.healthy)
            self.assertIn(RecoveryPhase.SEND_BACK, self._phases(result))
            self.assertIn("keyevent:4", fixture.emulator.calls)

    def test_repeated_same_screen_opens_incident_and_circuit_breaker(self) -> None:
        with self._fixture(same_screen_timeout_seconds=1) as fixture:
            first = fixture.watchdog.monitor(
                instance_id=fixture.instance_id,
                instance_index=0,
                instance_name="MEmu0",
            )
            fixture.now = 2.0
            second = fixture.watchdog.monitor(
                instance_id=fixture.instance_id,
                instance_index=0,
                instance_name="MEmu0",
            )

            self.assertTrue(first.healthy)
            self.assertFalse(second.healthy)
            self.assertTrue(second.circuit_opened)
            self.assertIn(RecoveryPhase.OPEN_INCIDENT, self._phases(second))
            self.assertTrue(fixture.breakers.is_open(fixture.instance_id))
            self.assertEqual(1, len(fixture.incidents.list_open()))

    def _fixture(
        self,
        *,
        scene_key: str = "city",
        same_screen_timeout_seconds: float = 120.0,
    ) -> "_Fixture":
        return _Fixture(
            scene_key=scene_key,
            same_screen_timeout_seconds=same_screen_timeout_seconds,
        )

    @staticmethod
    def _phases(result: object) -> list[RecoveryPhase]:
        return [record.phase for record in result.recovery_records]


class _Fixture:
    def __init__(self, *, scene_key: str, same_screen_timeout_seconds: float) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = Database(self.root / "app.sqlite3")
        self.db.initialize()
        self.instances = InstanceRepository(self.db)
        self.instance_id = self.instances.save(
            Instance(name="MEmu0", instance_index=0, instance_name="MEmu0")
        )
        self.emulator = FakeEmulator(self.root / "screens")
        self.attempts = RecoveryAttemptRepository(self.db)
        self.incidents = IncidentRepository(self.db)
        self.breakers = InstanceCircuitBreakerRepository(self.db)
        self.now = 0.0
        config = WatchdogConfig(
            screenshot_dir=self.root / "screens",
            same_screen_timeout_seconds=same_screen_timeout_seconds,
            same_screen_max_observations=100,
        )
        self.watchdog = RecoveryWatchdog(
            emulator=self.emulator,
            config=config,
            scene_resolver=lambda _path: scene_key,
            attempts=self.attempts,
            incidents=self.incidents,
            circuit_breakers=self.breakers,
            clock=lambda: self.now,
        )

    def __enter__(self) -> "_Fixture":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.db.close()
        self.temp.cleanup()


if __name__ == "__main__":
    unittest.main()
