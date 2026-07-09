from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests.db_helpers import SRC_ROOT  # noqa: F401

from rok_assistant.db.database import Database
from rok_assistant.db.models import Character, Instance, Job
from rok_assistant.db.repositories import (
    CharacterRepository,
    IncidentRepository,
    InstanceCircuitBreakerRepository,
    InstanceRepository,
    JobRepository,
    JobRunRepository,
    StepRunRepository,
)
from rok_assistant.tasks.game_reboot_workflow import (
    GAME_REBOOT_STATES,
    GAME_REBOOT_TEMPLATE_KEYS,
    GameReadinessResult,
    GameRebootActionResult,
    GameRebootConfig,
    GameRebootPolicy,
    GameRebootRequest,
    GameRebootWorkflow,
)
from rok_assistant.workflow_engine import WorkflowOutcome


class FakeGameRebootDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.preflight = GameRebootActionResult(True, data={"adb_connected": True})
        self.force_stop = GameRebootActionResult(True)
        self.stopped = GameRebootActionResult(True, data={"activity": ""})
        self.launch = GameRebootActionResult(True)
        self.readiness = GameReadinessResult(
            True,
            home_scene_verified=True,
            scene_key="city",
            activity="com.lilithgame.roc.gp/.UnityPlayerActivity",
            screenshot_path="runtime/screens/ready.png",
        )
        self.popup = GameRebootActionResult(True, data={"handled": False})
        self.home = GameReadinessResult(
            True,
            home_scene_verified=True,
            scene_key="city",
            activity="com.lilithgame.roc.gp/.UnityPlayerActivity",
            screenshot_path="runtime/screens/home.png",
        )
        self.emulator_reboot = GameReadinessResult(
            True,
            home_scene_verified=True,
            scene_key="city",
            activity="com.lilithgame.roc.gp/.UnityPlayerActivity",
            screenshot_path="runtime/screens/reboot-ready.png",
        )

    def validate_session(self, _request, _character, _policy):
        self.calls.append("validate_session")
        return self.preflight

    def force_stop_game(self, _request, _policy):
        self.calls.append("force_stop_game")
        return self.force_stop

    def verify_game_stopped(self, _request, _policy):
        self.calls.append("verify_game_stopped")
        return self.stopped

    def launch_game(self, _request, _policy):
        self.calls.append("launch_game")
        return self.launch

    def wait_for_readiness(self, _request, _character, _policy):
        self.calls.append("wait_for_readiness")
        return self.readiness

    def handle_safe_popups(self, _request, _character, _policy, _readiness):
        self.calls.append("handle_safe_popups")
        return self.popup

    def verify_home_scene(self, _request, _character, _policy):
        self.calls.append("verify_home_scene")
        return self.home

    def reboot_emulator(self, _request, _character, _policy):
        self.calls.append("reboot_emulator")
        return self.emulator_reboot


class FakeWatchdog:
    def __init__(self, *, healthy: bool = True) -> None:
        self.healthy = healthy
        self.calls: list[int | None] = []

    def monitor(
        self,
        *,
        instance_id: int,
        instance_index: int,
        instance_name: str,
        job_run_id: int | None = None,
    ) -> object:
        del instance_id, instance_index, instance_name
        self.calls.append(job_run_id)
        return SimpleNamespace(
            healthy=self.healthy,
            recovery_attempted=True,
            circuit_opened=not self.healthy,
            observation=SimpleNamespace(
                message="" if self.healthy else "still unhealthy",
                screenshot_path="" if self.healthy else "runtime/screens/unhealthy.png",
            ),
        )


class GameRebootWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "game-reboot.sqlite3")
        self.db.initialize()
        self.instances = InstanceRepository(self.db)
        self.characters = CharacterRepository(self.db)
        self.jobs = JobRepository(self.db)
        self.job_runs = JobRunRepository(self.db)
        self.step_runs = StepRunRepository(self.db)
        self.incidents = IncidentRepository(self.db)
        self.breakers = InstanceCircuitBreakerRepository(self.db)
        self.instance_id = self.instances.save(
            Instance(name="MEmu 1", instance_index=0, instance_name="MEmu 1")
        )
        self.character_id = self.characters.save(
            Character(
                id=None,
                name="Farm01",
                instance_id=self.instance_id,
            )
        )
        self.driver = FakeGameRebootDriver()
        self.watchdog = FakeWatchdog()

    def tearDown(self) -> None:
        self.db.close()
        self.temp_dir.cleanup()

    def _job(self, key: str) -> int:
        return self.jobs.save(
            Job(
                idempotency_key=key,
                job_type="workflow",
                scheduled_for="2026-07-09T00:00:00",
            )
        )

    def _workflow(self) -> GameRebootWorkflow:
        return GameRebootWorkflow(
            characters=self.characters,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            circuit_breakers=self.breakers,
            config=GameRebootConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "game-reboot-run",
        policy: GameRebootPolicy | None = None,
    ) -> GameRebootRequest:
        return GameRebootRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or GameRebootPolicy(),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_ops_states_and_template_keys(self) -> None:
        self.assertEqual(GAME_REBOOT_STATES, self._workflow().workflow_states)
        self.assertIn("city.home", GAME_REBOOT_TEMPLATE_KEYS)

    def test_normal_restart_persists_phase_attempts_and_readiness(self) -> None:
        job_id = self._job("game-reboot-normal")

        result = self._workflow().execute(
            self._request(job_id=job_id, run_key="game-reboot-normal")
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(
            [
                "validate_session",
                "force_stop_game",
                "verify_game_stopped",
                "launch_game",
                "wait_for_readiness",
                "verify_home_scene",
            ],
            self.driver.calls,
        )
        self.assertFalse(result.result["normal_restart_failed"])
        self.assertEqual("city", result.result["readiness_result"]["scene_key"])
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual("game-reboot", payload["workflow_key"])
        self.assertEqual("city", payload["result"]["readiness_result"]["scene_key"])
        self.assertGreaterEqual(len(payload["result"]["phase_attempts"]), 5)
        self.assertEqual("completed", run.status)  # type: ignore[union-attr]

    def test_adb_offline_preflight_uses_watchdog_then_restarts(self) -> None:
        self.driver.preflight = GameRebootActionResult(False, "ADB is offline.", retryable=True)

        result = self._workflow().execute(self._request(run_key="game-reboot-adb-offline"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual([None], self.watchdog.calls)
        self.assertIn("force_stop_game", self.driver.calls)
        self.assertTrue(result.result["recovery_outcome"]["healthy"])

    def test_package_launch_failure_escalates_to_emulator_reboot(self) -> None:
        self.driver.launch = GameRebootActionResult(
            False,
            "Launch game activity failed.",
            retryable=False,
        )

        result = self._workflow().execute(self._request(run_key="game-reboot-launch-failed"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertTrue(result.result["normal_restart_failed"])
        self.assertEqual("launch_game", result.result["phase_attempts"][3]["phase"])
        self.assertIn("reboot_emulator", self.driver.calls)
        self.assertEqual("city", result.result["emulator_reboot_result"]["scene_key"])

    def test_blank_first_screenshot_escalates_to_emulator_reboot(self) -> None:
        self.driver.readiness = GameReadinessResult(
            False,
            message="First screenshot is blank.",
            retryable=False,
            screenshot_path="runtime/screens/blank.png",
        )

        result = self._workflow().execute(self._request(run_key="game-reboot-blank"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertTrue(result.result["normal_restart_failed"])
        self.assertIn("reboot_emulator", self.driver.calls)
        self.assertEqual("runtime/screens/reboot-ready.png", result.result["emulator_reboot_result"]["screenshot_path"])

    def test_wrong_activity_escalates_to_emulator_reboot(self) -> None:
        self.driver.home = GameReadinessResult(
            False,
            message="Foreground activity is not the configured game: com.android.launcher/.Launcher",
            retryable=False,
            activity="com.android.launcher/.Launcher",
        )

        result = self._workflow().execute(self._request(run_key="game-reboot-wrong-activity"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertTrue(result.result["normal_restart_failed"])
        self.assertIn("reboot_emulator", self.driver.calls)

    def test_emulator_reboot_failure_opens_incident_and_circuit_breaker(self) -> None:
        self.driver.launch = GameRebootActionResult(False, "Launch game activity failed.", retryable=False)
        self.driver.emulator_reboot = GameReadinessResult(
            False,
            message="Game readiness was not verified after emulator reboot.",
            retryable=False,
            screenshot_path="runtime/screens/reboot-failed.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("game-reboot-failed"), run_key="game-reboot-failed")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("emulator_reboot", result.result["terminal_state"])
        self.assertTrue(result.result["incident_opened"])
        self.assertTrue(result.result["circuit_opened"])
        self.assertTrue(self.breakers.is_open(self.instance_id))
        self.assertEqual(1, len(self.incidents.list_open()))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("failed", run.status)  # type: ignore[union-attr]
        self.assertEqual("runtime/screens/reboot-failed.png", run.screenshot_path)  # type: ignore[union-attr]

    def test_observable_character_mismatch_blocks_after_reboot(self) -> None:
        self.driver.home = GameReadinessResult(
            True,
            home_scene_verified=True,
            scene_key="city",
            observed_character_id=999,
            character_verified=False,
            screenshot_path="runtime/screens/wrong-character.png",
        )
        self.driver.emulator_reboot = GameReadinessResult(
            False,
            message="Wrong character remained visible after emulator reboot.",
            retryable=False,
            screenshot_path="runtime/screens/wrong-character-reboot.png",
        )

        result = self._workflow().execute(self._request(run_key="game-reboot-wrong-character"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("emulator_reboot", result.result["terminal_state"])
        self.assertTrue(self.breakers.is_open(self.instance_id))


if __name__ == "__main__":
    unittest.main()
