from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import cv2

from rok_assistant.db.models import Incident, RecoveryAttempt, utc_now_iso
from rok_assistant.db.repositories import (
    IncidentRepository,
    InstanceCircuitBreakerRepository,
    RecoveryAttemptRepository,
)
from rok_assistant.emulator.provider import EmulatorCommandResult, EmulatorHealth
from rok_assistant.paths import SCREENSHOT_DIR


class WatchdogIssue(str, Enum):
    NONE = "none"
    EMULATOR_STOPPED = "emulator_stopped"
    ADB_OFFLINE = "adb_offline"
    ANDROID_NOT_BOOTED = "android_not_booted"
    GAME_NOT_RUNNING = "game_not_running"
    WRONG_ACTIVITY = "wrong_activity"
    FIRST_SCREENSHOT_FAILED = "first_screenshot_failed"
    BLANK_SCREENSHOT = "blank_screenshot"
    UNKNOWN_SCENE = "unknown_scene"
    BLOCKING_POPUP = "blocking_popup"
    SAME_SCREEN_TIMEOUT = "same_screen_timeout"
    CIRCUIT_OPEN = "circuit_open"


class RecoveryPhase(str, Enum):
    RECONNECT_ADB = "reconnect_adb"
    SEND_BACK = "send_back"
    NORMALIZE_HOME = "normalize_home"
    RELAUNCH_GAME = "relaunch_game"
    RESTART_EMULATOR = "restart_emulator"
    OPEN_INCIDENT = "open_incident"


@dataclass(frozen=True)
class WatchdogConfig:
    game_package: str = "com.lilithgame.roc.gp"
    game_activity: str = "com.lilithgame.roc.gp/.UnityPlayerActivity"
    known_scene_keys: tuple[str, ...] = ("city", "home", "map")
    blocking_scene_keys: tuple[str, ...] = ("android.anr", "android.crash")
    screenshot_dir: Path = SCREENSHOT_DIR
    blank_stddev_threshold: float = 1.0
    blank_mean_minimum: float = 1.0
    same_screen_timeout_seconds: float = 120.0
    same_screen_max_observations: int = 3
    phase_timeouts: dict[RecoveryPhase, float] = field(
        default_factory=lambda: {
            RecoveryPhase.RECONNECT_ADB: 15.0,
            RecoveryPhase.SEND_BACK: 5.0,
            RecoveryPhase.NORMALIZE_HOME: 5.0,
            RecoveryPhase.RELAUNCH_GAME: 30.0,
            RecoveryPhase.RESTART_EMULATOR: 120.0,
            RecoveryPhase.OPEN_INCIDENT: 5.0,
        }
    )

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "WatchdogConfig":
        phase_values = values.get("phase_timeouts", {})
        phase_timeouts = dict(cls().phase_timeouts)
        if isinstance(phase_values, dict):
            for phase in RecoveryPhase:
                raw_value = phase_values.get(phase.value)
                if isinstance(raw_value, int | float) and not isinstance(raw_value, bool):
                    phase_timeouts[phase] = max(0.1, float(raw_value))
        return cls(
            game_package=str(values.get("game_package") or cls().game_package),
            game_activity=str(values.get("game_activity") or cls().game_activity),
            same_screen_timeout_seconds=_positive_float(
                values.get("same_screen_timeout_seconds"),
                cls().same_screen_timeout_seconds,
            ),
            same_screen_max_observations=max(
                2,
                _positive_int(
                    values.get("same_screen_max_observations"),
                    cls().same_screen_max_observations,
                ),
            ),
            phase_timeouts=phase_timeouts,
        )


@dataclass(frozen=True)
class WatchdogObservation:
    healthy: bool
    issue: WatchdogIssue = WatchdogIssue.NONE
    message: str = ""
    screenshot_path: str = ""
    scene_key: str = ""
    activity: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryPhaseRecord:
    phase: RecoveryPhase
    success: bool
    message: str


@dataclass(frozen=True)
class WatchdogResult:
    healthy: bool
    observation: WatchdogObservation
    recovery_attempted: bool = False
    recovery_records: tuple[RecoveryPhaseRecord, ...] = ()
    circuit_opened: bool = False


class WatchdogEmulator(Protocol):
    def health_check(self, index: int) -> EmulatorCommandResult[EmulatorHealth]:
        ...

    def adb_connect(self, index: int) -> EmulatorCommandResult[None]:
        ...

    def keyevent(self, index: int, code: int) -> EmulatorCommandResult[None]:
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

    def stop(self, index: int) -> EmulatorCommandResult[None]:
        ...

    def start(self, index: int) -> EmulatorCommandResult[None]:
        ...

    def screenshot(
        self,
        index: int,
        instance_name: str,
        output_dir: Path = SCREENSHOT_DIR,
    ) -> EmulatorCommandResult[Path]:
        ...

    def run_adb(self, index: int, args: list[str]) -> EmulatorCommandResult[None]:
        ...


SceneResolver = Callable[[Path], str | None]
Clock = Callable[[], float]


class RecoveryWatchdog:
    def __init__(
        self,
        *,
        emulator: WatchdogEmulator,
        config: WatchdogConfig | None = None,
        scene_resolver: SceneResolver | None = None,
        attempts: RecoveryAttemptRepository | None = None,
        incidents: IncidentRepository | None = None,
        circuit_breakers: InstanceCircuitBreakerRepository | None = None,
        clock: Clock | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.emulator = emulator
        self.config = config or WatchdogConfig()
        self.scene_resolver = scene_resolver
        self.attempts = attempts
        self.incidents = incidents
        self.circuit_breakers = circuit_breakers
        self.clock = clock or time.monotonic
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._screens: dict[int, tuple[str, float, int]] = {}

    def monitor(
        self,
        *,
        instance_id: int,
        instance_index: int,
        instance_name: str,
        job_run_id: int | None = None,
    ) -> WatchdogResult:
        if self._is_circuit_open(instance_id):
            observation = WatchdogObservation(
                healthy=False,
                issue=WatchdogIssue.CIRCUIT_OPEN,
                message="Instance circuit breaker is open.",
            )
            return WatchdogResult(healthy=False, observation=observation, circuit_opened=True)

        observation = self._observe(instance_id, instance_index, instance_name)
        if observation.healthy:
            return WatchdogResult(healthy=True, observation=observation)

        records = self._recover(
            instance_id=instance_id,
            instance_index=instance_index,
            instance_name=instance_name,
            job_run_id=job_run_id,
            initial_observation=observation,
        )
        final_observation = self._observe(instance_id, instance_index, instance_name)
        circuit_opened = self._is_circuit_open(instance_id)
        return WatchdogResult(
            healthy=final_observation.healthy,
            observation=final_observation,
            recovery_attempted=True,
            recovery_records=tuple(records),
            circuit_opened=circuit_opened,
        )

    def _observe(
        self,
        instance_id: int,
        instance_index: int,
        instance_name: str,
    ) -> WatchdogObservation:
        health_result = self.emulator.health_check(instance_index)
        health = health_result.payload or EmulatorHealth(
            index=instance_index,
            running=False,
            adb_connected=False,
        )
        if not health.running:
            return self._unhealthy(WatchdogIssue.EMULATOR_STOPPED, "Emulator is not running.")
        if not health.adb_connected:
            return self._unhealthy(WatchdogIssue.ADB_OFFLINE, "ADB is offline.")
        if not self._android_booted(instance_index):
            return self._unhealthy(WatchdogIssue.ANDROID_NOT_BOOTED, "Android boot is not ready.")

        activity = self._current_activity(instance_index)
        if not activity:
            return self._unhealthy(WatchdogIssue.GAME_NOT_RUNNING, "No foreground activity detected.")
        if self.config.game_package not in activity:
            return self._unhealthy(
                WatchdogIssue.WRONG_ACTIVITY,
                f"Foreground activity is not the configured game: {activity}",
                activity=activity,
            )
        if self.config.game_activity and self.config.game_activity not in activity:
            return self._unhealthy(
                WatchdogIssue.WRONG_ACTIVITY,
                f"Foreground activity does not match configured activity: {activity}",
                activity=activity,
            )

        screenshot = self.emulator.screenshot(
            instance_index,
            instance_name,
            self.config.screenshot_dir,
        )
        if not screenshot.succeeded or screenshot.payload is None:
            return self._unhealthy(
                WatchdogIssue.FIRST_SCREENSHOT_FAILED,
                "First screenshot capture failed.",
                activity=activity,
            )
        screenshot_path = screenshot.payload
        if self._is_blank_screenshot(screenshot_path):
            return self._unhealthy(
                WatchdogIssue.BLANK_SCREENSHOT,
                "Screenshot is blank or nearly blank.",
                screenshot_path=str(screenshot_path),
                activity=activity,
            )

        scene_key = self._resolve_scene(screenshot_path)
        if scene_key in self.config.blocking_scene_keys:
            return self._unhealthy(
                WatchdogIssue.BLOCKING_POPUP,
                f"Blocking scene detected: {scene_key}",
                screenshot_path=str(screenshot_path),
                scene_key=scene_key or "",
                activity=activity,
            )
        if scene_key not in self.config.known_scene_keys:
            return self._unhealthy(
                WatchdogIssue.UNKNOWN_SCENE,
                "Screenshot did not classify as a known scene.",
                screenshot_path=str(screenshot_path),
                scene_key=scene_key or "",
                activity=activity,
            )
        same_screen = self._same_screen_timed_out(instance_id, screenshot_path)
        if same_screen:
            return self._unhealthy(
                WatchdogIssue.SAME_SCREEN_TIMEOUT,
                "The screen has not changed within the configured timeout.",
                screenshot_path=str(screenshot_path),
                scene_key=scene_key or "",
                activity=activity,
            )
        return WatchdogObservation(
            healthy=True,
            screenshot_path=str(screenshot_path),
            scene_key=scene_key or "",
            activity=activity,
        )

    def _recover(
        self,
        *,
        instance_id: int,
        instance_index: int,
        instance_name: str,
        job_run_id: int | None,
        initial_observation: WatchdogObservation,
    ) -> list[RecoveryPhaseRecord]:
        records: list[RecoveryPhaseRecord] = []
        for phase in (
            RecoveryPhase.RECONNECT_ADB,
            RecoveryPhase.SEND_BACK,
            RecoveryPhase.NORMALIZE_HOME,
            RecoveryPhase.RELAUNCH_GAME,
            RecoveryPhase.RESTART_EMULATOR,
        ):
            success, message = self._run_phase(phase, instance_index)
            records.append(RecoveryPhaseRecord(phase, success, message))
            self._persist_attempt(
                instance_id=instance_id,
                job_run_id=job_run_id,
                phase=phase,
                success=success,
                reason=message,
                screenshot_path=initial_observation.screenshot_path,
                metadata={
                    "issue": initial_observation.issue.value,
                    "timeout_seconds": self.config.phase_timeouts.get(phase),
                },
            )
            if success and self._observe(instance_id, instance_index, instance_name).healthy:
                return records

        incident_id = self._open_incident(
            instance_id=instance_id,
            job_run_id=job_run_id,
            observation=initial_observation,
        )
        if self.circuit_breakers is not None:
            self.circuit_breakers.open(
                instance_id=instance_id,
                reason=initial_observation.message,
                incident_id=incident_id or None,
                metadata_json=json.dumps({"issue": initial_observation.issue.value}),
            )
        records.append(
            RecoveryPhaseRecord(
                RecoveryPhase.OPEN_INCIDENT,
                True,
                "Incident opened and instance circuit breaker is open.",
            )
        )
        self._persist_attempt(
            instance_id=instance_id,
            job_run_id=job_run_id,
            phase=RecoveryPhase.OPEN_INCIDENT,
            success=True,
            reason=initial_observation.message,
            screenshot_path=initial_observation.screenshot_path,
            metadata={"issue": initial_observation.issue.value, "incident_id": incident_id},
        )
        return records

    def _run_phase(self, phase: RecoveryPhase, instance_index: int) -> tuple[bool, str]:
        if phase == RecoveryPhase.RECONNECT_ADB:
            return self._result_message(self.emulator.adb_connect(instance_index))
        if phase == RecoveryPhase.SEND_BACK:
            return self._result_message(self.emulator.keyevent(instance_index, 4))
        if phase == RecoveryPhase.NORMALIZE_HOME:
            return self._result_message(self.emulator.keyevent(instance_index, 3))
        if phase == RecoveryPhase.RELAUNCH_GAME:
            self.emulator.force_stop_game_package(instance_index, self.config.game_package)
            return self._result_message(
                self.emulator.launch_game_activity(instance_index, self.config.game_activity)
            )
        if phase == RecoveryPhase.RESTART_EMULATOR:
            stop_result = self.emulator.stop(instance_index)
            start_result = self.emulator.start(instance_index)
            if not stop_result.succeeded:
                return self._result_message(stop_result)
            return self._result_message(start_result)
        return False, f"Unsupported recovery phase: {phase.value}"

    def _persist_attempt(
        self,
        *,
        instance_id: int,
        job_run_id: int | None,
        phase: RecoveryPhase,
        success: bool,
        reason: str,
        screenshot_path: str,
        metadata: dict[str, object],
    ) -> None:
        if self.attempts is None:
            return
        attempt = RecoveryAttempt(
            attempt_key=f"recovery:{instance_id}:{phase.value}:{uuid4().hex}",
            instance_id=instance_id,
            job_run_id=job_run_id,
            phase=phase.value,
            state="succeeded" if success else "failed",
            started_at=utc_now_iso(),
            finished_at=utc_now_iso(),
            success=success,
            reason=reason,
            screenshot_path=screenshot_path,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        self.attempts.save(attempt)

    def _open_incident(
        self,
        *,
        instance_id: int,
        job_run_id: int | None,
        observation: WatchdogObservation,
    ) -> int:
        if self.incidents is None:
            return 0
        return self.incidents.save(
            Incident(
                incident_key=f"recovery:{instance_id}:{uuid4().hex}",
                severity="critical",
                status="open",
                title=f"Recovery exhausted for instance {instance_id}",
                details=observation.message,
                job_run_id=job_run_id,
                screenshot_path=observation.screenshot_path,
            )
        )

    def _android_booted(self, instance_index: int) -> bool:
        result = self.emulator.run_adb(
            instance_index,
            ["shell", "getprop", "sys.boot_completed"],
        )
        return result.succeeded and result.stdout.strip() == "1"

    def _current_activity(self, instance_index: int) -> str:
        result = self.emulator.run_adb(
            instance_index,
            ["shell", "dumpsys", "window", "windows"],
        )
        if not result.succeeded:
            return ""
        return _extract_activity(result.stdout)

    def _resolve_scene(self, screenshot_path: Path) -> str | None:
        if self.scene_resolver is None:
            return None
        return self.scene_resolver(screenshot_path)

    def _same_screen_timed_out(self, instance_id: int, screenshot_path: Path) -> bool:
        digest = _file_hash(screenshot_path)
        now = self.clock()
        previous = self._screens.get(instance_id)
        if previous is None or previous[0] != digest:
            self._screens[instance_id] = (digest, now, 1)
            return False
        first_seen, count = previous[1], previous[2] + 1
        self._screens[instance_id] = (digest, first_seen, count)
        elapsed = now - first_seen
        return (
            elapsed >= self.config.same_screen_timeout_seconds
            or count >= max(2, self.config.same_screen_max_observations)
        )

    def _is_circuit_open(self, instance_id: int) -> bool:
        return self.circuit_breakers is not None and self.circuit_breakers.is_open(instance_id)

    def _is_blank_screenshot(self, path: Path) -> bool:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            return True
        mean, stddev = cv2.meanStdDev(image)
        return (
            float(stddev[0][0]) <= self.config.blank_stddev_threshold
            and float(mean[0][0]) <= self.config.blank_mean_minimum
        )

    @staticmethod
    def _result_message(result: EmulatorCommandResult[object]) -> tuple[bool, str]:
        if result.succeeded:
            return True, "ok"
        return False, result.error_message or result.stderr.strip() or result.error_category.value

    @staticmethod
    def _unhealthy(
        issue: WatchdogIssue,
        message: str,
        *,
        screenshot_path: str = "",
        scene_key: str = "",
        activity: str = "",
        metadata: dict[str, object] | None = None,
    ) -> WatchdogObservation:
        return WatchdogObservation(
            healthy=False,
            issue=issue,
            message=message,
            screenshot_path=screenshot_path,
            scene_key=scene_key,
            activity=activity,
            metadata=metadata or {},
        )


def _extract_activity(output: str) -> str:
    for token in ("mCurrentFocus=", "mFocusedApp=", "topResumedActivity="):
        index = output.find(token)
        if index == -1:
            continue
        line = output[index:].splitlines()[0]
        for part in line.replace("}", " ").replace("{", " ").split():
            if "/" in part:
                return part.strip()
    return ""


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_float(value: object, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    parsed = float(value)
    return parsed if parsed > 0 else default


def _positive_int(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value if value > 0 else default
