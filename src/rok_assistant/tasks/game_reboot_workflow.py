from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

import cv2

from rok_assistant.db.models import Character, Incident, JobRun, utc_now_iso
from rok_assistant.emulator.provider import (
    EmulatorCommandResult,
    EmulatorHealth,
    command_failure_message,
)
from rok_assistant.tasks.resource_search_workflow import ResourceGatheringActionResult
from rok_assistant.workflow_context import (
    CancellationToken,
    StepBudget,
    WorkflowDeadline,
    WorkflowExecutionContext,
)
from rok_assistant.workflow_engine import (
    ActionRegistry,
    WorkflowEngine,
    WorkflowExecutionResult,
    WorkflowOutcome,
    WorkflowRunRepositoryRecorder,
    WorkflowStepResult,
    WorkflowStepSpec,
)


GAME_REBOOT_WORKFLOW_KEY = "game-reboot"
GAME_REBOOT_TEMPLATE_KEYS = (
    "city.home",
    "map.home",
    "startup.popup.safe_close",
)
GAME_REBOOT_STATES = (
    "validate_input",
    "load_character",
    "ensure_account",
    "ensure_character",
    "preflight_session",
    "force_stop_game",
    "verify_game_stopped",
    "launch_game",
    "wait_for_readiness",
    "handle_safe_popups",
    "verify_home_scene",
    "emulator_reboot",
    "complete",
    "failed",
    "cancelled",
)


@dataclass(frozen=True)
class GameRebootPolicy:
    game_package: str = "com.lilithgame.roc.gp"
    game_activity: str = "com.lilithgame.roc.gp/.UnityPlayerActivity"
    known_home_scene_keys: tuple[str, ...] = ("city", "home", "map")
    safe_popup_scene_keys: tuple[str, ...] = ("startup.popup.safe_close", "android.anr")
    verify_account_character_when_observable: bool = True
    allow_emulator_reboot: bool = True

    def normalized(self) -> GameRebootPolicy:
        package = self.game_package.strip()
        activity = self.game_activity.strip()
        if not package:
            raise ValueError("game_package must be configured.")
        if not activity or "/" not in activity:
            raise ValueError("game_activity must be an Android component string.")
        home_keys = _normalized_keys(self.known_home_scene_keys)
        if not home_keys:
            raise ValueError("At least one known home scene key must be configured.")
        return GameRebootPolicy(
            game_package=package,
            game_activity=activity,
            known_home_scene_keys=home_keys,
            safe_popup_scene_keys=_normalized_keys(self.safe_popup_scene_keys),
            verify_account_character_when_observable=bool(
                self.verify_account_character_when_observable
            ),
            allow_emulator_reboot=bool(self.allow_emulator_reboot),
        )

    def to_json(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "game_package": normalized.game_package,
            "game_activity": normalized.game_activity,
            "known_home_scene_keys": list(normalized.known_home_scene_keys),
            "safe_popup_scene_keys": list(normalized.safe_popup_scene_keys),
            "verify_account_character_when_observable": normalized.verify_account_character_when_observable,
            "allow_emulator_reboot": normalized.allow_emulator_reboot,
        }


@dataclass(frozen=True)
class GameRebootRequest:
    instance_id: int
    instance_index: int
    instance_name: str
    character_id: int
    target_account_id: int | None = None
    session_key: str = ""
    policy: GameRebootPolicy = field(default_factory=GameRebootPolicy)
    job_id: int | None = None
    run_key: str = ""
    run_attempt: int = 1
    actor: str = "system"


@dataclass(frozen=True)
class GameRebootConfig:
    workflow_timeout_seconds: float = 240.0
    step_timeout_seconds: float = 30.0
    precondition_retry_limit: int = 1
    restart_retry_limit: int = 1
    readiness_retry_limit: int = 2
    popup_retry_limit: int = 1
    emulator_reboot_retry_limit: int = 0
    retry_delay_seconds: float = 0.5


@dataclass(frozen=True)
class GameRebootActionResult:
    success: bool
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "message": self.message,
            "retryable": self.retryable,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


@dataclass(frozen=True)
class GameReadinessResult:
    ready: bool
    home_scene_verified: bool = False
    scene_key: str = ""
    activity: str = ""
    account_verified: bool | None = None
    character_verified: bool | None = None
    observed_account_id: int | None = None
    observed_character_id: int | None = None
    message: str = ""
    retryable: bool = True
    screenshot_path: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "home_scene_verified": self.home_scene_verified,
            "scene_key": self.scene_key,
            "activity": self.activity,
            "account_verified": self.account_verified,
            "character_verified": self.character_verified,
            "observed_account_id": self.observed_account_id,
            "observed_character_id": self.observed_character_id,
            "message": self.message,
            "retryable": self.retryable,
            "screenshot_path": self.screenshot_path,
            **self.data,
        }


class CharacterRepository(Protocol):
    def get(self, character_id: int) -> Character | None:
        ...


class JobRunRepository(Protocol):
    def get(self, run_id: int) -> JobRun | None:
        ...

    def save(self, run: JobRun) -> int:
        ...


class StepRunRepository(Protocol):
    ...


class IncidentRepository(Protocol):
    def save(self, incident: Incident) -> int:
        ...


class InstanceCircuitBreakerRepository(Protocol):
    def open(
        self,
        *,
        instance_id: int,
        reason: str,
        incident_id: int | None = None,
        metadata_json: str = "{}",
        opened_at: str | None = None,
    ) -> int:
        ...


class GameRebootAccountPrecondition(Protocol):
    def ensure_account(
        self,
        request: GameRebootRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class GameRebootCharacterPrecondition(Protocol):
    def ensure_character(
        self,
        request: GameRebootRequest,
        character: Character,
    ) -> ResourceGatheringActionResult:
        ...


class RecoveryWatchdog(Protocol):
    def monitor(
        self,
        *,
        instance_id: int,
        instance_index: int,
        instance_name: str,
        job_run_id: int | None = None,
    ) -> object:
        ...


class GameRebootDriver(Protocol):
    def validate_session(
        self,
        request: GameRebootRequest,
        character: Character,
        policy: GameRebootPolicy,
    ) -> GameRebootActionResult:
        ...

    def force_stop_game(
        self,
        request: GameRebootRequest,
        policy: GameRebootPolicy,
    ) -> GameRebootActionResult:
        ...

    def verify_game_stopped(
        self,
        request: GameRebootRequest,
        policy: GameRebootPolicy,
    ) -> GameRebootActionResult:
        ...

    def launch_game(
        self,
        request: GameRebootRequest,
        policy: GameRebootPolicy,
    ) -> GameRebootActionResult:
        ...

    def wait_for_readiness(
        self,
        request: GameRebootRequest,
        character: Character,
        policy: GameRebootPolicy,
    ) -> GameReadinessResult:
        ...

    def handle_safe_popups(
        self,
        request: GameRebootRequest,
        character: Character,
        policy: GameRebootPolicy,
        readiness: GameReadinessResult,
    ) -> GameRebootActionResult:
        ...

    def verify_home_scene(
        self,
        request: GameRebootRequest,
        character: Character,
        policy: GameRebootPolicy,
    ) -> GameReadinessResult:
        ...

    def reboot_emulator(
        self,
        request: GameRebootRequest,
        character: Character,
        policy: GameRebootPolicy,
    ) -> GameReadinessResult:
        ...


class GameRebootEmulator(Protocol):
    def health_check(self, index: int) -> EmulatorCommandResult[EmulatorHealth]:
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

    def screenshot(
        self,
        index: int,
        instance_name: str,
        output_dir: Path,
    ) -> EmulatorCommandResult[Path]:
        ...

    def keyevent(self, index: int, code: int) -> EmulatorCommandResult[None]:
        ...

    def run_adb(self, index: int, args: list[str]) -> EmulatorCommandResult[None]:
        ...


SceneResolver = Callable[[Path], str | None]


class EmulatorGameRebootDriver:
    def __init__(
        self,
        emulator: GameRebootEmulator,
        *,
        screenshot_dir: Path = Path("runtime/screens"),
        scene_resolver: SceneResolver | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        readiness_timeout_seconds: float = 90.0,
        readiness_poll_seconds: float = 2.0,
        blank_stddev_threshold: float = 2.0,
        blank_mean_minimum: float = 5.0,
    ) -> None:
        self.emulator = emulator
        self.screenshot_dir = screenshot_dir
        self.scene_resolver = scene_resolver
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self.readiness_timeout_seconds = readiness_timeout_seconds
        self.readiness_poll_seconds = readiness_poll_seconds
        self.blank_stddev_threshold = blank_stddev_threshold
        self.blank_mean_minimum = blank_mean_minimum

    def validate_session(
        self,
        request: GameRebootRequest,
        character: Character,
        policy: GameRebootPolicy,
    ) -> GameRebootActionResult:
        del policy
        if character.instance_id != request.instance_id:
            return GameRebootActionResult(
                False,
                "Target character is not assigned to the requested emulator instance.",
                retryable=False,
            )
        health = self.emulator.health_check(request.instance_index)
        if not health.succeeded:
            return _command_action("Emulator health check", health, retryable=True)
        payload = health.payload or EmulatorHealth(
            index=request.instance_index,
            running=False,
            adb_connected=False,
        )
        if not payload.running:
            return GameRebootActionResult(False, "Emulator is not running.", retryable=True)
        if not payload.adb_connected:
            return GameRebootActionResult(False, "ADB is offline.", retryable=True)
        return GameRebootActionResult(
            True,
            data={
                "running": payload.running,
                "adb_connected": payload.adb_connected,
                "serial": payload.serial,
            },
        )

    def force_stop_game(
        self,
        request: GameRebootRequest,
        policy: GameRebootPolicy,
    ) -> GameRebootActionResult:
        return _command_action(
            "Force-stop game package",
            self.emulator.force_stop_game_package(
                request.instance_index,
                policy.game_package,
            ),
            retryable=True,
        )

    def verify_game_stopped(
        self,
        request: GameRebootRequest,
        policy: GameRebootPolicy,
    ) -> GameRebootActionResult:
        activity = self._current_activity(request.instance_index)
        if policy.game_package in activity:
            return GameRebootActionResult(
                False,
                f"Game activity is still foreground after force-stop: {activity}",
                retryable=True,
                data={"activity": activity},
            )
        return GameRebootActionResult(True, data={"activity": activity})

    def launch_game(
        self,
        request: GameRebootRequest,
        policy: GameRebootPolicy,
    ) -> GameRebootActionResult:
        return _command_action(
            "Launch game activity",
            self.emulator.launch_game_activity(
                request.instance_index,
                policy.game_activity,
            ),
            retryable=True,
        )

    def wait_for_readiness(
        self,
        request: GameRebootRequest,
        character: Character,
        policy: GameRebootPolicy,
    ) -> GameReadinessResult:
        del character
        return self._wait_until_ready(request, policy)

    def handle_safe_popups(
        self,
        request: GameRebootRequest,
        character: Character,
        policy: GameRebootPolicy,
        readiness: GameReadinessResult,
    ) -> GameRebootActionResult:
        del character
        if readiness.scene_key not in policy.safe_popup_scene_keys:
            return GameRebootActionResult(True, data={"handled": False, "scene_key": readiness.scene_key})
        result = self.emulator.keyevent(request.instance_index, 4)
        return _command_action(
            "Close safe startup popup",
            result,
            retryable=True,
            data={"handled": True, "scene_key": readiness.scene_key},
        )

    def verify_home_scene(
        self,
        request: GameRebootRequest,
        character: Character,
        policy: GameRebootPolicy,
    ) -> GameReadinessResult:
        del character
        return self._observe_ready(request, policy)

    def reboot_emulator(
        self,
        request: GameRebootRequest,
        character: Character,
        policy: GameRebootPolicy,
    ) -> GameReadinessResult:
        reboot = self.emulator.reboot(request.instance_index)
        if not reboot.succeeded:
            return GameReadinessResult(
                False,
                message=command_failure_message("Emulator reboot", reboot),
                retryable=True,
            )
        launch = self.emulator.launch_game_activity(
            request.instance_index,
            policy.game_activity,
        )
        if not launch.succeeded:
            return GameReadinessResult(
                False,
                message=command_failure_message("Launch game after emulator reboot", launch),
                retryable=True,
            )
        return self.wait_for_readiness(request, character, policy)

    def _wait_until_ready(
        self,
        request: GameRebootRequest,
        policy: GameRebootPolicy,
    ) -> GameReadinessResult:
        deadline = self.clock() + self.readiness_timeout_seconds
        last = GameReadinessResult(False, message="Game readiness was not checked.")
        while self.clock() <= deadline:
            last = self._observe_ready(request, policy)
            if last.ready or not last.retryable:
                return last
            self.sleep(self.readiness_poll_seconds)
        return GameReadinessResult(
            False,
            message=last.message or "Game readiness timed out.",
            retryable=True,
            screenshot_path=last.screenshot_path,
            scene_key=last.scene_key,
            activity=last.activity,
        )

    def _observe_ready(
        self,
        request: GameRebootRequest,
        policy: GameRebootPolicy,
    ) -> GameReadinessResult:
        health = self.emulator.health_check(request.instance_index)
        payload = health.payload or EmulatorHealth(
            index=request.instance_index,
            running=False,
            adb_connected=False,
        )
        if not payload.running:
            return GameReadinessResult(False, message="Emulator is not running.", retryable=True)
        if not payload.adb_connected:
            return GameReadinessResult(False, message="ADB is offline.", retryable=True)
        booted = self.emulator.run_adb(
            request.instance_index,
            ["shell", "getprop", "sys.boot_completed"],
        )
        if not booted.succeeded or booted.stdout.strip() != "1":
            return GameReadinessResult(False, message="Android boot is not ready.", retryable=True)
        activity = self._current_activity(request.instance_index)
        if policy.game_package not in activity:
            return GameReadinessResult(
                False,
                message=f"Foreground activity is not the configured game: {activity}",
                retryable=True,
                activity=activity,
            )
        screenshot = self.emulator.screenshot(
            request.instance_index,
            request.instance_name,
            self.screenshot_dir,
        )
        if not screenshot.succeeded or screenshot.payload is None:
            return GameReadinessResult(
                False,
                message=command_failure_message("Readiness screenshot", screenshot),
                retryable=True,
                activity=activity,
            )
        screenshot_path = screenshot.payload
        if self._blank_screenshot(screenshot_path):
            return GameReadinessResult(
                False,
                message="First screenshot is blank.",
                retryable=True,
                screenshot_path=str(screenshot_path),
                activity=activity,
            )
        scene_key = self.scene_resolver(screenshot_path) if self.scene_resolver is not None else ""
        home_verified = scene_key in policy.known_home_scene_keys
        if not home_verified and scene_key in policy.safe_popup_scene_keys:
            return GameReadinessResult(
                False,
                home_scene_verified=False,
                scene_key=scene_key or "",
                activity=activity,
                message=f"Safe startup popup is visible: {scene_key}",
                retryable=True,
                screenshot_path=str(screenshot_path),
            )
        if not home_verified:
            return GameReadinessResult(
                False,
                home_scene_verified=False,
                scene_key=scene_key or "",
                activity=activity,
                message=f"Known home scene was not verified: {scene_key or 'unknown'}",
                retryable=True,
                screenshot_path=str(screenshot_path),
            )
        return GameReadinessResult(
            True,
            home_scene_verified=True,
            scene_key=scene_key or "",
            activity=activity,
            screenshot_path=str(screenshot_path),
        )

    def _current_activity(self, instance_index: int) -> str:
        result = self.emulator.run_adb(
            instance_index,
            ["shell", "dumpsys", "window", "windows"],
        )
        if not result.succeeded:
            return ""
        return _extract_activity(result.stdout)

    def _blank_screenshot(self, path: Path) -> bool:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            return True
        mean, stddev = cv2.meanStdDev(image)
        return (
            float(stddev[0][0]) <= self.blank_stddev_threshold
            and float(mean[0][0]) <= self.blank_mean_minimum
        )


@dataclass
class _GameRebootState:
    request: GameRebootRequest
    character: Character | None = None
    policy: GameRebootPolicy | None = None
    phase_attempts: list[dict[str, object]] = field(default_factory=list)
    readiness_result: GameReadinessResult | None = None
    popup_result: GameRebootActionResult | None = None
    emulator_reboot_result: GameReadinessResult | None = None
    recovery_outcome: dict[str, object] = field(default_factory=dict)
    normal_restart_failed: bool = False
    terminal_outcome: WorkflowOutcome | None = None
    terminal_state: str = ""
    terminal_reason: str = ""
    screenshot_path: str = ""
    incident_opened: bool = False
    circuit_opened: bool = False

    @property
    def stopped(self) -> bool:
        return self.terminal_outcome is not None

    @property
    def failed(self) -> bool:
        return self.terminal_outcome in {
            WorkflowOutcome.BLOCKED,
            WorkflowOutcome.FATAL_FAILURE,
            WorkflowOutcome.TIMEOUT,
            WorkflowOutcome.VALIDATION_FAILURE,
        }

    def stop(
        self,
        step_key: str,
        outcome: WorkflowOutcome,
        reason: str,
        *,
        screenshot_path: str = "",
        data: dict[str, object] | None = None,
    ) -> WorkflowStepResult:
        self.terminal_state = step_key
        self.terminal_reason = reason
        self.terminal_outcome = outcome
        if screenshot_path:
            self.screenshot_path = screenshot_path
        return _step_result(
            step_key,
            WorkflowOutcome.SUCCESS if outcome == WorkflowOutcome.SKIPPED else outcome,
            reason,
            data={
                "terminal_outcome": outcome.value,
                "terminal_state": step_key,
                "terminal_reason": reason,
                **(data or {}),
            },
            screenshot_path=screenshot_path,
        )

    def mark_normal_restart_failed(
        self,
        step_key: str,
        reason: str,
        *,
        screenshot_path: str = "",
        data: dict[str, object] | None = None,
    ) -> WorkflowStepResult:
        self.normal_restart_failed = True
        self.terminal_state = step_key
        self.terminal_reason = reason
        if screenshot_path:
            self.screenshot_path = screenshot_path
        return _step_result(
            step_key,
            WorkflowOutcome.SUCCESS,
            reason,
            data={
                "normal_restart_failed": True,
                "failure_state": step_key,
                "failure_reason": reason,
                **(data or {}),
            },
            screenshot_path=screenshot_path,
        )


class GameRebootWorkflow:
    def __init__(
        self,
        *,
        characters: CharacterRepository,
        driver: GameRebootDriver,
        account_precondition: GameRebootAccountPrecondition | None = None,
        character_precondition: GameRebootCharacterPrecondition | None = None,
        recovery_watchdog: RecoveryWatchdog | None = None,
        job_runs: JobRunRepository | None = None,
        step_runs: StepRunRepository | None = None,
        incidents: IncidentRepository | None = None,
        circuit_breakers: InstanceCircuitBreakerRepository | None = None,
        config: GameRebootConfig | None = None,
    ) -> None:
        self.characters = characters
        self.driver = driver
        self.account_precondition = account_precondition
        self.character_precondition = character_precondition
        self.recovery_watchdog = recovery_watchdog
        self.job_runs = job_runs
        self.step_runs = step_runs
        self.incidents = incidents
        self.circuit_breakers = circuit_breakers
        self.config = config or GameRebootConfig()
        self._states: dict[str, _GameRebootState] = {}

    @property
    def workflow_states(self) -> tuple[str, ...]:
        return GAME_REBOOT_STATES

    def execute(
        self,
        request: GameRebootRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> WorkflowExecutionResult:
        token = uuid4().hex
        state = _GameRebootState(request=request)
        self._states[token] = state
        persistence = None
        if self.job_runs is not None and self.step_runs is not None and request.job_id is not None:
            persistence = WorkflowRunRepositoryRecorder(self.job_runs, self.step_runs)
        try:
            context = WorkflowExecutionContext(
                cancellation_token=cancellation_token or CancellationToken(),
                deadline=WorkflowDeadline.from_timeout(
                    self.config.workflow_timeout_seconds,
                    time.monotonic,
                ),
                budget=StepBudget(max_steps=len(GAME_REBOOT_STATES) + 8),
                persistence=persistence,
                job_id=request.job_id,
                run_key=request.run_key or f"game-reboot:{request.instance_id}:{uuid4().hex}",
                run_attempt=request.run_attempt,
                metadata={"game_reboot_run_id": token},
            )
            result = self._engine().execute(self._definition(), context)
            self._record_engine_failure(result, state)
            if state.failed:
                self._open_terminal_incident(state, result.job_run_id)
            self._augment_result(result, state)
            self._update_persisted_run(result, state)
            return result
        finally:
            self._states.pop(token, None)

    def _engine(self) -> WorkflowEngine:
        registry = ActionRegistry()
        for state in GAME_REBOOT_STATES:
            registry.register(f"game_reboot.{state}", self._handler_for(state))
        registry.freeze()
        return WorkflowEngine(action_registry=registry)

    def _definition(self):
        from rok_assistant.workflow_types import WorkflowDefinitionSpec

        retry_limits = {
            "ensure_account": self.config.precondition_retry_limit,
            "ensure_character": self.config.precondition_retry_limit,
            "preflight_session": self.config.precondition_retry_limit,
            "force_stop_game": self.config.restart_retry_limit,
            "verify_game_stopped": self.config.restart_retry_limit,
            "launch_game": self.config.restart_retry_limit,
            "wait_for_readiness": self.config.readiness_retry_limit,
            "handle_safe_popups": self.config.popup_retry_limit,
            "verify_home_scene": self.config.readiness_retry_limit,
            "emulator_reboot": self.config.emulator_reboot_retry_limit,
        }
        return WorkflowDefinitionSpec(
            workflow_key=GAME_REBOOT_WORKFLOW_KEY,
            name="Reboot Game",
            steps=[
                WorkflowStepSpec(
                    step_key=state,
                    action_type=f"game_reboot.{state}",
                    timeout_seconds=self.config.step_timeout_seconds,
                    retry_limit=retry_limits.get(state, 0),
                    retry_delay_seconds=self.config.retry_delay_seconds,
                )
                for state in GAME_REBOOT_STATES
            ],
        )

    def _handler_for(self, state_name: str):
        def handler(
            context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            state = self._state_from_context(context)
            if state_name == "failed":
                return self._failed(step, state, context)
            if state_name == "cancelled":
                return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
            if state.stopped and state_name not in {"complete"}:
                return _step_result(
                    step.step_key,
                    WorkflowOutcome.SKIPPED,
                    data={"skipped_after_terminal_state": state.terminal_state},
                )
            if state.normal_restart_failed and state_name in {
                "force_stop_game",
                "verify_game_stopped",
                "launch_game",
                "wait_for_readiness",
                "handle_safe_popups",
                "verify_home_scene",
            }:
                return _step_result(
                    step.step_key,
                    WorkflowOutcome.SKIPPED,
                    data={"skipped_after_normal_restart_failure": state.terminal_state},
                )
            method = getattr(self, f"_{state_name}")
            return method(step, state, context)

        return handler

    def _validate_input(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        request = state.request
        if request.instance_id <= 0:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, "instance_id must be positive.")
        if request.instance_index < 0:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, "instance_index must be zero or greater.")
        if request.character_id <= 0:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, "character_id must be positive.")
        try:
            state.policy = request.policy.normalized()
        except ValueError as exc:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, str(exc))
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "policy": state.policy.to_json(),
                "template_keys": list(GAME_REBOOT_TEMPLATE_KEYS),
            },
        )

    def _load_character(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        character = self.characters.get(state.request.character_id)
        if character is None:
            return state.stop(step.step_key, WorkflowOutcome.VALIDATION_FAILURE, "Target character was not found.")
        if not character.enabled:
            return state.stop(step.step_key, WorkflowOutcome.SKIPPED, "Target character is disabled.")
        state.character = character
        return _step_result(
            step.step_key,
            WorkflowOutcome.SUCCESS,
            data={
                "character_id": character.id,
                "character_name": character.name,
                "instance_id": character.instance_id,
                "game_account_id": character.game_account_id,
            },
        )

    def _ensure_account(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.account_precondition is None:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        action = self.account_precondition.ensure_account(state.request, _require_character(state))
        return self._precondition_result(step, state, action, "Account precondition failed.")

    def _ensure_character(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if self.character_precondition is None:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        action = self.character_precondition.ensure_character(state.request, _require_character(state))
        return self._precondition_result(step, state, action, "Character precondition failed.")

    def _preflight_session(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        result = self._timed_action(
            state,
            step.step_key,
            lambda: self.driver.validate_session(
                state.request,
                _require_character(state),
                _require_policy(state),
            ),
        )
        if result.screenshot_path:
            state.screenshot_path = result.screenshot_path
        if result.success:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=result.to_json())
        recovery = self._monitor_recovery(state, _job_run_id(context))
        state.recovery_outcome = recovery
        if recovery.get("healthy"):
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                result.message,
                data={"preflight": result.to_json(), "recovery_outcome": recovery},
                screenshot_path=result.screenshot_path,
            )
        if result.retryable:
            return _step_result(
                step.step_key,
                WorkflowOutcome.RETRYABLE_FAILURE,
                result.message or "Preflight session validation failed.",
                data={"preflight": result.to_json(), "recovery_outcome": recovery},
                screenshot_path=result.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            result.message or "Preflight session validation failed.",
            screenshot_path=result.screenshot_path,
            data={"preflight": result.to_json(), "recovery_outcome": recovery},
        )

    def _force_stop_game(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        action = self._timed_action(
            state,
            step.step_key,
            lambda: self.driver.force_stop_game(state.request, _require_policy(state)),
        )
        return self._normal_restart_action(step, state, action, "Game package could not be force-stopped.")

    def _verify_game_stopped(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        action = self._timed_action(
            state,
            step.step_key,
            lambda: self.driver.verify_game_stopped(state.request, _require_policy(state)),
        )
        return self._normal_restart_action(step, state, action, "Game stopped state could not be verified.")

    def _launch_game(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        action = self._timed_action(
            state,
            step.step_key,
            lambda: self.driver.launch_game(state.request, _require_policy(state)),
        )
        return self._normal_restart_action(step, state, action, "Game activity could not be launched.")

    def _wait_for_readiness(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        readiness = self._timed_action(
            state,
            step.step_key,
            lambda: self.driver.wait_for_readiness(
                state.request,
                _require_character(state),
                _require_policy(state),
            ),
        )
        state.readiness_result = readiness
        return self._readiness_step(step, state, readiness, "Game readiness was not verified.")

    def _handle_safe_popups(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        readiness = state.readiness_result
        if readiness is None or readiness.ready:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        action = self._timed_action(
            state,
            step.step_key,
            lambda: self.driver.handle_safe_popups(
                state.request,
                _require_character(state),
                _require_policy(state),
                readiness,
            ),
        )
        state.popup_result = action
        return self._normal_restart_action(step, state, action, "Safe startup popup could not be handled.")

    def _verify_home_scene(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        readiness = self._timed_action(
            state,
            step.step_key,
            lambda: self.driver.verify_home_scene(
                state.request,
                _require_character(state),
                _require_policy(state),
            ),
        )
        state.readiness_result = readiness
        return self._readiness_step(step, state, readiness, "Expected home scene was not verified.")

    def _emulator_reboot(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        policy = _require_policy(state)
        if not state.normal_restart_failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        if not policy.allow_emulator_reboot:
            return state.stop(
                step.step_key,
                WorkflowOutcome.BLOCKED,
                "Normal game restart failed and emulator reboot is disabled by policy.",
                screenshot_path=state.screenshot_path,
            )
        readiness = self._timed_action(
            state,
            step.step_key,
            lambda: self.driver.reboot_emulator(
                state.request,
                _require_character(state),
                policy,
            ),
        )
        state.emulator_reboot_result = readiness
        if readiness.screenshot_path:
            state.screenshot_path = readiness.screenshot_path
        if readiness.ready and self._session_verified_after_reboot(state, readiness):
            state.terminal_state = ""
            state.terminal_reason = ""
            return _step_result(
                step.step_key,
                WorkflowOutcome.SUCCESS,
                data={"emulator_reboot_result": readiness.to_json()},
                screenshot_path=readiness.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            readiness.message or "Game readiness was not verified after emulator reboot.",
            screenshot_path=readiness.screenshot_path,
            data={"emulator_reboot_result": readiness.to_json()},
        )

    def _complete(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if state.stopped:
            return _step_result(
                step.step_key,
                WorkflowOutcome.SKIPPED,
                data={
                    "terminal_outcome": state.terminal_outcome.value if state.terminal_outcome else "",
                    "terminal_state": state.terminal_state,
                    "terminal_reason": state.terminal_reason,
                },
            )
        return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=self._payload(state))

    def _failed(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        _context: WorkflowExecutionContext,
    ) -> WorkflowStepResult:
        if not state.failed:
            return _step_result(step.step_key, WorkflowOutcome.SKIPPED)
        return _step_result(
            step.step_key,
            WorkflowOutcome.FATAL_FAILURE,
            state.terminal_reason,
            data=self._payload(state),
            screenshot_path=state.screenshot_path,
        )

    def _normal_restart_action(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        action: GameRebootActionResult,
        fallback_message: str,
    ) -> WorkflowStepResult:
        if action.screenshot_path:
            state.screenshot_path = action.screenshot_path
        if action.success:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=action.to_json(), screenshot_path=action.screenshot_path)
        return state.mark_normal_restart_failed(
            step.step_key,
            action.message or fallback_message,
            screenshot_path=action.screenshot_path,
            data=action.to_json(),
        )

    def _readiness_step(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        readiness: GameReadinessResult,
        fallback_message: str,
    ) -> WorkflowStepResult:
        if readiness.screenshot_path:
            state.screenshot_path = readiness.screenshot_path
        if readiness.ready and self._session_verified_after_reboot(state, readiness):
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=readiness.to_json(), screenshot_path=readiness.screenshot_path)
        return state.mark_normal_restart_failed(
            step.step_key,
            readiness.message or fallback_message,
            screenshot_path=readiness.screenshot_path,
            data=readiness.to_json(),
        )

    def _precondition_result(
        self,
        step: WorkflowStepSpec,
        state: _GameRebootState,
        action: ResourceGatheringActionResult,
        fallback: str,
    ) -> WorkflowStepResult:
        if action.screenshot_path:
            state.screenshot_path = action.screenshot_path
        if action.success:
            return _step_result(step.step_key, WorkflowOutcome.SUCCESS, data=action.data, screenshot_path=action.screenshot_path)
        if action.retryable:
            return _step_result(
                step.step_key,
                WorkflowOutcome.RETRYABLE_FAILURE,
                action.message or fallback,
                data=action.data,
                screenshot_path=action.screenshot_path,
            )
        return state.stop(
            step.step_key,
            WorkflowOutcome.BLOCKED,
            action.message or fallback,
            screenshot_path=action.screenshot_path,
            data=action.data,
        )

    def _session_verified_after_reboot(
        self,
        state: _GameRebootState,
        readiness: GameReadinessResult,
    ) -> bool:
        policy = _require_policy(state)
        if not policy.verify_account_character_when_observable:
            return True
        if readiness.account_verified is False or readiness.character_verified is False:
            return False
        request = state.request
        character = _require_character(state)
        if readiness.observed_account_id is not None:
            expected_account = request.target_account_id or character.game_account_id
            if expected_account is not None and readiness.observed_account_id != expected_account:
                return False
        if readiness.observed_character_id is not None and character.id is not None:
            return readiness.observed_character_id == character.id
        return True

    def _timed_action(self, state: _GameRebootState, phase: str, action):
        started = time.monotonic()
        started_at = utc_now_iso()
        result = action()
        duration = time.monotonic() - started
        payload = result.to_json() if hasattr(result, "to_json") else {}
        state.phase_attempts.append(
            {
                "phase": phase,
                "started_at": started_at,
                "duration_seconds": round(duration, 6),
                "success": bool(getattr(result, "success", getattr(result, "ready", False))),
                "message": str(getattr(result, "message", "")),
                "screenshot_path": str(getattr(result, "screenshot_path", "")),
                "result": payload,
            }
        )
        return result

    def _monitor_recovery(
        self,
        state: _GameRebootState,
        job_run_id: int | None,
    ) -> dict[str, object]:
        if self.recovery_watchdog is None:
            return {"attempted": False, "reason": "watchdog_not_configured"}
        result = self.recovery_watchdog.monitor(
            instance_id=state.request.instance_id,
            instance_index=state.request.instance_index,
            instance_name=state.request.instance_name,
            job_run_id=job_run_id,
        )
        observation = getattr(result, "observation", None)
        screenshot_path = str(getattr(observation, "screenshot_path", "") or "")
        if screenshot_path:
            state.screenshot_path = screenshot_path
        return {
            "attempted": bool(getattr(result, "recovery_attempted", False)),
            "healthy": bool(getattr(result, "healthy", False)),
            "circuit_opened": bool(getattr(result, "circuit_opened", False)),
            "message": str(getattr(observation, "message", "") or ""),
            "screenshot_path": screenshot_path,
        }

    def _state_from_context(self, context: WorkflowExecutionContext) -> _GameRebootState:
        token = str(context.metadata.get("game_reboot_run_id") or "")
        try:
            return self._states[token]
        except KeyError as exc:
            raise RuntimeError("Game reboot runtime state is missing.") from exc

    def _record_engine_failure(
        self,
        result: WorkflowExecutionResult,
        state: _GameRebootState,
    ) -> None:
        if not result.outcome.is_failure or state.stopped or result.outcome == WorkflowOutcome.CANCELLED:
            return
        last_step = result.steps[-1] if result.steps else None
        state.terminal_state = last_step.step_key if last_step is not None else ""
        state.terminal_reason = result.message or "Game reboot workflow failed."
        state.terminal_outcome = result.outcome
        state.screenshot_path = last_step.screenshot_path if last_step is not None else state.screenshot_path
        if state.terminal_state in {
            "force_stop_game",
            "verify_game_stopped",
            "launch_game",
            "wait_for_readiness",
            "handle_safe_popups",
            "verify_home_scene",
        }:
            state.normal_restart_failed = True

    def _open_terminal_incident(
        self,
        state: _GameRebootState,
        job_run_id: int | None,
    ) -> None:
        if state.incident_opened:
            return
        incident_id = None
        if self.incidents is not None:
            incident_id = self.incidents.save(
                Incident(
                    incident_key=f"game-reboot:{state.request.instance_id}:{uuid4().hex}",
                    severity="critical",
                    status="open",
                    title="Game reboot workflow failed",
                    details=state.terminal_reason,
                    job_run_id=job_run_id,
                    screenshot_path=state.screenshot_path,
                )
            )
            state.incident_opened = True
        if self.circuit_breakers is not None:
            self.circuit_breakers.open(
                instance_id=state.request.instance_id,
                reason=state.terminal_reason,
                incident_id=incident_id,
                metadata_json=json.dumps(
                    {
                        "workflow_key": GAME_REBOOT_WORKFLOW_KEY,
                        "terminal_state": state.terminal_state,
                    },
                    sort_keys=True,
                ),
            )
            state.circuit_opened = True

    def _payload(self, state: _GameRebootState) -> dict[str, object]:
        return {
            "policy": state.policy.to_json() if state.policy is not None else {},
            "phase_attempts": state.phase_attempts,
            "readiness_result": state.readiness_result.to_json() if state.readiness_result else {},
            "popup_result": state.popup_result.to_json() if state.popup_result else {},
            "emulator_reboot_result": state.emulator_reboot_result.to_json() if state.emulator_reboot_result else {},
            "normal_restart_failed": state.normal_restart_failed,
            "recovery_outcome": state.recovery_outcome,
            "terminal_state": state.terminal_state,
            "terminal_reason": state.terminal_reason,
            "failure_state": state.terminal_state if state.failed else "",
            "failure_reason": state.terminal_reason if state.failed else "",
            "failure_evidence": {
                "screenshot_path": state.screenshot_path,
                "terminal_state": state.terminal_state,
                "terminal_reason": state.terminal_reason,
            },
            "incident_opened": state.incident_opened,
            "circuit_opened": state.circuit_opened,
        }

    def _augment_result(
        self,
        result: WorkflowExecutionResult,
        state: _GameRebootState,
    ) -> None:
        if state.terminal_outcome in {WorkflowOutcome.SKIPPED, WorkflowOutcome.BLOCKED}:
            result.outcome = state.terminal_outcome
            result.message = state.terminal_reason
        result.result = {**dict(result.result), **self._payload(state)}

    def _update_persisted_run(
        self,
        result: WorkflowExecutionResult,
        state: _GameRebootState,
    ) -> None:
        if self.job_runs is None or result.job_run_id is None:
            return
        run = self.job_runs.get(result.job_run_id)
        if run is None:
            return
        run.status = "failed" if result.outcome.is_failure else "completed"
        run.result_json = json.dumps(result.to_json_dict(), sort_keys=True)
        run.error_message = state.terminal_reason if result.outcome.is_failure else ""
        run.screenshot_path = state.screenshot_path
        self.job_runs.save(run)


def _step_result(
    step_key: str,
    outcome: WorkflowOutcome,
    message: str = "",
    *,
    data: dict[str, object] | None = None,
    screenshot_path: str = "",
) -> WorkflowStepResult:
    return WorkflowStepResult(
        step_key=step_key,
        action_type=f"game_reboot.{step_key}",
        outcome=outcome,
        message=message,
        data=data or {},
        screenshot_path=screenshot_path,
    )


def _require_character(state: _GameRebootState) -> Character:
    if state.character is None:
        raise RuntimeError("Target character has not been loaded.")
    return state.character


def _require_policy(state: _GameRebootState) -> GameRebootPolicy:
    if state.policy is None:
        raise RuntimeError("Game reboot policy has not been validated.")
    return state.policy


def _job_run_id(context: WorkflowExecutionContext) -> int | None:
    value = context.metadata.get("_job_run_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalized_keys(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def _command_action(
    action: str,
    result: EmulatorCommandResult[object],
    *,
    retryable: bool,
    data: dict[str, object] | None = None,
) -> GameRebootActionResult:
    if result.succeeded:
        return GameRebootActionResult(True, data=data or {})
    return GameRebootActionResult(
        False,
        command_failure_message(action, result),
        retryable=retryable,
        data={
            **(data or {}),
            "error_category": result.error_category.value,
            "exit_code": result.exit_code,
            "diagnostics": list(result.diagnostics),
        },
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
