from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from rok_assistant.db.models import Instance
from rok_assistant.db.repositories import InstanceRepository
from rok_assistant.emulator.memu_manager import MEmuManager
from rok_assistant.emulator.provider import (
    CommandRunner,
    LEGACY_COMMAND_TIMEOUT_SECONDS,
    execute_legacy_command,
)


class EmulatorState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"


@dataclass
class ManagedProcess:
    instance_id: int
    command: tuple[str, ...]


class EmulatorManager:
    def __init__(
        self,
        instance_repository: InstanceRepository,
        memu_manager: MEmuManager | None = None,
        max_concurrent_provider: Callable[[], int] | None = None,
        command_timeout_seconds: int = LEGACY_COMMAND_TIMEOUT_SECONDS,
        command_runner: CommandRunner | None = None,
    ):
        self.instances = instance_repository
        self.memu_manager = memu_manager or MEmuManager()
        self.max_concurrent_provider = max_concurrent_provider or (lambda: 5)
        self.command_timeout_seconds = command_timeout_seconds
        self.command_runner = command_runner
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lock = threading.RLock()
        self._processes: dict[int, ManagedProcess] = {}
        self._states: dict[int, EmulatorState] = {}
        self._starting_instances: set[int] = set()

    def set_memu_install_path(self, install_path: str | Path) -> None:
        self.memu_manager.set_install_path(install_path)

    def launch_instance(self, instance: Instance) -> bool:
        if instance.id is None:
            raise ValueError("Cannot launch an unsaved emulator instance.")
        if not instance.enabled:
            self.logger.info("Instance %s is disabled; launch skipped.", instance.name)
            return False

        with self._lock:
            if self.is_running(instance):
                self.logger.info("[MEmu] Reusing running instance %s", instance.name)
                return True
            if self.running_count() >= self._max_concurrent_instances():
                self.logger.warning(
                    "[MEmu] Maximum concurrent instances reached; launch skipped for %s",
                    instance.name,
                )
                return False
            self._states[instance.id] = EmulatorState.STARTING
            self._starting_instances.add(instance.id)

        if instance.instance_index is not None:
            success = self.memu_manager.launch_instance(instance.instance_index)
            with self._lock:
                self._starting_instances.discard(instance.id)
                self._states[instance.id] = (
                    EmulatorState.RUNNING if success else EmulatorState.FAILED
                )
            if success:
                self.logger.info("[MEmu] Start instance %s success", instance.name)
            return success

        if self._command_is_empty(instance.launch_command):
            self.logger.warning("Instance %s has no launch command configured.", instance.name)
            with self._lock:
                self._starting_instances.discard(instance.id)
                self._states[instance.id] = EmulatorState.FAILED
            return False

        cwd = Path(instance.launch_path).expanduser() if instance.launch_path.strip() else None
        if cwd is not None and not cwd.exists():
            self.logger.error("Launch path does not exist for %s: %s", instance.name, cwd)
            with self._lock:
                self._starting_instances.discard(instance.id)
                self._states[instance.id] = EmulatorState.FAILED
            return False

        self.logger.info("Launching emulator instance %s", instance.name)
        result = execute_legacy_command(
            instance.launch_command,
            cwd=cwd,
            timeout_seconds=self.command_timeout_seconds,
            command_runner=self.command_runner,
        )
        if not result.succeeded:
            self.logger.error(
                "Launch failed for instance %s: %s",
                instance.name,
                result.error_message or result.stderr or result.error_category.value,
            )
            with self._lock:
                self._starting_instances.discard(instance.id)
                self._states[instance.id] = EmulatorState.FAILED
            return False

        with self._lock:
            self._processes[instance.id] = ManagedProcess(
                instance_id=instance.id,
                command=result.command,
            )
            self._starting_instances.discard(instance.id)
            self._states[instance.id] = EmulatorState.RUNNING
            return True

    def close_instance(self, instance: Instance) -> bool:
        if instance.id is None:
            return False

        self.logger.info("Closing emulator instance %s", instance.name)
        if instance.instance_index is not None:
            success = self.memu_manager.stop_instance(instance.instance_index)
            with self._lock:
                self._starting_instances.discard(instance.id)
                self._states[instance.id] = (
                    EmulatorState.STOPPED if success else EmulatorState.FAILED
                )
            if success:
                self.logger.info("[MEmu] Stop instance %s success", instance.name)
            return success

        with self._lock:
            if not self._command_is_empty(instance.close_command):
                cwd = Path(instance.launch_path).expanduser() if instance.launch_path.strip() else None
                result = execute_legacy_command(
                    instance.close_command,
                    cwd=cwd,
                    timeout_seconds=self.command_timeout_seconds,
                    command_runner=self.command_runner,
                )
                if not result.succeeded:
                    self.logger.error(
                        "Close command failed for instance %s: %s",
                        instance.name,
                        result.error_message or result.stderr or result.error_category.value,
                    )
                    self._states[instance.id] = EmulatorState.FAILED
                    return False

            self._processes.pop(instance.id, None)
            self._states[instance.id] = EmulatorState.STOPPED
            return True

    def restart_instance(self, instance: Instance) -> bool:
        self.logger.warning("Restarting emulator instance %s", instance.name)
        self.close_instance(instance)
        return self.launch_instance(instance)

    def is_running(self, instance: Instance) -> bool:
        if instance.id is None:
            return False
        if instance.instance_index is not None:
            running = self.memu_manager.is_running(instance.instance_index)
            with self._lock:
                self._states[instance.id] = (
                    EmulatorState.RUNNING if running else EmulatorState.STOPPED
                )
            return running
        with self._lock:
            managed = self._processes.get(instance.id)
            if managed is None:
                return self._states.get(instance.id) == EmulatorState.RUNNING
            return self._states.get(instance.id) == EmulatorState.RUNNING

    def state_for(self, instance_id: int) -> EmulatorState:
        instance = self.instances.get(instance_id)
        if instance and instance.instance_index is not None:
            return EmulatorState.RUNNING if self.is_running(instance) else EmulatorState.STOPPED
        with self._lock:
            return self._states.get(instance_id, EmulatorState.STOPPED)

    def running_count(self) -> int:
        instances = self.instances.list_all()
        with self._lock:
            starting = set(self._starting_instances)
        return sum(
            1
            for instance in instances
            if (instance.id in starting if instance.id is not None else False)
            or self.is_running(instance)
        )

    def _max_concurrent_instances(self) -> int:
        try:
            return max(1, int(self.max_concurrent_provider()))
        except (TypeError, ValueError):
            return 5

    @staticmethod
    def _command_is_empty(command: str | Sequence[str]) -> bool:
        if isinstance(command, str):
            return not command.strip()
        return not command
