from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from rok_assistant.db.models import Instance
from rok_assistant.db.repositories import InstanceRepository
from rok_assistant.emulator.memu_manager import MEmuManager


class EmulatorState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"


@dataclass
class ManagedProcess:
    instance_id: int
    process: subprocess.Popen
    command: str


class EmulatorManager:
    def __init__(
        self,
        instance_repository: InstanceRepository,
        memu_manager: MEmuManager | None = None,
        max_concurrent_provider: Callable[[], int] | None = None,
    ):
        self.instances = instance_repository
        self.memu_manager = memu_manager or MEmuManager()
        self.max_concurrent_provider = max_concurrent_provider or (lambda: 5)
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

        if not instance.launch_command.strip():
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
        with self._lock:
            try:
                process = subprocess.Popen(
                    instance.launch_command,
                    cwd=str(cwd) if cwd else None,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                self.logger.exception("Launch failed for instance %s", instance.name)
                self._starting_instances.discard(instance.id)
                self._states[instance.id] = EmulatorState.FAILED
                return False

            self._processes[instance.id] = ManagedProcess(
                instance_id=instance.id,
                process=process,
                command=instance.launch_command,
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
            if instance.close_command.strip():
                try:
                    subprocess.Popen(
                        instance.close_command,
                        cwd=instance.launch_path or None,
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except OSError:
                    self.logger.exception("Close command failed for instance %s", instance.name)
                    self._states[instance.id] = EmulatorState.FAILED
                    return False

            managed = self._processes.get(instance.id)
            if managed and managed.process.poll() is None:
                managed.process.terminate()
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
            running = managed.process.poll() is None
            self._states[instance.id] = EmulatorState.RUNNING if running else EmulatorState.STOPPED
            if not running:
                self._processes.pop(instance.id, None)
            return running

    def state_for(self, instance_id: int) -> EmulatorState:
        instance = self.instances.get(instance_id)
        if instance and instance.instance_index is not None:
            return EmulatorState.RUNNING if self.is_running(instance) else EmulatorState.STOPPED
        with self._lock:
            managed = self._processes.get(instance_id)
            if managed and managed.process.poll() is None:
                return EmulatorState.RUNNING
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
