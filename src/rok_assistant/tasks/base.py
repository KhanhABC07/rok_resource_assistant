from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rok_assistant.characters import CharacterManager
from rok_assistant.db.repositories import (
    CharacterRepository,
    InstanceRepository,
    MarchRepository,
    SettingsRepository,
)
from rok_assistant.emulator import EmulatorManager
from rok_assistant.task_result import TaskResult as TaskOutcome
from rok_assistant.vision import VisionOcrModule


@dataclass
class TaskResult:
    success: bool
    message: str = ""
    retry_after_seconds: int | None = None
    result: TaskOutcome | None = None

    def __post_init__(self) -> None:
        if self.result is None:
            self.result = TaskOutcome.SUCCESS if self.success else TaskOutcome.FAILED


@dataclass
class TaskContext:
    instances: InstanceRepository
    characters: CharacterRepository
    marches: MarchRepository
    settings: SettingsRepository
    emulator_manager: EmulatorManager
    character_manager: CharacterManager
    vision: VisionOcrModule
    logger: logging.Logger


class TaskPlugin(ABC):
    task_type: str = ""
    display_name: str = ""

    @abstractmethod
    def run(self, task_id: int, context: TaskContext) -> TaskResult:
        raise NotImplementedError
