from __future__ import annotations

import logging

from rok_assistant.tasks.base import TaskContext, TaskPlugin, TaskResult


class AllianceHelpTaskPlugin(TaskPlugin):
    task_type = "alliance_help"
    display_name = "Alliance Help"

    def run(self, task_id: int, context: TaskContext) -> TaskResult:
        task = next((item for item in context.task_lookup(task_id)), None)  # type: ignore[attr-defined]
        if task is None:
            return TaskResult(False, f"Task {task_id} no longer exists.")
        character = context.characters.get(task.character_id or 0)
        if character is None:
            return TaskResult(False, "Character no longer exists.")
        if not character.alliance_help_enabled:
            return TaskResult(True, "Alliance help disabled for character.")
        logging.getLogger(self.__class__.__name__).info(
            "Alliance help flow prepared for %s.", character.name
        )
        return TaskResult(True, "Alliance help flow prepared.")


class AllianceDonateTaskPlugin(TaskPlugin):
    task_type = "alliance_donate"
    display_name = "Alliance Donation"

    def run(self, task_id: int, context: TaskContext) -> TaskResult:
        task = next((item for item in context.task_lookup(task_id)), None)  # type: ignore[attr-defined]
        if task is None:
            return TaskResult(False, f"Task {task_id} no longer exists.")
        character = context.characters.get(task.character_id or 0)
        if character is None:
            return TaskResult(False, "Character no longer exists.")
        if not character.alliance_donate_enabled:
            return TaskResult(True, "Alliance donation disabled for character.")
        logging.getLogger(self.__class__.__name__).info(
            "Alliance donation flow prepared for %s.", character.name
        )
        return TaskResult(True, "Alliance donation flow prepared.")


class GiftCollectionTaskPlugin(TaskPlugin):
    task_type = "gift_collection"
    display_name = "Gift Collection"

    def run(self, task_id: int, context: TaskContext) -> TaskResult:
        task = next((item for item in context.task_lookup(task_id)), None)  # type: ignore[attr-defined]
        if task is None:
            return TaskResult(False, f"Task {task_id} no longer exists.")
        character = context.characters.get(task.character_id or 0)
        if character is None:
            return TaskResult(False, "Character no longer exists.")
        if not character.gift_collection_enabled:
            return TaskResult(True, "Gift collection disabled for character.")
        logging.getLogger(self.__class__.__name__).info(
            "Gift collection flow prepared for %s.", character.name
        )
        return TaskResult(True, "Gift collection flow prepared.")
