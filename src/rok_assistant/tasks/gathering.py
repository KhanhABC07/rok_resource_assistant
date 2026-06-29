from __future__ import annotations

import logging

from rok_assistant.tasks.base import TaskContext, TaskPlugin, TaskResult


class GatheringTaskPlugin(TaskPlugin):
    task_type = "gathering"
    display_name = "Gather Resources"

    def run(self, task_id: int, context: TaskContext) -> TaskResult:
        task = next((item for item in context.task_lookup(task_id)), None)  # type: ignore[attr-defined]
        if task is None:
            return TaskResult(False, f"Task {task_id} no longer exists.")

        character = context.characters.get(task.character_id or 0)
        if character is None:
            return TaskResult(False, "Character no longer exists.")
        instance = context.instances.get(character.instance_id or 0)
        if instance is None:
            return TaskResult(False, "Instance no longer exists.")

        logger = logging.getLogger(self.__class__.__name__)
        logger.info(
            "Executing gathering task %s for character=%s march=%s",
            task_id,
            character.name,
            task.march_slot,
        )

        if not context.emulator_manager.launch_instance(instance):
            return TaskResult(False, f"Unable to launch instance {instance.name}.")

        if not context.character_manager.switch_to_character(character):
            return TaskResult(False, f"Unable to switch to character {character.name}.")

        if context.vision.detect_unexpected_popup(instance.id or 0):
            return TaskResult(False, "Unexpected popup detected.")

        levels = context.settings.get_json("gathering.preferred_resource_levels", [8, 7, 6])
        minimum_level = context.settings.get_int("gathering.minimum_resource_level", 6)
        resource_type = task.resource_type

        node = context.vision.find_resource_node(
            instance.id or 0,
            resource_type,
            list(levels),
            minimum_level,
        )
        if node is None:
            retry_delay = context.settings.get_int("scheduler.retry_delay_minutes", 10) * 60
            return TaskResult(
                False,
                f"No valid {resource_type} node found at level {minimum_level}+.",
                retry_after_seconds=retry_delay,
            )

        return TaskResult(
            True,
            f"Gathering flow prepared for {resource_type} level {node.level}.",
        )
