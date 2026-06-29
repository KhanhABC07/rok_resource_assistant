from __future__ import annotations

import logging
import threading

from rok_assistant.db.models import Character
from rok_assistant.db.repositories import CharacterRepository


class CharacterManager:
    def __init__(self, character_repository: CharacterRepository):
        self.characters = character_repository
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lock = threading.RLock()
        self._active_character_by_instance: dict[int, int] = {}

    def switch_to_character(self, character: Character) -> bool:
        if character.id is None or character.instance_id is None:
            raise ValueError("Cannot switch to an unsaved character.")

        with self._lock:
            active_character_id = self._active_character_by_instance.get(character.instance_id)
            if active_character_id == character.id:
                self.logger.info(
                    "Character %s already active on instance %s.",
                    character.name,
                    character.instance_name or character.instance_id,
                )
                return True

            self.logger.info(
                "Switching instance %s to character %s.",
                character.instance_name or character.instance_id,
                character.name,
            )
            self._active_character_by_instance[character.instance_id] = character.id
            self.characters.mark_switched(character.id)
            return True

    def active_character_for_instance(self, instance_id: int) -> int | None:
        with self._lock:
            return self._active_character_by_instance.get(instance_id)
