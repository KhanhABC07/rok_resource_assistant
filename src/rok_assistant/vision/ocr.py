from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass
class ResourceNode:
    resource_type: str
    level: int
    confidence: float
    screen_x: int
    screen_y: int


class VisionOcrModule:
    """Interface-style OCR module.

    Replace these methods with an implementation based on your preferred OCR or
    computer-vision stack. The scheduler and task plugins should depend on this
    abstraction rather than importing OCR libraries directly.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect_unexpected_popup(self, instance_id: int) -> bool:
        self.logger.debug("Popup detection requested for instance %s.", instance_id)
        return False

    def read_screen_text(self, instance_id: int) -> str:
        self.logger.debug("OCR text read requested for instance %s.", instance_id)
        return ""

    def find_resource_node(
        self,
        instance_id: int,
        resource_type: str,
        preferred_levels: list[int],
        minimum_level: int,
    ) -> ResourceNode | None:
        self.logger.info(
            "Resource node scan requested: instance=%s resource=%s levels=%s min=%s",
            instance_id,
            resource_type,
            preferred_levels,
            minimum_level,
        )
        return None
