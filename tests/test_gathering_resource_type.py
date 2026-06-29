from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.db.models import Character, Instance, March, ScheduledTask
from rok_assistant.tasks.gathering import GatheringTaskPlugin
from rok_assistant.vision.ocr import ResourceNode


class StaticRepository:
    def __init__(self, item: object) -> None:
        self.item = item

    def get(self, _item_id: int) -> object:
        return self.item


class LegacyMarchRepository:
    def list_for_character(self, _character_id: int) -> list[March]:
        raise AssertionError("Gathering must not read legacy march resource configuration.")


class RecordingVision:
    def __init__(self) -> None:
        self.resource_type = ""

    def detect_unexpected_popup(self, _instance_id: int) -> bool:
        return False

    def find_resource_node(
        self,
        _instance_id: int,
        resource_type: str,
        _preferred_levels: list[int],
        _minimum_level: int,
    ) -> ResourceNode:
        self.resource_type = resource_type
        return ResourceNode(resource_type, 6, 1.0, 0, 0)


class GatheringResourceTypeTest(unittest.TestCase):
    def test_task_resource_type_is_not_overridden_by_march_configuration(self) -> None:
        task = ScheduledTask(
            id=1,
            character_id=2,
            march_slot=3,
            task_type="gathering",
            resource_type="Wood",
        )
        vision = RecordingVision()
        context = SimpleNamespace(
            task_lookup=lambda _task_id: [task],
            characters=StaticRepository(Character(id=2, name="Farm01", instance_id=4)),
            instances=StaticRepository(Instance(id=4, name="MEmu")),
            marches=LegacyMarchRepository(),
            settings=SimpleNamespace(
                get_json=lambda _key, default: default,
                get_int=lambda _key, default: default,
            ),
            emulator_manager=SimpleNamespace(launch_instance=lambda _instance: True),
            character_manager=SimpleNamespace(
                switch_to_character=lambda _character: True
            ),
            vision=vision,
        )

        result = GatheringTaskPlugin().run(1, context)

        self.assertTrue(result.success)
        self.assertEqual("Wood", vision.resource_type)


if __name__ == "__main__":
    unittest.main()
