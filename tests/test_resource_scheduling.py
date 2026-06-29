from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.app import AppContext
from rok_assistant.db.models import Character


class DisabledCharacterRepository:
    def list_all(self, include_disabled: bool = True) -> list[Character]:
        return [
            Character(
                id=1,
                name="Farm01",
                enabled=True,
                alliance_help_enabled=False,
                alliance_donate_enabled=False,
                gift_collection_enabled=False,
            )
        ]


class UnusedMarchRepository:
    def list_for_character(self, _character_id: int) -> list[object]:
        raise AssertionError(
            "Scheduler generation must not read deprecated march resource configuration."
        )


class ResourceSchedulingTest(unittest.TestCase):
    def test_scheduler_generation_ignores_legacy_march_resource_configuration(
        self,
    ) -> None:
        wake_calls: list[bool] = []
        context = SimpleNamespace(
            characters=DisabledCharacterRepository(),
            marches=UnusedMarchRepository(),
            tasks=SimpleNamespace(
                open_task_exists=lambda *_args: False,
                enqueue=lambda _task: self.fail("No task should be enqueued."),
            ),
            scheduler=SimpleNamespace(wake=lambda: wake_calls.append(True)),
        )

        created = AppContext.schedule_enabled_work(context)

        self.assertEqual(0, created)
        self.assertEqual([True], wake_calls)


if __name__ == "__main__":
    unittest.main()
