from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from PyQt6.QtWidgets import QApplication, QComboBox

from rok_assistant.db.models import Character, March
from rok_assistant.gui.march_config import MarchConfigWidget


class FakeCharacters:
    def list_all(self) -> list[Character]:
        return [
            Character(
                id=1,
                name="Farm01",
                instance_name="MEmu",
            )
        ]


class FakeMarches:
    def __init__(self) -> None:
        self.saved: list[March] = []

    def list_for_character(self, character_id: int) -> list[March]:
        return [
            March(character_id=character_id, march_slot=slot)
            for slot in range(1, 6)
        ]

    def save(self, march: March) -> int:
        self.saved.append(march)
        return march.march_slot


class MarchConfigWidgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.marches = FakeMarches()
        self.widget = MarchConfigWidget(
            SimpleNamespace(
                characters=FakeCharacters(),
                marches=self.marches,
                schedule_enabled_work=lambda: 0,
            )
        )

    def tearDown(self) -> None:
        self.widget.deleteLater()

    def test_marches_table_contains_only_runtime_state_columns(self) -> None:
        headers = [
            self.widget.table.horizontalHeaderItem(column).text()
            for column in range(self.widget.table.columnCount())
        ]

        self.assertEqual(
            ["Slot", "Status", "Next Action Time", "Expected Return Time"],
            headers,
        )
        self.assertNotIn("Resource Source", headers)
        self.assertNotIn("Resource Type", headers)
        self.assertEqual([], self.widget.table.findChildren(QComboBox))

    def test_saving_runtime_state_does_not_expose_resource_configuration(self) -> None:
        with patch("rok_assistant.gui.march_config.QMessageBox.information"):
            self.widget.save()

        self.assertEqual(5, len(self.marches.saved))
        self.assertTrue(all(march.status == "idle" for march in self.marches.saved))


if __name__ == "__main__":
    unittest.main()
