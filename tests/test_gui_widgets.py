from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
)

from rok_assistant.gui.widgets import (
    MetricLabel,
    SectionCard,
    StatusBadge,
    apply_button_variant,
    configure_table,
    set_empty_table_state,
    set_table_item,
)


class GuiWidgetHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_configure_table_applies_readability_defaults(self) -> None:
        table = QTableWidget(0, 2)

        configure_table(table, min_height=160)

        self.assertTrue(table.alternatingRowColors())
        self.assertFalse(table.showGrid())
        self.assertFalse(table.wordWrap())
        self.assertFalse(table.verticalHeader().isVisible())
        self.assertEqual(
            QAbstractItemView.SelectionBehavior.SelectRows,
            table.selectionBehavior(),
        )
        self.assertEqual(
            QAbstractItemView.EditTrigger.NoEditTriggers,
            table.editTriggers(),
        )
        self.assertGreaterEqual(table.minimumHeight(), 160)

    def test_set_table_item_returns_item_with_text_and_tooltip(self) -> None:
        table = QTableWidget(1, 1)

        item = set_table_item(table, 0, 0, "Farm queue")

        self.assertIs(item, table.item(0, 0))
        self.assertEqual("Farm queue", item.text())
        self.assertEqual("Farm queue", item.toolTip())

    def test_empty_table_state_toggles_without_adding_rows(self) -> None:
        table = QTableWidget(0, 2)

        set_empty_table_state(table, "No rows yet.")
        label = table.viewport().findChild(QLabel, "emptyTableState")

        self.assertIsNotNone(label)
        assert label is not None
        self.assertEqual("No rows yet.", label.text())
        self.assertFalse(label.isHidden())
        self.assertEqual(0, table.rowCount())

        table.setRowCount(1)
        set_empty_table_state(table, "No rows yet.")

        self.assertTrue(label.isHidden())
        self.assertEqual(1, table.rowCount())

    def test_status_badge_updates_text_and_style(self) -> None:
        badge = StatusBadge("READY", "success")

        badge.set_status("NOT READY", "danger")

        self.assertEqual("NOT READY", badge.text())
        self.assertEqual("danger", badge.property("status"))
        self.assertIn("#fee2e2", badge.styleSheet())

    def test_metric_label_keeps_compatible_value_api(self) -> None:
        metric = MetricLabel("Active Workers")

        metric.set_value(3)

        self.assertEqual("3", metric.value)
        self.assertIn("Active Workers", metric.text())
        self.assertIn("3", metric.text())

    def test_section_card_exposes_content_layout_helpers(self) -> None:
        card = SectionCard("Task Overview", "Recent work")
        child_layout = QVBoxLayout()

        card.addLayout(child_layout)

        self.assertEqual("sectionCard", card.objectName())
        self.assertEqual("Task Overview", card.title_label.text())
        self.assertIsNotNone(card.subtitle_label)
        self.assertGreaterEqual(card.content_layout.count(), 3)

    def test_apply_button_variant_sets_property_and_stylesheet(self) -> None:
        button = QPushButton("Refresh")

        apply_button_variant(button, "secondary")

        self.assertEqual("secondary", button.property("variant"))
        self.assertIn("QPushButton", button.styleSheet())


if __name__ == "__main__":
    unittest.main()
