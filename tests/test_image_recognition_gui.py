from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from PyQt6.QtCore import QRect
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QApplication, QTableWidget, QTabWidget, QWidget

from rok_assistant.db.models import Instance
from rok_assistant.gui.automation import AutomationWidget


class FakeInstances:
    def __init__(self) -> None:
        self.rows = [
            Instance(id=1, name="MEmu", instance_index=0, instance_name="Primary"),
            Instance(id=2, name="MEmu_1", instance_index=3, instance_name="Farm"),
        ]

    def list_all(self) -> list[Instance]:
        return self.rows

    def get(self, instance_id: int) -> Instance | None:
        return next((row for row in self.rows if row.id == instance_id), None)


class ImageRecognitionGuiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.template_path = self.root / "template.png"
        self.screenshot_path = self.root / "screenshot.png"
        self._write_image(self.template_path, 20, 12, QColor("blue"))
        self._write_image(self.screenshot_path, 120, 80, QColor("white"))
        self.adb = MagicMock()
        self.adb.capture_screenshot.return_value = self.screenshot_path
        self.context = SimpleNamespace(
            instances=FakeInstances(),
            memu_adb_manager=self.adb,
        )
        self.widget = AutomationWidget(self.context)
        self.widget._run_background = (
            lambda _action, callback, on_finished: on_finished(callback())
        )

    def tearDown(self) -> None:
        self.widget.deleteLater()
        self.temp_dir.cleanup()

    def test_selected_instance_is_used_for_screenshot_capture(self) -> None:
        self.widget.instance_combo.setCurrentIndex(1)

        self.widget.capture_screenshot()

        self.adb.capture_screenshot.assert_called_once_with(3, "Farm")
        self.assertEqual(self.screenshot_path, self.widget.latest_screenshot_path)

    def test_testing_area_no_longer_contains_full_instance_table(self) -> None:
        self.assertEqual([], self.widget.findChildren(QTableWidget))
        self.assertFalse(hasattr(self.widget, "target_table"))

    def test_compact_instance_selector_lists_discovered_instances_and_state(self) -> None:
        self.assertEqual(2, self.widget.instance_combo.count())
        self.assertIn("0 — Primary", self.widget.instance_combo.itemText(0))
        self.assertIn("ADB disconnected", self.widget.instance_combo.itemText(0))
        self.assertIn("3 — Farm", self.widget.instance_combo.itemText(1))

    def test_refresh_updates_both_instance_selectors(self) -> None:
        self.context.instances.rows.append(
            Instance(
                id=3,
                name="MEmu_2",
                instance_index=5,
                instance_name="New Farm",
                adb_connected=True,
            )
        )

        self.widget.refresh_targets_button.click()

        self.assertEqual(3, self.widget.instance_combo.count())
        self.assertEqual(3, self.widget.quick_instance_combo.count())
        self.assertIn("5 — New Farm [ADB connected]", self.widget.instance_combo.itemText(2))

    def test_open_instances_switches_to_instances_tab(self) -> None:
        tabs = QTabWidget()
        instances_tab = QWidget()
        tabs.addTab(instances_tab, "Instances")
        tabs.addTab(self.widget, "Automation")
        tabs.setCurrentWidget(self.widget)
        self.widget.open_instances_requested.connect(
            lambda: tabs.setCurrentWidget(instances_tab)
        )

        self.widget.open_instances_button.click()

        self.assertIs(instances_tab, tabs.currentWidget())
        tabs.deleteLater()

    def test_image_and_quick_action_share_selected_instance(self) -> None:
        self.widget.instance_combo.setCurrentIndex(1)
        self.assertEqual(2, self.widget.quick_instance_combo.currentData())

        self.widget.quick_instance_combo.setCurrentIndex(0)
        self.assertEqual(1, self.widget.instance_combo.currentData())

    def test_screenshot_preview_receives_expandable_space(self) -> None:
        self.widget.resize(1200, 650)
        self.widget.show()
        self.app.processEvents()

        sizes = self.widget.image_splitter.sizes()
        self.assertGreater(sizes[1], sizes[0])
        self.assertGreater(self.widget.screenshot_preview.height(), 180)

    def test_threshold_value_is_passed_to_find_template(self) -> None:
        self.widget.threshold_input.setValue(0.73)
        with patch(
            "rok_assistant.gui.automation.find_template",
            return_value={"found": False, "confidence": 0.5, "x": -1, "y": -1},
        ) as matcher:
            self.widget._find_template(
                self.screenshot_path,
                self.template_path,
                self.widget.threshold_input.value(),
            )

        matcher.assert_called_once_with(
            self.screenshot_path,
            self.template_path,
            threshold=0.73,
        )

    def test_template_preview_loads(self) -> None:
        self.widget._select_saved_template(str(self.template_path))

        self.assertFalse(self.widget.template_preview.source_pixmap.isNull())
        self.assertEqual(20, self.widget.template_preview.source_pixmap.width())

    def test_screenshot_preview_loads(self) -> None:
        self.widget._handle_screenshot_result(
            {
                "results": [
                    {
                        "name": "Primary",
                        "success": True,
                        "path": str(self.screenshot_path),
                    }
                ]
            }
        )

        self.assertFalse(self.widget.screenshot_preview.source_pixmap.isNull())
        self.assertEqual(120, self.widget.screenshot_preview.source_pixmap.width())

    def test_found_result_displays_confidence_and_coordinates(self) -> None:
        self.widget._handle_template_result(
            {
                "found": True,
                "confidence": 0.9876,
                "x": 14,
                "y": 9,
                "width": 20,
                "height": 12,
                "center_x": 24,
                "center_y": 15,
                "screenshot": str(self.screenshot_path),
                "template": str(self.template_path),
            }
        )

        self.assertEqual("FOUND", self.widget.result_status_label.text())
        self.assertEqual("0.9876", self.widget.result_confidence_label.text())
        self.assertEqual("(14, 9)", self.widget.result_position_label.text())
        self.assertEqual("20 × 12", self.widget.result_size_label.text())
        self.assertEqual("(24, 15)", self.widget.result_center_label.text())
        self.assertEqual("x: 14", self.widget.result_x_label.text())
        self.assertEqual("y: 9", self.widget.result_y_label.text())
        self.assertEqual("width: 20", self.widget.result_width_label.text())
        self.assertEqual("height: 12", self.widget.result_height_label.text())
        self.assertEqual("center x: 24", self.widget.result_center_x_label.text())
        self.assertEqual("center y: 15", self.widget.result_center_y_label.text())

    def test_not_found_clears_stale_coordinates(self) -> None:
        self.widget.result_x_label.setText("x: 99")
        self.widget.result_center_y_label.setText("center y: 101")

        self.widget._handle_template_result(
            {
                "found": False,
                "confidence": 0.42,
                "x": -1,
                "y": -1,
                "screenshot": str(self.screenshot_path),
                "template": str(self.template_path),
            }
        )

        self.assertEqual("NOT FOUND", self.widget.result_status_label.text())
        self.assertEqual("-", self.widget.result_position_label.text())
        self.assertEqual("-", self.widget.result_size_label.text())
        self.assertEqual("-", self.widget.result_center_label.text())
        self.assertEqual("x: -", self.widget.result_x_label.text())
        self.assertEqual("center y: -", self.widget.result_center_y_label.text())
        self.assertIsNone(self.widget.screenshot_preview.match_rect)

    def test_bounding_box_is_drawn_for_found_match(self) -> None:
        self.widget._handle_template_result(
            {
                "found": True,
                "confidence": 0.99,
                "x": 10,
                "y": 11,
                "width": 20,
                "height": 12,
                "center_x": 20,
                "center_y": 17,
                "screenshot": str(self.screenshot_path),
                "template": str(self.template_path),
            }
        )

        self.assertEqual(QRect(10, 11, 20, 12), self.widget.screenshot_preview.match_rect)
        self.assertIsNotNone(self.widget.screenshot_preview.pixmap())

    def test_run_match_is_disabled_until_template_and_screenshot_exist(self) -> None:
        self.assertFalse(self.widget.run_match_button.isEnabled())

        self.widget._select_saved_template(str(self.template_path))
        self.assertFalse(self.widget.run_match_button.isEnabled())

        self.widget.latest_screenshot_path = self.screenshot_path
        self.widget._update_run_match_enabled()
        self.assertTrue(self.widget.run_match_button.isEnabled())

        self.widget.latest_screenshot_path = self.root / "missing.png"
        self.widget._update_run_match_enabled()
        self.assertFalse(self.widget.run_match_button.isEnabled())

    def test_match_status_styling_changes_for_found_not_found_and_error(self) -> None:
        self.widget._set_match_status("FOUND")
        self.assertIn("#dcfce7", self.widget.result_status_label.styleSheet())

        self.widget._set_match_status("NOT FOUND")
        self.assertIn("#fef9c3", self.widget.result_status_label.styleSheet())

        with patch("rok_assistant.gui.automation.QMessageBox.warning"):
            self.widget._handle_worker_failure("run match", "bad image")
        self.assertEqual("ERROR", self.widget.result_status_label.text())
        self.assertIn("#fee2e2", self.widget.result_status_label.styleSheet())

    def test_selected_template_is_shared_with_quick_action_context(self) -> None:
        self.widget._select_saved_template(str(self.template_path))

        self.assertEqual(self.template_path, self.widget.selected_template_path)
        self.assertEqual(str(self.template_path), self.widget.quick_template_value.text())

    def test_open_template_capture_opens_capture_dialog(self) -> None:
        dialog = MagicMock()
        with patch(
            "rok_assistant.gui.automation.TemplateCaptureDialog",
            return_value=dialog,
        ) as dialog_class:
            self.widget.open_template_capture()

        dialog_class.assert_called_once_with(
            self.context,
            screenshot_path=None,
            parent=self.widget,
        )
        dialog.template_saved.connect.assert_called_once()
        dialog.exec.assert_called_once_with()

    def test_newly_saved_template_can_be_selected_automatically(self) -> None:
        self.widget._select_saved_template(str(self.template_path))

        self.assertEqual(self.template_path, self.widget.selected_template_path)
        self.assertIn(str(self.template_path), self.widget.template_label.text())
        self.assertFalse(self.widget.template_preview.source_pixmap.isNull())

    @staticmethod
    def _write_image(path: Path, width: int, height: int, color: QColor) -> None:
        image = QImage(width, height, QImage.Format.Format_RGB32)
        image.fill(color)
        assert image.save(str(path))


if __name__ == "__main__":
    unittest.main()
