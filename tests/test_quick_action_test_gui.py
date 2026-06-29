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

from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QApplication, QPushButton, QTableWidget

from rok_assistant.db.models import Instance
from rok_assistant.gui.automation import AutomationWidget


class FakeInstances:
    def __init__(self) -> None:
        self.rows = [
            Instance(
                id=1,
                name="MEmu",
                instance_index=0,
                instance_name="Primary",
                adb_connected=True,
            ),
            Instance(
                id=2,
                name="MEmu_1",
                instance_index=3,
                instance_name="Farm",
            ),
        ]

    def list_all(self) -> list[Instance]:
        return self.rows

    def get(self, instance_id: int) -> Instance | None:
        return next((row for row in self.rows if row.id == instance_id), None)


class FakeActionEngine:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def click_template(self, _path: Path, *, threshold: float) -> dict[str, object]:
        return {
            "success": True,
            "confidence": threshold,
            "x": 45,
            "y": 60,
            "elapsed_time": 0.2,
            "message": "",
        }

    def click_coordinates(self, x: int, y: int) -> dict[str, object]:
        return {
            "success": True,
            "confidence": 0.0,
            "x": x,
            "y": y,
            "elapsed_time": 0.1,
            "message": "",
        }

    def swipe_coordinates(
        self,
        _x1: int,
        _y1: int,
        x2: int,
        y2: int,
        _duration: int,
    ) -> dict[str, object]:
        return {
            "success": True,
            "confidence": 0.0,
            "x": x2,
            "y": y2,
            "elapsed_time": 0.3,
            "message": "",
        }

    def wait_for_template(
        self,
        _path: Path,
        *,
        threshold: float,
        timeout_seconds: float,
        retry_interval_seconds: float,
    ) -> dict[str, object]:
        return {
            "success": True,
            "confidence": threshold,
            "x": 12,
            "y": 18,
            "elapsed_time": min(timeout_seconds, retry_interval_seconds),
            "message": "",
        }


class QuickActionTestGuiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.template_path = self.root / "template.png"
        self.screenshot_path = self.root / "screenshot.png"
        self._write_image(self.template_path, 20, 12, QColor("blue"))
        self._write_image(self.screenshot_path, 100, 80, QColor("white"))
        self.saved_steps = MagicMock()
        self.context = SimpleNamespace(
            instances=FakeInstances(),
            memu_adb_manager=MagicMock(),
            automation_tasks=self.saved_steps,
        )
        self.widget = AutomationWidget(self.context)
        self.widget._run_background = (
            lambda _action, callback, on_finished: on_finished(callback())
        )

    def tearDown(self) -> None:
        self.widget.deleteLater()
        self.temp_dir.cleanup()

    def test_action_test_is_renamed_quick_action_test(self) -> None:
        self.assertEqual("Quick Action Test", self.widget.test_tabs.tabText(1))

    def test_quick_action_test_does_not_contain_full_task_editor(self) -> None:
        self.assertEqual([], self.widget.quick_action_panel.findChildren(QTableWidget))
        button_texts = {
            button.text()
            for button in self.widget.quick_action_panel.findChildren(QPushButton)
        }
        self.assertNotIn("Add Step", button_texts)
        self.assertNotIn("Save Step", button_texts)
        self.assertNotIn("Browse", button_texts)

    def test_action_specific_fields_appear_only_when_required(self) -> None:
        self.widget.quick_action_combo.setCurrentIndex(
            self.widget.quick_action_combo.findData("click_coordinates")
        )
        self.assertFalse(self.widget.quick_click_coordinates_widget.isHidden())
        self.assertTrue(self.widget.quick_threshold_input.isHidden())
        self.assertTrue(self.widget.quick_swipe_start_widget.isHidden())
        self.assertTrue(self.widget.quick_timeout_input.isHidden())

        self.widget.quick_action_combo.setCurrentIndex(
            self.widget.quick_action_combo.findData("wait_for_template")
        )
        self.assertFalse(self.widget.quick_threshold_input.isHidden())
        self.assertFalse(self.widget.quick_timeout_input.isHidden())
        self.assertFalse(self.widget.quick_retry_input.isHidden())
        self.assertTrue(self.widget.quick_click_coordinates_widget.isHidden())
        self.assertTrue(self.widget.quick_swipe_start_widget.isHidden())

        self.widget.quick_action_combo.setCurrentIndex(
            self.widget.quick_action_combo.findData("swipe_coordinates")
        )
        self.assertFalse(self.widget.quick_swipe_start_widget.isHidden())
        self.assertFalse(self.widget.quick_swipe_end_widget.isHidden())
        self.assertFalse(self.widget.quick_swipe_duration_input.isHidden())
        self.assertTrue(self.widget.quick_threshold_input.isHidden())

        self.widget.quick_action_combo.setCurrentIndex(
            self.widget.quick_action_combo.findData("click_template")
        )
        self.assertFalse(self.widget.quick_threshold_input.isHidden())
        self.assertTrue(self.widget.quick_timeout_input.isHidden())
        self.assertTrue(self.widget.quick_click_coordinates_widget.isHidden())
        self.assertTrue(self.widget.quick_swipe_start_widget.isHidden())

        self.widget._handle_template_result(
            {
                "found": True,
                "confidence": 0.98,
                "x": 10,
                "y": 20,
                "width": 20,
                "height": 12,
                "center_x": 20,
                "center_y": 26,
                "screenshot": str(self.screenshot_path),
                "template": str(self.template_path),
            }
        )
        self.widget.quick_action_combo.setCurrentIndex(
            self.widget.quick_action_combo.findData("click_last_match")
        )
        self.assertTrue(self.widget.quick_threshold_input.isHidden())
        self.assertTrue(self.widget.quick_timeout_input.isHidden())
        self.assertTrue(self.widget.quick_click_coordinates_widget.isHidden())
        self.assertTrue(self.widget.quick_swipe_start_widget.isHidden())

    def test_click_last_match_is_disabled_before_a_match(self) -> None:
        index = self.widget.quick_action_combo.findData("click_last_match")
        self.assertFalse(self.widget.quick_action_combo.model().item(index).isEnabled())

    def test_click_last_match_is_enabled_after_successful_match(self) -> None:
        self.widget._handle_template_result(
            {
                "found": True,
                "confidence": 0.98,
                "x": 10,
                "y": 20,
                "width": 20,
                "height": 12,
                "center_x": 20,
                "center_y": 26,
                "screenshot": str(self.screenshot_path),
                "template": str(self.template_path),
            }
        )

        index = self.widget.quick_action_combo.findData("click_last_match")
        self.assertTrue(self.widget.quick_action_combo.model().item(index).isEnabled())
        self.widget.quick_action_combo.setCurrentIndex(index)
        self.assertTrue(self.widget.quick_run_button.isEnabled())
        self.assertIn("center (20, 26)", self.widget.quick_last_match_value.text())

    def test_failed_match_disables_click_last_match(self) -> None:
        self.widget._handle_template_result(
            {
                "found": True,
                "confidence": 0.98,
                "x": 10,
                "y": 20,
                "width": 20,
                "height": 12,
                "center_x": 20,
                "center_y": 26,
                "screenshot": str(self.screenshot_path),
                "template": str(self.template_path),
            }
        )
        self.widget._handle_template_result(
            {
                "found": False,
                "confidence": 0.4,
                "x": -1,
                "y": -1,
                "screenshot": str(self.screenshot_path),
                "template": str(self.template_path),
            }
        )

        index = self.widget.quick_action_combo.findData("click_last_match")
        self.assertFalse(self.widget.quick_action_combo.model().item(index).isEnabled())
        self.assertIsNone(self.widget.last_match)
        self.assertEqual("Unavailable", self.widget.quick_last_match_value.text())

    def test_click_last_match_uses_previous_match_center(self) -> None:
        self.widget._handle_template_result(
            {
                "found": True,
                "confidence": 0.98,
                "x": 10,
                "y": 20,
                "width": 20,
                "height": 12,
                "center_x": 20,
                "center_y": 26,
                "screenshot": str(self.screenshot_path),
                "template": str(self.template_path),
            }
        )
        self.widget.quick_action_combo.setCurrentIndex(
            self.widget.quick_action_combo.findData("click_last_match")
        )

        with patch("rok_assistant.gui.automation.ActionEngine", FakeActionEngine):
            self.widget.run_quick_action()

        self.assertEqual("SUCCESS", self.widget.quick_status_label.text())
        self.assertEqual("(20, 26)", self.widget.quick_coordinates_label.text())
        self.assertIn("shell input tap 20 26", self.widget.quick_command_label.text())

    def test_quick_actions_do_not_create_or_modify_saved_task_steps(self) -> None:
        self.widget.quick_action_combo.setCurrentIndex(
            self.widget.quick_action_combo.findData("click_coordinates")
        )
        self.widget.quick_x_input.setValue(100)
        self.widget.quick_y_input.setValue(200)

        with patch("rok_assistant.gui.automation.ActionEngine", FakeActionEngine):
            self.widget.run_quick_action()

        self.saved_steps.assert_not_called()
        self.assertEqual([], self.saved_steps.method_calls)
        self.assertEqual("SUCCESS", self.widget.quick_status_label.text())
        self.assertIn("shell input tap 100 200", self.widget.quick_command_label.text())

    def test_execution_log_receives_action_progress_with_timestamps(self) -> None:
        self.widget.quick_action_combo.setCurrentIndex(
            self.widget.quick_action_combo.findData("click_coordinates")
        )

        with patch("rok_assistant.gui.automation.ActionEngine", FakeActionEngine):
            self.widget.run_quick_action()

        log = self.widget.quick_execution_log.toPlainText()
        self.assertRegex(log, r"\[\d{2}:\d{2}:\d{2}\] Starting Click Coordinates")
        self.assertIn("completed with SUCCESS", log)
        self.assertIn("ADB command:", log)

    def test_all_supported_quick_actions_are_available(self) -> None:
        actions = {
            self.widget.quick_action_combo.itemData(index)
            for index in range(self.widget.quick_action_combo.count())
        }
        self.assertEqual(
            {
                "click_template",
                "click_coordinates",
                "swipe_coordinates",
                "wait_for_template",
                "click_last_match",
            },
            actions,
        )

    @staticmethod
    def _write_image(path: Path, width: int, height: int, color: QColor) -> None:
        image = QImage(width, height, QImage.Format.Format_RGB32)
        image.fill(color)
        assert image.save(str(path))


if __name__ == "__main__":
    unittest.main()
