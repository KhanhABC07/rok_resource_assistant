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
from PyQt6.QtWidgets import QApplication

from rok_assistant.db.models import Instance
from rok_assistant.gui.template_capture import TemplateCaptureDialog


class FakeInstances:
    def __init__(self) -> None:
        self.instance = Instance(
            id=7,
            name="MEmu",
            instance_index=4,
            instance_name="Capture Source",
        )

    def list_all(self) -> list[Instance]:
        return [self.instance]

    def get(self, instance_id: int) -> Instance | None:
        return self.instance if instance_id == self.instance.id else None


class TemplateCaptureGuiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.screenshot_path = self.root / "screenshot.png"
        image = QImage(100, 60, QImage.Format.Format_RGB32)
        image.fill(QColor("green"))
        self.assertTrue(image.save(str(self.screenshot_path)))
        self.adb = MagicMock()
        self.adb.capture_screenshot.return_value = self.screenshot_path
        self.context = SimpleNamespace(
            instances=FakeInstances(),
            memu_adb_manager=self.adb,
        )
        self.dialog = TemplateCaptureDialog(self.context)

    def tearDown(self) -> None:
        self.dialog.deleteLater()
        self.temp_dir.cleanup()

    def test_instance_capture_loads_screenshot_for_cropping(self) -> None:
        self.dialog.capture_screenshot()

        self.adb.capture_screenshot.assert_called_once_with(4, "Capture Source")
        self.assertEqual(self.screenshot_path, self.dialog.screenshot_path)
        self.assertFalse(self.dialog.preview._source_pixmap.isNull())

    def test_saved_crop_emits_template_path(self) -> None:
        saved_paths: list[str] = []
        self.dialog.template_saved.connect(saved_paths.append)
        self.assertTrue(self.dialog.load_screenshot(self.screenshot_path))
        self.dialog.preview.set_source_selection(QRect(10, 8, 20, 15))
        self.dialog.template_name_input.setText("captured_template")

        with patch(
            "rok_assistant.gui.template_capture.TEMPLATE_DIR",
            self.root / "templates",
        ):
            self.dialog.save_template()

        self.assertEqual(1, len(saved_paths))
        saved = Path(saved_paths[0])
        self.assertTrue(saved.exists())
        result = QImage(str(saved))
        self.assertEqual(20, result.width())
        self.assertEqual(15, result.height())


if __name__ == "__main__":
    unittest.main()
