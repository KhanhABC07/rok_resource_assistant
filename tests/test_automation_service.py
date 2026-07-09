from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import MagicMock

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.application.automation import AutomationViewModel
from rok_assistant.db.models import Instance


class FakeInstances:
    def __init__(self, rows: list[Instance] | None = None) -> None:
        self.rows = rows or []

    def list_all(self) -> list[Instance]:
        return self.rows

    def get(self, item_id: int) -> Instance | None:
        return next((row for row in self.rows if row.id == item_id), None)


class FakeActionEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def click_coordinates(self, x: int, y: int) -> dict[str, object]:
        self.calls.append(("click_coordinates", (x, y), {}))
        return {
            "success": True,
            "confidence": 0.0,
            "x": x,
            "y": y,
            "elapsed_time": 0.25,
            "message": "",
        }

    def swipe_coordinates(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int,
    ) -> dict[str, object]:
        self.calls.append(("swipe_coordinates", (x1, y1, x2, y2, duration_ms), {}))
        return {
            "success": True,
            "confidence": 0.0,
            "x": x2,
            "y": y2,
            "elapsed_time": 0.5,
            "message": "",
        }

    def click_template(self, path: Path, *, threshold: float) -> dict[str, object]:
        self.calls.append(("click_template", (path,), {"threshold": threshold}))
        return {
            "success": True,
            "confidence": threshold,
            "x": 40,
            "y": 60,
            "elapsed_time": 0.1,
            "message": "",
        }


class AutomationViewModelTest(unittest.TestCase):
    def make_view_model(
        self,
        *,
        instances: FakeInstances | None = None,
        adb_manager: MagicMock | None = None,
        engine: FakeActionEngine | None = None,
    ) -> AutomationViewModel:
        selected_engine = engine or FakeActionEngine()
        return AutomationViewModel(
            instances
            or FakeInstances(
                [
                    Instance(
                        id=1,
                        name="MEmu",
                        instance_index=0,
                        instance_name="Primary",
                        adb_connected=True,
                    )
                ]
            ),
            adb_manager or MagicMock(),
            action_engine_factory=lambda *_args: selected_engine,
        )

    def test_target_rows_format_instance_identity_and_adb_state(self) -> None:
        instances = FakeInstances(
            [
                Instance(
                    id=1,
                    name="MEmu",
                    instance_index=0,
                    instance_name="Primary",
                    adb_connected=True,
                ),
                Instance(id=2, name="No Index", instance_index=None),
            ]
        )
        view_model = self.make_view_model(instances=instances)

        rows = view_model.list_target_rows()

        self.assertEqual(1, len(rows))
        self.assertEqual(1, rows[0].id)
        self.assertEqual("0 — Primary [ADB connected]", rows[0].label)

    def test_capture_screenshots_uses_instance_index_and_display_name(self) -> None:
        adb = MagicMock()
        adb.capture_screenshot.return_value = Path("screenshots/primary.png")
        view_model = self.make_view_model(adb_manager=adb)

        result = view_model.capture_screenshots([1])

        adb.capture_screenshot.assert_called_once_with(0, "Primary")
        self.assertEqual(
            [{"name": "MEmu", "success": True, "path": "screenshots\\primary.png"}],
            result["results"],
        )

    def test_find_template_adds_template_size_and_center_without_qt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            screenshot = root / "screenshot.png"
            template = root / "template.png"
            cv2.imwrite(str(screenshot), np.zeros((40, 60, 3), dtype=np.uint8))
            cv2.imwrite(str(template), np.zeros((12, 20, 3), dtype=np.uint8))
            matcher = MagicMock(
                return_value={"found": True, "confidence": 0.9, "x": 5, "y": 7}
            )
            view_model = self.make_view_model()

            result = view_model.find_template(screenshot, template, 0.75, matcher=matcher)

        matcher.assert_called_once_with(screenshot, template, threshold=0.75)
        self.assertEqual(20, result["width"])
        self.assertEqual(12, result["height"])
        self.assertEqual(15, result["center_x"])
        self.assertEqual(13, result["center_y"])

    def test_validate_quick_action_blocks_missing_template_and_last_match(self) -> None:
        view_model = self.make_view_model()

        missing_template = view_model.validate_quick_action(
            command="click_template",
            selected_instance_id=1,
            selected_template_path=None,
            last_match=None,
        )
        missing_match = view_model.validate_quick_action(
            command="click_last_match",
            selected_instance_id=1,
            selected_template_path=Path("template.png"),
            last_match=None,
        )

        self.assertFalse(missing_template.allowed)
        self.assertEqual(
            "Select a template in Image Recognition Test first.",
            missing_template.warning_message,
        )
        self.assertFalse(missing_match.allowed)
        self.assertEqual("Run a successful match first.", missing_match.warning_message)

    def test_run_quick_action_dispatches_to_engine_and_formats_adb_command(self) -> None:
        engine = FakeActionEngine()
        view_model = self.make_view_model(engine=engine)

        result = view_model.run_quick_action(
            instance_id=1,
            command="click_last_match",
            parameters={
                "last_match_x": 20,
                "last_match_y": 26,
            },
        )

        self.assertEqual([("click_coordinates", (20, 26), {})], engine.calls)
        self.assertTrue(result["success"])
        self.assertEqual("memuc adb -i 0 shell input tap 20 26", result["adb_command"])

    def test_quick_action_result_view_formats_status_for_gui_rendering(self) -> None:
        view_model = self.make_view_model()

        view = view_model.quick_action_result_view(
            {
                "action": "click_template",
                "success": True,
                "confidence": 0.81234,
                "x": 40,
                "y": 60,
                "elapsed_time": 1.25,
                "message": "",
                "adb_command": "memuc adb -i 0 shell input tap 40 60",
            }
        )

        self.assertEqual("SUCCESS", view.status)
        self.assertEqual("success", view.status_kind)
        self.assertEqual("0.8123", view.confidence_text)
        self.assertEqual("(40, 60)", view.coordinates_text)
        self.assertIn("Elapsed: 1.25s", view.result_summary)


if __name__ == "__main__":
    unittest.main()
