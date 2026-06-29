from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.action_engine import ActionEngine


class FakeAdbManager:
    def __init__(self, screenshot_path: Path):
        self.screenshot_path = screenshot_path
        self.capture_calls: list[tuple[int, str, Path]] = []

    def capture_screenshot(
        self,
        index: int,
        instance_name: str,
        output_dir: Path,
    ) -> Path:
        self.capture_calls.append((index, instance_name, output_dir))
        return self.screenshot_path


class FakeInputManager:
    def __init__(self):
        self.taps: list[tuple[int, int]] = []
        self.swipes: list[tuple[int, int, int, int, int]] = []

    def tap(self, x: int, y: int) -> bool:
        self.taps.append((x, y))
        return True

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> bool:
        self.swipes.append((x1, y1, x2, y2, duration_ms))
        return True


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class ActionEngineTest(unittest.TestCase):
    def test_click_template_taps_template_center_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            screenshot_path = temp_path / "screenshot.png"
            template_path = temp_path / "template.png"
            screenshot_path.write_bytes(b"fake screenshot")
            template_path.write_bytes(b"fake template")
            adb_manager = FakeAdbManager(screenshot_path)
            input_manager = FakeInputManager()

            def matcher(
                screenshot: str | Path,
                template: str | Path,
                threshold: float,
            ) -> dict[str, object]:
                self.assertEqual(screenshot_path, screenshot)
                self.assertEqual(template_path, template)
                self.assertEqual(0.85, threshold)
                return {"found": True, "confidence": 0.93, "x": 120, "y": 240}

            engine = ActionEngine(
                adb_manager,
                4,
                "MEmu4",
                input_manager=input_manager,
                matcher=matcher,
                template_size_reader=lambda _path: (20, 10),
            )

            result = engine.click_template(template_path, threshold=0.85)

            self.assertTrue(result["success"])
            self.assertEqual(0.93, result["confidence"])
            self.assertEqual(130, result["x"])
            self.assertEqual(245, result["y"])
            self.assertEqual(120, result["template_x"])
            self.assertEqual(240, result["template_y"])
            self.assertEqual(20, result["template_width"])
            self.assertEqual(10, result["template_height"])
            self.assertEqual([(130, 245)], input_manager.taps)
            self.assertEqual([(4, "MEmu4", engine.screenshot_dir)], adb_manager.capture_calls)

    def test_click_template_preserves_zero_top_left_coordinate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            screenshot_path = temp_path / "screenshot.png"
            template_path = temp_path / "template.png"
            screenshot_path.write_bytes(b"fake screenshot")
            template_path.write_bytes(b"fake template")
            adb_manager = FakeAdbManager(screenshot_path)
            input_manager = FakeInputManager()

            def matcher(
                _screenshot: str | Path,
                _template: str | Path,
                _threshold: float,
            ) -> dict[str, object]:
                return {"found": True, "confidence": 0.98, "x": 0, "y": 679}

            engine = ActionEngine(
                adb_manager,
                0,
                "MEmu",
                input_manager=input_manager,
                matcher=matcher,
                template_size_reader=lambda _path: (24, 18),
            )

            result = engine.click_template(template_path)

            self.assertTrue(result["success"])
            self.assertEqual(12, result["x"])
            self.assertEqual(688, result["y"])
            self.assertEqual(0, result["template_x"])
            self.assertEqual(679, result["template_y"])
            self.assertEqual([(12, 688)], input_manager.taps)

    def test_template_center_calculation_never_returns_negative_coordinates(self) -> None:
        self.assertEqual(
            (0, 0),
            ActionEngine.calculate_template_tap_coordinates(-20, -30, 10, 10),
        )
        self.assertEqual(
            (0, 4),
            ActionEngine.calculate_template_tap_coordinates(-5, -1, 10, 10),
        )

    def test_wait_for_template_retries_until_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            screenshot_path = temp_path / "screenshot.png"
            template_path = temp_path / "template.png"
            screenshot_path.write_bytes(b"fake screenshot")
            template_path.write_bytes(b"fake template")
            adb_manager = FakeAdbManager(screenshot_path)
            input_manager = FakeInputManager()
            clock = FakeClock()
            matches = [
                {"found": False, "confidence": 0.3, "x": -1, "y": -1},
                {"found": True, "confidence": 0.91, "x": 10, "y": 20},
            ]

            def matcher(
                _screenshot: str | Path,
                _template: str | Path,
                _threshold: float,
            ) -> dict[str, object]:
                return matches.pop(0)

            engine = ActionEngine(
                adb_manager,
                1,
                "MEmu1",
                input_manager=input_manager,
                matcher=matcher,
                clock=clock.monotonic,
                sleeper=clock.sleep,
            )

            result = engine.wait_for_template(
                template_path,
                threshold=0.8,
                timeout_seconds=2.0,
                retry_interval_seconds=0.5,
            )

            self.assertTrue(result["success"])
            self.assertEqual(0.91, result["confidence"])
            self.assertEqual(10, result["x"])
            self.assertEqual(20, result["y"])
            self.assertEqual(0.5, result["elapsed_time"])
            self.assertEqual([0.5], clock.sleeps)
            self.assertEqual(2, len(adb_manager.capture_calls))

    def test_wait_for_template_returns_timeout_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            screenshot_path = temp_path / "screenshot.png"
            template_path = temp_path / "template.png"
            screenshot_path.write_bytes(b"fake screenshot")
            template_path.write_bytes(b"fake template")
            adb_manager = FakeAdbManager(screenshot_path)
            clock = FakeClock()

            def matcher(
                _screenshot: str | Path,
                _template: str | Path,
                _threshold: float,
            ) -> dict[str, object]:
                return {"found": False, "confidence": 0.4, "x": -1, "y": -1}

            engine = ActionEngine(
                adb_manager,
                2,
                "MEmu2",
                input_manager=FakeInputManager(),
                matcher=matcher,
                clock=clock.monotonic,
                sleeper=clock.sleep,
            )

            result = engine.wait_for_template(
                template_path,
                timeout_seconds=1.0,
                retry_interval_seconds=0.5,
            )

            self.assertFalse(result["success"])
            self.assertEqual("timeout", result["message"])
            self.assertEqual(1.0, result["elapsed_time"])
            self.assertEqual(3, len(adb_manager.capture_calls))

    def test_coordinate_and_swipe_actions_use_input_manager(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot_path = Path(temp_dir) / "screenshot.png"
            adb_manager = FakeAdbManager(screenshot_path)
            input_manager = FakeInputManager()
            engine = ActionEngine(adb_manager, 0, "MEmu", input_manager=input_manager)

            click = engine.click_coordinates(11, 22)
            swipe = engine.swipe_coordinates(1, 2, 3, 4, 500)

            self.assertTrue(click["success"])
            self.assertEqual((11, 22), (click["x"], click["y"]))
            self.assertTrue(swipe["success"])
            self.assertEqual((3, 4), (swipe["x"], swipe["y"]))
            self.assertEqual([(11, 22)], input_manager.taps)
            self.assertEqual([(1, 2, 3, 4, 500)], input_manager.swipes)

    def test_abort_task_returns_default_reason_without_reason(self) -> None:
        engine = ActionEngine(FakeAdbManager(Path("screenshot.png")), 0, "MEmu")

        result = engine.abort_task()

        self.assertTrue(result["success"])
        self.assertTrue(result["aborted"])
        self.assertEqual("Task aborted intentionally", result["message"])
        self.assertEqual("Task aborted intentionally", result["abort_reason"])

    def test_abort_task_returns_custom_reason(self) -> None:
        engine = ActionEngine(FakeAdbManager(Path("screenshot.png")), 0, "MEmu")

        result = engine.abort_task("No free march")

        self.assertTrue(result["success"])
        self.assertTrue(result["aborted"])
        self.assertEqual("No free march", result["message"])
        self.assertEqual("No free march", result["abort_reason"])


if __name__ == "__main__":
    unittest.main()
