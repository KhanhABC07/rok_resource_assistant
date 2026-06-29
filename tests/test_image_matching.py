from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.vision import find_template


class ImageMatchingTest(unittest.TestCase):
    def test_find_template_returns_coordinates_and_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            screenshot_path = temp_path / "screenshot.png"
            template_path = temp_path / "template.png"

            screenshot = np.zeros((80, 100, 3), dtype=np.uint8)
            template = self._template()
            screenshot[30:42, 25:35] = template
            cv2.imwrite(str(screenshot_path), screenshot)
            cv2.imwrite(str(template_path), template)

            result = find_template(screenshot_path, template_path, threshold=0.95)

            self.assertTrue(result["found"])
            self.assertGreaterEqual(result["confidence"], 0.99)
            self.assertEqual(25, result["x"])
            self.assertEqual(30, result["y"])

    def test_find_template_returns_not_found_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            screenshot_path = temp_path / "screenshot.png"
            template_path = temp_path / "template.png"

            screenshot = np.zeros((80, 100, 3), dtype=np.uint8)
            template = self._template()
            screenshot[30:42, 25:35] = template
            cv2.imwrite(str(screenshot_path), screenshot)
            cv2.imwrite(str(template_path), template)

            result = find_template(screenshot_path, template_path, threshold=1.01)

            self.assertFalse(result["found"])
            self.assertEqual(-1, result["x"])
            self.assertEqual(-1, result["y"])

    @staticmethod
    def _template() -> np.ndarray:
        template = np.zeros((12, 10, 3), dtype=np.uint8)
        for y in range(template.shape[0]):
            for x in range(template.shape[1]):
                template[y, x] = [(x * 17) % 255, (y * 19) % 255, ((x + y) * 11) % 255]
        return template


if __name__ == "__main__":
    unittest.main()
