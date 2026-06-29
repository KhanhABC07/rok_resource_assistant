from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2


def find_template(
    screenshot_path: str | Path,
    template_path: str | Path,
    threshold: float = 0.8,
) -> dict[str, Any]:
    logger = logging.getLogger("ImageMatching")
    screenshot = Path(screenshot_path)
    template = Path(template_path)
    if not screenshot.exists():
        raise FileNotFoundError(f"Screenshot not found: {screenshot}")
    if not template.exists():
        raise FileNotFoundError(f"Template not found: {template}")

    screenshot_image = cv2.imread(str(screenshot), cv2.IMREAD_COLOR)
    template_image = cv2.imread(str(template), cv2.IMREAD_COLOR)
    if screenshot_image is None:
        raise ValueError(f"Screenshot is not a readable image: {screenshot}")
    if template_image is None:
        raise ValueError(f"Template is not a readable image: {template}")

    screenshot_height, screenshot_width = screenshot_image.shape[:2]
    template_height, template_width = template_image.shape[:2]
    if template_width > screenshot_width or template_height > screenshot_height:
        logger.info(
            "[ImageMatch] Template larger than screenshot: template=%s screenshot=%s",
            template,
            screenshot,
        )
        return {"found": False, "confidence": 0.0, "x": -1, "y": -1}

    result = cv2.matchTemplate(screenshot_image, template_image, cv2.TM_CCOEFF_NORMED)
    _minimum_value, maximum_value, _minimum_location, maximum_location = cv2.minMaxLoc(result)
    confidence = float(maximum_value)
    found = confidence >= threshold
    x, y = maximum_location if found else (-1, -1)
    logger.info(
        "[ImageMatch] template=%s screenshot=%s confidence=%.4f found=%s x=%s y=%s",
        template,
        screenshot,
        confidence,
        found,
        x,
        y,
    )
    return {
        "found": found,
        "confidence": confidence,
        "x": int(x),
        "y": int(y),
    }
