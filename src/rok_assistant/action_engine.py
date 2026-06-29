from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Mapping

import cv2

from rok_assistant.emulator import MEmuAdbInputManager, MEmuAdbManager
from rok_assistant.paths import SCREENSHOT_DIR
from rok_assistant.vision import find_template


TemplateMatcher = Callable[[str | Path, str | Path, float], Mapping[str, object]]
TemplateSizeReader = Callable[[str | Path], tuple[int, int]]
DEFAULT_ABORT_REASON = "Task aborted intentionally"


class ActionEngine:
    def __init__(
        self,
        adb_manager: MEmuAdbManager,
        instance_index: int,
        instance_name: str = "",
        *,
        input_manager: MEmuAdbInputManager | None = None,
        screenshot_dir: Path = SCREENSHOT_DIR,
        matcher: TemplateMatcher = find_template,
        template_size_reader: TemplateSizeReader | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.adb_manager = adb_manager
        self.instance_index = instance_index
        self.instance_name = instance_name or f"index {instance_index}"
        self.input_manager = input_manager or MEmuAdbInputManager(
            adb_manager,
            instance_index,
            self.instance_name,
        )
        self.screenshot_dir = screenshot_dir
        self.matcher = matcher
        self.template_size_reader = template_size_reader or self._read_template_size
        self.clock = clock
        self.sleeper = sleeper
        self.logger = logging.getLogger(self.__class__.__name__)

    def wait_for_template(
        self,
        template_path: str | Path,
        *,
        threshold: float = 0.8,
        timeout_seconds: float = 10.0,
        retry_interval_seconds: float = 1.0,
    ) -> dict[str, object]:
        action = "wait_for_template"
        started_at = self._start_action(
            action,
            "template=%s threshold=%.4f timeout=%.2fs retry=%.2fs",
            template_path,
            threshold,
            timeout_seconds,
            retry_interval_seconds,
        )
        timeout_seconds = max(0.0, float(timeout_seconds))
        retry_interval_seconds = max(0.01, float(retry_interval_seconds))
        last_match = {"confidence": 0.0, "x": -1, "y": -1}

        while True:
            match = self._capture_and_match(template_path, threshold, action, started_at)
            last_match = {
                "confidence": float(match.get("confidence", 0.0) or 0.0),
                "x": self._int_value(match.get("x"), -1),
                "y": self._int_value(match.get("y"), -1),
            }
            if match.get("success"):
                self._log_success(action, match)
                return match

            if match.get("fatal"):
                self._log_failure(action, str(match.get("message", "action failed")), match)
                return match

            elapsed = self.clock() - started_at
            if elapsed >= timeout_seconds:
                result = self._result(
                    False,
                    confidence=float(last_match["confidence"]),
                    x=int(last_match["x"]),
                    y=int(last_match["y"]),
                    elapsed_time=elapsed,
                    message="timeout",
                )
                self.logger.warning(
                    "[Action] %s timeout on %s after %.2fs",
                    action,
                    self.instance_name,
                    elapsed,
                )
                return result

            self.sleeper(min(retry_interval_seconds, timeout_seconds - elapsed))

    def click_template(
        self,
        template_path: str | Path,
        *,
        threshold: float = 0.8,
    ) -> dict[str, object]:
        action = "click_template"
        started_at = self._start_action(
            action,
            "template=%s threshold=%.4f",
            template_path,
            threshold,
        )
        match = self._capture_and_match(template_path, threshold, action, started_at)
        if not match.get("success"):
            self._log_failure(action, str(match.get("message", "template not found")), match)
            return match

        top_left_x = self._int_value(match.get("x"), -1)
        top_left_y = self._int_value(match.get("y"), -1)
        try:
            template_width, template_height = self.template_size_reader(template_path)
        except (FileNotFoundError, ValueError, OSError) as exc:
            result = self._result(
                False,
                confidence=float(match.get("confidence", 0.0) or 0.0),
                x=top_left_x,
                y=top_left_y,
                elapsed_time=self.clock() - started_at,
                message=str(exc),
                screenshot_path=str(match.get("screenshot_path", "")),
                fatal=True,
            )
            self._log_failure(action, str(exc), result)
            return result

        tap_x, tap_y = self.calculate_template_tap_coordinates(
            top_left_x,
            top_left_y,
            template_width,
            template_height,
        )
        self.logger.info(
            "[Action] %s coordinates on %s: template_top_left=(%s,%s) "
            "template_width=%s template_height=%s tap=(%s,%s)",
            action,
            self.instance_name,
            top_left_x,
            top_left_y,
            template_width,
            template_height,
            tap_x,
            tap_y,
        )
        tapped = self.input_manager.tap(tap_x, tap_y)
        elapsed = self.clock() - started_at
        result = self._result(
            tapped,
            confidence=float(match.get("confidence", 0.0) or 0.0),
            x=tap_x,
            y=tap_y,
            elapsed_time=elapsed,
            message="" if tapped else "tap failed",
            screenshot_path=str(match.get("screenshot_path", "")),
            template_x=top_left_x,
            template_y=top_left_y,
            template_width=template_width,
            template_height=template_height,
            tap_x=tap_x,
            tap_y=tap_y,
        )
        if tapped:
            self._log_success(action, result)
        else:
            self._log_failure(action, "tap failed", result)
        return result

    def click_coordinates(self, x: int, y: int) -> dict[str, object]:
        action = "click_coordinates"
        started_at = self._start_action(action, "x=%s y=%s", x, y)
        success = self.input_manager.tap(x, y)
        result = self._result(success, x=x, y=y, elapsed_time=self.clock() - started_at)
        if success:
            self._log_success(action, result)
        else:
            self._log_failure(action, "tap failed", result)
        return result

    def swipe_coordinates(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int,
    ) -> dict[str, object]:
        action = "swipe_coordinates"
        started_at = self._start_action(
            action,
            "x1=%s y1=%s x2=%s y2=%s duration_ms=%s",
            x1,
            y1,
            x2,
            y2,
            duration_ms,
        )
        success = self.input_manager.swipe(x1, y1, x2, y2, duration_ms)
        result = self._result(success, x=x2, y=y2, elapsed_time=self.clock() - started_at)
        if success:
            self._log_success(action, result)
        else:
            self._log_failure(action, "swipe failed", result)
        return result

    def abort_task(self, reason: str | None = None) -> dict[str, object]:
        abort_reason = str(reason or "").strip() or DEFAULT_ABORT_REASON
        self.logger.info("AbortTask executed: %s", abort_reason)
        return {
            "success": True,
            "aborted": True,
            "message": abort_reason,
            "abort_reason": abort_reason,
        }

    def _capture_and_match(
        self,
        template_path: str | Path,
        threshold: float,
        action: str,
        started_at: float,
    ) -> dict[str, object]:
        screenshot_path = self.adb_manager.capture_screenshot(
            self.instance_index,
            self.instance_name,
            self.screenshot_dir,
        )
        if screenshot_path is None:
            return self._result(
                False,
                elapsed_time=self.clock() - started_at,
                message="screenshot capture failed",
            )

        try:
            match = self.matcher(screenshot_path, template_path, threshold)
        except (FileNotFoundError, ValueError, OSError) as exc:
            return self._result(
                False,
                elapsed_time=self.clock() - started_at,
                message=str(exc),
                screenshot_path=str(screenshot_path),
                fatal=True,
            )

        found = bool(match.get("found"))
        return self._result(
            found,
            confidence=float(match.get("confidence", 0.0) or 0.0),
            x=self._int_value(match.get("x"), -1),
            y=self._int_value(match.get("y"), -1),
            elapsed_time=self.clock() - started_at,
            message="" if found else "template not found",
            screenshot_path=str(screenshot_path),
        )

    @staticmethod
    def calculate_template_tap_coordinates(
        template_x: int,
        template_y: int,
        template_width: int,
        template_height: int,
    ) -> tuple[int, int]:
        tap_x = template_x + max(0, template_width) // 2
        tap_y = template_y + max(0, template_height) // 2
        return max(0, tap_x), max(0, tap_y)

    @staticmethod
    def _read_template_size(template_path: str | Path) -> tuple[int, int]:
        template = Path(template_path)
        if not template.exists():
            raise FileNotFoundError(f"Template not found: {template}")
        image = cv2.imread(str(template), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Template is not a readable image: {template}")
        height, width = image.shape[:2]
        return int(width), int(height)

    @staticmethod
    def _int_value(value: object, default: int) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _start_action(self, action: str, details: str, *args: object) -> float:
        started_at = self.clock()
        self.logger.info(
            "[Action] %s started on %s: " + details,
            action,
            self.instance_name,
            *args,
        )
        return started_at

    def _log_success(self, action: str, result: Mapping[str, object]) -> None:
        self.logger.info(
            "[Action] %s success on %s confidence=%.4f x=%s y=%s elapsed=%.2fs",
            action,
            self.instance_name,
            float(result.get("confidence", 0.0) or 0.0),
            result.get("x", -1),
            result.get("y", -1),
            float(result.get("elapsed_time", 0.0) or 0.0),
        )

    def _log_failure(
        self,
        action: str,
        message: str,
        result: Mapping[str, object],
    ) -> None:
        self.logger.error(
            "[Action] %s failed on %s: %s confidence=%.4f x=%s y=%s elapsed=%.2fs",
            action,
            self.instance_name,
            message,
            float(result.get("confidence", 0.0) or 0.0),
            result.get("x", -1),
            result.get("y", -1),
            float(result.get("elapsed_time", 0.0) or 0.0),
        )

    @staticmethod
    def _result(
        success: bool,
        *,
        confidence: float = 0.0,
        x: int = -1,
        y: int = -1,
        elapsed_time: float = 0.0,
        message: str = "",
        screenshot_path: str = "",
        fatal: bool = False,
        **extra: object,
    ) -> dict[str, object]:
        result = {
            "success": success,
            "confidence": confidence,
            "x": x,
            "y": y,
            "elapsed_time": elapsed_time,
            "message": message,
            "screenshot_path": screenshot_path,
            "fatal": fatal,
        }
        result.update(extra)
        return result
