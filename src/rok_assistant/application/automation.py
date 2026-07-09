from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

import cv2

from rok_assistant.action_engine import ActionEngine
from rok_assistant.db.models import Instance
from rok_assistant.vision import find_template


TemplateMatcher = Callable[..., Mapping[str, object]]
ActionEngineFactory = Callable[[object, int, str], object]


class InstanceRepositoryPort(Protocol):
    def list_all(self) -> list[Instance]: ...

    def get(self, item_id: int) -> Instance | None: ...


class ScreenshotCapturePort(Protocol):
    def capture_screenshot(
        self,
        instance_index: int,
        instance_name: str,
        *args: object,
        **kwargs: object,
    ) -> str | Path | None: ...


@dataclass(frozen=True)
class TargetInstanceRow:
    id: int
    label: str


@dataclass(frozen=True)
class QuickActionValidation:
    allowed: bool
    log_message: str = ""
    warning_title: str = ""
    warning_message: str = ""


@dataclass(frozen=True)
class QuickActionResultView:
    success: bool
    status: str
    status_kind: str
    confidence: float
    x: int
    y: int
    elapsed_time: float
    message: str
    adb_command: str
    coordinates_text: str
    confidence_text: str
    elapsed_text: str
    message_text: str
    adb_command_text: str
    result_summary: str
    log_summary: str


@dataclass(frozen=True)
class TemplateMatchView:
    found: bool
    status: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    screenshot: str
    template: str
    position_text: str
    size_text: str
    center_text: str
    coordinates_text: str

    @property
    def last_match(self) -> dict[str, object] | None:
        if not self.found:
            return None
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "confidence": self.confidence,
        }


def default_action_engine_factory(
    adb_manager: object,
    instance_index: int,
    instance_name: str,
) -> object:
    return ActionEngine(adb_manager, instance_index, instance_name)


class AutomationViewModel:
    def __init__(
        self,
        instances: InstanceRepositoryPort,
        adb_manager: ScreenshotCapturePort,
        *,
        action_engine_factory: ActionEngineFactory = default_action_engine_factory,
        matcher: TemplateMatcher = find_template,
    ) -> None:
        self.instances = instances
        self.adb_manager = adb_manager
        self.action_engine_factory = action_engine_factory
        self.matcher = matcher

    def list_target_rows(self) -> list[TargetInstanceRow]:
        rows: list[TargetInstanceRow] = []
        for instance in self.instances.list_all():
            if instance.id is None or instance.instance_index is None:
                continue
            name = instance.instance_name or instance.name
            state = "ADB connected" if instance.adb_connected else "ADB disconnected"
            rows.append(
                TargetInstanceRow(
                    id=instance.id,
                    label=f"{instance.instance_index} — {name} [{state}]",
                )
            )
        return rows

    def get_instance(self, instance_id: int) -> Instance | None:
        return self.instances.get(instance_id)

    def capture_screenshots(self, instance_ids: list[int]) -> dict[str, object]:
        results = []
        for instance_id in instance_ids:
            instance = self.instances.get(instance_id)
            if instance is None or instance.instance_index is None:
                continue
            name = instance.instance_name or instance.name
            path = self.adb_manager.capture_screenshot(instance.instance_index, name)
            results.append(
                {
                    "name": instance.name,
                    "success": path is not None,
                    "path": "" if path is None else str(path),
                }
            )
        return {"results": results}

    def find_template(
        self,
        screenshot_path: Path,
        template_path: Path,
        threshold: float,
        *,
        matcher: TemplateMatcher | None = None,
    ) -> dict[str, object]:
        effective_matcher = matcher or self.matcher
        result = dict(effective_matcher(screenshot_path, template_path, threshold=threshold))
        width, height = self._template_size(template_path)
        found = bool(result.get("found"))
        x = self.int_value(result.get("x"), -1)
        y = self.int_value(result.get("y"), -1)
        return {
            "screenshot": str(screenshot_path),
            "template": str(template_path),
            **result,
            "width": width if found else 0,
            "height": height if found else 0,
            "center_x": x + width // 2 if found else -1,
            "center_y": y + height // 2 if found else -1,
        }

    def validate_quick_action(
        self,
        *,
        command: str,
        selected_instance_id: int | None,
        selected_template_path: Path | None,
        last_match: Mapping[str, object] | None,
    ) -> QuickActionValidation:
        if selected_instance_id is None or self.instances.get(selected_instance_id) is None:
            return QuickActionValidation(
                allowed=False,
                log_message="Action blocked: no MEmu instance selected.",
                warning_title="Quick Action Test",
                warning_message="Select a MEmu instance first.",
            )
        if command in {"wait_for_template", "click_template"} and selected_template_path is None:
            return QuickActionValidation(
                allowed=False,
                log_message="Action blocked: no shared template selected.",
                warning_title="Quick Action Test",
                warning_message="Select a template in Image Recognition Test first.",
            )
        if command == "click_last_match" and last_match is None:
            return QuickActionValidation(
                allowed=False,
                log_message="Action blocked: no successful match is available.",
                warning_title="Quick Action Test",
                warning_message="Run a successful match first.",
            )
        return QuickActionValidation(allowed=True)

    def run_quick_action(
        self,
        *,
        instance_id: int,
        command: str,
        parameters: Mapping[str, object],
    ) -> dict[str, object]:
        instance = self.instances.get(instance_id)
        if instance is None or instance.instance_index is None:
            return self._missing_instance_result(command)

        name = instance.instance_name or instance.name
        engine = self.action_engine_factory(
            self.adb_manager,
            instance.instance_index,
            name,
        )
        result = self._execute_quick_action(engine, command, parameters)
        command_text = self.generated_adb_command(
            instance.instance_index,
            command,
            result,
            parameters,
        )
        return {
            "action": command,
            "instance": name,
            "adb_command": command_text,
            **result,
        }

    def action_parameters(
        self,
        *,
        template_path: Path | None,
        threshold: float,
        timeout_seconds: float,
        retry_interval_seconds: float,
        x: int,
        y: int,
        swipe_x1: int,
        swipe_y1: int,
        swipe_x2: int,
        swipe_y2: int,
        swipe_duration_ms: int,
        last_match: Mapping[str, object] | None,
    ) -> dict[str, object]:
        match = last_match or {}
        return {
            "template_path": template_path,
            "threshold": threshold,
            "timeout_seconds": timeout_seconds,
            "retry_interval_seconds": retry_interval_seconds,
            "x": x,
            "y": y,
            "swipe_x1": swipe_x1,
            "swipe_y1": swipe_y1,
            "swipe_x2": swipe_x2,
            "swipe_y2": swipe_y2,
            "swipe_duration_ms": swipe_duration_ms,
            "last_match_x": self.int_value(match.get("center_x"), -1),
            "last_match_y": self.int_value(match.get("center_y"), -1),
        }

    def template_match_view(self, result: Mapping[str, object]) -> TemplateMatchView:
        found = bool(result.get("found"))
        confidence = float(result.get("confidence", 0.0) or 0.0)
        x = self.int_value(result.get("x"), -1)
        y = self.int_value(result.get("y"), -1)
        width = self.int_value(result.get("width"), 0) if found else 0
        height = self.int_value(result.get("height"), 0) if found else 0
        center_x = self.int_value(result.get("center_x"), -1) if found else -1
        center_y = self.int_value(result.get("center_y"), -1) if found else -1
        return TemplateMatchView(
            found=found,
            status="FOUND" if found else "NOT FOUND",
            confidence=confidence,
            x=x,
            y=y,
            width=width,
            height=height,
            center_x=center_x,
            center_y=center_y,
            screenshot=str(result.get("screenshot", "") or ""),
            template=str(result.get("template", "") or ""),
            position_text=f"({x}, {y})" if found else "-",
            size_text=f"{width} x {height}" if found else "-",
            center_text=f"({center_x}, {center_y})" if found else "-",
            coordinates_text=f"Coordinates: ({x}, {y})" if found else "Coordinates: -",
        )

    def quick_action_result_view(
        self,
        result: Mapping[str, object],
    ) -> QuickActionResultView:
        success = bool(result.get("success"))
        confidence = float(result.get("confidence", 0.0) or 0.0)
        x = self.int_value(result.get("x"), -1)
        y = self.int_value(result.get("y"), -1)
        elapsed = float(result.get("elapsed_time", 0.0) or 0.0)
        message = str(result.get("message", "") or "")
        adb_command = str(result.get("adb_command", "") or "")
        status = "SUCCESS" if success else "FAILED"
        suffix = f" | {message}" if message else ""
        action = str(result.get("action", "Action") or "Action")
        return QuickActionResultView(
            success=success,
            status=status,
            status_kind="success" if success else "error",
            confidence=confidence,
            x=x,
            y=y,
            elapsed_time=elapsed,
            message=message,
            adb_command=adb_command,
            coordinates_text=f"({x}, {y})" if x >= 0 and y >= 0 else "-",
            confidence_text=(
                f"{confidence:.4f}"
                if action in {"click_template", "wait_for_template"}
                else "-"
            ),
            elapsed_text=f"{elapsed:.2f}s",
            message_text=message or "-",
            adb_command_text=adb_command or "-",
            result_summary=(
                f"Result: {status} | Confidence: {confidence:.4f} | "
                f"Coordinates: ({x}, {y}) | Elapsed: {elapsed:.2f}s{suffix}"
            ),
            log_summary=(
                f"{action} completed with {status} in {elapsed:.2f}s at ({x}, {y})."
            ),
        )

    @staticmethod
    def generated_adb_command(
        instance_index: int,
        command: str,
        result: Mapping[str, object],
        parameters: Mapping[str, object],
    ) -> str:
        if command in {"click_template", "click_coordinates", "click_last_match"}:
            x = int(result.get("x", -1))
            y = int(result.get("y", -1))
            if x >= 0 and y >= 0:
                return f"memuc adb -i {instance_index} shell input tap {x} {y}"
        if command == "swipe_coordinates":
            return (
                f"memuc adb -i {instance_index} shell input swipe "
                f"{parameters['swipe_x1']} {parameters['swipe_y1']} "
                f"{parameters['swipe_x2']} {parameters['swipe_y2']} "
                f"{parameters['swipe_duration_ms']}"
            )
        return ""

    @staticmethod
    def int_value(value: object, default: int) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _template_size(template_path: Path) -> tuple[int, int]:
        image = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if image is None:
            return 0, 0
        height, width = image.shape[:2]
        return int(width), int(height)

    @staticmethod
    def _missing_instance_result(command: str) -> dict[str, object]:
        return {
            "action": command,
            "instance": "",
            "success": False,
            "confidence": 0.0,
            "x": -1,
            "y": -1,
            "elapsed_time": 0.0,
            "message": "No MEmu instance selected.",
        }

    @staticmethod
    def _execute_quick_action(
        engine: object,
        command: str,
        parameters: Mapping[str, object],
    ) -> dict[str, object]:
        template_path = parameters.get("template_path")
        if command == "wait_for_template":
            return engine.wait_for_template(
                Path(str(template_path)),
                threshold=float(parameters["threshold"]),
                timeout_seconds=float(parameters["timeout_seconds"]),
                retry_interval_seconds=float(parameters["retry_interval_seconds"]),
            )
        if command == "click_template":
            return engine.click_template(
                Path(str(template_path)),
                threshold=float(parameters["threshold"]),
            )
        if command == "click_coordinates":
            return engine.click_coordinates(
                int(parameters["x"]),
                int(parameters["y"]),
            )
        if command == "swipe_coordinates":
            return engine.swipe_coordinates(
                int(parameters["swipe_x1"]),
                int(parameters["swipe_y1"]),
                int(parameters["swipe_x2"]),
                int(parameters["swipe_y2"]),
                int(parameters["swipe_duration_ms"]),
            )
        if command == "click_last_match":
            return engine.click_coordinates(
                int(parameters["last_match_x"]),
                int(parameters["last_match_y"]),
            )
        return {
            "success": False,
            "confidence": 0.0,
            "x": -1,
            "y": -1,
            "elapsed_time": 0.0,
            "message": f"Unknown action: {command}",
        }
