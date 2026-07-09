from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, QRect, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rok_assistant.action_engine import ActionEngine
from rok_assistant.application.automation import AutomationViewModel
from rok_assistant.app import AppContext
from rok_assistant.db.models import Instance
from rok_assistant.gui.style import preview_surface_qss, status_badge_qss
from rok_assistant.gui.template_capture import TemplateCaptureDialog
from rok_assistant.gui.widgets import StatusBadge, apply_button_variant
from rok_assistant.paths import SCREENSHOT_DIR, TEMPLATE_DIR
from rok_assistant.vision import find_template


class AutomationOperationWorker(QObject):
    finished = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)

    def __init__(self, action: str, callback: Callable[[], object]):
        super().__init__()
        self.action = action
        self.callback = callback

    def run(self) -> None:
        try:
            result = self.callback()
        except Exception as exc:
            self.failed.emit(self.action, str(exc))
            return
        self.finished.emit(self.action, result)


class ImagePreviewLabel(QLabel):
    def __init__(self, empty_text: str) -> None:
        super().__init__(empty_text)
        self.empty_text = empty_text
        self.source_pixmap = QPixmap()
        self.match_rect: QRect | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 180)
        self.setStyleSheet(preview_surface_qss())

    def load_image(self, path: str | Path, match_rect: QRect | None = None) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.clear_image()
            return False
        self.source_pixmap = pixmap
        self.match_rect = match_rect
        self._update_display()
        return True

    def clear_image(self) -> None:
        self.source_pixmap = QPixmap()
        self.match_rect = None
        self.clear()
        self.setText(self.empty_text)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_display()

    def _update_display(self) -> None:
        if self.source_pixmap.isNull():
            return
        rendered = self.source_pixmap.copy()
        if self.match_rect is not None and not self.match_rect.isEmpty():
            painter = QPainter(rendered)
            painter.setPen(QPen(Qt.GlobalColor.red, 3))
            painter.drawRect(self.match_rect)
            painter.end()
        self.setPixmap(
            rendered.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class AutomationWidget(QWidget):
    open_instances_requested = pyqtSignal()

    def __init__(self, context: AppContext):
        super().__init__()
        self.context = context
        self.view_model = AutomationViewModel(
            context.instances,
            context.memu_adb_manager,
            action_engine_factory=lambda adb_manager, instance_index, instance_name: ActionEngine(
                adb_manager,
                instance_index,
                instance_name,
            ),
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.latest_screenshot_path: Path | None = None
        self.selected_template_path: Path | None = None
        self.last_match: dict[str, object] | None = None
        self._selected_instance_id: int | None = None
        self._syncing_instance_selectors = False
        self._workers: list[tuple[QThread, AutomationOperationWorker]] = []
        self._busy_count = 0

        self.instance_combo = QComboBox()
        self.instance_combo.setMinimumContentsLength(24)
        self.refresh_targets_button = QPushButton("Refresh")
        self.open_instances_button = QPushButton("Open Instances")
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0.0, 1.0)
        self.threshold_input.setSingleStep(0.05)
        self.threshold_input.setDecimals(2)
        self.threshold_input.setValue(0.70)
        self.browse_template_button = QPushButton("Browse Template")
        self.capture_screenshot_button = QPushButton("Capture Screenshot")
        self.run_match_button = QPushButton("Run Match")
        self.open_template_capture_button = QPushButton("Open Template Capture")
        self.open_screenshot_folder_button = QPushButton("Open Screenshot Folder")
        for button in (
            self.refresh_targets_button,
            self.open_instances_button,
            self.open_template_capture_button,
            self.open_screenshot_folder_button,
        ):
            apply_button_variant(button, "secondary")
        self.template_preview = ImagePreviewLabel("No template selected")
        self.screenshot_preview = ImagePreviewLabel("No screenshot captured")

        # Compatibility aliases for callers of the original panel.
        self.select_template_button = self.browse_template_button
        self.image_capture_button = self.capture_screenshot_button
        self.find_template_button = self.run_match_button
        self.template_label = QLabel("Template: -")
        self.match_confidence_label = QLabel("Confidence: -")
        self.match_coordinates_label = QLabel("Coordinates: -")

        self.result_status_label = StatusBadge("-", "neutral")
        self.result_confidence_label = QLabel("-")
        self.result_position_label = QLabel("-")
        self.result_size_label = QLabel("-")
        self.result_center_label = QLabel("-")
        self.result_x_label = QLabel("x: -")
        self.result_y_label = QLabel("y: -")
        self.result_width_label = QLabel("width: -")
        self.result_height_label = QLabel("height: -")
        self.result_center_x_label = QLabel("center x: -")
        self.result_center_y_label = QLabel("center y: -")
        self.result_screenshot_path_label = QLabel("-")
        self.result_template_path_label = QLabel("-")
        self.result_message_label = QLabel("")
        self.result_message_label.setWordWrap(True)
        self.result_screenshot_path_label.setWordWrap(True)
        self.result_template_path_label.setWordWrap(True)

        self.quick_instance_combo = QComboBox()
        self.quick_instance_combo.setMinimumContentsLength(24)
        self.quick_refresh_instances_button = QPushButton("Refresh")
        self.quick_open_instances_button = QPushButton("Open Instances")
        apply_button_variant(self.quick_refresh_instances_button, "secondary")
        apply_button_variant(self.quick_open_instances_button, "secondary")
        self.quick_action_combo = QComboBox()
        for label, command in (
            ("Click Last Match", "click_last_match"),
            ("Click Matched Template Center", "click_template"),
            ("Wait For Selected Template", "wait_for_template"),
            ("Click Coordinates", "click_coordinates"),
            ("Swipe Coordinates", "swipe_coordinates"),
        ):
            self.quick_action_combo.addItem(label, command)
        self.quick_run_button = QPushButton("Run Once")

        self.quick_threshold_input = QDoubleSpinBox()
        self.quick_threshold_input.setRange(0.0, 1.0)
        self.quick_threshold_input.setSingleStep(0.05)
        self.quick_threshold_input.setDecimals(2)
        self.quick_threshold_input.setValue(self.threshold_input.value())
        self.quick_timeout_input = QDoubleSpinBox()
        self.quick_timeout_input.setRange(0.1, 300.0)
        self.quick_timeout_input.setDecimals(1)
        self.quick_timeout_input.setSuffix(" s")
        self.quick_timeout_input.setValue(10.0)
        self.quick_retry_input = QDoubleSpinBox()
        self.quick_retry_input.setRange(0.1, 60.0)
        self.quick_retry_input.setDecimals(1)
        self.quick_retry_input.setSuffix(" s")
        self.quick_retry_input.setValue(1.0)

        self.quick_x_input = QSpinBox()
        self.quick_x_input.setRange(0, 10000)
        self.quick_x_input.setValue(540)
        self.quick_y_input = QSpinBox()
        self.quick_y_input.setRange(0, 10000)
        self.quick_y_input.setValue(960)
        self.quick_swipe_x1_input = QSpinBox()
        self.quick_swipe_x1_input.setRange(0, 10000)
        self.quick_swipe_x1_input.setValue(540)
        self.quick_swipe_y1_input = QSpinBox()
        self.quick_swipe_y1_input.setRange(0, 10000)
        self.quick_swipe_y1_input.setValue(1500)
        self.quick_swipe_x2_input = QSpinBox()
        self.quick_swipe_x2_input.setRange(0, 10000)
        self.quick_swipe_x2_input.setValue(540)
        self.quick_swipe_y2_input = QSpinBox()
        self.quick_swipe_y2_input.setRange(0, 10000)
        self.quick_swipe_y2_input.setValue(600)
        self.quick_swipe_duration_input = QSpinBox()
        self.quick_swipe_duration_input.setRange(0, 60000)
        self.quick_swipe_duration_input.setSuffix(" ms")
        self.quick_swipe_duration_input.setValue(500)

        self.quick_template_value = QLabel("-")
        self.quick_last_match_value = QLabel("Unavailable")
        self.quick_template_value.setWordWrap(True)
        self.quick_status_label = StatusBadge("-", "neutral")
        self.quick_elapsed_label = QLabel("-")
        self.quick_coordinates_label = QLabel("-")
        self.quick_confidence_label = QLabel("-")
        self.quick_message_label = QLabel("-")
        self.quick_command_label = QLabel("-")
        self.quick_message_label.setWordWrap(True)
        self.quick_command_label.setWordWrap(True)
        self.quick_execution_log = QPlainTextEdit()
        self.quick_execution_log.setReadOnly(True)
        self.quick_execution_log.setPlaceholderText("Quick action execution events appear here.")
        self.quick_execution_log.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.action_result_label = QLabel("Result: -")
        self.action_result_label.hide()

        image_panel = QGroupBox("Image Recognition Test")
        image_layout = QVBoxLayout(image_panel)
        instance_controls = QHBoxLayout()
        instance_controls.addWidget(QLabel("Instance"))
        instance_controls.addWidget(self.instance_combo, 1)
        instance_controls.addWidget(self.refresh_targets_button)
        instance_controls.addWidget(self.open_instances_button)
        image_layout.addLayout(instance_controls)

        match_controls = QHBoxLayout()
        match_controls.addWidget(QLabel("Threshold"))
        match_controls.addWidget(self.threshold_input)
        match_controls.addWidget(self.browse_template_button)
        match_controls.addWidget(self.capture_screenshot_button)
        match_controls.addWidget(self.run_match_button)
        match_controls.addWidget(self.open_template_capture_button)
        match_controls.addStretch(1)
        image_layout.addLayout(match_controls)

        template_preview_group = QGroupBox("Template Preview")
        template_preview_layout = QVBoxLayout(template_preview_group)
        template_preview_layout.addWidget(self.template_preview, 1)

        screenshot_preview_group = QGroupBox("Screenshot Preview")
        screenshot_preview_layout = QVBoxLayout(screenshot_preview_group)
        screenshot_preview_layout.addWidget(self.screenshot_preview, 1)

        self.image_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.image_splitter.addWidget(template_preview_group)
        self.image_splitter.addWidget(screenshot_preview_group)
        self.image_splitter.setStretchFactor(0, 30)
        self.image_splitter.setStretchFactor(1, 70)
        self.image_splitter.setSizes([340, 800])
        self.image_splitter.setChildrenCollapsible(False)
        self.image_splitter.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        image_layout.addWidget(self.image_splitter, 1)

        match_result_group = QGroupBox("Match Result")
        match_result_layout = QFormLayout(match_result_group)
        match_summary = QHBoxLayout()
        match_summary.addWidget(QLabel("Status"))
        match_summary.addWidget(self.result_status_label)
        match_summary.addSpacing(18)
        match_summary.addWidget(QLabel("Confidence"))
        match_summary.addWidget(self.result_confidence_label)
        match_summary.addStretch(1)
        match_result_layout.addRow(match_summary)
        match_result_layout.addRow("Position", self.result_position_label)
        match_result_layout.addRow("Size", self.result_size_label)
        match_result_layout.addRow("Center", self.result_center_label)
        match_result_layout.addRow("Screenshot path", self.result_screenshot_path_label)
        match_result_layout.addRow("Template path", self.result_template_path_label)
        match_result_layout.addRow("Message", self.result_message_label)
        match_result_layout.addRow(self.open_screenshot_folder_button)
        image_layout.addWidget(match_result_group)

        self.quick_action_panel = QWidget()
        action_layout = QVBoxLayout(self.quick_action_panel)
        quick_header = QHBoxLayout()
        quick_header.addWidget(QLabel("Instance"))
        quick_header.addWidget(self.quick_instance_combo, 1)
        quick_header.addWidget(self.quick_refresh_instances_button)
        quick_header.addWidget(self.quick_open_instances_button)
        action_layout.addLayout(quick_header)

        self.quick_form = QFormLayout()
        self.quick_form.addRow("Quick Action", self.quick_action_combo)
        self.quick_form.addRow("Threshold", self.quick_threshold_input)
        self.quick_form.addRow("Timeout", self.quick_timeout_input)
        self.quick_form.addRow("Retry Interval", self.quick_retry_input)

        self.quick_click_coordinates_widget = self._coordinate_pair_widget(
            self.quick_x_input,
            self.quick_y_input,
            "X",
            "Y",
        )
        self.quick_swipe_start_widget = self._coordinate_pair_widget(
            self.quick_swipe_x1_input,
            self.quick_swipe_y1_input,
            "X1",
            "Y1",
        )
        self.quick_swipe_end_widget = self._coordinate_pair_widget(
            self.quick_swipe_x2_input,
            self.quick_swipe_y2_input,
            "X2",
            "Y2",
        )
        self.quick_form.addRow("Coordinates", self.quick_click_coordinates_widget)
        self.quick_form.addRow("Swipe Start", self.quick_swipe_start_widget)
        self.quick_form.addRow("Swipe End", self.quick_swipe_end_widget)
        self.quick_form.addRow("Swipe Duration", self.quick_swipe_duration_input)
        action_layout.addLayout(self.quick_form)

        quick_context_group = QGroupBox("Shared Test Context")
        quick_context_layout = QFormLayout(quick_context_group)
        quick_context_layout.addRow("Current template", self.quick_template_value)
        quick_context_layout.addRow("Last match", self.quick_last_match_value)
        action_layout.addWidget(quick_context_group)

        run_row = QHBoxLayout()
        run_row.addWidget(self.quick_run_button)
        run_row.addStretch(1)
        action_layout.addLayout(run_row)

        quick_result_group = QGroupBox("Quick Action Result")
        quick_result_layout = QFormLayout(quick_result_group)
        quick_result_layout.addRow("Status", self.quick_status_label)
        quick_result_layout.addRow("Elapsed time", self.quick_elapsed_label)
        quick_result_layout.addRow("Coordinates", self.quick_coordinates_label)
        quick_result_layout.addRow("Confidence", self.quick_confidence_label)
        quick_result_layout.addRow("ADB command", self.quick_command_label)
        quick_result_layout.addRow("Error", self.quick_message_label)

        quick_output_splitter = QSplitter(Qt.Orientation.Vertical)
        quick_output_splitter.addWidget(quick_result_group)
        quick_output_splitter.addWidget(self.quick_execution_log)
        quick_output_splitter.setStretchFactor(0, 0)
        quick_output_splitter.setStretchFactor(1, 1)
        quick_output_splitter.setSizes([180, 260])
        quick_output_splitter.setChildrenCollapsible(False)
        action_layout.addWidget(quick_output_splitter, 1)

        self.test_tabs = QTabWidget()
        self.test_tabs.addTab(image_panel, "Image Recognition Test")
        self.test_tabs.addTab(self.quick_action_panel, "Quick Action Test")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.test_tabs, 1)

        self.refresh_targets_button.clicked.connect(self.refresh)
        self.quick_refresh_instances_button.clicked.connect(self.refresh)
        self.open_instances_button.clicked.connect(self.open_instances_requested.emit)
        self.quick_open_instances_button.clicked.connect(self.open_instances_requested.emit)
        self.instance_combo.currentIndexChanged.connect(
            lambda _index: self._instance_selector_changed(self.instance_combo)
        )
        self.quick_instance_combo.currentIndexChanged.connect(
            lambda _index: self._instance_selector_changed(self.quick_instance_combo)
        )
        self.select_template_button.clicked.connect(self.select_template)
        self.image_capture_button.clicked.connect(self.capture_screenshot)
        self.find_template_button.clicked.connect(self.find_template_in_latest_screenshot)
        self.open_template_capture_button.clicked.connect(self.open_template_capture)
        self.open_screenshot_folder_button.clicked.connect(self.open_screenshot_folder)
        self.quick_action_combo.currentIndexChanged.connect(self._update_quick_action_fields)
        self.quick_run_button.clicked.connect(self.run_quick_action)
        self.threshold_input.valueChanged.connect(self.quick_threshold_input.setValue)
        self.refresh()
        self._update_run_match_enabled()
        self._update_quick_action_fields()

    def refresh(self) -> None:
        rows = self.view_model.list_target_rows()
        selected_id = self._selected_instance_id
        self._syncing_instance_selectors = True
        try:
            for combo in (self.instance_combo, self.quick_instance_combo):
                combo.clear()
                for row in rows:
                    combo.addItem(row.label, row.id)
            if selected_id is None and self.instance_combo.count():
                selected_id = int(self.instance_combo.itemData(0))
            self._selected_instance_id = selected_id
            self._sync_instance_selectors()
        finally:
            self._syncing_instance_selectors = False
        self._update_quick_action_enabled()

    def _instance_selector_changed(self, source: QComboBox) -> None:
        if self._syncing_instance_selectors:
            return
        instance_id = source.currentData()
        self._selected_instance_id = int(instance_id) if instance_id is not None else None
        self._syncing_instance_selectors = True
        try:
            self._sync_instance_selectors()
        finally:
            self._syncing_instance_selectors = False
        self._update_quick_action_enabled()

    def _sync_instance_selectors(self) -> None:
        for combo in (self.instance_combo, self.quick_instance_combo):
            index = combo.findData(self._selected_instance_id)
            combo.setCurrentIndex(index)

    def capture_screenshot(self) -> None:
        instance_id = self._selected_instance_id
        if instance_id is None:
            QMessageBox.warning(self, "Image Recognition", "Select a MEmu instance first.")
            return
        self.logger.info(
            "[MEmu][ADB] Image recognition screenshot requested for instance id %s",
            instance_id,
        )
        self._run_background(
            "capture screenshot",
            lambda: self._capture_screenshots([int(instance_id)]),
            self._handle_screenshot_result,
        )

    def select_template(self) -> None:
        TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Browse Template",
            str(TEMPLATE_DIR),
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
        )
        if not path:
            return
        self.selected_template_path = Path(path)
        self._set_template_labels()
        self.template_preview.load_image(self.selected_template_path)
        self._clear_match_result()
        self._update_run_match_enabled()
        self.logger.info("[ImageMatch] Selected template: %s", self.selected_template_path)

    def find_template_in_latest_screenshot(self) -> None:
        if self.latest_screenshot_path is None or self.selected_template_path is None:
            QMessageBox.warning(
                self,
                "Image Matching",
                "Select a template and capture a screenshot first.",
            )
            return
        screenshot_path = self.latest_screenshot_path
        template_path = self.selected_template_path
        threshold = self.threshold_input.value()
        self.logger.info(
            "[ImageMatch] Find template requested: screenshot=%s template=%s",
            screenshot_path,
            template_path,
        )
        self._run_background(
            "run match",
            lambda: self._find_template(screenshot_path, template_path, threshold),
            self._handle_template_result,
        )

    def open_template_capture(self) -> None:
        dialog = TemplateCaptureDialog(
            self.context,
            screenshot_path=self.latest_screenshot_path,
            parent=self,
        )
        dialog.template_saved.connect(self._select_saved_template)
        dialog.exec()

    def open_screenshot_folder(self) -> None:
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(SCREENSHOT_DIR.resolve())))

    def _select_saved_template(self, path: str) -> None:
        self.selected_template_path = Path(path)
        self._set_template_labels()
        self.template_preview.load_image(path)
        self._clear_match_result()
        self._update_run_match_enabled()

    def wait_for_template_action(self) -> None:
        self._select_and_run_quick_action("wait_for_template")

    def click_template_action(self) -> None:
        self._select_and_run_quick_action("click_template")

    def click_coordinates_action(self) -> None:
        self._select_and_run_quick_action("click_coordinates")

    def swipe_coordinates_action(self) -> None:
        self._select_and_run_quick_action("swipe_coordinates")

    def selected_instances(self) -> list[Instance]:
        instance = self._selected_action_instance()
        return [instance] if instance is not None else []

    def _selected_action_instance(self) -> Instance | None:
        if self._selected_instance_id is None:
            return None
        return self.view_model.get_instance(self._selected_instance_id)

    def run_quick_action(self) -> None:
        command = str(self.quick_action_combo.currentData() or "")
        self._run_action_test(command)

    def _select_and_run_quick_action(self, command: str) -> None:
        index = self.quick_action_combo.findData(command)
        if index >= 0:
            self.quick_action_combo.setCurrentIndex(index)
        self.run_quick_action()

    def _run_action_test(self, command: str) -> None:
        instance = self._selected_action_instance()
        validation = self.view_model.validate_quick_action(
            command=command,
            selected_instance_id=self._selected_instance_id,
            selected_template_path=self.selected_template_path,
            last_match=self.last_match,
        )
        if not validation.allowed:
            self._append_quick_log(validation.log_message)
            QMessageBox.warning(
                self,
                validation.warning_title,
                validation.warning_message,
            )
            return
        assert instance is not None and instance.id is not None
        parameters = self._action_parameters()
        action_name = self.quick_action_combo.currentText()
        self._append_quick_log(
            f"Starting {action_name} on {instance.instance_name or instance.name}."
        )
        self.logger.info(
            "[Action] %s requested for instance %s",
            command,
            instance.instance_name or instance.name,
        )
        self._run_background(
            f"quick action {command}",
            lambda: self._run_action_test_instance(instance.id, command, parameters),
            self._handle_action_result,
        )

    def _capture_screenshots(self, instance_ids: list[int]) -> dict[str, object]:
        return self.view_model.capture_screenshots(instance_ids)

    def _find_template(
        self,
        screenshot_path: Path,
        template_path: Path,
        threshold: float | None = None,
    ) -> dict[str, object]:
        effective_threshold = self.threshold_input.value() if threshold is None else threshold
        return self.view_model.find_template(
            screenshot_path,
            template_path,
            effective_threshold,
            matcher=find_template,
        )

    def _run_action_test_instance(
        self,
        instance_id: int,
        command: str,
        parameters: dict[str, object],
    ) -> dict[str, object]:
        return self.view_model.run_quick_action(
            instance_id=instance_id,
            command=command,
            parameters=parameters,
        )

    def _handle_control_result(self, result: object) -> None:
        self.refresh()
        data = result if isinstance(result, dict) else {}
        failures = [
            item["name"]
            for item in data.get("results", [])
            if isinstance(item, dict) and not item.get("success")
        ]
        if failures:
            QMessageBox.warning(
                self,
                "Automation",
                "Some automation actions failed:\n" + "\n".join(str(name) for name in failures),
            )

    def _handle_screenshot_result(self, result: object) -> None:
        self.refresh()
        data = result if isinstance(result, dict) else {}
        results = [
            item
            for item in data.get("results", [])
            if isinstance(item, dict)
        ]
        failures = [item["name"] for item in results if not item.get("success")]
        for item in results:
            if item.get("success"):
                path = Path(str(item.get("path")))
                self.latest_screenshot_path = path
                self.screenshot_preview.load_image(path)
                self._clear_match_result()
                self._update_run_match_enabled()
                self.logger.info(
                    "[MEmu][ADB] Screenshot capture result for %s: %s",
                    item.get("name"),
                    item.get("path"),
                )
        if failures:
            QMessageBox.warning(
                self,
                "Automation",
                "Some screenshots failed:\n" + "\n".join(str(name) for name in failures),
            )

    def _handle_template_result(self, result: object) -> None:
        data = result if isinstance(result, dict) else {}
        view = self.view_model.template_match_view(data)
        self.match_confidence_label.setText(f"Confidence: {view.confidence:.4f}")
        self.match_coordinates_label.setText(view.coordinates_text)
        self._set_match_status(view.status)
        self.result_confidence_label.setText(f"{view.confidence:.4f}")
        self.result_screenshot_path_label.setText(view.screenshot or "-")
        self.result_template_path_label.setText(view.template or "-")
        self.result_message_label.setText("")
        if view.found:
            self.result_position_label.setText(view.position_text)
            self.result_size_label.setText(f"{view.width} × {view.height}")
            self.result_center_label.setText(view.center_text)
            self.result_x_label.setText(f"x: {view.x}")
            self.result_y_label.setText(f"y: {view.y}")
            self.result_width_label.setText(f"width: {view.width}")
            self.result_height_label.setText(f"height: {view.height}")
            self.result_center_x_label.setText(f"center x: {view.center_x}")
            self.result_center_y_label.setText(f"center y: {view.center_y}")
            self.screenshot_preview.load_image(
                view.screenshot,
                QRect(view.x, view.y, view.width, view.height),
            )
            self.last_match = view.last_match
        else:
            self.last_match = None
            self._clear_result_coordinates()
            self.screenshot_preview.load_image(view.screenshot)
        self._update_last_match_state()
        self.logger.info(
            "[ImageMatch] Result found=%s confidence=%.4f x=%s y=%s screenshot=%s template=%s",
            view.found,
            view.confidence,
            view.x,
            view.y,
            view.screenshot,
            view.template,
        )

    def _handle_action_result(self, result: object) -> None:
        data = result if isinstance(result, dict) else {}
        view = self.view_model.quick_action_result_view(data)
        self.action_result_label.setText(view.result_summary)
        self.quick_status_label.setText(view.status)
        self._set_status_style(
            self.quick_status_label,
            view.status_kind,
        )
        self.quick_elapsed_label.setText(view.elapsed_text)
        self.quick_coordinates_label.setText(view.coordinates_text)
        self.quick_confidence_label.setText(view.confidence_text)
        self.quick_message_label.setText(view.message_text)
        self.quick_command_label.setText(view.adb_command_text)
        self._append_quick_log(view.log_summary)
        if view.adb_command:
            self._append_quick_log(f"ADB command: {view.adb_command}")
        if view.message:
            self._append_quick_log(f"Error: {view.message}")
        self.logger.info(
            "[Action] GUI result action=%s instance=%s success=%s confidence=%.4f x=%s y=%s "
            "elapsed=%.2fs message=%s",
            data.get("action", ""),
            data.get("instance", ""),
            view.success,
            view.confidence,
            view.x,
            view.y,
            view.elapsed_time,
            view.message,
        )

    def _run_background(
        self,
        action: str,
        callback: Callable[[], object],
        on_finished: Callable[[object], None],
    ) -> None:
        self._set_busy(True)
        thread = QThread(self)
        worker = AutomationOperationWorker(action, callback)
        worker.moveToThread(thread)
        self._workers.append((thread, worker))

        thread.started.connect(worker.run)
        worker.finished.connect(lambda _action, result: on_finished(result))
        worker.failed.connect(self._handle_worker_failure)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda: self._cleanup_worker(thread))
        thread.start()

    def _handle_worker_failure(self, action: str, message: str) -> None:
        self.logger.error("[Automation] %s failed: %s", action, message)
        if action == "run match":
            self._set_match_status("ERROR")
            self.result_confidence_label.setText("-")
            self.result_message_label.setText(message)
            self._clear_result_coordinates()
            self.match_coordinates_label.setText("Coordinates: -")
            self.last_match = None
            self._update_last_match_state()
        elif action.startswith("quick action"):
            self.quick_status_label.setText("FAILED")
            self._set_status_style(self.quick_status_label, "error")
            self.quick_message_label.setText(message)
            self._append_quick_log(f"Action failed: {message}")
        QMessageBox.warning(self, "Automation", f"{action} failed:\n{message}")

    def _cleanup_worker(self, thread: QThread) -> None:
        self._workers = [
            (existing_thread, worker)
            for existing_thread, worker in self._workers
            if existing_thread is not thread
        ]
        thread.deleteLater()
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self._busy_count += 1 if busy else -1
        self._busy_count = max(0, self._busy_count)
        enabled = self._busy_count == 0
        for button in (
            self.refresh_targets_button,
            self.quick_refresh_instances_button,
            self.open_instances_button,
            self.quick_open_instances_button,
            self.select_template_button,
            self.image_capture_button,
            self.find_template_button,
            self.open_template_capture_button,
            self.quick_run_button,
        ):
            button.setEnabled(enabled)
        self._update_run_match_enabled()
        self._update_quick_action_enabled()

    def _action_parameters(self) -> dict[str, object]:
        return self.view_model.action_parameters(
            template_path=self.selected_template_path,
            threshold=self.quick_threshold_input.value(),
            timeout_seconds=self.quick_timeout_input.value(),
            retry_interval_seconds=self.quick_retry_input.value(),
            x=self.quick_x_input.value(),
            y=self.quick_y_input.value(),
            swipe_x1=self.quick_swipe_x1_input.value(),
            swipe_y1=self.quick_swipe_y1_input.value(),
            swipe_x2=self.quick_swipe_x2_input.value(),
            swipe_y2=self.quick_swipe_y2_input.value(),
            swipe_duration_ms=self.quick_swipe_duration_input.value(),
            last_match=self.last_match,
        )

    def _set_template_labels(self) -> None:
        text = f"Template: {self.selected_template_path}" if self.selected_template_path else "Template: -"
        self.template_label.setText(text)
        self.quick_template_value.setText(
            str(self.selected_template_path)
            if self.selected_template_path
            else "-"
        )

    def _clear_result_coordinates(self) -> None:
        self.result_position_label.setText("-")
        self.result_size_label.setText("-")
        self.result_center_label.setText("-")
        self.result_x_label.setText("x: -")
        self.result_y_label.setText("y: -")
        self.result_width_label.setText("width: -")
        self.result_height_label.setText("height: -")
        self.result_center_x_label.setText("center x: -")
        self.result_center_y_label.setText("center y: -")

    def _clear_match_result(self) -> None:
        self.last_match = None
        self._set_match_status("-")
        self.result_confidence_label.setText("-")
        self.result_screenshot_path_label.setText("-")
        self.result_template_path_label.setText("-")
        self.result_message_label.setText("")
        self.match_confidence_label.setText("Confidence: -")
        self.match_coordinates_label.setText("Coordinates: -")
        self._clear_result_coordinates()
        self._update_last_match_state()

    def _update_run_match_enabled(self) -> None:
        ready = (
            self._busy_count == 0
            and self.selected_template_path is not None
            and self.selected_template_path.exists()
            and self.latest_screenshot_path is not None
            and self.latest_screenshot_path.exists()
        )
        self.run_match_button.setEnabled(ready)
        self._update_quick_action_enabled()

    def _update_last_match_state(self) -> None:
        if self.last_match is None:
            self.quick_last_match_value.setText("Unavailable")
        else:
            self.quick_last_match_value.setText(
                f"Position ({self.last_match['x']}, {self.last_match['y']}), "
                f"center ({self.last_match['center_x']}, {self.last_match['center_y']})"
            )
        self._update_quick_action_enabled()

    def _update_quick_action_enabled(self) -> None:
        last_match_index = self.quick_action_combo.findData("click_last_match")
        if last_match_index >= 0:
            item = self.quick_action_combo.model().item(last_match_index)
            if item is not None:
                item.setEnabled(self.last_match is not None)
        if (
            self.last_match is None
            and self.quick_action_combo.currentData() == "click_last_match"
        ):
            fallback_index = self.quick_action_combo.findData("click_template")
            self.quick_action_combo.setCurrentIndex(fallback_index)
        command_ready = (
            self.quick_action_combo.currentData() != "click_last_match"
            or self.last_match is not None
        )
        self.quick_run_button.setEnabled(
            self._busy_count == 0
            and self._selected_instance_id is not None
            and command_ready
        )

    def _update_quick_action_fields(self) -> None:
        command = str(self.quick_action_combo.currentData() or "")
        visibility = {
            self.quick_threshold_input: command in {"click_template", "wait_for_template"},
            self.quick_timeout_input: command == "wait_for_template",
            self.quick_retry_input: command == "wait_for_template",
            self.quick_click_coordinates_widget: command == "click_coordinates",
            self.quick_swipe_start_widget: command == "swipe_coordinates",
            self.quick_swipe_end_widget: command == "swipe_coordinates",
            self.quick_swipe_duration_input: command == "swipe_coordinates",
        }
        for widget, visible in visibility.items():
            self._set_quick_field_visible(widget, visible)
        self._update_quick_action_enabled()

    def _set_match_status(self, status: str) -> None:
        self.result_status_label.setText(status)
        style = {
            "FOUND": "success",
            "NOT FOUND": "warning",
            "ERROR": "error",
        }.get(status, "neutral")
        self._set_status_style(self.result_status_label, style)

    @staticmethod
    def _set_status_style(label: QLabel, style: str) -> None:
        if isinstance(label, StatusBadge):
            label.set_status(label.text(), style)
            return
        label.setStyleSheet(status_badge_qss(style))

    def _append_quick_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.quick_execution_log.appendPlainText(f"[{timestamp}] {message}")

    def _set_quick_field_visible(self, widget: QWidget, visible: bool) -> None:
        widget.setVisible(visible)
        label = self.quick_form.labelForField(widget)
        if label is not None:
            label.setVisible(visible)

    @staticmethod
    def _coordinate_pair_widget(
        first: QSpinBox,
        second: QSpinBox,
        first_label: str,
        second_label: str,
    ) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(first_label))
        layout.addWidget(first)
        layout.addWidget(QLabel(second_label))
        layout.addWidget(second)
        layout.addStretch(1)
        return widget

    @staticmethod
    def _generated_adb_command(
        instance_index: int,
        command: str,
        result: dict[str, object],
        parameters: dict[str, object],
    ) -> str:
        return AutomationViewModel.generated_adb_command(
            instance_index,
            command,
            result,
            parameters,
        )

    @staticmethod
    def _int_value(value: object, default: int) -> int:
        return AutomationViewModel.int_value(value, default)
