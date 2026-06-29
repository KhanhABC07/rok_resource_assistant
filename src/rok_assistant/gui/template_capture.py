from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from rok_assistant.app import AppContext
from rok_assistant.paths import TEMPLATE_DIR


class CropPreviewLabel(QLabel):
    selection_changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__("Capture or browse a screenshot")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 300)
        self.setStyleSheet("QLabel { background: #171717; border: 1px solid #555; }")
        self._source_pixmap = QPixmap()
        self._display_rect = QRect()
        self._drag_start: QPoint | None = None
        self._drag_end: QPoint | None = None

    def set_image(self, path: str | Path) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return False
        self._source_pixmap = pixmap
        self._drag_start = None
        self._drag_end = None
        self._update_display()
        self.selection_changed.emit()
        return True

    def selected_source_rect(self) -> QRect:
        if (
            self._source_pixmap.isNull()
            or self._drag_start is None
            or self._drag_end is None
            or self._display_rect.isEmpty()
        ):
            return QRect()
        selected = QRect(self._drag_start, self._drag_end).normalized().intersected(
            self._display_rect
        )
        if selected.width() < 2 or selected.height() < 2:
            return QRect()
        scale_x = self._source_pixmap.width() / self._display_rect.width()
        scale_y = self._source_pixmap.height() / self._display_rect.height()
        return QRect(
            round((selected.x() - self._display_rect.x()) * scale_x),
            round((selected.y() - self._display_rect.y()) * scale_y),
            max(1, round(selected.width() * scale_x)),
            max(1, round(selected.height() * scale_y)),
        ).intersected(self._source_pixmap.rect())

    def set_source_selection(self, rect: QRect) -> None:
        if self._source_pixmap.isNull() or self._display_rect.isEmpty():
            return
        scale_x = self._display_rect.width() / self._source_pixmap.width()
        scale_y = self._display_rect.height() / self._source_pixmap.height()
        self._drag_start = QPoint(
            self._display_rect.x() + round(rect.x() * scale_x),
            self._display_rect.y() + round(rect.y() * scale_y),
        )
        self._drag_end = QPoint(
            self._display_rect.x() + round((rect.x() + rect.width()) * scale_x),
            self._display_rect.y() + round((rect.y() + rect.height()) * scale_y),
        )
        self.update()
        self.selection_changed.emit()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_display()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._display_rect.contains(event.position().toPoint())
        ):
            self._drag_start = event.position().toPoint()
            self._drag_end = self._drag_start
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_start is not None:
            self._drag_end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_start is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag_end = event.position().toPoint()
            self.update()
            self.selection_changed.emit()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if self._drag_start is None or self._drag_end is None:
            return
        painter = QPainter(self)
        painter.setPen(QPen(Qt.GlobalColor.red, 2))
        painter.drawRect(
            QRect(self._drag_start, self._drag_end).normalized().intersected(
                self._display_rect
            )
        )

    def _update_display(self) -> None:
        if self._source_pixmap.isNull():
            return
        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self._display_rect = QRect(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
            scaled.width(),
            scaled.height(),
        )


class TemplateCaptureDialog(QDialog):
    template_saved = pyqtSignal(str)

    def __init__(
        self,
        context: AppContext,
        screenshot_path: str | Path | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.context = context
        self.screenshot_path: Path | None = None
        self.setWindowTitle("Template Capture")
        self.resize(900, 620)

        self.instance_combo = QComboBox()
        self._load_instances()
        self.capture_button = QPushButton("Capture Screenshot")
        self.browse_screenshot_button = QPushButton("Browse Screenshot")
        self.preview = CropPreviewLabel()
        self.template_name_input = QLineEdit()
        self.template_name_input.setPlaceholderText("template_name")
        self.selection_label = QLabel("Selection: -")
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.save_button = self.button_box.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setEnabled(False)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("MEmu instance"))
        source_row.addWidget(self.instance_combo, 1)
        source_row.addWidget(self.capture_button)
        source_row.addWidget(self.browse_screenshot_button)

        form = QFormLayout()
        form.addRow("Template name", self.template_name_input)
        form.addRow("", self.selection_label)

        layout = QVBoxLayout(self)
        layout.addLayout(source_row)
        layout.addWidget(self.preview, 1)
        layout.addLayout(form)
        layout.addWidget(self.button_box)

        self.capture_button.clicked.connect(self.capture_screenshot)
        self.browse_screenshot_button.clicked.connect(self.browse_screenshot)
        self.preview.selection_changed.connect(self._selection_changed)
        self.template_name_input.textChanged.connect(self._update_save_enabled)
        self.button_box.accepted.connect(self.save_template)
        self.button_box.rejected.connect(self.reject)

        if screenshot_path is not None:
            self.load_screenshot(screenshot_path)

    def _load_instances(self) -> None:
        self.instance_combo.clear()
        for instance in self.context.instances.list_all():
            if instance.id is None or instance.instance_index is None:
                continue
            name = instance.instance_name or instance.name
            self.instance_combo.addItem(f"{instance.instance_index} — {name}", instance.id)

    def capture_screenshot(self) -> None:
        instance_id = self.instance_combo.currentData()
        instance = self.context.instances.get(int(instance_id)) if instance_id is not None else None
        if instance is None or instance.instance_index is None:
            QMessageBox.warning(self, "Template Capture", "Select a MEmu instance first.")
            return
        path = self.context.memu_adb_manager.capture_screenshot(
            instance.instance_index,
            instance.instance_name or instance.name,
        )
        if path is None or not self.load_screenshot(path):
            QMessageBox.warning(self, "Template Capture", "Screenshot capture failed.")

    def browse_screenshot(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Browse Screenshot",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
        )
        if path:
            self.load_screenshot(path)

    def load_screenshot(self, path: str | Path) -> bool:
        if not self.preview.set_image(path):
            QMessageBox.warning(self, "Template Capture", f"Cannot load image:\n{path}")
            return False
        self.screenshot_path = Path(path)
        return True

    def save_template(self) -> None:
        selection = self.preview.selected_source_rect()
        name = self.template_name_input.text().strip()
        if selection.isEmpty() or not name:
            return
        safe_name = Path(name).stem
        if not safe_name:
            QMessageBox.warning(self, "Template Capture", "Enter a valid template name.")
            return
        TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
        target = TEMPLATE_DIR / f"{safe_name}.png"
        cropped = self.preview._source_pixmap.copy(selection)
        if cropped.isNull() or not cropped.save(str(target), "PNG"):
            QMessageBox.warning(self, "Template Capture", "Could not save the template.")
            return
        self.template_saved.emit(str(target))
        self.accept()

    def _selection_changed(self) -> None:
        rect = self.preview.selected_source_rect()
        self.selection_label.setText(
            f"Selection: x={rect.x()}, y={rect.y()}, width={rect.width()}, height={rect.height()}"
            if not rect.isEmpty()
            else "Selection: -"
        )
        self._update_save_enabled()

    def _update_save_enabled(self) -> None:
        self.save_button.setEnabled(
            bool(self.template_name_input.text().strip())
            and not self.preview.selected_source_rect().isEmpty()
        )
