from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from rok_assistant.app import AppContext
from rok_assistant.gui.widgets import SectionCard, apply_button_variant
from rok_assistant.observability import DEFAULT_LOG_TAIL_BYTES, read_text_tail


class LogViewerWidget(QWidget):
    def __init__(self, context: AppContext):
        super().__init__()
        self.context = context
        self.viewer = QPlainTextEdit()
        self.viewer.setObjectName("logViewer")
        self.viewer.setReadOnly(True)
        self.viewer.setPlaceholderText("No log entries yet.")
        self._last_log_signature: tuple[int, int] | None = None
        self.refresh_button = QPushButton("Refresh")
        self.clear_button = QPushButton("Clear Log")
        apply_button_variant(self.refresh_button, "secondary")
        apply_button_variant(self.clear_button, "danger")

        buttons = QHBoxLayout()
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        log_card = SectionCard("Logs", "Recent application log output.")
        log_card.addLayout(buttons)
        log_card.addWidget(self.viewer, 1)
        layout.addWidget(log_card, 1)

        self.refresh_button.clicked.connect(self.refresh)
        self.clear_button.clicked.connect(self.clear_log)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(5000)
        self.refresh()

    def refresh(self) -> None:
        log_file = self.context.config.log_file
        try:
            stat = log_file.stat()
        except OSError:
            if self._last_log_signature is not None:
                self.viewer.setPlainText("")
                self._last_log_signature = None
            return
        signature = (stat.st_size, stat.st_mtime_ns)
        if signature == self._last_log_signature:
            return
        self._last_log_signature = signature
        self.viewer.setPlainText(read_text_tail(log_file, DEFAULT_LOG_TAIL_BYTES))
        self.viewer.verticalScrollBar().setValue(self.viewer.verticalScrollBar().maximum())

    def clear_log(self) -> None:
        self.context.config.log_file.write_text("", encoding="utf-8")
        self._last_log_signature = None
        self.refresh()
