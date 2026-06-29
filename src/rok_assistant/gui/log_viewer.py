from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from rok_assistant.app import AppContext


class LogViewerWidget(QWidget):
    def __init__(self, context: AppContext):
        super().__init__()
        self.context = context
        self.viewer = QPlainTextEdit()
        self.viewer.setReadOnly(True)
        self.refresh_button = QPushButton("Refresh")
        self.clear_button = QPushButton("Clear Log")

        buttons = QHBoxLayout()
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addWidget(self.viewer)

        self.refresh_button.clicked.connect(self.refresh)
        self.clear_button.clicked.connect(self.clear_log)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(5000)
        self.refresh()

    def refresh(self) -> None:
        log_file = self.context.config.log_file
        if not log_file.exists():
            self.viewer.setPlainText("")
            return
        text = log_file.read_text(encoding="utf-8", errors="replace")
        self.viewer.setPlainText(text[-120_000:])
        self.viewer.verticalScrollBar().setValue(self.viewer.verticalScrollBar().maximum())

    def clear_log(self) -> None:
        self.context.config.log_file.write_text("", encoding="utf-8")
        self.refresh()
