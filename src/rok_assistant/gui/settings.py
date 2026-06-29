from __future__ import annotations

import json

from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rok_assistant.app import AppContext
from rok_assistant.emulator import DEFAULT_MEMU_INSTALL_PATH


class SettingsWidget(QWidget):
    def __init__(self, context: AppContext):
        super().__init__()
        self.context = context

        self.max_workers = QSpinBox()
        self.max_workers.setRange(1, 64)
        self.max_instances = QSpinBox()
        self.max_instances.setRange(1, 64)
        self.retry_delay = QSpinBox()
        self.retry_delay.setRange(1, 1440)
        self.pre_launch = QSpinBox()
        self.pre_launch.setRange(0, 1440)
        self.minimum_level = QSpinBox()
        self.minimum_level.setRange(1, 10)
        self.preferred_levels = QLineEdit()
        self.memu_path = QLineEdit()
        self.browse_memu_button = QPushButton("Browse")

        memu_path_row = QHBoxLayout()
        memu_path_row.addWidget(self.memu_path)
        memu_path_row.addWidget(self.browse_memu_button)

        form = QFormLayout()
        form.addRow("MEmu Install Path", memu_path_row)
        form.addRow("Max Workers", self.max_workers)
        form.addRow("Maximum Concurrent Instances", self.max_instances)
        form.addRow("Retry Delay Minutes", self.retry_delay)
        form.addRow("Pre-Launch Minutes", self.pre_launch)
        form.addRow("Preferred Resource Levels", self.preferred_levels)
        form.addRow("Minimum Resource Level", self.minimum_level)

        self.save_button = QPushButton("Save Settings")
        self.reload_button = QPushButton("Reload")
        buttons = QHBoxLayout()
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.reload_button)
        buttons.addStretch(1)

        note = QLabel("Worker count changes apply the next time the scheduler starts.")
        note.setStyleSheet("color: #5c6675")

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(note)
        layout.addStretch(1)

        self.save_button.clicked.connect(self.save)
        self.reload_button.clicked.connect(self.load)
        self.browse_memu_button.clicked.connect(self.browse_memu_path)
        self.load()

    def load(self) -> None:
        self.memu_path.setText(
            self.context.settings.get("emulator.memu_install_path", DEFAULT_MEMU_INSTALL_PATH)
        )
        self.max_workers.setValue(self.context.settings.get_int("scheduler.max_workers", 5))
        self.max_instances.setValue(
            self.context.settings.get_int("scheduler.max_active_instances", 5)
        )
        self.retry_delay.setValue(
            self.context.settings.get_int("scheduler.retry_delay_minutes", 10)
        )
        self.pre_launch.setValue(
            self.context.settings.get_int("scheduler.pre_launch_minutes", 2)
        )
        levels = self.context.settings.get_json(
            "gathering.preferred_resource_levels", [8, 7, 6]
        )
        self.preferred_levels.setText(",".join(str(level) for level in levels))
        self.minimum_level.setValue(
            self.context.settings.get_int("gathering.minimum_resource_level", 6)
        )

    def browse_memu_path(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select MEmu Install Path",
            self.memu_path.text() or DEFAULT_MEMU_INSTALL_PATH,
        )
        if path:
            self.memu_path.setText(path)

    def save(self) -> None:
        try:
            levels = [
                int(part.strip())
                for part in self.preferred_levels.text().split(",")
                if part.strip()
            ]
        except ValueError:
            QMessageBox.warning(self, "Validation", "Preferred levels must be comma-separated numbers.")
            return

        values = {
            "emulator.memu_install_path": self.memu_path.text().strip()
            or DEFAULT_MEMU_INSTALL_PATH,
            "scheduler.max_workers": self.max_workers.value(),
            "scheduler.max_active_instances": self.max_instances.value(),
            "scheduler.retry_delay_minutes": self.retry_delay.value(),
            "scheduler.pre_launch_minutes": self.pre_launch.value(),
            "gathering.preferred_resource_levels": levels,
            "gathering.minimum_resource_level": self.minimum_level.value(),
        }
        for key, value in values.items():
            self.context.settings.set(key, value)
        self.context.worker_pool.max_workers = self.max_workers.value()
        self.context.emulator_manager.set_memu_install_path(
            values["emulator.memu_install_path"]
        )
        self.context.memu_adb_manager.set_install_path(values["emulator.memu_install_path"])

        self.context.config.set("emulator.memu_install_path", values["emulator.memu_install_path"])
        self.context.config.set("scheduler.max_workers", self.max_workers.value())
        self.context.config.set("scheduler.max_active_instances", self.max_instances.value())
        self.context.config.set("scheduler.retry_delay_minutes", self.retry_delay.value())
        self.context.config.set("scheduler.pre_launch_minutes", self.pre_launch.value())
        self.context.config.set("gathering.preferred_resource_levels", levels)
        self.context.config.set("gathering.minimum_resource_level", self.minimum_level.value())
        self.context.config.save()
        QMessageBox.information(self, "Saved", "Settings saved.")
