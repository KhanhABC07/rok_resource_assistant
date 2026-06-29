from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QToolBar,
)

from rok_assistant.app import AppContext
from rok_assistant.gui.automation import AutomationWidget
from rok_assistant.gui.character_manager import CharacterManagerWidget
from rok_assistant.gui.dashboard import DashboardWidget
from rok_assistant.gui.instance_manager import InstanceManagerWidget
from rok_assistant.gui.log_viewer import LogViewerWidget
from rok_assistant.gui.march_config import MarchConfigWidget
from rok_assistant.gui.settings import SettingsWidget
from rok_assistant.gui.style import APP_STYLE
from rok_assistant.gui.task_queue import TaskQueueWidget


class MainWindow(QMainWindow):
    def __init__(self, context: AppContext):
        super().__init__()
        self.context = context
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setWindowTitle("Rise of Kingdoms Resource Assistant")
        self.setStyleSheet(APP_STYLE)

        self.tabs = QTabWidget()
        self.dashboard = DashboardWidget(context)
        self.instances = InstanceManagerWidget(context)
        self.automation = AutomationWidget(context)
        self.characters = CharacterManagerWidget(context)
        self.marches = MarchConfigWidget(context)
        self.tasks = TaskQueueWidget(context)
        self.settings = SettingsWidget(context)
        self.logs = LogViewerWidget(context)

        self.tabs.addTab(self.dashboard, "Dashboard")
        self.tabs.addTab(self.instances, "Instances")
        self.tabs.addTab(self.automation, "Automation")
        self.tabs.addTab(self.characters, "Characters")
        self.tabs.addTab(self.marches, "Marches")
        self.tabs.addTab(self.tasks, "Tasks")
        self.tabs.addTab(self.settings, "Settings")
        self.tabs.addTab(self.logs, "Logs")
        self.setCentralWidget(self.tabs)
        self.automation.open_instances_requested.connect(
            lambda: self.tabs.setCurrentWidget(self.instances)
        )

        self._build_actions()
        self.statusBar().showMessage("Ready")

    def _build_actions(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        start_action = QAction("Start Scheduler", self)
        stop_action = QAction("Stop Scheduler", self)
        create_action = QAction("Create Tasks", self)
        export_action = QAction("Export JSON", self)
        import_action = QAction("Import JSON", self)
        backup_action = QAction("Backup DB", self)
        restore_action = QAction("Restore DB", self)

        start_action.triggered.connect(self.start_scheduler)
        stop_action.triggered.connect(self.stop_scheduler)
        create_action.triggered.connect(self.create_tasks)
        export_action.triggered.connect(self.export_json)
        import_action.triggered.connect(self.import_json)
        backup_action.triggered.connect(self.backup_database)
        restore_action.triggered.connect(self.restore_database)

        for action in (
            start_action,
            stop_action,
            create_action,
            export_action,
            import_action,
            backup_action,
            restore_action,
        ):
            toolbar.addAction(action)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(export_action)
        file_menu.addAction(import_action)
        file_menu.addAction(backup_action)
        file_menu.addAction(restore_action)

        scheduler_menu = self.menuBar().addMenu("Scheduler")
        scheduler_menu.addAction(start_action)
        scheduler_menu.addAction(stop_action)
        scheduler_menu.addAction(create_action)

    def start_scheduler(self) -> None:
        self.context.scheduler.start()
        self.statusBar().showMessage("Scheduler running")

    def stop_scheduler(self) -> None:
        self.context.scheduler.stop()
        self.statusBar().showMessage("Scheduler stopped")

    def create_tasks(self) -> None:
        created = self.context.schedule_enabled_work()
        self.tasks.refresh()
        self.dashboard.refresh()
        QMessageBox.information(self, "Tasks Created", f"Created {created} task(s).")

    def export_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Configuration",
            "rok_configuration.json",
            "JSON Files (*.json)",
        )
        if not path:
            return
        self.context.configuration_service.export_json(Path(path))
        QMessageBox.information(self, "Exported", f"Configuration exported to:\n{path}")

    def import_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Configuration",
            "",
            "JSON Files (*.json)",
        )
        if not path:
            return
        self.context.configuration_service.import_json(Path(path))
        self.refresh_all()
        QMessageBox.information(self, "Imported", "Configuration imported.")

    def backup_database(self) -> None:
        path = self.context.configuration_service.backup_database()
        QMessageBox.information(self, "Backup Created", f"Backup saved to:\n{path}")

    def restore_database(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Restore SQLite Backup",
            "",
            "SQLite Database (*.sqlite3 *.db);;All Files (*)",
        )
        if not path:
            return
        self.context.scheduler.stop()
        self.context.configuration_service.restore_database(Path(path))
        self.refresh_all()
        QMessageBox.information(self, "Restored", "Database backup restored.")

    def refresh_all(self) -> None:
        self.instances.refresh()
        self.automation.refresh()
        self.characters.refresh()
        self.marches.refresh_characters()
        self.tasks.refresh()
        self.settings.load()
        self.logs.refresh()
        self.dashboard.refresh()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.context.shutdown()
        event.accept()
