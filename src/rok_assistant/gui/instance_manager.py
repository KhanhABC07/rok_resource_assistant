from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rok_assistant.app import AppContext
from rok_assistant.db.models import Instance


class MEmuOperationWorker(QObject):
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


class InstanceManagerWidget(QWidget):
    def __init__(self, context: AppContext):
        super().__init__()
        self.context = context
        self.logger = logging.getLogger(self.__class__.__name__)
        self.selected_id: int | None = None
        self.latest_scan_by_index: dict[int, dict[str, object]] = {}
        self.latest_screenshot_path: Path | None = None
        self._workers: list[tuple[QThread, MEmuOperationWorker]] = []
        self._busy_count = 0

        self.index_value = QLabel("-")
        self.name_value = QLabel("-")
        self.count_label = QLabel("Total: 0    Running: 0    Stopped: 0")
        self.enabled_input = QCheckBox("Enabled")
        self.enabled_input.setChecked(True)

        form = QFormLayout()
        form.addRow("Index", self.index_value)
        form.addRow("Name", self.name_value)
        form.addRow("", self.enabled_input)

        self.scan_button = QPushButton("Scan MEmu")
        self.refresh_status_button = QPushButton("Refresh Status")
        self.start_selected_button = QPushButton("Start Selected")
        self.stop_selected_button = QPushButton("Stop Selected")
        self.start_all_button = QPushButton("Start All")
        self.stop_all_button = QPushButton("Stop All")
        self.connect_adb_button = QPushButton("Connect ADB")
        self.disconnect_adb_button = QPushButton("Disconnect ADB")
        self.refresh_adb_button = QPushButton("Refresh ADB Status")
        self.capture_screenshot_button = QPushButton("Capture Screenshot")
        self.save_button = QPushButton("Save")
        self.clear_button = QPushButton("Clear")

        buttons = QHBoxLayout()
        for button in (
            self.scan_button,
            self.refresh_status_button,
            self.start_selected_button,
            self.stop_selected_button,
            self.start_all_button,
            self.stop_all_button,
            self.connect_adb_button,
            self.disconnect_adb_button,
            self.refresh_adb_button,
            self.capture_screenshot_button,
            self.save_button,
            self.clear_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Index", "Name", "Running", "PID", "ADB Serial", "ADB Connected", "Enabled"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.count_label)
        layout.addWidget(self.table, 1)

        self.scan_button.clicked.connect(self.scan_memu)
        self.refresh_status_button.clicked.connect(self.refresh_status)
        self.start_selected_button.clicked.connect(self.start_selected)
        self.stop_selected_button.clicked.connect(self.stop_selected)
        self.start_all_button.clicked.connect(self.start_all)
        self.stop_all_button.clicked.connect(self.stop_all)
        self.connect_adb_button.clicked.connect(self.connect_adb_selected)
        self.disconnect_adb_button.clicked.connect(self.disconnect_adb_selected)
        self.refresh_adb_button.clicked.connect(self.refresh_adb_status)
        self.capture_screenshot_button.clicked.connect(self.capture_screenshot)
        self.save_button.clicked.connect(self.save)
        self.clear_button.clicked.connect(self.clear_form)
        self.table.cellClicked.connect(self.load_selected)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.refresh()

    def refresh(self) -> None:
        rows = self.context.instances.list_all()
        self.table.setRowCount(len(rows))
        for row, instance in enumerate(rows):
            runtime = self._runtime_for(instance)
            index_item = self._set_table_item(self.table, row, 0, instance.instance_index)
            index_item.setData(Qt.ItemDataRole.UserRole, instance.id)
            self._set_table_item(self.table, row, 1, instance.instance_name or instance.name)
            self._set_table_item(self.table, row, 2, runtime["running"])
            self._set_table_item(self.table, row, 3, runtime["pid"])
            self._set_table_item(self.table, row, 4, instance.adb_serial)
            self._set_table_item(self.table, row, 5, "Yes" if instance.adb_connected else "No")
            self._set_table_item(self.table, row, 6, "Yes" if instance.enabled else "No")
        self.update_counts(rows)

    def scan_memu(self) -> None:
        self.logger.info("[MEmu] Scan instances requested from GUI")
        self._run_background("scan", self._scan_and_import, self._handle_scan_result)

    def refresh_status(self) -> None:
        self.logger.info("[MEmu] Refresh status requested from GUI")
        self._run_background("refresh", self._scan_and_import, self._handle_scan_result)

    def start_selected(self) -> None:
        instances = self.selected_instances()
        if not instances:
            return
        instance_ids = [instance.id for instance in instances if instance.id is not None]
        self.logger.info("[MEmu] Start instance requested for %s selected row(s)", len(instance_ids))
        self._run_background(
            "start selected",
            lambda: self._start_instances(instance_ids),
            self._handle_control_result,
        )

    def stop_selected(self) -> None:
        instances = self.selected_instances()
        if not instances:
            return
        instance_ids = [instance.id for instance in instances if instance.id is not None]
        self.logger.info("[MEmu] Stop instance requested for %s selected row(s)", len(instance_ids))
        self._run_background(
            "stop selected",
            lambda: self._stop_instances(instance_ids),
            self._handle_control_result,
        )

    def start_all(self) -> None:
        instance_ids = [
            instance.id
            for instance in self.context.instances.list_all()
            if instance.id is not None
        ]
        if not instance_ids:
            return
        self.logger.info("[MEmu] Start instance requested for all rows")
        self._run_background(
            "start all",
            lambda: self._start_instances(instance_ids),
            self._handle_control_result,
        )

    def stop_all(self) -> None:
        self.logger.info("[MEmu] Stop all instances requested from GUI")
        self._run_background("stop all", self._stop_all_instances, self._handle_control_result)

    def connect_adb_selected(self) -> None:
        instances = self.selected_instances()
        if not instances:
            return
        instance_ids = [instance.id for instance in instances if instance.id is not None]
        self.logger.info("[MEmu][ADB] Connect requested for %s selected row(s)", len(instance_ids))
        self._run_background(
            "connect adb",
            lambda: self._connect_adb_instances(instance_ids),
            self._handle_control_result,
        )

    def disconnect_adb_selected(self) -> None:
        instances = self.selected_instances()
        if not instances:
            return
        instance_ids = [instance.id for instance in instances if instance.id is not None]
        self.logger.info(
            "[MEmu][ADB] Disconnect requested for %s selected row(s)",
            len(instance_ids),
        )
        self._run_background(
            "disconnect adb",
            lambda: self._disconnect_adb_instances(instance_ids),
            self._handle_control_result,
        )

    def refresh_adb_status(self) -> None:
        instance_ids = [
            instance.id
            for instance in self.context.instances.list_all()
            if instance.id is not None
        ]
        if not instance_ids:
            return
        self.logger.info("[MEmu][ADB] Refresh status requested from GUI")
        self._run_background(
            "refresh adb status",
            lambda: self._refresh_adb_instances(instance_ids),
            self._handle_control_result,
        )

    def capture_screenshot(self) -> None:
        instances = self.selected_instances()
        if not instances:
            return
        instance_ids = [instance.id for instance in instances if instance.id is not None]
        self.logger.info(
            "[MEmu][ADB] Screenshot capture requested for %s selected row(s)",
            len(instance_ids),
        )
        self._run_background(
            "capture screenshot",
            lambda: self._capture_screenshots(instance_ids),
            self._handle_screenshot_result,
        )

    def show_context_menu(self, position) -> None:  # type: ignore[no-untyped-def]
        item = self.table.itemAt(position)
        selected_rows = {index.row() for index in self.table.selectionModel().selectedRows(0)}
        if item is not None and item.row() not in selected_rows:
            self.table.selectRow(item.row())

        menu = QMenu(self)
        start_action = menu.addAction("Start")
        stop_action = menu.addAction("Stop")
        refresh_action = menu.addAction("Refresh")
        menu.addSeparator()
        connect_adb_action = menu.addAction("Connect ADB")
        disconnect_adb_action = menu.addAction("Disconnect ADB")
        refresh_adb_action = menu.addAction("Refresh ADB Status")
        capture_screenshot_action = menu.addAction("Capture Screenshot")
        selected_action = menu.exec(self.table.viewport().mapToGlobal(position))
        if selected_action == start_action:
            self.start_selected()
        elif selected_action == stop_action:
            self.stop_selected()
        elif selected_action == refresh_action:
            self.refresh_status()
        elif selected_action == connect_adb_action:
            self.connect_adb_selected()
        elif selected_action == disconnect_adb_action:
            self.disconnect_adb_selected()
        elif selected_action == refresh_adb_action:
            self.refresh_adb_status()
        elif selected_action == capture_screenshot_action:
            self.capture_screenshot()

    def load_selected(self, row: int, _column: int) -> None:
        instance = self.instance_for_row(row)
        if instance is None:
            return
        self.selected_id = instance.id
        self.index_value.setText("" if instance.instance_index is None else str(instance.instance_index))
        self.name_value.setText(instance.instance_name or instance.name)
        self.enabled_input.setChecked(instance.enabled)

    def save(self) -> None:
        if self.selected_id is None:
            return
        instance = self.context.instances.get(self.selected_id)
        if instance is None:
            return
        self.context.instances.save(
            Instance(
                id=instance.id,
                name=instance.name,
                instance_index=instance.instance_index,
                instance_name=instance.instance_name,
                adb_serial=instance.adb_serial,
                adb_connected=instance.adb_connected,
                launch_path=instance.launch_path,
                launch_command=instance.launch_command,
                close_command=instance.close_command,
                enabled=self.enabled_input.isChecked(),
            )
        )
        self.refresh()

    def selected_instances(self) -> list[Instance]:
        selected_rows = self.table.selectionModel().selectedRows(0)
        instances: list[Instance] = []
        seen_ids: set[int] = set()
        for index in selected_rows:
            instance = self.instance_for_row(index.row())
            if instance is None or instance.id is None or instance.id in seen_ids:
                continue
            instances.append(instance)
            seen_ids.add(instance.id)
        if instances:
            return instances
        if self.selected_id is None:
            return []
        instance = self.context.instances.get(self.selected_id)
        return [instance] if instance is not None else []

    def instance_for_row(self, row: int) -> Instance | None:
        item = self.table.item(row, 0)
        if item is None:
            return None
        instance_id = item.data(Qt.ItemDataRole.UserRole)
        if instance_id is None:
            return None
        return self.context.instances.get(int(instance_id))

    def clear_form(self) -> None:
        self.selected_id = None
        self.table.clearSelection()
        self.index_value.setText("-")
        self.name_value.setText("-")
        self.enabled_input.setChecked(True)

    def update_counts(self, rows: list[Instance]) -> None:
        total = len(rows)
        running = sum(
            1
            for instance in rows
            if instance.instance_index is not None
            and self.latest_scan_by_index.get(instance.instance_index, {}).get("running") is True
        )
        stopped = max(0, total - running)
        self.count_label.setText(f"Total: {total}    Running: {running}    Stopped: {stopped}")

    def _scan_and_import(self) -> dict[str, object]:
        instances = self.context.memu_manager.scan_instances()
        imported = self.context.instances.upsert_memu_instances(instances)
        indexes = [int(item["index"]) for item in instances]
        adb_statuses = self.context.memu_adb_manager.refresh_adb_status(indexes) if indexes else {}
        self.context.instances.update_adb_statuses(adb_statuses)
        return {"instances": instances, "imported": imported, "adb_statuses": adb_statuses}

    def _start_instances(self, instance_ids: list[int]) -> dict[str, object]:
        results = []
        for instance_id in instance_ids:
            instance = self.context.instances.get(instance_id)
            if instance is None:
                continue
            self.logger.info("[MEmu] Start instance %s requested", instance.name)
            results.append(
                {
                    "name": instance.name,
                    "success": self.context.emulator_manager.launch_instance(instance),
                }
            )
        scan = self._scan_and_import()
        return {"results": results, **scan}

    def _stop_instances(self, instance_ids: list[int]) -> dict[str, object]:
        results = []
        for instance_id in instance_ids:
            instance = self.context.instances.get(instance_id)
            if instance is None:
                continue
            self.logger.info("[MEmu] Stop instance %s requested", instance.name)
            results.append(
                {
                    "name": instance.name,
                    "success": self.context.emulator_manager.close_instance(instance),
                }
            )
        scan = self._scan_and_import()
        return {"results": results, **scan}

    def _stop_all_instances(self) -> dict[str, object]:
        success = self.context.memu_manager.stop_all_instances()
        scan = self._scan_and_import()
        return {"results": [{"name": "All MEmu instances", "success": success}], **scan}

    def _connect_adb_instances(self, instance_ids: list[int]) -> dict[str, object]:
        results = []
        indexes: list[int] = []
        for instance_id in instance_ids:
            instance = self.context.instances.get(instance_id)
            if instance is None or instance.instance_index is None:
                continue
            indexes.append(instance.instance_index)
            self.logger.info("[MEmu][ADB] Connect instance %s requested", instance.name)
            results.append(
                {
                    "name": instance.name,
                    "success": self.context.memu_adb_manager.connect_instance(
                        instance.instance_index
                    ),
                }
            )
        adb_statuses = self.context.memu_adb_manager.refresh_adb_status(indexes)
        self.context.instances.update_adb_statuses(adb_statuses)
        return {"results": results, "adb_statuses": adb_statuses}

    def _disconnect_adb_instances(self, instance_ids: list[int]) -> dict[str, object]:
        results = []
        indexes: list[int] = []
        for instance_id in instance_ids:
            instance = self.context.instances.get(instance_id)
            if instance is None or instance.instance_index is None:
                continue
            indexes.append(instance.instance_index)
            self.logger.info("[MEmu][ADB] Disconnect instance %s requested", instance.name)
            results.append(
                {
                    "name": instance.name,
                    "success": self.context.memu_adb_manager.disconnect_instance(
                        instance.instance_index
                    ),
                }
            )
        adb_statuses = self.context.memu_adb_manager.refresh_adb_status(indexes)
        self.context.instances.update_adb_statuses(adb_statuses)
        return {"results": results, "adb_statuses": adb_statuses}

    def _refresh_adb_instances(self, instance_ids: list[int]) -> dict[str, object]:
        indexes = []
        for instance_id in instance_ids:
            instance = self.context.instances.get(instance_id)
            if instance is not None and instance.instance_index is not None:
                indexes.append(instance.instance_index)
        adb_statuses = self.context.memu_adb_manager.refresh_adb_status(indexes)
        self.context.instances.update_adb_statuses(adb_statuses)
        return {"results": [], "adb_statuses": adb_statuses}

    def _capture_screenshots(self, instance_ids: list[int]) -> dict[str, object]:
        results = []
        for instance_id in instance_ids:
            instance = self.context.instances.get(instance_id)
            if instance is None or instance.instance_index is None:
                continue
            path = self.context.memu_adb_manager.capture_screenshot(
                instance.instance_index,
                instance.instance_name or instance.name,
            )
            results.append(
                {
                    "name": instance.name,
                    "success": path is not None,
                    "path": "" if path is None else str(path),
                }
            )
        return {"results": results}

    def _handle_scan_result(self, result: object) -> None:
        data = result if isinstance(result, dict) else {}
        has_instance_scan = "instances" in data
        instances = data.get("instances", [])
        if isinstance(instances, list):
            self.latest_scan_by_index = {
                int(item["index"]): item
                for item in instances
                if isinstance(item, dict) and "index" in item
            }
        self.refresh()
        imported = int(data.get("imported", 0) or 0)
        self.logger.info("[MEmu] Scan instances GUI update complete: %s persisted", imported)
        if has_instance_scan and not instances:
            QMessageBox.warning(self, "MEmu", "No MEmu instances were detected.")

    def _handle_control_result(self, result: object) -> None:
        self._handle_scan_result(result)
        data = result if isinstance(result, dict) else {}
        failures = [
            item["name"]
            for item in data.get("results", [])
            if isinstance(item, dict) and not item.get("success")
        ]
        if failures:
            QMessageBox.warning(
                self,
                "MEmu",
                "Some MEmu actions failed:\n" + "\n".join(str(name) for name in failures),
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
                self.logger.info(
                    "[MEmu][ADB] Screenshot capture result for %s: %s",
                    item.get("name"),
                    item.get("path"),
                )
        if failures:
            QMessageBox.warning(
                self,
                "MEmu",
                "Some screenshots failed:\n" + "\n".join(str(name) for name in failures),
            )

    def _runtime_for(self, instance: Instance) -> dict[str, object]:
        if instance.instance_index is None:
            return {"running": "Unknown", "pid": ""}
        runtime = self.latest_scan_by_index.get(instance.instance_index)
        if runtime is None:
            return {"running": "Unknown", "pid": ""}
        return {
            "running": "Yes" if runtime["running"] else "No",
            "pid": runtime["pid"] or "",
        }

    def _run_background(
        self,
        action: str,
        callback: Callable[[], object],
        on_finished: Callable[[object], None],
    ) -> None:
        self._set_busy(True)
        thread = QThread(self)
        worker = MEmuOperationWorker(action, callback)
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
        self.logger.error("[MEmu] %s failed: %s", action, message)
        QMessageBox.warning(self, "MEmu", f"{action} failed:\n{message}")

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
            self.scan_button,
            self.refresh_status_button,
            self.start_selected_button,
            self.stop_selected_button,
            self.start_all_button,
            self.stop_all_button,
            self.connect_adb_button,
            self.disconnect_adb_button,
            self.refresh_adb_button,
            self.capture_screenshot_button,
            self.save_button,
            self.clear_button,
        ):
            button.setEnabled(enabled)

    @staticmethod
    def _set_table_item(
        table: QTableWidget,
        row: int,
        column: int,
        value: object,
    ) -> QTableWidgetItem:
        item = QTableWidgetItem("" if value is None else str(value))
        table.setItem(row, column, item)
        return item
