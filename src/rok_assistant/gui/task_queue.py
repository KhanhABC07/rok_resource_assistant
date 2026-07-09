from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rok_assistant.app import AppContext
from rok_assistant.application.task_queue import (
    ACTION_PARAMETER_FIELDS,
    FIELD_DEFAULTS,
    FIELD_PARAMETER_KEYS,
    MAX_RESOURCE_LEVEL,
    MIN_RESOURCE_LEVEL,
    ReadinessView,
    ResourceType,
    TaskExecutionResult,
    TaskExecutionService,
    TaskQueueViewModel,
    TaskResult,
)
from rok_assistant.db.models import AUTOMATION_ACTION_TYPES, Instance
from rok_assistant.gui.widgets import set_table_item
from rok_assistant.paths import PROJECT_ROOT, TEMPLATE_DIR


class TaskExecutionWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, callback: Callable[[], TaskExecutionResult]):
        super().__init__()
        self.callback = callback

    def run(self) -> None:
        try:
            self.finished.emit(self.callback())
        except Exception as exc:
            self.failed.emit(str(exc))


class TaskQueueWidget(QWidget):
    def __init__(self, context: AppContext):
        super().__init__()
        self.context = context
        self.view_model = TaskQueueViewModel(
            context.automation_tasks,
            context.instances,
            context.tasks,
            schedule_enabled_work=context.schedule_enabled_work,
            task_run_history=getattr(context, "task_run_history", None),
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.selected_task_id: int | None = None
        self.selected_step_id: int | None = None
        self._workers: list[tuple[QThread, TaskExecutionWorker]] = []
        self._busy_count = 0
        self._selected_task_templates_ready = True

        self.tabs = QTabWidget()
        self.automation_tab = QWidget()
        self.scheduler_tab = QWidget()
        self.run_history_tab = QWidget()
        self.tabs.addTab(self.automation_tab, "Automation Tasks")
        self.tabs.addTab(self.scheduler_tab, "Scheduler Queue")
        self.tabs.addTab(self.run_history_tab, "Run History")

        self._build_automation_tab()
        self._build_scheduler_tab()
        self._build_run_history_tab()

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_scheduler_queue)
        self.timer.start(3000)
        self.refresh()

    def _build_automation_tab(self) -> None:
        self.task_name_input = QLineEdit()
        self.task_enabled_input = QCheckBox("Enabled")
        self.task_enabled_input.setChecked(True)
        self.target_instance_combo = QComboBox()
        self.resource_workflow_type_combo = QComboBox()
        for resource_type in ResourceType:
            self.resource_workflow_type_combo.addItem(
                resource_type.value.title(),
                resource_type,
            )
        self.resource_workflow_level_input = QSpinBox()
        self.resource_workflow_level_input.setRange(
            MIN_RESOURCE_LEVEL,
            MAX_RESOURCE_LEVEL,
        )
        self.resource_workflow_level_input.setValue(MAX_RESOURCE_LEVEL)
        self.resource_workflow_march_required_input = QCheckBox("March Required")
        self.resource_workflow_march_required_input.setChecked(True)
        self.resource_workflow_fallback_enabled_input = QCheckBox("Fallback Enabled")

        task_form = QFormLayout()
        task_form.addRow("Task Name", self.task_name_input)
        task_form.addRow("", self.task_enabled_input)
        task_form.addRow("Target Instance", self.target_instance_combo)
        task_form.addRow("Resource Type", self.resource_workflow_type_combo)
        task_form.addRow("Target Level", self.resource_workflow_level_input)
        task_form.addRow("", self.resource_workflow_march_required_input)
        task_form.addRow("", self.resource_workflow_fallback_enabled_input)

        self.create_task_button = QPushButton("Create Task")
        self.create_resource_workflow_button = QPushButton("Create Resource Workflow")
        self.save_task_button = QPushButton("Save Task")
        self.delete_task_button = QPushButton("Delete Task")
        self.duplicate_task_button = QPushButton("Duplicate Task")
        self.run_task_button = QPushButton("Run Task")
        self.refresh_tasks_button = QPushButton("Refresh")

        task_buttons = QHBoxLayout()
        for button in (
            self.create_task_button,
            self.create_resource_workflow_button,
            self.save_task_button,
            self.delete_task_button,
            self.duplicate_task_button,
            self.run_task_button,
            self.refresh_tasks_button,
        ):
            task_buttons.addWidget(button)
        task_buttons.addStretch(1)

        self.tasks_table = QTableWidget(0, 4)
        self.tasks_table.setHorizontalHeaderLabels(["ID", "Name", "Enabled", "Created At"])
        self.tasks_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tasks_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tasks_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tasks_table.horizontalHeader().setStretchLastSection(True)
        self.tasks_table.setMinimumHeight(120)

        self.action_type_combo = QComboBox()
        self.action_type_combo.addItems(AUTOMATION_ACTION_TYPES)
        self.template_path_input = QLineEdit()
        self.browse_template_button = QPushButton("Browse")
        template_row = QHBoxLayout()
        template_row.addWidget(self.template_path_input)
        template_row.addWidget(self.browse_template_button)

        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0.0, 1.0)
        self.threshold_input.setSingleStep(0.05)
        self.threshold_input.setDecimals(2)
        self.threshold_input.setValue(0.8)
        self.timeout_input = QDoubleSpinBox()
        self.timeout_input.setRange(0.1, 300.0)
        self.timeout_input.setDecimals(1)
        self.timeout_input.setSuffix(" s")
        self.timeout_input.setValue(10.0)
        self.retry_input = QDoubleSpinBox()
        self.retry_input.setRange(0.1, 60.0)
        self.retry_input.setDecimals(1)
        self.retry_input.setSuffix(" s")
        self.retry_input.setValue(1.0)
        self.x_input = QSpinBox()
        self.x_input.setRange(0, 10000)
        self.x_input.setValue(540)
        self.y_input = QSpinBox()
        self.y_input.setRange(0, 10000)
        self.y_input.setValue(960)
        self.x1_input = QSpinBox()
        self.x1_input.setRange(0, 10000)
        self.x1_input.setValue(540)
        self.y1_input = QSpinBox()
        self.y1_input.setRange(0, 10000)
        self.y1_input.setValue(1500)
        self.x2_input = QSpinBox()
        self.x2_input.setRange(0, 10000)
        self.x2_input.setValue(540)
        self.y2_input = QSpinBox()
        self.y2_input.setRange(0, 10000)
        self.y2_input.setValue(600)
        self.duration_input = QSpinBox()
        self.duration_input.setRange(0, 60000)
        self.duration_input.setSuffix(" ms")
        self.duration_input.setValue(500)
        self.delay_input = QDoubleSpinBox()
        self.delay_input.setRange(0.0, 3600.0)
        self.delay_input.setDecimals(2)
        self.delay_input.setSuffix(" s")
        self.delay_input.setValue(1.0)
        self.repeat_count_input = QSpinBox()
        self.repeat_count_input.setRange(1, 100000)
        self.repeat_count_input.setValue(5)
        self.reason_input = QLineEdit()

        self.step_form = QFormLayout()
        self.step_form.addRow("Action", self.action_type_combo)
        self.step_form.addRow("Template", template_row)
        self.step_form.addRow("Threshold", self.threshold_input)
        self.step_form.addRow("Timeout", self.timeout_input)
        self.step_form.addRow("Polling Interval", self.retry_input)
        self.step_form.addRow("X", self.x_input)
        self.step_form.addRow("Y", self.y_input)
        self.step_form.addRow("Swipe X1", self.x1_input)
        self.step_form.addRow("Swipe Y1", self.y1_input)
        self.step_form.addRow("Swipe X2", self.x2_input)
        self.step_form.addRow("Swipe Y2", self.y2_input)
        self.step_form.addRow("Swipe Duration", self.duration_input)
        self.step_form.addRow("Duration", self.delay_input)
        self.step_form.addRow("Count", self.repeat_count_input)
        self.step_form.addRow("Reason", self.reason_input)
        self.parameter_fields = {
            "template": (template_row, (self.template_path_input, self.browse_template_button)),
            "threshold": (self.threshold_input, (self.threshold_input,)),
            "timeout": (self.timeout_input, (self.timeout_input,)),
            "polling_interval": (self.retry_input, (self.retry_input,)),
            "x": (self.x_input, (self.x_input,)),
            "y": (self.y_input, (self.y_input,)),
            "x1": (self.x1_input, (self.x1_input,)),
            "y1": (self.y1_input, (self.y1_input,)),
            "x2": (self.x2_input, (self.x2_input,)),
            "y2": (self.y2_input, (self.y2_input,)),
            "swipe_duration": (self.duration_input, (self.duration_input,)),
            "delay_duration": (self.delay_input, (self.delay_input,)),
            "count": (self.repeat_count_input, (self.repeat_count_input,)),
            "reason": (self.reason_input, (self.reason_input,)),
        }
        self._loading_parameters = False

        self.add_step_button = QPushButton("Add Step")
        self.save_step_button = QPushButton("Save Step")
        self.remove_step_button = QPushButton("Remove Step")
        self.move_step_up_button = QPushButton("Move Up")
        self.move_step_down_button = QPushButton("Move Down")

        step_buttons_row_one = QHBoxLayout()
        for button in (
            self.add_step_button,
            self.save_step_button,
            self.remove_step_button,
        ):
            button.setMinimumWidth(92)
            step_buttons_row_one.addWidget(button)
        step_buttons_row_one.addStretch(1)

        step_buttons_row_two = QHBoxLayout()
        for button in (self.move_step_up_button, self.move_step_down_button):
            button.setMinimumWidth(100)
            step_buttons_row_two.addWidget(button)
        step_buttons_row_two.addStretch(1)

        self.steps_table = QTableWidget(0, 4)
        self.steps_table.setHorizontalHeaderLabels(
            ["Order", "Action", "Parameters", "Status"]
        )
        self.steps_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.steps_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.steps_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        steps_header = self.steps_table.horizontalHeader()
        steps_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        steps_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        steps_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        steps_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        self.steps_table.setColumnWidth(0, 60)
        self.steps_table.setColumnWidth(1, 150)
        self.steps_table.setColumnWidth(3, 130)
        self.steps_table.setMinimumHeight(180)

        self.task_status_label = QLabel("Ready")
        self.task_status_label.setWordWrap(True)

        task_section = QWidget()
        task_section_layout = QVBoxLayout(task_section)
        task_section_layout.setContentsMargins(0, 0, 0, 0)
        task_section_layout.addLayout(task_form)
        task_section_layout.addLayout(task_buttons)
        task_section_layout.addWidget(self.tasks_table, 1)

        self.step_editor_group = QGroupBox("Step Editor")
        step_editor_group_layout = QVBoxLayout(self.step_editor_group)
        step_editor_group_layout.setContentsMargins(8, 8, 8, 8)

        step_editor_widget = QWidget()
        step_editor_layout = QVBoxLayout(step_editor_widget)
        step_editor_layout.setContentsMargins(4, 4, 4, 4)
        step_editor_layout.addLayout(self.step_form)
        step_editor_layout.addSpacing(8)
        step_editor_layout.addLayout(step_buttons_row_one)
        step_editor_layout.addLayout(step_buttons_row_two)
        step_editor_layout.addStretch(1)

        step_editor_scroll = QScrollArea()
        step_editor_scroll.setWidgetResizable(True)
        step_editor_scroll.setWidget(step_editor_widget)
        step_editor_scroll.setMinimumWidth(340)
        step_editor_group_layout.addWidget(step_editor_scroll)
        self.step_editor_group.setMinimumWidth(360)

        self.workflow_steps_group = QGroupBox("Workflow Steps")
        workflow_steps_layout = QVBoxLayout(self.workflow_steps_group)
        workflow_steps_layout.setContentsMargins(8, 8, 8, 8)
        workflow_steps_layout.addWidget(self.steps_table)
        self.workflow_steps_group.setMinimumWidth(520)

        self.steps_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.steps_splitter.addWidget(self.step_editor_group)
        self.steps_splitter.addWidget(self.workflow_steps_group)
        self.steps_splitter.setStretchFactor(0, 35)
        self.steps_splitter.setStretchFactor(1, 65)
        self.steps_splitter.setSizes([420, 780])
        self.steps_splitter.setChildrenCollapsible(False)

        self.steps_tab = QWidget()
        steps_tab_layout = QVBoxLayout(self.steps_tab)
        steps_tab_layout.setContentsMargins(4, 4, 4, 4)
        steps_tab_layout.addWidget(self.steps_splitter, 1)

        self.readiness_state_label = QLabel("READY")
        self.readiness_state_label.setStyleSheet(
            "color: #1f7a3a; font-size: 16px; font-weight: 600;"
        )
        self.readiness_count_label = QLabel("0 missing templates")
        readiness_summary = QHBoxLayout()
        readiness_summary.addWidget(self.readiness_state_label)
        readiness_summary.addSpacing(12)
        readiness_summary.addWidget(self.readiness_count_label)
        readiness_summary.addStretch(1)

        self.template_readiness_table = QTableWidget(0, 4)
        self.template_readiness_table.setHorizontalHeaderLabels(
            ["Template", "Path", "Status", "Action"]
        )
        self.template_readiness_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.template_readiness_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        readiness_header = self.template_readiness_table.horizontalHeader()
        readiness_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        readiness_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        readiness_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        readiness_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.template_readiness_table.setColumnWidth(0, 220)
        self.template_readiness_table.setColumnWidth(2, 100)
        self.template_readiness_table.setColumnWidth(3, 100)
        self.template_readiness_table.setMinimumHeight(180)

        self.recheck_templates_button = QPushButton("Recheck Templates")
        self.open_template_folder_button = QPushButton("Open Template Folder")
        readiness_buttons = QHBoxLayout()
        readiness_buttons.addWidget(self.recheck_templates_button)
        readiness_buttons.addWidget(self.open_template_folder_button)
        readiness_buttons.addStretch(1)

        self.template_readiness_tab = QWidget()
        template_readiness_layout = QVBoxLayout(self.template_readiness_tab)
        template_readiness_layout.setContentsMargins(8, 8, 8, 8)
        template_readiness_layout.addLayout(readiness_summary)
        template_readiness_layout.addWidget(self.template_readiness_table, 1)
        template_readiness_layout.addLayout(readiness_buttons)

        self.lower_tabs = QTabWidget()
        self.lower_tabs.addTab(self.steps_tab, "Steps")
        self.lower_tabs.addTab(self.template_readiness_tab, "Template Readiness")
        self.lower_tabs.setMinimumHeight(300)

        step_section = QWidget()
        step_section_layout = QVBoxLayout(step_section)
        step_section_layout.setContentsMargins(0, 0, 0, 0)
        step_section_layout.addWidget(self.lower_tabs, 1)
        step_section_layout.addWidget(self.task_status_label)

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(task_section)
        main_splitter.addWidget(step_section)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)
        main_splitter.setSizes([230, 470])
        main_splitter.setChildrenCollapsible(False)

        layout = QVBoxLayout(self.automation_tab)
        layout.addWidget(main_splitter, 1)

        self.create_task_button.clicked.connect(self.create_task)
        self.create_resource_workflow_button.clicked.connect(
            self.create_resource_workflow
        )
        self.save_task_button.clicked.connect(self.save_task)
        self.delete_task_button.clicked.connect(self.delete_task)
        self.duplicate_task_button.clicked.connect(self.duplicate_task)
        self.run_task_button.clicked.connect(self.run_task)
        self.refresh_tasks_button.clicked.connect(self.refresh)
        self.browse_template_button.clicked.connect(self.browse_template)
        self.add_step_button.clicked.connect(self.add_step)
        self.save_step_button.clicked.connect(self.save_step)
        self.remove_step_button.clicked.connect(self.remove_step)
        self.move_step_up_button.clicked.connect(self.move_step_up)
        self.move_step_down_button.clicked.connect(self.move_step_down)
        self.recheck_templates_button.clicked.connect(self.recheck_templates)
        self.open_template_folder_button.clicked.connect(self.open_template_folder)
        self.action_type_combo.currentTextChanged.connect(
            self._handle_action_type_changed
        )
        self.tasks_table.cellClicked.connect(self.load_task_from_row)
        self.steps_table.cellClicked.connect(self.load_step_from_row)
        self.update_parameter_fields()

    def _build_scheduler_tab(self) -> None:
        self.create_scheduled_button = QPushButton("Create Tasks From Config")
        self.refresh_scheduled_button = QPushButton("Refresh")
        controls = QHBoxLayout()
        controls.addWidget(self.create_scheduled_button)
        controls.addWidget(self.refresh_scheduled_button)
        controls.addStretch(1)

        self.scheduler_table = QTableWidget(0, 10)
        self.scheduler_table.setHorizontalHeaderLabels(
            [
                "ID",
                "Instance",
                "Character",
                "Type",
                "March",
                "Priority",
                "Status",
                "Scheduled For",
                "Attempts",
                "Message",
            ]
        )
        self.scheduler_table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout(self.scheduler_tab)
        layout.addLayout(controls)
        layout.addWidget(self.scheduler_table)

        self.create_scheduled_button.clicked.connect(self.create_scheduled_tasks)
        self.refresh_scheduled_button.clicked.connect(self.refresh_scheduler_queue)

    def _build_run_history_tab(self) -> None:
        self.refresh_history_button = QPushButton("Refresh")
        controls = QHBoxLayout()
        controls.addWidget(self.refresh_history_button)
        controls.addStretch(1)

        self.run_history_table = QTableWidget(0, 7)
        self.run_history_table.setHorizontalHeaderLabels(
            [
                "Task",
                "Instance Index",
                "Instance Name",
                "Started At",
                "Finished At",
                "Result",
                "Error / Abort Reason",
            ]
        )
        self.run_history_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.run_history_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.run_history_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.run_history_table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout(self.run_history_tab)
        layout.addLayout(controls)
        layout.addWidget(self.run_history_table)

        self.refresh_history_button.clicked.connect(self.refresh_run_history)

    def create_task(self) -> None:
        task_id = self.view_model.create_task()
        self.selected_task_id = task_id
        self.refresh()
        self.select_task(task_id)

    def create_resource_workflow(self) -> None:
        try:
            task_id = self.view_model.create_resource_workflow(
                resource_type=self.resource_workflow_type_combo.currentData(),
                target_level=self.resource_workflow_level_input.value(),
                march_required=self.resource_workflow_march_required_input.isChecked(),
                fallback_enabled=(
                    self.resource_workflow_fallback_enabled_input.isChecked()
                ),
            )
        except ValueError as exc:
            message = str(exc)
            self.task_status_label.setText(f"Resource workflow error: {message}")
            self._set_task_status_style(TaskResult.FAILED)
            QMessageBox.warning(self, "Resource Workflow", message)
            return

        self.selected_task_id = task_id
        self.selected_step_id = None
        self.refresh()
        self.select_task(task_id)

    def save_task(self) -> None:
        if self.selected_task_id is None:
            return
        if not self.view_model.save_task(
            self.selected_task_id,
            name=self.task_name_input.text(),
            enabled=self.task_enabled_input.isChecked(),
        ):
            return
        self.refresh()
        self.select_task(self.selected_task_id)

    def delete_task(self) -> None:
        if self.selected_task_id is None:
            return
        self.view_model.delete_task(self.selected_task_id)
        self.selected_task_id = None
        self.selected_step_id = None
        self.refresh()

    def duplicate_task(self) -> None:
        if self.selected_task_id is None:
            return
        new_task_id = self.view_model.duplicate_task(self.selected_task_id)
        self.selected_task_id = new_task_id
        self.refresh()
        self.select_task(new_task_id)

    def add_step(self) -> None:
        if self.selected_task_id is None:
            return
        step_id = self.view_model.add_step(
            self.selected_task_id,
            self.action_type_combo.currentText(),
            self.current_step_parameters(),
        )
        self.selected_step_id = step_id
        self.refresh_steps()
        self.select_step(step_id)

    def save_step(self) -> None:
        if self.selected_step_id is None or self.selected_task_id is None:
            return
        if not self.view_model.save_step(
            task_id=self.selected_task_id,
            step_id=self.selected_step_id,
            action_type=self.action_type_combo.currentText(),
            parameters=self.current_step_parameters(),
        ):
            return
        self.refresh_steps()
        self.select_step(self.selected_step_id)

    def remove_step(self) -> None:
        if self.selected_step_id is None:
            return
        self.view_model.delete_step(self.selected_step_id)
        self.selected_step_id = None
        self.refresh_steps()

    def move_step_up(self) -> None:
        if self.selected_step_id is None:
            return
        self.view_model.move_step_up(self.selected_step_id)
        self.refresh_steps()
        self.select_step(self.selected_step_id)

    def move_step_down(self) -> None:
        if self.selected_step_id is None:
            return
        self.view_model.move_step_down(self.selected_step_id)
        self.refresh_steps()
        self.select_step(self.selected_step_id)

    def run_task(self) -> None:
        preparation = self.view_model.prepare_task_run(
            task_id=self.selected_task_id,
            instance_id=self.target_instance_combo.currentData(),
        )
        if preparation.readiness is not None:
            self._show_template_readiness(preparation.readiness)
        if not preparation.ready:
            if preparation.warning_message:
                QMessageBox.warning(
                    self,
                    preparation.warning_title,
                    preparation.warning_message,
                )
            return

        run = preparation.run
        if run is None:
            return
        self.logger.info("[TaskEngine] Run requested for task %s", run.task.name)
        service = TaskExecutionService(
            self.context.memu_adb_manager,
            history_repository=getattr(self.context, "task_run_history", None),
        )
        self._run_background(
            lambda: service.run_task(run)
        )

    def browse_template(self) -> None:
        TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select Template",
            str(TEMPLATE_DIR),
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
        )
        if path:
            self.template_path_input.setText(path)

    def current_step_parameters(self) -> dict[str, object]:
        action_type = self.action_type_combo.currentText()
        parameters = {
            FIELD_PARAMETER_KEYS[field_name]: self._parameter_field_value(field_name)
            for field_name in self._action_parameter_fields(action_type)
        }
        if action_type == "AbortTask" and not parameters.get("reason"):
            parameters.pop("reason", None)
        return parameters

    def load_parameters(self, action_type: str, parameters: dict[str, object]) -> None:
        self._loading_parameters = True
        try:
            self.action_type_combo.setCurrentText(action_type)
            self._clear_all_parameter_values()
            for field_name in self._action_parameter_fields(action_type):
                parameter_key = FIELD_PARAMETER_KEYS[field_name]
                value = parameters.get(parameter_key, FIELD_DEFAULTS[field_name])
                self._set_parameter_field_value(field_name, value)
        finally:
            self._loading_parameters = False
        self.update_parameter_fields()
        self._clear_parameter_error_state()

    def _handle_action_type_changed(self, action_type: str) -> None:
        if not self._loading_parameters:
            self._clear_fields_not_in_action(action_type)
        self.update_parameter_fields(action_type)
        self._clear_parameter_error_state()

    def update_parameter_fields(self, action_type: str | None = None) -> None:
        visible_fields = set(
            self._action_parameter_fields(action_type or self.action_type_combo.currentText())
        )
        for name, (field, widgets) in self.parameter_fields.items():
            visible = name in visible_fields
            self.step_form.setRowVisible(field, visible)
            label = self.step_form.labelForField(field)
            if label is not None:
                label.setVisible(visible)
            for widget in widgets:
                widget.setVisible(visible)
                widget.setEnabled(visible)

    @staticmethod
    def _action_parameter_fields(action_type: str) -> tuple[str, ...]:
        return ACTION_PARAMETER_FIELDS.get(action_type, ())

    def _clear_fields_not_in_action(self, action_type: str) -> None:
        visible_fields = set(self._action_parameter_fields(action_type))
        for field_name in self.parameter_fields:
            if field_name not in visible_fields:
                self._reset_parameter_field(field_name)

    def _clear_all_parameter_values(self) -> None:
        for field_name in self.parameter_fields:
            self._reset_parameter_field(field_name)

    def _reset_parameter_field(self, field_name: str) -> None:
        self._set_parameter_field_value(field_name, FIELD_DEFAULTS[field_name])

    def _parameter_field_value(self, field_name: str) -> object:
        if field_name == "template":
            return self.template_path_input.text().strip()
        if field_name == "threshold":
            return self.threshold_input.value()
        if field_name == "timeout":
            return self.timeout_input.value()
        if field_name == "polling_interval":
            return self.retry_input.value()
        if field_name == "x":
            return self.x_input.value()
        if field_name == "y":
            return self.y_input.value()
        if field_name == "x1":
            return self.x1_input.value()
        if field_name == "y1":
            return self.y1_input.value()
        if field_name == "x2":
            return self.x2_input.value()
        if field_name == "y2":
            return self.y2_input.value()
        if field_name == "swipe_duration":
            return self.duration_input.value()
        if field_name == "delay_duration":
            return self.delay_input.value()
        if field_name == "count":
            return self.repeat_count_input.value()
        if field_name == "reason":
            return self.reason_input.text().strip()
        return FIELD_DEFAULTS[field_name]

    def _set_parameter_field_value(self, field_name: str, value: object) -> None:
        value = self._value_or_default(field_name, value)
        if field_name == "template":
            self.template_path_input.setText(str(value))
        elif field_name == "threshold":
            self.threshold_input.setValue(float(value))
        elif field_name == "timeout":
            self.timeout_input.setValue(float(value))
        elif field_name == "polling_interval":
            self.retry_input.setValue(float(value))
        elif field_name == "x":
            self.x_input.setValue(int(value))
        elif field_name == "y":
            self.y_input.setValue(int(value))
        elif field_name == "x1":
            self.x1_input.setValue(int(value))
        elif field_name == "y1":
            self.y1_input.setValue(int(value))
        elif field_name == "x2":
            self.x2_input.setValue(int(value))
        elif field_name == "y2":
            self.y2_input.setValue(int(value))
        elif field_name == "swipe_duration":
            self.duration_input.setValue(int(value))
        elif field_name == "delay_duration":
            self.delay_input.setValue(float(value))
        elif field_name == "count":
            self.repeat_count_input.setValue(int(value))
        elif field_name == "reason":
            self.reason_input.setText(str(value))

    @staticmethod
    def _value_or_default(field_name: str, value: object) -> object:
        if value is None or value == "":
            return FIELD_DEFAULTS[field_name]
        return value

    def _clear_parameter_error_state(self) -> None:
        for field, widgets in self.parameter_fields.values():
            label = self.step_form.labelForField(field)
            if label is not None:
                label.setStyleSheet("")
                label.setToolTip("")
            for widget in widgets:
                widget.setStyleSheet("")
                widget.setToolTip("")

    def load_task_from_row(self, row: int, _column: int) -> None:
        item = self.tasks_table.item(row, 0)
        if item is None:
            return
        self.select_task(int(item.data(Qt.ItemDataRole.UserRole)))

    def load_step_from_row(self, row: int, _column: int) -> None:
        item = self.steps_table.item(row, 0)
        if item is None:
            return
        self.select_step(int(item.data(Qt.ItemDataRole.UserRole)))

    def select_task(self, task_id: int) -> None:
        task = self.view_model.get_task(task_id)
        if task is None:
            return
        self.selected_task_id = task.id
        self.selected_step_id = None
        self.task_name_input.setText(task.name)
        self.task_enabled_input.setChecked(task.enabled)
        for row in range(self.tasks_table.rowCount()):
            item = self.tasks_table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == task_id:
                self.tasks_table.selectRow(row)
                break
        self.refresh_steps()

    def select_step(self, step_id: int) -> None:
        step = self.view_model.get_step(step_id)
        if step is None:
            return
        self.selected_step_id = step.id
        self.load_parameters(step.action_type, step.parameters or {})
        for row in range(self.steps_table.rowCount()):
            item = self.steps_table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == step_id:
                self.steps_table.selectRow(row)
                break

    def refresh(self) -> None:
        self.refresh_targets()
        tasks = self.view_model.list_task_rows()
        self.tasks_table.setRowCount(len(tasks))
        for row, task in enumerate(tasks):
            id_item = QTableWidgetItem("" if task.id is None else str(task.id))
            id_item.setData(Qt.ItemDataRole.UserRole, task.id)
            self.tasks_table.setItem(row, 0, id_item)
            set_table_item(self.tasks_table, row, 1, task.name)
            set_table_item(self.tasks_table, row, 2, task.enabled)
            set_table_item(self.tasks_table, row, 3, task.created_at)
        if self.selected_task_id is not None and self.view_model.task_exists(
            self.selected_task_id
        ):
            self.select_task(self.selected_task_id)
        else:
            self.steps_table.setRowCount(0)
        self.refresh_scheduler_queue()
        self.refresh_run_history()

    def refresh_steps(self) -> None:
        if self.selected_task_id is None:
            self.steps_table.setRowCount(0)
            self.template_readiness_table.setRowCount(0)
            self._selected_task_templates_ready = True
            self._set_readiness_summary(True, 0)
            self._update_run_button_state()
            return
        rows = self.view_model.list_step_rows(self.selected_task_id)
        self.steps_table.setRowCount(len(rows))
        for row, step in enumerate(rows):
            order_item = QTableWidgetItem(str(step.order))
            order_item.setData(Qt.ItemDataRole.UserRole, step.id)
            self.steps_table.setItem(row, 0, order_item)
            set_table_item(self.steps_table, row, 1, step.action_type)
            set_table_item(self.steps_table, row, 2, step.parameters)
            status_item = QTableWidgetItem(step.status)
            if step.status_kind == "ready":
                status_item.setForeground(QColor("#1f7a3a"))
            elif step.status_kind == "missing":
                status_item.setForeground(QColor("#b00020"))
            else:
                status_item.setForeground(QColor("#9a5a00"))
            self.steps_table.setItem(row, 3, status_item)
        self._refresh_template_readiness()

    def _refresh_template_readiness(self) -> None:
        if self.selected_task_id is None:
            self._selected_task_templates_ready = True
            self._update_run_button_state()
            return
        task = self.view_model.get_task(self.selected_task_id)
        readiness = self.view_model.readiness_view(self.selected_task_id)
        if task is None or not task.template_readiness_required:
            self._selected_task_templates_ready = True
            self.task_status_label.setText("Ready")
            self.task_status_label.setStyleSheet("")
            self._populate_template_readiness_table(
                readiness,
                readiness_required=False,
            )
            self._set_readiness_summary(True, 0)
            self._update_run_button_state()
            return
        self._populate_template_readiness_table(readiness, readiness_required=True)
        self._show_template_readiness(readiness)

    def _show_template_readiness(
        self,
        readiness: ReadinessView,
    ) -> None:
        self._selected_task_templates_ready = readiness.ready
        self._set_readiness_summary(
            self._selected_task_templates_ready,
            readiness.missing_count,
            readiness.invalid_count,
        )
        self.task_status_label.setText("Ready")
        self.task_status_label.setStyleSheet("")
        self._update_run_button_state()

    def _set_readiness_summary(
        self,
        ready: bool,
        missing_count: int,
        invalid_count: int = 0,
    ) -> None:
        self.readiness_state_label.setText("READY" if ready else "NOT READY")
        color = "#1f7a3a" if ready else "#b00020"
        self.readiness_state_label.setStyleSheet(
            f"color: {color}; font-size: 16px; font-weight: 600;"
        )
        summary = f"{missing_count} missing template"
        if missing_count != 1:
            summary += "s"
        if invalid_count:
            summary += f", {invalid_count} invalid step"
            if invalid_count != 1:
                summary += "s"
        self.readiness_count_label.setText(summary)

    def _populate_template_readiness_table(
        self,
        readiness: ReadinessView,
        *,
        readiness_required: bool,
    ) -> None:
        self.template_readiness_table.setRowCount(len(readiness.rows))
        for row, template_row in enumerate(readiness.rows):
            set_table_item(
                self.template_readiness_table,
                row,
                0,
                template_row.template_name,
            )
            set_table_item(
                self.template_readiness_table,
                row,
                1,
                template_row.template_path,
            )
            status_item = QTableWidgetItem(template_row.status)
            if template_row.status_kind == "ready":
                status_item.setForeground(QColor("#1f7a3a"))
            elif template_row.status_kind == "missing":
                status_item.setForeground(QColor("#b00020"))
            else:
                status_item.setForeground(QColor("#9a5a00"))
            self.template_readiness_table.setItem(row, 2, status_item)
            if template_row.status_kind == "missing":
                browse_button = QPushButton("Browse")
                browse_button.setProperty(
                    "template_step_ids",
                    list(template_row.step_ids),
                )
                browse_button.clicked.connect(
                    lambda _checked=False, path=template_row.template_path: (
                        self.browse_readiness_template(path)
                    )
                )
                self.template_readiness_table.setCellWidget(
                    row, 3, browse_button
                )
            elif template_row.status_kind == "invalid":
                browse_button = QPushButton("Browse")
                browse_button.clicked.connect(
                    lambda _checked=False, row_step_id=template_row.invalid_step_id: (
                        self.browse_readiness_template("", step_id=row_step_id)
                    )
                )
                self.template_readiness_table.setCellWidget(row, 3, browse_button)

        self.template_readiness_table.setEnabled(readiness_required)

    def recheck_templates(self) -> None:
        self.refresh_steps()

    def browse_readiness_template(
        self,
        template_path: str,
        *,
        step_id: int | None = None,
    ) -> None:
        resource_template_dir = PROJECT_ROOT / "templates" / "resource_search"
        resource_template_dir.mkdir(parents=True, exist_ok=True)
        selected_path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select Template",
            str(resource_template_dir),
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
        )
        if not selected_path or self.selected_task_id is None:
            return

        self.view_model.update_template_paths(
            task_id=self.selected_task_id,
            selected_path=selected_path,
            template_path=template_path,
            step_id=step_id,
        )
        self.refresh_steps()
        if self.selected_step_id is not None:
            self.select_step(self.selected_step_id)

    def open_template_folder(self) -> None:
        template_folder = PROJECT_ROOT / "templates" / "resource_search"
        template_folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(template_folder)))

    def _update_run_button_state(self) -> None:
        self.run_task_button.setEnabled(
            self._busy_count == 0 and self._selected_task_templates_ready
        )

    def refresh_targets(self) -> None:
        current_id = self.target_instance_combo.currentData()
        self.target_instance_combo.clear()
        for instance in self.view_model.list_target_rows():
            self.target_instance_combo.addItem(instance.label, instance.id)
        if current_id is not None:
            for index in range(self.target_instance_combo.count()):
                if self.target_instance_combo.itemData(index) == current_id:
                    self.target_instance_combo.setCurrentIndex(index)
                    break

    def current_target_instance(self) -> Instance | None:
        instance_id = self.target_instance_combo.currentData()
        if instance_id is None:
            return None
        return self.view_model.get_instance(int(instance_id))

    def create_scheduled_tasks(self) -> None:
        created = self.view_model.create_scheduled_tasks()
        QMessageBox.information(self, "Tasks Created", f"Created {created} task(s).")
        self.refresh_scheduler_queue()

    def refresh_scheduler_queue(self) -> None:
        tasks = self.view_model.list_scheduler_rows(limit=300)
        self.scheduler_table.setRowCount(len(tasks))
        for row, task in enumerate(tasks):
            set_table_item(self.scheduler_table, row, 0, task.id)
            set_table_item(self.scheduler_table, row, 1, task.instance_name)
            set_table_item(self.scheduler_table, row, 2, task.character_name)
            set_table_item(self.scheduler_table, row, 3, task.task_type)
            set_table_item(self.scheduler_table, row, 4, task.march_slot or "")
            set_table_item(self.scheduler_table, row, 5, task.priority)
            set_table_item(self.scheduler_table, row, 6, task.status)
            set_table_item(self.scheduler_table, row, 7, task.scheduled_for)
            set_table_item(self.scheduler_table, row, 8, task.attempts)
            set_table_item(self.scheduler_table, row, 9, task.error_message)

    def refresh_run_history(self) -> None:
        self.view_model.task_run_history = getattr(self.context, "task_run_history", None)
        runs = self.view_model.list_run_history_rows(limit=200)
        self.run_history_table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            set_table_item(self.run_history_table, row, 0, run.task_name)
            set_table_item(self.run_history_table, row, 1, run.instance_index)
            set_table_item(self.run_history_table, row, 2, run.instance_name)
            set_table_item(self.run_history_table, row, 3, run.started_at)
            set_table_item(self.run_history_table, row, 4, run.finished_at)
            result_item = QTableWidgetItem(run.result)
            result_item.setForeground(self._result_color(run.result))
            self.run_history_table.setItem(row, 5, result_item)
            set_table_item(
                self.run_history_table,
                row,
                6,
                run.error_or_abort_reason,
            )

    def _run_background(self, callback: Callable[[], TaskExecutionResult]) -> None:
        self._set_busy(True)
        thread = QThread(self)
        worker = TaskExecutionWorker(callback)
        worker.moveToThread(thread)
        self._workers.append((thread, worker))

        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_run_finished)
        worker.failed.connect(self._handle_run_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda: self._cleanup_worker(thread))
        thread.start()

    def _handle_run_finished(self, result: object) -> None:
        if not isinstance(result, TaskExecutionResult):
            return
        task_result = result.result or (
            TaskResult.SUCCESS if result.success else TaskResult.FAILED
        )
        status_text = task_result.value
        detail = f": {result.message}" if result.message else ""
        self.task_status_label.setText(
            f"Task result {status_text}: {result.task_name} "
            f"({len(result.steps)} step(s)){detail}"
        )
        self._set_task_status_style(task_result)
        self.logger.info(
            "[TaskEngine] Task Result: %s task=%s message=%s",
            status_text,
            result.task_name,
            result.message,
        )
        if task_result == TaskResult.ABORTED:
            QMessageBox.information(self, "Tasks", f"Task aborted:\n{result.message}")
        elif not result.success:
            QMessageBox.warning(self, "Tasks", f"Task failed:\n{result.message}")
        self.refresh_run_history()

    def _handle_run_failed(self, message: str) -> None:
        self.logger.error("[TaskEngine] Task worker failed: %s", message)
        self.task_status_label.setText(f"Task failed: {message}")
        self._set_task_status_style(TaskResult.FAILED)
        QMessageBox.warning(self, "Tasks", f"Task failed:\n{message}")
        self.refresh_run_history()

    def _set_task_status_style(self, result: TaskResult) -> None:
        if result == TaskResult.SUCCESS:
            self.task_status_label.setStyleSheet("color: #1f7a3a; font-weight: 600;")
        elif result == TaskResult.ABORTED:
            self.task_status_label.setStyleSheet("color: #9a5a00; font-weight: 600;")
        else:
            self.task_status_label.setStyleSheet("color: #b00020; font-weight: 600;")

    @staticmethod
    def _result_color(result: str) -> QColor:
        if result == TaskResult.SUCCESS.value:
            return QColor("#1f7a3a")
        if result == TaskResult.ABORTED.value:
            return QColor("#9a5a00")
        return QColor("#b00020")

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
            self.create_task_button,
            self.create_resource_workflow_button,
            self.save_task_button,
            self.delete_task_button,
            self.duplicate_task_button,
            self.refresh_tasks_button,
            self.add_step_button,
            self.save_step_button,
            self.remove_step_button,
            self.move_step_up_button,
            self.move_step_down_button,
        ):
            button.setEnabled(enabled)
        self._update_run_button_state()
