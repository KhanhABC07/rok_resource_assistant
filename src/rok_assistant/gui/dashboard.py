from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QTableWidget, QVBoxLayout, QWidget

from rok_assistant.app import AppContext
from rok_assistant.gui.widgets import MetricLabel, set_table_item


class DashboardWidget(QWidget):
    def __init__(self, context: AppContext):
        super().__init__()
        self.context = context
        self.active_workers = MetricLabel("Active Workers")
        self.running_instances = MetricLabel("Running Instances")
        self.total_characters = MetricLabel("Total Characters")
        self.pending_tasks = MetricLabel("Pending Tasks")
        self.next_task = MetricLabel("Next Scheduled Task")
        self.success_rate = MetricLabel("Success Rate")
        self.failures = MetricLabel("Failures")
        self.blocked_retry = MetricLabel("Blocked/Retry")
        self.queue_depth = MetricLabel("Queue Depth")
        self.active_jobs = MetricLabel("Active Jobs")
        self.concurrency = MetricLabel("Concurrency")
        self.incidents = MetricLabel("Open Incidents")
        self.last_run = MetricLabel("Last Run")

        metrics = QHBoxLayout()
        for widget in (
            self.active_workers,
            self.running_instances,
            self.total_characters,
            self.pending_tasks,
            self.next_task,
            self.success_rate,
            self.failures,
            self.blocked_retry,
            self.queue_depth,
            self.active_jobs,
            self.concurrency,
            self.incidents,
            self.last_run,
        ):
            metrics.addWidget(widget)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Instance", "Character", "Task", "Status", "Next Action"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout(self)
        layout.addLayout(metrics)
        layout.addWidget(QLabel("Task Overview"))
        layout.addWidget(self.table)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(3000)
        self.refresh()

    def refresh(self) -> None:
        stats = self.context.dashboard_stats()
        self.active_workers.set_value(stats.active_workers)
        self.running_instances.set_value(stats.running_instances)
        self.total_characters.set_value(stats.total_characters)
        self.pending_tasks.set_value(stats.pending_tasks)
        self.next_task.set_value(stats.next_scheduled_task)
        self.success_rate.set_value(f"{stats.success_rate:.0%}")
        self.failures.set_value(stats.failure_count)
        self.blocked_retry.set_value(stats.blocked_retry_count)
        self.queue_depth.set_value(stats.queue_depth)
        self.active_jobs.set_value(stats.active_jobs)
        self.concurrency.set_value(f"{stats.concurrency_in_use}/{stats.concurrency_limit}")
        self.incidents.set_value(stats.open_incident_count)
        self.last_run.set_value(stats.last_run_at)

        tasks = self.context.tasks.list_recent(limit=100)
        self.table.setRowCount(len(tasks))
        for row, task in enumerate(tasks):
            set_table_item(self.table, row, 0, task.instance_name)
            set_table_item(self.table, row, 1, task.character_name)
            label = task.task_type if task.march_slot is None else f"{task.task_type} M{task.march_slot}"
            set_table_item(self.table, row, 2, label)
            set_table_item(self.table, row, 3, task.status)
            set_table_item(self.table, row, 4, task.scheduled_for)
