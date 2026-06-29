from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QPushButton, QSplitter

from rok_assistant.db import AutomationTaskRepository, Database
from rok_assistant.db.models import Task, TaskRunHistory, TaskStep
from rok_assistant.gui.task_queue import ACTION_PARAMETER_FIELDS, TaskQueueWidget
from rok_assistant.task_engine import TaskExecutionResult, TaskResult


class EmptyRepository:
    def list_all(self) -> list[object]:
        return []

    def list_recent(self, limit: int = 300) -> list[object]:
        return []

    def get(self, _item_id: int) -> None:
        return None


class FakeAutomationTaskRepository(EmptyRepository):
    def __init__(self) -> None:
        self.tasks: dict[int, Task] = {}
        self.steps: dict[int, TaskStep] = {}
        self.task_steps: dict[int, list[int]] = {}
        self.added_steps: list[tuple[int, str, dict[str, object]]] = []
        self.next_task_id = 1
        self.next_step_id = 1

    def list_all(self) -> list[Task]:
        return list(self.tasks.values())

    def get(self, item_id: int) -> Task | None:
        return self.tasks.get(item_id)

    def save_task(self, task: Task) -> int:
        task_id = task.id or self.next_task_id
        if task.id is None:
            self.next_task_id += 1
        self.tasks[task_id] = Task(
            id=task_id,
            name=task.name,
            enabled=task.enabled,
            template_readiness_required=task.template_readiness_required,
            created_at=task.created_at,
        )
        self.task_steps.setdefault(task_id, [])
        return task_id

    def add_step(
        self,
        task_id: int,
        action_type: str,
        parameters: dict[str, object],
    ) -> int:
        self.added_steps.append((task_id, action_type, parameters))
        step_id = self.next_step_id
        self.next_step_id += 1
        order = len(self.task_steps.setdefault(task_id, [])) + 1
        self.steps[step_id] = TaskStep(
            id=step_id,
            task_id=task_id,
            order=order,
            action_type=action_type,
            parameters=parameters,
        )
        self.task_steps[task_id].append(step_id)
        return step_id

    def list_steps(self, task_id: int) -> list[TaskStep]:
        return [self.steps[step_id] for step_id in self.task_steps.get(task_id, [])]

    def get_step(self, step_id: int) -> TaskStep | None:
        return self.steps.get(step_id)

    def save_step(self, step: TaskStep) -> int:
        if step.id is None:
            raise ValueError("Test step must already exist.")
        self.steps[step.id] = step
        return step.id


class FakeTaskRunHistoryRepository:
    def __init__(self, runs: list[TaskRunHistory]):
        self.runs = runs
        self.list_recent_calls: list[int] = []

    def list_recent(self, limit: int = 200) -> list[TaskRunHistory]:
        self.list_recent_calls.append(limit)
        return self.runs[:limit]


class TaskQueueWidgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.automation_tasks = FakeAutomationTaskRepository()
        self.widget = TaskQueueWidget(
            SimpleNamespace(
                instances=EmptyRepository(),
                automation_tasks=self.automation_tasks,
                tasks=EmptyRepository(),
                task_run_history=EmptyRepository(),
                memu_adb_manager=object(),
                schedule_enabled_work=lambda: 0,
            )
        )

    def tearDown(self) -> None:
        self.widget.timer.stop()
        self.widget.deleteLater()

    def visible_parameter_fields(self) -> set[str]:
        return {
            name
            for name, (_field, widgets) in self.widget.parameter_fields.items()
            if any(not widget.isHidden() for widget in widgets)
        }

    def test_abort_task_is_selectable_and_saves_optional_reason(self) -> None:
        self.assertGreaterEqual(self.widget.action_type_combo.findText("AbortTask"), 0)

        self.widget.action_type_combo.setCurrentText("AbortTask")

        self.assertEqual({}, self.widget.current_step_parameters())
        self.assertEqual({"reason"}, self.visible_parameter_fields())

        self.widget.reason_input.setText("No free march")

        self.assertEqual(
            {"reason": "No free march"},
            self.widget.current_step_parameters(),
        )

    def test_click_template_to_abort_task_hides_and_clears_template_fields(self) -> None:
        self.widget.action_type_combo.setCurrentText("ClickTemplate")
        self.widget.template_path_input.setText("button.png")
        self.widget.threshold_input.setValue(0.42)

        self.widget.action_type_combo.setCurrentText("AbortTask")

        self.assertEqual({"reason"}, self.visible_parameter_fields())
        self.assertEqual("", self.widget.template_path_input.text())
        self.assertEqual(0.8, self.widget.threshold_input.value())
        self.assertEqual({}, self.widget.current_step_parameters())

    def test_click_coordinates_to_abort_task_hides_and_clears_coordinates(self) -> None:
        self.widget.action_type_combo.setCurrentText("ClickCoordinates")
        self.widget.x_input.setValue(123)
        self.widget.y_input.setValue(456)

        self.widget.action_type_combo.setCurrentText("AbortTask")

        self.assertEqual({"reason"}, self.visible_parameter_fields())
        self.assertEqual(540, self.widget.x_input.value())
        self.assertEqual(960, self.widget.y_input.value())
        self.assertEqual({}, self.widget.current_step_parameters())

    def test_delay_to_abort_task_hides_and_clears_duration(self) -> None:
        self.widget.action_type_combo.setCurrentText("Delay")
        self.widget.delay_input.setValue(12.5)

        self.widget.action_type_combo.setCurrentText("AbortTask")

        self.assertEqual({"reason"}, self.visible_parameter_fields())
        self.assertEqual(1.0, self.widget.delay_input.value())
        self.assertEqual({}, self.widget.current_step_parameters())

    def test_abort_task_to_click_template_shows_only_template_related_fields(self) -> None:
        self.widget.action_type_combo.setCurrentText("AbortTask")

        self.widget.action_type_combo.setCurrentText("ClickTemplate")

        self.assertEqual({"template", "threshold"}, self.visible_parameter_fields())
        self.assertEqual(
            {"template_path": "", "threshold": 0.8},
            self.widget.current_step_parameters(),
        )

    def test_block_delimiters_show_no_parameter_fields(self) -> None:
        for action_type in ("RepeatEnd", "Else", "EndIf"):
            with self.subTest(action_type=action_type):
                self.widget.action_type_combo.setCurrentText("ClickCoordinates")
                self.widget.x_input.setValue(123)
                self.widget.action_type_combo.setCurrentText(action_type)

                self.assertEqual(set(), self.visible_parameter_fields())
                self.assertEqual({}, self.widget.current_step_parameters())

    def test_saving_abort_task_contains_no_stale_parameters(self) -> None:
        self.widget.selected_task_id = 99
        self.widget.action_type_combo.setCurrentText("ClickCoordinates")
        self.widget.x_input.setValue(123)
        self.widget.y_input.setValue(456)
        self.widget.action_type_combo.setCurrentText("AbortTask")

        self.widget.add_step()

        self.assertEqual([(99, "AbortTask", {})], self.automation_tasks.added_steps)

    def test_saving_abort_task_includes_reason(self) -> None:
        self.widget.selected_task_id = 99
        self.widget.action_type_combo.setCurrentText("AbortTask")
        self.widget.reason_input.setText("No free march")

        self.widget.add_step()

        self.assertEqual(
            [(99, "AbortTask", {"reason": "No free march"})],
            self.automation_tasks.added_steps,
        )

    def test_create_resource_workflow_generates_new_editable_task(self) -> None:
        self.widget.resource_workflow_type_combo.setCurrentText("Wood")
        self.widget.resource_workflow_level_input.setValue(6)

        self.widget.create_resource_workflow()

        self.assertEqual(1, self.widget.selected_task_id)
        task = self.automation_tasks.get(1)
        self.assertIsNotNone(task)
        self.assertEqual(
            "Resource Workflow - Wood L6",
            task.name,  # type: ignore[union-attr]
        )
        self.assertEqual(
            [
                "IfTemplateExists",
                "AbortTask",
                "EndIf",
                "ClickTemplate",
                "ClickTemplate",
                "WaitTemplate",
                "ClickTemplate",
                "ClickTemplate",
                "ClickTemplate",
                "WaitTemplate",
                "ClickTemplate",
                "WaitTemplate",
                "ClickTemplate",
                "WaitTemplate",
                "ClickTemplate",
                "WaitTemplate",
            ],
            [
                action_type
                for _task_id, action_type, _parameters in (
                    self.automation_tasks.added_steps
                )
            ],
        )
        self.assertEqual(16, self.widget.steps_table.rowCount())
        self.assertEqual(
            {
                "template_path": (
                    "templates/resource_search/wood_resource_icon.png"
                )
            },
            self.automation_tasks.added_steps[6][2],
        )
        self.assertEqual(
            {
                "template_path": (
                    "templates/resource_search/resource_level_6_selector.png"
                )
            },
            self.automation_tasks.added_steps[7][2],
        )
        self.assertEqual("NOT READY", self.widget.readiness_state_label.text())
        readiness_paths = [
            self.widget.template_readiness_table.item(row, 1).text()
            for row in range(self.widget.template_readiness_table.rowCount())
        ]
        self.assertIn(
            "templates/resource_search/world_map_button.png",
            readiness_paths,
        )
        self.assertFalse(self.widget.run_task_button.isEnabled())

    def test_resource_workflow_controls_only_offer_natural_resources(self) -> None:
        options = [
            self.widget.resource_workflow_type_combo.itemText(index)
            for index in range(self.widget.resource_workflow_type_combo.count())
        ]

        self.assertEqual(["Food", "Wood", "Stone", "Gold"], options)
        self.assertNotIn("Alliance Resource", " ".join(options))
        self.assertEqual(1, self.widget.resource_workflow_level_input.minimum())
        self.assertEqual(8, self.widget.resource_workflow_level_input.maximum())
        self.assertTrue(self.widget.resource_workflow_march_required_input.isChecked())
        self.assertFalse(self.widget.resource_workflow_fallback_enabled_input.isChecked())

    def test_create_resource_workflow_respects_march_required_flag(self) -> None:
        self.widget.resource_workflow_march_required_input.setChecked(False)

        self.widget.create_resource_workflow()

        self.assertEqual(13, len(self.automation_tasks.added_steps))
        self.assertNotIn(
            "AbortTask",
            [
                action_type
                for _task_id, action_type, _parameters in (
                    self.automation_tasks.added_steps
                )
            ],
        )

    def test_create_resource_workflow_shows_validation_errors(self) -> None:
        self.widget.resource_workflow_type_combo.addItem("Invalid", "Invalid")
        self.widget.resource_workflow_type_combo.setCurrentText("Invalid")

        with patch("rok_assistant.gui.task_queue.QMessageBox.warning") as warning:
            self.widget.create_resource_workflow()

        self.assertEqual([], self.automation_tasks.added_steps)
        self.assertIn("Resource workflow error", self.widget.task_status_label.text())
        warning.assert_called_once()

    def test_unready_resource_workflow_can_still_be_saved(self) -> None:
        self.widget.create_resource_workflow()
        self.assertFalse(self.widget.run_task_button.isEnabled())
        self.assertTrue(self.widget.save_task_button.isEnabled())

        self.widget.task_name_input.setText("Edited Unready Workflow")
        self.widget.save_task()

        saved = self.automation_tasks.get(1)
        self.assertIsNotNone(saved)
        self.assertEqual("Edited Unready Workflow", saved.name)  # type: ignore[union-attr]
        self.assertTrue(  # type: ignore[union-attr]
            saved.template_readiness_required
        )
        self.assertFalse(self.widget.run_task_button.isEnabled())

    def test_loading_existing_resource_workflow_rechecks_readiness(self) -> None:
        task_id = self.automation_tasks.save_task(
            Task(
                name="Existing Resource Workflow",
                template_readiness_required=True,
            )
        )
        self.automation_tasks.add_step(
            task_id,
            "ClickTemplate",
            {"template_path": "templates/resource_search/missing.png"},
        )

        self.widget.select_task(task_id)

        self.assertEqual("NOT READY", self.widget.readiness_state_label.text())
        self.assertEqual(
            "templates/resource_search/missing.png",
            self.widget.template_readiness_table.item(0, 1).text(),
        )
        self.assertFalse(self.widget.run_task_button.isEnabled())

    def test_manual_task_keeps_existing_run_behavior(self) -> None:
        task_id = self.automation_tasks.save_task(Task(name="Manual Task"))
        self.automation_tasks.add_step(
            task_id,
            "ClickTemplate",
            {"template_path": "missing.png"},
        )

        self.widget.select_task(task_id)

        self.assertEqual("Ready", self.widget.task_status_label.text())
        self.assertTrue(self.widget.run_task_button.isEnabled())

    def test_lower_editor_contains_steps_and_template_readiness_tabs(self) -> None:
        self.assertEqual(2, self.widget.lower_tabs.count())
        self.assertEqual("Steps", self.widget.lower_tabs.tabText(0))
        self.assertEqual("Template Readiness", self.widget.lower_tabs.tabText(1))

    def test_steps_tab_uses_resizable_horizontal_splitter(self) -> None:
        self.assertIsInstance(self.widget.steps_splitter, QSplitter)
        self.assertEqual(
            Qt.Orientation.Horizontal,
            self.widget.steps_splitter.orientation(),
        )
        self.assertFalse(self.widget.steps_splitter.childrenCollapsible())
        self.assertEqual(2, self.widget.steps_splitter.count())

    def test_step_editor_buttons_are_accessible_at_normal_window_size(self) -> None:
        self.widget.resize(1366, 768)
        self.widget.show()
        self.app.processEvents()

        for button in (
            self.widget.add_step_button,
            self.widget.save_step_button,
            self.widget.remove_step_button,
            self.widget.move_step_up_button,
            self.widget.move_step_down_button,
        ):
            with self.subTest(button=button.text()):
                self.assertFalse(button.isHidden())
                self.assertTrue(button.isVisible())
                self.assertGreaterEqual(button.width(), button.minimumWidth())
        self.assertEqual("Move Down", self.widget.move_step_down_button.text())

    def test_workflow_steps_table_has_status_column(self) -> None:
        headers = [
            self.widget.steps_table.horizontalHeaderItem(column).text()
            for column in range(self.widget.steps_table.columnCount())
        ]

        self.assertEqual(["Order", "Action", "Parameters", "Status"], headers)

    def test_missing_templates_appear_as_readiness_rows(self) -> None:
        self.widget.create_resource_workflow()

        statuses = [
            self.widget.template_readiness_table.item(row, 2).text()
            for row in range(self.widget.template_readiness_table.rowCount())
        ]
        self.assertTrue(statuses)
        self.assertTrue(all(status == "Missing" for status in statuses))
        self.assertEqual("NOT READY", self.widget.readiness_state_label.text())
        self.assertIn("missing template", self.widget.readiness_count_label.text())

    def test_browse_updates_corresponding_template_paths(self) -> None:
        task_id = self.automation_tasks.save_task(
            Task(name="Browse Workflow", template_readiness_required=True)
        )
        first_step_id = self.automation_tasks.add_step(
            task_id,
            "WaitTemplate",
            {"template_path": "templates/resource_search/shared_missing.png"},
        )
        second_step_id = self.automation_tasks.add_step(
            task_id,
            "ClickTemplate",
            {"template_path": "templates/resource_search/shared_missing.png"},
        )
        self.widget.select_task(task_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            replacement = Path(temp_dir) / "replacement.png"
            replacement.touch()
            with patch(
                "rok_assistant.gui.task_queue.QFileDialog.getOpenFileName",
                return_value=(str(replacement), "Images"),
            ):
                browse = self.widget.template_readiness_table.cellWidget(0, 3)
                self.assertIsInstance(browse, QPushButton)
                browse.click()  # type: ignore[union-attr]

            self.assertEqual(
                str(replacement),
                (
                    self.automation_tasks.get_step(first_step_id).parameters
                    or {}
                )["template_path"],
            )
            self.assertEqual(
                str(replacement),
                (
                    self.automation_tasks.get_step(second_step_id).parameters
                    or {}
                )["template_path"],
            )

    def test_recheck_changes_missing_template_to_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            template = Path(temp_dir) / "appears_later.png"
            task_id = self.automation_tasks.save_task(
                Task(name="Recheck Workflow", template_readiness_required=True)
            )
            self.automation_tasks.add_step(
                task_id,
                "ClickTemplate",
                {"template_path": str(template)},
            )
            self.widget.select_task(task_id)
            self.assertEqual(
                "Missing",
                self.widget.template_readiness_table.item(0, 2).text(),
            )

            template.touch()
            self.widget.recheck_templates_button.click()

            self.assertEqual(
                "Ready",
                self.widget.template_readiness_table.item(0, 2).text(),
            )
            self.assertEqual("READY", self.widget.readiness_state_label.text())
            self.assertTrue(self.widget.run_task_button.isEnabled())

    def test_click_template_changed_to_abort_task_persists_without_stale_parameters(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite3"
            db = Database(db_path)
            db.initialize()
            repo = AutomationTaskRepository(db)
            task_id = repo.save_task(Task(name="Abort persistence", enabled=True))
            step_id = repo.add_step(
                task_id,
                "ClickTemplate",
                {"template_path": "button.png", "threshold": 0.42},
            )

            widget = TaskQueueWidget(
                SimpleNamespace(
                    instances=EmptyRepository(),
                    automation_tasks=repo,
                    tasks=EmptyRepository(),
                    task_run_history=EmptyRepository(),
                    memu_adb_manager=object(),
                    schedule_enabled_work=lambda: 0,
                )
            )
            try:
                original_step = repo.get_step(step_id)
                self.assertIsNotNone(original_step)
                widget.selected_task_id = task_id
                widget.selected_step_id = step_id
                widget.load_parameters(
                    original_step.action_type,  # type: ignore[union-attr]
                    original_step.parameters or {},  # type: ignore[union-attr]
                )

                widget.action_type_combo.setCurrentText("AbortTask")
                widget.save_step()
            finally:
                widget.timer.stop()
                widget.deleteLater()
                db.close()

            reloaded_db = Database(db_path)
            reloaded_db.initialize()
            try:
                reloaded_repo = AutomationTaskRepository(reloaded_db)
                reloaded_steps = reloaded_repo.list_steps(task_id)
            finally:
                reloaded_db.close()

        self.assertEqual(1, len(reloaded_steps))
        loaded_step = reloaded_steps[0]
        self.assertEqual("AbortTask", loaded_step.action_type)
        loaded_parameters = loaded_step.parameters or {}
        stale_keys = {
            "template_path",
            "threshold",
            "timeout_seconds",
            "retry_interval_seconds",
            "x",
            "y",
            "x1",
            "y1",
            "x2",
            "y2",
            "duration_ms",
            "seconds",
            "count",
        }
        self.assertFalse(stale_keys.intersection(loaded_parameters))
        self.assertLessEqual(set(loaded_parameters), {"reason"})

    def test_loading_abort_task_shows_no_unrelated_parameters(self) -> None:
        self.widget.load_parameters(
            "AbortTask",
            {
                "template_path": "stale.png",
                "threshold": 0.55,
                "x": 123,
                "y": 456,
                "seconds": 10.0,
                "count": 7,
            },
        )

        self.assertEqual({"reason"}, self.visible_parameter_fields())
        self.assertEqual("", self.widget.template_path_input.text())
        self.assertEqual(540, self.widget.x_input.value())
        self.assertEqual(1.0, self.widget.delay_input.value())
        self.assertEqual({}, self.widget.current_step_parameters())

    def test_loading_abort_task_reason_populates_reason_field(self) -> None:
        self.widget.load_parameters(
            "AbortTask",
            {"reason": "No free march", "template_path": "stale.png"},
        )

        self.assertEqual({"reason"}, self.visible_parameter_fields())
        self.assertEqual("No free march", self.widget.reason_input.text())
        self.assertEqual(
            {"reason": "No free march"},
            self.widget.current_step_parameters(),
        )

    def test_existing_actions_display_their_mapped_fields(self) -> None:
        for action_type, expected_fields in ACTION_PARAMETER_FIELDS.items():
            with self.subTest(action_type=action_type):
                self.widget.action_type_combo.setCurrentText(action_type)
                self.assertEqual(set(expected_fields), self.visible_parameter_fields())

    def test_aborted_result_status_is_distinct_from_failed(self) -> None:
        result = TaskExecutionResult(
            task_id=1,
            task_name="Abort flow",
            success=False,
            message="Task aborted intentionally",
            result=TaskResult.ABORTED,
        )

        with patch("rok_assistant.gui.task_queue.QMessageBox.information") as information:
            self.widget._handle_run_finished(result)

        self.assertIn("ABORTED", self.widget.task_status_label.text())
        self.assertNotIn("failed", self.widget.task_status_label.text().lower())
        self.assertIn("#9a5a00", self.widget.task_status_label.styleSheet())
        information.assert_called_once()

    def test_run_history_table_displays_recent_runs(self) -> None:
        self.widget.context.task_run_history = FakeTaskRunHistoryRepository(
            [
                TaskRunHistory(
                    task_name="Success task",
                    instance_index=1,
                    instance_name="MEmu1",
                    started_at="2026-06-22T10:00:00Z",
                    finished_at="2026-06-22T10:01:00Z",
                    result="SUCCESS",
                ),
                TaskRunHistory(
                    task_name="Failed task",
                    instance_index=2,
                    instance_name="MEmu2",
                    started_at="2026-06-22T11:00:00Z",
                    finished_at="2026-06-22T11:01:00Z",
                    result="FAILED",
                    error_message="click failed",
                ),
                TaskRunHistory(
                    task_name="Aborted task",
                    instance_index=3,
                    instance_name="MEmu3",
                    started_at="2026-06-22T12:00:00Z",
                    finished_at="2026-06-22T12:01:00Z",
                    result="ABORTED",
                    abort_reason="Stopped by task action",
                ),
            ]
        )

        self.widget.refresh_run_history()

        self.assertEqual(3, self.widget.run_history_table.rowCount())
        self.assertEqual("Success task", self.widget.run_history_table.item(0, 0).text())
        self.assertEqual("1", self.widget.run_history_table.item(0, 1).text())
        self.assertEqual("MEmu1", self.widget.run_history_table.item(0, 2).text())
        self.assertEqual("SUCCESS", self.widget.run_history_table.item(0, 5).text())
        self.assertEqual("FAILED", self.widget.run_history_table.item(1, 5).text())
        self.assertEqual("click failed", self.widget.run_history_table.item(1, 6).text())
        self.assertEqual("ABORTED", self.widget.run_history_table.item(2, 5).text())
        self.assertEqual(
            "Stopped by task action",
            self.widget.run_history_table.item(2, 6).text(),
        )
        self.assertEqual(
            "#1f7a3a",
            self.widget.run_history_table.item(0, 5).foreground().color().name(),
        )
        self.assertEqual(
            "#b00020",
            self.widget.run_history_table.item(1, 5).foreground().color().name(),
        )
        self.assertEqual(
            "#9a5a00",
            self.widget.run_history_table.item(2, 5).foreground().color().name(),
        )

    def test_run_history_refreshes_after_task_completion(self) -> None:
        history = FakeTaskRunHistoryRepository(
            [
                TaskRunHistory(
                    task_name="Completed task",
                    instance_index=4,
                    instance_name="MEmu4",
                    result="SUCCESS",
                )
            ]
        )
        self.widget.context.task_run_history = history

        self.widget._handle_run_finished(
            TaskExecutionResult(
                task_id=4,
                task_name="Completed task",
                success=True,
                result=TaskResult.SUCCESS,
            )
        )

        self.assertEqual([200], history.list_recent_calls)
        self.assertEqual(1, self.widget.run_history_table.rowCount())
        self.assertEqual("Completed task", self.widget.run_history_table.item(0, 0).text())


if __name__ == "__main__":
    unittest.main()
