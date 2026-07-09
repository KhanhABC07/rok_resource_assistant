from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.application.task_queue import TaskQueueViewModel
from rok_assistant.db.models import Instance, ScheduledTask, Task, TaskRunHistory, TaskStep
from rok_assistant.tasks.resource_search_workflow import ResourceType


class FakeAutomationTaskRepository:
    def __init__(self) -> None:
        self.tasks: dict[int, Task] = {}
        self.steps: dict[int, TaskStep] = {}
        self.task_steps: dict[int, list[int]] = {}
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

    def delete_task(self, task_id: int) -> None:
        self.tasks.pop(task_id, None)
        self.task_steps.pop(task_id, None)

    def duplicate_task(self, task_id: int) -> int:
        task = self.tasks[task_id]
        new_task_id = self.save_task(
            Task(
                name=f"{task.name} Copy",
                enabled=task.enabled,
                template_readiness_required=task.template_readiness_required,
            )
        )
        for step in self.list_steps(task_id):
            self.add_step(new_task_id, step.action_type, step.parameters or {})
        return new_task_id

    def list_steps(self, task_id: int) -> list[TaskStep]:
        return [self.steps[step_id] for step_id in self.task_steps.get(task_id, [])]

    def get_step(self, step_id: int) -> TaskStep | None:
        return self.steps.get(step_id)

    def add_step(
        self,
        task_id: int,
        action_type: str,
        parameters: dict[str, object],
    ) -> int:
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

    def save_step(self, step: TaskStep) -> int:
        if step.id is None:
            raise ValueError("step id required")
        self.steps[step.id] = step
        return step.id

    def delete_step(self, step_id: int) -> None:
        step = self.steps.pop(step_id, None)
        if step is not None and step.task_id is not None:
            self.task_steps[step.task_id].remove(step_id)

    def move_step_up(self, step_id: int) -> None:
        self._move_step(step_id, -1)

    def move_step_down(self, step_id: int) -> None:
        self._move_step(step_id, 1)

    def _move_step(self, step_id: int, direction: int) -> None:
        step = self.steps[step_id]
        if step.task_id is None:
            return
        ids = self.task_steps[step.task_id]
        index = ids.index(step_id)
        target = index + direction
        if target < 0 or target >= len(ids):
            return
        ids[index], ids[target] = ids[target], ids[index]


class FakeInstanceRepository:
    def __init__(self, instances: list[Instance] | None = None) -> None:
        self.instances = {instance.id: instance for instance in instances or []}

    def list_all(self) -> list[Instance]:
        return list(self.instances.values())

    def get(self, item_id: int) -> Instance | None:
        return self.instances.get(item_id)


class FakeScheduledTaskRepository:
    def __init__(self, tasks: list[ScheduledTask] | None = None) -> None:
        self.tasks = tasks or []
        self.limits: list[int] = []

    def list_recent(self, limit: int = 300) -> list[ScheduledTask]:
        self.limits.append(limit)
        return self.tasks[:limit]


class FakeTaskRunHistoryRepository:
    def __init__(self, runs: list[TaskRunHistory] | None = None) -> None:
        self.runs = runs or []
        self.limits: list[int] = []

    def list_recent(self, limit: int = 200) -> list[TaskRunHistory]:
        self.limits.append(limit)
        return self.runs[:limit]


class TaskQueueViewModelTest(unittest.TestCase):
    def make_view_model(
        self,
        *,
        automation_tasks: FakeAutomationTaskRepository | None = None,
        instances: FakeInstanceRepository | None = None,
        scheduled_tasks: FakeScheduledTaskRepository | None = None,
        history: FakeTaskRunHistoryRepository | None = None,
    ) -> TaskQueueViewModel:
        return TaskQueueViewModel(
            automation_tasks or FakeAutomationTaskRepository(),
            instances or FakeInstanceRepository(),
            scheduled_tasks or FakeScheduledTaskRepository(),
            schedule_enabled_work=lambda: 2,
            task_run_history=history,
        )

    def test_task_rows_format_enabled_state(self) -> None:
        repo = FakeAutomationTaskRepository()
        repo.save_task(Task(name="Enabled", enabled=True, created_at="2026-07-01"))
        repo.save_task(Task(name="Disabled", enabled=False, created_at="2026-07-02"))
        view_model = self.make_view_model(automation_tasks=repo)

        rows = view_model.list_task_rows()

        self.assertEqual(["Enabled", "Disabled"], [row.name for row in rows])
        self.assertEqual(["Yes", "No"], [row.enabled for row in rows])

    def test_create_resource_workflow_persists_task_and_steps(self) -> None:
        repo = FakeAutomationTaskRepository()
        view_model = self.make_view_model(automation_tasks=repo)

        task_id = view_model.create_resource_workflow(
            resource_type=ResourceType.WOOD,
            target_level=6,
            march_required=False,
            fallback_enabled=False,
        )

        task = repo.get(task_id)
        self.assertIsNotNone(task)
        self.assertEqual("Resource Workflow - Wood L6", task.name)  # type: ignore[union-attr]
        self.assertTrue(task.template_readiness_required)  # type: ignore[union-attr]
        actions = [step.action_type for step in repo.list_steps(task_id)]
        self.assertEqual(13, len(actions))
        self.assertNotIn("AbortTask", actions)

    def test_readiness_rows_group_shared_missing_templates_and_invalid_steps(self) -> None:
        repo = FakeAutomationTaskRepository()
        task_id = repo.save_task(Task(name="Workflow", template_readiness_required=True))
        first_step_id = repo.add_step(
            task_id,
            "WaitTemplate",
            {"template_path": "templates/resource_search/missing.png"},
        )
        second_step_id = repo.add_step(
            task_id,
            "ClickTemplate",
            {"template_path": "templates/resource_search/missing.png"},
        )
        invalid_step_id = repo.add_step(task_id, "ClickTemplate", {})
        view_model = self.make_view_model(automation_tasks=repo)

        readiness = view_model.readiness_view(task_id)

        self.assertFalse(readiness.ready)
        self.assertEqual(1, readiness.missing_count)
        self.assertEqual(1, readiness.invalid_count)
        self.assertEqual("Missing", readiness.rows[0].status)
        self.assertEqual((first_step_id, second_step_id), readiness.rows[0].step_ids)
        self.assertEqual("Invalid", readiness.rows[1].status)
        self.assertEqual(invalid_step_id, readiness.rows[1].invalid_step_id)

    def test_update_template_paths_updates_matching_steps_only(self) -> None:
        repo = FakeAutomationTaskRepository()
        task_id = repo.save_task(Task(name="Workflow", template_readiness_required=True))
        first_step_id = repo.add_step(
            task_id,
            "WaitTemplate",
            {"template_path": "shared.png"},
        )
        second_step_id = repo.add_step(
            task_id,
            "ClickTemplate",
            {"template_path": "shared.png"},
        )
        other_step_id = repo.add_step(
            task_id,
            "ClickTemplate",
            {"template_path": "other.png"},
        )
        view_model = self.make_view_model(automation_tasks=repo)

        view_model.update_template_paths(
            task_id=task_id,
            selected_path="replacement.png",
            template_path="shared.png",
        )

        self.assertEqual("replacement.png", repo.get_step(first_step_id).parameters["template_path"])  # type: ignore[union-attr]
        self.assertEqual("replacement.png", repo.get_step(second_step_id).parameters["template_path"])  # type: ignore[union-attr]
        self.assertEqual("other.png", repo.get_step(other_step_id).parameters["template_path"])  # type: ignore[union-attr]

    def test_prepare_task_run_returns_warning_for_unready_templates(self) -> None:
        repo = FakeAutomationTaskRepository()
        task_id = repo.save_task(Task(name="Workflow", template_readiness_required=True))
        repo.add_step(task_id, "ClickTemplate", {"template_path": "missing.png"})
        instances = FakeInstanceRepository(
            [Instance(id=1, name="MEmu", instance_index=1, instance_name="MEmu1")]
        )
        view_model = self.make_view_model(automation_tasks=repo, instances=instances)

        preparation = view_model.prepare_task_run(task_id=task_id, instance_id=1)

        self.assertFalse(preparation.ready)
        self.assertEqual("Resource Workflow", preparation.warning_title)
        self.assertIsNotNone(preparation.readiness)

    def test_prepare_task_run_returns_task_steps_and_instance_identity(self) -> None:
        repo = FakeAutomationTaskRepository()
        task_id = repo.save_task(Task(name="Manual"))
        repo.add_step(task_id, "Delay", {"seconds": 1.0})
        instances = FakeInstanceRepository(
            [Instance(id=1, name="Fallback", instance_index=4, instance_name="MEmu4")]
        )
        view_model = self.make_view_model(automation_tasks=repo, instances=instances)

        preparation = view_model.prepare_task_run(task_id=task_id, instance_id=1)

        self.assertTrue(preparation.ready)
        self.assertIsNotNone(preparation.run)
        self.assertEqual(4, preparation.run.instance_index)  # type: ignore[union-attr]
        self.assertEqual("MEmu4", preparation.run.instance_name)  # type: ignore[union-attr]
        self.assertEqual("Manual", preparation.run.task.name)  # type: ignore[union-attr]

    def test_scheduler_and_history_rows_are_table_ready(self) -> None:
        scheduler = FakeScheduledTaskRepository(
            [
                ScheduledTask(
                    id=9,
                    instance_name="MEmu1",
                    character_name="Builder",
                    task_type="gathering",
                    march_slot=None,
                    priority=10,
                    status="queued",
                    scheduled_for="2026-07-09T01:00:00",
                    attempts=1,
                    error_message="",
                )
            ]
        )
        history = FakeTaskRunHistoryRepository(
            [
                TaskRunHistory(
                    task_name="Failed",
                    instance_index=None,
                    instance_name="",
                    result="FAILED",
                    error_message="click failed",
                )
            ]
        )
        view_model = self.make_view_model(scheduled_tasks=scheduler, history=history)

        scheduler_rows = view_model.list_scheduler_rows(limit=50)
        history_rows = view_model.list_run_history_rows(limit=25)

        self.assertEqual([50], scheduler.limits)
        self.assertEqual("", scheduler_rows[0].march_slot)
        self.assertEqual("queued", scheduler_rows[0].status)
        self.assertEqual([25], history.limits)
        self.assertEqual("", history_rows[0].instance_index)
        self.assertEqual("click failed", history_rows[0].error_or_abort_reason)

    def test_step_status_reports_ready_for_existing_absolute_template(self) -> None:
        repo = FakeAutomationTaskRepository()
        with tempfile.TemporaryDirectory() as temp_dir:
            template = Path(temp_dir) / "button.png"
            template.touch()
            task_id = repo.save_task(
                Task(name="Workflow", template_readiness_required=True)
            )
            repo.add_step(task_id, "ClickTemplate", {"template_path": str(template)})
            view_model = self.make_view_model(automation_tasks=repo)

            rows = view_model.list_step_rows(task_id)

        self.assertEqual("Ready", rows[0].status)
        self.assertEqual("ready", rows[0].status_kind)


if __name__ == "__main__":
    unittest.main()
