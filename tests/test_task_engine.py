from __future__ import annotations

import sys
import tempfile
import unittest
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.db import AutomationTaskRepository, Database
from rok_assistant.db.models import Task, TaskStep
from rok_assistant.task_engine import TaskResult, TaskRunner


class FakeAdbManager:
    pass


class FakeActionEngine:
    def __init__(self):
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.fail_click = False
        self.template_exists = True

    def wait_for_template(
        self,
        template_path: str,
        *,
        threshold: float,
        timeout_seconds: float,
        retry_interval_seconds: float,
    ) -> dict[str, object]:
        self.calls.append(
            (
                "WaitTemplate",
                (template_path, threshold, timeout_seconds, retry_interval_seconds),
            )
        )
        return {
            "success": self.template_exists,
            "confidence": 0.9 if self.template_exists else 0.2,
            "x": 10 if self.template_exists else -1,
            "y": 20 if self.template_exists else -1,
            "message": "" if self.template_exists else "timeout",
        }

    def click_template(self, template_path: str, *, threshold: float) -> dict[str, object]:
        self.calls.append(("ClickTemplate", (template_path, threshold)))
        return {"success": not self.fail_click, "message": "click failed" if self.fail_click else ""}

    def click_coordinates(self, x: int, y: int) -> dict[str, object]:
        self.calls.append(("ClickCoordinates", (x, y)))
        return {"success": True, "x": x, "y": y}

    def swipe_coordinates(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int,
    ) -> dict[str, object]:
        self.calls.append(("SwipeCoordinates", (x1, y1, x2, y2, duration_ms)))
        return {"success": True, "x": x2, "y": y2}

    def abort_task(self, reason: str | None = None) -> dict[str, object]:
        abort_reason = str(reason or "").strip() or "Task aborted intentionally"
        self.calls.append(("AbortTask", (reason,) if reason is not None else ()))
        logging.getLogger("ActionEngine").info("AbortTask executed: %s", abort_reason)
        return {
            "success": True,
            "aborted": True,
            "message": abort_reason,
            "abort_reason": abort_reason,
        }


class AutomationTaskRepositoryTest(unittest.TestCase):
    def test_task_and_step_crud_duplicate_and_reorder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "test.sqlite3")
            db.initialize()
            repo = AutomationTaskRepository(db)

            task_id = repo.save_task(Task(name="Farm", enabled=True))
            first_id = repo.add_step(task_id, "Delay", {"seconds": 1.0})
            second_id = repo.add_step(task_id, "ClickCoordinates", {"x": 12, "y": 34})
            repeat_start_id = repo.add_step(task_id, "RepeatStart", {"count": 5})
            repeat_end_id = repo.add_step(task_id, "RepeatEnd", {})

            steps = repo.list_steps(task_id)
            self.assertEqual([1, 2, 3, 4], [step.order for step in steps])
            self.assertEqual(
                ["Delay", "ClickCoordinates", "RepeatStart", "RepeatEnd"],
                [step.action_type for step in steps],
            )

            repo.move_step_down(first_id)
            steps = repo.list_steps(task_id)
            self.assertEqual(
                [second_id, first_id, repeat_start_id, repeat_end_id],
                [step.id for step in steps],
            )
            self.assertEqual([1, 2, 3, 4], [step.order for step in steps])

            duplicate_id = repo.duplicate_task(task_id)
            duplicate = repo.get(duplicate_id)
            duplicate_steps = repo.list_steps(duplicate_id)
            self.assertIsNotNone(duplicate)
            self.assertEqual("Farm Copy", duplicate.name)  # type: ignore[union-attr]
            self.assertEqual(4, len(duplicate_steps))

            abort_step_id = repo.add_step(task_id, "AbortTask", {})
            self.assertIsInstance(abort_step_id, int)

            repo.delete_step(first_id)
            self.assertEqual(
                [1, 2, 3, 4],
                [step.order for step in repo.list_steps(task_id)],
            )
            repo.delete_task(task_id)
            self.assertIsNone(repo.get(task_id))
            db.close()

    def test_abort_task_serializes_deserializes_and_duplicates_reason(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "test.sqlite3")
            db.initialize()
            repo = AutomationTaskRepository(db)

            task_id = repo.save_task(Task(name="Abort flow", enabled=True))
            step_id = repo.add_step(task_id, "AbortTask", {"reason": "No free march"})
            repo.save_step(
                TaskStep(
                    id=step_id,
                    task_id=task_id,
                    order=1,
                    action_type="AbortTask",
                    parameters={"reason": "No free march"},
                )
            )

            step = repo.get_step(step_id)
            self.assertIsNotNone(step)
            self.assertEqual("AbortTask", step.action_type)  # type: ignore[union-attr]
            self.assertEqual({"reason": "No free march"}, step.parameters)  # type: ignore[union-attr]

            duplicate_id = repo.duplicate_task(task_id)
            duplicate_steps = repo.list_steps(duplicate_id)
            self.assertEqual(1, len(duplicate_steps))
            self.assertEqual("AbortTask", duplicate_steps[0].action_type)
            self.assertEqual({"reason": "No free march"}, duplicate_steps[0].parameters)
            db.close()

    def test_abort_task_serializes_deserializes_without_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "test.sqlite3")
            db.initialize()
            repo = AutomationTaskRepository(db)

            task_id = repo.save_task(Task(name="Abort flow", enabled=True))
            step_id = repo.add_step(task_id, "AbortTask", {"template_path": "ignored.png"})
            repo.save_step(
                TaskStep(
                    id=step_id,
                    task_id=task_id,
                    order=1,
                    action_type="AbortTask",
                    parameters={},
                )
            )

            step = repo.get_step(step_id)
            self.assertIsNotNone(step)
            self.assertEqual("AbortTask", step.action_type)  # type: ignore[union-attr]
            self.assertEqual({}, step.parameters)  # type: ignore[union-attr]
            db.close()


class TaskRunnerTest(unittest.TestCase):
    def test_runner_executes_supported_steps_in_order(self) -> None:
        fake_engine = FakeActionEngine()
        sleeps: list[float] = []
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: fake_engine,  # type: ignore[arg-type]
            sleeper=sleeps.append,
        )
        task = Task(id=7, name="Farm", enabled=True)
        steps = [
            TaskStep(order=2, action_type="ClickCoordinates", parameters={"x": 50, "y": 60}),
            TaskStep(order=1, action_type="Delay", parameters={"seconds": 0.25}),
            TaskStep(
                order=3,
                action_type="SwipeCoordinates",
                parameters={"x1": 1, "y1": 2, "x2": 3, "y2": 4, "duration_ms": 500},
            ),
        ]

        with self.assertLogs("TaskRunner", level="INFO") as logs:
            result = runner.run_task(task, steps, instance_index=0, instance_name="MEmu")

        self.assertTrue(result.success)
        self.assertEqual(TaskResult.SUCCESS, result.result)
        self.assertTrue(
            any("Task Finished: Farm result=SUCCESS" in message for message in logs.output)
        )
        self.assertEqual([0.25], sleeps)
        self.assertEqual(
            [
                ("ClickCoordinates", (50, 60)),
                ("SwipeCoordinates", (1, 2, 3, 4, 500)),
            ],
            fake_engine.calls,
        )

    def test_runner_stops_on_step_failure(self) -> None:
        fake_engine = FakeActionEngine()
        fake_engine.fail_click = True
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: fake_engine,  # type: ignore[arg-type]
            sleeper=lambda _seconds: None,
        )
        task = Task(id=8, name="Farm", enabled=True)
        steps = [
            TaskStep(
                order=1,
                action_type="ClickTemplate",
                parameters={"template_path": "button.png", "threshold": 0.8},
            ),
            TaskStep(order=2, action_type="ClickCoordinates", parameters={"x": 1, "y": 2}),
        ]

        result = runner.run_task(task, steps, instance_index=0, instance_name="MEmu")

        self.assertFalse(result.success)
        self.assertEqual(TaskResult.FAILED, result.result)
        self.assertEqual("click failed", result.message)
        self.assertEqual([("ClickTemplate", ("button.png", 0.8))], fake_engine.calls)

    def test_runner_abort_task_only_returns_aborted(self) -> None:
        fake_engine = FakeActionEngine()
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: fake_engine,  # type: ignore[arg-type]
            sleeper=lambda _seconds: None,
        )
        task = Task(id=19, name="Abort", enabled=True)

        with self.assertLogs(level="INFO") as logs:
            result = runner.run_task(
                task,
                [TaskStep(order=1, action_type="AbortTask", parameters={})],
                instance_index=0,
                instance_name="MEmu",
            )

        self.assertFalse(result.success)
        self.assertEqual(TaskResult.ABORTED, result.result)
        self.assertNotEqual(TaskResult.FAILED, result.result)
        self.assertEqual("Task aborted intentionally", result.message)
        self.assertEqual([("AbortTask", ())], fake_engine.calls)
        self.assertTrue(any("AbortTask executed" in message for message in logs.output))
        self.assertTrue(
            any("Task aborted intentionally" in message for message in logs.output)
        )

    def test_runner_abort_task_uses_custom_reason(self) -> None:
        fake_engine = FakeActionEngine()
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: fake_engine,  # type: ignore[arg-type]
            sleeper=lambda _seconds: None,
        )
        task = Task(id=20, name="Abort", enabled=True)

        with self.assertLogs(level="INFO") as logs:
            result = runner.run_task(
                task,
                [
                    TaskStep(
                        order=1,
                        action_type="AbortTask",
                        parameters={"reason": "No free march"},
                    )
                ],
                instance_index=0,
                instance_name="MEmu",
            )

        self.assertFalse(result.success)
        self.assertEqual(TaskResult.ABORTED, result.result)
        self.assertEqual("No free march", result.message)
        self.assertEqual([("AbortTask", ("No free march",))], fake_engine.calls)
        self.assertTrue(any("No free march" in message for message in logs.output))

    def test_runner_executes_repeat_block_requested_number_of_times(self) -> None:
        fake_engine = FakeActionEngine()
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: fake_engine,  # type: ignore[arg-type]
            sleeper=lambda _seconds: None,
        )
        task = Task(id=9, name="Loop Farm", enabled=True)
        steps = [
            TaskStep(order=1, action_type="RepeatStart", parameters={"count": 3}),
            TaskStep(order=2, action_type="ClickCoordinates", parameters={"x": 10, "y": 20}),
            TaskStep(
                order=3,
                action_type="SwipeCoordinates",
                parameters={"x1": 1, "y1": 2, "x2": 3, "y2": 4, "duration_ms": 250},
            ),
            TaskStep(order=4, action_type="RepeatEnd", parameters={}),
            TaskStep(order=5, action_type="ClickCoordinates", parameters={"x": 30, "y": 40}),
        ]

        result = runner.run_task(task, steps, instance_index=0, instance_name="MEmu")

        self.assertTrue(result.success)
        self.assertEqual(
            [
                ("ClickCoordinates", (10, 20)),
                ("SwipeCoordinates", (1, 2, 3, 4, 250)),
                ("ClickCoordinates", (10, 20)),
                ("SwipeCoordinates", (1, 2, 3, 4, 250)),
                ("ClickCoordinates", (10, 20)),
                ("SwipeCoordinates", (1, 2, 3, 4, 250)),
                ("ClickCoordinates", (30, 40)),
            ],
            fake_engine.calls,
        )

    def test_runner_abort_task_inside_repeat_stops_whole_task(self) -> None:
        fake_engine = FakeActionEngine()
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: fake_engine,  # type: ignore[arg-type]
            sleeper=lambda _seconds: None,
        )
        task = Task(id=20, name="Loop Abort", enabled=True)
        steps = [
            TaskStep(order=1, action_type="RepeatStart", parameters={"count": 3}),
            TaskStep(order=2, action_type="ClickCoordinates", parameters={"x": 10, "y": 20}),
            TaskStep(order=3, action_type="AbortTask", parameters={}),
            TaskStep(order=4, action_type="RepeatEnd", parameters={}),
            TaskStep(order=5, action_type="ClickCoordinates", parameters={"x": 30, "y": 40}),
        ]

        result = runner.run_task(task, steps, instance_index=0, instance_name="MEmu")

        self.assertFalse(result.success)
        self.assertEqual(TaskResult.ABORTED, result.result)
        self.assertEqual(
            [
                ("ClickCoordinates", (10, 20)),
                ("AbortTask", ()),
            ],
            fake_engine.calls,
        )

    def test_runner_detects_missing_repeat_end(self) -> None:
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: FakeActionEngine(),  # type: ignore[arg-type]
            sleeper=lambda _seconds: None,
        )
        result = runner.run_task(
            Task(id=10, name="Broken", enabled=True),
            [
                TaskStep(order=1, action_type="RepeatStart", parameters={"count": 2}),
                TaskStep(order=2, action_type="ClickCoordinates", parameters={"x": 1, "y": 2}),
            ],
            instance_index=0,
            instance_name="MEmu",
        )

        self.assertFalse(result.success)
        self.assertIn("Missing RepeatEnd", result.message)

    def test_runner_executes_nested_repeat_blocks(self) -> None:
        fake_engine = FakeActionEngine()
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: fake_engine,  # type: ignore[arg-type]
            sleeper=lambda _seconds: None,
        )
        result = runner.run_task(
            Task(id=11, name="Nested", enabled=True),
            [
                TaskStep(order=1, action_type="RepeatStart", parameters={"count": 2}),
                TaskStep(order=2, action_type="RepeatStart", parameters={"count": 2}),
                TaskStep(order=3, action_type="ClickCoordinates", parameters={"x": 1, "y": 2}),
                TaskStep(order=4, action_type="RepeatEnd", parameters={}),
                TaskStep(order=5, action_type="RepeatEnd", parameters={}),
            ],
            instance_index=0,
            instance_name="MEmu",
        )

        self.assertTrue(result.success)
        self.assertEqual(
            [
                ("ClickCoordinates", (1, 2)),
                ("ClickCoordinates", (1, 2)),
                ("ClickCoordinates", (1, 2)),
                ("ClickCoordinates", (1, 2)),
            ],
            fake_engine.calls,
        )

    def test_runner_detects_repeat_end_without_start(self) -> None:
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: FakeActionEngine(),  # type: ignore[arg-type]
            sleeper=lambda _seconds: None,
        )
        result = runner.run_task(
            Task(id=12, name="Stray End", enabled=True),
            [TaskStep(order=1, action_type="RepeatEnd", parameters={})],
            instance_index=0,
            instance_name="MEmu",
        )

        self.assertFalse(result.success)
        self.assertIn("RepeatEnd without RepeatStart", result.message)

    def test_runner_executes_true_branch_when_template_exists(self) -> None:
        fake_engine = FakeActionEngine()
        fake_engine.template_exists = True
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: fake_engine,  # type: ignore[arg-type]
            sleeper=lambda _seconds: None,
        )
        steps = [
            TaskStep(
                order=1,
                action_type="IfTemplateExists",
                parameters={
                    "template_path": "ready.png",
                    "threshold": 0.8,
                    "timeout_seconds": 2.0,
                    "retry_interval_seconds": 0.5,
                },
            ),
            TaskStep(order=2, action_type="ClickCoordinates", parameters={"x": 10, "y": 20}),
            TaskStep(order=3, action_type="Else", parameters={}),
            TaskStep(order=4, action_type="ClickCoordinates", parameters={"x": 30, "y": 40}),
            TaskStep(order=5, action_type="EndIf", parameters={}),
        ]

        result = runner.run_task(
            Task(id=13, name="If True", enabled=True),
            steps,
            instance_index=0,
            instance_name="MEmu",
        )

        self.assertTrue(result.success)
        self.assertEqual(
            [
                ("WaitTemplate", ("ready.png", 0.8, 0.25, 0.25)),
                ("ClickCoordinates", (10, 20)),
            ],
            fake_engine.calls,
        )
        self.assertTrue(result.steps[0].result["condition_result"])

    def test_runner_abort_task_inside_if_true_branch_returns_aborted(self) -> None:
        fake_engine = FakeActionEngine()
        fake_engine.template_exists = True
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: fake_engine,  # type: ignore[arg-type]
            sleeper=lambda _seconds: None,
        )
        steps = [
            TaskStep(
                order=1,
                action_type="IfTemplateExists",
                parameters={
                    "template_path": "ready.png",
                    "threshold": 0.8,
                    "timeout_seconds": 2.0,
                    "retry_interval_seconds": 0.5,
                },
            ),
            TaskStep(order=2, action_type="AbortTask", parameters={}),
            TaskStep(order=3, action_type="Else", parameters={}),
            TaskStep(order=4, action_type="ClickCoordinates", parameters={"x": 30, "y": 40}),
            TaskStep(order=5, action_type="EndIf", parameters={}),
        ]

        result = runner.run_task(
            Task(id=21, name="If True Abort", enabled=True),
            steps,
            instance_index=0,
            instance_name="MEmu",
        )

        self.assertFalse(result.success)
        self.assertEqual(TaskResult.ABORTED, result.result)
        self.assertEqual(
            [
                ("WaitTemplate", ("ready.png", 0.8, 0.25, 0.25)),
                ("AbortTask", ()),
            ],
            fake_engine.calls,
        )

    def test_runner_executes_false_branch_when_template_missing(self) -> None:
        fake_engine = FakeActionEngine()
        fake_engine.template_exists = False
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: fake_engine,  # type: ignore[arg-type]
            sleeper=lambda _seconds: None,
        )
        steps = [
            TaskStep(
                order=1,
                action_type="IfTemplateExists",
                parameters={
                    "template_path": "missing.png",
                    "threshold": 0.8,
                    "timeout_seconds": 2.0,
                    "retry_interval_seconds": 0.5,
                },
            ),
            TaskStep(order=2, action_type="ClickCoordinates", parameters={"x": 10, "y": 20}),
            TaskStep(order=3, action_type="Else", parameters={}),
            TaskStep(order=4, action_type="ClickCoordinates", parameters={"x": 30, "y": 40}),
            TaskStep(order=5, action_type="EndIf", parameters={}),
        ]

        result = runner.run_task(
            Task(id=14, name="If False", enabled=True),
            steps,
            instance_index=0,
            instance_name="MEmu",
        )

        self.assertTrue(result.success)
        self.assertEqual(
            [
                ("WaitTemplate", ("missing.png", 0.8, 0.25, 0.25)),
                ("ClickCoordinates", (30, 40)),
            ],
            fake_engine.calls,
        )
        self.assertFalse(result.steps[0].result["condition_result"])

    def test_runner_abort_task_inside_if_false_branch_does_not_run(self) -> None:
        fake_engine = FakeActionEngine()
        fake_engine.template_exists = False
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: fake_engine,  # type: ignore[arg-type]
            sleeper=lambda _seconds: None,
        )
        steps = [
            TaskStep(
                order=1,
                action_type="IfTemplateExists",
                parameters={
                    "template_path": "missing.png",
                    "threshold": 0.8,
                    "timeout_seconds": 2.0,
                    "retry_interval_seconds": 0.5,
                },
            ),
            TaskStep(order=2, action_type="AbortTask", parameters={}),
            TaskStep(order=3, action_type="Else", parameters={}),
            TaskStep(order=4, action_type="ClickCoordinates", parameters={"x": 30, "y": 40}),
            TaskStep(order=5, action_type="EndIf", parameters={}),
        ]

        result = runner.run_task(
            Task(id=22, name="If False Skip Abort", enabled=True),
            steps,
            instance_index=0,
            instance_name="MEmu",
        )

        self.assertTrue(result.success)
        self.assertEqual(TaskResult.SUCCESS, result.result)
        self.assertEqual(
            [
                ("WaitTemplate", ("missing.png", 0.8, 0.25, 0.25)),
                ("ClickCoordinates", (30, 40)),
            ],
            fake_engine.calls,
        )

    def test_runner_detects_missing_endif(self) -> None:
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: FakeActionEngine(),  # type: ignore[arg-type]
            sleeper=lambda _seconds: None,
        )
        result = runner.run_task(
            Task(id=15, name="Missing EndIf", enabled=True),
            [TaskStep(order=1, action_type="IfTemplateExists", parameters={})],
            instance_index=0,
            instance_name="MEmu",
        )

        self.assertFalse(result.success)
        self.assertIn("Missing EndIf", result.message)

    def test_runner_detects_else_without_if(self) -> None:
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: FakeActionEngine(),  # type: ignore[arg-type]
            sleeper=lambda _seconds: None,
        )
        result = runner.run_task(
            Task(id=16, name="Else Without If", enabled=True),
            [TaskStep(order=1, action_type="Else", parameters={})],
            instance_index=0,
            instance_name="MEmu",
        )

        self.assertFalse(result.success)
        self.assertIn("Else without IfTemplateExists", result.message)

    def test_runner_executes_nested_if_blocks(self) -> None:
        fake_engine = FakeActionEngine()
        fake_engine.template_exists = True
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: fake_engine,  # type: ignore[arg-type]
            sleeper=lambda _seconds: None,
        )
        result = runner.run_task(
            Task(id=17, name="Nested If", enabled=True),
            [
                TaskStep(
                    order=1,
                    action_type="IfTemplateExists",
                    parameters={"template_path": "outer.png"},
                ),
                TaskStep(
                    order=2,
                    action_type="IfTemplateExists",
                    parameters={"template_path": "inner.png"},
                ),
                TaskStep(order=3, action_type="ClickCoordinates", parameters={"x": 5, "y": 6}),
                TaskStep(order=4, action_type="EndIf", parameters={}),
                TaskStep(order=5, action_type="EndIf", parameters={}),
            ],
            instance_index=0,
            instance_name="MEmu",
        )

        self.assertTrue(result.success)
        self.assertEqual(
            [
                ("WaitTemplate", ("outer.png", 0.8, 0.25, 0.25)),
                ("WaitTemplate", ("inner.png", 0.8, 0.25, 0.25)),
                ("ClickCoordinates", (5, 6)),
            ],
            fake_engine.calls,
        )

    def test_runner_detects_duplicate_else(self) -> None:
        runner = TaskRunner(
            FakeAdbManager(),  # type: ignore[arg-type]
            action_engine_factory=lambda _index, _name: FakeActionEngine(),  # type: ignore[arg-type]
            sleeper=lambda _seconds: None,
        )
        result = runner.run_task(
            Task(id=18, name="Duplicate Else", enabled=True),
            [
                TaskStep(order=1, action_type="IfTemplateExists", parameters={}),
                TaskStep(order=2, action_type="Else", parameters={}),
                TaskStep(order=3, action_type="Else", parameters={}),
                TaskStep(order=4, action_type="EndIf", parameters={}),
            ],
            instance_index=0,
            instance_name="MEmu",
        )

        self.assertFalse(result.success)
        self.assertIn("Duplicate Else", result.message)


if __name__ == "__main__":
    unittest.main()
