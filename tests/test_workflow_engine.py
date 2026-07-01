from __future__ import annotations

import copy
import json
import math
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tests.db_helpers import SRC_ROOT  # noqa: F401

from rok_assistant.db.database import Database
from rok_assistant.db.models import (
    Job,
    JobRun,
    StepRun,
    Task,
    TaskStep,
    WorkflowDefinition,
    WorkflowStep,
)
from rok_assistant.db.repositories import JobRepository, JobRunRepository, StepRunRepository
from rok_assistant.workflow_engine import (
    ActionRegistry,
    CancellationToken,
    ConditionEvaluation,
    ConditionRegistry,
    DuplicateRegistrationError,
    LegacyAutomationTaskAdapter,
    MAX_CALCULATED_BACKOFF_SECONDS,
    MAX_REPEAT_ITERATIONS,
    MAX_RETRY_BACKOFF_MULTIPLIER,
    MAX_RETRY_DELAY_SECONDS,
    MAX_RETRY_LIMIT,
    MAX_RUNTIME_SLEEP_CHUNK_SECONDS,
    MAX_SAFE_COLLECTION_SIZE,
    MAX_SAFE_METADATA_DEPTH,
    MAX_SUB_WORKFLOW_DEPTH,
    NormalizerRegistry,
    RegistryFrozenError,
    RecoveryPhase,
    SemanticTemplate,
    StepBudget,
    UNKNOWN_FIELD_POLICY,
    UnknownRegistrationError,
    WorkflowDefinitionSpec,
    WorkflowDeadline,
    WorkflowEngine,
    WorkflowExecutionContext,
    WorkflowOutcome,
    WorkflowRunRepositoryRecorder,
    WorkflowStepResult,
    WorkflowStepSpec,
    WorkflowValidationError,
    WorkflowValidationLimits,
    default_action_registry,
    default_condition_registry,
    default_normalizer_registry,
    safe_json_payload,
    safe_serialize_metadata,
)


class FakeActionEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.template_matches: dict[str, bool] = {}
        self.fail_next_clicks = 0

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
        matched = self.template_matches.get(template_path, True)
        return {
            "success": matched,
            "message": "" if matched else "timeout",
            "confidence": 0.9 if matched else 0.1,
            "screenshot_path": f"runtime/screens/{template_path}.png",
        }

    def click_template(self, template_path: str, *, threshold: float) -> dict[str, object]:
        self.calls.append(("ClickTemplate", (template_path, threshold)))
        if self.fail_next_clicks > 0:
            self.fail_next_clicks -= 1
            return {
                "success": False,
                "message": "click failed",
                "screenshot_path": "runtime/screens/click-failed.png",
            }
        return {
            "success": True,
            "template_path": template_path,
            "threshold": threshold,
            "screenshot_path": "runtime/screens/click.png",
        }

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


class MutableClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class RecordingPersistence:
    def __init__(self) -> None:
        self.started = False

    def start_workflow_run(
        self,
        workflow: WorkflowDefinitionSpec,
        context: WorkflowExecutionContext,
        started_at: str,
    ) -> int | None:
        del workflow, context, started_at
        self.started = True
        return 1

    def finish_workflow_run(
        self,
        job_run_id: int | None,
        result: object,
    ) -> None:
        del job_run_id, result

    def get_completed_step(
        self,
        job_run_id: int | None,
        step: WorkflowStepSpec,
        attempt: int,
    ) -> WorkflowStepResult | None:
        del job_run_id, step, attempt
        return None

    def start_step_run(
        self,
        job_run_id: int | None,
        step: WorkflowStepSpec,
        attempt: int,
        started_at: str,
    ) -> int | None:
        del job_run_id, step, attempt, started_at
        return 1

    def finish_step_run(
        self,
        step_run_id: int | None,
        result: WorkflowStepResult,
    ) -> None:
        del step_run_id, result


class FailingStepStartRecorder(WorkflowRunRepositoryRecorder):
    def _after_step_start_saved(self, _step_run_id: int) -> None:
        raise RuntimeError("injected step start failure")


class FailingStepFinishRecorder(WorkflowRunRepositoryRecorder):
    def _after_step_finish_saved(self, _step_run_id: int) -> None:
        raise RuntimeError("injected step finish failure")


class WorkflowEngineTest(unittest.TestCase):
    def _workflow_with_tap(
        self,
        *,
        postcondition: dict[str, object] | None = None,
    ) -> WorkflowDefinitionSpec:
        return WorkflowDefinitionSpec(
            workflow_key="persisted",
            steps=[
                WorkflowStepSpec(
                    step_key="tap",
                    action_type="tap",
                    parameters={"x": 7, "y": 8},
                    postcondition=postcondition,
                )
            ],
        )

    def _database_job_context(
        self,
        temp_dir: str,
        *,
        run_key: str = "run-1",
    ) -> tuple[Database, JobRunRepository, StepRunRepository, int, WorkflowRunRepositoryRecorder]:
        db = Database(Path(temp_dir) / "workflow.sqlite3")
        db.initialize()
        jobs = JobRepository(db)
        job_runs = JobRunRepository(db)
        step_runs = StepRunRepository(db)
        job_id = jobs.save(
            Job(
                idempotency_key=f"job-{run_key}",
                job_type="workflow",
                scheduled_for="2026-01-01T00:00:00",
            )
        )
        recorder = WorkflowRunRepositoryRecorder(job_runs, step_runs)
        return db, job_runs, step_runs, job_id, recorder

    def test_validation_rejects_malformed_workflow(self) -> None:
        workflow = WorkflowDefinitionSpec(
            workflow_key="",
            schema_version=1,
            steps=[
                WorkflowStepSpec(step_key="same", action_type="tap", parameters={}),
                WorkflowStepSpec(step_key="same", action_type="missing", parameters={}),
            ],
        )

        errors = WorkflowEngine().validation_errors(workflow)

        self.assertTrue(any("schema_version" in error for error in errors))
        self.assertTrue(any("workflow_key is required" in error for error in errors))
        self.assertTrue(any("Duplicate step_key" in error for error in errors))
        self.assertTrue(any("unsupported action_type missing" in error for error in errors))

    def test_strict_workflow_json_rejects_malformed_missing_and_unknown_fields(self) -> None:
        with self.assertRaises(WorkflowValidationError) as malformed:
            WorkflowDefinitionSpec.from_json("{")
        self.assertTrue(malformed.exception.field_errors)
        self.assertIn("workflow_json", malformed.exception.errors[0])

        with self.assertRaises(WorkflowValidationError) as missing:
            WorkflowDefinitionSpec.from_json(
                json.dumps({"schema_version": 2, "steps": []})
            )
        self.assertIn("workflow.workflow_key", missing.exception.errors[0])
        with self.assertRaises(WorkflowValidationError) as missing_steps:
            WorkflowDefinitionSpec.from_json(
                json.dumps({"workflow_key": "missing-steps", "schema_version": 2})
            )
        self.assertIn("workflow.steps", missing_steps.exception.errors[0])

        with self.assertRaises(WorkflowValidationError) as unknown:
            WorkflowDefinitionSpec.from_json(
                json.dumps(
                    {
                        "workflow_key": "strict",
                        "schema_version": 2,
                        "steps": [],
                        "unexpected": True,
                    }
                )
            )
        self.assertIn("unknown field", str(unknown.exception))
        self.assertIn("reject unknown", UNKNOWN_FIELD_POLICY)

    def test_parsing_and_validation_do_not_mutate_caller_owned_structures(self) -> None:
        workflow_mapping: dict[str, object] = {
            "workflow_key": "caller-owned",
            "schema_version": 2,
            "config": {"max_steps": 10, "labels": ["kept"]},
            "steps": [
                {
                    "step_key": "branch",
                    "action_type": "if_else",
                    "parameters": {
                        "condition_type": "always",
                        "condition_metadata": {"labels": ["ready"]},
                    },
                    "then_steps": [
                        {
                            "step_key": "tap-then",
                            "action_type": "tap",
                            "parameters": {"x": 1, "y": 2},
                            "postcondition": {
                                "condition_type": "always",
                                "condition_metadata": {"labels": ["done"]},
                            },
                        }
                    ],
                    "else_steps": [
                        {
                            "step_key": "delay-else",
                            "action_type": "delay",
                            "parameters": {"seconds": 0.0},
                        }
                    ],
                }
            ],
        }
        original = copy.deepcopy(workflow_mapping)
        config = workflow_mapping["config"]
        steps = workflow_mapping["steps"]
        branch = steps[0]  # type: ignore[index]
        branch_parameters = branch["parameters"]  # type: ignore[index]
        then_steps = branch["then_steps"]  # type: ignore[index]
        then_step = then_steps[0]  # type: ignore[index]
        then_parameters = then_step["parameters"]  # type: ignore[index]
        postcondition = then_step["postcondition"]  # type: ignore[index]
        else_steps = branch["else_steps"]  # type: ignore[index]

        workflow = WorkflowDefinitionSpec.from_mapping(workflow_mapping)
        errors = WorkflowEngine().validation_errors(workflow)

        self.assertEqual([], errors)
        self.assertEqual(original, workflow_mapping)
        self.assertIsNot(workflow.config, config)
        self.assertIsNot(workflow.steps, steps)
        self.assertIsNot(workflow.steps[0].parameters, branch_parameters)
        self.assertIsNot(workflow.steps[0].then_steps, then_steps)
        self.assertIsNot(workflow.steps[0].then_steps[0].parameters, then_parameters)
        self.assertIsNot(workflow.steps[0].then_steps[0].postcondition, postcondition)
        self.assertIsNot(workflow.steps[0].else_steps, else_steps)

    def test_strict_workflow_json_rejects_unsupported_schema_before_steps(self) -> None:
        with self.assertRaises(WorkflowValidationError) as unsupported:
            WorkflowDefinitionSpec.from_json(
                json.dumps(
                    {
                        "workflow_key": "unsupported",
                        "schema_version": 99,
                        "steps": [{"unexpected": True}],
                    }
                )
            )

        self.assertIn("workflow.schema_version", unsupported.exception.errors[0])
        self.assertIn("unsupported schema_version 99", unsupported.exception.errors[0])
        self.assertNotIn("steps[0]", str(unsupported.exception))

    def test_persisted_workflow_rejects_unsupported_schema_before_steps(self) -> None:
        with self.assertRaises(WorkflowValidationError) as unsupported:
            WorkflowDefinitionSpec.from_persisted(
                WorkflowDefinition(
                    workflow_key="persisted-unsupported",
                    config_json=json.dumps({"schema_version": 99}),
                ),
                [
                    WorkflowStep(
                        step_key="malformed",
                        action_type="tap",
                        parameters_json="{",
                    )
                ],
            )

        self.assertIn("config_json.schema_version", unsupported.exception.errors[0])
        self.assertIn("unsupported schema_version 99", unsupported.exception.errors[0])

    def test_validation_happens_before_persistence_or_side_effects(self) -> None:
        persistence = RecordingPersistence()
        fake_engine = FakeActionEngine()
        workflow = WorkflowDefinitionSpec(
            workflow_key="invalid",
            schema_version=99,
            steps=[
                WorkflowStepSpec(
                    step_key="tap",
                    action_type="tap",
                    parameters={"x": 1, "y": 2},
                )
            ],
        )

        result = WorkflowEngine().execute(
            workflow,
            WorkflowExecutionContext(
                action_engine=fake_engine,
                persistence=persistence,  # type: ignore[arg-type]
                job_id=1,
                run_key="invalid-run",
            ),
        )

        self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
        self.assertFalse(persistence.started)
        self.assertEqual([], fake_engine.calls)

    def test_registry_rejects_duplicate_and_frozen_mutation(self) -> None:
        def action(
            _context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            return WorkflowStepResult(step.step_key, step.action_type, WorkflowOutcome.SUCCESS)

        registry = ActionRegistry()
        registry.register("custom", action)
        with self.assertRaises(DuplicateRegistrationError):
            registry.register("custom", action)
        registry.freeze()
        with self.assertRaises(RegistryFrozenError):
            registry.register("other", action)

    def test_condition_and_normalizer_registries_reject_duplicates(self) -> None:
        def condition(
            _context: WorkflowExecutionContext,
            _step: WorkflowStepSpec,
        ) -> ConditionEvaluation:
            return ConditionEvaluation(True)

        def normalizer(
            _context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            return WorkflowStepResult(step.step_key, step.action_type, WorkflowOutcome.SUCCESS)

        conditions = ConditionRegistry()
        conditions.register("ready", condition)
        normalizers = NormalizerRegistry()
        normalizers.register("home", normalizer)

        with self.assertRaises(DuplicateRegistrationError):
            conditions.register("ready", condition)
        with self.assertRaises(DuplicateRegistrationError):
            normalizers.register("home", normalizer)
        conditions.freeze()
        normalizers.freeze()
        with self.assertRaises(RegistryFrozenError):
            conditions.register("other", condition)
        with self.assertRaises(RegistryFrozenError):
            normalizers.register("other", normalizer)
        with self.assertRaises(UnknownRegistrationError):
            normalizers.require("missing")

    def test_registry_unknown_lookup_and_deterministic_frozen_defaults(self) -> None:
        registry = ActionRegistry()
        self.assertIsNone(registry.get("missing"))
        with self.assertRaises(UnknownRegistrationError):
            registry.require("missing")

        actions = default_action_registry()
        conditions = default_condition_registry()
        normalizers = default_normalizer_registry()

        self.assertTrue(actions.frozen)
        self.assertTrue(conditions.frozen)
        self.assertTrue(normalizers.frozen)
        self.assertEqual(
            (
                "wait",
                "click_semantic_template",
                "tap",
                "swipe",
                "delay",
                "normalize_scene",
                "cancel",
            ),
            actions.names(),
        )
        self.assertEqual(("always", "template_exists"), conditions.names())
        self.assertEqual((), normalizers.names())

    def test_registry_lookup_is_thread_safe(self) -> None:
        registry = default_action_registry()
        ready = threading.Barrier(8)

        def lookup() -> tuple[str, ...]:
            ready.wait(timeout=5)
            return registry.names()

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _index: lookup(), range(8)))

        self.assertTrue(all(result == registry.names() for result in results))

    def test_plugin_registry_initialization_freeze_and_execution(self) -> None:
        calls: list[str] = []

        def plugin_action(
            _context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            calls.append("action")
            return WorkflowStepResult(step.step_key, step.action_type, WorkflowOutcome.SUCCESS)

        def plugin_condition(
            _context: WorkflowExecutionContext,
            _step: WorkflowStepSpec,
        ) -> ConditionEvaluation:
            calls.append("condition")
            return ConditionEvaluation(True)

        def plugin_normalize_action(
            context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            calls.append("normalize-action")
            registration = context.normalizer_registry.require("plugin_home")
            return registration.handler(context, step)

        def plugin_normalizer(
            _context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            calls.append("normalizer")
            return WorkflowStepResult(step.step_key, step.action_type, WorkflowOutcome.SUCCESS)

        actions = ActionRegistry()
        conditions = ConditionRegistry()
        normalizers = NormalizerRegistry()
        self.assertFalse(actions.frozen)
        self.assertFalse(conditions.frozen)
        self.assertFalse(normalizers.frozen)

        actions.register("plugin_action", plugin_action)
        actions.register("plugin_normalize", plugin_normalize_action)
        conditions.register("plugin_ready", plugin_condition)
        normalizers.register("plugin_home", plugin_normalizer)
        engine = WorkflowEngine(
            action_registry=actions,
            condition_registry=conditions,
            normalizer_registry=normalizers,
        )
        actions.freeze()
        conditions.freeze()
        normalizers.freeze()

        with self.assertRaises(RegistryFrozenError):
            actions.register("late_plugin_action", plugin_action)
        with self.assertRaises(RegistryFrozenError):
            conditions.register("late_plugin_condition", plugin_condition)
        with self.assertRaises(RegistryFrozenError):
            normalizers.register("late_plugin_normalizer", plugin_normalizer)

        workflow = WorkflowDefinitionSpec(
            workflow_key="plugin",
            steps=[
                WorkflowStepSpec(
                    step_key="branch",
                    action_type="if_else",
                    parameters={"condition_type": "plugin_ready"},
                    then_steps=[
                        WorkflowStepSpec(
                            step_key="custom",
                            action_type="plugin_action",
                        )
                    ],
                ),
                WorkflowStepSpec(
                    step_key="normalize",
                    action_type="plugin_normalize",
                ),
            ],
        )

        result = engine.execute(workflow)

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["condition", "action", "normalize-action", "normalizer"], calls)

    def test_sub_workflow_validation_rejects_missing_child(self) -> None:
        workflow = WorkflowDefinitionSpec(
            workflow_key="root",
            steps=[
                WorkflowStepSpec(
                    step_key="call-missing",
                    action_type="sub_workflow",
                    parameters={"workflow_key": "missing"},
                )
            ],
        )

        errors = WorkflowEngine().validation_errors(
            workflow,
            WorkflowExecutionContext(workflow_resolver=lambda _key: None),
        )

        self.assertTrue(any("sub-workflow not found: missing" in error for error in errors))

    def test_sub_workflow_validation_rejects_direct_self_cycle(self) -> None:
        workflow = WorkflowDefinitionSpec(
            workflow_key="self",
            steps=[
                WorkflowStepSpec(
                    step_key="call-self",
                    action_type="sub_workflow",
                    parameters={"workflow_key": "self"},
                )
            ],
        )

        errors = WorkflowEngine().validation_errors(
            workflow,
            WorkflowExecutionContext(workflow_resolver=lambda key: workflow if key == "self" else None),
        )

        self.assertTrue(any("self -> self" in error for error in errors))

    def test_sub_workflow_validation_rejects_two_workflow_cycle(self) -> None:
        workflow_a = WorkflowDefinitionSpec(
            workflow_key="a",
            steps=[
                WorkflowStepSpec(
                    step_key="call-b",
                    action_type="sub_workflow",
                    parameters={"workflow_key": "b"},
                )
            ],
        )
        workflow_b = WorkflowDefinitionSpec(
            workflow_key="b",
            steps=[
                WorkflowStepSpec(
                    step_key="call-a",
                    action_type="sub_workflow",
                    parameters={"workflow_key": "a"},
                )
            ],
        )
        workflows = {"a": workflow_a, "b": workflow_b}

        errors = WorkflowEngine().validation_errors(
            workflow_a,
            WorkflowExecutionContext(workflow_resolver=workflows.get),
        )

        self.assertTrue(any("a -> b -> a" in error for error in errors))

    def test_sub_workflow_validation_rejects_longer_cycle(self) -> None:
        workflow_a = WorkflowDefinitionSpec(
            workflow_key="a",
            steps=[
                WorkflowStepSpec(
                    step_key="call-b",
                    action_type="sub_workflow",
                    parameters={"workflow_key": "b"},
                )
            ],
        )
        workflow_b = WorkflowDefinitionSpec(
            workflow_key="b",
            steps=[
                WorkflowStepSpec(
                    step_key="call-c",
                    action_type="sub_workflow",
                    parameters={"workflow_key": "c"},
                )
            ],
        )
        workflow_c = WorkflowDefinitionSpec(
            workflow_key="c",
            steps=[
                WorkflowStepSpec(
                    step_key="call-a",
                    action_type="sub_workflow",
                    parameters={"workflow_key": "a"},
                )
            ],
        )
        workflows = {"a": workflow_a, "b": workflow_b, "c": workflow_c}

        errors = WorkflowEngine().validation_errors(
            workflow_a,
            WorkflowExecutionContext(workflow_resolver=workflows.get),
        )

        self.assertTrue(any("a -> b -> c -> a" in error for error in errors))

    def test_sub_workflow_validation_allows_valid_nested_workflows(self) -> None:
        fake_engine = FakeActionEngine()
        grandchild = WorkflowDefinitionSpec(
            workflow_key="grandchild",
            steps=[
                WorkflowStepSpec(
                    step_key="tap-grandchild",
                    action_type="tap",
                    parameters={"x": 5, "y": 6},
                )
            ],
        )
        child = WorkflowDefinitionSpec(
            workflow_key="child",
            steps=[
                WorkflowStepSpec(
                    step_key="call-grandchild",
                    action_type="sub_workflow",
                    parameters={"workflow_key": "grandchild"},
                )
            ],
        )
        root = WorkflowDefinitionSpec(
            workflow_key="root",
            steps=[
                WorkflowStepSpec(
                    step_key="call-child",
                    action_type="sub_workflow",
                    parameters={"workflow_key": "child"},
                )
            ],
        )
        workflows = {"child": child, "grandchild": grandchild}

        result = WorkflowEngine().execute(
            root,
            WorkflowExecutionContext(
                action_engine=fake_engine,
                workflow_resolver=workflows.get,
            ),
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual([("ClickCoordinates", (5, 6))], fake_engine.calls)

    def test_sub_workflow_validation_allows_shared_child_dag(self) -> None:
        shared = WorkflowDefinitionSpec(
            workflow_key="shared",
            steps=[
                WorkflowStepSpec(
                    step_key="shared-tap",
                    action_type="tap",
                    parameters={"x": 1, "y": 2},
                )
            ],
        )
        parent_a = WorkflowDefinitionSpec(
            workflow_key="parent-a",
            steps=[
                WorkflowStepSpec(
                    step_key="call-shared-a",
                    action_type="sub_workflow",
                    parameters={"workflow_key": "shared"},
                )
            ],
        )
        parent_b = WorkflowDefinitionSpec(
            workflow_key="parent-b",
            steps=[
                WorkflowStepSpec(
                    step_key="call-shared-b",
                    action_type="sub_workflow",
                    parameters={"workflow_key": "shared"},
                )
            ],
        )
        root = WorkflowDefinitionSpec(
            workflow_key="root",
            steps=[
                WorkflowStepSpec(
                    step_key="call-parent-a",
                    action_type="sub_workflow",
                    parameters={"workflow_key": "parent-a"},
                ),
                WorkflowStepSpec(
                    step_key="call-parent-b",
                    action_type="sub_workflow",
                    parameters={"workflow_key": "parent-b"},
                ),
            ],
        )
        workflows = {
            "parent-a": parent_a,
            "parent-b": parent_b,
            "shared": shared,
        }

        errors = WorkflowEngine().validation_errors(
            root,
            WorkflowExecutionContext(workflow_resolver=workflows.get),
        )

        self.assertEqual([], errors)

    def test_sub_workflow_validation_enforces_maximum_depth(self) -> None:
        grandchild = WorkflowDefinitionSpec(
            workflow_key="grandchild",
            steps=[
                WorkflowStepSpec(
                    step_key="tap-grandchild",
                    action_type="tap",
                    parameters={"x": 5, "y": 6},
                )
            ],
        )
        child = WorkflowDefinitionSpec(
            workflow_key="child",
            steps=[
                WorkflowStepSpec(
                    step_key="call-grandchild",
                    action_type="sub_workflow",
                    parameters={"workflow_key": "grandchild"},
                )
            ],
        )
        root = WorkflowDefinitionSpec(
            workflow_key="root",
            steps=[
                WorkflowStepSpec(
                    step_key="call-child",
                    action_type="sub_workflow",
                    parameters={"workflow_key": "child"},
                )
            ],
        )
        workflows = {"child": child, "grandchild": grandchild}

        errors = WorkflowEngine(
            validation_limits=WorkflowValidationLimits(max_sub_workflow_depth=1)
        ).validation_errors(
            root,
            WorkflowExecutionContext(workflow_resolver=workflows.get),
        )

        self.assertTrue(any("depth" in error for error in errors))

    def test_nested_branches_repeats_and_sub_workflow_execute(self) -> None:
        fake_engine = FakeActionEngine()
        fake_engine.template_matches["outer.png"] = True
        fake_engine.template_matches["inner.png"] = False
        sub_workflow = WorkflowDefinitionSpec(
            workflow_key="sub",
            steps=[
                WorkflowStepSpec(
                    step_key="sub-tap",
                    action_type="tap",
                    parameters={"x": 9, "y": 10},
                )
            ],
        )
        workflow = WorkflowDefinitionSpec(
            workflow_key="main",
            steps=[
                WorkflowStepSpec(
                    step_key="branch",
                    action_type="if_else",
                    parameters={
                        "condition_type": "template_exists",
                        "template_path": "outer.png",
                    },
                    then_steps=[
                        WorkflowStepSpec(
                            step_key="repeat",
                            action_type="bounded_repeat",
                            parameters={"count": 2, "max_count": 2},
                            steps=[
                                WorkflowStepSpec(
                                    step_key="inner-branch",
                                    action_type="if_else",
                                    parameters={
                                        "condition_type": "template_exists",
                                        "template_path": "inner.png",
                                    },
                                    then_steps=[
                                        WorkflowStepSpec(
                                            step_key="unused",
                                            action_type="tap",
                                            parameters={"x": 1, "y": 2},
                                        )
                                    ],
                                    else_steps=[
                                        WorkflowStepSpec(
                                            step_key="else-tap",
                                            action_type="tap",
                                            parameters={"x": 3, "y": 4},
                                        )
                                    ],
                                )
                            ],
                        ),
                        WorkflowStepSpec(
                            step_key="call-sub",
                            action_type="sub_workflow",
                            parameters={"workflow_key": "sub"},
                        ),
                    ],
                )
            ],
        )

        result = WorkflowEngine().execute(
            workflow,
            WorkflowExecutionContext(
                action_engine=fake_engine,
                workflow_resolver=lambda key: sub_workflow if key == "sub" else None,
            ),
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(
            [
                ("WaitTemplate", ("outer.png", 0.8, 0.25, 0.25)),
                ("WaitTemplate", ("inner.png", 0.8, 0.25, 0.25)),
                ("ClickCoordinates", (3, 4)),
                ("WaitTemplate", ("inner.png", 0.8, 0.25, 0.25)),
                ("ClickCoordinates", (3, 4)),
                ("ClickCoordinates", (9, 10)),
            ],
            fake_engine.calls,
        )

    def test_retry_policy_retries_retryable_failure(self) -> None:
        calls: list[int] = []

        def flaky_action(
            _context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            calls.append(1)
            if len(calls) == 1:
                return WorkflowStepResult(
                    step.step_key,
                    step.action_type,
                    WorkflowOutcome.RETRYABLE_FAILURE,
                    "try again",
                )
            return WorkflowStepResult(
                step.step_key,
                step.action_type,
                WorkflowOutcome.SUCCESS,
            )

        registry = ActionRegistry()
        registry.register("flaky", flaky_action)
        workflow = WorkflowDefinitionSpec(
            workflow_key="retry",
            steps=[
                WorkflowStepSpec(
                    step_key="flaky",
                    action_type="flaky",
                    retry_limit=1,
                )
            ],
        )

        result = WorkflowEngine(action_registry=registry).execute(workflow)

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(2, len(calls))
        self.assertEqual([1, 2], [step.attempt for step in result.steps])

    def test_handler_exceptions_become_structured_outcomes(self) -> None:
        def action_raises(
            _context: WorkflowExecutionContext,
            _step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            raise RuntimeError("password=hunter2")

        def condition_raises(
            _context: WorkflowExecutionContext,
            _step: WorkflowStepSpec,
        ) -> ConditionEvaluation:
            raise ValueError("token=abc123")

        def normalizer_raises(
            _context: WorkflowExecutionContext,
            _step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            raise LookupError("credential=secret")

        def postcondition_raises(
            _context: WorkflowExecutionContext,
            _step: WorkflowStepSpec,
        ) -> ConditionEvaluation:
            raise OSError("api_key=secret")

        action_registry = ActionRegistry()
        action_registry.register("explode", action_raises)
        condition_registry = ConditionRegistry()
        condition_registry.register("explode_condition", condition_raises)
        condition_registry.register("post_explode", postcondition_raises)
        normalizer_registry = NormalizerRegistry()
        normalizer_registry.register("explode_normalizer", normalizer_raises)

        cases = (
            (
                "action",
                WorkflowEngine(action_registry=action_registry),
                WorkflowDefinitionSpec(
                    workflow_key="action-exception",
                    steps=[WorkflowStepSpec(step_key="explode", action_type="explode")],
                ),
            ),
            (
                "condition",
                WorkflowEngine(condition_registry=condition_registry),
                WorkflowDefinitionSpec(
                    workflow_key="condition-exception",
                    steps=[
                        WorkflowStepSpec(
                            step_key="branch",
                            action_type="if_else",
                            parameters={"condition_type": "explode_condition"},
                            then_steps=[
                                WorkflowStepSpec(
                                    step_key="delay",
                                    action_type="delay",
                                    parameters={"seconds": 0},
                                )
                            ],
                        )
                    ],
                ),
            ),
            (
                "normalizer",
                WorkflowEngine(normalizer_registry=normalizer_registry),
                WorkflowDefinitionSpec(
                    workflow_key="normalizer-exception",
                    steps=[
                        WorkflowStepSpec(
                            step_key="normalize",
                            action_type="normalize_scene",
                            parameters={"normalizer_type": "explode_normalizer"},
                        )
                    ],
                ),
            ),
            (
                "postcondition",
                WorkflowEngine(condition_registry=condition_registry),
                WorkflowDefinitionSpec(
                    workflow_key="postcondition-exception",
                    steps=[
                        WorkflowStepSpec(
                            step_key="delay",
                            action_type="delay",
                            parameters={"seconds": 0},
                            postcondition={"condition_type": "post_explode"},
                        )
                    ],
                ),
            ),
        )

        for case_name, engine, workflow in cases:
            with self.subTest(case=case_name):
                result = engine.execute(workflow)

                self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
                self.assertIn("handler raised", result.steps[0].message)
                self.assertNotIn("hunter2", result.steps[0].message)
                self.assertNotIn("abc123", result.steps[0].message)
                self.assertNotIn("secret", result.steps[0].message)
                self.assertIn("exception", result.steps[0].data if case_name != "postcondition" else result.steps[0].data["postcondition"])

    def test_invalid_handler_returns_are_validation_failures(self) -> None:
        def invalid_action(
            _context: WorkflowExecutionContext,
            _step: WorkflowStepSpec,
        ) -> object:
            return object()

        def invalid_condition(
            _context: WorkflowExecutionContext,
            _step: WorkflowStepSpec,
        ) -> object:
            return {"matched": "yes"}

        def invalid_normalizer(
            _context: WorkflowExecutionContext,
            _step: WorkflowStepSpec,
        ) -> object:
            return {"success": "yes"}

        action_registry = ActionRegistry()
        action_registry.register("invalid_action", invalid_action)
        condition_registry = ConditionRegistry()
        condition_registry.register("invalid_condition", invalid_condition)
        normalizer_registry = NormalizerRegistry()
        normalizer_registry.register("invalid_normalizer", invalid_normalizer)

        cases = (
            (
                "action",
                WorkflowEngine(action_registry=action_registry),
                WorkflowDefinitionSpec(
                    workflow_key="invalid-action",
                    steps=[
                        WorkflowStepSpec(
                            step_key="invalid",
                            action_type="invalid_action",
                            retry_limit=1,
                        )
                    ],
                ),
            ),
            (
                "condition",
                WorkflowEngine(condition_registry=condition_registry),
                WorkflowDefinitionSpec(
                    workflow_key="invalid-condition",
                    steps=[
                        WorkflowStepSpec(
                            step_key="branch",
                            action_type="if_else",
                            parameters={"condition_type": "invalid_condition"},
                            then_steps=[
                                WorkflowStepSpec(
                                    step_key="delay",
                                    action_type="delay",
                                    parameters={"seconds": 0},
                                )
                            ],
                        )
                    ],
                ),
            ),
            (
                "normalizer",
                WorkflowEngine(normalizer_registry=normalizer_registry),
                WorkflowDefinitionSpec(
                    workflow_key="invalid-normalizer",
                    steps=[
                        WorkflowStepSpec(
                            step_key="normalize",
                            action_type="normalize_scene",
                            parameters={"normalizer_type": "invalid_normalizer"},
                        )
                    ],
                ),
            ),
        )

        for case_name, engine, workflow in cases:
            with self.subTest(case=case_name):
                result = engine.execute(workflow)

                self.assertEqual(WorkflowOutcome.VALIDATION_FAILURE, result.outcome)
                self.assertEqual(1, len(result.steps))
                self.assertIn("invalid", json.dumps(result.steps[0].data))

    def test_safe_serialization_rejects_unsupported_cycles_and_limits(self) -> None:
        class CustomMetadata:
            pass

        recursive_mapping: dict[str, object] = {}
        recursive_mapping["self"] = recursive_mapping
        recursive_list: list[object] = []
        recursive_list.append(recursive_list)
        oversized = list(range(MAX_SAFE_COLLECTION_SIZE + 1))
        deep: object = "leaf"
        for _index in range(MAX_SAFE_METADATA_DEPTH + 1):
            deep = {"next": deep}

        cases = (
            CustomMetadata(),
            recursive_mapping,
            recursive_list,
            b"bytes",
            {1, 2},
            RuntimeError("token=secret"),
            oversized,
            deep,
        )

        for value in cases:
            with self.subTest(value_type=type(value).__name__):
                result = safe_serialize_metadata({"payload": value}, source="test")
                payload = safe_json_payload({"payload": value}, source="test")

                self.assertFalse(result.ok)
                self.assertIn("serialization", json.dumps(payload))

    def test_non_serializable_handler_metadata_is_not_retried(self) -> None:
        calls: list[int] = []

        def unsafe_metadata(
            _context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            calls.append(1)
            return WorkflowStepResult(
                step.step_key,
                step.action_type,
                WorkflowOutcome.RETRYABLE_FAILURE,
                "unsafe",
                data={"metadata": object()},
            )

        registry = ActionRegistry()
        registry.register("unsafe", unsafe_metadata)
        workflow = WorkflowDefinitionSpec(
            workflow_key="unsafe-metadata",
            steps=[
                WorkflowStepSpec(
                    step_key="unsafe",
                    action_type="unsafe",
                    retry_limit=3,
                )
            ],
        )

        result = WorkflowEngine(action_registry=registry).execute(workflow)

        self.assertEqual(WorkflowOutcome.VALIDATION_FAILURE, result.outcome)
        self.assertEqual(1, len(calls))
        self.assertIn("serialization_failure", result.steps[0].data)

    def test_cancellation_checkpoints_before_after_handler_postcondition_delay_and_template_wait(self) -> None:
        def success_action(
            context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            context.cancellation_token.cancel("after handler")
            return WorkflowStepResult(step.step_key, step.action_type, WorkflowOutcome.SUCCESS)

        action_registry = ActionRegistry()
        action_registry.register("success_cancel", success_action)

        token = CancellationToken()
        token.cancel("before handler")
        before_calls: list[int] = []

        def should_not_run(
            _context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            before_calls.append(1)
            return WorkflowStepResult(step.step_key, step.action_type, WorkflowOutcome.SUCCESS)

        before_registry = ActionRegistry()
        before_registry.register("never", should_not_run)
        before_result = WorkflowEngine(action_registry=before_registry).execute(
            WorkflowDefinitionSpec(
                workflow_key="cancel-before",
                steps=[WorkflowStepSpec(step_key="never", action_type="never")],
            ),
            WorkflowExecutionContext(cancellation_token=token),
        )
        after_result = WorkflowEngine(action_registry=action_registry).execute(
            WorkflowDefinitionSpec(
                workflow_key="cancel-after",
                steps=[
                    WorkflowStepSpec(
                        step_key="success",
                        action_type="success_cancel",
                    )
                ],
            )
        )

        self.assertEqual(WorkflowOutcome.CANCELLED, before_result.outcome)
        self.assertEqual([], before_calls)
        self.assertEqual(WorkflowOutcome.CANCELLED, after_result.outcome)
        self.assertEqual("uncertain", after_result.steps[0].data["side_effect_state"])

        def postcondition_cancels(
            context: WorkflowExecutionContext,
            _step: WorkflowStepSpec,
        ) -> bool:
            context.cancellation_token.cancel("post cancelled")
            return True

        postconditions = ConditionRegistry()
        postconditions.register("cancel_post", postcondition_cancels)
        post_result = WorkflowEngine(condition_registry=postconditions).execute(
            WorkflowDefinitionSpec(
                workflow_key="cancel-post",
                steps=[
                    WorkflowStepSpec(
                        step_key="delay",
                        action_type="delay",
                        parameters={"seconds": 0},
                        postcondition={"condition_type": "cancel_post"},
                    )
                ],
            )
        )

        self.assertEqual(WorkflowOutcome.CANCELLED, post_result.outcome)
        self.assertEqual("post cancelled", post_result.message)
        self.assertEqual("uncertain", post_result.steps[0].data["postcondition"]["side_effect_state"])

        delay_token = CancellationToken()
        delay_sleeps: list[float] = []

        def cancel_during_delay(seconds: float) -> None:
            delay_sleeps.append(seconds)
            delay_token.cancel("delay cancelled")

        delay_result = WorkflowEngine().execute(
            WorkflowDefinitionSpec(
                workflow_key="cancel-delay",
                steps=[
                    WorkflowStepSpec(
                        step_key="delay",
                        action_type="delay",
                        parameters={"seconds": 5},
                    )
                ],
            ),
            WorkflowExecutionContext(
                cancellation_token=delay_token,
                sleeper=cancel_during_delay,
            ),
        )

        self.assertEqual(WorkflowOutcome.CANCELLED, delay_result.outcome)
        self.assertLessEqual(max(delay_sleeps), MAX_RUNTIME_SLEEP_CHUNK_SECONDS)

        wait_token = CancellationToken()

        class CancellingTemplateEngine(FakeActionEngine):
            def wait_for_template(
                self,
                template_path: str,
                *,
                threshold: float,
                timeout_seconds: float,
                retry_interval_seconds: float,
            ) -> dict[str, object]:
                wait_token.cancel("template cancelled")
                return super().wait_for_template(
                    template_path,
                    threshold=threshold,
                    timeout_seconds=timeout_seconds,
                    retry_interval_seconds=retry_interval_seconds,
                )

        wait_result = WorkflowEngine().execute(
            WorkflowDefinitionSpec(
                workflow_key="cancel-template",
                steps=[
                    WorkflowStepSpec(
                        step_key="wait",
                        action_type="wait",
                        parameters={
                            "condition_type": "template_exists",
                            "template_path": "ready.png",
                        },
                    )
                ],
            ),
            WorkflowExecutionContext(
                action_engine=CancellingTemplateEngine(),
                cancellation_token=wait_token,
            ),
        )

        self.assertEqual(WorkflowOutcome.CANCELLED, wait_result.outcome)
        self.assertEqual("template cancelled", wait_result.message)

    def test_cancellation_and_timeout_during_retry_backoff_are_not_retried(self) -> None:
        calls: list[int] = []

        def retryable(
            _context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            calls.append(1)
            return WorkflowStepResult(
                step.step_key,
                step.action_type,
                WorkflowOutcome.RETRYABLE_FAILURE,
                "retry me",
            )

        registry = ActionRegistry()
        registry.register("retryable", retryable)
        cancel_token = CancellationToken()
        cancel_sleeps: list[float] = []

        def cancel_backoff(seconds: float) -> None:
            cancel_sleeps.append(seconds)
            cancel_token.cancel("backoff cancelled")

        workflow = WorkflowDefinitionSpec(
            workflow_key="retry-cancel",
            steps=[
                WorkflowStepSpec(
                    step_key="retry",
                    action_type="retryable",
                    retry_limit=2,
                    retry_delay_seconds=5,
                )
            ],
        )
        cancelled = WorkflowEngine(action_registry=registry).execute(
            workflow,
            WorkflowExecutionContext(
                cancellation_token=cancel_token,
                sleeper=cancel_backoff,
            ),
        )

        self.assertEqual(WorkflowOutcome.CANCELLED, cancelled.outcome)
        self.assertEqual(1, len(calls))
        self.assertLessEqual(max(cancel_sleeps), MAX_RUNTIME_SLEEP_CHUNK_SECONDS)
        self.assertTrue(cancelled.steps[0].data["retry"]["interrupted"])

        timeout_calls: list[int] = []

        def retryable_timeout(
            _context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            timeout_calls.append(1)
            return WorkflowStepResult(
                step.step_key,
                step.action_type,
                WorkflowOutcome.RETRYABLE_FAILURE,
                "retry me",
            )

        timeout_registry = ActionRegistry()
        timeout_registry.register("retryable_timeout", retryable_timeout)
        clock = MutableClock()

        def advance(seconds: float) -> None:
            clock.advance(seconds)

        timed_out = WorkflowEngine(action_registry=timeout_registry).execute(
            WorkflowDefinitionSpec(
                workflow_key="retry-timeout",
                steps=[
                    WorkflowStepSpec(
                        step_key="retry",
                        action_type="retryable_timeout",
                        retry_limit=2,
                        retry_delay_seconds=5,
                    )
                ],
            ),
            WorkflowExecutionContext(
                clock=clock,
                sleeper=advance,
                deadline=WorkflowDeadline.from_timeout(0.5, clock),
            ),
        )

        self.assertEqual(WorkflowOutcome.TIMEOUT, timed_out.outcome)
        self.assertEqual(1, len(timeout_calls))
        self.assertEqual(0.5, timed_out.steps[0].data["retry"]["applied_delay_seconds"])

    def test_timeout_fatal_invalid_serialization_and_uncertain_side_effects_are_not_retried(self) -> None:
        cases: list[tuple[str, WorkflowOutcome, object]] = []

        def fatal(
            _context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            return WorkflowStepResult(step.step_key, step.action_type, WorkflowOutcome.FATAL_FAILURE)

        def invalid(
            _context: WorkflowExecutionContext,
            _step: WorkflowStepSpec,
        ) -> object:
            return object()

        def unsafe(
            _context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            return WorkflowStepResult(
                step.step_key,
                step.action_type,
                WorkflowOutcome.RETRYABLE_FAILURE,
                data={"bad": object()},
            )

        def uncertain(
            context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            context.cancellation_token.cancel("uncertain")
            return WorkflowStepResult(step.step_key, step.action_type, WorkflowOutcome.SUCCESS)

        cases.extend(
            [
                ("fatal", WorkflowOutcome.FATAL_FAILURE, fatal),
                ("invalid", WorkflowOutcome.VALIDATION_FAILURE, invalid),
                ("unsafe", WorkflowOutcome.VALIDATION_FAILURE, unsafe),
                ("uncertain", WorkflowOutcome.CANCELLED, uncertain),
            ]
        )

        for action_type, expected, handler in cases:
            with self.subTest(action_type=action_type):
                calls: list[int] = []

                def counted(
                    context: WorkflowExecutionContext,
                    step: WorkflowStepSpec,
                    *,
                    delegate: object = handler,
                ) -> object:
                    calls.append(1)
                    return delegate(context, step)  # type: ignore[misc, operator]

                registry = ActionRegistry()
                registry.register(action_type, counted)
                result = WorkflowEngine(action_registry=registry).execute(
                    WorkflowDefinitionSpec(
                        workflow_key=f"{action_type}-not-retried",
                        steps=[
                            WorkflowStepSpec(
                                step_key=action_type,
                                action_type=action_type,
                                retry_limit=3,
                            )
                        ],
                    )
                )

                self.assertEqual(expected, result.outcome)
                self.assertEqual(1, len(calls))

    def test_child_deadline_metadata_isolation_and_explicit_result_metadata(self) -> None:
        clock = MutableClock()
        child_deadlines: list[float | None] = []
        sibling_metadata_seen: list[object] = []
        mutation_errors: list[str] = []

        def first(
            context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            context.add_result_metadata("first", "recorded")
            try:
                context.metadata["shared"] = "mutated"  # type: ignore[index]
            except TypeError as exc:
                mutation_errors.append(type(exc).__name__)
            return WorkflowStepResult(step.step_key, step.action_type, WorkflowOutcome.SUCCESS)

        def second(
            context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            sibling_metadata_seen.append(context.metadata.get("first"))
            return WorkflowStepResult(step.step_key, step.action_type, WorkflowOutcome.SUCCESS)

        def child_action(
            context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            child_deadlines.append(context.deadline.expires_at)
            context.add_result_metadata("child", "returned")
            return WorkflowStepResult(step.step_key, step.action_type, WorkflowOutcome.SUCCESS)

        registry = ActionRegistry()
        registry.register("first", first)
        registry.register("second", second)
        registry.register("child_action", child_action)
        child = WorkflowDefinitionSpec(
            workflow_key="child",
            steps=[WorkflowStepSpec(step_key="child-action", action_type="child_action")],
        )
        workflow = WorkflowDefinitionSpec(
            workflow_key="isolation",
            steps=[
                WorkflowStepSpec(step_key="first", action_type="first"),
                WorkflowStepSpec(step_key="second", action_type="second"),
                WorkflowStepSpec(
                    step_key="child",
                    action_type="sub_workflow",
                    parameters={"workflow_key": "child"},
                    timeout_seconds=20,
                ),
            ],
        )
        base_context = WorkflowExecutionContext(
            clock=clock,
            deadline=WorkflowDeadline.from_timeout(5, clock),
            metadata={"shared": "original"},
            workflow_resolver=lambda key: child if key == "child" else None,
        )

        first_run = WorkflowEngine(action_registry=registry).execute(workflow, base_context)
        second_run = WorkflowEngine(action_registry=registry).execute(workflow, base_context)

        self.assertEqual(WorkflowOutcome.SUCCESS, first_run.outcome)
        self.assertEqual(WorkflowOutcome.SUCCESS, second_run.outcome)
        self.assertEqual(["TypeError", "TypeError"], mutation_errors)
        self.assertEqual([None, None], sibling_metadata_seen)
        self.assertEqual("original", base_context.metadata["shared"])
        self.assertEqual({"first": "recorded"}, first_run.steps[0].data["result_metadata"])
        self.assertNotIn("result_metadata", first_run.steps[2].data)
        child_step_data = first_run.steps[2].data["child_results"][0]["data"]
        self.assertEqual({"child": "returned"}, child_step_data["result_metadata"])
        self.assertTrue(all(deadline == 5.0 for deadline in child_deadlines))

    def test_retry_exhaustion_and_backoff_upper_bound_metadata(self) -> None:
        calls: list[int] = []
        sleeps: list[float] = []

        def retryable(
            _context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            calls.append(1)
            return WorkflowStepResult(
                step.step_key,
                step.action_type,
                WorkflowOutcome.RETRYABLE_FAILURE,
                "still failing",
            )

        registry = ActionRegistry()
        registry.register("retryable", retryable)
        workflow = WorkflowDefinitionSpec(
            workflow_key="retry-exhausted",
            steps=[
                WorkflowStepSpec(
                    step_key="retry",
                    action_type="retryable",
                    retry_limit=2,
                    retry_delay_seconds=60,
                    retry_backoff_multiplier=MAX_RETRY_BACKOFF_MULTIPLIER,
                    max_retry_delay_seconds=1.5,
                )
            ],
        )

        result = WorkflowEngine(action_registry=registry).execute(
            workflow,
            WorkflowExecutionContext(sleeper=sleeps.append),
        )

        self.assertEqual(WorkflowOutcome.RETRYABLE_FAILURE, result.outcome)
        self.assertEqual(3, len(calls))
        self.assertEqual([1, 2, 3], [step.attempt for step in result.steps])
        self.assertEqual(1.5, result.steps[0].data["retry"]["applied_delay_seconds"])
        self.assertEqual(60.0, result.steps[0].data["retry"]["requested_delay_seconds"])
        self.assertIn("retry_exhausted", result.steps[-1].data)
        self.assertTrue(all(0 < sleep <= MAX_RUNTIME_SLEEP_CHUNK_SECONDS for sleep in sleeps))

    def test_validation_rejects_retry_and_control_flow_limits(self) -> None:
        workflow = WorkflowDefinitionSpec(
            workflow_key="limits",
            config={
                "max_steps": 9999,
                "max_sub_workflow_depth": MAX_SUB_WORKFLOW_DEPTH + 1,
            },
            steps=[
                WorkflowStepSpec(
                    step_key="repeat",
                    action_type="bounded_repeat",
                    parameters={
                        "count": MAX_REPEAT_ITERATIONS + 1,
                        "max_count": MAX_REPEAT_ITERATIONS + 1,
                    },
                    steps=[
                        WorkflowStepSpec(
                            step_key="tap-child",
                            action_type="tap",
                            parameters={"x": 1, "y": 2},
                        )
                    ],
                ),
                WorkflowStepSpec(
                    step_key="retry",
                    action_type="tap",
                    parameters={"x": 3, "y": 4},
                    retry_limit=MAX_RETRY_LIMIT + 1,
                    retry_delay_seconds=MAX_RETRY_DELAY_SECONDS + 1,
                    retry_backoff_multiplier=MAX_RETRY_BACKOFF_MULTIPLIER + 1,
                    max_retry_delay_seconds=MAX_CALCULATED_BACKOFF_SECONDS + 1,
                ),
            ],
        )

        errors = WorkflowEngine().validation_errors(workflow)

        self.assertTrue(any("max_steps" in error for error in errors))
        self.assertTrue(any("max_sub_workflow_depth" in error for error in errors))
        self.assertTrue(any("parameters.count" in error for error in errors))
        self.assertTrue(any("retry_limit" in error for error in errors))
        self.assertTrue(any("retry_delay_seconds" in error for error in errors))
        self.assertTrue(any("retry_backoff_multiplier" in error for error in errors))
        self.assertTrue(any("max_retry_delay_seconds" in error for error in errors))

    def test_parsing_and_validation_reject_non_finite_float_step_fields(self) -> None:
        fields = (
            "timeout_seconds",
            "retry_delay_seconds",
            "retry_backoff_multiplier",
            "max_retry_delay_seconds",
        )
        values = (math.nan, math.inf, -math.inf)

        for field_name in fields:
            for value in values:
                with self.subTest(stage="parse", field=field_name, value=value):
                    with self.assertRaises(WorkflowValidationError) as raised:
                        WorkflowDefinitionSpec.from_mapping(
                            {
                                "workflow_key": "non-finite",
                                "schema_version": 2,
                                "steps": [
                                    {
                                        "step_key": "tap",
                                        "action_type": "tap",
                                        "parameters": {"x": 1, "y": 2},
                                        field_name: value,
                                    }
                                ],
                            }
                        )
                    self.assertIn(field_name, str(raised.exception))
                    self.assertIn("finite number", str(raised.exception))

                with self.subTest(stage="validate", field=field_name, value=value):
                    workflow = WorkflowDefinitionSpec(
                        workflow_key="non-finite",
                        steps=[
                            WorkflowStepSpec(
                                step_key="tap",
                                action_type="tap",
                                parameters={"x": 1, "y": 2},
                                **{field_name: value},
                            )
                        ],
                    )
                    errors = WorkflowEngine().validation_errors(workflow)

                    self.assertTrue(
                        any(field_name in error and "finite number" in error for error in errors),
                        errors,
                    )

    def test_validation_rejects_non_finite_builtin_float_parameters(self) -> None:
        values = (math.nan, math.inf, -math.inf)
        cases = (
            (
                "delay.seconds",
                lambda value: WorkflowStepSpec(
                    step_key="delay",
                    action_type="delay",
                    parameters={"seconds": value},
                ),
                "seconds",
            ),
            (
                "wait.seconds",
                lambda value: WorkflowStepSpec(
                    step_key="wait",
                    action_type="wait",
                    parameters={"seconds": value},
                ),
                "seconds",
            ),
            (
                "template.threshold",
                lambda value: WorkflowStepSpec(
                    step_key="click-template",
                    action_type="click_semantic_template",
                    parameters={"template_path": "ready.png", "threshold": value},
                ),
                "threshold",
            ),
            (
                "wait-template.threshold",
                lambda value: WorkflowStepSpec(
                    step_key="wait-template",
                    action_type="wait",
                    parameters={
                        "condition_type": "template_exists",
                        "template_path": "ready.png",
                        "threshold": value,
                    },
                ),
                "threshold",
            ),
            (
                "wait-template.timeout_seconds",
                lambda value: WorkflowStepSpec(
                    step_key="wait-template",
                    action_type="wait",
                    parameters={
                        "condition_type": "template_exists",
                        "template_path": "ready.png",
                        "timeout_seconds": value,
                    },
                ),
                "timeout_seconds",
            ),
            (
                "wait-template.retry_interval_seconds",
                lambda value: WorkflowStepSpec(
                    step_key="wait-template",
                    action_type="wait",
                    parameters={
                        "condition_type": "template_exists",
                        "template_path": "ready.png",
                        "retry_interval_seconds": value,
                    },
                ),
                "retry_interval_seconds",
            ),
            (
                "postcondition.threshold",
                lambda value: WorkflowStepSpec(
                    step_key="tap",
                    action_type="tap",
                    parameters={"x": 1, "y": 2},
                    postcondition={
                        "condition_type": "template_exists",
                        "template_path": "done.png",
                        "threshold": value,
                    },
                ),
                "threshold",
            ),
        )

        for case_name, step_factory, field_name in cases:
            for value in values:
                with self.subTest(case=case_name, value=value):
                    workflow = WorkflowDefinitionSpec(
                        workflow_key="non-finite-parameters",
                        steps=[step_factory(value)],
                    )
                    errors = WorkflowEngine().validation_errors(workflow)

                    self.assertTrue(
                        any(field_name in error and "finite number" in error for error in errors),
                        errors,
                    )

    def test_validation_rejects_booleans_as_numeric_values(self) -> None:
        workflow = WorkflowDefinitionSpec(
            workflow_key="boolean-numerics",
            config={"max_steps": True},
            steps=[
                WorkflowStepSpec(
                    step_key="repeat",
                    action_type="bounded_repeat",
                    parameters={"count": True, "max_count": False},
                    steps=[
                        WorkflowStepSpec(
                            step_key="repeat-child",
                            action_type="tap",
                            parameters={"x": 1, "y": 2},
                        )
                    ],
                ),
                WorkflowStepSpec(
                    step_key="retry",
                    action_type="tap",
                    parameters={"x": 3, "y": 4},
                    timeout_seconds=True,
                    retry_limit=True,
                    retry_delay_seconds=True,
                    retry_backoff_multiplier=True,
                    max_retry_delay_seconds=True,
                ),
                WorkflowStepSpec(
                    step_key="delay",
                    action_type="delay",
                    parameters={"seconds": True},
                ),
                WorkflowStepSpec(
                    step_key="click-template",
                    action_type="click_semantic_template",
                    parameters={"template_path": "ready.png", "threshold": True},
                ),
                WorkflowStepSpec(
                    step_key="wait-template",
                    action_type="wait",
                    parameters={
                        "condition_type": "template_exists",
                        "template_path": "ready.png",
                        "threshold": True,
                        "timeout_seconds": True,
                        "retry_interval_seconds": True,
                    },
                ),
                WorkflowStepSpec(
                    step_key="postcondition",
                    action_type="tap",
                    parameters={"x": 5, "y": 6},
                    postcondition={
                        "condition_type": "template_exists",
                        "template_path": "done.png",
                        "threshold": True,
                    },
                ),
            ],
        )

        errors = WorkflowEngine().validation_errors(workflow)

        for field_name in (
            "config.max_steps",
            "parameters.count",
            "parameters.max_count",
            "timeout_seconds",
            "retry_limit",
            "retry_delay_seconds",
            "retry_backoff_multiplier",
            "max_retry_delay_seconds",
            "seconds",
            "threshold",
            "timeout_seconds",
            "retry_interval_seconds",
        ):
            with self.subTest(field=field_name):
                self.assertTrue(
                    any(field_name in error and ("integer" in error or "number" in error) for error in errors),
                    errors,
                )

    def test_validation_rejects_malformed_control_flow_types(self) -> None:
        workflow = WorkflowDefinitionSpec(
            workflow_key="malformed-types",
            config={"max_steps": True},
            steps=[
                WorkflowStepSpec(
                    step_key="repeat",
                    action_type="bounded_repeat",
                    parameters={"count": "2"},
                    steps=[
                        WorkflowStepSpec(
                            step_key="repeat-child",
                            action_type="tap",
                            parameters={"x": 1, "y": 2},
                        )
                    ],
                ),
                WorkflowStepSpec(
                    step_key="branch",
                    action_type="if_else",
                    parameters={"condition_type": 42},
                    then_steps=[
                        WorkflowStepSpec(
                            step_key="branch-child",
                            action_type="tap",
                            parameters={"x": 3, "y": 4},
                        )
                    ],
                ),
                WorkflowStepSpec(
                    step_key="sub",
                    action_type="sub_workflow",
                    parameters={"workflow_key": 7},
                ),
                WorkflowStepSpec(
                    step_key="retry",
                    action_type="tap",
                    parameters={"x": 5, "y": 6},
                    retry_limit="1",  # type: ignore[arg-type]
                ),
                WorkflowStepSpec(
                    step_key="post",
                    action_type="tap",
                    parameters={"x": 7, "y": 8},
                    postcondition={"condition_type": False},
                ),
            ],
        )

        errors = WorkflowEngine().validation_errors(
            workflow,
            WorkflowExecutionContext(workflow_resolver=lambda _key: None),
        )

        self.assertTrue(any("config.max_steps" in error and "integer" in error for error in errors))
        self.assertTrue(any("parameters.count" in error and "integer" in error for error in errors))
        self.assertTrue(any("parameters.condition_type" in error and "string" in error for error in errors))
        self.assertTrue(any("parameters.workflow_key" in error and "string" in error for error in errors))
        self.assertTrue(any("retry_limit" in error and "integer" in error for error in errors))
        self.assertTrue(any("postcondition.condition_type" in error and "string" in error for error in errors))

    def test_step_budget_rejects_negative_limits(self) -> None:
        with self.assertRaises(ValueError):
            StepBudget(max_steps=-1)
        with self.assertRaises(ValueError):
            StepBudget(max_depth=-1)
        with self.assertRaises(ValueError):
            StepBudget(max_repeat_iterations=-1)

    def test_step_deadline_converts_late_success_to_timeout_without_retry(self) -> None:
        clock = MutableClock()
        calls: list[int] = []

        def slow_success(
            context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            del step
            calls.append(1)
            clock.advance(2.0)
            return WorkflowStepResult(
                "slow",
                "slow",
                WorkflowOutcome.SUCCESS,
                data={"now": context.clock()},
            )

        registry = ActionRegistry()
        registry.register("slow", slow_success)
        workflow = WorkflowDefinitionSpec(
            workflow_key="timeout",
            steps=[
                WorkflowStepSpec(
                    step_key="slow",
                    action_type="slow",
                    timeout_seconds=1,
                    retry_limit=1,
                )
            ],
        )

        result = WorkflowEngine(action_registry=registry).execute(
            workflow,
            WorkflowExecutionContext(clock=clock),
        )

        self.assertEqual(WorkflowOutcome.TIMEOUT, result.outcome)
        self.assertIn("deadline", result.message)
        self.assertEqual(1, len(calls))

    def test_cancellation_token_stops_before_next_step(self) -> None:
        token = CancellationToken()
        calls: list[str] = []

        def cancel_after_first(seconds: float) -> None:
            del seconds
            token.cancel("user stopped")

        workflow = WorkflowDefinitionSpec(
            workflow_key="cancel",
            steps=[
                WorkflowStepSpec(
                    step_key="delay",
                    action_type="delay",
                    parameters={"seconds": 1},
                ),
                WorkflowStepSpec(
                    step_key="tap",
                    action_type="tap",
                    parameters={"x": 1, "y": 2},
                ),
            ],
        )

        result = WorkflowEngine().execute(
            workflow,
            WorkflowExecutionContext(
                action_engine=FakeActionEngine(),
                cancellation_token=token,
                sleeper=cancel_after_first,
                metadata={"calls": calls},
            ),
        )

        self.assertEqual(WorkflowOutcome.CANCELLED, result.outcome)
        self.assertEqual("user stopped", result.message)

    def test_persistence_records_runs_and_resume_skips_completed_step(self) -> None:
        fake_engine = FakeActionEngine()
        workflow = WorkflowDefinitionSpec(
            workflow_key="persisted",
            steps=[
                WorkflowStepSpec(
                    step_key="tap",
                    action_type="tap",
                    parameters={"x": 7, "y": 8},
                    workflow_step_id=None,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "workflow.sqlite3")
            db.initialize()
            jobs = JobRepository(db)
            job_runs = JobRunRepository(db)
            step_runs = StepRunRepository(db)
            job_id = jobs.save(
                Job(
                    idempotency_key="job-1",
                    job_type="workflow",
                    scheduled_for="2026-01-01T00:00:00",
                )
            )
            recorder = WorkflowRunRepositoryRecorder(job_runs, step_runs)
            context = WorkflowExecutionContext(
                action_engine=fake_engine,
                persistence=recorder,
                job_id=job_id,
                run_key="run-1",
            )

            first = WorkflowEngine().execute(workflow, context)
            second = WorkflowEngine().execute(
                workflow,
                WorkflowExecutionContext(
                    action_engine=fake_engine,
                    persistence=recorder,
                    job_id=job_id,
                    run_key="run-1",
                ),
            )
            stored_run = job_runs.get_by_key("run-1")
            stored_steps = step_runs.list_for_job_run(first.job_run_id or 0)

            self.assertEqual(WorkflowOutcome.SUCCESS, first.outcome)
            self.assertEqual(WorkflowOutcome.SUCCESS, second.outcome)
            self.assertEqual([("ClickCoordinates", (7, 8))], fake_engine.calls)
            self.assertIsNotNone(stored_run)
            self.assertEqual("completed", stored_run.status)  # type: ignore[union-attr]
            self.assertEqual(["completed"], [step.status for step in stored_steps])
            self.assertEqual(
                "SUCCESS",
                json.loads(stored_run.result_json)["outcome"],  # type: ignore[union-attr]
            )
            db.close()

    def test_atomic_step_start_rollback_removes_partial_step_run(self) -> None:
        workflow = self._workflow_with_tap()
        with tempfile.TemporaryDirectory() as temp_dir:
            db, job_runs, step_runs, job_id, _recorder = self._database_job_context(
                temp_dir,
                run_key="atomic-start",
            )
            recorder = FailingStepStartRecorder(job_runs, step_runs)
            context = WorkflowExecutionContext(
                persistence=recorder,
                job_id=job_id,
                run_key="atomic-start",
            )
            start = recorder.start_workflow_run(workflow, context, "2026-01-01T00:00:00")
            self.assertIsInstance(start, int)

            with self.assertRaisesRegex(RuntimeError, "step start"):
                recorder.start_step_run(
                    start,
                    workflow.steps[0],
                    1,
                    "2026-01-01T00:00:01",
                )

            self.assertEqual([], step_runs.list_for_job_run(start))
            stored_run = job_runs.get(start)
            self.assertIsNotNone(stored_run)
            self.assertEqual(
                RecoveryPhase.NOT_STARTED.value,
                json.loads(stored_run.result_json)["recovery"]["phase"],  # type: ignore[union-attr]
            )
            db.close()

    def test_atomic_step_finish_rollback_preserves_running_step_and_job_phase(self) -> None:
        workflow = self._workflow_with_tap()
        with tempfile.TemporaryDirectory() as temp_dir:
            db, job_runs, step_runs, job_id, _recorder = self._database_job_context(
                temp_dir,
                run_key="atomic-finish",
            )
            recorder = FailingStepFinishRecorder(job_runs, step_runs)
            context = WorkflowExecutionContext(
                persistence=recorder,
                job_id=job_id,
                run_key="atomic-finish",
            )
            start = recorder.start_workflow_run(workflow, context, "2026-01-01T00:00:00")
            self.assertIsInstance(start, int)
            step_run_id = recorder.start_step_run(
                start,
                workflow.steps[0],
                1,
                "2026-01-01T00:00:01",
            )
            self.assertIsInstance(step_run_id, int)
            result = WorkflowStepResult(
                step_key="tap",
                action_type="tap",
                outcome=WorkflowOutcome.SUCCESS,
                attempt=1,
                started_at="2026-01-01T00:00:01",
                finished_at="2026-01-01T00:00:02",
            )

            with self.assertRaisesRegex(RuntimeError, "step finish"):
                recorder.finish_step_run(step_run_id, result)

            stored_step = step_runs.get(step_run_id)
            stored_run = job_runs.get(start)
            self.assertEqual("running", stored_step.status)  # type: ignore[union-attr]
            self.assertEqual(
                RecoveryPhase.PRECONDITION_VERIFIED.value,
                json.loads(stored_step.result_json)["recovery"]["phase"],  # type: ignore[union-attr]
            )
            self.assertEqual(
                RecoveryPhase.PRECONDITION_VERIFIED.value,
                json.loads(stored_run.result_json)["recovery"]["phase"],  # type: ignore[union-attr]
            )
            db.close()

    def test_repository_failure_after_handler_success_does_not_report_success(self) -> None:
        workflow = self._workflow_with_tap()
        fake_engine = FakeActionEngine()
        with tempfile.TemporaryDirectory() as temp_dir:
            db, job_runs, step_runs, job_id, _recorder = self._database_job_context(
                temp_dir,
                run_key="finish-fails",
            )
            recorder = FailingStepFinishRecorder(job_runs, step_runs)

            result = WorkflowEngine().execute(
                workflow,
                WorkflowExecutionContext(
                    action_engine=fake_engine,
                    persistence=recorder,
                    job_id=job_id,
                    run_key="finish-fails",
                ),
            )

            self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
            self.assertEqual([("ClickCoordinates", (7, 8))], fake_engine.calls)
            stored_steps = step_runs.list_for_job_run(result.job_run_id or 0)
            self.assertEqual(["running"], [step.status for step in stored_steps])
            self.assertEqual("failed", job_runs.get(result.job_run_id or 0).status)  # type: ignore[union-attr]
            db.close()

    def test_interrupted_before_side_effect_recovers_without_duplicate_attempt(self) -> None:
        workflow = self._workflow_with_tap()
        fake_engine = FakeActionEngine()
        with tempfile.TemporaryDirectory() as temp_dir:
            db, job_runs, step_runs, job_id, recorder = self._database_job_context(
                temp_dir,
                run_key="before-side-effect",
            )
            context = WorkflowExecutionContext(
                persistence=recorder,
                job_id=job_id,
                run_key="before-side-effect",
            )
            start = recorder.start_workflow_run(workflow, context, "2026-01-01T00:00:00")
            self.assertIsInstance(start, int)
            recorder.start_step_run(start, workflow.steps[0], 1, "2026-01-01T00:00:01")

            result = WorkflowEngine().execute(
                workflow,
                WorkflowExecutionContext(
                    action_engine=fake_engine,
                    persistence=recorder,
                    job_id=job_id,
                    run_key="before-side-effect",
                ),
            )

            self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
            self.assertEqual([("ClickCoordinates", (7, 8))], fake_engine.calls)
            self.assertEqual(1, len(step_runs.list_for_job_run(start)))
            self.assertEqual(1, len(job_runs.list_for_job(job_id)))
            db.close()

    def test_side_effect_uncertain_recovery_blocks_without_reexecuting_action(self) -> None:
        workflow = self._workflow_with_tap()
        fake_engine = FakeActionEngine()
        with tempfile.TemporaryDirectory() as temp_dir:
            db, job_runs, step_runs, job_id, recorder = self._database_job_context(
                temp_dir,
                run_key="uncertain",
            )
            context = WorkflowExecutionContext(
                persistence=recorder,
                job_id=job_id,
                run_key="uncertain",
            )
            start = recorder.start_workflow_run(workflow, context, "2026-01-01T00:00:00")
            self.assertIsInstance(start, int)
            step_run_id = recorder.start_step_run(
                start,
                workflow.steps[0],
                1,
                "2026-01-01T00:00:01",
            )
            recorder.mark_step_phase(
                step_run_id,
                workflow.steps[0],
                1,
                RecoveryPhase.SIDE_EFFECT_STARTED,
            )

            result = WorkflowEngine().execute(
                workflow,
                WorkflowExecutionContext(
                    action_engine=fake_engine,
                    persistence=recorder,
                    job_id=job_id,
                    run_key="uncertain",
                ),
            )

            self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
            self.assertEqual([], fake_engine.calls)
            stored_step = step_runs.get(step_run_id or 0)
            self.assertEqual("failed", stored_step.status)  # type: ignore[union-attr]
            self.assertEqual(
                RecoveryPhase.SIDE_EFFECT_UNCERTAIN.value,
                json.loads(stored_step.result_json)["recovery"]["phase"],  # type: ignore[union-attr]
            )
            db.close()

    def test_recovery_rechecks_postcondition_without_reexecuting_action(self) -> None:
        workflow = self._workflow_with_tap(
            postcondition={
                "condition_type": "template_exists",
                "template_path": "confirmed.png",
                "timeout_seconds": 0.25,
            }
        )
        fake_engine = FakeActionEngine()
        with tempfile.TemporaryDirectory() as temp_dir:
            db, _job_runs, step_runs, job_id, recorder = self._database_job_context(
                temp_dir,
                run_key="postcondition-recovery",
            )
            context = WorkflowExecutionContext(
                persistence=recorder,
                job_id=job_id,
                run_key="postcondition-recovery",
            )
            start = recorder.start_workflow_run(workflow, context, "2026-01-01T00:00:00")
            self.assertIsInstance(start, int)
            step_run_id = recorder.start_step_run(
                start,
                workflow.steps[0],
                1,
                "2026-01-01T00:00:01",
            )
            recorder.mark_step_phase(
                step_run_id,
                workflow.steps[0],
                1,
                RecoveryPhase.SIDE_EFFECT_STARTED,
            )

            result = WorkflowEngine().execute(
                workflow,
                WorkflowExecutionContext(
                    action_engine=fake_engine,
                    persistence=recorder,
                    job_id=job_id,
                    run_key="postcondition-recovery",
                ),
            )

            self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
            self.assertEqual(
                [("WaitTemplate", ("confirmed.png", 0.8, 0.25, 0.25))],
                fake_engine.calls,
            )
            self.assertEqual("completed", step_runs.get(step_run_id or 0).status)  # type: ignore[union-attr]
            db.close()

    def test_malformed_running_metadata_returns_structured_recovery_failure(self) -> None:
        workflow = self._workflow_with_tap()
        fake_engine = FakeActionEngine()
        with tempfile.TemporaryDirectory() as temp_dir:
            db, job_runs, _step_runs, job_id, recorder = self._database_job_context(
                temp_dir,
                run_key="malformed",
            )
            with db.transaction():
                db.execute(
                    """
                    INSERT INTO job_runs(job_id, run_key, status, attempt, started_at, result_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        "malformed",
                        "running",
                        1,
                        "2026-01-01T00:00:00",
                        "{",
                    ),
                )

            result = WorkflowEngine().execute(
                workflow,
                WorkflowExecutionContext(
                    action_engine=fake_engine,
                    persistence=recorder,
                    job_id=job_id,
                    run_key="malformed",
                ),
            )

            self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
            self.assertIn("malformed JSON", result.message)
            self.assertEqual([], fake_engine.calls)
            self.assertEqual("failed", job_runs.get_by_key("malformed").status)  # type: ignore[union-attr]
            db.close()

    def test_repeated_restart_recovery_is_deterministic_and_creates_no_duplicates(self) -> None:
        workflow = self._workflow_with_tap()
        fake_engine = FakeActionEngine()
        with tempfile.TemporaryDirectory() as temp_dir:
            db, job_runs, step_runs, job_id, recorder = self._database_job_context(
                temp_dir,
                run_key="deterministic",
            )
            context = WorkflowExecutionContext(
                persistence=recorder,
                job_id=job_id,
                run_key="deterministic",
            )
            start = recorder.start_workflow_run(workflow, context, "2026-01-01T00:00:00")
            self.assertIsInstance(start, int)
            step_run_id = recorder.start_step_run(
                start,
                workflow.steps[0],
                1,
                "2026-01-01T00:00:01",
            )
            recorder.mark_step_phase(
                step_run_id,
                workflow.steps[0],
                1,
                RecoveryPhase.SIDE_EFFECT_STARTED,
            )

            first = WorkflowEngine().execute(
                workflow,
                WorkflowExecutionContext(
                    action_engine=fake_engine,
                    persistence=recorder,
                    job_id=job_id,
                    run_key="deterministic",
                ),
            )
            second = WorkflowEngine().execute(
                workflow,
                WorkflowExecutionContext(
                    action_engine=fake_engine,
                    persistence=recorder,
                    job_id=job_id,
                    run_key="deterministic",
                ),
            )

            self.assertEqual(WorkflowOutcome.BLOCKED, first.outcome)
            self.assertEqual(WorkflowOutcome.BLOCKED, second.outcome)
            self.assertEqual([], fake_engine.calls)
            self.assertEqual(1, len(job_runs.list_for_job(job_id)))
            self.assertEqual(1, len(step_runs.list_for_job_run(start)))
            db.close()

    def test_public_import_facade_exposes_split_modules(self) -> None:
        from rok_assistant import workflow_engine as public

        self.assertIs(public.WorkflowRunRepositoryRecorder, WorkflowRunRepositoryRecorder)
        self.assertIs(public.LegacyAutomationTaskAdapter, LegacyAutomationTaskAdapter)
        self.assertIs(public.RecoveryPhase, RecoveryPhase)
        self.assertTrue(hasattr(public, "StepRecoveryDecision"))

    def test_legacy_adapter_converts_nested_if_blocks(self) -> None:
        workflow = LegacyAutomationTaskAdapter().to_workflow(
            Task(id=1, name="Legacy"),
            [
                TaskStep(order=1, action_type="IfTemplateExists", parameters={"template_path": "a.png"}),
                TaskStep(order=2, action_type="IfTemplateExists", parameters={"template_path": "b.png"}),
                TaskStep(order=3, action_type="ClickCoordinates", parameters={"x": 1, "y": 2}),
                TaskStep(order=4, action_type="EndIf", parameters={}),
                TaskStep(order=5, action_type="EndIf", parameters={}),
            ],
        )

        self.assertEqual("if_else", workflow.steps[0].action_type)
        self.assertEqual("if_else", workflow.steps[0].then_steps[0].action_type)

    def test_step_budget_blocks_unbounded_work(self) -> None:
        workflow = WorkflowDefinitionSpec(
            workflow_key="budget",
            steps=[
                WorkflowStepSpec(
                    step_key="repeat",
                    action_type="bounded_repeat",
                    parameters={"count": 3, "max_count": 3},
                    steps=[
                        WorkflowStepSpec(
                            step_key="tap",
                            action_type="tap",
                            parameters={"x": 1, "y": 2},
                        )
                    ],
                )
            ],
        )

        result = WorkflowEngine().execute(
            workflow,
            WorkflowExecutionContext(
                action_engine=FakeActionEngine(),
                budget=StepBudget(max_steps=2),
            ),
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertIn("budget", result.message)

    def test_semantic_template_resolver_avoids_hard_coded_paths(self) -> None:
        fake_engine = FakeActionEngine()
        workflow = WorkflowDefinitionSpec(
            workflow_key="template-key",
            steps=[
                WorkflowStepSpec(
                    step_key="click-help",
                    action_type="click_semantic_template",
                    parameters={"template_key": "help", "threshold": 0.95},
                )
            ],
        )

        result = WorkflowEngine().execute(
            workflow,
            WorkflowExecutionContext(
                action_engine=fake_engine,
                template_resolver=lambda key: SemanticTemplate(
                    template_key=key,
                    file_path=f"{key}.png",
                    threshold=0.7,
                ),
            ),
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual([("ClickTemplate", ("help.png", 0.95))], fake_engine.calls)

    def test_postcondition_verification_can_fail_successful_action(self) -> None:
        fake_engine = FakeActionEngine()
        fake_engine.template_matches["done.png"] = False
        workflow = WorkflowDefinitionSpec(
            workflow_key="postcondition",
            steps=[
                WorkflowStepSpec(
                    step_key="tap",
                    action_type="tap",
                    parameters={"x": 1, "y": 2},
                    postcondition={
                        "condition_type": "template_exists",
                        "template_path": "done.png",
                    },
                )
            ],
        )

        result = WorkflowEngine().execute(
            workflow,
            WorkflowExecutionContext(action_engine=fake_engine),
        )

        self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
        self.assertEqual(
            [
                ("ClickCoordinates", (1, 2)),
                ("WaitTemplate", ("done.png", 0.8, 0.25, 0.25)),
            ],
            fake_engine.calls,
        )

    def test_normalize_scene_uses_injected_boundary(self) -> None:
        workflow = WorkflowDefinitionSpec(
            workflow_key="normalize",
            steps=[
                WorkflowStepSpec(
                    step_key="normalize-home",
                    action_type="normalize_scene",
                )
            ],
        )

        def normalizer(
            _context: WorkflowExecutionContext,
            step: WorkflowStepSpec,
        ) -> WorkflowStepResult:
            return WorkflowStepResult(
                step.step_key,
                step.action_type,
                WorkflowOutcome.SUCCESS,
                data={"scene": "home"},
            )

        result = WorkflowEngine().execute(
            workflow,
            WorkflowExecutionContext(scene_normalizer=normalizer),
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual({"scene": "home"}, result.steps[0].data)


if __name__ == "__main__":
    unittest.main()
