from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.db.database import Database  # noqa: E402
from rok_assistant.db.models import Character, Instance, Job, TaskStep  # noqa: E402
from rok_assistant.db.repositories import (  # noqa: E402
    CharacterRepository,
    IncidentRepository,
    InstanceRepository,
    JobRepository,
    JobRunRepository,
    MarchRepository,
    StepRunRepository,
)
from rok_assistant.task_engine import TaskRunner  # noqa: E402
from rok_assistant.tasks.resource_search_workflow import (  # noqa: E402
    RESOURCE_GATHERING_STATES,
    MarchAvailability,
    MarchDispatchResult,
    ResourceGatheringActionResult,
    ResourceGatheringConfig,
    ResourceGatheringRequest,
    ResourceGatheringWorkflow,
    ResourceNodeSearchResult,
    ResourcePreference,
    ResourceSearchWorkflow,
    ResourceType,
    check_template_readiness,
)
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


def step_data(
    workflow: ResourceSearchWorkflow,
) -> list[tuple[int, str, dict[str, object]]]:
    return [
        (step.order, step.action_type, step.parameters or {})
        for step in workflow.to_task_steps()
    ]


class ResourceSearchWorkflowTest(unittest.TestCase):
    def test_gold_level_8_workflow_follows_ordered_game_flow(self) -> None:
        workflow = ResourceSearchWorkflow(
            resource_type=ResourceType.GOLD,
            target_level=8,
            march_required=False,
        )

        self.assertEqual(
            [
                (
                    1,
                    "ClickTemplate",
                    {
                        "template_path": (
                            "templates/resource_search/world_map_button.png"
                        )
                    },
                ),
                (
                    2,
                    "ClickTemplate",
                    {
                        "template_path": (
                            "templates/resource_search/"
                            "open_resource_search_button.png"
                        )
                    },
                ),
                (
                    3,
                    "WaitTemplate",
                    {
                        "template_path": (
                            "templates/resource_search/resource_search_panel.png"
                        )
                    },
                ),
                (
                    4,
                    "ClickTemplate",
                    {
                        "template_path": (
                            "templates/resource_search/gold_resource_icon.png"
                        )
                    },
                ),
                (
                    5,
                    "ClickTemplate",
                    {
                        "template_path": (
                            "templates/resource_search/"
                            "resource_level_8_selector.png"
                        )
                    },
                ),
                (
                    6,
                    "ClickTemplate",
                    {
                        "template_path": (
                            "templates/resource_search/"
                            "resource_search_submit_button.png"
                        )
                    },
                ),
                (
                    7,
                    "WaitTemplate",
                    {
                        "template_path": (
                            "templates/resource_search/gold_node_level_8.png"
                        )
                    },
                ),
                (
                    8,
                    "ClickTemplate",
                    {
                        "template_path": (
                            "templates/resource_search/gold_node_level_8.png"
                        )
                    },
                ),
                (
                    9,
                    "WaitTemplate",
                    {
                        "template_path": (
                            "templates/resource_search/gather_button.png"
                        )
                    },
                ),
                (
                    10,
                    "ClickTemplate",
                    {
                        "template_path": (
                            "templates/resource_search/gather_button.png"
                        )
                    },
                ),
                (
                    11,
                    "WaitTemplate",
                    {
                        "template_path": (
                            "templates/resource_search/new_troop_window.png"
                        )
                    },
                ),
                (
                    12,
                    "ClickTemplate",
                    {
                        "template_path": (
                            "templates/resource_search/"
                            "new_troop_march_button.png"
                        )
                    },
                ),
                (
                    13,
                    "WaitTemplate",
                    {
                        "template_path": (
                            "templates/resource_search/"
                            "march_started_indicator.png"
                        )
                    },
                ),
            ],
            step_data(workflow),
        )

    def test_confirm_is_not_generated(self) -> None:
        templates = [
            str((step.parameters or {}).get("template_path", ""))
            for step in ResourceSearchWorkflow("GOLD", 8).to_task_steps()
        ]

        self.assertFalse(any("confirm" in template.lower() for template in templates))

    def test_open_search_and_submit_search_use_separate_templates(self) -> None:
        templates = [
            str((step.parameters or {}).get("template_path", ""))
            for step in ResourceSearchWorkflow(
                "GOLD", 8, march_required=False
            ).to_task_steps()
        ]

        self.assertIn(
            "templates/resource_search/open_resource_search_button.png",
            templates,
        )
        self.assertIn(
            "templates/resource_search/resource_search_submit_button.png",
            templates,
        )

    def test_all_supported_resources_generate_natural_resource_assets(self) -> None:
        for resource_type in ResourceType:
            with self.subTest(resource_type=resource_type):
                resource_name = resource_type.value.lower()
                templates = [
                    str((step.parameters or {}).get("template_path", ""))
                    for step in ResourceSearchWorkflow(
                        resource_type,
                        5,
                        fallback_enabled=True,
                        march_required=False,
                    ).to_task_steps()
                ]

                self.assertIn(
                    f"templates/resource_search/{resource_name}_resource_icon.png",
                    templates,
                )
                self.assertIn(
                    f"templates/resource_search/{resource_name}_node_level_5.png",
                    templates,
                )

    def test_no_free_march_check_is_optional(self) -> None:
        required_steps = ResourceSearchWorkflow("WOOD", 3).to_task_steps()
        optional_steps = ResourceSearchWorkflow(
            "WOOD", 3, march_required=False
        ).to_task_steps()

        self.assertEqual(
            ["IfTemplateExists", "AbortTask", "EndIf"],
            [step.action_type for step in required_steps[:3]],
        )
        self.assertEqual(
            "templates/resource_search/no_free_march.png",
            (required_steps[0].parameters or {})["template_path"],
        )
        self.assertNotIn(
            "IfTemplateExists",
            [step.action_type for step in optional_steps],
        )

    def test_target_level_uses_explicit_placeholder_asset(self) -> None:
        templates = [
            str((step.parameters or {}).get("template_path", ""))
            for step in ResourceSearchWorkflow(
                "STONE", 6, march_required=False
            ).to_task_steps()
        ]

        self.assertIn(
            "templates/resource_search/resource_level_6_selector.png",
            templates,
        )

    def test_missing_template_files_are_not_ready(self) -> None:
        steps = ResourceSearchWorkflow(
            "GOLD", 8, march_required=False
        ).to_task_steps()
        with tempfile.TemporaryDirectory() as temp_dir:
            readiness = check_template_readiness(steps, Path(temp_dir))

        self.assertFalse(readiness.ready)
        self.assertIn(
            "templates/resource_search/world_map_button.png",
            readiness.missing_templates,
        )
        self.assertIn(
            "templates/resource_search/gold_node_level_8.png",
            readiness.missing_templates,
        )

    def test_existing_template_files_are_ready(self) -> None:
        steps = ResourceSearchWorkflow(
            "GOLD", 8, march_required=False
        ).to_task_steps()
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            for step in steps:
                template_path = str(
                    (step.parameters or {}).get("template_path", "")
                )
                if not template_path:
                    continue
                path = base_dir / template_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            readiness = check_template_readiness(steps, base_dir)

        self.assertTrue(readiness.ready)
        self.assertEqual([], readiness.missing_templates)

    def test_generated_items_are_task_step_instances(self) -> None:
        steps = ResourceSearchWorkflow("FOOD", 5).to_task_steps()

        self.assertTrue(all(isinstance(step, TaskStep) for step in steps))

    def test_generated_workflow_passes_task_engine_structural_validation(self) -> None:
        workflow = ResourceSearchWorkflow("GOLD", 8, march_required=True)
        runner = TaskRunner(object())  # type: ignore[arg-type]

        self.assertEqual("", runner.validate_steps(workflow.to_task_steps()))

    def test_invalid_resource_type_fails_validation(self) -> None:
        with self.assertRaises(ValueError):
            ResourceSearchWorkflow(resource_type="GEMS", target_level=5)

    def test_invalid_target_level_fails_validation(self) -> None:
        with self.assertRaises(ValueError):
            ResourceSearchWorkflow(resource_type="STONE", target_level=0)


class FakeResourceDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.search_levels: list[int] = []
        self.available = MarchAvailability(True, march_slot=1, available_count=1)
        self.dispatch = MarchDispatchResult(
            True,
            march_slot=1,
            dispatch_id="dispatch-1",
            expected_return_time="2026-07-07T01:00:00",
        )
        self.navigation = ResourceGatheringActionResult(True)
        self.search_timeout = False
        self.found_levels: set[int] = {8}

    def navigate_to_resource_search(self, _request, _character):
        self.calls.append("navigate_to_resource_search")
        return self.navigation

    def select_resource(self, _request, selection):
        self.calls.append(f"select_resource:{selection.resource_type.value}:{selection.level}")
        return ResourceGatheringActionResult(True)

    def search_resource(self, _request, selection):
        self.calls.append(f"search_resource:{selection.resource_type.value}:{selection.level}")
        self.search_levels.append(selection.level)
        if self.search_timeout:
            raise TimeoutError("resource search timed out")
        if selection.level not in self.found_levels:
            return ResourceNodeSearchResult(
                False,
                selection.resource_type,
                selection.level,
                message="not found",
                retryable=False,
                screenshot_path=f"runtime/screens/{selection.level}-missing.png",
            )
        return ResourceNodeSearchResult(
            True,
            selection.resource_type,
            selection.level,
            confidence=0.92,
            x=100,
            y=200,
        )

    def validate_march_availability(self, _request, selection):
        self.calls.append(f"validate_march:{selection.level}")
        return self.available

    def dispatch_gather_march(self, _request, selection, availability):
        self.calls.append(f"dispatch_march:{selection.level}:{availability.march_slot}")
        return self.dispatch

    def verify_dispatch(self, _request, dispatch):
        self.calls.append(f"verify_dispatch:{dispatch.dispatch_id}")
        return ResourceGatheringActionResult(True, data={"verified": True})


class FakeWatchdog:
    def __init__(self, *, healthy: bool = True) -> None:
        self.healthy = healthy
        self.calls: list[int | None] = []

    def monitor(
        self,
        *,
        instance_id: int,
        instance_index: int,
        instance_name: str,
        job_run_id: int | None = None,
    ) -> object:
        del instance_id, instance_index, instance_name
        self.calls.append(job_run_id)
        return SimpleNamespace(
            healthy=self.healthy,
            recovery_attempted=not self.healthy,
            circuit_opened=not self.healthy,
            observation=SimpleNamespace(
                message="unhealthy" if not self.healthy else "",
                screenshot_path="runtime/screens/unhealthy.png" if not self.healthy else "",
            ),
        )


class ResourceGatheringWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "resource.sqlite3")
        self.db.initialize()
        self.instances = InstanceRepository(self.db)
        self.characters = CharacterRepository(self.db)
        self.marches = MarchRepository(self.db)
        self.jobs = JobRepository(self.db)
        self.job_runs = JobRunRepository(self.db)
        self.step_runs = StepRunRepository(self.db)
        self.incidents = IncidentRepository(self.db)
        self.instance_id = self.instances.save(
            Instance(name="MEmu 1", instance_index=0, instance_name="MEmu 1")
        )
        self.character_id = self.characters.save(
            Character(id=None, name="Farm01", instance_id=self.instance_id)
        )
        self.driver = FakeResourceDriver()
        self.watchdog = FakeWatchdog()

    def tearDown(self) -> None:
        self.db.close()
        self.temp_dir.cleanup()

    def _job(self, key: str) -> int:
        return self.jobs.save(
            Job(
                idempotency_key=key,
                job_type="workflow",
                scheduled_for="2026-07-07T00:00:00",
            )
        )

    def _workflow(self) -> ResourceGatheringWorkflow:
        return ResourceGatheringWorkflow(
            characters=self.characters,
            marches=self.marches,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=ResourceGatheringConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "resource-run",
        preferences: tuple[ResourcePreference, ...] = (),
    ) -> ResourceGatheringRequest:
        return ResourceGatheringRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            resource_type=ResourceType.GOLD,
            target_level=8,
            resource_preferences=preferences,
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states(self) -> None:
        self.assertEqual(RESOURCE_GATHERING_STATES, self._workflow().workflow_states)

    def test_no_march_available_fails_before_dispatch_and_persists_reason(self) -> None:
        self.driver.available = MarchAvailability(
            False,
            available_count=0,
            message="No march available",
            screenshot_path="runtime/screens/no-march.png",
        )
        job_id = self._job("resource-no-march")

        result = self._workflow().execute(
            self._request(job_id=job_id, run_key="resource-no-march")
        )

        self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
        self.assertEqual("validate_march", result.result["failure_state"])
        self.assertNotIn("dispatch_march:8:1", self.driver.calls)
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual("No march available", payload["result"]["failure_reason"])

    def test_resource_not_found_fails_with_search_metadata(self) -> None:
        self.driver.found_levels = set()

        result = self._workflow().execute(self._request(run_key="resource-not-found"))

        self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
        self.assertEqual("search_resource", result.result["failure_state"])
        self.assertEqual([8], self.driver.search_levels)

    def test_lower_level_fallback_is_used_when_allowed(self) -> None:
        self.driver.found_levels = {7}
        preferences = (
            ResourcePreference(
                ResourceType.WOOD,
                target_level=8,
                minimum_level=6,
                fallback_allowed=True,
            ),
        )

        result = self._workflow().execute(
            self._request(run_key="resource-fallback", preferences=preferences)
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual([8, 7], self.driver.search_levels)
        self.assertIn("select_resource:WOOD:7", self.driver.calls)
        self.assertEqual(
            {"resource_type": "WOOD", "level": 7, "fallback_from_level": 8},
            result.result["selected_resource"],
        )

    def test_search_timeout_records_recovery_outcome(self) -> None:
        self.driver.search_timeout = True

        result = self._workflow().execute(
            self._request(
                job_id=self._job("resource-timeout"),
                run_key="resource-timeout",
            )
        )

        self.assertEqual(WorkflowOutcome.TIMEOUT, result.outcome)
        self.assertEqual("search_resource", result.result["failure_state"])
        self.assertEqual(
            {"attempted": False, "healthy": True, "circuit_opened": False},
            result.result["recovery_outcome"],
        )

    def test_dispatch_success_persists_selected_resource_and_march_metadata(self) -> None:
        result = self._workflow().execute(
            self._request(
                job_id=self._job("resource-success"),
                run_key="resource-success",
            )
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual("dispatch-1", result.result["march_dispatch"]["dispatch_id"])
        marches = self.marches.list_for_character(self.character_id)
        self.assertEqual("gathering", marches[0].status)
        self.assertEqual("2026-07-07T01:00:00", marches[0].expected_return_time)
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual("GOLD", payload["result"]["selected_resource"]["resource_type"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_recovery_after_navigation_failure_opens_incident(self) -> None:
        self.driver.navigation = ResourceGatheringActionResult(
            False,
            "navigation failed",
            retryable=False,
            screenshot_path="runtime/screens/navigation.png",
        )

        result = self._workflow().execute(
            self._request(
                job_id=self._job("resource-navigation-failure"),
                run_key="resource-navigation-failure",
            )
        )

        self.assertEqual(WorkflowOutcome.FATAL_FAILURE, result.outcome)
        self.assertEqual("navigate_to_resource_search", result.result["failure_state"])
        self.assertEqual(
            {"attempted": False, "healthy": True, "circuit_opened": False},
            result.result["recovery_outcome"],
        )
        self.assertEqual(1, len(self.incidents.list_open()))


if __name__ == "__main__":
    unittest.main()
