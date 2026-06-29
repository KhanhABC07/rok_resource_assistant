from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.db.models import TaskStep  # noqa: E402
from rok_assistant.task_engine import TaskRunner  # noqa: E402
from rok_assistant.tasks.resource_search_workflow import (  # noqa: E402
    ResourceSearchWorkflow,
    ResourceType,
    check_template_readiness,
)


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


if __name__ == "__main__":
    unittest.main()
