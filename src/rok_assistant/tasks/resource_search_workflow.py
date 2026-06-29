from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from rok_assistant.db.models import TaskStep
from rok_assistant.paths import PROJECT_ROOT


class ResourceType(StrEnum):
    FOOD = "FOOD"
    WOOD = "WOOD"
    STONE = "STONE"
    GOLD = "GOLD"


RESOURCE_SEARCH_TEMPLATE_ROOT = "templates/resource_search"
MIN_RESOURCE_LEVEL = 1
MAX_RESOURCE_LEVEL = 8


@dataclass(frozen=True)
class TemplateReadiness:
    ready: bool
    missing_templates: list[str]


def check_template_readiness(
    steps: list[TaskStep],
    base_dir: Path = PROJECT_ROOT,
) -> TemplateReadiness:
    missing_templates: list[str] = []
    checked_templates: set[str] = set()
    for step in sorted(steps, key=lambda item: item.order):
        template_path = str((step.parameters or {}).get("template_path", "")).strip()
        if not template_path or template_path in checked_templates:
            continue
        checked_templates.add(template_path)
        path = Path(template_path)
        resolved_path = path if path.is_absolute() else base_dir / path
        if not resolved_path.is_file():
            missing_templates.append(template_path)
    return TemplateReadiness(not missing_templates, missing_templates)


@dataclass
class ResourceSearchWorkflow:
    resource_type: ResourceType | str
    target_level: int
    fallback_enabled: bool = False
    march_required: bool = True

    def __post_init__(self) -> None:
        self.resource_type = self._validate_resource_type(self.resource_type)
        self.target_level = self._validate_target_level(self.target_level)

    def to_task_steps(self) -> list[TaskStep]:
        steps: list[TaskStep] = []

        if self.march_required:
            steps.extend(
                [
                    self._step(
                        "IfTemplateExists",
                        {"template_path": self._no_free_march_template},
                    ),
                    self._step(
                        "AbortTask",
                        {"reason": "No free march"},
                    ),
                    self._step("EndIf", {}),
                ]
            )

        steps.extend(
            [
                self._step(
                    "ClickTemplate",
                    {"template_path": self._world_map_button_template},
                ),
                self._step(
                    "ClickTemplate",
                    {"template_path": self._open_resource_search_button_template},
                ),
                self._step(
                    "WaitTemplate",
                    {"template_path": self._resource_search_panel_template},
                ),
                self._step(
                    "ClickTemplate",
                    {"template_path": self._resource_icon_template},
                ),
                self._step(
                    "ClickTemplate",
                    {"template_path": self._level_button_template},
                ),
                self._step(
                    "ClickTemplate",
                    {"template_path": self._resource_search_submit_button_template},
                ),
                self._step(
                    "WaitTemplate",
                    {"template_path": self._resource_node_template},
                ),
                self._step(
                    "ClickTemplate",
                    {"template_path": self._resource_node_template},
                ),
                self._step(
                    "WaitTemplate",
                    {"template_path": self._gather_button_template},
                ),
                self._step(
                    "ClickTemplate",
                    {"template_path": self._gather_button_template},
                ),
                self._step(
                    "WaitTemplate",
                    {"template_path": self._new_troop_window_template},
                ),
                self._step(
                    "ClickTemplate",
                    {"template_path": self._new_troop_march_button_template},
                ),
                self._step(
                    "WaitTemplate",
                    {"template_path": self._march_started_indicator_template},
                ),
            ]
        )

        for order, step in enumerate(steps, start=1):
            step.order = order
        return steps

    @property
    def _no_free_march_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/no_free_march.png"

    @property
    def _world_map_button_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/world_map_button.png"

    @property
    def _open_resource_search_button_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/open_resource_search_button.png"

    @property
    def _resource_search_panel_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/resource_search_panel.png"

    @property
    def _resource_icon_template(self) -> str:
        resource_name = self.resource_type.value.lower()
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/{resource_name}_resource_icon.png"

    @property
    def _level_button_template(self) -> str:
        # Placeholder until a real level-selector crop is supplied for each level.
        return (
            f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/"
            f"resource_level_{self.target_level}_selector.png"
        )

    @property
    def _resource_search_submit_button_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/resource_search_submit_button.png"

    @property
    def _resource_node_template(self) -> str:
        resource_name = self.resource_type.value.lower()
        return (
            f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/{resource_name}_node_level_"
            f"{self.target_level}.png"
        )

    @property
    def _gather_button_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/gather_button.png"

    @property
    def _new_troop_window_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/new_troop_window.png"

    @property
    def _new_troop_march_button_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/new_troop_march_button.png"

    @property
    def _march_started_indicator_template(self) -> str:
        return f"{RESOURCE_SEARCH_TEMPLATE_ROOT}/march_started_indicator.png"

    @staticmethod
    def _step(action_type: str, parameters: dict[str, object]) -> TaskStep:
        return TaskStep(action_type=action_type, parameters=parameters)

    @staticmethod
    def _validate_resource_type(resource_type: ResourceType | str) -> ResourceType:
        if isinstance(resource_type, ResourceType):
            return resource_type
        value = str(resource_type).strip().upper()
        try:
            return ResourceType(value)
        except ValueError as exc:
            valid = ", ".join(item.value for item in ResourceType)
            raise ValueError(
                f"Invalid resource_type: {resource_type!r}. Expected one of: {valid}."
            ) from exc

    @staticmethod
    def _validate_target_level(target_level: int) -> int:
        if isinstance(target_level, bool):
            raise ValueError("target_level must be an integer resource level.")
        try:
            value = int(target_level)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_level must be an integer resource level.") from exc
        if value < MIN_RESOURCE_LEVEL or value > MAX_RESOURCE_LEVEL:
            raise ValueError(
                f"target_level must be between {MIN_RESOURCE_LEVEL} and {MAX_RESOURCE_LEVEL}."
            )
        return value
