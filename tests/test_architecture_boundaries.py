from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
SRC_ROOT = SRC_PATH / "rok_assistant"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imported_modules(path: Path) -> set[str]:
    tree = parse_module(path)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def imports_module(path: Path, forbidden_prefixes: tuple[str, ...]) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imported_modules(path)
        for prefix in forbidden_prefixes
    )


def test_application_services_do_not_import_pyqt_or_gui_modules() -> None:
    forbidden_prefixes = ("PyQt6", "rok_assistant.gui")
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in sorted((SRC_ROOT / "application").glob("*.py"))
        if imports_module(path, forbidden_prefixes)
    ]

    assert offenders == []


def test_extracted_application_services_import_without_qt_setup() -> None:
    task_queue = importlib.import_module("rok_assistant.application.task_queue")
    automation = importlib.import_module("rok_assistant.application.automation")

    assert task_queue.TaskQueueViewModel is not None
    assert task_queue.TaskExecutionService is not None
    assert automation.AutomationViewModel is not None


def test_task_queue_and_automation_gui_do_not_import_lower_level_engines() -> None:
    forbidden_prefixes = (
        "rok_assistant.db.repositories",
        "rok_assistant.scheduler",
        "rok_assistant.task_engine",
        "rok_assistant.tasks.resource_search_workflow",
        "rok_assistant.workflow_engine",
        "rok_assistant.workflow_runtime",
    )
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (
            SRC_ROOT / "gui" / "task_queue.py",
            SRC_ROOT / "gui" / "automation.py",
        )
        if imports_module(path, forbidden_prefixes)
    ]

    assert offenders == []


def test_gui_modules_only_define_presentation_classes() -> None:
    allowed_suffixes = ("Dialog", "Label", "Window", "Widget", "Worker")
    offenders: list[str] = []
    for path in sorted((SRC_ROOT / "gui").glob("*.py")):
        tree = parse_module(path)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.endswith(allowed_suffixes):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{node.name}")

    assert offenders == []


def test_task_queue_and_automation_business_logic_lives_in_application_services() -> None:
    task_queue_imports = imported_modules(SRC_ROOT / "gui" / "task_queue.py")
    automation_imports = imported_modules(SRC_ROOT / "gui" / "automation.py")
    task_queue_tree = parse_module(SRC_ROOT / "application" / "task_queue.py")
    automation_tree = parse_module(SRC_ROOT / "application" / "automation.py")

    task_queue_classes = {
        node.name for node in task_queue_tree.body if isinstance(node, ast.ClassDef)
    }
    automation_classes = {
        node.name for node in automation_tree.body if isinstance(node, ast.ClassDef)
    }

    assert "rok_assistant.application.task_queue" in task_queue_imports
    assert "rok_assistant.application.automation" in automation_imports
    assert {"TaskQueueViewModel", "TaskExecutionService"}.issubset(task_queue_classes)
    assert "AutomationViewModel" in automation_classes
