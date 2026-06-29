from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
LOG_DIR = RUNTIME_DIR / "logs"
BACKUP_DIR = RUNTIME_DIR / "backups"
SCREENSHOT_DIR = RUNTIME_DIR / "screenshots"
TEMPLATE_DIR = RUNTIME_DIR / "assets" / "templates"


def ensure_runtime_dirs() -> None:
    for directory in (RUNTIME_DIR, LOG_DIR, BACKUP_DIR, SCREENSHOT_DIR, TEMPLATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
