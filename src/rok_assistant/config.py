from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import CONFIG_DIR, resolve_project_path


DEFAULT_CONFIG: dict[str, Any] = {
    "database": {"path": "runtime/rok_assistant.sqlite3"},
    "logging": {"level": "INFO", "file": "runtime/logs/app.log"},
    "emulator": {"memu_install_path": r"C:\MEmu\Microvirt\MEmu"},
    "scheduler": {
        "max_workers": 5,
        "max_active_instances": 5,
        "retry_delay_minutes": 10,
        "pre_launch_minutes": 2,
        "poll_interval_seconds": 5,
    },
    "watchdog": {
        "game_package": "com.lilithgame.roc.gp",
        "game_activity": "com.lilithgame.roc.gp/.UnityPlayerActivity",
        "same_screen_timeout_seconds": 120,
        "same_screen_max_observations": 3,
        "phase_timeouts": {
            "reconnect_adb": 15,
            "send_back": 5,
            "normalize_home": 5,
            "relaunch_game": 30,
            "restart_emulator": 120,
            "open_incident": 5,
        },
    },
    "gathering": {
        "preferred_resource_levels": [8, 7, 6],
        "minimum_resource_level": 6,
    },
    "alliance_pit": {
        "enabled_resource_types": ["FOOD", "WOOD", "STONE", "GOLD"],
        "march_preset": "default",
    },
    "plugins": {"packages": ["rok_assistant.plugins"]},
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class AppConfig:
    path: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        config_path = path or CONFIG_DIR / "app_config.json"
        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")

        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(path=config_path, data=deep_merge(DEFAULT_CONFIG, loaded))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def get(self, dotted_key: str, default: Any = None) -> Any:
        current: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def set(self, dotted_key: str, value: Any) -> None:
        current = self.data
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    @property
    def database_path(self) -> Path:
        return resolve_project_path(self.get("database.path"))

    @property
    def log_file(self) -> Path:
        return resolve_project_path(self.get("logging.file"))

    @property
    def plugin_packages(self) -> list[str]:
        return list(self.get("plugins.packages", []))
