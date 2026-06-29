from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from rok_assistant.db.database import Database
from rok_assistant.db.models import Character, Instance, March
from rok_assistant.db.repositories import (
    CharacterRepository,
    InstanceRepository,
    MarchRepository,
    SettingsRepository,
)
from rok_assistant.paths import BACKUP_DIR


class ConfigurationService:
    def __init__(self, db: Database):
        self.db = db
        self.instances = InstanceRepository(db)
        self.characters = CharacterRepository(db)
        self.marches = MarchRepository(db)
        self.settings = SettingsRepository(db)

    def export_json(self, path: Path) -> None:
        data = {
            "version": 1,
            "exported_at": datetime.now(UTC).replace(tzinfo=None, microsecond=0).isoformat(),
            "settings": self.settings.all(),
            "instances": [],
        }
        for instance in self.instances.list_all():
            instance_data = {
                "id": instance.id,
                "name": instance.name,
                "instance_index": instance.instance_index,
                "instance_name": instance.instance_name,
                "adb_serial": instance.adb_serial,
                "adb_connected": instance.adb_connected,
                "launch_path": instance.launch_path,
                "launch_command": instance.launch_command,
                "close_command": instance.close_command,
                "enabled": instance.enabled,
                "characters": [],
            }
            characters = [
                character
                for character in self.characters.list_all()
                if character.instance_id == instance.id
            ]
            for character in characters:
                character_data = {
                    "id": character.id,
                    "name": character.name,
                    "account_name": character.account_name,
                    "enabled": character.enabled,
                    "alliance_help_enabled": character.alliance_help_enabled,
                    "alliance_donate_enabled": character.alliance_donate_enabled,
                    "gift_collection_enabled": character.gift_collection_enabled,
                    "marches": [
                        {
                            "march_slot": march.march_slot,
                            "status": march.status,
                            "next_action_time": march.next_action_time,
                            "expected_return_time": march.expected_return_time,
                        }
                        for march in self.marches.list_for_character(character.id or 0)
                    ],
                }
                instance_data["characters"].append(character_data)
            data["instances"].append(instance_data)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def import_json(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        with self.db.transaction() as connection:
            connection.execute("DELETE FROM scheduled_tasks")
            connection.execute("DELETE FROM marches")
            connection.execute("DELETE FROM characters")
            connection.execute("DELETE FROM instances")

            for key, value in data.get("settings", {}).items():
                self.settings.set(key, value)

            for item in data.get("instances", []):
                instance_id = self.instances.save(
                    Instance(
                        name=item["name"],
                        instance_index=item.get("instance_index"),
                        instance_name=item.get("instance_name", item["name"]),
                        adb_serial=item.get("adb_serial", ""),
                        adb_connected=bool(item.get("adb_connected", False)),
                        launch_path=item.get("launch_path", ""),
                        launch_command=item.get("launch_command", ""),
                        close_command=item.get("close_command", ""),
                        enabled=bool(item.get("enabled", True)),
                    )
                )
                for character_item in item.get("characters", []):
                    character_id = self.characters.save(
                        Character(
                            name=character_item["name"],
                            instance_id=instance_id,
                            account_name=character_item.get("account_name", ""),
                            enabled=bool(character_item.get("enabled", True)),
                            alliance_help_enabled=bool(
                                character_item.get("alliance_help_enabled", True)
                            ),
                            alliance_donate_enabled=bool(
                                character_item.get("alliance_donate_enabled", True)
                            ),
                            gift_collection_enabled=bool(
                                character_item.get("gift_collection_enabled", True)
                            ),
                        )
                    )
                    for march_item in character_item.get("marches", []):
                        self.marches.save(
                            March(
                                character_id=character_id,
                                march_slot=int(march_item.get("march_slot", 1)),
                                status=march_item.get("status", "idle"),
                                next_action_time=march_item.get("next_action_time"),
                                expected_return_time=march_item.get("expected_return_time"),
                            )
                        )

    def backup_database(self) -> Path:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        destination = BACKUP_DIR / f"rok_assistant_{timestamp}.sqlite3"
        shutil.copy2(self.db.path, destination)
        return destination

    def restore_database(self, backup_path: Path) -> None:
        self.db.close()
        shutil.copy2(backup_path, self.db.path)
        self.db.reopen()
