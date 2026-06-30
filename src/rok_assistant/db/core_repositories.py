from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .account_repositories import GameAccountRepository
from .database import Database
from .models import (
    AUTOMATION_ACTION_TYPES,
    Character,
    Instance,
    March,
    ScheduledTask,
    Task,
    TaskRunHistory,
    TaskStep,
    row_bool,
    utc_now_iso,
)


class InstanceRepository:
    def __init__(self, db: Database):
        self.db = db

    def list_all(self) -> list[Instance]:
        rows = self.db.fetch_all(
            """
            SELECT * FROM instances
            ORDER BY
                CASE WHEN instance_index IS NULL THEN 1 ELSE 0 END,
                instance_index,
                name
            """
        )
        return [self._from_row(row) for row in rows]

    def get(self, instance_id: int) -> Instance | None:
        row = self.db.fetch_one("SELECT * FROM instances WHERE id = ?", (instance_id,))
        return self._from_row(row) if row else None

    def get_by_instance_index(self, instance_index: int) -> Instance | None:
        row = self.db.fetch_one(
            "SELECT * FROM instances WHERE instance_index = ?",
            (instance_index,),
        )
        return self._from_row(row) if row else None

    def save(self, instance: Instance) -> int:
        name = (instance.name or instance.instance_name).strip()
        instance_name = (instance.instance_name or instance.name).strip()
        if instance.id is None:
            cursor = self.db.execute(
                """
                INSERT INTO instances(
                    name, instance_index, instance_name,
                    adb_serial, adb_connected,
                    launch_path, launch_command, close_command, enabled
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    instance.instance_index,
                    instance_name,
                    instance.adb_serial.strip(),
                    int(instance.adb_connected),
                    instance.launch_path.strip(),
                    instance.launch_command.strip(),
                    instance.close_command.strip(),
                    int(instance.enabled),
                ),
            )
            return int(cursor.lastrowid)

        self.db.execute(
            """
            UPDATE instances
            SET name = ?,
                instance_index = ?,
                instance_name = ?,
                adb_serial = ?,
                adb_connected = ?,
                launch_path = ?,
                launch_command = ?,
                close_command = ?,
                enabled = ?
            WHERE id = ?
            """,
            (
                name,
                instance.instance_index,
                instance_name,
                instance.adb_serial.strip(),
                int(instance.adb_connected),
                instance.launch_path.strip(),
                instance.launch_command.strip(),
                instance.close_command.strip(),
                int(instance.enabled),
                instance.id,
            ),
        )
        return instance.id

    def upsert_memu_instance(self, instance_index: int, instance_name: str) -> int:
        name = instance_name.strip()
        with self.db.transaction():
            row = self.db.fetch_one(
                """
                SELECT *
                FROM instances
                WHERE instance_index = ? OR name = ? OR instance_name = ?
                ORDER BY CASE WHEN instance_index = ? THEN 0 ELSE 1 END
                LIMIT 1
                """,
                (instance_index, name, name, instance_index),
            )
            if row is None:
                cursor = self.db.execute(
                    """
                    INSERT INTO instances(
                        name, instance_index, instance_name,
                        adb_serial, adb_connected,
                        launch_path, launch_command, close_command, enabled
                    )
                    VALUES (?, ?, ?, '', 0, '', '', '', 1)
                    """,
                    (name, instance_index, name),
                )
                return int(cursor.lastrowid)

            self.db.execute(
                """
                UPDATE instances
                SET name = ?,
                    instance_index = ?,
                    instance_name = ?,
                    launch_path = '',
                    launch_command = '',
                    close_command = ''
                WHERE id = ?
                """,
                (name, instance_index, name, row["id"]),
            )
            return int(row["id"])

    def update_adb_status(
        self,
        instance_index: int,
        adb_serial: str,
        adb_connected: bool,
    ) -> None:
        self.db.execute(
            """
            UPDATE instances
            SET adb_serial = ?, adb_connected = ?
            WHERE instance_index = ?
            """,
            (adb_serial.strip(), int(adb_connected), instance_index),
        )

    def update_adb_statuses(self, statuses: dict[int, dict[str, object]]) -> None:
        with self.db.transaction():
            for instance_index, status in statuses.items():
                self.update_adb_status(
                    instance_index,
                    str(status.get("serial") or ""),
                    bool(status.get("connected")),
                )

    def upsert_memu_instances(self, instances: list[dict[str, object]]) -> int:
        imported = 0
        with self.db.transaction():
            for item in instances:
                self.upsert_memu_instance(
                    int(item["index"]),
                    str(item["name"]),
                )
                imported += 1
        return imported

    def delete(self, instance_id: int) -> None:
        self.db.execute("DELETE FROM instances WHERE id = ?", (instance_id,))

    @staticmethod
    def _from_row(row: Any) -> Instance:
        instance_name = row["instance_name"] if "instance_name" in row.keys() else ""
        return Instance(
            id=row["id"],
            name=row["name"],
            instance_index=row["instance_index"] if "instance_index" in row.keys() else None,
            instance_name=instance_name or row["name"],
            adb_serial=row["adb_serial"] if "adb_serial" in row.keys() else "",
            adb_connected=(
                row_bool(row["adb_connected"]) if "adb_connected" in row.keys() else False
            ),
            launch_path=row["launch_path"],
            launch_command=row["launch_command"],
            close_command=row["close_command"],
            enabled=row_bool(row["enabled"]),
        )


class CharacterRepository:
    def __init__(self, db: Database):
        self.db = db

    def list_all(self, include_disabled: bool = True) -> list[Character]:
        where = "" if include_disabled else "WHERE c.enabled = 1 AND i.enabled = 1"
        rows = self.db.fetch_all(
            f"""
            SELECT c.*, i.name AS instance_name
            FROM characters c
            JOIN instances i ON i.id = c.instance_id
            {where}
            ORDER BY i.name, c.account_name, c.name
            """
        )
        return [self._from_row(row) for row in rows]

    def get(self, character_id: int) -> Character | None:
        row = self.db.fetch_one(
            """
            SELECT c.*, i.name AS instance_name
            FROM characters c
            JOIN instances i ON i.id = c.instance_id
            WHERE c.id = ?
            """,
            (character_id,),
        )
        return self._from_row(row) if row else None

    def save(self, character: Character) -> int:
        if character.instance_id is None:
            raise ValueError("Character must be assigned to an instance.")
        with self.db.transaction():
            account_name, game_account_id = self._resolve_game_account(character)
            if character.id is None:
                cursor = self.db.execute(
                    """
                    INSERT INTO characters(
                        name, instance_id, account_name, game_account_id, enabled,
                        alliance_help_enabled, alliance_donate_enabled, gift_collection_enabled
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        character.name.strip(),
                        character.instance_id,
                        account_name,
                        game_account_id,
                        int(character.enabled),
                        int(character.alliance_help_enabled),
                        int(character.alliance_donate_enabled),
                        int(character.gift_collection_enabled),
                    ),
                )
                character_id = int(cursor.lastrowid)
                MarchRepository(self.db).initialize_for_character(character_id)
                return character_id

            self.db.execute(
                """
                UPDATE characters
                SET name = ?, instance_id = ?, account_name = ?, game_account_id = ?, enabled = ?,
                    alliance_help_enabled = ?, alliance_donate_enabled = ?, gift_collection_enabled = ?
                WHERE id = ?
                """,
                (
                    character.name.strip(),
                    character.instance_id,
                    account_name,
                    game_account_id,
                    int(character.enabled),
                    int(character.alliance_help_enabled),
                    int(character.alliance_donate_enabled),
                    int(character.gift_collection_enabled),
                    character.id,
                ),
            )
            MarchRepository(self.db).initialize_for_character(character.id)
            return character.id

    def _resolve_game_account(self, character: Character) -> tuple[str, int | None]:
        account_name = character.account_name.strip()
        if character.game_account_id is not None:
            account = GameAccountRepository(self.db).get(character.game_account_id)
            if account is None:
                raise ValueError(
                    f"Game account does not exist: {character.game_account_id}"
                )
            if account_name and account.account_name.casefold() != account_name.casefold():
                raise ValueError(
                    "Character account_name does not match the referenced game account."
                )
            return account_name or account.account_name, character.game_account_id

        if not account_name:
            return "", None

        account = GameAccountRepository(self.db).get_by_name(account_name)
        if account is None or account.id is None:
            raise ValueError(
                f"Game account must exist before assigning character: {account_name}"
            )
        return account.account_name, account.id

    def delete(self, character_id: int) -> None:
        self.db.execute("DELETE FROM characters WHERE id = ?", (character_id,))

    def count_all(self) -> int:
        row = self.db.fetch_one("SELECT COUNT(*) AS total FROM characters")
        return int(row["total"] if row else 0)

    def mark_switched(self, character_id: int) -> None:
        self.db.execute(
            "UPDATE characters SET last_switch_at = ? WHERE id = ?",
            (utc_now_iso(), character_id),
        )

    @staticmethod
    def _from_row(row: Any) -> Character:
        return Character(
            id=row["id"],
            name=row["name"],
            instance_id=row["instance_id"],
            account_name=row["account_name"],
            enabled=row_bool(row["enabled"]),
            alliance_help_enabled=row_bool(row["alliance_help_enabled"]),
            alliance_donate_enabled=row_bool(row["alliance_donate_enabled"]),
            gift_collection_enabled=row_bool(row["gift_collection_enabled"]),
            instance_name=row["instance_name"] if "instance_name" in row.keys() else "",
            game_account_id=(
                row["game_account_id"] if "game_account_id" in row.keys() else None
            ),
        )


class MarchRepository:
    def __init__(self, db: Database):
        self.db = db

    def initialize_for_character(self, character_id: int) -> None:
        with self.db.transaction():
            for slot in range(1, 6):
                self.db.execute(
                    """
                    INSERT OR IGNORE INTO marches(character_id, march_slot)
                    VALUES (?, ?)
                    """,
                    (character_id, slot),
                )

    def list_for_character(self, character_id: int) -> list[March]:
        self.initialize_for_character(character_id)
        rows = self.db.fetch_all(
            """
            SELECT * FROM marches
            WHERE character_id = ?
            ORDER BY march_slot
            """,
            (character_id,),
        )
        return [self._from_row(row) for row in rows]

    def save(self, march: March) -> int:
        if march.character_id is None:
            raise ValueError("March must belong to a character.")

        cursor = self.db.execute(
            """
            INSERT INTO marches(
                character_id, march_slot, status, next_action_time, expected_return_time
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(character_id, march_slot) DO UPDATE SET
                status = excluded.status,
                next_action_time = excluded.next_action_time,
                expected_return_time = excluded.expected_return_time
            """,
            (
                march.character_id,
                march.march_slot,
                march.status,
                march.next_action_time,
                march.expected_return_time,
            ),
        )
        return int(cursor.lastrowid or march.id or 0)

    @staticmethod
    def _from_row(row: Any) -> March:
        return March(
            id=row["id"],
            character_id=row["character_id"],
            march_slot=row["march_slot"],
            resource_type=row["resource_type"],
            resource_source=row["resource_source"],
            status=row["status"],
            next_action_time=row["next_action_time"],
            expected_return_time=row["expected_return_time"],
        )


class TaskRepository:
    def __init__(self, db: Database):
        self.db = db

    def enqueue(self, task: ScheduledTask) -> int:
        scheduled_for = task.scheduled_for or utc_now_iso()
        payload = json.loads(task.payload_json or "{}")
        if task.task_type == "gathering":
            payload["resource_type"] = task.resource_type
        cursor = self.db.execute(
            """
            INSERT INTO scheduled_tasks(
                character_id, march_slot, task_type, priority, status,
                scheduled_for, attempts, error_message, result, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task.character_id,
                task.march_slot,
                task.task_type,
                task.priority,
                task.status,
                scheduled_for,
                task.attempts,
                task.error_message,
                task.result,
                json.dumps(payload),
            ),
        )
        return int(cursor.lastrowid)

    def open_task_exists(self, character_id: int, march_slot: int | None, task_type: str) -> bool:
        row = self.db.fetch_one(
            """
            SELECT id
            FROM scheduled_tasks
            WHERE character_id = ?
              AND COALESCE(march_slot, -1) = COALESCE(?, -1)
              AND task_type = ?
              AND status IN ('pending', 'queued', 'running', 'retrying')
            LIMIT 1
            """,
            (character_id, march_slot, task_type),
        )
        return row is not None

    def list_due(self, limit: int) -> list[ScheduledTask]:
        rows = self.db.fetch_all(
            """
            SELECT t.*, c.name AS character_name, i.id AS instance_id, i.name AS instance_name
            FROM scheduled_tasks t
            JOIN characters c ON c.id = t.character_id
            JOIN instances i ON i.id = c.instance_id
            WHERE t.status IN ('pending', 'retrying') AND t.scheduled_for <= ?
              AND c.enabled = 1 AND i.enabled = 1
            ORDER BY t.priority ASC, t.scheduled_for ASC
            LIMIT ?
            """,
            (utc_now_iso(), limit),
        )
        return [self._from_row(row) for row in rows]

    def list_recent(self, limit: int = 200) -> list[ScheduledTask]:
        rows = self.db.fetch_all(
            """
            SELECT t.*, c.name AS character_name, i.id AS instance_id, i.name AS instance_name
            FROM scheduled_tasks t
            JOIN characters c ON c.id = t.character_id
            JOIN instances i ON i.id = c.instance_id
            ORDER BY t.scheduled_for DESC, t.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [self._from_row(row) for row in rows]

    def mark_queued(self, task_id: int) -> None:
        self.db.execute(
            "UPDATE scheduled_tasks SET status = 'queued', result = '' WHERE id = ?",
            (task_id,),
        )

    def mark_running(self, task_id: int) -> None:
        self.db.execute(
            """
            UPDATE scheduled_tasks
            SET status = 'running', started_at = ?, attempts = attempts + 1, result = ''
            WHERE id = ?
            """,
            (utc_now_iso(), task_id),
        )

    def mark_completed(self, task_id: int, message: str = "") -> None:
        self.db.execute(
            """
            UPDATE scheduled_tasks
            SET status = 'completed', completed_at = ?, error_message = ?, result = 'SUCCESS'
            WHERE id = ?
            """,
            (utc_now_iso(), message, task_id),
        )

    def mark_aborted(self, task_id: int, message: str = "") -> None:
        self.db.execute(
            """
            UPDATE scheduled_tasks
            SET status = 'aborted', completed_at = ?, error_message = ?, result = 'ABORTED'
            WHERE id = ?
            """,
            (utc_now_iso(), message, task_id),
        )

    def mark_failed(self, task_id: int, message: str) -> None:
        self.db.execute(
            """
            UPDATE scheduled_tasks
            SET status = 'failed', completed_at = ?, error_message = ?, result = 'FAILED'
            WHERE id = ?
            """,
            (utc_now_iso(), message, task_id),
        )

    def schedule_retry(self, task_id: int, scheduled_for: str, message: str) -> None:
        self.db.execute(
            """
            UPDATE scheduled_tasks
            SET status = 'retrying', scheduled_for = ?, error_message = ?, result = 'FAILED'
            WHERE id = ?
            """,
            (scheduled_for, message, task_id),
        )

    def count_pending(self) -> int:
        row = self.db.fetch_one(
            """
            SELECT COUNT(*) AS total
            FROM scheduled_tasks
            WHERE status IN ('pending', 'queued', 'running', 'retrying')
            """
        )
        return int(row["total"] if row else 0)

    def next_scheduled(self) -> str:
        row = self.db.fetch_one(
            """
            SELECT scheduled_for
            FROM scheduled_tasks
            WHERE status IN ('pending', 'retrying')
            ORDER BY scheduled_for ASC
            LIMIT 1
            """
        )
        return row["scheduled_for"] if row else "-"

    @staticmethod
    def _from_row(row: Any) -> ScheduledTask:
        payload_json = row["payload_json"]
        payload = json.loads(payload_json or "{}")
        return ScheduledTask(
            id=row["id"],
            character_id=row["character_id"],
            march_slot=row["march_slot"],
            task_type=row["task_type"],
            priority=row["priority"],
            status=row["status"],
            scheduled_for=row["scheduled_for"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            attempts=row["attempts"],
            error_message=row["error_message"],
            result=row["result"] if "result" in row.keys() else "",
            payload_json=payload_json,
            resource_type=str(payload.get("resource_type", "Gold")),
            character_name=row["character_name"] if "character_name" in row.keys() else "",
            instance_id=row["instance_id"] if "instance_id" in row.keys() else None,
            instance_name=row["instance_name"] if "instance_name" in row.keys() else "",
        )


class AutomationTaskRepository:
    def __init__(self, db: Database):
        self.db = db

    def list_all(self) -> list[Task]:
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM automation_tasks
            ORDER BY enabled DESC, name COLLATE NOCASE, id
            """
        )
        return [self._task_from_row(row) for row in rows]

    def get(self, task_id: int) -> Task | None:
        row = self.db.fetch_one("SELECT * FROM automation_tasks WHERE id = ?", (task_id,))
        return self._task_from_row(row) if row else None

    def save_task(self, task: Task) -> int:
        name = task.name.strip() or "Untitled Task"
        if task.id is None:
            cursor = self.db.execute(
                """
                INSERT INTO automation_tasks(
                    name, enabled, template_readiness_required
                )
                VALUES (?, ?, ?)
                """,
                (
                    name,
                    int(task.enabled),
                    int(task.template_readiness_required),
                ),
            )
            return int(cursor.lastrowid)

        self.db.execute(
            """
            UPDATE automation_tasks
            SET name = ?, enabled = ?, template_readiness_required = ?
            WHERE id = ?
            """,
            (
                name,
                int(task.enabled),
                int(task.template_readiness_required),
                task.id,
            ),
        )
        return task.id

    def delete_task(self, task_id: int) -> None:
        self.db.execute("DELETE FROM automation_tasks WHERE id = ?", (task_id,))

    def duplicate_task(self, task_id: int) -> int:
        with self.db.transaction():
            task = self.get(task_id)
            if task is None:
                raise ValueError(f"Task not found: {task_id}")
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
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM automation_task_steps
            WHERE task_id = ?
            ORDER BY step_order
            """,
            (task_id,),
        )
        return [self._step_from_row(row) for row in rows]

    def add_step(
        self,
        task_id: int,
        action_type: str,
        parameters: dict[str, object] | None = None,
    ) -> int:
        self._validate_action(action_type)
        with self.db.transaction():
            next_order = self._next_order(task_id)
            cursor = self.db.execute(
                """
                INSERT INTO automation_task_steps(task_id, step_order, action_type, parameters)
                VALUES (?, ?, ?, ?)
                """,
                (task_id, next_order, action_type, json.dumps(parameters or {})),
            )
            return int(cursor.lastrowid)

    def save_step(self, step: TaskStep) -> int:
        if step.task_id is None:
            raise ValueError("Task step must belong to a task.")
        self._validate_action(step.action_type)
        parameters = json.dumps(step.parameters or {})
        with self.db.transaction():
            if step.id is None:
                cursor = self.db.execute(
                    """
                    INSERT INTO automation_task_steps(
                        task_id, step_order, action_type, parameters
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (step.task_id, step.order, step.action_type, parameters),
                )
                self.reorder_steps(step.task_id)
                return int(cursor.lastrowid)

            self.db.execute(
                """
                UPDATE automation_task_steps
                SET step_order = ?, action_type = ?, parameters = ?
                WHERE id = ?
                """,
                (step.order, step.action_type, parameters, step.id),
            )
            self.reorder_steps(step.task_id)
            return step.id

    def delete_step(self, step_id: int) -> None:
        with self.db.transaction():
            step = self.get_step(step_id)
            if step is None or step.task_id is None:
                return
            self.db.execute("DELETE FROM automation_task_steps WHERE id = ?", (step_id,))
            self.reorder_steps(step.task_id)

    def get_step(self, step_id: int) -> TaskStep | None:
        row = self.db.fetch_one(
            "SELECT * FROM automation_task_steps WHERE id = ?",
            (step_id,),
        )
        return self._step_from_row(row) if row else None

    def move_step_up(self, step_id: int) -> None:
        self._move_step(step_id, -1)

    def move_step_down(self, step_id: int) -> None:
        self._move_step(step_id, 1)

    def reorder_steps(self, task_id: int) -> None:
        with self.db.transaction():
            steps = self.list_steps(task_id)
            for index, step in enumerate(steps, start=1):
                if step.id is None or step.order == index:
                    continue
                self.db.execute(
                    """
                    UPDATE automation_task_steps
                    SET step_order = ?
                    WHERE id = ?
                    """,
                    (index, step.id),
                )

    def _move_step(self, step_id: int, direction: int) -> None:
        with self.db.transaction():
            step = self.get_step(step_id)
            if step is None or step.task_id is None or step.id is None:
                return
            steps = self.list_steps(step.task_id)
            index = next((i for i, item in enumerate(steps) if item.id == step.id), -1)
            target_index = index + direction
            if index < 0 or target_index < 0 or target_index >= len(steps):
                return
            steps[index], steps[target_index] = steps[target_index], steps[index]
            temporary_offset = len(steps) + 1000
            for temporary_order, item in enumerate(steps, start=temporary_offset):
                self.db.execute(
                    "UPDATE automation_task_steps SET step_order = ? WHERE id = ?",
                    (temporary_order, item.id),
                )
            for final_order, item in enumerate(steps, start=1):
                self.db.execute(
                    "UPDATE automation_task_steps SET step_order = ? WHERE id = ?",
                    (final_order, item.id),
                )

    def _next_order(self, task_id: int) -> int:
        row = self.db.fetch_one(
            """
            SELECT COALESCE(MAX(step_order), 0) + 1 AS next_order
            FROM automation_task_steps
            WHERE task_id = ?
            """,
            (task_id,),
        )
        return int(row["next_order"] if row else 1)

    @staticmethod
    def _validate_action(action_type: str) -> None:
        if action_type not in AUTOMATION_ACTION_TYPES:
            raise ValueError(f"Unsupported action type: {action_type}")

    @staticmethod
    def _task_from_row(row: Any) -> Task:
        return Task(
            id=row["id"],
            name=row["name"],
            enabled=row_bool(row["enabled"]),
            template_readiness_required=(
                row_bool(row["template_readiness_required"])
                if "template_readiness_required" in row.keys()
                else False
            ),
            created_at=row["created_at"] if "created_at" in row.keys() else "",
        )

    @staticmethod
    def _step_from_row(row: Any) -> TaskStep:
        try:
            parameters = json.loads(row["parameters"] or "{}")
        except json.JSONDecodeError:
            parameters = {}
        if not isinstance(parameters, dict):
            parameters = {}
        return TaskStep(
            id=row["id"],
            task_id=row["task_id"],
            order=row["step_order"],
            action_type=row["action_type"],
            parameters=parameters,
        )


class TaskRunHistoryRepository:
    def __init__(self, db: Database):
        self.db = db

    def create(self, history: TaskRunHistory) -> int:
        cursor = self.db.execute(
            """
            INSERT INTO task_run_history(
                task_id, task_name, instance_index, instance_name,
                started_at, finished_at, result, error_message, abort_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                history.task_id,
                history.task_name.strip() or "Untitled Task",
                history.instance_index,
                history.instance_name.strip(),
                history.started_at,
                history.finished_at,
                history.result,
                history.error_message,
                history.abort_reason,
            ),
        )
        return int(cursor.lastrowid)

    def list_recent(self, limit: int = 200) -> list[TaskRunHistory]:
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM task_run_history
            ORDER BY started_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: Any) -> TaskRunHistory:
        return TaskRunHistory(
            id=row["id"],
            task_id=row["task_id"],
            task_name=row["task_name"],
            instance_index=row["instance_index"],
            instance_name=row["instance_name"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            result=row["result"],
            error_message=row["error_message"],
            abort_reason=row["abort_reason"],
            created_at=row["created_at"],
        )


class SettingsRepository:
    def __init__(self, db: Database):
        self.db = db

    def get(self, key: str, default: str = "") -> str:
        row = self.db.fetch_one("SELECT value FROM settings WHERE key = ?", (key,))
        return row["value"] if row else default

    def set(self, key: str, value: Any) -> None:
        stored = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        self.db.execute(
            """
            INSERT INTO settings(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
            """,
            (key, stored),
        )

    def set_defaults(self, values: dict[str, Any]) -> None:
        with self.db.transaction():
            for key, value in values.items():
                if self.get(key, "") == "":
                    self.set(key, value)

    def all(self) -> dict[str, str]:
        rows = self.db.fetch_all("SELECT key, value FROM settings ORDER BY key")
        return {row["key"]: row["value"] for row in rows}

    def get_int(self, key: str, default: int) -> int:
        try:
            return int(self.get(key, str(default)))
        except ValueError:
            return default

    def get_json(self, key: str, default: Any) -> Any:
        value = self.get(key, "")
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
