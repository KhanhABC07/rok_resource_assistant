from __future__ import annotations

from typing import Any

from .database import Database
from .models import GameAccount, row_bool
from .repository_helpers import json_object_text, require_text, row_id


class GameAccountRepository:
    def __init__(self, db: Database):
        self.db = db

    def list_all(self, include_disabled: bool = True) -> list[GameAccount]:
        where = "" if include_disabled else "WHERE enabled = 1"
        rows = self.db.fetch_all(
            f"""
            SELECT *
            FROM game_accounts
            {where}
            ORDER BY account_name COLLATE NOCASE
            """
        )
        return [self._from_row(row) for row in rows]

    def get(self, account_id: int) -> GameAccount | None:
        row = self.db.fetch_one("SELECT * FROM game_accounts WHERE id = ?", (account_id,))
        return self._from_row(row) if row else None

    def get_by_name(self, account_name: str) -> GameAccount | None:
        row = self.db.fetch_one(
            "SELECT * FROM game_accounts WHERE account_name = ?",
            (require_text(account_name, "account_name"),),
        )
        return self._from_row(row) if row else None

    def save(self, account: GameAccount) -> int:
        account_name = require_text(account.account_name, "account_name")
        display_name = account.display_name.strip() or account_name
        metadata_json = json_object_text(account.metadata_json, "metadata_json")
        with self.db.transaction():
            if account.id is None:
                self.db.execute(
                    """
                    INSERT INTO game_accounts(
                        account_name, display_name, provider, external_id, enabled, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(account_name) DO UPDATE SET
                        display_name = excluded.display_name,
                        provider = excluded.provider,
                        external_id = excluded.external_id,
                        enabled = excluded.enabled,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        account_name,
                        display_name,
                        account.provider.strip(),
                        account.external_id.strip(),
                        int(account.enabled),
                        metadata_json,
                    ),
                )
                return row_id(self.get_by_name(account_name))

            self.db.execute(
                """
                UPDATE game_accounts
                SET account_name = ?,
                    display_name = ?,
                    provider = ?,
                    external_id = ?,
                    enabled = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    account_name,
                    display_name,
                    account.provider.strip(),
                    account.external_id.strip(),
                    int(account.enabled),
                    metadata_json,
                    account.id,
                ),
            )
            return account.id

    def delete(self, account_id: int) -> None:
        self.db.execute("DELETE FROM game_accounts WHERE id = ?", (account_id,))

    @staticmethod
    def _from_row(row: Any) -> GameAccount:
        return GameAccount(
            id=row["id"],
            account_name=row["account_name"],
            display_name=row["display_name"],
            provider=row["provider"],
            external_id=row["external_id"],
            enabled=row_bool(row["enabled"]),
            metadata_json=row["metadata_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
