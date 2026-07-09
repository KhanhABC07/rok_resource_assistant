from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.db_helpers import SRC_ROOT  # noqa: F401

from rok_assistant.account_config import (
    AccountConfigError,
    AccountConfigInput,
    AccountConfigService,
)
from rok_assistant.db.database import Database
from rok_assistant.db.models import GameAccount
from rok_assistant.db.repositories import GameAccountRepository
from rok_assistant.security import InMemorySecretStore


class AccountConfigServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "accounts.sqlite3")
        self.db.initialize()
        self.accounts = GameAccountRepository(self.db)
        self.secret_store = InMemorySecretStore()
        self.service = AccountConfigService(self.accounts, self.secret_store)

    def tearDown(self) -> None:
        self.db.close()
        self.temp_dir.cleanup()

    def test_validation_requires_account_identity_and_credential(self) -> None:
        with self.assertRaisesRegex(AccountConfigError, "Account name is required"):
            self.service.save_account(
                AccountConfigInput(
                    account_name=" ",
                    provider="email",
                    external_id="farm@example.test",
                    password="secret",
                )
            )

        with self.assertRaisesRegex(AccountConfigError, "Login provider is required"):
            self.service.save_account(
                AccountConfigInput(
                    account_name="Farm",
                    provider=" ",
                    external_id="farm@example.test",
                    password="secret",
                )
            )

        with self.assertRaisesRegex(AccountConfigError, "Provider account ID is required"):
            self.service.save_account(
                AccountConfigInput(
                    account_name="Farm",
                    provider="email",
                    external_id=" ",
                    password="secret",
                )
            )

        with self.assertRaisesRegex(AccountConfigError, "Password or token is required"):
            self.service.save_account(
                AccountConfigInput(
                    account_name="Farm",
                    provider="email",
                    external_id="farm@example.test",
                )
            )

    def test_saves_account_with_secret_ref_without_plaintext_database_secret(self) -> None:
        account_id = self.service.save_account(
            AccountConfigInput(
                account_name="Farm",
                display_name="Farm Account",
                provider="email",
                external_id="farm@example.test",
                username="farm@example.test",
                password="super-secret-password",
            )
        )

        account = self.accounts.get(account_id)
        self.assertIsNotNone(account)
        self.assertEqual("Farm", account.account_name)  # type: ignore[union-attr]
        self.assertTrue(account.secret_ref.startswith("mem://account/"))  # type: ignore[union-attr]
        self.assertEqual(
            "super-secret-password",
            self.secret_store.get(account.secret_ref).password,  # type: ignore[union-attr]
        )

        rows = self.db.fetch_all("SELECT * FROM game_accounts")
        persisted = "\n".join(str(dict(row)) for row in rows)
        self.assertNotIn("super-secret-password", persisted)

    def test_edit_without_new_secret_preserves_existing_secret_ref(self) -> None:
        account_id = self.service.save_account(
            AccountConfigInput(
                account_name="Farm",
                provider="email",
                external_id="farm@example.test",
                password="secret",
            )
        )
        original_ref = self.accounts.get(account_id).secret_ref  # type: ignore[union-attr]

        self.service.save_account(
            AccountConfigInput(
                account_id=account_id,
                account_name="Farm",
                display_name="Renamed",
                provider="email",
                external_id="farm@example.test",
            )
        )

        updated = self.accounts.get(account_id)
        self.assertEqual("Renamed", updated.display_name)  # type: ignore[union-attr]
        self.assertEqual(original_ref, updated.secret_ref)  # type: ignore[union-attr]

    def test_duplicate_provider_account_id_is_rejected_before_database_error(self) -> None:
        self.service.save_account(
            AccountConfigInput(
                account_name="Farm",
                provider="email",
                external_id="farm@example.test",
                password="secret",
            )
        )

        with self.assertRaisesRegex(AccountConfigError, "Provider account ID"):
            self.service.save_account(
                AccountConfigInput(
                    account_name="Farm Duplicate",
                    provider="email",
                    external_id="farm@example.test",
                    password="secret",
                )
            )

    def test_more_than_six_enabled_accounts_is_rejected(self) -> None:
        for index in range(6):
            self.accounts.save(
                GameAccount(
                    account_name=f"Farm {index}",
                    provider="email",
                    external_id=f"farm-{index}",
                    secret_ref=f"mem://account/{index}",
                    enabled=True,
                )
            )

        with self.assertRaisesRegex(AccountConfigError, "Only 6 enabled accounts"):
            self.service.save_account(
                AccountConfigInput(
                    account_name="Farm 7",
                    provider="email",
                    external_id="farm-7",
                    password="secret",
                    enabled=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
