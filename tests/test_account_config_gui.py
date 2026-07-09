from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.db_helpers import SRC_ROOT  # noqa: F401

from PyQt6.QtWidgets import QApplication, QMessageBox

from rok_assistant.db.models import GameAccount
from rok_assistant.gui.account_config import AccountConfigWidget
from rok_assistant.security import InMemorySecretStore, SecretMaterial, SecretStoreError


class FakeAccounts:
    def __init__(self) -> None:
        self.items: dict[int, GameAccount] = {}
        self.next_id = 1
        self.deleted: list[int] = []

    def list_all(self, include_disabled: bool = True) -> list[GameAccount]:
        accounts = list(self.items.values())
        if include_disabled:
            return accounts
        return [account for account in accounts if account.enabled]

    def get(self, account_id: int) -> GameAccount | None:
        return self.items.get(account_id)

    def get_by_name(self, account_name: str) -> GameAccount | None:
        for account in self.items.values():
            if account.account_name == account_name:
                return account
        return None

    def save(self, account: GameAccount) -> int:
        account_id = account.id or self.next_id
        if account.id is None:
            self.next_id += 1
        self.items[account_id] = GameAccount(
            id=account_id,
            account_name=account.account_name,
            display_name=account.display_name,
            provider=account.provider,
            external_id=account.external_id,
            secret_ref=account.secret_ref,
            enabled=account.enabled,
        )
        return account_id

    def delete(self, account_id: int) -> None:
        self.deleted.append(account_id)
        self.items.pop(account_id, None)


class AccountConfigWidgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.accounts = FakeAccounts()
        self.secret_store = InMemorySecretStore()
        self.widget = AccountConfigWidget(
            SimpleNamespace(accounts=self.accounts, secret_store=self.secret_store)
        )

    def tearDown(self) -> None:
        self.widget.deleteLater()

    def test_save_shows_validation_error_without_required_fields(self) -> None:
        with patch("rok_assistant.gui.account_config.QMessageBox.warning") as warning:
            self.widget.save()

        warning.assert_called_once()
        self.assertEqual({}, self.accounts.items)

    def test_save_persists_account_and_clears_secret_fields(self) -> None:
        self.widget.account_name_input.setText("Farm")
        self.widget.display_name_input.setText("Farm Account")
        self.widget.provider_input.setCurrentText("email")
        self.widget.external_id_input.setText("farm@example.test")
        self.widget.username_input.setText("farm@example.test")
        self.widget.password_input.setText("secret-password")

        with patch("rok_assistant.gui.account_config.QMessageBox.information"):
            self.widget.save()

        self.assertEqual(1, len(self.accounts.items))
        saved = self.accounts.get(1)
        self.assertIsNotNone(saved)
        self.assertEqual("Farm", saved.account_name)  # type: ignore[union-attr]
        self.assertTrue(saved.secret_ref.startswith("mem://account/"))  # type: ignore[union-attr]
        self.assertEqual("Stored", self.widget.table.item(0, 6).text())
        self.assertEqual("", self.widget.password_input.text())
        self.assertEqual("", self.widget.token_input.text())

    def test_selecting_account_loads_non_secret_fields_only(self) -> None:
        ref = self.secret_store.put(SecretMaterial(password="secret"))
        self.accounts.save(
            GameAccount(
                account_name="Farm",
                display_name="Farm Account",
                provider="email",
                external_id="farm@example.test",
                secret_ref=ref,
            )
        )
        self.widget.refresh()

        self.widget.load_selected(0, 0)

        self.assertEqual(1, self.widget.selected_id)
        self.assertEqual("Farm", self.widget.account_name_input.text())
        self.assertEqual("email", self.widget.provider_input.currentText())
        self.assertEqual("", self.widget.password_input.text())
        self.assertEqual("", self.widget.token_input.text())

    def test_delete_removes_account_and_secret_after_confirmation(self) -> None:
        self.widget.account_name_input.setText("Farm")
        self.widget.provider_input.setCurrentText("email")
        self.widget.external_id_input.setText("farm@example.test")
        self.widget.password_input.setText("secret-password")
        with patch("rok_assistant.gui.account_config.QMessageBox.information"):
            self.widget.save()
        secret_ref = self.accounts.get(1).secret_ref  # type: ignore[union-attr]
        self.widget.load_selected(0, 0)

        with patch(
            "rok_assistant.gui.account_config.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            self.widget.delete()

        self.assertEqual([1], self.accounts.deleted)
        with self.assertRaises(SecretStoreError):
            self.secret_store.get(secret_ref)


if __name__ == "__main__":
    unittest.main()
