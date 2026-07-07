from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

from tests.db_helpers import SRC_ROOT  # noqa: F401

from rok_assistant.db.database import Database
from rok_assistant.db.models import Character, GameAccount, Instance
from rok_assistant.db.repositories import CharacterRepository, GameAccountRepository, InstanceRepository
from rok_assistant.export_import import ConfigurationService
from rok_assistant.logging_setup import configure_logging
from rok_assistant.security import (
    CredentialFailureReason,
    InMemorySecretStore,
    SecretMaterial,
    SecretStoreError,
    validate_account_credentials,
)


class CredentialSecurityTest(unittest.TestCase):
    def test_fake_secret_store_validates_structured_failure_reasons(self) -> None:
        store = InMemorySecretStore()
        secret_ref = store.put(SecretMaterial(username="farm@example.test", password="s3cr3t"))

        self.assertTrue(validate_account_credentials(secret_ref, store).ok)

        missing_ref = validate_account_credentials("", store)
        self.assertFalse(missing_ref.ok)
        self.assertEqual(CredentialFailureReason.MISSING_REFERENCE, missing_ref.reason)

        missing_secret = validate_account_credentials("mem://missing", store)
        self.assertFalse(missing_secret.ok)
        self.assertEqual(CredentialFailureReason.MISSING_SECRET, missing_secret.reason)

        with self.assertRaises(SecretStoreError) as raised:
            store.put(SecretMaterial(username="farm@example.test"))
        self.assertEqual(CredentialFailureReason.INVALID, raised.exception.reason)

    def test_account_persists_only_secret_ref_not_secret_material(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "accounts.sqlite3")
            db.initialize()
            store = InMemorySecretStore()
            secret = "plain-text-password"
            secret_ref = store.put(SecretMaterial(username="farm", password=secret))
            accounts = GameAccountRepository(db)

            account_id = accounts.save(
                GameAccount(account_name="Account A", secret_ref=secret_ref)
            )
            row = db.fetch_one("SELECT * FROM game_accounts WHERE id = ?", (account_id,))

            self.assertEqual(secret_ref, row["secret_ref"])
            self.assertNotIn(secret, json.dumps(dict(row)))
            self.assertEqual(secret, store.get(secret_ref).password)
            db.close()

    def test_logging_redacts_secret_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "app.log"
            configure_logging(log_file, "INFO")
            secret = "plain-text-password"

            logging.getLogger("test.security").info(
                "credential password=%s token=%s",
                secret,
                "token-value",
            )
            logging.getLogger("test.security").info({"password": secret})
            for handler in logging.getLogger().handlers:
                handler.flush()

            log_text = log_file.read_text(encoding="utf-8")
            self.assertNotIn(secret, log_text)
            self.assertNotIn("token-value", log_text)
            self.assertIn("[REDACTED]", log_text)
            for handler in logging.getLogger().handlers[:]:
                logging.getLogger().removeHandler(handler)
                handler.close()

    def test_export_and_import_keep_secret_material_out_of_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "source.sqlite3")
            db.initialize()
            store = InMemorySecretStore()
            secret = "plain-text-password"
            secret_ref = store.put(SecretMaterial(username="farm", password=secret))
            instances = InstanceRepository(db)
            accounts = GameAccountRepository(db)
            characters = CharacterRepository(db)
            instance_id = instances.save(Instance(name="MEmu0"))
            accounts.save(GameAccount(account_name="Account A", secret_ref=secret_ref))
            characters.save(
                Character(name="Farm01", instance_id=instance_id, account_name="Account A")
            )

            export_path = Path(temp_dir) / "config.json"
            ConfigurationService(db).export_json(export_path)
            exported_text = export_path.read_text(encoding="utf-8")
            exported = json.loads(exported_text)

            self.assertNotIn(secret, exported_text)
            self.assertEqual(secret_ref, exported["game_accounts"][0]["secret_ref"])

            imported_db = Database(Path(temp_dir) / "imported.sqlite3")
            imported_db.initialize()
            imported_store = InMemorySecretStore()
            ConfigurationService(imported_db).import_json(
                export_path,
                credential_payloads={
                    "Account A": SecretMaterial(username="farm", password=secret)
                },
                secret_store=imported_store,
            )
            imported_account = GameAccountRepository(imported_db).get_by_name("Account A")
            self.assertIsNotNone(imported_account)
            self.assertEqual(secret, imported_store.get(imported_account.secret_ref).password)

            imported_row = imported_db.fetch_one(
                "SELECT * FROM game_accounts WHERE account_name = ?",
                ("Account A",),
            )
            self.assertNotIn(secret, json.dumps(dict(imported_row)))
            db.close()
            imported_db.close()


if __name__ == "__main__":
    unittest.main()
