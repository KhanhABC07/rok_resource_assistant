from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rok_assistant.db.models import GameAccount
from rok_assistant.security import SecretMaterial, SecretStore


MAX_CONFIGURED_ACCOUNTS = 6


class AccountConfigError(ValueError):
    pass


class GameAccountRepository(Protocol):
    def list_all(self, include_disabled: bool = True) -> list[GameAccount]:
        ...

    def get(self, account_id: int) -> GameAccount | None:
        ...

    def get_by_name(self, account_name: str) -> GameAccount | None:
        ...

    def save(self, account: GameAccount) -> int:
        ...

    def delete(self, account_id: int) -> None:
        ...


@dataclass(frozen=True)
class AccountConfigInput:
    account_name: str
    display_name: str = ""
    provider: str = ""
    external_id: str = ""
    username: str = ""
    password: str = ""
    token: str = ""
    enabled: bool = True
    account_id: int | None = None


class AccountConfigService:
    def __init__(
        self,
        accounts: GameAccountRepository,
        secret_store: SecretStore,
        *,
        max_enabled_accounts: int = MAX_CONFIGURED_ACCOUNTS,
    ) -> None:
        self.accounts = accounts
        self.secret_store = secret_store
        self.max_enabled_accounts = max_enabled_accounts

    def list_accounts(self) -> list[GameAccount]:
        return self.accounts.list_all(include_disabled=True)

    def get_account(self, account_id: int) -> GameAccount | None:
        return self.accounts.get(account_id)

    def save_account(self, data: AccountConfigInput) -> int:
        account_name = data.account_name.strip()
        provider = data.provider.strip()
        external_id = data.external_id.strip()
        display_name = data.display_name.strip() or account_name

        if not account_name:
            raise AccountConfigError("Account name is required.")
        if not provider:
            raise AccountConfigError("Login provider is required.")
        if not external_id:
            raise AccountConfigError("Provider account ID is required.")

        existing = self.accounts.get(data.account_id) if data.account_id is not None else None
        duplicate = self.accounts.get_by_name(account_name)
        if duplicate is not None and duplicate.id != data.account_id:
            raise AccountConfigError("Account name is already configured.")
        for account in self.accounts.list_all(include_disabled=True):
            if account.id == data.account_id:
                continue
            if account.provider == provider and account.external_id == external_id:
                raise AccountConfigError("Provider account ID is already configured.")

        self._validate_enabled_limit(data.account_id, data.enabled)

        secret_ref = existing.secret_ref if existing is not None else ""
        if data.password or data.token:
            secret_ref = self.secret_store.put(
                SecretMaterial(
                    username=data.username.strip(),
                    password=data.password,
                    token=data.token,
                ),
                ref=secret_ref or None,
            )
        elif not secret_ref:
            raise AccountConfigError("Password or token is required for a new account.")

        return self.accounts.save(
            GameAccount(
                id=data.account_id,
                account_name=account_name,
                display_name=display_name,
                provider=provider,
                external_id=external_id,
                secret_ref=secret_ref,
                enabled=data.enabled,
            )
        )

    def delete_account(self, account_id: int) -> None:
        account = self.accounts.get(account_id)
        if account is None:
            return
        if account.secret_ref:
            self.secret_store.delete(account.secret_ref)
        self.accounts.delete(account_id)

    def _validate_enabled_limit(self, account_id: int | None, enabled: bool) -> None:
        if not enabled:
            return
        enabled_accounts = self.accounts.list_all(include_disabled=False)
        enabled_count = sum(1 for account in enabled_accounts if account.id != account_id)
        if enabled_count >= self.max_enabled_accounts:
            raise AccountConfigError(
                f"Only {self.max_enabled_accounts} enabled accounts are supported."
            )
