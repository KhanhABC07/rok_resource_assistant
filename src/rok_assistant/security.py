from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes
import json
import sys
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from .paths import CONFIG_DIR


REDACTED = "[REDACTED]"
SENSITIVE_KEY_PARTS = (
    "access_token",
    "auth",
    "credential",
    "password",
    "refresh_token",
    "secret",
    "token",
)


class CredentialFailureReason(StrEnum):
    MISSING_REFERENCE = "missing_reference"
    MISSING_SECRET = "missing_secret"
    INACCESSIBLE = "inaccessible"
    INVALID = "invalid"


@dataclass(frozen=True)
class SecretMaterial:
    username: str = ""
    password: str = ""
    token: str = ""
    metadata: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "password": self.password,
            "token": self.token,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecretMaterial":
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        return cls(
            username=str(data.get("username", "")),
            password=str(data.get("password", "")),
            token=str(data.get("token", "")),
            metadata={str(key): str(value) for key, value in metadata.items()},
        )


@dataclass(frozen=True)
class CredentialValidation:
    ok: bool
    reason: CredentialFailureReason | None = None
    message: str = ""


class SecretStoreError(RuntimeError):
    def __init__(self, message: str, reason: CredentialFailureReason) -> None:
        super().__init__(message)
        self.reason = reason


class SecretStore(Protocol):
    def put(self, material: SecretMaterial, *, ref: str | None = None) -> str:
        """Store secret material and return a non-secret reference."""

    def get(self, ref: str) -> SecretMaterial:
        """Return secret material for a non-secret reference."""

    def delete(self, ref: str) -> None:
        """Remove secret material if present."""


class InMemorySecretStore:
    def __init__(self) -> None:
        self._items: dict[str, SecretMaterial] = {}

    def put(self, material: SecretMaterial, *, ref: str | None = None) -> str:
        validate_secret_material(material)
        secret_ref = ref or f"mem://account/{uuid.uuid4()}"
        self._items[secret_ref] = material
        return secret_ref

    def get(self, ref: str) -> SecretMaterial:
        try:
            return self._items[ref]
        except KeyError as exc:
            raise SecretStoreError(
                "Credential reference was not found.",
                CredentialFailureReason.MISSING_SECRET,
            ) from exc

    def delete(self, ref: str) -> None:
        self._items.pop(ref, None)


class DpapiFileSecretStore:
    """Current-user DPAPI encrypted credential store for the Windows desktop runtime."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or CONFIG_DIR / "secrets.dpapi.json"

    def put(self, material: SecretMaterial, *, ref: str | None = None) -> str:
        validate_secret_material(material)
        secret_ref = ref or f"dpapi://account/{uuid.uuid4()}"
        items = self._read_items()
        plaintext = json.dumps(material.to_dict(), sort_keys=True).encode("utf-8")
        items[secret_ref] = base64.b64encode(_dpapi_protect(plaintext)).decode("ascii")
        self._write_items(items)
        return secret_ref

    def get(self, ref: str) -> SecretMaterial:
        items = self._read_items()
        encrypted = items.get(ref)
        if encrypted is None:
            raise SecretStoreError(
                "Credential reference was not found.",
                CredentialFailureReason.MISSING_SECRET,
            )
        try:
            plaintext = _dpapi_unprotect(base64.b64decode(encrypted.encode("ascii")))
            data = json.loads(plaintext.decode("utf-8"))
            return SecretMaterial.from_dict(data)
        except SecretStoreError:
            raise
        except Exception as exc:
            raise SecretStoreError(
                "Credential reference could not be read.",
                CredentialFailureReason.INACCESSIBLE,
            ) from exc

    def delete(self, ref: str) -> None:
        items = self._read_items()
        if ref in items:
            del items[ref]
            self._write_items(items)

    def _read_items(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise SecretStoreError(
                "Credential store is inaccessible.",
                CredentialFailureReason.INACCESSIBLE,
            ) from exc
        if not isinstance(data, dict):
            raise SecretStoreError(
                "Credential store data is invalid.",
                CredentialFailureReason.INVALID,
            )
        return {str(key): str(value) for key, value in data.items()}

    def _write_items(self, items: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(items, indent=2, sort_keys=True), encoding="utf-8")


def validate_secret_material(material: SecretMaterial) -> None:
    if not material.password and not material.token:
        raise SecretStoreError(
            "Credential material must include a password or token.",
            CredentialFailureReason.INVALID,
        )
    if material.password.strip() != material.password or material.token.strip() != material.token:
        raise SecretStoreError(
            "Credential material contains invalid surrounding whitespace.",
            CredentialFailureReason.INVALID,
        )


def validate_account_credentials(secret_ref: str, secret_store: SecretStore) -> CredentialValidation:
    if not secret_ref.strip():
        return CredentialValidation(
            ok=False,
            reason=CredentialFailureReason.MISSING_REFERENCE,
            message="Account has no credential reference.",
        )
    try:
        validate_secret_material(secret_store.get(secret_ref))
    except SecretStoreError as exc:
        return CredentialValidation(ok=False, reason=exc.reason, message=str(exc))
    return CredentialValidation(ok=True)


def redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: REDACTED if _is_sensitive_key(str(key)) else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_text(text: str) -> str:
    redacted = text
    for key_part in SENSITIVE_KEY_PARTS:
        redacted = _redact_assignment(redacted, key_part)
    return redacted


def redacted_exception_message(exc: BaseException) -> str:
    return redact_text(str(exc))


def _is_sensitive_key(key: str) -> bool:
    key_lower = key.lower()
    if key_lower in {"secret_ref", "credential_ref"}:
        return False
    return any(part in key_lower for part in SENSITIVE_KEY_PARTS)


def _redact_assignment(text: str, key: str) -> str:
    import re

    pattern = re.compile(
        rf"(['\"]?{re.escape(key)}['\"]?\s*[:=]\s*)(\[[^\]]*\]|\"[^\"]*\"|'[^']*'|[^\s,;}}\]]+)",
        re.IGNORECASE,
    )
    return pattern.sub(lambda match: f"{match.group(1)}{REDACTED}", text)


class RedactingLogFilter:
    def filter(self, record: Any) -> bool:
        record.msg = redact_text(record.getMessage())
        record.args = ()
        return True


def _dpapi_protect(data: bytes) -> bytes:
    if sys.platform != "win32":
        raise SecretStoreError(
            "DPAPI credential store is only available on Windows.",
            CredentialFailureReason.INACCESSIBLE,
        )
    return _crypt_protect_data(data)


def _dpapi_unprotect(data: bytes) -> bytes:
    if sys.platform != "win32":
        raise SecretStoreError(
            "DPAPI credential store is only available on Windows.",
            CredentialFailureReason.INACCESSIBLE,
        )
    return _crypt_unprotect_data(data)


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _blob_from_bytes(data: bytes) -> _DataBlob:
    buffer = ctypes.create_string_buffer(data)
    blob = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    blob._buffer = buffer  # type: ignore[attr-defined]
    return blob


def _bytes_from_blob(blob: _DataBlob) -> bytes:
    try:
        return ctypes.string_at(blob.pbData, blob.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob.pbData)


def _crypt_protect_data(data: bytes) -> bytes:
    in_blob = _blob_from_bytes(data)
    out_blob = _DataBlob()
    if not ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(in_blob),
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(out_blob),
    ):
        raise SecretStoreError(
            f"DPAPI protect failed with Windows error {ctypes.get_last_error()}.",
            CredentialFailureReason.INACCESSIBLE,
        )
    return _bytes_from_blob(out_blob)


def _crypt_unprotect_data(data: bytes) -> bytes:
    in_blob = _blob_from_bytes(data)
    out_blob = _DataBlob()
    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(in_blob),
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(out_blob),
    ):
        raise SecretStoreError(
            f"DPAPI unprotect failed with Windows error {ctypes.get_last_error()}.",
            CredentialFailureReason.INACCESSIBLE,
        )
    return _bytes_from_blob(out_blob)
