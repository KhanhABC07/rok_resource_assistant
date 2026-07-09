from __future__ import annotations

import argparse
import json
import sys
import tempfile
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.db.database import Database
from rok_assistant.vision.template_registry import validate_template_pack


def _load_project_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def _validate_entry_point() -> list[str]:
    diagnostics: list[str] = []
    main_file = REPO_ROOT / "main.py"
    if not main_file.is_file():
        diagnostics.append("main.py entry point is missing.")
        return diagnostics
    source = main_file.read_text(encoding="utf-8")
    if "from rok_assistant.app import run_app" not in source or "run_app()" not in source:
        diagnostics.append("main.py must call rok_assistant.app.run_app().")
    return diagnostics


def _validate_migrations() -> list[str]:
    diagnostics: list[str] = []
    with tempfile.TemporaryDirectory(prefix="rok-packaging-check-", ignore_cleanup_errors=True) as temp_dir:
        temp_db = Path(temp_dir) / "packaging-check.sqlite3"
        db = Database(temp_db)
        try:
            db.initialize()
            versions = [
                row["version"]
                for row in db.fetch_all("SELECT version FROM schema_migrations ORDER BY version")
            ]
            if versions != [1, 2, 3, 4, 5]:
                diagnostics.append(f"Unexpected migration versions: {versions!r}.")
        finally:
            db.close()
    return diagnostics


def _validate_template_packs() -> list[str]:
    diagnostics: list[str] = []
    templates_root = REPO_ROOT / "templates"
    if not templates_root.exists():
        return diagnostics
    for manifest in templates_root.rglob("template-pack.json"):
        report = validate_template_pack(manifest.parent)
        if not report.is_valid:
            for diagnostic in report.diagnostics:
                diagnostics.append(f"{manifest}: {diagnostic.code} {diagnostic.field} {diagnostic.message}")
    return diagnostics


def _validate_spec() -> list[str]:
    diagnostics: list[str] = []
    spec_file = REPO_ROOT / "packaging" / "pyinstaller" / "rok_resource_assistant.spec"
    if not spec_file.is_file():
        diagnostics.append("PyInstaller spec is missing.")
        return diagnostics
    spec = spec_file.read_text(encoding="utf-8")
    required_fragments = [
        "main.py",
        "config",
        "templates",
        "rok_resource_assistant",
    ]
    for fragment in required_fragments:
        if fragment not in spec:
            diagnostics.append(f"PyInstaller spec does not reference {fragment!r}.")
    forbidden_fragments = ["runtime/", "runtime\\", "venv/", "venv\\", ".venv", ".sqlite", "screenshots"]
    for fragment in forbidden_fragments:
        if fragment in spec:
            diagnostics.append(f"PyInstaller spec must not bundle {fragment!r}.")
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Windows packaging inputs.")
    parser.add_argument("--channel", choices=("staging", "canary", "stable"), default="staging")
    parser.add_argument("--template-pack-version", default="unversioned")
    args = parser.parse_args()

    diagnostics = []
    diagnostics.extend(_validate_entry_point())
    diagnostics.extend(_validate_spec())
    diagnostics.extend(_validate_migrations())
    diagnostics.extend(_validate_template_packs())

    if diagnostics:
        for diagnostic in diagnostics:
            print(f"PACKAGING_CHECK_FAILED: {diagnostic}", file=sys.stderr)
        return 1

    result = {
        "application": "rok-resource-assistant",
        "version": _load_project_version(),
        "channel": args.channel,
        "template_pack_version": args.template_pack_version,
        "checks": ["entry_point", "pyinstaller_spec", "database_migrations", "template_packs"],
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
