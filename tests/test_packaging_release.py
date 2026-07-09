from __future__ import annotations

from pathlib import Path

import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyinstaller_spec_references_entry_point_and_assets() -> None:
    spec = (REPO_ROOT / "packaging" / "pyinstaller" / "rok_resource_assistant.spec").read_text(encoding="utf-8")

    assert "main.py" in spec
    assert "config" in spec
    assert "templates" in spec
    assert "rok_resource_assistant" in spec
    assert "runtime/" not in spec
    assert "runtime\\" not in spec
    assert "venv/" not in spec
    assert "venv\\" not in spec
    assert ".venv" not in spec
    assert "screenshots" not in spec
    assert ".sqlite" not in spec


def test_windows_ci_runs_tests_builds_artifact_and_uploads_release_outputs() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "windows-ci.yml").read_text(encoding="utf-8")

    assert "runs-on: windows-latest" in workflow
    assert "requirements-lock.txt" in workflow
    assert "QT_QPA_PLATFORM: offscreen" in workflow
    assert "python -m pytest tests -q" in workflow
    assert "scripts\\build_windows.ps1" in workflow or "scripts/build_windows.ps1" in workflow
    assert "dist/*.zip" in workflow
    assert "dist/*.sha256" in workflow
    assert "dist/*.cyclonedx.json" in workflow


def test_locked_requirements_include_runtime_test_and_packaging_tools() -> None:
    locked = (REPO_ROOT / "requirements-lock.txt").read_text(encoding="utf-8")

    for package in ("PyQt6==", "opencv-python-headless==", "pytest==", "pyinstaller==", "cyclonedx-bom=="):
        assert package in locked


def test_packaging_script_generates_checksum_and_sbom() -> None:
    script = (REPO_ROOT / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")

    assert "Get-FileHash -Algorithm SHA256" in script
    assert "cyclonedx_py" in script
    assert "release.json" in script
    assert "ValidateSet(\"staging\", \"canary\", \"stable\")" in script
    assert "Compress-Archive" in script


def test_gitignore_excludes_generated_packaging_outputs() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    for pattern in ("dist/", "build/", "*.exe", "*.msi", "*.zip", "*.sha256", "*.cyclonedx.json", "release/"):
        assert pattern in gitignore


def test_project_version_is_available_for_packaging_metadata() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert data["project"]["name"] == "rok-resource-assistant"
    assert data["project"]["version"]
