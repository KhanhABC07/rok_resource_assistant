# Windows packaging and release

This project ships as a Windows-first Python 3.11+ PyQt6 desktop application.
Packaging is intentionally separate from runtime state: releases include the
application code, default configuration, and checked-in template packs, but do
not include `runtime/`, local SQLite databases, screenshots, logs, support
bundles, `venv/`, `.venv/`, or generated build folders.

## Release channels

Use one of these channels for every build:

- `staging`: internal validation builds from active branches.
- `canary`: limited rollout builds after staging validation passes.
- `stable`: user-facing builds after canary validation and release approval.

The application version comes from `pyproject.toml`. The template-pack version
is supplied independently to the build script because template packs can be
validated and released on a different cadence.

## Local build

Run from the repository root on Windows:

```powershell
venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\scripts\build_windows.ps1 -Channel staging -TemplatePackVersion unversioned -Python venv\Scripts\python.exe
```

The script validates the packaging entry point, database migrations, and any
`templates/**/template-pack.json` manifests before building. It then creates:

- `dist/rok-resource-assistant-<version>-<channel>-windows.zip`
- `dist/rok-resource-assistant-<version>-<channel>-windows.zip.sha256`
- `dist/rok-resource-assistant-<version>-<channel>-windows.cyclonedx.json`

Generated files stay under ignored build output paths and must not be committed.

## CI build

`.github/workflows/windows-ci.yml` runs on Windows. It installs
`requirements-lock.txt`, sets `QT_QPA_PLATFORM=offscreen`, validates packaging
inputs, runs the full test suite, builds the PyInstaller artifact, generates a
SHA256 checksum and CycloneDX SBOM, then uploads the generated files as a GitHub
Actions artifact.

## Release checklist

1. Build the `staging` channel and verify tests, migration validation, template
   pack validation, checksum generation, and SBOM generation pass.
2. Promote the same source revision to `canary` with an explicit
   `TemplatePackVersion`.
3. Inspect the uploaded `.sha256` file and verify the zip locally with
   `Get-FileHash -Algorithm SHA256`.
4. Promote to `stable` only after canary validation and release-note review.
5. Publish the zip, checksum, SBOM, application version, template-pack version,
   channel, source commit, and known limitations together.

## Rollback

Rollback means publishing the previous known-good artifact for the target
channel, with its original checksum and SBOM. Do not rebuild from a different
commit and call it a rollback. Keep the application version and template-pack
version visible in the release notes so operators can confirm which artifact is
installed.
