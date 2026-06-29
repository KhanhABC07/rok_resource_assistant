# Rise of Kingdoms Resource Assistant

A Windows desktop management application scaffold for coordinating Android emulator instances, characters, marches, scheduled tasks, logs, and JSON configuration.

The project is intentionally built as a maintainable automation framework:

- Python application code under `src/rok_assistant`
- PyQt6 GUI with dashboard and configuration tabs
- SQLite persistence with repository classes
- JSON import/export and backup/restore
- Event-driven scheduler with worker pool
- Emulator manager abstraction
- Plugin-style task modules
- Vision/OCR abstraction for future implementation

The built-in task plugins are conservative stubs. They launch configured emulator commands and record task flow, but actual game-specific screen interaction and OCR must be implemented in task or vision plugins.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Runtime files are created under `runtime/`:

- `runtime/rok_assistant.sqlite3`
- `runtime/logs/app.log`
- JSON backups/exports as selected in the GUI

## Project Layout

```text
rok_resource_assistant/
  main.py
  requirements.txt
  pyproject.toml
  config/
    app_config.json
  docs/
    architecture.md
  src/rok_assistant/
    app.py
    config.py
    logging_setup.py
    paths.py
    recovery.py
    db/
    scheduler/
    emulator/
    characters/
    tasks/
    vision/
    gui/
    plugins/
  tests/
```

## Notes

Use the application only in ways that comply with the terms of service for the software and services you control. Emulator commands and automation actions are user-configured and should be tested on non-critical profiles first.
