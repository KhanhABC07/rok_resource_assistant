# Rise of Kingdoms Automation Assistant

A Windows desktop automation framework for coordinating Rise of Kingdoms workflows across Android emulator instances, accounts, characters, and scheduled jobs.

The application is built with Python, PyQt6, SQLite, OpenCV, and MEmu/ADB integrations. It is under active development. The repository currently provides the desktop shell, persistence, scheduler, emulator adapters, task infrastructure, template capture, image matching, logging, and initial resource workflows. Game-specific workflows are implemented incrementally through task-scoped changes.

## Project Status

### Available foundation

- PyQt6 desktop application and management screens
- SQLite database and repository layer
- JSON configuration, import/export, and backups
- Scheduler and worker pool
- MEmu and ADB command adapters
- Character, instance, march, and task management
- Task engine and plugin infrastructure
- Template capture and OpenCV image matching
- Runtime logging, screenshots, and recovery utilities
- Automated unit and GUI-oriented tests

### In development

- Account switching for up to 6 configured accounts
- Character switching for up to 12 characters per account
- Multi-instance resource gathering
- Alliance pit and gem gathering
- City, alliance, quest, VIP, troop, inventory, map, and canyon workflows
- Rally participation, peace shield handling, and game reboot recovery
- Stronger workflow state validation, bounded retries, and postcondition verification

A feature must not be treated as production-ready merely because its menu, model, plugin, or placeholder exists. Check the relevant task implementation and automated tests before enabling it on a live profile.

## Technology Stack

- Python 3.11+
- PyQt6
- SQLite
- OpenCV
- MEmu emulator
- ADB and MEmu command-line tools
- pytest for automated tests

## Requirements

- Windows 10 or Windows 11
- Python 3.11 or newer
- MEmu installed with ADB access enabled
- A consistent emulator display profile; the initial reference profile is 1280 × 720 at 100% DPI
- Game and emulator profiles prepared manually before automation is enabled

The default MEmu installation path is:

```text
C:\MEmu\Microvirt\MEmu
```

Change it in `config/app_config.json` when MEmu is installed elsewhere.

## Quick Start

Run the following commands from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
python main.py
```

Alternatively, install the application dependencies from `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
python -m pip install pytest
python main.py
```

## Configuration

The main configuration file is:

```text
config/app_config.json
```

Important settings include:

| Setting | Purpose |
|---|---|
| `database.path` | SQLite database location |
| `logging.level` | Application log level |
| `logging.file` | Main log file location |
| `emulator.memu_install_path` | MEmu installation directory |
| `scheduler.max_workers` | Maximum worker threads |
| `scheduler.max_active_instances` | Maximum emulator profiles handled concurrently |
| `scheduler.retry_delay_minutes` | Delay before retrying recoverable failures |
| `scheduler.poll_interval_seconds` | Scheduler polling interval |
| `gathering.preferred_resource_levels` | Preferred resource node levels |
| `gathering.minimum_resource_level` | Lowest accepted resource level |
| `plugins.packages` | Python packages scanned for task plugins |

Do not commit credentials, account secrets, emulator identity data, or private runtime data.

## Running Tests

Run a focused test while developing:

```powershell
python -m pytest tests/test_resource_search_workflow.py -q
```

Run the full test suite before completing a task:

```powershell
python -m pytest tests -q
```

Some GUI and emulator integration tests require Windows, PyQt6, MEmu, or a configured test profile. Tests that cannot run in the current environment must be reported explicitly; they must not be marked as passed without execution.

## Project Layout

```text
rok_resource_assistant/
├── AGENTS.md
├── README.md
├── main.py
├── pyproject.toml
├── requirements.txt
├── config/
│   └── app_config.json
├── docs/
│   ├── architecture.md
│   └── codex/
│       ├── 00_SHARED_CONTEXT.md
│       └── tasks/
├── runtime/
│   ├── assets/templates/
│   ├── backups/
│   ├── logs/
│   ├── screenshots/
│   └── rok_assistant.sqlite3
├── src/rok_assistant/
│   ├── characters/
│   ├── db/
│   ├── emulator/
│   ├── gui/
│   ├── plugins/
│   ├── scheduler/
│   ├── tasks/
│   └── vision/
└── tests/
```

`runtime/` contains local data and generated evidence. It must remain outside source control.

## Architecture Direction

The target architecture is a modular monolith with clear boundaries:

- **Domain:** entities, value objects, policies, and task results
- **Application:** use cases, workflow orchestration, scheduling, and recovery
- **Automation:** stateful game workflows with explicit preconditions and postconditions
- **Adapters:** SQLite, MEmu, ADB, OpenCV, filesystem, and credential storage
- **Presentation:** PyQt6 windows, dialogs, view models, and user actions

Core rules:

- GUI widgets must not contain game automation logic.
- Database access must go through repositories and migrations.
- Emulator operations must go through emulator adapters.
- Image recognition must go through vision interfaces.
- Workflows must support timeout, cancellation, bounded retries, scene/session revalidation, and structured failure results.
- Scheduling uses a durable queue and configurable concurrency. The project does not use exclusive-lock or lease subsystems.

See `docs/architecture.md` for the current architecture description. Keep that document synchronized when a task changes important boundaries or contracts.

## Codex Development Workflow

Repository-wide agent rules are defined in `AGENTS.md`.

Task-specific instructions should be stored under:

```text
docs/codex/tasks/
```

For each Codex run, provide only:

1. `docs/codex/00_SHARED_CONTEXT.md`
2. One task file, such as `docs/codex/tasks/DATA-001.md`
3. The dependencies already merged
4. The current focused and full test results
5. A clear instruction to implement only that task

Do not ask Codex to implement the complete task catalog in one run. Use one branch or pull request per task unless a task has been deliberately divided into reviewed subtasks.

## Optional CodeGraph Integration

CodeGraph may be used for repository navigation and impact analysis:

- Identify callers, callees, imports, symbols, and dependency paths
- Find related tests and implementations before editing
- Estimate the impact of a proposed change

CodeGraph is not the source of truth. Verify graph results against the actual source code and tests. Do not expand a task merely because the graph reveals neighboring modules.

Keep only one `.codegraph/` directory at the actual Git repository root. Its local database, WAL files, logs, and process files must not be committed. Rebuild or refresh the graph after major package moves, interface changes, or large branch merges.

## Runtime Data and Source Control

The following content must remain local:

```text
.venv/
.codegraph/
runtime/
.pytest_cache/
__pycache__/
dist/
build/
*.egg-info/
```

Template images may be source-controlled only when they are intentionally reviewed project assets and their origin and usage rights are clear. Failure screenshots, logs, database files, and captured account data must not be committed.

## Safety and Operational Constraints

- Use the application only where you are authorized to do so and in compliance with applicable terms and policies.
- Test new automation on non-critical profiles first.
- Stop for manual intervention when verification or CAPTCHA screens are detected.
- Do not implement CAPTCHA solving, anti-detection, fingerprint spoofing, or evasion mechanisms.
- Do not copy proprietary code or assets from packaged reference applications.
- Never log account credentials or secret values.

## Documentation

When behavior changes, update the relevant documentation in the same task:

- `README.md` for setup, supported behavior, or operator-facing changes
- `AGENTS.md` for repository-wide engineering rules
- `docs/architecture.md` for architecture and contracts
- `docs/codex/tasks/` for task-specific requirements
