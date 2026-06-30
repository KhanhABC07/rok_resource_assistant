# Repository Instructions for Coding Agents

These instructions apply to the entire repository unless a more specific `AGENTS.md` exists in a subdirectory.

## 1. Project Context

This repository contains a Windows desktop automation application for coordinating Rise of Kingdoms workflows across MEmu emulator instances, accounts, characters, marches, and scheduled jobs.

Primary technologies:

- Python 3.11+
- PyQt6
- SQLite
- OpenCV
- MEmu and ADB command-line tools
- pytest

The repository is being developed as a modular monolith. Preserve existing behavior and user data while improving boundaries incrementally.

## 2. Sources of Truth

Use the following order of authority:

1. The explicit user request for the current run
2. The assigned task file under `docs/codex/tasks/`
3. `docs/codex/00_SHARED_CONTEXT.md`
4. This `AGENTS.md`
5. Current source code and automated tests
6. `docs/architecture.md` and other repository documentation
7. CodeGraph or other generated indexes

When requirements conflict, stop and report the conflict instead of silently choosing a broader interpretation.

## 3. Task Scope

Implement only the explicitly assigned task.

Before editing:

1. Read the shared context and the assigned task file.
2. Inspect the relevant source files and tests.
3. Identify direct dependencies and affected contracts.
4. State a concise implementation plan.
5. List any unsupported requirement or ambiguity that materially blocks the task.

During implementation:

- Do not implement neighboring catalog tasks.
- Do not perform unrelated refactors.
- Do not change public behavior outside the task without a documented compatibility reason.
- Do not replace working infrastructure solely to match personal preferences.
- Keep changes reviewable and suitable for one branch or pull request.
- When a task is too large, propose technical subtasks while preserving the parent acceptance criteria.

## 4. Architecture Rules

Maintain clear boundaries between:

- **Domain:** entities, value objects, policies, and structured results
- **Application:** use cases, workflow orchestration, scheduling, and recovery
- **Automation:** game-specific state machines and interaction steps
- **Adapters:** SQLite, filesystem, MEmu, ADB, OpenCV, and credential storage
- **Presentation:** PyQt6 UI and user interaction

Required constraints:

- Do not put automation logic in PyQt6 widgets.
- Do not access SQLite directly from GUI code or workflow steps.
- Use repositories for persistence.
- Use migrations for schema changes.
- Use emulator interfaces/adapters for MEmu and ADB commands.
- Use vision interfaces for screenshots, matching, OCR, and scene detection.
- Prefer dependency injection over hidden global state.
- Preserve the modular-monolith deployment model unless a task explicitly changes it.
- Do not introduce microservices, message brokers, or external infrastructure without an explicit requirement.

## 5. Scheduling and Concurrency

Use the existing scheduler queue, worker pool, priorities, cancellation, and configurable concurrency policies.

Do not add any of the following:

- Exclusive instance locks
- Instance leases
- March leases
- Heartbeat leases
- Distributed-lock services
- Lock ownership tables or renewal daemons

Prevent unsafe overlap through queue dispatch rules, target-aware scheduling, workflow state validation, cancellation, and configurable concurrency limits.

Do not assume cached UI state is still valid when a queued job begins. Revalidate the active emulator, account, character, and scene before executing consequential actions.

## 6. Workflow Requirements

Every automation workflow must define:

- Input validation
- Explicit preconditions
- Named states or clearly bounded steps
- Per-step and total timeouts
- Cancellation checks
- Bounded retries with backoff where appropriate
- Scene and session revalidation
- Structured success and failure results
- Postcondition verification
- Recovery behavior
- Screenshot evidence for actionable failures
- Safe behavior when the screen is unknown

Actions that spend premium currency, consume rare items, change account identity, join combat, or alter defensive state must require stronger verification than routine collection actions.

Do not report success based only on a click. Verify the resulting screen, value, state, or other observable postcondition.

## 7. Emulator and Windows Rules

- Treat Windows as the primary production platform.
- Build command arguments as lists rather than manually concatenated shell strings.
- Use `pathlib.Path`, but verify Windows path behavior in tests.
- Avoid `shell=True` unless an explicit, reviewed requirement makes it unavoidable.
- Capture command, exit code, stdout, stderr, timeout, and target instance in structured diagnostics without exposing secrets.
- Do not assume MEmu is installed at the default path; read configuration.
- Keep emulator-specific logic behind adapters so future providers can be added without rewriting workflows.
- Validate ADB connectivity and target identity before input commands.

## 8. Vision and Template Rules

- Access image recognition through the vision layer.
- Prefer semantic template keys over hard-coded filenames in workflows.
- Keep thresholds, regions of interest, scales, language, and display profile configurable where practical.
- Record confidence and the selected match region for diagnostics.
- Use deterministic screenshot fixtures for vision tests.
- Do not treat low-confidence matches as confirmed UI state.
- Avoid full-screen matching when a stable region of interest is available.
- Preserve original failure screenshots; do not overwrite evidence from a different run.

Template assets must have a known source and permission for use. Do not extract or copy proprietary assets from reference binaries.

## 9. Database and Migration Rules

- Preserve existing SQLite data whenever practical.
- Use forward-only, versioned migrations.
- Never silently delete, reset, or recreate an existing user database.
- Make migrations transactional where SQLite permits.
- Add indexes and constraints deliberately and test migration from the previous schema.
- Keep SQL and persistence details inside the database/repository layer.
- Use UTC timestamps for persisted execution data unless an existing contract requires otherwise.
- Never store plaintext credentials in SQLite.

A task that changes schema must report:

- Previous and new schema version
- Tables, columns, constraints, and indexes changed
- Migration and rollback/recovery considerations
- Tests covering both fresh initialization and upgrade of existing data

## 10. Security and Privacy

- Never commit or log credentials, authentication tokens, email secrets, account identifiers intended to remain private, or emulator identity data.
- Use an approved credential adapter, such as Windows Credential Manager, when the assigned task requires secret storage.
- Redact secrets from exceptions, structured logs, screenshots, fixtures, and test output.
- Detect verification or CAPTCHA screens and stop for manual intervention.
- Do not implement CAPTCHA solving.
- Do not implement anti-detection, behavioral evasion, fingerprint spoofing, or mechanisms intended to bypass platform protections.
- Do not copy proprietary source code or assets from packaged reference applications.

## 11. Code Quality

Follow the existing repository style unless a task introduces an approved formatter or linter.

Required practices:

- Use `from __future__ import annotations` in new Python modules where consistent with the package.
- Add type hints to public functions, methods, dataclasses, and service interfaces.
- Prefer small functions with explicit inputs and outputs.
- Use descriptive domain names instead of generic utility abstractions.
- Avoid broad `except Exception` unless the boundary records context and re-raises or returns a structured failure.
- Avoid mutable module-level state.
- Use structured logging instead of `print` in application code.
- Keep comments focused on intent, constraints, and non-obvious behavior.
- Remove dead code introduced by the task.
- Do not add a new dependency when the standard library or an existing dependency is sufficient.
- Document any added dependency and explain why it is necessary.

## 12. PyQt6 Rules

- Keep the UI responsive; do not run emulator commands, database migrations, or long vision operations on the GUI thread.
- Communicate from workers to the UI through signals, queued callbacks, or existing application services.
- Keep widgets focused on presentation and input validation.
- Do not duplicate domain or workflow rules in multiple screens.
- Preserve existing window startup and shutdown behavior.
- Add GUI tests for meaningful user-facing changes when the environment supports them.

## 13. CodeGraph Usage

CodeGraph is optional and may be used to:

- Locate symbols, imports, callers, and callees
- Find related tests and implementations
- Estimate impact before editing
- Verify that a refactor has not left obsolete references

Rules:

- The assigned task remains the authoritative scope.
- Verify CodeGraph findings against source files and tests.
- Do not edit generated `.codegraph` database, WAL, shared-memory, PID, or log files.
- Do not commit `.codegraph` runtime files.
- Keep only one CodeGraph index at the actual Git repository root.
- Refresh or rebuild the graph after major package moves, interface changes, or large merges.
- Do not expand the task simply because CodeGraph exposes adjacent modules.

## 14. Files That Must Not Be Modified or Committed Accidentally

Do not commit generated or private content from:

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

Do not use the local SQLite database, logs, screenshots, or captured templates as test fixtures unless they have been deliberately sanitized and copied into a reviewed test-fixture directory.

## 15. Testing Requirements

Add or update tests for every behavioral change.

Test in this order:

1. Tests directly related to the changed module
2. Contract or integration tests using fakes/mocks
3. Relevant screenshot replay tests for vision behavior
4. Relevant GUI tests
5. Full suite

Common commands:

```powershell
python -m pytest tests/test_target_module.py -q
python -m pytest tests -q
```

Testing rules:

- Tests must be deterministic.
- Do not depend on execution order.
- Do not call a live game account in unit tests.
- Mock or fake MEmu, ADB, clock, filesystem, and vision boundaries where appropriate.
- Use temporary directories and temporary SQLite databases.
- Cover timeout, cancellation, retry exhaustion, invalid state, and recovery paths.
- For Windows command builders, test paths containing spaces and backslashes.
- Never claim a test passed unless it was executed successfully.
- If a test cannot run because of the environment, report the exact command, blocker, and remaining validation required.

## 16. Documentation Requirements

Update documentation in the same task when the change affects:

- Setup or configuration
- User-visible behavior
- Architecture or dependency direction
- Database schema
- Internal service contracts
- Task execution or recovery
- Required developer commands

Do not copy the entire task catalog into `README.md` or `AGENTS.md`. Keep detailed acceptance criteria in the individual task files under `docs/codex/tasks/`.

## 17. Completion Report

At the end of every task, report:

1. Implementation summary
2. Files changed
3. Important design decisions
4. Database or configuration changes
5. Tests added or updated
6. Exact test commands executed
7. Test results
8. Known limitations
9. Follow-up tasks not implemented

Do not state that the whole feature is complete when only infrastructure, placeholders, UI controls, or partial workflow states were implemented.

## 18. Definition of Done

A task is complete only when:

- The assigned acceptance criteria are implemented.
- Unrelated behavior remains backward-compatible.
- Required migrations preserve existing data.
- Focused tests pass.
- The full available test suite has been executed or environmental blockers are documented.
- Failure and recovery paths have test coverage appropriate to the task.
- Logging contains enough context to diagnose failure without exposing secrets.
- Relevant documentation is updated.
- The completion report is accurate and separates verified behavior from remaining limitations.
