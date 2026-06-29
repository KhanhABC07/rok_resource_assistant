# Shared Context for All Codex Tasks

```text
Work in the existing Python project at `outputs/rok_resource_assistant`. Preserve the current PyQt6 desktop application and SQLite data where practical, but implement changes using a modular-monolith architecture with domain, application, automation, adapter, and presentation boundaries.

Use typed Python, dependency injection, repository migrations, structured logging, and deterministic tests. Every workflow must have explicit preconditions, bounded retries, timeouts, cancellation, scene/session revalidation, postcondition verification, and screenshot evidence on failure. Use scheduler queue and configurable concurrency policies. Never log credentials. Detect verification/CAPTCHA screens and stop for manual intervention; do not implement CAPTCHA solving, anti-detection, or evasion. Do not copy proprietary code or assets from reference binaries.

Run or add unit tests, integration tests with fakes, screenshot replay tests, and update documentation. Keep unrelated behavior backward-compatible. Report changed files, migration steps, test results, and known limitations.
```
