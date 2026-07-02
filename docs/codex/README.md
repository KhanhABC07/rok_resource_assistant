# ROK Codex Tasks - Usage

Use one task at a time.

Recommended input for each Codex run:

1. Attach or paste `00_SHARED_CONTEXT.md`.
2. Attach exactly one task file from `docs/codex/`.
3. State which dependencies are already merged and provide the current test command/result.
4. Ask Codex to implement only that task, keep unrelated behavior unchanged, run tests, and report changed files and limitations.

Do not ask Codex to implement the entire catalog in one run. The complete catalog is useful only for planning, dependency review, or selecting the next task.

Suggested execution order:

`ARCH-001 -> DATA-001/EMU-001 -> EXEC-001 -> SCHED-001/VISION-001 -> REC-001/OBS-001/SEC-001/PACK-001 -> ACC-001/CHR-001/OPS-001 -> functional tasks`
