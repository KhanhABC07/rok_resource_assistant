from __future__ import annotations

import sqlite3
from collections.abc import Iterator


def iter_sql_statements(script: str) -> Iterator[str]:
    buffer: list[str] = []
    in_trigger = False
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped and not buffer:
            continue

        buffer.append(line)
        upper = stripped.upper()
        if upper.startswith("CREATE TRIGGER"):
            in_trigger = True

        if in_trigger:
            if upper == "END;":
                yield "\n".join(buffer).strip()
                buffer = []
                in_trigger = False
            continue

        if stripped.endswith(";"):
            yield "\n".join(buffer).strip()
            buffer = []

    trailing = "\n".join(buffer).strip()
    if trailing:
        yield trailing


def execute_script(connection: sqlite3.Connection, script: str) -> None:
    for statement in iter_sql_statements(script):
        connection.execute(statement)
