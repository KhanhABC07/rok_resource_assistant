from __future__ import annotations

import json
import platform
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import AppConfig
from .db.database import Database
from .db.models import DashboardStats
from .paths import LOG_DIR, RUNTIME_DIR
from .security import redact_text, redact_value


@dataclass(frozen=True)
class SupportBundleResult:
    path: Path


class DashboardMetricsService:
    def __init__(self, db: Database):
        self.db = db

    def collect(self, *, active_workers: int, running_instances: int, max_workers: int) -> DashboardStats:
        task_counts = self._counts_by_status("scheduled_tasks")
        job_counts = self._counts_by_status("jobs")
        recent_runs = self._recent_task_results()
        total_recent = sum(recent_runs.values())
        success_count = int(recent_runs.get("SUCCESS", 0))
        success_rate = (success_count / total_recent) if total_recent else 0.0
        blocked_retry_count = int(task_counts.get("retrying", 0) + job_counts.get("aborted", 0))
        active_jobs = int(job_counts.get("running", 0) + job_counts.get("queued", 0))
        return DashboardStats(
            active_workers=active_workers,
            running_instances=running_instances,
            total_characters=self._scalar_int("SELECT COUNT(*) AS value FROM characters"),
            pending_tasks=int(
                task_counts.get("pending", 0)
                + task_counts.get("queued", 0)
                + task_counts.get("running", 0)
                + task_counts.get("retrying", 0)
            ),
            next_scheduled_task=self._next_scheduled(),
            success_count=success_count,
            failure_count=int(recent_runs.get("FAILED", 0)),
            blocked_retry_count=blocked_retry_count,
            queue_depth=int(task_counts.get("pending", 0) + task_counts.get("queued", 0) + job_counts.get("queued", 0)),
            active_jobs=active_jobs,
            concurrency_in_use=active_workers,
            concurrency_limit=max_workers,
            open_incident_count=self._scalar_int(
                "SELECT COUNT(*) AS value FROM incidents WHERE status IN ('open', 'acknowledged')"
            ),
            recent_incident_count=self._scalar_int(
                "SELECT COUNT(*) AS value FROM incidents WHERE created_at >= datetime('now', '-24 hours')"
            ),
            success_rate=success_rate,
            last_run_at=self._last_run_at(),
        )

    def _counts_by_status(self, table: str) -> dict[str, int]:
        rows = self.db.fetch_all(f"SELECT status, COUNT(*) AS total FROM {table} GROUP BY status")
        return {str(row["status"]): int(row["total"]) for row in rows}

    def _recent_task_results(self) -> dict[str, int]:
        rows = self.db.fetch_all(
            """
            SELECT result, COUNT(*) AS total
            FROM (
                SELECT result
                FROM task_run_history
                ORDER BY started_at DESC, id DESC
                LIMIT 100
            )
            GROUP BY result
            """
        )
        return {str(row["result"]): int(row["total"]) for row in rows}

    def _next_scheduled(self) -> str:
        row = self.db.fetch_one(
            """
            SELECT scheduled_for
            FROM scheduled_tasks
            WHERE status IN ('pending', 'retrying')
            ORDER BY scheduled_for ASC
            LIMIT 1
            """
        )
        return str(row["scheduled_for"]) if row else "-"

    def _last_run_at(self) -> str:
        row = self.db.fetch_one(
            """
            SELECT MAX(started_at) AS value
            FROM (
                SELECT started_at FROM task_run_history
                UNION ALL
                SELECT started_at FROM job_runs
            )
            """
        )
        return str(row["value"]) if row and row["value"] else "-"

    def _scalar_int(self, sql: str) -> int:
        row = self.db.fetch_one(sql)
        return int(row["value"] if row else 0)


class SupportBundleExporter:
    def __init__(
        self,
        *,
        db: Database,
        config: AppConfig,
        output_dir: Path | None = None,
        runtime_dir: Path = RUNTIME_DIR,
        log_dir: Path = LOG_DIR,
    ) -> None:
        self.db = db
        self.config = config
        self.output_dir = output_dir or (runtime_dir / "support_bundles")
        self.runtime_dir = runtime_dir
        self.log_dir = log_dir

    def export(self) -> SupportBundleResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        generated_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"support-bundle-{generated_at}.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            self._write_json(archive, "metadata.json", self._metadata())
            self._write_json(archive, "configuration/redacted_config.json", redact_value(self.config.data))
            self._write_json(archive, "configuration/redacted_settings.json", self._settings_snapshot())
            self._write_json(archive, "incidents/recent_incidents.json", self._recent_incidents())
            self._write_json(archive, "evidence/recent_references.json", self._recent_evidence_references())
            for log_file in self._recent_logs():
                self._write_redacted_log(archive, log_file)
        return SupportBundleResult(path=path)

    def _metadata(self) -> dict[str, object]:
        return {
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "app_version": __version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "database_path": str(self.config.database_path),
        }

    def _settings_snapshot(self) -> dict[str, object]:
        rows = self.db.fetch_all("SELECT key, value FROM settings ORDER BY key")
        return redact_value({str(row["key"]): str(row["value"]) for row in rows})

    def _recent_incidents(self) -> list[dict[str, object]]:
        rows = self.db.fetch_all(
            """
            SELECT id, incident_key, severity, status, title, details, job_run_id,
                   step_run_id, screenshot_path, created_at, resolved_at, updated_at
            FROM incidents
            ORDER BY created_at DESC, id DESC
            LIMIT 100
            """
        )
        return [redact_value(dict(row)) for row in rows]

    def _recent_evidence_references(self) -> list[dict[str, object]]:
        rows = self.db.fetch_all(
            """
            SELECT 'incident' AS source, id AS source_id, screenshot_path AS path, created_at AS captured_at
            FROM incidents
            WHERE screenshot_path <> ''
            UNION ALL
            SELECT 'screen_observation' AS source, id AS source_id, screenshot_path AS path, observed_at AS captured_at
            FROM screen_observations
            WHERE screenshot_path <> ''
            ORDER BY captured_at DESC
            LIMIT 100
            """
        )
        return [dict(row) for row in rows]

    def _recent_logs(self) -> list[Path]:
        if not self.log_dir.exists():
            return []
        return sorted(
            (path for path in self.log_dir.glob("*.log*") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:6]

    @staticmethod
    def _write_json(archive: zipfile.ZipFile, name: str, payload: Any) -> None:
        archive.writestr(
            name,
            json.dumps(redact_value(payload), sort_keys=True, indent=2) + "\n",
        )

    @staticmethod
    def _write_redacted_log(archive: zipfile.ZipFile, path: Path) -> None:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = ""
        archive.writestr(f"logs/{path.name}", redact_text(content))
