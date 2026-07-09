from __future__ import annotations

import json
import logging
import tempfile
import unittest
import zipfile
from pathlib import Path

from tests.db_helpers import SRC_ROOT  # noqa: F401

from rok_assistant.config import AppConfig, DEFAULT_CONFIG
from rok_assistant.db.database import Database
from rok_assistant.db.models import Character, Incident, Instance, TaskRunHistory
from rok_assistant.db.repositories import (
    CharacterRepository,
    IncidentRepository,
    InstanceRepository,
    SettingsRepository,
    TaskRunHistoryRepository,
)
from rok_assistant.logging_setup import configure_logging, log_context
from rok_assistant.observability import DashboardMetricsService, SupportBundleExporter


class ObservabilityTest(unittest.TestCase):
    def tearDown(self) -> None:
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()

    def test_json_log_event_shape_correlation_and_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "app.log"
            configure_logging(log_file, "INFO")

            with log_context(job_id=7, run_id=11, instance_id=3, character_id=5):
                logging.getLogger("obs-test").info(
                    "credential={'password':'plain'}",
                    extra={"feature_key": "gathering"},
                )

            for handler in logging.getLogger().handlers:
                handler.flush()
            event = json.loads(log_file.read_text(encoding="utf-8").splitlines()[0])
            self._close_root_handlers()

            self.assertEqual("INFO", event["level"])
            self.assertEqual("obs-test", event["logger"])
            self.assertEqual(7, event["job_id"])
            self.assertEqual(11, event["run_id"])
            self.assertEqual(3, event["instance_id"])
            self.assertEqual(5, event["character_id"])
            self.assertEqual("gathering", event["feature_key"])
            self.assertIn("timestamp", event)
            self.assertNotIn("plain", json.dumps(event))
            self.assertIn("[REDACTED]", event["message"])

    def test_dashboard_metrics_aggregate_existing_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = self._db(temp_dir)
            instances = InstanceRepository(db)
            characters = CharacterRepository(db)
            histories = TaskRunHistoryRepository(db)
            incidents = IncidentRepository(db)

            instance_id = instances.save(Instance(name="MEmu0", instance_name="MEmu0"))
            character_id = characters.save(Character(name="Farm01", instance_id=instance_id))
            db.execute(
                """
                INSERT INTO scheduled_tasks(character_id, task_type, status, scheduled_for)
                VALUES (?, 'gathering', 'pending', '2026-07-09T01:00:00')
                """,
                (character_id,),
            )
            db.execute(
                """
                INSERT INTO scheduled_tasks(character_id, task_type, status, scheduled_for)
                VALUES (?, 'gathering', 'retrying', '2026-07-09T02:00:00')
                """,
                (character_id,),
            )
            db.execute(
                """
                INSERT INTO jobs(idempotency_key, job_type, status, scheduled_for)
                VALUES ('job-1', 'workflow', 'queued', '2026-07-09T00:00:00')
                """
            )
            histories.create(
                TaskRunHistory(
                    task_name="Gather",
                    started_at="2026-07-09T00:00:00",
                    finished_at="2026-07-09T00:01:00",
                    result="SUCCESS",
                )
            )
            histories.create(
                TaskRunHistory(
                    task_name="Gather",
                    started_at="2026-07-09T00:02:00",
                    finished_at="2026-07-09T00:03:00",
                    result="FAILED",
                )
            )
            incidents.save(Incident(incident_key="obs-incident", title="Blocked"))

            stats = DashboardMetricsService(db).collect(
                active_workers=1,
                running_instances=2,
                max_workers=5,
            )

            self.assertEqual(1, stats.success_count)
            self.assertEqual(1, stats.failure_count)
            self.assertEqual(1, stats.blocked_retry_count)
            self.assertEqual(2, stats.queue_depth)
            self.assertEqual(1, stats.active_jobs)
            self.assertEqual(1, stats.open_incident_count)
            self.assertEqual(0.5, stats.success_rate)
            self.assertEqual("2026-07-09T00:02:00", stats.last_run_at)
            db.close()

    def test_support_bundle_contains_redacted_config_logs_incidents_and_evidence_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = self._db(temp_dir)
            SettingsRepository(db).set("api.token", "secret-token")
            IncidentRepository(db).save(
                Incident(
                    incident_key="incident-1",
                    title="Verification screen",
                    details="password='secret'",
                    screenshot_path="screenshots/failure.png",
                )
            )
            log_dir = root / "logs"
            log_dir.mkdir()
            (log_dir / "app.log").write_text("token=secret-token\n", encoding="utf-8")
            config = AppConfig(
                path=root / "app_config.json",
                data={**DEFAULT_CONFIG, "credentials": {"password": "secret"}},
            )

            result = SupportBundleExporter(
                db=db,
                config=config,
                output_dir=root / "bundles",
                runtime_dir=root,
                log_dir=log_dir,
            ).export()

            with zipfile.ZipFile(result.path) as bundle:
                names = set(bundle.namelist())
                self.assertIn("metadata.json", names)
                self.assertIn("configuration/redacted_config.json", names)
                self.assertIn("configuration/redacted_settings.json", names)
                self.assertIn("incidents/recent_incidents.json", names)
                self.assertIn("evidence/recent_references.json", names)
                self.assertIn("logs/app.log", names)
                config_snapshot = bundle.read("configuration/redacted_config.json").decode("utf-8")
                settings_snapshot = bundle.read("configuration/redacted_settings.json").decode("utf-8")
                incidents = bundle.read("incidents/recent_incidents.json").decode("utf-8")
                log_content = bundle.read("logs/app.log").decode("utf-8")

            self.assertNotIn("secret", config_snapshot)
            self.assertNotIn("secret-token", settings_snapshot)
            self.assertNotIn("password='secret'", incidents)
            self.assertNotIn("secret-token", log_content)
            self.assertIn("[REDACTED]", config_snapshot)
            self.assertIn("[REDACTED]", settings_snapshot)
            self.assertIn("[REDACTED]", log_content)
            db.close()

    @staticmethod
    def _db(temp_dir: str) -> Database:
        db = Database(Path(temp_dir) / "observability.sqlite3")
        db.initialize()
        return db

    @staticmethod
    def _close_root_handlers() -> None:
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    unittest.main()
