from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.db_helpers import SRC_ROOT  # noqa: F401

from rok_assistant.db.database import Database
from rok_assistant.db.models import (
    AuditLog,
    AutomationProfile,
    Character,
    FeatureConfig,
    GameAccount,
    Incident,
    Instance,
    InstanceSession,
    Job,
    JobRun,
    ScheduleDefinition,
    ScreenObservation,
    StepRun,
    Template,
    TemplatePack,
    WorkflowDefinition,
    WorkflowStep,
)
from rok_assistant.db.repositories import (
    AuditLogRepository,
    AutomationProfileRepository,
    CharacterRepository,
    FeatureConfigRepository,
    GameAccountRepository,
    IncidentRepository,
    InstanceRepository,
    InstanceSessionRepository,
    JobRepository,
    JobRunRepository,
    ScheduleDefinitionRepository,
    ScreenObservationRepository,
    StepRunRepository,
    TemplatePackRepository,
    TemplateRepository,
    WorkflowDefinitionRepository,
    WorkflowStepRepository,
)


class DataV2RepositoryTest(unittest.TestCase):
    def test_v2_repositories_round_trip_all_new_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "repositories_v2.sqlite3")
            db.initialize()
            instances = InstanceRepository(db)
            characters = CharacterRepository(db)
            accounts = GameAccountRepository(db)
            sessions = InstanceSessionRepository(db)
            profiles = AutomationProfileRepository(db)
            features = FeatureConfigRepository(db)
            schedules = ScheduleDefinitionRepository(db)
            workflows = WorkflowDefinitionRepository(db)
            workflow_steps = WorkflowStepRepository(db)
            jobs = JobRepository(db)
            job_runs = JobRunRepository(db)
            step_runs = StepRunRepository(db)
            template_packs = TemplatePackRepository(db)
            templates = TemplateRepository(db)
            observations = ScreenObservationRepository(db)
            incidents = IncidentRepository(db)
            audits = AuditLogRepository(db)

            instance_id = instances.save(Instance(name="MEmu0", instance_name="MEmu0"))
            account_id = accounts.save(
                GameAccount(
                    account_name="Account A",
                    secret_ref="mem://account/a",
                    metadata_json='{"tier": "farm"}',
                )
            )
            self.assertEqual(
                account_id,
                accounts.save(
                    GameAccount(
                        account_name="Account A",
                        display_name="Account A",
                        secret_ref="mem://account/a",
                        metadata_json='{"tier": "farm"}',
                    )
                ),
            )
            self.assertEqual("mem://account/a", accounts.get(account_id).secret_ref)
            character_id = characters.save(
                Character(
                    name="Farm01",
                    instance_id=instance_id,
                    account_name="Account A",
                    character_slot=1,
                    display_fingerprint="farm01-k1001",
                    kingdom_id=1001,
                    verification_metadata_json='{"source": "replay"}',
                )
            )
            self.assertEqual(account_id, characters.get(character_id).game_account_id)
            stored_character = characters.get(character_id)
            self.assertEqual(1, stored_character.character_slot)
            self.assertEqual("farm01-k1001", stored_character.display_fingerprint)
            self.assertEqual(1001, stored_character.kingdom_id)
            self.assertEqual('{"source": "replay"}', stored_character.verification_metadata_json)

            session_id = sessions.save(
                InstanceSession(
                    instance_id=instance_id,
                    session_key="session-1",
                    status="running",
                    started_at="2026-01-01T00:00:00",
                    metadata_json='{"source": "test"}',
                )
            )
            self.assertEqual("running", sessions.get(session_id).status)

            profile_id = profiles.save(
                AutomationProfile(name="Default", metadata_json='{"region": "home"}')
            )
            feature_id = features.save(
                FeatureConfig(
                    profile_id=profile_id,
                    feature_key="alliance_help",
                    config_json='{"max_attempts": 3}',
                )
            )
            schedule_id = schedules.save(
                ScheduleDefinition(
                    profile_id=profile_id,
                    schedule_key="daily-help",
                    name="Daily Help",
                    interval_seconds=3600,
                    config_json='{"jitter": 5}',
                )
            )
            workflow_id = workflows.save(
                WorkflowDefinition(
                    profile_id=profile_id,
                    workflow_key="help-flow",
                    name="Help Flow",
                    config_json='{"kind": "test"}',
                )
            )
            workflow_step_id = workflow_steps.save(
                WorkflowStep(
                    workflow_id=workflow_id,
                    step_order=1,
                    step_key="open-help",
                    action_type="ClickTemplate",
                    parameters_json='{"template": "help"}',
                    timeout_seconds=30,
                    retry_limit=1,
                )
            )

            job_id = jobs.save(
                Job(
                    workflow_id=workflow_id,
                    schedule_id=schedule_id,
                    character_id=character_id,
                    idempotency_key="job-1",
                    job_type="workflow",
                    scheduled_for="2026-01-01T00:05:00",
                    payload_json='{"source": "schedule"}',
                )
            )
            self.assertEqual(
                job_id,
                jobs.save(
                    Job(
                        workflow_id=workflow_id,
                        schedule_id=schedule_id,
                        character_id=character_id,
                        idempotency_key="job-1",
                        job_type="workflow",
                        status="queued",
                        scheduled_for="2026-01-01T00:05:00",
                    )
                ),
            )
            self.assertEqual("queued", jobs.get_by_key("job-1").status)

            job_run_id = job_runs.save(
                JobRun(
                    job_id=job_id,
                    run_key="run-1",
                    started_at="2026-01-01T00:06:00",
                    result_json='{"started": true}',
                )
            )
            step_run_id = step_runs.save(
                StepRun(
                    job_run_id=job_run_id,
                    workflow_step_id=workflow_step_id,
                    step_key="open-help",
                    started_at="2026-01-01T00:06:01",
                    result_json='{"clicked": true}',
                )
            )

            pack_id = template_packs.save(
                TemplatePack(
                    pack_key="core",
                    name="Core",
                    version="1",
                    metadata_json='{"owner": "test"}',
                )
            )
            template_id = templates.save(
                Template(
                    pack_id=pack_id,
                    template_key="help-button",
                    name="Help Button",
                    file_path="templates/help.png",
                    image_hash="hash-help",
                    threshold=0.9,
                )
            )
            observation_id = observations.save(
                ScreenObservation(
                    observation_key="obs-1",
                    instance_id=instance_id,
                    character_id=character_id,
                    job_run_id=job_run_id,
                    observed_at="2026-01-01T00:06:02",
                    scene_name="home",
                    screenshot_path="runtime/screens/obs-1.png",
                    metadata_json='{"template_id": %d}' % template_id,
                )
            )
            incident_id = incidents.save(
                Incident(
                    incident_key="incident-1",
                    severity="warning",
                    status="open",
                    title="Template slow",
                    job_run_id=job_run_id,
                    step_run_id=step_run_id,
                    screenshot_path="runtime/screens/incident-1.png",
                )
            )
            audit_id = audits.append(
                AuditLog(
                    audit_key="audit-1",
                    action="created",
                    entity_type="incident",
                    entity_id=incident_id,
                    occurred_at="2026-01-01T00:06:03",
                    details_json='{"observation_id": %d}' % observation_id,
                )
            )

            self.assertEqual([feature_id], [item.id for item in features.list_for_profile(profile_id)])
            self.assertEqual([schedule_id], [item.id for item in schedules.list_for_profile(profile_id)])
            self.assertEqual([workflow_id], [item.id for item in workflows.list_for_profile(profile_id)])
            self.assertEqual(
                [workflow_step_id],
                [item.id for item in workflow_steps.list_for_workflow(workflow_id)],
            )
            self.assertEqual([job_id], [item.id for item in jobs.list_by_status("queued")])
            self.assertEqual([job_run_id], [item.id for item in job_runs.list_for_job(job_id)])
            self.assertEqual(
                [step_run_id],
                [item.id for item in step_runs.list_for_job_run(job_run_id)],
            )
            self.assertEqual([pack_id], [item.id for item in template_packs.list_all()])
            self.assertEqual([template_id], [item.id for item in templates.list_for_pack(pack_id)])
            self.assertEqual(
                [observation_id],
                [item.id for item in observations.list_recent(limit=10)],
            )
            self.assertEqual([incident_id], [item.id for item in incidents.list_open()])
            self.assertEqual([audit_id], [item.id for item in audits.list_recent(limit=10)])
            self.assertEqual(
                audit_id,
                audits.append(
                    AuditLog(
                        audit_key="audit-1",
                        action="ignored",
                        entity_type="incident",
                        occurred_at="2026-01-01T00:06:04",
                    )
                ),
            )
            self.assertEqual("created", audits.get_by_key("audit-1").action)
            db.close()

    def test_v2_repository_validation_rejects_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "validation_v2.sqlite3")
            db.initialize()
            profiles = AutomationProfileRepository(db)
            jobs = JobRepository(db)
            template_packs = TemplatePackRepository(db)
            templates = TemplateRepository(db)

            with self.assertRaises(ValueError):
                profiles.save(AutomationProfile(name="Invalid", metadata_json="[]"))
            with self.assertRaises(ValueError):
                jobs.save(
                    Job(
                        idempotency_key="job-invalid",
                        job_type="workflow",
                        status="paused",
                    )
                )
            pack_id = template_packs.save(TemplatePack(pack_key="core", name="Core"))
            with self.assertRaises(ValueError):
                templates.save(
                    Template(
                        pack_id=pack_id,
                        template_key="bad",
                        name="Bad",
                        file_path="bad.png",
                        threshold=1.5,
                    )
                )
            db.close()

    def test_character_save_requires_existing_game_account(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "account_validation.sqlite3")
            db.initialize()
            instances = InstanceRepository(db)
            characters = CharacterRepository(db)
            accounts = GameAccountRepository(db)

            instance_id = instances.save(Instance(name="MEmu0"))
            with self.assertRaisesRegex(ValueError, "Game account must exist"):
                characters.save(
                    Character(
                        name="Farm01",
                        instance_id=instance_id,
                        account_name="Account A",
                    )
                )

            account_id = accounts.save(GameAccount(account_name="Account A"))
            character_id = characters.save(
                Character(
                    name="Farm01",
                    instance_id=instance_id,
                    account_name="Account A",
                )
            )
            self.assertEqual(account_id, characters.get(character_id).game_account_id)
            db.close()


if __name__ == "__main__":
    unittest.main()
