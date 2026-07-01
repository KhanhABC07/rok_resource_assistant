from __future__ import annotations

from typing import Any

from .database import Database
from .models import (
    INCIDENT_SEVERITIES,
    INCIDENT_STATUSES,
    INSTANCE_SESSION_STATUSES,
    JOB_STATUSES,
    RUN_STATUSES,
    AuditLog,
    AutomationProfile,
    FeatureConfig,
    Incident,
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
    row_bool,
    utc_now_iso,
)
from .repository_helpers import (
    json_object_text,
    require_id,
    require_text,
    row_id,
    validate_choice,
    validate_positive,
)


class InstanceSessionRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, session: InstanceSession) -> int:
        instance_id = require_id(session.instance_id, "instance_id")
        session_key = require_text(session.session_key, "session_key")
        status = validate_choice(session.status, INSTANCE_SESSION_STATUSES, "status")
        metadata_json = json_object_text(session.metadata_json, "metadata_json")
        with self.db.transaction():
            if session.id is None:
                self.db.execute(
                    """
                    INSERT INTO instance_sessions(
                        instance_id, session_key, status, started_at, ended_at,
                        emulator_pid, adb_serial, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_key) DO UPDATE SET
                        instance_id = excluded.instance_id,
                        status = excluded.status,
                        started_at = excluded.started_at,
                        ended_at = excluded.ended_at,
                        emulator_pid = excluded.emulator_pid,
                        adb_serial = excluded.adb_serial,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        instance_id,
                        session_key,
                        status,
                        session.started_at,
                        session.ended_at,
                        session.emulator_pid,
                        session.adb_serial.strip(),
                        metadata_json,
                    ),
                )
                return row_id(self.get_by_key(session_key))

            self.db.execute(
                """
                UPDATE instance_sessions
                SET instance_id = ?,
                    session_key = ?,
                    status = ?,
                    started_at = ?,
                    ended_at = ?,
                    emulator_pid = ?,
                    adb_serial = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    instance_id,
                    session_key,
                    status,
                    session.started_at,
                    session.ended_at,
                    session.emulator_pid,
                    session.adb_serial.strip(),
                    metadata_json,
                    session.id,
                ),
            )
            return session.id

    def get(self, session_id: int) -> InstanceSession | None:
        row = self.db.fetch_one("SELECT * FROM instance_sessions WHERE id = ?", (session_id,))
        return self._from_row(row) if row else None

    def get_by_key(self, session_key: str) -> InstanceSession | None:
        row = self.db.fetch_one(
            "SELECT * FROM instance_sessions WHERE session_key = ?",
            (require_text(session_key, "session_key"),),
        )
        return self._from_row(row) if row else None

    def list_for_instance(self, instance_id: int) -> list[InstanceSession]:
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM instance_sessions
            WHERE instance_id = ?
            ORDER BY COALESCE(started_at, created_at) DESC, id DESC
            """,
            (instance_id,),
        )
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: Any) -> InstanceSession:
        return InstanceSession(
            id=row["id"],
            instance_id=row["instance_id"],
            session_key=row["session_key"],
            status=row["status"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            emulator_pid=row["emulator_pid"],
            adb_serial=row["adb_serial"],
            metadata_json=row["metadata_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class AutomationProfileRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, profile: AutomationProfile) -> int:
        name = require_text(profile.name, "name")
        metadata_json = json_object_text(profile.metadata_json, "metadata_json")
        with self.db.transaction():
            if profile.id is None:
                self.db.execute(
                    """
                    INSERT INTO automation_profiles(name, description, enabled, metadata_json)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        description = excluded.description,
                        enabled = excluded.enabled,
                        metadata_json = excluded.metadata_json
                    """,
                    (name, profile.description.strip(), int(profile.enabled), metadata_json),
                )
                return row_id(self.get_by_name(name))

            self.db.execute(
                """
                UPDATE automation_profiles
                SET name = ?, description = ?, enabled = ?, metadata_json = ?
                WHERE id = ?
                """,
                (
                    name,
                    profile.description.strip(),
                    int(profile.enabled),
                    metadata_json,
                    profile.id,
                ),
            )
            return profile.id

    def get(self, profile_id: int) -> AutomationProfile | None:
        row = self.db.fetch_one(
            "SELECT * FROM automation_profiles WHERE id = ?",
            (profile_id,),
        )
        return self._from_row(row) if row else None

    def get_by_name(self, name: str) -> AutomationProfile | None:
        row = self.db.fetch_one(
            "SELECT * FROM automation_profiles WHERE name = ?",
            (require_text(name, "name"),),
        )
        return self._from_row(row) if row else None

    def list_all(self) -> list[AutomationProfile]:
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM automation_profiles
            ORDER BY enabled DESC, name COLLATE NOCASE
            """
        )
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: Any) -> AutomationProfile:
        return AutomationProfile(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            enabled=row_bool(row["enabled"]),
            metadata_json=row["metadata_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class FeatureConfigRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, config: FeatureConfig) -> int:
        profile_id = require_id(config.profile_id, "profile_id")
        feature_key = require_text(config.feature_key, "feature_key")
        config_json = json_object_text(config.config_json, "config_json")
        with self.db.transaction():
            if config.id is None:
                self.db.execute(
                    """
                    INSERT INTO feature_configs(profile_id, feature_key, enabled, config_json)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(profile_id, feature_key) DO UPDATE SET
                        enabled = excluded.enabled,
                        config_json = excluded.config_json
                    """,
                    (profile_id, feature_key, int(config.enabled), config_json),
                )
                return row_id(self.get_by_key(profile_id, feature_key))

            self.db.execute(
                """
                UPDATE feature_configs
                SET profile_id = ?, feature_key = ?, enabled = ?, config_json = ?
                WHERE id = ?
                """,
                (profile_id, feature_key, int(config.enabled), config_json, config.id),
            )
            return config.id

    def get_by_key(self, profile_id: int, feature_key: str) -> FeatureConfig | None:
        row = self.db.fetch_one(
            """
            SELECT *
            FROM feature_configs
            WHERE profile_id = ? AND feature_key = ?
            """,
            (profile_id, require_text(feature_key, "feature_key")),
        )
        return self._from_row(row) if row else None

    def list_for_profile(self, profile_id: int) -> list[FeatureConfig]:
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM feature_configs
            WHERE profile_id = ?
            ORDER BY feature_key COLLATE NOCASE
            """,
            (profile_id,),
        )
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: Any) -> FeatureConfig:
        return FeatureConfig(
            id=row["id"],
            profile_id=row["profile_id"],
            feature_key=row["feature_key"],
            enabled=row_bool(row["enabled"]),
            config_json=row["config_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class ScheduleDefinitionRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, schedule: ScheduleDefinition) -> int:
        profile_id = require_id(schedule.profile_id, "profile_id")
        schedule_key = require_text(schedule.schedule_key, "schedule_key")
        name = schedule.name.strip() or schedule_key
        config_json = json_object_text(schedule.config_json, "config_json")
        if schedule.interval_seconds is not None and schedule.interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero.")
        with self.db.transaction():
            if schedule.id is None:
                self.db.execute(
                    """
                    INSERT INTO schedule_definitions(
                        profile_id, schedule_key, name, enabled, cron_expression,
                        interval_seconds, timezone, config_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(profile_id, schedule_key) DO UPDATE SET
                        name = excluded.name,
                        enabled = excluded.enabled,
                        cron_expression = excluded.cron_expression,
                        interval_seconds = excluded.interval_seconds,
                        timezone = excluded.timezone,
                        config_json = excluded.config_json
                    """,
                    (
                        profile_id,
                        schedule_key,
                        name,
                        int(schedule.enabled),
                        schedule.cron_expression.strip(),
                        schedule.interval_seconds,
                        schedule.timezone.strip() or "UTC",
                        config_json,
                    ),
                )
                return row_id(self.get_by_key(profile_id, schedule_key))

            self.db.execute(
                """
                UPDATE schedule_definitions
                SET profile_id = ?,
                    schedule_key = ?,
                    name = ?,
                    enabled = ?,
                    cron_expression = ?,
                    interval_seconds = ?,
                    timezone = ?,
                    config_json = ?
                WHERE id = ?
                """,
                (
                    profile_id,
                    schedule_key,
                    name,
                    int(schedule.enabled),
                    schedule.cron_expression.strip(),
                    schedule.interval_seconds,
                    schedule.timezone.strip() or "UTC",
                    config_json,
                    schedule.id,
                ),
            )
            return schedule.id

    def get_by_key(
        self, profile_id: int, schedule_key: str
    ) -> ScheduleDefinition | None:
        row = self.db.fetch_one(
            """
            SELECT *
            FROM schedule_definitions
            WHERE profile_id = ? AND schedule_key = ?
            """,
            (profile_id, require_text(schedule_key, "schedule_key")),
        )
        return self._from_row(row) if row else None

    def list_for_profile(self, profile_id: int) -> list[ScheduleDefinition]:
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM schedule_definitions
            WHERE profile_id = ?
            ORDER BY enabled DESC, name COLLATE NOCASE
            """,
            (profile_id,),
        )
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: Any) -> ScheduleDefinition:
        return ScheduleDefinition(
            id=row["id"],
            profile_id=row["profile_id"],
            schedule_key=row["schedule_key"],
            name=row["name"],
            enabled=row_bool(row["enabled"]),
            cron_expression=row["cron_expression"],
            interval_seconds=row["interval_seconds"],
            timezone=row["timezone"],
            config_json=row["config_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class WorkflowDefinitionRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, workflow: WorkflowDefinition) -> int:
        profile_id = require_id(workflow.profile_id, "profile_id")
        workflow_key = require_text(workflow.workflow_key, "workflow_key")
        version = validate_positive(workflow.version, "version")
        name = workflow.name.strip() or workflow_key
        config_json = json_object_text(workflow.config_json, "config_json")
        with self.db.transaction():
            if workflow.id is None:
                self.db.execute(
                    """
                    INSERT INTO workflow_definitions(
                        profile_id, workflow_key, name, version, enabled,
                        trigger_type, config_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(profile_id, workflow_key, version) DO UPDATE SET
                        name = excluded.name,
                        enabled = excluded.enabled,
                        trigger_type = excluded.trigger_type,
                        config_json = excluded.config_json
                    """,
                    (
                        profile_id,
                        workflow_key,
                        name,
                        version,
                        int(workflow.enabled),
                        workflow.trigger_type.strip() or "manual",
                        config_json,
                    ),
                )
                return row_id(self.get_by_key(profile_id, workflow_key, version))

            self.db.execute(
                """
                UPDATE workflow_definitions
                SET profile_id = ?,
                    workflow_key = ?,
                    name = ?,
                    version = ?,
                    enabled = ?,
                    trigger_type = ?,
                    config_json = ?
                WHERE id = ?
                """,
                (
                    profile_id,
                    workflow_key,
                    name,
                    version,
                    int(workflow.enabled),
                    workflow.trigger_type.strip() or "manual",
                    config_json,
                    workflow.id,
                ),
            )
            return workflow.id

    def get(self, workflow_id: int) -> WorkflowDefinition | None:
        row = self.db.fetch_one(
            "SELECT * FROM workflow_definitions WHERE id = ?",
            (workflow_id,),
        )
        return self._from_row(row) if row else None

    def get_by_key(
        self, profile_id: int, workflow_key: str, version: int = 1
    ) -> WorkflowDefinition | None:
        row = self.db.fetch_one(
            """
            SELECT *
            FROM workflow_definitions
            WHERE profile_id = ? AND workflow_key = ? AND version = ?
            """,
            (profile_id, require_text(workflow_key, "workflow_key"), version),
        )
        return self._from_row(row) if row else None

    def list_for_profile(self, profile_id: int) -> list[WorkflowDefinition]:
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM workflow_definitions
            WHERE profile_id = ?
            ORDER BY workflow_key COLLATE NOCASE, version DESC
            """,
            (profile_id,),
        )
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: Any) -> WorkflowDefinition:
        return WorkflowDefinition(
            id=row["id"],
            profile_id=row["profile_id"],
            workflow_key=row["workflow_key"],
            name=row["name"],
            version=row["version"],
            enabled=row_bool(row["enabled"]),
            trigger_type=row["trigger_type"],
            config_json=row["config_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class WorkflowStepRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, step: WorkflowStep) -> int:
        workflow_id = require_id(step.workflow_id, "workflow_id")
        step_order = validate_positive(step.step_order, "step_order")
        step_key = require_text(step.step_key, "step_key")
        action_type = require_text(step.action_type, "action_type")
        parameters_json = json_object_text(step.parameters_json, "parameters_json")
        if step.retry_limit < 0:
            raise ValueError("retry_limit must be zero or greater.")
        if step.timeout_seconds is not None and step.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        with self.db.transaction():
            if step.id is None:
                self.db.execute(
                    """
                    INSERT INTO workflow_steps(
                        workflow_id, step_order, step_key, action_type, parameters_json,
                        timeout_seconds, retry_limit, enabled
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(workflow_id, step_key) DO UPDATE SET
                        step_order = excluded.step_order,
                        action_type = excluded.action_type,
                        parameters_json = excluded.parameters_json,
                        timeout_seconds = excluded.timeout_seconds,
                        retry_limit = excluded.retry_limit,
                        enabled = excluded.enabled
                    """,
                    (
                        workflow_id,
                        step_order,
                        step_key,
                        action_type,
                        parameters_json,
                        step.timeout_seconds,
                        step.retry_limit,
                        int(step.enabled),
                    ),
                )
                return row_id(self.get_by_key(workflow_id, step_key))

            self.db.execute(
                """
                UPDATE workflow_steps
                SET workflow_id = ?,
                    step_order = ?,
                    step_key = ?,
                    action_type = ?,
                    parameters_json = ?,
                    timeout_seconds = ?,
                    retry_limit = ?,
                    enabled = ?
                WHERE id = ?
                """,
                (
                    workflow_id,
                    step_order,
                    step_key,
                    action_type,
                    parameters_json,
                    step.timeout_seconds,
                    step.retry_limit,
                    int(step.enabled),
                    step.id,
                ),
            )
            return step.id

    def get_by_key(self, workflow_id: int, step_key: str) -> WorkflowStep | None:
        row = self.db.fetch_one(
            """
            SELECT *
            FROM workflow_steps
            WHERE workflow_id = ? AND step_key = ?
            """,
            (workflow_id, require_text(step_key, "step_key")),
        )
        return self._from_row(row) if row else None

    def list_for_workflow(self, workflow_id: int) -> list[WorkflowStep]:
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM workflow_steps
            WHERE workflow_id = ?
            ORDER BY step_order
            """,
            (workflow_id,),
        )
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: Any) -> WorkflowStep:
        return WorkflowStep(
            id=row["id"],
            workflow_id=row["workflow_id"],
            step_order=row["step_order"],
            step_key=row["step_key"],
            action_type=row["action_type"],
            parameters_json=row["parameters_json"],
            timeout_seconds=row["timeout_seconds"],
            retry_limit=row["retry_limit"],
            enabled=row_bool(row["enabled"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class JobRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, job: Job) -> int:
        idempotency_key = require_text(job.idempotency_key, "idempotency_key")
        job_type = require_text(job.job_type, "job_type")
        status = validate_choice(job.status, JOB_STATUSES, "status")
        scheduled_for = job.scheduled_for or utc_now_iso()
        payload_json = json_object_text(job.payload_json, "payload_json")
        with self.db.transaction():
            if job.id is None:
                self.db.execute(
                    """
                    INSERT INTO jobs(
                        workflow_id, schedule_id, character_id, idempotency_key,
                        job_type, status, priority, scheduled_for, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(idempotency_key) DO UPDATE SET
                        workflow_id = excluded.workflow_id,
                        schedule_id = excluded.schedule_id,
                        character_id = excluded.character_id,
                        job_type = excluded.job_type,
                        status = excluded.status,
                        priority = excluded.priority,
                        scheduled_for = excluded.scheduled_for,
                        payload_json = excluded.payload_json
                    """,
                    (
                        job.workflow_id,
                        job.schedule_id,
                        job.character_id,
                        idempotency_key,
                        job_type,
                        status,
                        job.priority,
                        scheduled_for,
                        payload_json,
                    ),
                )
                return row_id(self.get_by_key(idempotency_key))

            self.db.execute(
                """
                UPDATE jobs
                SET workflow_id = ?,
                    schedule_id = ?,
                    character_id = ?,
                    idempotency_key = ?,
                    job_type = ?,
                    status = ?,
                    priority = ?,
                    scheduled_for = ?,
                    payload_json = ?
                WHERE id = ?
                """,
                (
                    job.workflow_id,
                    job.schedule_id,
                    job.character_id,
                    idempotency_key,
                    job_type,
                    status,
                    job.priority,
                    scheduled_for,
                    payload_json,
                    job.id,
                ),
            )
            return job.id

    def get_by_key(self, idempotency_key: str) -> Job | None:
        row = self.db.fetch_one(
            "SELECT * FROM jobs WHERE idempotency_key = ?",
            (require_text(idempotency_key, "idempotency_key"),),
        )
        return self._from_row(row) if row else None

    def list_by_status(self, status: str) -> list[Job]:
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM jobs
            WHERE status = ?
            ORDER BY scheduled_for, priority, id
            """,
            (validate_choice(status, JOB_STATUSES, "status"),),
        )
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: Any) -> Job:
        return Job(
            id=row["id"],
            workflow_id=row["workflow_id"],
            schedule_id=row["schedule_id"],
            character_id=row["character_id"],
            idempotency_key=row["idempotency_key"],
            job_type=row["job_type"],
            status=row["status"],
            priority=row["priority"],
            scheduled_for=row["scheduled_for"],
            payload_json=row["payload_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class JobRunRepository:
    def __init__(self, db: Database):
        self.db = db

    def get(self, run_id: int) -> JobRun | None:
        row = self.db.fetch_one(
            "SELECT * FROM job_runs WHERE id = ?",
            (run_id,),
        )
        return self._from_row(row) if row else None

    def save(self, run: JobRun) -> int:
        job_id = require_id(run.job_id, "job_id")
        run_key = require_text(run.run_key, "run_key")
        status = validate_choice(run.status, RUN_STATUSES, "status")
        started_at = run.started_at or utc_now_iso()
        result_json = json_object_text(run.result_json, "result_json")
        attempt = validate_positive(run.attempt, "attempt")
        with self.db.transaction():
            if run.id is None:
                self.db.execute(
                    """
                    INSERT INTO job_runs(
                        job_id, run_key, status, attempt, started_at, finished_at,
                        result_json, error_message, screenshot_path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_key) DO UPDATE SET
                        job_id = excluded.job_id,
                        status = excluded.status,
                        attempt = excluded.attempt,
                        started_at = excluded.started_at,
                        finished_at = excluded.finished_at,
                        result_json = excluded.result_json,
                        error_message = excluded.error_message,
                        screenshot_path = excluded.screenshot_path
                    """,
                    (
                        job_id,
                        run_key,
                        status,
                        attempt,
                        started_at,
                        run.finished_at,
                        result_json,
                        run.error_message.strip(),
                        run.screenshot_path.strip(),
                    ),
                )
                return row_id(self.get_by_key(run_key))

            self.db.execute(
                """
                UPDATE job_runs
                SET job_id = ?,
                    run_key = ?,
                    status = ?,
                    attempt = ?,
                    started_at = ?,
                    finished_at = ?,
                    result_json = ?,
                    error_message = ?,
                    screenshot_path = ?
                WHERE id = ?
                """,
                (
                    job_id,
                    run_key,
                    status,
                    attempt,
                    started_at,
                    run.finished_at,
                    result_json,
                    run.error_message.strip(),
                    run.screenshot_path.strip(),
                    run.id,
                ),
            )
            return run.id

    def get_by_key(self, run_key: str) -> JobRun | None:
        row = self.db.fetch_one(
            "SELECT * FROM job_runs WHERE run_key = ?",
            (require_text(run_key, "run_key"),),
        )
        return self._from_row(row) if row else None

    def list_for_job(self, job_id: int) -> list[JobRun]:
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM job_runs
            WHERE job_id = ?
            ORDER BY started_at DESC, id DESC
            """,
            (job_id,),
        )
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: Any) -> JobRun:
        return JobRun(
            id=row["id"],
            job_id=row["job_id"],
            run_key=row["run_key"],
            status=row["status"],
            attempt=row["attempt"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            result_json=row["result_json"],
            error_message=row["error_message"],
            screenshot_path=row["screenshot_path"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class StepRunRepository:
    def __init__(self, db: Database):
        self.db = db

    def get(self, run_id: int) -> StepRun | None:
        row = self.db.fetch_one(
            "SELECT * FROM step_runs WHERE id = ?",
            (run_id,),
        )
        return self._from_row(row) if row else None

    def save(self, run: StepRun) -> int:
        job_run_id = require_id(run.job_run_id, "job_run_id")
        step_key = require_text(run.step_key, "step_key")
        status = validate_choice(run.status, RUN_STATUSES, "status")
        started_at = run.started_at or utc_now_iso()
        result_json = json_object_text(run.result_json, "result_json")
        attempt = validate_positive(run.attempt, "attempt")
        with self.db.transaction():
            if run.id is None:
                self.db.execute(
                    """
                    INSERT INTO step_runs(
                        job_run_id, workflow_step_id, step_key, status, attempt,
                        started_at, finished_at, result_json, error_message, screenshot_path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_run_id, step_key, attempt) DO UPDATE SET
                        workflow_step_id = excluded.workflow_step_id,
                        status = excluded.status,
                        started_at = excluded.started_at,
                        finished_at = excluded.finished_at,
                        result_json = excluded.result_json,
                        error_message = excluded.error_message,
                        screenshot_path = excluded.screenshot_path
                    """,
                    (
                        job_run_id,
                        run.workflow_step_id,
                        step_key,
                        status,
                        attempt,
                        started_at,
                        run.finished_at,
                        result_json,
                        run.error_message.strip(),
                        run.screenshot_path.strip(),
                    ),
                )
                return row_id(self.get_by_key(job_run_id, step_key, attempt))

            self.db.execute(
                """
                UPDATE step_runs
                SET job_run_id = ?,
                    workflow_step_id = ?,
                    step_key = ?,
                    status = ?,
                    attempt = ?,
                    started_at = ?,
                    finished_at = ?,
                    result_json = ?,
                    error_message = ?,
                    screenshot_path = ?
                WHERE id = ?
                """,
                (
                    job_run_id,
                    run.workflow_step_id,
                    step_key,
                    status,
                    attempt,
                    started_at,
                    run.finished_at,
                    result_json,
                    run.error_message.strip(),
                    run.screenshot_path.strip(),
                    run.id,
                ),
            )
            return run.id

    def get_by_key(self, job_run_id: int, step_key: str, attempt: int) -> StepRun | None:
        row = self.db.fetch_one(
            """
            SELECT *
            FROM step_runs
            WHERE job_run_id = ? AND step_key = ? AND attempt = ?
            """,
            (job_run_id, require_text(step_key, "step_key"), attempt),
        )
        return self._from_row(row) if row else None

    def list_for_job_run(self, job_run_id: int) -> list[StepRun]:
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM step_runs
            WHERE job_run_id = ?
            ORDER BY started_at, id
            """,
            (job_run_id,),
        )
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: Any) -> StepRun:
        return StepRun(
            id=row["id"],
            job_run_id=row["job_run_id"],
            workflow_step_id=row["workflow_step_id"],
            step_key=row["step_key"],
            status=row["status"],
            attempt=row["attempt"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            result_json=row["result_json"],
            error_message=row["error_message"],
            screenshot_path=row["screenshot_path"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class TemplatePackRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, pack: TemplatePack) -> int:
        pack_key = require_text(pack.pack_key, "pack_key")
        version = require_text(pack.version, "version")
        name = pack.name.strip() or pack_key
        metadata_json = json_object_text(pack.metadata_json, "metadata_json")
        with self.db.transaction():
            if pack.id is None:
                self.db.execute(
                    """
                    INSERT INTO template_packs(
                        pack_key, name, version, source_path, enabled, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(pack_key, version) DO UPDATE SET
                        name = excluded.name,
                        source_path = excluded.source_path,
                        enabled = excluded.enabled,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        pack_key,
                        name,
                        version,
                        pack.source_path.strip(),
                        int(pack.enabled),
                        metadata_json,
                    ),
                )
                return row_id(self.get_by_key(pack_key, version))

            self.db.execute(
                """
                UPDATE template_packs
                SET pack_key = ?,
                    name = ?,
                    version = ?,
                    source_path = ?,
                    enabled = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    pack_key,
                    name,
                    version,
                    pack.source_path.strip(),
                    int(pack.enabled),
                    metadata_json,
                    pack.id,
                ),
            )
            return pack.id

    def get_by_key(self, pack_key: str, version: str = "1") -> TemplatePack | None:
        row = self.db.fetch_one(
            """
            SELECT *
            FROM template_packs
            WHERE pack_key = ? AND version = ?
            """,
            (require_text(pack_key, "pack_key"), require_text(version, "version")),
        )
        return self._from_row(row) if row else None

    def list_all(self) -> list[TemplatePack]:
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM template_packs
            ORDER BY pack_key COLLATE NOCASE, version
            """
        )
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: Any) -> TemplatePack:
        return TemplatePack(
            id=row["id"],
            pack_key=row["pack_key"],
            name=row["name"],
            version=row["version"],
            source_path=row["source_path"],
            enabled=row_bool(row["enabled"]),
            metadata_json=row["metadata_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class TemplateRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, template: Template) -> int:
        pack_id = require_id(template.pack_id, "pack_id")
        template_key = require_text(template.template_key, "template_key")
        name = template.name.strip() or template_key
        file_path = require_text(template.file_path, "file_path")
        if template.threshold < 0.0 or template.threshold > 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0.")
        metadata_json = json_object_text(template.metadata_json, "metadata_json")
        with self.db.transaction():
            if template.id is None:
                self.db.execute(
                    """
                    INSERT INTO templates(
                        pack_id, template_key, name, file_path, image_hash,
                        threshold, enabled, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(pack_id, template_key) DO UPDATE SET
                        name = excluded.name,
                        file_path = excluded.file_path,
                        image_hash = excluded.image_hash,
                        threshold = excluded.threshold,
                        enabled = excluded.enabled,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        pack_id,
                        template_key,
                        name,
                        file_path,
                        template.image_hash.strip(),
                        template.threshold,
                        int(template.enabled),
                        metadata_json,
                    ),
                )
                return row_id(self.get_by_key(pack_id, template_key))

            self.db.execute(
                """
                UPDATE templates
                SET pack_id = ?,
                    template_key = ?,
                    name = ?,
                    file_path = ?,
                    image_hash = ?,
                    threshold = ?,
                    enabled = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    pack_id,
                    template_key,
                    name,
                    file_path,
                    template.image_hash.strip(),
                    template.threshold,
                    int(template.enabled),
                    metadata_json,
                    template.id,
                ),
            )
            return template.id

    def get_by_key(self, pack_id: int, template_key: str) -> Template | None:
        row = self.db.fetch_one(
            """
            SELECT *
            FROM templates
            WHERE pack_id = ? AND template_key = ?
            """,
            (pack_id, require_text(template_key, "template_key")),
        )
        return self._from_row(row) if row else None

    def list_for_pack(self, pack_id: int) -> list[Template]:
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM templates
            WHERE pack_id = ?
            ORDER BY template_key COLLATE NOCASE
            """,
            (pack_id,),
        )
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: Any) -> Template:
        return Template(
            id=row["id"],
            pack_id=row["pack_id"],
            template_key=row["template_key"],
            name=row["name"],
            file_path=row["file_path"],
            image_hash=row["image_hash"],
            threshold=float(row["threshold"]),
            enabled=row_bool(row["enabled"]),
            metadata_json=row["metadata_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class ScreenObservationRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, observation: ScreenObservation) -> int:
        observation_key = require_text(observation.observation_key, "observation_key")
        observed_at = observation.observed_at or utc_now_iso()
        metadata_json = json_object_text(observation.metadata_json, "metadata_json")
        with self.db.transaction():
            if observation.id is None:
                self.db.execute(
                    """
                    INSERT INTO screen_observations(
                        observation_key, instance_id, character_id, job_run_id,
                        observed_at, scene_name, screenshot_path, ocr_text, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(observation_key) DO UPDATE SET
                        instance_id = excluded.instance_id,
                        character_id = excluded.character_id,
                        job_run_id = excluded.job_run_id,
                        observed_at = excluded.observed_at,
                        scene_name = excluded.scene_name,
                        screenshot_path = excluded.screenshot_path,
                        ocr_text = excluded.ocr_text,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        observation_key,
                        observation.instance_id,
                        observation.character_id,
                        observation.job_run_id,
                        observed_at,
                        observation.scene_name.strip(),
                        observation.screenshot_path.strip(),
                        observation.ocr_text,
                        metadata_json,
                    ),
                )
                return row_id(self.get_by_key(observation_key))

            self.db.execute(
                """
                UPDATE screen_observations
                SET observation_key = ?,
                    instance_id = ?,
                    character_id = ?,
                    job_run_id = ?,
                    observed_at = ?,
                    scene_name = ?,
                    screenshot_path = ?,
                    ocr_text = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    observation_key,
                    observation.instance_id,
                    observation.character_id,
                    observation.job_run_id,
                    observed_at,
                    observation.scene_name.strip(),
                    observation.screenshot_path.strip(),
                    observation.ocr_text,
                    metadata_json,
                    observation.id,
                ),
            )
            return observation.id

    def get_by_key(self, observation_key: str) -> ScreenObservation | None:
        row = self.db.fetch_one(
            "SELECT * FROM screen_observations WHERE observation_key = ?",
            (require_text(observation_key, "observation_key"),),
        )
        return self._from_row(row) if row else None

    def list_recent(self, limit: int = 200) -> list[ScreenObservation]:
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM screen_observations
            ORDER BY observed_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: Any) -> ScreenObservation:
        return ScreenObservation(
            id=row["id"],
            observation_key=row["observation_key"],
            instance_id=row["instance_id"],
            character_id=row["character_id"],
            job_run_id=row["job_run_id"],
            observed_at=row["observed_at"],
            scene_name=row["scene_name"],
            screenshot_path=row["screenshot_path"],
            ocr_text=row["ocr_text"],
            metadata_json=row["metadata_json"],
            created_at=row["created_at"],
        )


class IncidentRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, incident: Incident) -> int:
        incident_key = require_text(incident.incident_key, "incident_key")
        severity = validate_choice(incident.severity, INCIDENT_SEVERITIES, "severity")
        status = validate_choice(incident.status, INCIDENT_STATUSES, "status")
        title = require_text(incident.title, "title")
        with self.db.transaction():
            if incident.id is None:
                self.db.execute(
                    """
                    INSERT INTO incidents(
                        incident_key, severity, status, title, details, job_run_id,
                        step_run_id, screenshot_path, resolved_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(incident_key) DO UPDATE SET
                        severity = excluded.severity,
                        status = excluded.status,
                        title = excluded.title,
                        details = excluded.details,
                        job_run_id = excluded.job_run_id,
                        step_run_id = excluded.step_run_id,
                        screenshot_path = excluded.screenshot_path,
                        resolved_at = excluded.resolved_at
                    """,
                    (
                        incident_key,
                        severity,
                        status,
                        title,
                        incident.details,
                        incident.job_run_id,
                        incident.step_run_id,
                        incident.screenshot_path.strip(),
                        incident.resolved_at,
                    ),
                )
                return row_id(self.get_by_key(incident_key))

            self.db.execute(
                """
                UPDATE incidents
                SET incident_key = ?,
                    severity = ?,
                    status = ?,
                    title = ?,
                    details = ?,
                    job_run_id = ?,
                    step_run_id = ?,
                    screenshot_path = ?,
                    resolved_at = ?
                WHERE id = ?
                """,
                (
                    incident_key,
                    severity,
                    status,
                    title,
                    incident.details,
                    incident.job_run_id,
                    incident.step_run_id,
                    incident.screenshot_path.strip(),
                    incident.resolved_at,
                    incident.id,
                ),
            )
            return incident.id

    def get_by_key(self, incident_key: str) -> Incident | None:
        row = self.db.fetch_one(
            "SELECT * FROM incidents WHERE incident_key = ?",
            (require_text(incident_key, "incident_key"),),
        )
        return self._from_row(row) if row else None

    def list_open(self) -> list[Incident]:
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM incidents
            WHERE status IN ('open', 'acknowledged')
            ORDER BY created_at DESC, id DESC
            """
        )
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: Any) -> Incident:
        return Incident(
            id=row["id"],
            incident_key=row["incident_key"],
            severity=row["severity"],
            status=row["status"],
            title=row["title"],
            details=row["details"],
            job_run_id=row["job_run_id"],
            step_run_id=row["step_run_id"],
            screenshot_path=row["screenshot_path"],
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
            updated_at=row["updated_at"],
        )


class AuditLogRepository:
    def __init__(self, db: Database):
        self.db = db

    def append(self, log: AuditLog) -> int:
        audit_key = require_text(log.audit_key, "audit_key")
        actor = log.actor.strip() or "system"
        action = require_text(log.action, "action")
        entity_type = require_text(log.entity_type, "entity_type")
        occurred_at = log.occurred_at or utc_now_iso()
        details_json = json_object_text(log.details_json, "details_json")
        with self.db.transaction():
            self.db.execute(
                """
                INSERT INTO audit_logs(
                    audit_key, actor, action, entity_type, entity_id, occurred_at, details_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(audit_key) DO NOTHING
                """,
                (
                    audit_key,
                    actor,
                    action,
                    entity_type,
                    log.entity_id,
                    occurred_at,
                    details_json,
                ),
            )
            return row_id(self.get_by_key(audit_key))

    def get_by_key(self, audit_key: str) -> AuditLog | None:
        row = self.db.fetch_one(
            "SELECT * FROM audit_logs WHERE audit_key = ?",
            (require_text(audit_key, "audit_key"),),
        )
        return self._from_row(row) if row else None

    def list_recent(self, limit: int = 200) -> list[AuditLog]:
        rows = self.db.fetch_all(
            """
            SELECT *
            FROM audit_logs
            ORDER BY occurred_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: Any) -> AuditLog:
        return AuditLog(
            id=row["id"],
            audit_key=row["audit_key"],
            actor=row["actor"],
            action=row["action"],
            entity_type=row["entity_type"],
            entity_id=row["entity_id"],
            occurred_at=row["occurred_at"],
            details_json=row["details_json"],
            created_at=row["created_at"],
        )
