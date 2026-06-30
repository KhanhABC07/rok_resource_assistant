LATEST_SCHEMA_VERSION = 2

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    instance_index INTEGER,
    instance_name TEXT NOT NULL DEFAULT '',
    adb_serial TEXT NOT NULL DEFAULT '',
    adb_connected INTEGER NOT NULL DEFAULT 0,
    launch_path TEXT NOT NULL DEFAULT '',
    launch_command TEXT NOT NULL DEFAULT '',
    close_command TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS game_accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_name TEXT NOT NULL COLLATE NOCASE
        CHECK(length(trim(account_name)) > 0),
    display_name TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    external_id TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_name)
);

CREATE TABLE IF NOT EXISTS characters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    instance_id INTEGER NOT NULL,
    account_name TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    alliance_help_enabled INTEGER NOT NULL DEFAULT 1,
    alliance_donate_enabled INTEGER NOT NULL DEFAULT 1,
    gift_collection_enabled INTEGER NOT NULL DEFAULT 1,
    last_switch_at TEXT,
    game_account_id INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(instance_id) REFERENCES instances(id) ON DELETE CASCADE,
    FOREIGN KEY(game_account_id) REFERENCES game_accounts(id) ON DELETE SET NULL,
    UNIQUE(instance_id, name)
);

CREATE TABLE IF NOT EXISTS marches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_id INTEGER NOT NULL,
    march_slot INTEGER NOT NULL CHECK(march_slot BETWEEN 1 AND 5),
    resource_type TEXT NOT NULL DEFAULT 'Gold'
        CHECK(resource_type IN ('Gold', 'Stone', 'Wood', 'Food')),
    resource_source TEXT NOT NULL DEFAULT 'Disabled'
        CHECK(resource_source IN ('Alliance Resource Pit', 'Wild Resource Node', 'Disabled')),
    status TEXT NOT NULL DEFAULT 'idle',
    next_action_time TEXT,
    expected_return_time TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE CASCADE,
    UNIQUE(character_id, march_slot)
);

CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_id INTEGER NOT NULL,
    march_slot INTEGER,
    task_type TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending', 'queued', 'running', 'completed', 'failed', 'aborted', 'retrying')),
    scheduled_for TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    error_message TEXT NOT NULL DEFAULT '',
    result TEXT NOT NULL DEFAULT ''
        CHECK(result IN ('', 'SUCCESS', 'FAILED', 'ABORTED')),
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS automation_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    template_readiness_required INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS automation_task_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    step_order INTEGER NOT NULL,
    action_type TEXT NOT NULL
        CHECK(action_type IN (
            'WaitTemplate',
            'ClickTemplate',
            'ClickCoordinates',
            'SwipeCoordinates',
            'Delay',
            'AbortTask',
            'RepeatStart',
            'RepeatEnd',
            'IfTemplateExists',
            'Else',
            'EndIf'
        )),
    parameters TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(task_id) REFERENCES automation_tasks(id) ON DELETE CASCADE,
    UNIQUE(task_id, step_order)
);

CREATE TABLE IF NOT EXISTS task_run_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER,
    task_name TEXT NOT NULL,
    instance_index INTEGER,
    instance_name TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    result TEXT NOT NULL CHECK(result IN ('SUCCESS', 'FAILED', 'ABORTED')),
    error_message TEXT NOT NULL DEFAULT '',
    abort_reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS instance_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id INTEGER NOT NULL,
    session_key TEXT NOT NULL COLLATE NOCASE
        CHECK(length(trim(session_key)) > 0),
    status TEXT NOT NULL DEFAULT 'created'
        CHECK(status IN ('created', 'starting', 'running', 'stopping', 'stopped', 'failed')),
    started_at TEXT,
    ended_at TEXT,
    emulator_pid INTEGER,
    adb_serial TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(instance_id) REFERENCES instances(id) ON DELETE CASCADE,
    UNIQUE(session_key)
);

CREATE TABLE IF NOT EXISTS automation_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL COLLATE NOCASE
        CHECK(length(trim(name)) > 0),
    description TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS feature_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL,
    feature_key TEXT NOT NULL COLLATE NOCASE
        CHECK(length(trim(feature_key)) > 0),
    enabled INTEGER NOT NULL DEFAULT 1,
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(profile_id) REFERENCES automation_profiles(id) ON DELETE CASCADE,
    UNIQUE(profile_id, feature_key)
);

CREATE TABLE IF NOT EXISTS schedule_definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL,
    schedule_key TEXT NOT NULL COLLATE NOCASE
        CHECK(length(trim(schedule_key)) > 0),
    name TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    cron_expression TEXT NOT NULL DEFAULT '',
    interval_seconds INTEGER,
    timezone TEXT NOT NULL DEFAULT 'UTC',
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(profile_id) REFERENCES automation_profiles(id) ON DELETE CASCADE,
    UNIQUE(profile_id, schedule_key)
);

CREATE TABLE IF NOT EXISTS workflow_definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL,
    workflow_key TEXT NOT NULL COLLATE NOCASE
        CHECK(length(trim(workflow_key)) > 0),
    name TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1 CHECK(version > 0),
    enabled INTEGER NOT NULL DEFAULT 1,
    trigger_type TEXT NOT NULL DEFAULT 'manual',
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(profile_id) REFERENCES automation_profiles(id) ON DELETE CASCADE,
    UNIQUE(profile_id, workflow_key, version)
);

CREATE TABLE IF NOT EXISTS workflow_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id INTEGER NOT NULL,
    step_order INTEGER NOT NULL CHECK(step_order > 0),
    step_key TEXT NOT NULL COLLATE NOCASE
        CHECK(length(trim(step_key)) > 0),
    action_type TEXT NOT NULL,
    parameters_json TEXT NOT NULL DEFAULT '{}',
    timeout_seconds INTEGER,
    retry_limit INTEGER NOT NULL DEFAULT 0 CHECK(retry_limit >= 0),
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(workflow_id) REFERENCES workflow_definitions(id) ON DELETE CASCADE,
    UNIQUE(workflow_id, step_order),
    UNIQUE(workflow_id, step_key)
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id INTEGER,
    schedule_id INTEGER,
    character_id INTEGER,
    idempotency_key TEXT NOT NULL COLLATE NOCASE
        CHECK(length(trim(idempotency_key)) > 0),
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending', 'queued', 'running', 'completed', 'failed', 'aborted', 'cancelled')),
    priority INTEGER NOT NULL DEFAULT 100,
    scheduled_for TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(workflow_id) REFERENCES workflow_definitions(id) ON DELETE SET NULL,
    FOREIGN KEY(schedule_id) REFERENCES schedule_definitions(id) ON DELETE SET NULL,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE SET NULL,
    UNIQUE(idempotency_key)
);

CREATE TABLE IF NOT EXISTS job_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    run_key TEXT NOT NULL COLLATE NOCASE
        CHECK(length(trim(run_key)) > 0),
    status TEXT NOT NULL DEFAULT 'running'
        CHECK(status IN ('running', 'completed', 'failed', 'aborted', 'cancelled')),
    attempt INTEGER NOT NULL DEFAULT 1 CHECK(attempt > 0),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    result_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT NOT NULL DEFAULT '',
    screenshot_path TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    UNIQUE(run_key)
);

CREATE TABLE IF NOT EXISTS step_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_run_id INTEGER NOT NULL,
    workflow_step_id INTEGER,
    step_key TEXT NOT NULL COLLATE NOCASE
        CHECK(length(trim(step_key)) > 0),
    status TEXT NOT NULL DEFAULT 'running'
        CHECK(status IN ('running', 'completed', 'failed', 'aborted', 'cancelled')),
    attempt INTEGER NOT NULL DEFAULT 1 CHECK(attempt > 0),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    result_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT NOT NULL DEFAULT '',
    screenshot_path TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(job_run_id) REFERENCES job_runs(id) ON DELETE CASCADE,
    FOREIGN KEY(workflow_step_id) REFERENCES workflow_steps(id) ON DELETE SET NULL,
    UNIQUE(job_run_id, step_key, attempt)
);

CREATE TABLE IF NOT EXISTS template_packs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pack_key TEXT NOT NULL COLLATE NOCASE
        CHECK(length(trim(pack_key)) > 0),
    name TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '1',
    source_path TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pack_key, version)
);

CREATE TABLE IF NOT EXISTS templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pack_id INTEGER NOT NULL,
    template_key TEXT NOT NULL COLLATE NOCASE
        CHECK(length(trim(template_key)) > 0),
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    image_hash TEXT NOT NULL DEFAULT '',
    threshold REAL NOT NULL DEFAULT 0.8 CHECK(threshold >= 0.0 AND threshold <= 1.0),
    enabled INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(pack_id) REFERENCES template_packs(id) ON DELETE CASCADE,
    UNIQUE(pack_id, template_key)
);

CREATE TABLE IF NOT EXISTS screen_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_key TEXT NOT NULL COLLATE NOCASE
        CHECK(length(trim(observation_key)) > 0),
    instance_id INTEGER,
    character_id INTEGER,
    job_run_id INTEGER,
    observed_at TEXT NOT NULL,
    scene_name TEXT NOT NULL DEFAULT '',
    screenshot_path TEXT NOT NULL DEFAULT '',
    ocr_text TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(instance_id) REFERENCES instances(id) ON DELETE SET NULL,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE SET NULL,
    FOREIGN KEY(job_run_id) REFERENCES job_runs(id) ON DELETE SET NULL,
    UNIQUE(observation_key)
);

CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_key TEXT NOT NULL COLLATE NOCASE
        CHECK(length(trim(incident_key)) > 0),
    severity TEXT NOT NULL DEFAULT 'error'
        CHECK(severity IN ('info', 'warning', 'error', 'critical')),
    status TEXT NOT NULL DEFAULT 'open'
        CHECK(status IN ('open', 'acknowledged', 'resolved')),
    title TEXT NOT NULL,
    details TEXT NOT NULL DEFAULT '',
    job_run_id INTEGER,
    step_run_id INTEGER,
    screenshot_path TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(job_run_id) REFERENCES job_runs(id) ON DELETE SET NULL,
    FOREIGN KEY(step_run_id) REFERENCES step_runs(id) ON DELETE SET NULL,
    UNIQUE(incident_key)
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_key TEXT NOT NULL COLLATE NOCASE
        CHECK(length(trim(audit_key)) > 0),
    actor TEXT NOT NULL DEFAULT 'system',
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id INTEGER,
    occurred_at TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(audit_key)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_instances_instance_index
    ON instances(instance_index)
    WHERE instance_index IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_game_accounts_external
    ON game_accounts(provider, external_id)
    WHERE external_id <> '';

CREATE INDEX IF NOT EXISTS idx_characters_game_account
    ON characters(game_account_id);

CREATE INDEX IF NOT EXISTS idx_tasks_due
    ON scheduled_tasks(status, scheduled_for, priority);

CREATE INDEX IF NOT EXISTS idx_automation_task_steps_task
    ON automation_task_steps(task_id, step_order);

CREATE INDEX IF NOT EXISTS idx_task_run_history_task
    ON task_run_history(task_id, started_at);

CREATE INDEX IF NOT EXISTS idx_marches_character
    ON marches(character_id, march_slot);

CREATE INDEX IF NOT EXISTS idx_instance_sessions_instance
    ON instance_sessions(instance_id, status, started_at);

CREATE INDEX IF NOT EXISTS idx_feature_configs_feature
    ON feature_configs(feature_key, enabled);

CREATE INDEX IF NOT EXISTS idx_schedule_definitions_profile
    ON schedule_definitions(profile_id, enabled);

CREATE INDEX IF NOT EXISTS idx_workflow_definitions_profile
    ON workflow_definitions(profile_id, workflow_key, version);

CREATE INDEX IF NOT EXISTS idx_workflow_steps_workflow
    ON workflow_steps(workflow_id, step_order);

CREATE INDEX IF NOT EXISTS idx_jobs_status_due
    ON jobs(status, scheduled_for, priority);

CREATE INDEX IF NOT EXISTS idx_job_runs_job
    ON job_runs(job_id, started_at);

CREATE INDEX IF NOT EXISTS idx_step_runs_job_run
    ON step_runs(job_run_id, workflow_step_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_templates_image_hash
    ON templates(image_hash)
    WHERE image_hash <> '';

CREATE INDEX IF NOT EXISTS idx_screen_observations_time
    ON screen_observations(observed_at, scene_name);

CREATE INDEX IF NOT EXISTS idx_incidents_status
    ON incidents(status, severity, created_at);

CREATE INDEX IF NOT EXISTS idx_audit_logs_entity
    ON audit_logs(entity_type, entity_id, occurred_at);

CREATE TRIGGER IF NOT EXISTS trg_instances_updated
AFTER UPDATE ON instances
BEGIN
    UPDATE instances SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_game_accounts_updated
AFTER UPDATE ON game_accounts
BEGIN
    UPDATE game_accounts SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_characters_updated
AFTER UPDATE ON characters
BEGIN
    UPDATE characters SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_marches_updated
AFTER UPDATE ON marches
BEGIN
    UPDATE marches SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_tasks_updated
AFTER UPDATE ON scheduled_tasks
BEGIN
    UPDATE scheduled_tasks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_automation_tasks_updated
AFTER UPDATE ON automation_tasks
BEGIN
    UPDATE automation_tasks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_automation_task_steps_updated
AFTER UPDATE ON automation_task_steps
BEGIN
    UPDATE automation_task_steps SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_instance_sessions_updated
AFTER UPDATE ON instance_sessions
BEGIN
    UPDATE instance_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_automation_profiles_updated
AFTER UPDATE ON automation_profiles
BEGIN
    UPDATE automation_profiles SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_feature_configs_updated
AFTER UPDATE ON feature_configs
BEGIN
    UPDATE feature_configs SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_schedule_definitions_updated
AFTER UPDATE ON schedule_definitions
BEGIN
    UPDATE schedule_definitions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_workflow_definitions_updated
AFTER UPDATE ON workflow_definitions
BEGIN
    UPDATE workflow_definitions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_workflow_steps_updated
AFTER UPDATE ON workflow_steps
BEGIN
    UPDATE workflow_steps SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_jobs_updated
AFTER UPDATE ON jobs
BEGIN
    UPDATE jobs SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_job_runs_updated
AFTER UPDATE ON job_runs
BEGIN
    UPDATE job_runs SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_step_runs_updated
AFTER UPDATE ON step_runs
BEGIN
    UPDATE step_runs SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_template_packs_updated
AFTER UPDATE ON template_packs
BEGIN
    UPDATE template_packs SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_templates_updated
AFTER UPDATE ON templates
BEGIN
    UPDATE templates SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_incidents_updated
AFTER UPDATE ON incidents
BEGIN
    UPDATE incidents SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
"""
