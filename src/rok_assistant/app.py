from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

from rok_assistant.characters import CharacterManager
from rok_assistant.config import AppConfig
from rok_assistant.db import (
    AutomationTaskRepository,
    CharacterRepository,
    Database,
    GameAccountRepository,
    IncidentRepository,
    InstanceCircuitBreakerRepository,
    InstanceRepository,
    MarchRepository,
    RecoveryAttemptRepository,
    SettingsRepository,
    TaskRunHistoryRepository,
    TaskRepository,
)
from rok_assistant.db.models import DashboardStats, ScheduledTask, utc_now_iso
from rok_assistant.emulator import (
    DEFAULT_MEMU_INSTALL_PATH,
    EmulatorManager,
    MEmuAdbManager,
    MEmuManager,
)
from rok_assistant.export_import import ConfigurationService
from rok_assistant.logging_setup import configure_logging, shutdown_logging
from rok_assistant.observability import DashboardMetricsService, SupportBundleExporter
from rok_assistant.paths import ensure_runtime_dirs, resolve_project_path
from rok_assistant.recovery import ErrorRecoveryPolicy
from rok_assistant.scheduler import Scheduler, WorkerPool
from rok_assistant.security import DpapiFileSecretStore, SecretStore
from rok_assistant.tasks import TaskContext, TaskManager
from rok_assistant.vision import VisionOcrModule
from rok_assistant.vision import EvidenceRetentionPolicy, FileSystemEvidenceStore


@dataclass
class AppContext:
    config: AppConfig
    db: Database
    accounts: GameAccountRepository
    instances: InstanceRepository
    characters: CharacterRepository
    marches: MarchRepository
    settings: SettingsRepository
    tasks: TaskRepository
    automation_tasks: AutomationTaskRepository
    task_run_history: TaskRunHistoryRepository
    incidents: IncidentRepository
    recovery_attempts: RecoveryAttemptRepository
    circuit_breakers: InstanceCircuitBreakerRepository
    memu_manager: MEmuManager
    memu_adb_manager: MEmuAdbManager
    emulator_manager: EmulatorManager
    character_manager: CharacterManager
    vision: VisionOcrModule
    task_manager: TaskManager
    worker_pool: WorkerPool
    scheduler: Scheduler
    configuration_service: ConfigurationService
    secret_store: SecretStore
    dashboard_metrics: DashboardMetricsService
    support_bundles: SupportBundleExporter
    closed: bool = False

    @classmethod
    def create(cls) -> "AppContext":
        ensure_runtime_dirs()
        config = AppConfig.load()
        configure_logging(config.log_file, config.get("logging.level", "INFO"))
        logger = logging.getLogger("rok_assistant")
        logger.info("Starting Rise of Kingdoms Resource Assistant.")

        db = Database(config.database_path)
        db.initialize()

        accounts = GameAccountRepository(db)
        instances = InstanceRepository(db)
        characters = CharacterRepository(db)
        marches = MarchRepository(db)
        settings = SettingsRepository(db)
        tasks = TaskRepository(db)
        automation_tasks = AutomationTaskRepository(db)
        task_run_history = TaskRunHistoryRepository(db)
        incidents = IncidentRepository(db)
        recovery_attempts = RecoveryAttemptRepository(db)
        circuit_breakers = InstanceCircuitBreakerRepository(db)
        settings.set_defaults(
            {
                "scheduler.max_workers": config.get("scheduler.max_workers", 5),
                "scheduler.max_active_instances": config.get(
                    "scheduler.max_active_instances", 5
                ),
                "scheduler.retry_delay_minutes": config.get(
                    "scheduler.retry_delay_minutes", 10
                ),
                "scheduler.pre_launch_minutes": config.get(
                    "scheduler.pre_launch_minutes", 2
                ),
                "scheduler.poll_interval_seconds": config.get(
                    "scheduler.poll_interval_seconds", 5
                ),
                "watchdog.game_package": config.get(
                    "watchdog.game_package", "com.lilithgame.roc.gp"
                ),
                "watchdog.game_activity": config.get(
                    "watchdog.game_activity",
                    "com.lilithgame.roc.gp/.UnityPlayerActivity",
                ),
                "watchdog.same_screen_timeout_seconds": config.get(
                    "watchdog.same_screen_timeout_seconds", 120
                ),
                "watchdog.same_screen_max_observations": config.get(
                    "watchdog.same_screen_max_observations", 3
                ),
                "observability.evidence_retention_days": config.get(
                    "observability.evidence_retention_days", 14
                ),
                "observability.evidence_retention_max_files": config.get(
                    "observability.evidence_retention_max_files", 500
                ),
                "observability.support_bundle_dir": config.get(
                    "observability.support_bundle_dir", "runtime/support_bundles"
                ),
                "emulator.memu_install_path": config.get(
                    "emulator.memu_install_path", DEFAULT_MEMU_INSTALL_PATH
                ),
                "gathering.preferred_resource_levels": config.get(
                    "gathering.preferred_resource_levels", [8, 7, 6]
                ),
                "gathering.minimum_resource_level": config.get(
                    "gathering.minimum_resource_level", 6
                ),
            }
        )

        memu_manager = MEmuManager(
            settings.get("emulator.memu_install_path", DEFAULT_MEMU_INSTALL_PATH)
        )
        memu_adb_manager = MEmuAdbManager(
            settings.get("emulator.memu_install_path", DEFAULT_MEMU_INSTALL_PATH)
        )
        emulator_manager = EmulatorManager(
            instances,
            memu_manager=memu_manager,
            max_concurrent_provider=lambda: settings.get_int(
                "scheduler.max_active_instances", 5
            ),
        )
        character_manager = CharacterManager(characters)
        vision = VisionOcrModule()
        secret_store = DpapiFileSecretStore()
        task_context = TaskContext(
            instances=instances,
            characters=characters,
            marches=marches,
            settings=settings,
            emulator_manager=emulator_manager,
            character_manager=character_manager,
            vision=vision,
            logger=logging.getLogger("TaskContext"),
        )
        recovery = ErrorRecoveryPolicy(
            retry_delay_minutes=settings.get_int("scheduler.retry_delay_minutes", 10)
        )
        task_manager = TaskManager(
            task_repository=tasks,
            context=task_context,
            recovery_policy=recovery,
            plugin_packages=config.plugin_packages,
        )
        worker_pool = WorkerPool(
            task_manager=task_manager,
            max_workers=settings.get_int("scheduler.max_workers", 5),
        )
        scheduler = Scheduler(
            task_repository=tasks,
            worker_pool=worker_pool,
            poll_interval_seconds=settings.get_int("scheduler.poll_interval_seconds", 5),
            instance_repository=instances,
            emulator_manager=emulator_manager,
            settings=settings,
            circuit_breakers=circuit_breakers,
        )
        FileSystemEvidenceStore(resolve_project_path("runtime/screenshots")).apply_retention(
            EvidenceRetentionPolicy(
                max_age_days=settings.get_int("observability.evidence_retention_days", 14),
                max_files=settings.get_int("observability.evidence_retention_max_files", 500),
            )
        )
        dashboard_metrics = DashboardMetricsService(db)
        return cls(
            config=config,
            db=db,
            accounts=accounts,
            instances=instances,
            characters=characters,
            marches=marches,
            settings=settings,
            tasks=tasks,
            automation_tasks=automation_tasks,
            task_run_history=task_run_history,
            incidents=incidents,
            recovery_attempts=recovery_attempts,
            circuit_breakers=circuit_breakers,
            memu_manager=memu_manager,
            memu_adb_manager=memu_adb_manager,
            emulator_manager=emulator_manager,
            character_manager=character_manager,
            vision=vision,
            task_manager=task_manager,
            worker_pool=worker_pool,
            scheduler=scheduler,
            configuration_service=ConfigurationService(db),
            secret_store=secret_store,
            dashboard_metrics=dashboard_metrics,
            support_bundles=SupportBundleExporter(
                db=db,
                config=config,
                output_dir=resolve_project_path(
                    settings.get("observability.support_bundle_dir", "runtime/support_bundles")
                ),
            ),
        )

    def dashboard_stats(self, *, force_refresh: bool = False) -> DashboardStats:
        return self.dashboard_metrics.collect(
            active_workers=self.scheduler.active_workers,
            running_instances=self.emulator_manager.running_count(),
            max_workers=self.worker_pool.max_workers,
            force_refresh=force_refresh,
        )

    def schedule_enabled_work(self) -> int:
        created = 0
        now = utc_now_iso()
        for character in self.characters.list_all(include_disabled=False):
            if character.id is None:
                continue

            if character.alliance_help_enabled and not self.tasks.open_task_exists(
                character.id, None, "alliance_help"
            ):
                self.tasks.enqueue(
                    ScheduledTask(
                        character_id=character.id,
                        task_type="alliance_help",
                        priority=40,
                        scheduled_for=now,
                    )
                )
                created += 1

            if character.alliance_donate_enabled and not self.tasks.open_task_exists(
                character.id, None, "alliance_donate"
            ):
                self.tasks.enqueue(
                    ScheduledTask(
                        character_id=character.id,
                        task_type="alliance_donate",
                        priority=50,
                        scheduled_for=now,
                    )
                )
                created += 1

            if character.gift_collection_enabled and not self.tasks.open_task_exists(
                character.id, None, "gift_collection"
            ):
                self.tasks.enqueue(
                    ScheduledTask(
                        character_id=character.id,
                        task_type="gift_collection",
                        priority=60,
                        scheduled_for=now,
                    )
                )
                created += 1

        self.scheduler.wake()
        dashboard_metrics = getattr(self, "dashboard_metrics", None)
        if dashboard_metrics is not None:
            dashboard_metrics.invalidate()
        logging.getLogger("rok_assistant").info("Created %s scheduled task(s).", created)
        return created

    def shutdown(self) -> None:
        if self.closed:
            return
        self.scheduler.stop()
        self.db.close()
        self.closed = True
        shutdown_logging()


def run_app() -> int:
    from PyQt6.QtWidgets import QApplication

    from rok_assistant.gui.main_window import MainWindow

    context = AppContext.create()
    app = QApplication(sys.argv)
    app.setApplicationName("Rise of Kingdoms Resource Assistant")
    window = MainWindow(context)
    window.resize(1280, 800)
    window.show()
    try:
        return app.exec()
    finally:
        context.shutdown()
