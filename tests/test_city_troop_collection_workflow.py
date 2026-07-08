from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.db.database import Database  # noqa: E402
from rok_assistant.db.models import Character, Instance, Job  # noqa: E402
from rok_assistant.db.repositories import (  # noqa: E402
    CharacterRepository,
    IncidentRepository,
    InstanceRepository,
    JobRepository,
    JobRunRepository,
    StepRunRepository,
)
from rok_assistant.tasks.city_troop_collection_workflow import (  # noqa: E402
    CITY_TROOP_COLLECTION_STATES,
    CityTroopCollectionConfig,
    CityTroopCollectionPolicy,
    CityTroopCollectionRequest,
    CityTroopCollectionResult,
    CityTroopCollectionWorkflow,
    CityTroopLayoutProfile,
    CityTroopObservation,
    CityTroopScan,
    CityTroopScanStatus,
    TroopBuilding,
    TroopBuildingRoi,
    TroopIndicatorType,
)
from rok_assistant.tasks.resource_search_workflow import ResourceGatheringActionResult  # noqa: E402
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


def _profile() -> CityTroopLayoutProfile:
    return CityTroopLayoutProfile(
        profile_id="standard",
        screen_width=500,
        screen_height=300,
        rois=(
            TroopBuildingRoi(TroopBuilding.BARRACKS, 10, 10, 80, 80),
            TroopBuildingRoi(TroopBuilding.STABLE, 110, 10, 80, 80),
            TroopBuildingRoi(TroopBuilding.ARCHERY_RANGE, 210, 10, 80, 80),
            TroopBuildingRoi(TroopBuilding.SIEGE_WORKSHOP, 310, 10, 80, 80),
        ),
    )


def _ready(
    building: TroopBuilding,
    x: int,
    *,
    indicator_id: str,
    indicator_type: TroopIndicatorType = TroopIndicatorType.COMPLETED_TRAINING,
    confidence: float = 0.93,
) -> CityTroopObservation:
    return CityTroopObservation(
        building,
        indicator_type,
        confidence,
        x=x,
        y=30,
        indicator_id=indicator_id,
    )


class FakeTroopDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.scans: list[CityTroopScan] = [
            CityTroopScan(
                CityTroopScanStatus.READY,
                observations=(
                    _ready(TroopBuilding.BARRACKS, 20, indicator_id="barracks-1"),
                ),
            ),
            CityTroopScan(CityTroopScanStatus.NONE_READY),
        ]
        self.normalize = ResourceGatheringActionResult(True, data={"scene": "CITY_HOME"})
        self.click = CityTroopCollectionResult(True, data={"clicked": True})
        self.verify = CityTroopCollectionResult(True, changed=True, data={"indicator_cleared": True})
        self.panel = CityTroopCollectionResult(True, changed=True, data={"panel_closed": True})

    def normalize_to_city_home(self, _request, _character, policy):
        self.calls.append(f"normalize:{policy.layout_profile.profile_id}")
        return self.normalize

    def scan_completed_troops(self, _request, _character, _policy, pass_number):
        self.calls.append(f"scan:{pass_number}")
        if self.scans:
            return self.scans.pop(0)
        return CityTroopScan(CityTroopScanStatus.NONE_READY)

    def click_completed_troop_indicator(self, _request, _character, observation, _policy):
        self.calls.append(f"click:{observation.normalized_building().value}:{observation.indicator_id}")
        return self.click

    def verify_completed_troop_collected(self, _request, _character, observation, _policy):
        self.calls.append(f"verify:{observation.normalized_building().value}:{observation.indicator_id}")
        return self.verify

    def handle_troop_collection_result_panel(self, _request, _character, _policy):
        self.calls.append("handle_panel")
        return self.panel


class FakeWatchdog:
    def __init__(self, *, healthy: bool = True) -> None:
        self.healthy = healthy
        self.calls: list[int | None] = []

    def monitor(
        self,
        *,
        instance_id: int,
        instance_index: int,
        instance_name: str,
        job_run_id: int | None = None,
    ) -> object:
        del instance_id, instance_index, instance_name
        self.calls.append(job_run_id)
        return SimpleNamespace(
            healthy=self.healthy,
            recovery_attempted=not self.healthy,
            circuit_opened=not self.healthy,
            observation=SimpleNamespace(
                message="unhealthy" if not self.healthy else "",
                screenshot_path="runtime/screens/unhealthy.png" if not self.healthy else "",
            ),
        )


class CityTroopCollectionWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "city-troops.sqlite3")
        self.db.initialize()
        self.instances = InstanceRepository(self.db)
        self.characters = CharacterRepository(self.db)
        self.jobs = JobRepository(self.db)
        self.job_runs = JobRunRepository(self.db)
        self.step_runs = StepRunRepository(self.db)
        self.incidents = IncidentRepository(self.db)
        self.instance_id = self.instances.save(
            Instance(name="MEmu 1", instance_index=0, instance_name="MEmu 1")
        )
        self.character_id = self.characters.save(
            Character(id=None, name="Farm01", instance_id=self.instance_id)
        )
        self.driver = FakeTroopDriver()
        self.watchdog = FakeWatchdog()

    def tearDown(self) -> None:
        self.db.close()
        self.temp_dir.cleanup()

    def _job(self, key: str) -> int:
        return self.jobs.save(
            Job(
                idempotency_key=key,
                job_type="workflow",
                scheduled_for="2026-07-08T00:00:00",
            )
        )

    def _workflow(self, *, max_scan_passes: int = 4) -> CityTroopCollectionWorkflow:
        return CityTroopCollectionWorkflow(
            characters=self.characters,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=CityTroopCollectionConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
                max_scan_passes=max_scan_passes,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "troop-run",
        policy: CityTroopCollectionPolicy | None = None,
    ) -> CityTroopCollectionRequest:
        return CityTroopCollectionRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or CityTroopCollectionPolicy(layout_profile=_profile()),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states(self) -> None:
        self.assertEqual(CITY_TROOP_COLLECTION_STATES, self._workflow().workflow_states)

    def test_collects_one_completed_troop_building_and_persists_metadata(self) -> None:
        job_id = self._job("troop-one")

        result = self._workflow().execute(self._request(job_id=job_id, run_key="troop-one"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(1, result.result["collected_building_count"])
        self.assertEqual(["BARRACKS"], result.result["collected_buildings"])
        self.assertEqual(
            [
                "normalize:standard",
                "scan:1",
                "click:BARRACKS:barracks-1",
                "handle_panel",
                "verify:BARRACKS:barracks-1",
                "scan:2",
                "handle_panel",
            ],
            self.driver.calls,
        )
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual(1, payload["result"]["collected_building_count"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_collects_multiple_completed_troop_buildings(self) -> None:
        self.driver.scans = [
            CityTroopScan(
                CityTroopScanStatus.READY,
                observations=(
                    _ready(TroopBuilding.BARRACKS, 20, indicator_id="barracks-1"),
                    _ready(TroopBuilding.STABLE, 120, indicator_id="stable-1"),
                    _ready(TroopBuilding.ARCHERY_RANGE, 220, indicator_id="archery-1"),
                ),
            ),
            CityTroopScan(CityTroopScanStatus.NONE_READY),
        ]

        result = self._workflow().execute(self._request(run_key="troop-many"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(3, result.result["collected_building_count"])
        self.assertEqual(["ARCHERY_RANGE", "BARRACKS", "STABLE"], result.result["collected_buildings"])
        self.assertIn("click:STABLE:stable-1", self.driver.calls)
        self.assertIn("click:ARCHERY_RANGE:archery-1", self.driver.calls)

    def test_no_completed_troops_returns_skipped_without_clicking(self) -> None:
        self.driver.scans = [CityTroopScan(CityTroopScanStatus.NONE_READY)]

        result = self._workflow().execute(
            self._request(job_id=self._job("troop-none"), run_key="troop-none")
        )

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual(0, result.result["collected_building_count"])
        self.assertEqual("collect_completed_troops", result.result["terminal_state"])
        self.assertFalse(any(call.startswith("click:") for call in self.driver.calls))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("completed", run.status)  # type: ignore[union-attr]
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_result_popup_is_handled_before_scanning_continues(self) -> None:
        self.driver.scans = [
            CityTroopScan(CityTroopScanStatus.RESULT_PANEL, message="Completion panel visible."),
            CityTroopScan(
                CityTroopScanStatus.READY,
                observations=(
                    _ready(TroopBuilding.STABLE, 120, indicator_id="stable-1"),
                ),
            ),
            CityTroopScan(CityTroopScanStatus.NONE_READY),
        ]

        result = self._workflow().execute(self._request(run_key="troop-popup"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(1, result.result["collected_building_count"])
        self.assertEqual(
            [
                "normalize:standard",
                "scan:1",
                "handle_panel",
                "scan:2",
                "click:STABLE:stable-1",
                "handle_panel",
                "verify:STABLE:stable-1",
                "scan:3",
                "handle_panel",
            ],
            self.driver.calls,
        )

    def test_ambiguous_speedup_and_upgrade_icons_are_ignored(self) -> None:
        self.driver.scans = [
            CityTroopScan(
                CityTroopScanStatus.READY,
                observations=(
                    _ready(
                        TroopBuilding.BARRACKS,
                        20,
                        indicator_id="ambiguous",
                        indicator_type=TroopIndicatorType.AMBIGUOUS,
                    ),
                    _ready(
                        TroopBuilding.STABLE,
                        120,
                        indicator_id="speedup",
                        indicator_type=TroopIndicatorType.SPEED_UP,
                    ),
                    _ready(
                        TroopBuilding.ARCHERY_RANGE,
                        220,
                        indicator_id="upgrade",
                        indicator_type=TroopIndicatorType.UPGRADE,
                    ),
                ),
            ),
            CityTroopScan(CityTroopScanStatus.NONE_READY),
        ]

        result = self._workflow().execute(self._request(run_key="troop-ambiguous"))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual(0, result.result["collected_building_count"])
        self.assertEqual(3, result.result["scan_attempts"][0]["ignored_observation_count"])
        self.assertFalse(any(call.startswith("click:") for call in self.driver.calls))

    def test_loop_budget_exhaustion_blocks_with_failure_evidence(self) -> None:
        self.driver.scans = [
            CityTroopScan(
                CityTroopScanStatus.READY,
                observations=(
                    _ready(TroopBuilding.BARRACKS, 20, indicator_id="barracks-1"),
                ),
                screenshot_path="runtime/screens/pass-1.png",
            ),
            CityTroopScan(
                CityTroopScanStatus.READY,
                observations=(
                    _ready(TroopBuilding.BARRACKS, 20, indicator_id="barracks-2"),
                ),
                screenshot_path="runtime/screens/pass-2.png",
            ),
        ]

        result = self._workflow(max_scan_passes=2).execute(
            self._request(job_id=self._job("troop-budget"), run_key="troop-budget")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("collect_completed_troops", result.result["terminal_state"])
        self.assertEqual(2, result.result["collected_building_count"])
        self.assertEqual("runtime/screens/pass-2.png", result.result["failure_evidence"]["screenshot_path"])
        self.assertEqual(1, len(self.incidents.list_open()))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("failed", run.status)  # type: ignore[union-attr]
        self.assertIn("loop budget", run.error_message)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
