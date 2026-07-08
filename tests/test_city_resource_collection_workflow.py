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
from rok_assistant.tasks.city_resource_collection_workflow import (  # noqa: E402
    CITY_RESOURCE_COLLECTION_STATES,
    CityLayoutProfile,
    CityResourceCollectionConfig,
    CityResourceCollectionPolicy,
    CityResourceCollectionRequest,
    CityResourceCollectionResult,
    CityResourceCollectionWorkflow,
    CityResourceObservation,
    CityResourceRoi,
    CityResourceScan,
    CityResourceScanStatus,
    CityResourceType,
)
from rok_assistant.tasks.resource_search_workflow import ResourceGatheringActionResult  # noqa: E402
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


def _profile(*, include_crystal: bool = False, shifted: bool = False) -> CityLayoutProfile:
    offset = 100 if shifted else 0
    rois = [
        CityResourceRoi(CityResourceType.FOOD, 10 + offset, 10, 50, 50),
        CityResourceRoi(CityResourceType.WOOD, 70 + offset, 10, 50, 50),
        CityResourceRoi(CityResourceType.STONE, 130 + offset, 10, 50, 50),
        CityResourceRoi(CityResourceType.GOLD, 190 + offset, 10, 50, 50),
    ]
    if include_crystal:
        rois.append(CityResourceRoi(CityResourceType.CRYSTAL, 250 + offset, 10, 50, 50))
    return CityLayoutProfile(
        profile_id="shifted" if shifted else "standard",
        screen_width=400,
        screen_height=200,
        rois=tuple(rois),
    )


class FakeCityDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.scans: list[CityResourceScan] = [
            CityResourceScan(
                CityResourceScanStatus.READY,
                observations=(
                    CityResourceObservation(CityResourceType.FOOD, 0.92, x=20, y=20, indicator_id="food-1"),
                    CityResourceObservation(CityResourceType.WOOD, 0.91, x=80, y=20, indicator_id="wood-1"),
                ),
            ),
            CityResourceScan(CityResourceScanStatus.NONE_READY),
        ]
        self.normalize = ResourceGatheringActionResult(True, data={"scene": "CITY_HOME"})
        self.clear = ResourceGatheringActionResult(True, data={"overlay_cleared": True})
        self.click = CityResourceCollectionResult(True, data={"clicked": True})
        self.verify = CityResourceCollectionResult(True, changed=True, data={"bubble_changed": True})

    def normalize_to_city_home(self, _request, _character, policy):
        self.calls.append(f"normalize:{policy.layout_profile.profile_id}")
        return self.normalize

    def scan_city_resources(self, _request, _character, _policy, pass_number):
        self.calls.append(f"scan:{pass_number}")
        if self.scans:
            return self.scans.pop(0)
        return CityResourceScan(CityResourceScanStatus.NONE_READY)

    def clear_overlays(self, _request, _character, _policy):
        self.calls.append("clear_overlays")
        return self.clear

    def collect_city_resource(self, _request, _character, observation, _policy):
        self.calls.append(f"collect:{observation.normalized_resource_type().value}:{observation.indicator_id}")
        return self.click

    def verify_city_resource_collected(self, _request, _character, observation, _policy):
        self.calls.append(f"verify:{observation.normalized_resource_type().value}:{observation.indicator_id}")
        return self.verify


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


class CityResourceCollectionWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "city.sqlite3")
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
        self.driver = FakeCityDriver()
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

    def _workflow(self) -> CityResourceCollectionWorkflow:
        return CityResourceCollectionWorkflow(
            characters=self.characters,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=CityResourceCollectionConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
                max_passes=3,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "city-run",
        policy: CityResourceCollectionPolicy | None = None,
    ) -> CityResourceCollectionRequest:
        return CityResourceCollectionRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or CityResourceCollectionPolicy(layout_profile=_profile()),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states(self) -> None:
        self.assertEqual(CITY_RESOURCE_COLLECTION_STATES, self._workflow().workflow_states)

    def test_collects_ready_resources_once_per_pass_and_persists_metadata(self) -> None:
        job_id = self._job("city-success")

        result = self._workflow().execute(
            self._request(job_id=job_id, run_key="city-success")
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["FOOD", "WOOD"], result.result["collected_resource_types"])
        self.assertEqual(
            [
                "normalize:standard",
                "scan:1",
                "collect:FOOD:food-1",
                "verify:FOOD:food-1",
                "collect:WOOD:wood-1",
                "verify:WOOD:wood-1",
                "scan:2",
            ],
            self.driver.calls,
        )
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual(["FOOD", "WOOD"], payload["result"]["collected_resource_types"])
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_shifted_layout_with_optional_crystal_collects_only_configured_rois(self) -> None:
        policy = CityResourceCollectionPolicy(
            layout_profile=_profile(include_crystal=True, shifted=True),
            enabled_resource_types=(
                CityResourceType.FOOD,
                CityResourceType.WOOD,
                CityResourceType.STONE,
                CityResourceType.GOLD,
                CityResourceType.CRYSTAL,
            ),
        )
        self.driver.scans = [
            CityResourceScan(
                CityResourceScanStatus.READY,
                observations=(
                    CityResourceObservation(CityResourceType.FOOD, 0.95, x=20, y=20, indicator_id="old-food"),
                    CityResourceObservation(CityResourceType.CRYSTAL, 0.94, x=360, y=20, indicator_id="crystal-1"),
                ),
            ),
            CityResourceScan(CityResourceScanStatus.NONE_READY),
        ]

        result = self._workflow().execute(
            self._request(run_key="city-crystal", policy=policy)
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["CRYSTAL"], result.result["collected_resource_types"])
        self.assertIn("collect:CRYSTAL:crystal-1", self.driver.calls)
        self.assertNotIn("collect:FOOD:old-food", self.driver.calls)

    def test_partial_readiness_ignores_disabled_and_below_threshold_resources(self) -> None:
        policy = CityResourceCollectionPolicy(
            layout_profile=_profile(),
            enabled_resource_types=(CityResourceType.FOOD, CityResourceType.GOLD),
            minimum_detector_confidence=0.9,
        )
        self.driver.scans = [
            CityResourceScan(
                CityResourceScanStatus.READY,
                observations=(
                    CityResourceObservation(CityResourceType.FOOD, 0.93, x=20, y=20, indicator_id="food-1"),
                    CityResourceObservation(CityResourceType.WOOD, 0.99, x=80, y=20, indicator_id="wood-disabled"),
                    CityResourceObservation(CityResourceType.GOLD, 0.70, x=200, y=20, indicator_id="gold-low"),
                ),
            ),
            CityResourceScan(CityResourceScanStatus.NONE_READY),
        ]

        result = self._workflow().execute(
            self._request(run_key="city-partial", policy=policy)
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["FOOD"], result.result["collected_resource_types"])
        self.assertEqual(1, len(result.result["collection_attempts"]))

    def test_overlapping_icons_keep_highest_confidence_and_do_not_double_click(self) -> None:
        profile = CityLayoutProfile(
            profile_id="overlap",
            screen_width=300,
            screen_height=200,
            rois=(
                CityResourceRoi(CityResourceType.FOOD, 10, 10, 60, 60),
                CityResourceRoi(CityResourceType.WOOD, 40, 10, 60, 60),
            ),
        )
        policy = CityResourceCollectionPolicy(
            layout_profile=profile,
            enabled_resource_types=(CityResourceType.FOOD, CityResourceType.WOOD),
            overlap_distance_pixels=20,
        )
        self.driver.scans = [
            CityResourceScan(
                CityResourceScanStatus.READY,
                observations=(
                    CityResourceObservation(CityResourceType.FOOD, 0.91, x=50, y=30, indicator_id="food-overlap"),
                    CityResourceObservation(CityResourceType.WOOD, 0.96, x=52, y=31, indicator_id="wood-overlap"),
                ),
            ),
            CityResourceScan(CityResourceScanStatus.NONE_READY),
        ]

        result = self._workflow().execute(
            self._request(run_key="city-overlap", policy=policy)
        )

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["WOOD"], result.result["collected_resource_types"])
        self.assertIn("collect:WOOD:wood-overlap", self.driver.calls)
        self.assertNotIn("collect:FOOD:food-overlap", self.driver.calls)

    def test_popup_overlay_is_cleared_then_collection_retries_next_pass(self) -> None:
        self.driver.scans = [
            CityResourceScan(
                CityResourceScanStatus.POPUP_OVERLAY,
                message="Daily offer popup visible.",
                screenshot_path="runtime/screens/popup.png",
            ),
            CityResourceScan(
                CityResourceScanStatus.READY,
                observations=(
                    CityResourceObservation(CityResourceType.GOLD, 0.94, x=200, y=20, indicator_id="gold-1"),
                ),
            ),
            CityResourceScan(CityResourceScanStatus.NONE_READY),
        ]

        result = self._workflow().execute(self._request(run_key="city-popup"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(["GOLD"], result.result["collected_resource_types"])
        self.assertEqual(["normalize:standard", "scan:1", "clear_overlays", "scan:2", "collect:GOLD:gold-1", "verify:GOLD:gold-1", "scan:3"], self.driver.calls)

    def test_no_resources_ready_returns_skipped_without_clicking(self) -> None:
        self.driver.scans = [CityResourceScan(CityResourceScanStatus.NONE_READY)]

        result = self._workflow().execute(
            self._request(job_id=self._job("city-none"), run_key="city-none")
        )

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("collect_resources", result.result["terminal_state"])
        self.assertFalse(any(call.startswith("collect:") for call in self.driver.calls))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("completed", run.status)  # type: ignore[union-attr]
        self.assertEqual("", run.error_message)  # type: ignore[union-attr]

    def test_unverified_collection_blocks_and_records_recovery(self) -> None:
        self.driver.verify = CityResourceCollectionResult(
            False,
            changed=False,
            message="Food bubble did not change.",
            retryable=False,
            screenshot_path="runtime/screens/food-not-changed.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("city-unverified"), run_key="city-unverified")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("collect_resources", result.result["terminal_state"])
        self.assertEqual("Food bubble did not change.", result.result["terminal_reason"])
        self.assertEqual(
            {"attempted": False, "healthy": True, "circuit_opened": False},
            result.result["recovery_outcome"],
        )
        self.assertEqual(1, len(self.incidents.list_open()))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("failed", run.status)  # type: ignore[union-attr]
        self.assertEqual("Food bubble did not change.", run.error_message)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
