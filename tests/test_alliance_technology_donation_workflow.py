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
from rok_assistant.tasks.alliance_technology_donation_workflow import (  # noqa: E402
    ALLIANCE_TECHNOLOGY_DONATION_STATES,
    ALLIANCE_TECHNOLOGY_DONATION_TEMPLATE_KEYS,
    AllianceDonationAction,
    AllianceDonationAttemptResult,
    AllianceDonationConfirmation,
    AllianceDonationCostType,
    AllianceDonationState,
    AllianceDonationStateStatus,
    AllianceTechnologyDonationConfig,
    AllianceTechnologyDonationPolicy,
    AllianceTechnologyDonationRequest,
    AllianceTechnologyDonationWorkflow,
    AllianceTechnologyObservation,
    AllianceTechnologyScan,
    AllianceTechnologyScanStatus,
)
from rok_assistant.tasks.resource_search_workflow import ResourceGatheringActionResult  # noqa: E402
from rok_assistant.workflow_engine import WorkflowOutcome  # noqa: E402


def _tech(
    key: str,
    *,
    recommended: bool = True,
    confidence: float = 0.94,
    selected: bool = False,
) -> AllianceTechnologyObservation:
    return AllianceTechnologyObservation(
        key,
        display_name=key.title(),
        recommended=recommended,
        confidence=confidence,
        selected=selected,
    )


class FakeAllianceTechnologyDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.open_alliance_result = ResourceGatheringActionResult(True, data={"scene": "ALLIANCE_HOME"})
        self.open_technology = ResourceGatheringActionResult(True, data={"scene": "ALLIANCE_TECHNOLOGY"})
        self.scan = AllianceTechnologyScan(
            AllianceTechnologyScanStatus.READY,
            observations=(
                _tech("ARCHITECTURE", recommended=False),
                _tech("DONATION", recommended=True),
            ),
        )
        self.select_result = ResourceGatheringActionResult(True, data={"selected": True})
        self.states: list[AllianceDonationState] = [
            AllianceDonationState(
                AllianceDonationStateStatus.READY,
                "DONATION",
                contribution_count=10,
                contribution_limit=20,
                available_actions=(AllianceDonationAction("normal", AllianceDonationCostType.FOOD, 100),),
            )
        ]
        self.donate_results: list[AllianceDonationAttemptResult] = [
            AllianceDonationAttemptResult(True, changed=True, donation_count=1, contribution_count=11)
        ]
        self.verify_results: list[AllianceDonationAttemptResult] = [
            AllianceDonationAttemptResult(True, changed=True, donation_count=1, contribution_count=11)
        ]
        self.confirm_result = AllianceDonationAttemptResult(True, changed=True, donation_count=1)

    def open_alliance(self, _request, _character, _policy):
        self.calls.append("open_alliance")
        return self.open_alliance_result

    def open_alliance_technology(self, _request, _character, _policy):
        self.calls.append("open_alliance_technology")
        return self.open_technology

    def scan_technologies(self, _request, _character, _policy):
        self.calls.append("scan_technologies")
        return self.scan

    def select_technology(self, _request, _character, technology, _policy):
        self.calls.append(f"select:{technology.normalized_technology_key()}")
        return self.select_result

    def scan_donation_state(self, _request, _character, technology, _policy):
        self.calls.append(f"scan_donation_state:{technology.normalized_technology_key()}")
        if len(self.states) > 1:
            return self.states.pop(0)
        return self.states[0]

    def donate_to_technology(self, _request, _character, technology, action, _policy):
        self.calls.append(
            f"donate:{technology.normalized_technology_key()}:{action.action_key}:{action.normalized_cost_type().value}"
        )
        if len(self.donate_results) > 1:
            return self.donate_results.pop(0)
        return self.donate_results[0]

    def handle_donation_confirmation(self, _request, _character, technology, action, attempt, _policy):
        del attempt
        self.calls.append(f"confirm:{technology.normalized_technology_key()}:{action.action_key}")
        return self.confirm_result

    def verify_donation_state(self, _request, _character, technology, before, attempt, _policy):
        del before, attempt
        self.calls.append(f"verify:{technology.normalized_technology_key()}")
        if len(self.verify_results) > 1:
            return self.verify_results.pop(0)
        return self.verify_results[0]


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


class AllianceTechnologyDonationWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp_dir.name) / "alliance-tech.sqlite3")
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
        self.driver = FakeAllianceTechnologyDriver()
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

    def _workflow(self) -> AllianceTechnologyDonationWorkflow:
        return AllianceTechnologyDonationWorkflow(
            characters=self.characters,
            driver=self.driver,
            recovery_watchdog=self.watchdog,
            job_runs=self.job_runs,
            step_runs=self.step_runs,
            incidents=self.incidents,
            config=AllianceTechnologyDonationConfig(
                workflow_timeout_seconds=30,
                step_timeout_seconds=2,
                retry_delay_seconds=0,
            ),
        )

    def _request(
        self,
        *,
        job_id: int | None = None,
        run_key: str = "alliance-tech-run",
        policy: AllianceTechnologyDonationPolicy | None = None,
    ) -> AllianceTechnologyDonationRequest:
        return AllianceTechnologyDonationRequest(
            instance_id=self.instance_id,
            instance_index=0,
            instance_name="MEmu 1",
            character_id=self.character_id,
            policy=policy or AllianceTechnologyDonationPolicy(),
            job_id=job_id,
            run_key=run_key,
        )

    def test_workflow_exposes_required_states_and_template_keys(self) -> None:
        self.assertEqual(ALLIANCE_TECHNOLOGY_DONATION_STATES, self._workflow().workflow_states)
        self.assertIn("alliance.technology.donate_button", ALLIANCE_TECHNOLOGY_DONATION_TEMPLATE_KEYS)
        self.assertIn("alliance.technology.gem_confirmation", ALLIANCE_TECHNOLOGY_DONATION_TEMPLATE_KEYS)

    def test_no_donation_available_returns_skipped(self) -> None:
        job_id = self._job("alliance-tech-none")
        self.driver.states = [
            AllianceDonationState(
                AllianceDonationStateStatus.NO_DONATION_AVAILABLE,
                "DONATION",
                message="No donation available.",
                screenshot_path="runtime/screens/alliance-tech-none.png",
            )
        ]

        result = self._workflow().execute(self._request(job_id=job_id, run_key="alliance-tech-none"))

        self.assertEqual(WorkflowOutcome.SKIPPED, result.outcome)
        self.assertEqual("scan_donation_state", result.result["terminal_state"])
        self.assertEqual("No donation available.", result.result["skipped_reason"])
        self.assertFalse(any(call.startswith("donate:") for call in self.driver.calls))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("completed", run.status)  # type: ignore[union-attr]

    def test_configured_technology_is_selected(self) -> None:
        policy = AllianceTechnologyDonationPolicy(
            recommended_only=False,
            target_technology_key="ARCHITECTURE",
        )
        self.driver.states = [
            AllianceDonationState(
                AllianceDonationStateStatus.READY,
                "ARCHITECTURE",
                contribution_count=4,
                contribution_limit=20,
                available_actions=(AllianceDonationAction("normal", AllianceDonationCostType.WOOD, 50),),
            )
        ]

        result = self._workflow().execute(self._request(run_key="alliance-tech-explicit", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertIn("select:ARCHITECTURE", self.driver.calls)
        self.assertEqual("ARCHITECTURE", result.result["selected_technology"]["technology_key"])

    def test_one_successful_donation_is_persisted(self) -> None:
        job_id = self._job("alliance-tech-one")

        result = self._workflow().execute(self._request(job_id=job_id, run_key="alliance-tech-one"))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(1, result.result["donation_count"])
        self.assertEqual(
            [
                "open_alliance",
                "open_alliance_technology",
                "scan_technologies",
                "select:DONATION",
                "scan_donation_state:DONATION",
                "donate:DONATION:normal:FOOD",
                "verify:DONATION",
            ],
            self.driver.calls,
        )
        run = self.job_runs.get(result.job_run_id or 0)
        payload = json.loads(run.result_json)  # type: ignore[union-attr]
        self.assertEqual(1, payload["result"]["donation_count"])
        self.assertEqual("DONATION", payload["result"]["donation_attempts"][0]["technology_key"])

    def test_multiple_donations_within_attempt_budget(self) -> None:
        self.driver.states = [
            AllianceDonationState(
                AllianceDonationStateStatus.READY,
                "DONATION",
                contribution_count=10,
                contribution_limit=20,
                available_actions=(AllianceDonationAction("normal", AllianceDonationCostType.FOOD, 100),),
            ),
            AllianceDonationState(
                AllianceDonationStateStatus.READY,
                "DONATION",
                contribution_count=11,
                contribution_limit=20,
                available_actions=(AllianceDonationAction("normal", AllianceDonationCostType.FOOD, 100),),
            ),
        ]
        self.driver.donate_results = [
            AllianceDonationAttemptResult(True, changed=True, donation_count=1, contribution_count=11),
            AllianceDonationAttemptResult(True, changed=True, donation_count=1, contribution_count=12),
        ]
        self.driver.verify_results = [
            AllianceDonationAttemptResult(True, changed=True, donation_count=1, contribution_count=11),
            AllianceDonationAttemptResult(True, changed=True, donation_count=1, contribution_count=12),
        ]
        policy = AllianceTechnologyDonationPolicy(max_donations_per_run=2)

        result = self._workflow().execute(self._request(run_key="alliance-tech-two", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(2, result.result["donation_count"])
        self.assertEqual(2, len(result.result["donation_attempts"]))
        self.assertEqual(2, len([call for call in self.driver.calls if call.startswith("donate:")]))

    def test_donation_stops_at_configured_limit(self) -> None:
        policy = AllianceTechnologyDonationPolicy(
            max_donations_per_run=3,
            max_donations_per_day=5,
            donations_used_today=4,
        )

        result = self._workflow().execute(self._request(run_key="alliance-tech-daily-limit", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, result.outcome)
        self.assertEqual(1, result.result["donation_count"])
        self.assertEqual(1, len(result.result["donation_attempts"]))
        self.assertEqual(1, result.result["policy"]["effective_run_budget"])

    def test_unsafe_gem_confirmation_is_blocked_unless_allowed(self) -> None:
        self.driver.donate_results = [
            AllianceDonationAttemptResult(
                True,
                changed=False,
                donation_count=0,
                contribution_count=10,
                confirmation=AllianceDonationConfirmation.PREMIUM_CURRENCY,
                screenshot_path="runtime/screens/tech-premium-confirm.png",
            )
        ]

        result = self._workflow().execute(self._request(run_key="alliance-tech-gems"))

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("donate", result.result["terminal_state"])
        self.assertIn("Premium currency confirmation", result.result["terminal_reason"])
        self.assertEqual(
            "runtime/screens/tech-premium-confirm.png",
            result.result["failure_evidence"]["screenshot_path"],
        )

        policy = AllianceTechnologyDonationPolicy(
            allow_premium_currency=True,
            allowed_cost_types=(AllianceDonationCostType.GEMS,),
        )
        self.driver.calls.clear()
        self.driver.states = [
            AllianceDonationState(
                AllianceDonationStateStatus.READY,
                "DONATION",
                contribution_count=10,
                contribution_limit=20,
                available_actions=(AllianceDonationAction("gem", AllianceDonationCostType.GEMS, 50, premium=True),),
            )
        ]
        self.driver.donate_results = [
            AllianceDonationAttemptResult(
                True,
                changed=True,
                donation_count=1,
                contribution_count=11,
                confirmation=AllianceDonationConfirmation.PREMIUM_CURRENCY,
            )
        ]

        allowed = self._workflow().execute(self._request(run_key="alliance-tech-gems-allowed", policy=policy))

        self.assertEqual(WorkflowOutcome.SUCCESS, allowed.outcome)
        self.assertIn("confirm:DONATION:gem", self.driver.calls)

    def test_missing_configured_technology_records_failure_evidence(self) -> None:
        policy = AllianceTechnologyDonationPolicy(
            recommended_only=False,
            target_technology_key="WAR_TECH",
        )
        self.driver.scan = AllianceTechnologyScan(
            AllianceTechnologyScanStatus.READY,
            observations=(_tech("DONATION", recommended=True),),
            screenshot_path="runtime/screens/tech-missing.png",
        )

        result = self._workflow().execute(
            self._request(job_id=self._job("alliance-tech-missing"), run_key="alliance-tech-missing", policy=policy)
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("select_technology", result.result["terminal_state"])
        self.assertIn("WAR_TECH", result.result["terminal_reason"])
        self.assertEqual("runtime/screens/tech-missing.png", result.result["failure_evidence"]["screenshot_path"])
        self.assertEqual(1, len(self.incidents.list_open()))
        run = self.job_runs.get(result.job_run_id or 0)
        self.assertEqual("failed", run.status)  # type: ignore[union-attr]

    def test_verification_failure_triggers_bounded_recovery_or_blocked_result(self) -> None:
        self.driver.verify_results = [
            AllianceDonationAttemptResult(
                False,
                changed=False,
                donation_count=0,
                contribution_count=10,
                message="Contribution count did not change.",
                retryable=False,
                screenshot_path="runtime/screens/tech-verify-failed.png",
            )
        ]

        result = self._workflow().execute(
            self._request(job_id=self._job("alliance-tech-verify-failed"), run_key="alliance-tech-verify-failed")
        )

        self.assertEqual(WorkflowOutcome.BLOCKED, result.outcome)
        self.assertEqual("donate", result.result["terminal_state"])
        self.assertIn("did not change", result.result["terminal_reason"])
        self.assertEqual(
            {"attempted": False, "healthy": True, "circuit_opened": False},
            result.result["recovery_outcome"],
        )
        self.assertEqual([None, result.job_run_id], self.watchdog.calls)


if __name__ == "__main__":
    unittest.main()
