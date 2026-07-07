from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.vision import (
    DetectionResult,
    MatchingMetadata,
    ReplayCaseDefinition,
    ReplayDatasetDefinition,
    ReplayDetectionExpectation,
    ReplayExpectation,
    ReplaySceneExpectation,
    RegionOfInterest,
    ResolutionProfile,
    ScaleRange,
    SceneClassificationResult,
    SceneClassificationStatus,
    SceneDefinition,
    SceneRule,
    TemplateDefinition,
    TemplatePack,
    TemplateRegistry,
    ThresholdCalibrationCase,
    ThresholdCalibrationRequest,
    ThresholdCalibrationRunner,
    ThresholdCalibrationTarget,
    ThresholdCandidate,
    ThresholdCandidateResult,
    ThresholdCaseObservation,
    ThresholdMetrics,
    ThresholdObservationStatus,
    ValidationDiagnostic,
)


class ThresholdCalibrationTest(unittest.TestCase):
    def test_candidate_validation_rejects_invalid_values(self) -> None:
        self.assertEqual(
            "template_detection_threshold:strict:0.9",
            ThresholdCandidate("strict", ThresholdCalibrationTarget.TEMPLATE_DETECTION, 0.9).semantic_id,
        )
        for value in (" ",):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ThresholdCandidate(value, ThresholdCalibrationTarget.TEMPLATE_DETECTION, 0.5)
        for value in (True, math.nan, math.inf, -math.inf, -0.01, 1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ThresholdCandidate("bad", ThresholdCalibrationTarget.TEMPLATE_DETECTION, value)

    def test_candidate_duplicates_and_ordering_are_deterministic(self) -> None:
        candidates = (
            ThresholdCandidate("b", ThresholdCalibrationTarget.TEMPLATE_DETECTION, 0.8),
            ThresholdCandidate("a", ThresholdCalibrationTarget.TEMPLATE_DETECTION, 0.7),
        )
        request = self._request(candidates=candidates, cases=(self._case("one", True),))

        self.assertEqual(("a", "b"), tuple(candidate.identifier for candidate in request.candidates))

        duplicate_id = self._runner().run(
            self._request(
                candidates=(
                    ThresholdCandidate("same", ThresholdCalibrationTarget.TEMPLATE_DETECTION, 0.7),
                    ThresholdCandidate("same", ThresholdCalibrationTarget.TEMPLATE_DETECTION, 0.8),
                ),
                cases=(self._case("one", True),),
            )
        )
        duplicate_value = self._runner().run(
            self._request(
                candidates=(
                    ThresholdCandidate("one", ThresholdCalibrationTarget.TEMPLATE_DETECTION, 0.7),
                    ThresholdCandidate("two", ThresholdCalibrationTarget.TEMPLATE_DETECTION, 0.7),
                ),
                cases=(self._case("one", True),),
            )
        )

        self.assertDiagnostic(duplicate_id.diagnostics, "calibration.candidate.duplicate_identifier")
        self.assertDiagnostic(duplicate_value.diagnostics, "calibration.candidate.duplicate_threshold")

    def test_request_validation_prevents_pipeline_calls(self) -> None:
        analyzer = _RecordingAnalyzer({"city.collect": DetectionResult("city.collect", 0.9)})
        result = self._runner().run(
            ThresholdCalibrationRequest(
                candidates=(),
                cases=(),
                dataset_root=None,
                registry=None,
                analyzer=analyzer,
            )
        )

        self.assertFalse(result.is_valid)
        self.assertDiagnostic(result.diagnostics, "calibration.request.empty_candidates")
        self.assertDiagnostic(result.diagnostics, "calibration.request.empty_cases")
        self.assertEqual((), analyzer.called_keys())

    def test_case_contract_rejects_missing_ground_truth_duplicates_and_unsupported_targets(self) -> None:
        with self.assertRaises(ValueError):
            ThresholdCalibrationCase.from_replay_case(
                ReplayCaseDefinition(
                    "case",
                    "screens/one.png",
                    "city.collect",
                    ReplayExpectation(detection=ReplayDetectionExpectation()),
                ),
                ThresholdCalibrationTarget.TEMPLATE_DETECTION,
            )
        unsupported = self._runner().run(
            self._request(
                candidates=(ThresholdCandidate("c", ThresholdCalibrationTarget.TEMPLATE_DETECTION, 0.5),),
                cases=(
                    self._case("same", True),
                    self._case("same", False),
                    ThresholdCalibrationCase(
                        "scene",
                        ThresholdCalibrationTarget.SCENE_MINIMUM_SCORE,
                        True,
                        "screens/one.png",
                        scene_key="city",
                    ),
                ),
            )
        )

        self.assertDiagnostic(unsupported.diagnostics, "calibration.case.duplicate_identifier")
        self.assertDiagnostic(unsupported.diagnostics, "calibration.case.unsupported_target")

    def test_detection_equality_and_near_threshold_cases(self) -> None:
        analyzer = _RecordingAnalyzer(
            {
                "equal": DetectionResult("equal", 0.75),
                "below": DetectionResult("below", 0.75 - 1e-9),
                "above": DetectionResult("above", 0.75 + 1e-9),
                "negative-above": DetectionResult("negative-above", 0.8),
                "negative-equal": DetectionResult("negative-equal", 0.75),
                "negative-below": DetectionResult("negative-below", 0.74),
            }
        )
        cases = tuple(
            ThresholdCalibrationCase(case_id, ThresholdCalibrationTarget.TEMPLATE_DETECTION, expected, "screens/one.png", template_key=case_id)
            for case_id, expected in (
                ("equal", True),
                ("below", True),
                ("above", True),
                ("negative-above", False),
                ("negative-equal", False),
                ("negative-below", False),
            )
        )

        result = self._runner().run(
            self._request(
                candidates=(ThresholdCandidate("candidate", ThresholdCalibrationTarget.TEMPLATE_DETECTION, 0.75),),
                cases=cases,
                analyzer=analyzer,
                registry=self._registry(tuple(case.template_key for case in cases)),
            )
        )
        observations = {observation.case_id: observation for observation in result.candidate_results[0].observations}

        self.assertEqual(ThresholdObservationStatus.TRUE_POSITIVE, observations["equal"].status)
        self.assertEqual(ThresholdObservationStatus.FALSE_NEGATIVE, observations["below"].status)
        self.assertEqual(ThresholdObservationStatus.TRUE_POSITIVE, observations["above"].status)
        self.assertEqual(ThresholdObservationStatus.FALSE_POSITIVE, observations["negative-above"].status)
        self.assertEqual(ThresholdObservationStatus.FALSE_POSITIVE, observations["negative-equal"].status)
        self.assertEqual(ThresholdObservationStatus.TRUE_NEGATIVE, observations["negative-below"].status)
        self.assertEqual(tuple(sorted(case.template_key for case in cases)), analyzer.called_keys())

    def test_detection_pipeline_diagnostics_and_later_cases_continue(self) -> None:
        malformed = object()
        non_finite = object.__new__(DetectionResult)
        object.__setattr__(non_finite, "matched_semantic_key", "nonfinite")
        object.__setattr__(non_finite, "confidence", float("nan"))
        analyzer = _RecordingAnalyzer(
            {
                "raises": cv2.error("raw failure"),
                "malformed": malformed,
                "nonfinite": non_finite,
                "wrong": DetectionResult("other", 0.9),
                "later": DetectionResult("later", 0.9),
            }
        )
        cases = tuple(self._case(key, True, template_key=key) for key in ("raises", "malformed", "nonfinite", "wrong", "later"))

        result = self._runner().run(
            self._request(
                candidates=(ThresholdCandidate("candidate", ThresholdCalibrationTarget.TEMPLATE_DETECTION, 0.5),),
                cases=cases,
                analyzer=analyzer,
                registry=self._registry(tuple(case.template_key for case in cases) + ("other",)),
            )
        )
        observations = {observation.case_id: observation for observation in result.candidate_results[0].observations}

        self.assertEqual(ThresholdObservationStatus.PIPELINE_ERROR, observations["raises"].status)
        self.assertEqual(ThresholdObservationStatus.PIPELINE_ERROR, observations["malformed"].status)
        self.assertEqual(ThresholdObservationStatus.PIPELINE_ERROR, observations["nonfinite"].status)
        self.assertEqual(ThresholdObservationStatus.FALSE_NEGATIVE, observations["wrong"].status)
        self.assertDiagnostic(observations["wrong"].diagnostics, "calibration.detection.wrong_template")
        self.assertEqual(ThresholdObservationStatus.TRUE_POSITIVE, observations["later"].status)
        self.assertEqual(("later", "malformed", "nonfinite", "raises", "wrong"), analyzer.called_keys())

    def test_scene_calibration_evaluates_minimum_score_without_mutating_definitions(self) -> None:
        classifier = _RecordingClassifier(
            {
                "above": SceneClassificationResult(SceneClassificationStatus.CLASSIFIED, scene_key="city", score=0.81),
                "equal": SceneClassificationResult(SceneClassificationStatus.CLASSIFIED, scene_key="city", score=0.8),
                "below": SceneClassificationResult(SceneClassificationStatus.CLASSIFIED, scene_key="city", score=0.79),
                "wrong": SceneClassificationResult(SceneClassificationStatus.CLASSIFIED, scene_key="map", score=0.9),
                "unknown": SceneClassificationResult(SceneClassificationStatus.UNKNOWN, score=0.0),
                "ambiguous": SceneClassificationResult(
                    SceneClassificationStatus.AMBIGUOUS,
                    score=0.9,
                    candidates=(
                        self._scene_candidate("city", 0.9),
                        self._scene_candidate("map", 0.9),
                    ),
                    diagnostics=(ValidationDiagnostic("scene.ambiguous", "scene", "ambiguous"),),
                ),
                "invalid": SceneClassificationResult(
                    SceneClassificationStatus.INVALID,
                    diagnostics=(ValidationDiagnostic("scene.invalid", "scene", "invalid"),),
                ),
            }
        )
        definitions = (SceneDefinition("city", SceneRule(required_template_keys=("city.collect",), minimum_score=0.2)),)
        cases = tuple(
            ThresholdCalibrationCase(case_id, ThresholdCalibrationTarget.SCENE_MINIMUM_SCORE, expected, "screens/one.png", scene_key="city")
            for case_id, expected in (
                ("above", True),
                ("equal", True),
                ("below", True),
                ("wrong", True),
                ("unknown", False),
                ("ambiguous", False),
                ("invalid", False),
            )
        )

        result = self._runner().run(
            self._request(
                candidates=(ThresholdCandidate("scene", ThresholdCalibrationTarget.SCENE_MINIMUM_SCORE, 0.8),),
                cases=cases,
                scene_classifier=classifier,
                scene_definitions=definitions,
            )
        )
        observations = {observation.case_id: observation for observation in result.candidate_results[0].observations}

        self.assertEqual(ThresholdObservationStatus.TRUE_POSITIVE, observations["above"].status)
        self.assertEqual(ThresholdObservationStatus.TRUE_POSITIVE, observations["equal"].status)
        self.assertEqual(ThresholdObservationStatus.FALSE_NEGATIVE, observations["below"].status)
        self.assertEqual(ThresholdObservationStatus.FALSE_NEGATIVE, observations["wrong"].status)
        self.assertEqual(ThresholdObservationStatus.TRUE_NEGATIVE, observations["unknown"].status)
        self.assertEqual(ThresholdObservationStatus.TRUE_NEGATIVE, observations["ambiguous"].status)
        self.assertEqual(ThresholdObservationStatus.TRUE_NEGATIVE, observations["invalid"].status)
        self.assertEqual(7, classifier.calls)
        self.assertEqual(0.2, definitions[0].rule.minimum_score)

    def test_scene_pipeline_failures_are_structured_and_do_not_stop_later_cases(self) -> None:
        non_finite = object.__new__(SceneClassificationResult)
        object.__setattr__(non_finite, "status", SceneClassificationStatus.CLASSIFIED)
        object.__setattr__(non_finite, "scene_key", "city")
        object.__setattr__(non_finite, "score", float("nan"))
        classifier = _RecordingClassifier(
            {
                "exception": cv2.error("bad scene"),
                "malformed": object(),
                "nonfinite": non_finite,
                "success": SceneClassificationResult(SceneClassificationStatus.CLASSIFIED, scene_key="city", score=0.9),
            }
        )
        cases = tuple(
            ThresholdCalibrationCase(
                case_id,
                ThresholdCalibrationTarget.SCENE_MINIMUM_SCORE,
                True,
                "screens/one.png",
                scene_key="city",
            )
            for case_id in ("exception", "malformed", "nonfinite", "success")
        )

        result = self._runner().run(
            self._request(
                candidates=(ThresholdCandidate("scene", ThresholdCalibrationTarget.SCENE_MINIMUM_SCORE, 0.8),),
                cases=cases,
                scene_classifier=classifier,
                scene_definitions=(SceneDefinition("city", SceneRule(required_template_keys=("city.collect",))),),
            )
        )
        observations = {observation.case_id: observation for observation in result.candidate_results[0].observations}

        self.assertEqual(ThresholdObservationStatus.PIPELINE_ERROR, observations["exception"].status)
        self.assertEqual(ThresholdObservationStatus.PIPELINE_ERROR, observations["malformed"].status)
        self.assertEqual(ThresholdObservationStatus.PIPELINE_ERROR, observations["nonfinite"].status)
        self.assertEqual(ThresholdObservationStatus.TRUE_POSITIVE, observations["success"].status)
        self.assertEqual(4, classifier.calls)

    def test_multiple_candidates_repeated_execution_and_exact_analyzer_call_count(self) -> None:
        analyzer = _RecordingAnalyzer(
            {
                "city.collect": DetectionResult("city.collect", 0.75),
                "city.help": DetectionResult("city.help", 0.25),
            }
        )
        second_analyzer = _RecordingAnalyzer(
            {
                "city.collect": DetectionResult("city.collect", 0.75),
                "city.help": DetectionResult("city.help", 0.25),
            }
        )
        cases = (
            self._case("collect", True, template_key="city.collect"),
            self._case("help", False, template_key="city.help"),
        )
        request = self._request(
            candidates=(
                ThresholdCandidate("loose", ThresholdCalibrationTarget.TEMPLATE_DETECTION, 0.2),
                ThresholdCandidate("strict", ThresholdCalibrationTarget.TEMPLATE_DETECTION, 0.8),
            ),
            cases=cases,
            analyzer=analyzer,
            registry=self._registry(("city.collect", "city.help")),
        )

        second_request = self._request(
            candidates=request.candidates,
            cases=cases,
            analyzer=second_analyzer,
            registry=self._registry(("city.collect", "city.help")),
        )

        first = self._runner().run(request)
        second = self._runner().run(second_request)

        self.assertEqual(first.candidate_results, second.candidate_results)
        self.assertEqual(("city.collect", "city.collect", "city.help", "city.help"), analyzer.called_keys())
        self.assertEqual(("city.collect", "city.collect", "city.help", "city.help"), second_analyzer.called_keys())
        self.assertEqual(("loose", "strict"), tuple(result.candidate.identifier for result in first.candidate_results))
        loose = first.candidate_results[0].metrics
        strict = first.candidate_results[1].metrics
        self.assertEqual(1, loose.true_positives)
        self.assertEqual(1, loose.false_positives)
        self.assertEqual(1, strict.false_negatives)
        self.assertEqual(1, strict.true_negatives)

    def test_metrics_have_deterministic_zero_denominator_policy(self) -> None:
        perfect = ThresholdMetrics.from_observations(
            (
                self._observation("tp", True, True),
                self._observation("tn", False, False),
            )
        )
        all_false_positive = ThresholdMetrics.from_observations((self._observation("fp", False, True),))
        all_false_negative = ThresholdMetrics.from_observations((self._observation("fn", True, False),))
        zero_positive = ThresholdMetrics.from_observations((self._observation("tn", False, False),))
        zero_negative = ThresholdMetrics.from_observations((self._observation("tp", True, True),))

        self.assertEqual(1.0, perfect.precision)
        self.assertEqual(1.0, perfect.recall)
        self.assertEqual(1.0, perfect.specificity)
        self.assertEqual(1.0, perfect.accuracy)
        self.assertEqual(1.0, perfect.f1_score)
        self.assertEqual(0.0, all_false_positive.precision)
        self.assertEqual(0.0, all_false_negative.recall)
        self.assertEqual(0.0, zero_positive.recall)
        self.assertEqual(0.0, zero_negative.specificity)
        with self.assertRaises(ValueError):
            ThresholdMetrics(1, 0, 0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 1.0, 1.0)
        for metric in perfect.precision, perfect.recall, perfect.specificity, perfect.accuracy, perfect.f1_score:
            self.assertTrue(math.isfinite(metric))

    def test_replay_dataset_integration_and_path_diagnostics_are_safe(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            manifest = {
                "schema_version": 1,
                "dataset_id": "vision.synthetic",
                "cases": [
                    {
                        "case_id": "case",
                        "screenshot": "screens/one.png",
                        "template_key": "city.collect",
                        "expectation": {"detection": {"expected_match": True}},
                    }
                ],
            }
            (root / "replay.json").write_text(json.dumps(manifest), encoding="utf-8")

            result = self._runner().run(
                ThresholdCalibrationRequest(
                    candidates=(ThresholdCandidate("c", ThresholdCalibrationTarget.TEMPLATE_DETECTION, 0.5),),
                    manifest_path="replay.json",
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    analyzer=_RecordingAnalyzer({"city.collect": DetectionResult("city.collect", 0.9)}),
                )
            )

            self.assertTrue(result.is_valid, result.diagnostics)
            self.assertEqual(ThresholdObservationStatus.TRUE_POSITIVE, result.candidate_results[0].observations[0].status)

            bad = self._runner().run(
                self._request(
                    cases=(self._case("bad", True, screenshot="../outside.png"),),
                    analyzer=_RecordingAnalyzer({"city.collect": DetectionResult("city.collect", 0.9)}),
                )
            )
            combined = self._diagnostic_text(bad.candidate_results[0].observations[0].diagnostics)
            self.assertEqual(ThresholdObservationStatus.INPUT_ERROR, bad.candidate_results[0].observations[0].status)
            self.assertNotIn(str(root), combined)
            self.assertNotIn(str(Path(root).drive), combined)

    def test_path_rejections_and_malformed_images_are_structured(self) -> None:
        invalid_paths = ("../outside.png", str(Path.cwd()), "C:relative.png", "\\\\server\\share\\shot.png")
        for screenshot in invalid_paths:
            with self.subTest(screenshot=screenshot):
                result = self._runner().run(
                    self._request(
                        cases=(self._case("bad", True, screenshot=screenshot),),
                        analyzer=_RecordingAnalyzer({"city.collect": DetectionResult("city.collect", 0.9)}),
                    )
                )
                self.assertEqual(ThresholdObservationStatus.INPUT_ERROR, result.candidate_results[0].observations[0].status)

        with self._dataset_root() as root:
            bad = root / "screens" / "bad.png"
            bad.parent.mkdir(parents=True)
            bad.write_bytes(b"not an image")
            result = self._runner(root).run(
                self._request(
                    cases=(self._case("bad", True, screenshot="screens/bad.png"),),
                    analyzer=_RecordingAnalyzer({"city.collect": DetectionResult("city.collect", 0.9)}),
                    dataset_root=root,
                )
            )
            self.assertEqual(ThresholdObservationStatus.INPUT_ERROR, result.candidate_results[0].observations[0].status)
            self.assertDiagnostic(result.candidate_results[0].observations[0].diagnostics, "calibration.image.invalid")

    def test_symlink_escape_is_rejected_when_supported(self) -> None:
        with self._dataset_root() as root:
            outside = root.parent / "outside.png"
            outside.write_bytes(b"outside")
            link = root / "screens" / "escape.png"
            link.parent.mkdir(parents=True)
            try:
                os.symlink(outside, link)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")

            result = self._runner(root).run(
                self._request(
                    cases=(self._case("escape", True, screenshot="screens/escape.png"),),
                    analyzer=_RecordingAnalyzer({"city.collect": DetectionResult("city.collect", 0.9)}),
                    dataset_root=root,
                )
            )

            self.assertEqual(ThresholdObservationStatus.INPUT_ERROR, result.candidate_results[0].observations[0].status)

    def test_failed_run_then_successful_run_and_instances_are_isolated(self) -> None:
        runner = self._runner()
        failed_analyzer = _RecordingAnalyzer({"city.collect": cv2.error("bad")})
        success_analyzer = _RecordingAnalyzer({"city.collect": DetectionResult("city.collect", 0.9)})

        failed = runner.run(self._request(analyzer=failed_analyzer))
        recovered = runner.run(self._request(analyzer=success_analyzer))
        separate = self._runner().run(self._request(analyzer=success_analyzer))

        self.assertEqual(ThresholdObservationStatus.PIPELINE_ERROR, failed.candidate_results[0].observations[0].status)
        self.assertEqual(ThresholdObservationStatus.TRUE_POSITIVE, recovered.candidate_results[0].observations[0].status)
        self.assertEqual(recovered.candidate_results[0].metrics, separate.candidate_results[0].metrics)

    def test_registry_analyzer_and_classifier_remain_reusable_after_calibration(self) -> None:
        registry = self._registry(("city.collect",))
        original_definition = registry.get("city.collect")
        analyzer = _RecordingAnalyzer({"city.collect": DetectionResult("city.collect", 0.9)})
        classifier = _RecordingClassifier(
            {"scene": SceneClassificationResult(SceneClassificationStatus.CLASSIFIED, scene_key="city", score=0.9)}
        )
        runner = self._runner()

        detection = runner.run(self._request(registry=registry, analyzer=analyzer))
        scene = runner.run(
            self._request(
                candidates=(ThresholdCandidate("scene", ThresholdCalibrationTarget.SCENE_MINIMUM_SCORE, 0.8),),
                cases=(
                    ThresholdCalibrationCase(
                        "scene",
                        ThresholdCalibrationTarget.SCENE_MINIMUM_SCORE,
                        True,
                        "screens/one.png",
                        scene_key="city",
                    ),
                ),
                registry=registry,
                scene_classifier=classifier,
                scene_definitions=(SceneDefinition("city", SceneRule(required_template_keys=("city.collect",), minimum_score=0.1)),),
            )
        )

        self.assertTrue(detection.is_valid, detection.diagnostics)
        self.assertTrue(scene.is_valid, scene.diagnostics)
        self.assertEqual(original_definition, registry.get("city.collect"))
        self.assertEqual(0.9, analyzer.match(np.zeros((2, 2), dtype=np.uint8), "city.collect", registry).confidence)
        classifier.classify(np.zeros((2, 2), dtype=np.uint8), (), registry)
        self.assertEqual(2, classifier.calls)

    def test_model_invariants_reject_contradictory_states(self) -> None:
        diagnostic = ValidationDiagnostic("calibration.test", "field", "message")
        candidate = ThresholdCandidate("c", ThresholdCalibrationTarget.TEMPLATE_DETECTION, 0.5)
        observation = self._observation("case", True, True)

        with self.assertRaises(ValueError):
            ThresholdCaseObservation(
                "case",
                "c",
                ThresholdCalibrationTarget.TEMPLATE_DETECTION,
                True,
                None,
                ThresholdObservationStatus.TRUE_POSITIVE,
            )
        with self.assertRaises(ValueError):
            ThresholdCaseObservation(
                "case",
                "c",
                ThresholdCalibrationTarget.TEMPLATE_DETECTION,
                True,
                True,
                ThresholdObservationStatus.PIPELINE_ERROR,
            )
        with self.assertRaises(ValueError):
            ThresholdCandidateResult(candidate, ())
        with self.assertRaises(ValueError):
            ThresholdCandidateResult(
                candidate,
                (ThresholdCaseObservation("case", "other", ThresholdCalibrationTarget.TEMPLATE_DETECTION, True, True, ThresholdObservationStatus.TRUE_POSITIVE),),
            )
        with self.assertRaises(ValueError):
            ThresholdCaseObservation(
                "case",
                "c",
                ThresholdCalibrationTarget.TEMPLATE_DETECTION,
                True,
                None,
                ThresholdObservationStatus.PIPELINE_ERROR,
                diagnostics=(diagnostic,),
                confidence=float("nan"),
            )
        self.assertEqual(1, ThresholdCandidateResult(candidate, (observation,)).metrics.true_positives)

    def assertDiagnostic(self, diagnostics: tuple[ValidationDiagnostic, ...], code: str) -> None:
        self.assertIn(code, {diagnostic.code for diagnostic in diagnostics})

    @staticmethod
    def _diagnostic_text(diagnostics: tuple[ValidationDiagnostic, ...]) -> str:
        return "\n".join(diagnostic.message for diagnostic in diagnostics)

    def _request(self, **overrides: object) -> ThresholdCalibrationRequest:
        root = overrides.pop("dataset_root", None)
        if root is None:
            root = self._ensure_dataset_root()
        defaults: dict[str, object] = {
            "candidates": (ThresholdCandidate("c", ThresholdCalibrationTarget.TEMPLATE_DETECTION, 0.5),),
            "cases": (self._case("city.collect", True),),
            "dataset_root": root,
            "registry": self._registry(("city.collect",)),
            "analyzer": _RecordingAnalyzer({"city.collect": DetectionResult("city.collect", 0.9)}),
        }
        defaults.update(overrides)
        return ThresholdCalibrationRequest(**defaults)

    def _runner(self, root: Path | None = None) -> ThresholdCalibrationRunner:
        self._ensure_dataset_root()
        if root is not None:
            self.dataset_root = root
        return ThresholdCalibrationRunner()

    def _ensure_dataset_root(self) -> Path:
        if not hasattr(self, "dataset_context"):
            self.dataset_context = self._dataset_root()
            self.dataset_root = self.dataset_context.__enter__()
            self._write_screenshot(self.dataset_root, "screens/one.png")
        return self.dataset_root

    def tearDown(self) -> None:
        context = getattr(self, "dataset_context", None)
        if context is not None:
            context.__exit__(None, None, None)

    @staticmethod
    def _case(
        case_id: str,
        expected: bool,
        *,
        screenshot: str = "screens/one.png",
        template_key: str = "city.collect",
    ) -> ThresholdCalibrationCase:
        return ThresholdCalibrationCase(
            case_id,
            ThresholdCalibrationTarget.TEMPLATE_DETECTION,
            expected,
            screenshot,
            template_key=template_key,
        )

    @staticmethod
    def _observation(case_id: str, expected: bool, actual: bool) -> ThresholdCaseObservation:
        if expected and actual:
            status = ThresholdObservationStatus.TRUE_POSITIVE
        elif expected and not actual:
            status = ThresholdObservationStatus.FALSE_NEGATIVE
        elif not expected and actual:
            status = ThresholdObservationStatus.FALSE_POSITIVE
        else:
            status = ThresholdObservationStatus.TRUE_NEGATIVE
        return ThresholdCaseObservation(
            case_id,
            "c",
            ThresholdCalibrationTarget.TEMPLATE_DETECTION,
            expected,
            actual,
            status,
        )

    @staticmethod
    def _scene_candidate(scene_key: str, score: float):
        from rok_assistant.vision import SceneCandidateResult

        return SceneCandidateResult(scene_key=scene_key, score=score, priority=1)

    @staticmethod
    def _registry(keys: tuple[str, ...]) -> TemplateRegistry:
        templates = tuple(
            TemplateDefinition(
                semantic_key=key,
                template_pack_version="2026.07",
                language="en",
                resolution_profile="phone.720p",
                source=Path(f"templates/{key.replace('.', '_')}.png"),
                region_of_interest=RegionOfInterest(0, 0, 20, 20),
                confidence_threshold=0.95,
                scale_range=ScaleRange(),
                source_reference="synthetic test fixture",
            )
            for key in keys
        )
        return TemplateRegistry(
            TemplatePack(
                version="2026.07",
                languages=("en",),
                resolution_profiles=(ResolutionProfile("phone.720p", 20, 20),),
                templates=templates,
                root=Path("synthetic-pack"),
            )
        )

    @staticmethod
    def _dataset_root() -> _DatasetRootContext:
        return _DatasetRootContext()

    @staticmethod
    def _write_screenshot(root: Path, relative_path: str) -> None:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((20, 20), dtype=np.uint8)
        if not cv2.imwrite(str(path), image):
            raise AssertionError(f"Could not write synthetic screenshot: {relative_path}")


class _DatasetRootContext:
    def __init__(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)

    def __enter__(self) -> Path:
        return self.root

    def __exit__(self, *args: object) -> None:
        self._temp_dir.cleanup()


class _RecordingAnalyzer:
    def __init__(self, results: dict[str, object]) -> None:
        self.results = dict(results)
        self.calls: list[str] = []

    def match(
        self,
        screenshot: object,
        semantic_key: str,
        registry: TemplateRegistry,
        *,
        scene: str | None = None,
    ) -> object:
        self.calls.append(semantic_key)
        result = self.results.get(semantic_key)
        if isinstance(result, BaseException):
            raise result
        return result if result is not None else DetectionResult(None, 0.0, metadata=MatchingMetadata())

    def called_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.calls))


class _RecordingClassifier:
    def __init__(self, results: dict[str, object]) -> None:
        self.results = dict(results)
        self.calls = 0

    def classify(
        self,
        screenshot: object,
        scene_definitions: object,
        registry: TemplateRegistry,
        *,
        analyzer: object = None,
    ) -> object:
        keys = sorted(self.results)
        key = keys[self.calls % len(keys)]
        self.calls += 1
        result = self.results[key]
        if isinstance(result, BaseException):
            raise result
        return result


if __name__ == "__main__":
    unittest.main()
