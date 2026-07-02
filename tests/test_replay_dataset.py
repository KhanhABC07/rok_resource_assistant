from __future__ import annotations

from datetime import UTC, datetime
import json
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
    BoundingBox,
    DetectionResult,
    EvidenceCaptureRequest,
    EvidenceReference,
    FileSystemEvidenceStore,
    MatchingMetadata,
    ReplayCaseDefinition,
    ReplayCaseResult,
    ReplayCaseStatus,
    ReplayDatasetDefinition,
    ReplayDatasetLoader,
    ReplayDetectionExpectation,
    ReplayExpectation,
    ReplayRunRequest,
    ReplayRunResult,
    ReplayRunSummary,
    ReplayRunner,
    ReplaySceneExpectation,
    ResolutionProfile,
    ScaleRange,
    SceneCandidateResult,
    SceneClassificationResult,
    SceneClassificationStatus,
    SceneClassifier,
    SceneDefinition,
    SceneRule,
    TemplateDefinition,
    TemplatePack,
    TemplateRegistry,
    TemplateScreenAnalyzer,
    ValidationDiagnostic,
)


class ReplayDatasetTest(unittest.TestCase):
    def test_valid_dataset_loading_orders_cases_deterministically(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/two.png")
            self._write_screenshot(root, "screens/one.png")
            self._write_manifest(
                root,
                cases=[
                    self._case("z-case", "screens/two.png"),
                    self._case("a-case", "screens/one.png"),
                ],
            )

            result = ReplayDatasetLoader(root).load("replay.json")

            self.assertTrue(result.is_valid, result.diagnostics)
            self.assertEqual("vision.synthetic", result.dataset.dataset_id)
            self.assertEqual(("a-case", "z-case"), tuple(case.case_id for case in result.dataset.cases))

    def test_malformed_json_wrong_type_and_unsupported_schema_return_diagnostics(self) -> None:
        with self._dataset_root() as root:
            (root / "bad.json").write_text("{bad json", encoding="utf-8")
            malformed = ReplayDatasetLoader(root).load("bad.json")
            wrong_type = ReplayDatasetLoader(root).load_mapping([])
            unsupported = ReplayDatasetLoader(root).load_mapping(
                {"schema_version": 2, "dataset_id": "vision.synthetic", "cases": []}
            )

            self.assertDiagnostic(malformed.diagnostics, "replay.manifest.invalid_json")
            self.assertDiagnostic(wrong_type.diagnostics, "replay.manifest.invalid_type")
            self.assertDiagnostic(unsupported.diagnostics, "replay.manifest.unsupported_version")

    def test_unknown_manifest_case_and_expectation_fields_are_rejected(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            manifest = {
                "schema_version": 1,
                "dataset_id": "vision.synthetic",
                "extra": "unsupported",
                "cases": [self._case("case", "screens/one.png")],
            }
            top_level = ReplayDatasetLoader(root).load_mapping(manifest)
            non_string_key = ReplayDatasetLoader(root).load_mapping(
                {
                    "schema_version": 1,
                    "dataset_id": "vision.synthetic",
                    "cases": [self._case("case", "screens/one.png")],
                    99: "unsupported",
                }
            )
            case_level = ReplayDatasetLoader(root).load_mapping(
                {
                    "schema_version": 1,
                    "dataset_id": "vision.synthetic",
                    "cases": [
                        {
                            **self._case("case", "screens/one.png"),
                            "extra": "unsupported",
                            "expectation": {
                                "detection": {
                                    "expected_match": True,
                                    "extra": "unsupported",
                                },
                                "scene": {
                                    "status": "unknown",
                                    "extra": "unsupported",
                                },
                                "extra": "unsupported",
                            },
                        }
                    ],
                }
            )

            self.assertDiagnostic(top_level.diagnostics, "replay.unknown_field")
            self.assertDiagnostic(non_string_key.diagnostics, "replay.unknown_field")
            self.assertTrue(case_level.is_valid, case_level.diagnostics)
            diagnostics = case_level.dataset.cases[0].diagnostics
            self.assertEqual(
                sorted(code for code in (diagnostic.code for diagnostic in diagnostics) if code == "replay.unknown_field"),
                ["replay.unknown_field", "replay.unknown_field", "replay.unknown_field", "replay.unknown_field"],
            )

    def test_empty_dataset_and_no_executable_cases_are_rejected(self) -> None:
        with self._dataset_root() as root:
            empty = ReplayDatasetLoader(root).load_mapping(
                {"schema_version": 1, "dataset_id": "vision.synthetic", "cases": []}
            )
            self._write_screenshot(root, "screens/one.png")
            no_executable = ReplayDatasetLoader(root).load_mapping(
                {
                    "schema_version": 1,
                    "dataset_id": "vision.synthetic",
                    "cases": [{"case_id": "one", "screenshot": "screens/one.png"}],
                }
            )

            self.assertDiagnostic(empty.diagnostics, "replay.dataset.empty")
            self.assertDiagnostic(no_executable.diagnostics, "replay.dataset.no_executable_cases")

    def test_invalid_identifiers_numeric_bounds_and_bounding_boxes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReplayDatasetDefinition("  ", (ReplayCaseDefinition("case", "screens/one.png", "city.collect"),))
        with self.assertRaises(ValueError):
            ReplayCaseDefinition("  ", "screens/one.png", "city.collect")
        with self.assertRaises(ValueError):
            ReplayDetectionExpectation(confidence_min=True)
        with self.assertRaises(ValueError):
            ReplayDetectionExpectation(confidence_min=0.8, confidence_max=0.7)
        with self.assertRaises(ValueError):
            ReplayDetectionExpectation(matched_scale_min=float("inf"))
        with self.assertRaises(ValueError):
            ReplayDetectionExpectation(bounding_box_tolerance=-1)
        with self.assertRaises(ValueError):
            ReplaySceneExpectation(status=SceneClassificationStatus.UNKNOWN, semantic_scene_key="city")

        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            invalid = ReplayDatasetLoader(root).load_mapping(
                {
                    "schema_version": 1,
                    "dataset_id": "vision.synthetic",
                    "cases": [
                        {
                            "case_id": "bad-box",
                            "screenshot": "screens/one.png",
                            "template_key": "city.collect",
                            "expectation": {
                                "detection": {
                                    "bounding_box": {"x": True, "y": 0, "width": 1, "height": 1}
                                }
                            },
                        }
                    ],
                }
            )

            self.assertDiagnostic(invalid.dataset.cases[0].diagnostics, "replay.expectation.invalid_detection")

    def test_duplicate_case_identifier_becomes_invalid_case(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            result = ReplayDatasetLoader(root).load_mapping(
                {
                    "schema_version": 1,
                    "dataset_id": "vision.synthetic",
                    "cases": [
                        self._case("duplicate", "screens/one.png"),
                        self._case("duplicate", "screens/one.png"),
                    ],
                }
            )

            self.assertTrue(result.is_valid, result.diagnostics)
            duplicate = result.dataset.cases[1]
            self.assertDiagnostic(duplicate.diagnostics, "replay.case.duplicate_id")

    def test_missing_screenshot_and_bad_paths_are_case_diagnostics_without_absolute_path_exposure(self) -> None:
        invalid_paths = (
            "screens/missing.png",
            "../outside.png",
            str(Path.cwd()),
            "C:relative.png",
            "\\\\server\\share\\shot.png",
            "\\rooted\\shot.png",
            "\\\\?\\C:\\shot.png",
            "screens\\..\\outside.png",
            "/tmp/shot.png",
            "screens//one.png",
            "screens/CON.png",
            "screens/bad .png",
            "screens/bad.png.",
            "screens/name:stream.png",
        )
        with self._dataset_root() as root:
            for path in invalid_paths:
                with self.subTest(path=path):
                    result = ReplayDatasetLoader(root).load_mapping(
                        {
                            "schema_version": 1,
                            "dataset_id": "vision.synthetic",
                            "cases": [self._case("bad", path)],
                        }
                    )

                    self.assertTrue(result.is_valid, result.diagnostics)
                    text = self._diagnostic_text(result.dataset.cases[0].diagnostics)
                    self.assertNotIn(str(root), text)
                    self.assertTrue(result.dataset.cases[0].diagnostics)

    def test_manifest_escape_directory_screenshot_and_malformed_image_are_structured(self) -> None:
        with self._dataset_root() as root:
            outside_manifest = root.parent / "outside.json"
            outside_manifest.write_text("{}", encoding="utf-8")
            manifest_escape = ReplayDatasetLoader(root).load(outside_manifest)

            directory = root / "screens" / "directory.png"
            directory.mkdir(parents=True)
            directory_case = ReplayDatasetLoader(root).load_mapping(
                {
                    "schema_version": 1,
                    "dataset_id": "vision.synthetic",
                    "cases": [self._case("directory", "screens/directory.png")],
                }
            )

            bad_image = root / "screens" / "bad.png"
            bad_image.parent.mkdir(parents=True, exist_ok=True)
            bad_image.write_bytes(b"not an image")
            malformed_case = ReplayDatasetDefinition(
                "vision.synthetic",
                (
                    ReplayCaseDefinition(
                        "bad-image",
                        "screens/bad.png",
                        "city.collect",
                        ReplayExpectation(detection=ReplayDetectionExpectation(expected_match=True)),
                    ),
                ),
            )
            malformed = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=malformed_case,
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    analyzer=_RecordingAnalyzer({"city.collect": self._match("city.collect", 0.8)}),
                )
            )

            self.assertDiagnostic(manifest_escape.diagnostics, "replay.manifest.unreadable")
            self.assertDiagnostic(directory_case.dataset.cases[0].diagnostics, "replay.screenshot.not_file")
            self.assertEqual(ReplayCaseStatus.INPUT_ERROR, malformed.case_results[0].status)
            self.assertDiagnostic(malformed.case_results[0].diagnostics, "replay.screenshot.invalid_image")
            text = self._diagnostic_text(malformed.case_results[0].diagnostics)
            self.assertNotIn(str(root), text)

    def test_dataset_root_symlink_is_rejected_when_supported(self) -> None:
        with self._dataset_root() as root:
            linked_root = root.parent / "linked-root"
            try:
                os.symlink(root, linked_root, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")

            result = ReplayDatasetLoader(linked_root).load_mapping(
                {
                    "schema_version": 1,
                    "dataset_id": "vision.synthetic",
                    "cases": [self._case("case", "screens/one.png")],
                }
            )

            self.assertDiagnostic(result.dataset.cases[0].diagnostics, "replay.path.invalid")

    def test_symlink_escape_is_rejected_when_supported(self) -> None:
        with self._dataset_root() as root:
            outside = root.parent / "outside.png"
            outside.write_bytes(b"outside")
            link = root / "screens" / "escape.png"
            link.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(outside, link)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")

            result = ReplayDatasetLoader(root).load_mapping(
                {
                    "schema_version": 1,
                    "dataset_id": "vision.synthetic",
                    "cases": [self._case("escape", "screens/escape.png")],
                }
            )

            self.assertDiagnostic(result.dataset.cases[0].diagnostics, "replay.path_escape")

    def test_malformed_expectation_and_case_are_invalid_case_results(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            load = ReplayDatasetLoader(root).load_mapping(
                {
                    "schema_version": 1,
                    "dataset_id": "vision.synthetic",
                    "cases": [
                        {
                            "case_id": "bad-expectation",
                            "screenshot": "screens/one.png",
                            "template_key": "city.collect",
                            "expectation": {"detection": {"confidence_min": float("nan")}},
                        }
                    ],
                }
            )

            result = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=load.dataset,
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    analyzer=_RecordingAnalyzer({"city.collect": self._match("city.collect", 0.9)}),
                )
            )

            self.assertEqual(ReplayCaseStatus.INVALID_CASE, result.case_results[0].status)
            self.assertDiagnostic(result.case_results[0].diagnostics, "replay.expectation.invalid_detection")

    def test_expected_positive_detection_passes_with_real_analyzer(self) -> None:
        with self._real_template_pack() as pack, self._dataset_root() as root:
            template = self._template()
            screenshot = self._screenshot()
            screenshot[5:13, 7:17] = template
            self._write_screenshot(root, "screens/positive.png", screenshot)
            dataset = ReplayDatasetDefinition(
                dataset_id="vision.synthetic",
                cases=(
                    ReplayCaseDefinition(
                        case_id="positive",
                        screenshot_path="screens/positive.png",
                        template_key="city.collect",
                        expectation=ReplayExpectation(
                            detection=ReplayDetectionExpectation(
                                expected_match=True,
                                semantic_key="city.collect",
                                confidence_min=0.95,
                                matched_scale_min=1.0,
                                matched_scale_max=1.0,
                                bounding_box=BoundingBox(7, 5, 10, 8),
                            )
                        ),
                    ),
                ),
            )

            result = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=dataset,
                    dataset_root=root,
                    registry=pack.registry(),
                    analyzer=TemplateScreenAnalyzer(clock=lambda: 1.0),
                )
            )

            self.assertTrue(result.is_valid, result)
            self.assertEqual(ReplayCaseStatus.PASS, result.case_results[0].status)
            self.assertEqual("city.collect", result.case_results[0].detection_result.matched_semantic_key)

    def test_expected_no_match_passes_and_unexpected_outcomes_fail(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            cases = (
                ReplayCaseDefinition(
                    case_id="expected-no-match",
                    screenshot_path="screens/one.png",
                    template_key="city.empty",
                    expectation=ReplayExpectation(
                        detection=ReplayDetectionExpectation(expected_match=False, diagnostic_code="match.below_threshold")
                    ),
                ),
                ReplayCaseDefinition(
                    case_id="unexpected-match",
                    screenshot_path="screens/one.png",
                    template_key="city.collect",
                    expectation=ReplayExpectation(detection=ReplayDetectionExpectation(expected_match=False)),
                ),
                ReplayCaseDefinition(
                    case_id="unexpected-no-match",
                    screenshot_path="screens/one.png",
                    template_key="city.missing",
                    expectation=ReplayExpectation(detection=ReplayDetectionExpectation(expected_match=True)),
                ),
            )
            analyzer = _RecordingAnalyzer(
                {
                    "city.empty": self._no_match(),
                    "city.collect": self._match("city.collect", 0.9),
                    "city.missing": self._no_match(),
                }
            )

            result = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=ReplayDatasetDefinition("vision.synthetic", cases),
                    dataset_root=root,
                    registry=self._registry(("city.empty", "city.collect", "city.missing")),
                    analyzer=analyzer,
                )
            )

            by_id = {case.case_id: case for case in result.case_results}
            self.assertEqual(ReplayCaseStatus.EXPECTATION_MISMATCH, by_id["unexpected-match"].status)
            self.assertEqual(ReplayCaseStatus.EXPECTATION_MISMATCH, by_id["unexpected-no-match"].status)
            self.assertEqual(1, result.summary.passed_cases)
            self.assertEqual(2, result.summary.failed_cases)

    def test_confidence_bounds_scale_and_bounding_box_tolerance(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            detection = self._match("city.collect", 0.75, box=BoundingBox(10, 11, 5, 6), scale=1.25)
            passing = ReplayCaseDefinition(
                "bounds-pass",
                "screens/one.png",
                "city.collect",
                ReplayExpectation(
                    detection=ReplayDetectionExpectation(
                        expected_match=True,
                        confidence_min=0.75,
                        confidence_max=0.75,
                        matched_scale_min=1.25,
                        matched_scale_max=1.25,
                        bounding_box=BoundingBox(11, 12, 5, 6),
                        bounding_box_tolerance=1,
                    )
                ),
            )
            failing = ReplayCaseDefinition(
                "bounds-fail",
                "screens/one.png",
                "city.collect",
                ReplayExpectation(
                    detection=ReplayDetectionExpectation(
                        expected_match=True,
                        confidence_min=0.76,
                        confidence_max=0.8,
                        matched_scale_min=1.5,
                        bounding_box=BoundingBox(20, 20, 5, 6),
                    )
                ),
            )

            result = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=ReplayDatasetDefinition("vision.synthetic", (failing, passing)),
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    analyzer=_RecordingAnalyzer({"city.collect": detection}),
                )
            )

            by_id = {case.case_id: case for case in result.case_results}
            self.assertEqual(ReplayCaseStatus.PASS, by_id["bounds-pass"].status)
            self.assertEqual(ReplayCaseStatus.EXPECTATION_MISMATCH, by_id["bounds-fail"].status)
            self.assertDiagnostic(by_id["bounds-fail"].diagnostics, "replay.detection.confidence_low")
            self.assertDiagnostic(by_id["bounds-fail"].diagnostics, "replay.detection.scale_low")
            self.assertDiagnostic(by_id["bounds-fail"].diagnostics, "replay.detection.bounding_box_mismatch")

    def test_near_boundary_confidence_and_scale_values_are_deterministic(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            detection = self._match("city.collect", 0.75, scale=1.25)
            pass_case = ReplayCaseDefinition(
                "near-pass",
                "screens/one.png",
                "city.collect",
                ReplayExpectation(
                    detection=ReplayDetectionExpectation(
                        expected_match=True,
                        confidence_min=0.7500000000005,
                        confidence_max=0.7500000000005,
                        matched_scale_min=1.2500000000005,
                        matched_scale_max=1.2500000000005,
                    )
                ),
            )
            fail_case = ReplayCaseDefinition(
                "near-fail",
                "screens/one.png",
                "city.collect",
                ReplayExpectation(
                    detection=ReplayDetectionExpectation(
                        expected_match=True,
                        confidence_min=0.750000000002,
                        matched_scale_min=1.250000000002,
                    )
                ),
            )

            result = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=ReplayDatasetDefinition("vision.synthetic", (fail_case, pass_case)),
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    analyzer=_RecordingAnalyzer({"city.collect": detection}),
                )
            )

            by_id = {case.case_id: case for case in result.case_results}
            self.assertEqual(ReplayCaseStatus.PASS, by_id["near-pass"].status)
            self.assertDiagnostic(by_id["near-fail"].diagnostics, "replay.detection.confidence_low")
            self.assertDiagnostic(by_id["near-fail"].diagnostics, "replay.detection.scale_low")

    def test_scene_classified_unknown_ambiguous_and_invalid_expectations(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            scene_results = {
                "classified": SceneClassificationResult(
                    status=SceneClassificationStatus.CLASSIFIED,
                    scene_key="city",
                    score=0.8,
                    candidates=(SceneCandidateResult("city", 0.8, 1),),
                ),
                "unknown": SceneClassificationResult(status=SceneClassificationStatus.UNKNOWN),
                "ambiguous": SceneClassificationResult(
                    status=SceneClassificationStatus.AMBIGUOUS,
                    candidates=(SceneCandidateResult("city", 0.8, 1), SceneCandidateResult("map", 0.8, 1)),
                    diagnostics=(ValidationDiagnostic("scene.ambiguous", "scene", "ambiguous"),),
                ),
                "invalid": SceneClassificationResult(
                    status=SceneClassificationStatus.INVALID,
                    diagnostics=(ValidationDiagnostic("scene.match_failed", "scene", "failed"),),
                ),
            }
            cases = tuple(
                ReplayCaseDefinition(
                    case_id=case_id,
                    screenshot_path="screens/one.png",
                    expectation=ReplayExpectation(
                        scene=ReplaySceneExpectation(
                            status=result.status,
                            semantic_scene_key=result.scene_key,
                            diagnostic_code=(result.diagnostics[0].code if result.diagnostics else None),
                        )
                    ),
                )
                for case_id, result in scene_results.items()
            )

            replay = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=ReplayDatasetDefinition("vision.synthetic", cases),
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    scene_classifier=_RecordingClassifier(scene_results),
                )
            )

            self.assertTrue(replay.is_valid, replay)
            self.assertEqual(4, replay.summary.scene_classification_cases)

    def test_wrong_scene_and_unexpected_status_fail(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            actual = SceneClassificationResult(
                status=SceneClassificationStatus.CLASSIFIED,
                scene_key="map",
                score=0.8,
            )
            case = ReplayCaseDefinition(
                "wrong-scene",
                "screens/one.png",
                expectation=ReplayExpectation(
                    scene=ReplaySceneExpectation(
                        status=SceneClassificationStatus.CLASSIFIED,
                        semantic_scene_key="city",
                        candidate_count=2,
                    )
                ),
            )

            replay = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=ReplayDatasetDefinition("vision.synthetic", (case,)),
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    scene_classifier=_StaticClassifier(actual),
                )
            )

            result = replay.case_results[0]
            self.assertEqual(ReplayCaseStatus.EXPECTATION_MISMATCH, result.status)
            self.assertDiagnostic(result.diagnostics, "replay.scene.key_mismatch")
            self.assertDiagnostic(result.diagnostics, "replay.scene.candidate_count_mismatch")

    def test_malformed_detection_and_scene_results_become_pipeline_errors(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            detection_case = ReplayCaseDefinition(
                "bad-detection",
                "screens/one.png",
                "city.collect",
                ReplayExpectation(detection=ReplayDetectionExpectation(expected_match=True)),
            )
            scene_case = ReplayCaseDefinition(
                "bad-scene",
                "screens/one.png",
                expectation=ReplayExpectation(scene=ReplaySceneExpectation(status=SceneClassificationStatus.UNKNOWN)),
            )
            detection = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=ReplayDatasetDefinition("vision.synthetic", (detection_case,)),
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    analyzer=_MalformedAnalyzer(),
                )
            )
            scene = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=ReplayDatasetDefinition("vision.synthetic", (scene_case,)),
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    scene_classifier=_MalformedClassifier(),
                )
            )

            self.assertEqual(ReplayCaseStatus.PIPELINE_ERROR, detection.case_results[0].status)
            self.assertDiagnostic(detection.case_results[0].diagnostics, "replay.pipeline.detection_failed")
            self.assertEqual(ReplayCaseStatus.PIPELINE_ERROR, scene.case_results[0].status)
            self.assertDiagnostic(scene.case_results[0].diagnostics, "replay.pipeline.scene_failed")

    def test_one_failing_case_does_not_stop_later_cases_and_order_is_deterministic(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            cases = (
                ReplayCaseDefinition(
                    "z-pass",
                    "screens/one.png",
                    "city.collect",
                    ReplayExpectation(detection=ReplayDetectionExpectation(expected_match=True)),
                ),
                ReplayCaseDefinition(
                    "a-fail",
                    "screens/one.png",
                    "city.collect",
                    ReplayExpectation(detection=ReplayDetectionExpectation(confidence_min=1.0)),
                ),
            )

            replay = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=ReplayDatasetDefinition("vision.synthetic", cases),
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    analyzer=_RecordingAnalyzer({"city.collect": self._match("city.collect", 0.8)}),
                )
            )

            self.assertEqual(("a-fail", "z-pass"), tuple(case.case_id for case in replay.case_results))
            self.assertEqual(1, replay.summary.passed_cases)
            self.assertEqual(1, replay.summary.failed_cases)

    def test_repeated_execution_is_identical_and_inputs_are_not_mutated(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            case = ReplayCaseDefinition(
                "stable",
                "screens/one.png",
                "city.collect",
                ReplayExpectation(detection=ReplayDetectionExpectation(expected_match=True)),
            )
            dataset = ReplayDatasetDefinition("vision.synthetic", [case])
            original_cases = dataset.cases
            runner = ReplayRunner()
            request = ReplayRunRequest(
                dataset=dataset,
                dataset_root=root,
                registry=self._registry(("city.collect",)),
                analyzer=_RecordingAnalyzer({"city.collect": self._match("city.collect", 0.8)}),
            )

            first = runner.run(request)
            second = runner.run(request)

            self.assertEqual(first, second)
            self.assertEqual(original_cases, dataset.cases)

    def test_separate_runners_are_isolated_and_registry_reusable_after_failure(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            dataset = ReplayDatasetDefinition(
                "vision.synthetic",
                (
                    ReplayCaseDefinition(
                        "case",
                        "screens/one.png",
                        "city.collect",
                        ReplayExpectation(detection=ReplayDetectionExpectation(expected_match=True)),
                    ),
                ),
            )
            registry = self._registry(("city.collect",))
            failed = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=dataset,
                    dataset_root=root,
                    registry=registry,
                    analyzer=_RaisingAnalyzer(),
                )
            )
            recovered = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=dataset,
                    dataset_root=root,
                    registry=registry,
                    analyzer=_RecordingAnalyzer({"city.collect": self._match("city.collect", 0.9)}),
                )
            )

            self.assertEqual(ReplayCaseStatus.PIPELINE_ERROR, failed.case_results[0].status)
            self.assertEqual(ReplayCaseStatus.PASS, recovered.case_results[0].status)

    def test_analyzer_and_classifier_calls_are_scoped(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            analyzer = _RecordingAnalyzer({"city.collect": self._match("city.collect", 0.8)})
            classifier = _StaticClassifier(SceneClassificationResult(status=SceneClassificationStatus.UNKNOWN))
            dataset = ReplayDatasetDefinition(
                "vision.synthetic",
                (
                    ReplayCaseDefinition(
                        "both",
                        "screens/one.png",
                        "city.collect",
                        ReplayExpectation(
                            detection=ReplayDetectionExpectation(expected_match=True),
                            scene=ReplaySceneExpectation(status=SceneClassificationStatus.UNKNOWN),
                        ),
                    ),
                ),
            )

            ReplayRunner().run(
                ReplayRunRequest(
                    dataset=dataset,
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    analyzer=analyzer,
                    scene_classifier=classifier,
                )
            )

            self.assertEqual(("city.collect",), analyzer.called_keys())
            self.assertEqual(1, classifier.calls)

    def test_invalid_dataset_prevents_calls_and_later_failed_run_does_not_poison_runner(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            analyzer = _RecordingAnalyzer({"city.collect": self._match("city.collect", 0.8)})
            invalid = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=ReplayDatasetDefinition(
                        "vision.synthetic",
                        (
                            ReplayCaseDefinition(
                                "invalid",
                                "screens/one.png",
                                "city.collect",
                                diagnostics=(ValidationDiagnostic("replay.invalid", "case", "invalid"),),
                            ),
                        ),
                        diagnostics=(ValidationDiagnostic("replay.dataset.invalid", "dataset", "invalid"),),
                    ),
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    analyzer=analyzer,
                )
            )
            runner = ReplayRunner()
            failed = runner.run(
                ReplayRunRequest(
                    dataset=ReplayDatasetDefinition(
                        "vision.synthetic",
                        (
                            ReplayCaseDefinition(
                                "failure",
                                "screens/one.png",
                                "city.collect",
                                ReplayExpectation(detection=ReplayDetectionExpectation(expected_match=True)),
                            ),
                        ),
                    ),
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    analyzer=_RaisingAnalyzer(),
                )
            )
            recovered = runner.run(
                ReplayRunRequest(
                    dataset=ReplayDatasetDefinition(
                        "vision.synthetic",
                        (
                            ReplayCaseDefinition(
                                "success",
                                "screens/one.png",
                                "city.collect",
                                ReplayExpectation(detection=ReplayDetectionExpectation(expected_match=True)),
                            ),
                        ),
                    ),
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    analyzer=_RecordingAnalyzer({"city.collect": self._match("city.collect", 0.8)}),
                )
            )

            self.assertFalse(invalid.is_valid)
            self.assertEqual((), analyzer.called_keys())
            self.assertEqual(ReplayCaseStatus.PIPELINE_ERROR, failed.case_results[0].status)
            self.assertTrue(recovered.is_valid)

    def test_no_raw_opencv_or_filesystem_exceptions_leak(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            dataset = ReplayDatasetDefinition(
                "vision.synthetic",
                (
                    ReplayCaseDefinition(
                        "bad",
                        "screens/one.png",
                        "city.collect",
                        ReplayExpectation(detection=ReplayDetectionExpectation(expected_match=True)),
                    ),
                ),
            )

            result = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=dataset,
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    analyzer=_RaisingAnalyzer(),
                )
            )

            self.assertEqual(ReplayCaseStatus.PIPELINE_ERROR, result.case_results[0].status)
            self.assertDiagnostic(result.case_results[0].diagnostics, "replay.pipeline.detection_failed")

    def test_summary_counts_and_invariants(self) -> None:
        diagnostic = ValidationDiagnostic("replay.test", "case", "diagnostic")
        with self.assertRaises(ValueError):
            ReplayRunSummary(0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(ValueError):
            ReplayRunSummary(1, 1, 1, 0, 0, 0, 0)
        with self.assertRaises(ValueError):
            ReplayCaseResult("bad", ReplayCaseStatus.EXPECTATION_MISMATCH)
        with self.assertRaises(ValueError):
            ReplayCaseResult("bad", ReplayCaseStatus.PASS, diagnostics=(diagnostic,))
        with self.assertRaises(ValueError):
            ReplayRunResult(
                dataset_id="vision.synthetic",
                case_results=(ReplayCaseResult("pass", ReplayCaseStatus.PASS),),
                summary=ReplayRunSummary(2, 2, 0, 0, 0, 1, 0),
            )
        passing = ReplayCaseResult("pass", ReplayCaseStatus.PASS)
        failing = ReplayCaseResult("fail", ReplayCaseStatus.EXPECTATION_MISMATCH, diagnostics=(diagnostic,))
        summary = ReplayRunSummary.from_results(
            (
                ReplayCaseDefinition("pass", "screens/one.png", "city.collect"),
                ReplayCaseDefinition("fail", "screens/one.png"),
            ),
            (passing, failing),
        )
        self.assertEqual(2, summary.total_cases)
        self.assertEqual(1, summary.passed_cases)
        self.assertEqual(1, summary.failed_cases)

    def test_evidence_capture_disabled_by_default_and_enabled_on_failure(self) -> None:
        with self._dataset_root() as root, tempfile.TemporaryDirectory() as evidence_dir:
            self._write_screenshot(root, "screens/one.png")
            dataset = ReplayDatasetDefinition(
                "vision.synthetic",
                (
                    ReplayCaseDefinition(
                        "fail",
                        "screens/one.png",
                        "city.collect",
                        ReplayExpectation(detection=ReplayDetectionExpectation(confidence_min=1.0)),
                    ),
                ),
            )
            request = ReplayRunRequest(
                dataset=dataset,
                dataset_root=root,
                registry=self._registry(("city.collect",)),
                analyzer=_RecordingAnalyzer({"city.collect": self._match("city.collect", 0.8)}),
            )
            no_evidence = ReplayRunner().run(request)
            evidence = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=dataset,
                    dataset_root=root,
                    registry=request.registry,
                    analyzer=request.analyzer,
                    evidence_store=FileSystemEvidenceStore(
                        evidence_dir,
                        clock=lambda: datetime(2026, 7, 2, tzinfo=UTC),
                        identifier_factory=lambda: "fixed-id",
                    ),
                    capture_evidence_on=(ReplayCaseStatus.EXPECTATION_MISMATCH,),
                )
            )

            self.assertIsNone(no_evidence.case_results[0].evidence_reference)
            self.assertIsNotNone(evidence.case_results[0].evidence_reference)
            self.assertFalse(Path(evidence.case_results[0].evidence_reference.image_path).is_absolute())

    def test_evidence_failure_becomes_replay_diagnostic_without_corrupting_summary(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            dataset = ReplayDatasetDefinition(
                "vision.synthetic",
                (
                    ReplayCaseDefinition(
                        "fail",
                        "screens/one.png",
                        "city.collect",
                        ReplayExpectation(detection=ReplayDetectionExpectation(confidence_min=1.0)),
                    ),
                ),
            )

            result = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=dataset,
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    analyzer=_RecordingAnalyzer({"city.collect": self._match("city.collect", 0.8)}),
                    evidence_store=_FailingEvidenceStore(),
                    capture_evidence_on=(ReplayCaseStatus.EXPECTATION_MISMATCH,),
                )
            )

            self.assertEqual(1, result.summary.total_cases)
            self.assertDiagnostic(result.case_results[0].diagnostics, "replay.evidence_failed")

    def test_evidence_failure_on_pass_becomes_pipeline_error_and_later_case_continues(self) -> None:
        with self._dataset_root() as root:
            self._write_screenshot(root, "screens/one.png")
            dataset = ReplayDatasetDefinition(
                "vision.synthetic",
                (
                    ReplayCaseDefinition(
                        "a-fail-evidence",
                        "screens/one.png",
                        "city.collect",
                        ReplayExpectation(detection=ReplayDetectionExpectation(expected_match=True)),
                    ),
                    ReplayCaseDefinition(
                        "b-pass-evidence",
                        "screens/one.png",
                        "city.collect",
                        ReplayExpectation(detection=ReplayDetectionExpectation(expected_match=True)),
                    ),
                ),
            )

            result = ReplayRunner().run(
                ReplayRunRequest(
                    dataset=dataset,
                    dataset_root=root,
                    registry=self._registry(("city.collect",)),
                    analyzer=_RecordingAnalyzer({"city.collect": self._match("city.collect", 0.8)}),
                    evidence_store=_FirstFailingEvidenceStore(),
                    capture_evidence_on=(ReplayCaseStatus.PASS,),
                )
            )

            by_id = {case.case_id: case for case in result.case_results}
            self.assertEqual(ReplayCaseStatus.PIPELINE_ERROR, by_id["a-fail-evidence"].status)
            self.assertDiagnostic(by_id["a-fail-evidence"].diagnostics, "replay.evidence_failed")
            self.assertEqual(ReplayCaseStatus.PASS, by_id["b-pass-evidence"].status)
            self.assertIsNotNone(by_id["b-pass-evidence"].evidence_reference)

    def test_diagnostics_are_deterministic_and_do_not_expose_absolute_paths(self) -> None:
        with self._dataset_root() as root:
            request = ReplayRunRequest(
                dataset=ReplayDatasetDefinition(
                    "vision.synthetic",
                    (
                        ReplayCaseDefinition(
                            "bad",
                            "../outside.png",
                            "city.collect",
                            ReplayExpectation(detection=ReplayDetectionExpectation(expected_match=True)),
                            diagnostics=(ValidationDiagnostic("replay.path.invalid", "screenshot", "bad path"),),
                        ),
                    ),
                ),
                dataset_root=root,
                registry=self._registry(("city.collect",)),
                analyzer=_RecordingAnalyzer({}),
            )
            first = ReplayRunner().run(request)
            second = ReplayRunner().run(request)

            self.assertEqual(
                [(item.code, item.field) for item in first.case_results[0].diagnostics],
                [(item.code, item.field) for item in second.case_results[0].diagnostics],
            )
            self.assertNotIn(str(root), self._diagnostic_text(first.case_results[0].diagnostics))

    def assertDiagnostic(self, diagnostics: tuple[ValidationDiagnostic, ...], code: str) -> None:
        self.assertIn(code, {diagnostic.code for diagnostic in diagnostics})

    @staticmethod
    def _diagnostic_text(diagnostics: tuple[ValidationDiagnostic, ...]) -> str:
        return "\n".join(diagnostic.message for diagnostic in diagnostics)

    @staticmethod
    def _case(case_id: str, screenshot: str) -> dict[str, object]:
        return {
            "case_id": case_id,
            "screenshot": screenshot,
            "template_key": "city.collect",
            "expectation": {"detection": {"expected_match": True}},
        }

    @staticmethod
    def _write_manifest(root: Path, *, cases: list[dict[str, object]]) -> None:
        (root / "replay.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset_id": "vision.synthetic",
                    "cases": cases,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _write_screenshot(root: Path, relative_path: str, image: np.ndarray | None = None) -> None:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        image = image if image is not None else np.zeros((20, 20), dtype=np.uint8)
        if not cv2.imwrite(str(path), image):
            raise AssertionError(f"Could not write synthetic screenshot: {relative_path}")

    @staticmethod
    def _screenshot() -> np.ndarray:
        return np.zeros((20, 20), dtype=np.uint8)

    @staticmethod
    def _template() -> np.ndarray:
        template = np.zeros((8, 10), dtype=np.uint8)
        for y in range(template.shape[0]):
            for x in range(template.shape[1]):
                template[y, x] = 40 + ((x * 17 + y * 23) % 200)
        return template

    @staticmethod
    def _match(
        key: str,
        confidence: float,
        *,
        box: BoundingBox | None = None,
        scale: float | None = None,
    ) -> DetectionResult:
        return DetectionResult(
            matched_semantic_key=key,
            confidence=confidence,
            bounding_box=box,
            matched_scale=scale,
        )

    @staticmethod
    def _no_match() -> DetectionResult:
        return DetectionResult(
            matched_semantic_key=None,
            confidence=0.0,
            metadata=MatchingMetadata(
                diagnostics=(ValidationDiagnostic("match.below_threshold", "confidence", "below"),)
            ),
        )

    @staticmethod
    def _registry(keys: tuple[str, ...]) -> TemplateRegistry:
        templates = tuple(
            TemplateDefinition(
                semantic_key=key,
                template_pack_version="2026.07",
                language="en",
                resolution_profile="phone.720p",
                source=Path(f"templates/{key.replace('.', '_')}.png"),
                region_of_interest=BoundingBox(0, 0, 20, 20),
                confidence_threshold=0.9,
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

    def _real_template_pack(self) -> _TemplatePackContext:
        return _TemplatePackContext(self._template())


class _TemplatePackContext:
    def __init__(self, template: np.ndarray) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        self.template = template

    def __enter__(self) -> _TemplatePackContext:
        template_path = self.root / "templates" / "collect.png"
        template_path.parent.mkdir(parents=True)
        if not cv2.imwrite(str(template_path), self.template):
            raise AssertionError("Could not write template")
        (self.root / "template-pack.json").write_text(
            json.dumps(
                {
                    "manifest_version": 1,
                    "version": "2026.07",
                    "languages": ["en"],
                    "resolution_profiles": {"phone.720p": {"width": 20, "height": 20}},
                    "templates": [
                        {
                            "key": "city.collect",
                            "source": "templates/collect.png",
                            "language": "en",
                            "resolution_profile": "phone.720p",
                            "roi": {"x": 0, "y": 0, "width": 20, "height": 20},
                            "threshold": 0.95,
                            "scale_range": {"min": 1.0, "max": 1.0},
                            "source_reference": "synthetic test fixture",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return self

    def __exit__(self, *args: object) -> None:
        self._temp_dir.cleanup()

    def registry(self) -> TemplateRegistry:
        return TemplateRegistry.from_pack_root(self.root)


class _DatasetRootContext:
    def __init__(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)

    def __enter__(self) -> Path:
        return self.root

    def __exit__(self, *args: object) -> None:
        self._temp_dir.cleanup()


class _RecordingAnalyzer:
    def __init__(self, results: dict[str, DetectionResult]) -> None:
        self.results = dict(results)
        self.calls: list[str] = []

    def match(
        self,
        screenshot: object,
        semantic_key: str,
        registry: TemplateRegistry,
        *,
        scene: str | None = None,
    ) -> DetectionResult:
        self.calls.append(semantic_key)
        return self.results.get(semantic_key, ReplayDatasetTest._no_match())

    def called_keys(self) -> tuple[str, ...]:
        return tuple(self.calls)


class _RaisingAnalyzer:
    def match(
        self,
        screenshot: object,
        semantic_key: str,
        registry: TemplateRegistry,
        *,
        scene: str | None = None,
    ) -> DetectionResult:
        raise cv2.error("bad match")


class _MalformedAnalyzer:
    def match(
        self,
        screenshot: object,
        semantic_key: str,
        registry: TemplateRegistry,
        *,
        scene: str | None = None,
    ) -> object:
        return object()


class _StaticClassifier:
    def __init__(self, result: SceneClassificationResult) -> None:
        self.result = result
        self.calls = 0

    def classify(
        self,
        screenshot: object,
        scene_definitions: object,
        registry: TemplateRegistry,
        *,
        analyzer: object = None,
    ) -> SceneClassificationResult:
        self.calls += 1
        return self.result


class _MalformedClassifier:
    def classify(
        self,
        screenshot: object,
        scene_definitions: object,
        registry: TemplateRegistry,
        *,
        analyzer: object = None,
    ) -> object:
        return object()


class _RecordingClassifier:
    def __init__(self, results: dict[str, SceneClassificationResult]) -> None:
        self.results = dict(results)
        self.calls = 0

    def classify(
        self,
        screenshot: object,
        scene_definitions: object,
        registry: TemplateRegistry,
        *,
        analyzer: object = None,
    ) -> SceneClassificationResult:
        keys = sorted(self.results)
        key = keys[self.calls]
        self.calls += 1
        return self.results[key]


class _FailingEvidenceStore:
    def capture(self, request: EvidenceCaptureRequest) -> object:
        class Result:
            is_valid = False
            reference = None
            diagnostics = (ValidationDiagnostic("evidence.write_failed", "evidence", "failed"),)

        return Result()


class _FirstFailingEvidenceStore:
    def __init__(self) -> None:
        self.calls = 0

    def capture(self, request: EvidenceCaptureRequest) -> object:
        self.calls += 1
        if self.calls == 1:
            class FailedResult:
                is_valid = False
                reference = None
                diagnostics = (ValidationDiagnostic("evidence.write_failed", "evidence", "failed"),)

            return FailedResult()

        class PassedResult:
            is_valid = True
            reference = EvidenceReference(
                image_path="screenshots/replay.png",
                metadata_path="screenshots/replay.json",
                content_hash="a" * 64,
            )

        return PassedResult()


if __name__ == "__main__":
    unittest.main()
