from __future__ import annotations

from datetime import UTC, datetime
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import rok_assistant.vision.evidence_capture as evidence_capture
from rok_assistant.vision import (
    BoundingBox,
    DetectionResult,
    EvidenceCaptureRequest,
    EvidenceReference,
    FileSystemEvidenceStore,
    SceneClassificationResult,
    SceneClassificationStatus,
)


class EvidenceCaptureTest(unittest.TestCase):
    def test_successful_full_frame_capture_writes_png_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)

            result = store.capture(EvidenceCaptureRequest(image=self._image()))

            self.assertTrue(result.is_valid, result.diagnostics)
            self.assertFalse(Path(result.reference.image_path).is_absolute())
            self.assertFalse(Path(result.reference.metadata_path).is_absolute())
            image_path = Path(temp_dir) / result.reference.image_path
            metadata_path = Path(temp_dir) / result.reference.metadata_path
            self.assertTrue(image_path.is_file())
            self.assertTrue(metadata_path.is_file())
            metadata = self._read_metadata(temp_dir, result)
            self.assertEqual(1, metadata["schema_version"])
            self.assertEqual("screenshot", metadata["evidence_kind"])
            self.assertEqual(7, metadata["screenshot_width"])
            self.assertEqual(6, metadata["screenshot_height"])
            self.assertEqual(result.reference.content_hash, metadata["content_hash"])

    def test_successful_valid_crop_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)

            result = store.capture(
                EvidenceCaptureRequest(
                    image=self._image(),
                    crop=BoundingBox(2, 1, 3, 4),
                )
            )

            self.assertTrue(result.is_valid, result.diagnostics)
            metadata = self._read_metadata(temp_dir, result)
            self.assertEqual(7, metadata["screenshot_width"])
            self.assertEqual(6, metadata["screenshot_height"])
            self.assertEqual(3, metadata["evidence_width"])
            self.assertEqual(4, metadata["evidence_height"])
            self.assertEqual({"x": 2, "y": 1, "width": 3, "height": 4}, metadata["bounding_box"])
            persisted = cv2.imread(str(Path(temp_dir) / result.reference.image_path), cv2.IMREAD_UNCHANGED)
            self.assertEqual((4, 3, 3), persisted.shape)

    def test_exact_right_and_bottom_crop_boundary_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._store(temp_dir).capture(
                EvidenceCaptureRequest(
                    image=self._image(),
                    crop=BoundingBox(4, 2, 3, 4),
                )
            )

            self.assertTrue(result.is_valid, result.diagnostics)
            metadata = self._read_metadata(temp_dir, result)
            self.assertEqual(3, metadata["evidence_width"])
            self.assertEqual(4, metadata["evidence_height"])

    def test_caller_owned_image_is_not_mutated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image = self._image()
            original = image.copy()

            result = self._store(temp_dir).capture(EvidenceCaptureRequest(image=image))

            self.assertTrue(result.is_valid, result.diagnostics)
            self.assertTrue(np.array_equal(original, image))

    def test_metadata_serialization_is_deterministic_with_injected_clock_and_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)

            result = store.capture(
                EvidenceCaptureRequest(
                    image=self._image(),
                    evidence_kind="failure",
                    correlation_id="run-42",
                )
            )

            self.assertTrue(result.is_valid, result.diagnostics)
            self.assertIn("2026-07-02T01-02-03Z-failure-fixed-id", result.reference.image_path)
            raw = (Path(temp_dir) / result.reference.metadata_path).read_text(encoding="utf-8")
            self.assertEqual(raw, json.dumps(json.loads(raw), sort_keys=True, indent=2) + "\n")
            self.assertEqual("run-42", json.loads(raw)["correlation_id"])

    def test_content_hash_is_for_exact_persisted_png_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._store(temp_dir).capture(EvidenceCaptureRequest(image=self._image()))

            self.assertTrue(result.is_valid, result.diagnostics)
            image_bytes = (Path(temp_dir) / result.reference.image_path).read_bytes()
            self.assertEqual(evidence_capture.hashlib.sha256(image_bytes).hexdigest(), result.reference.content_hash)

    def test_collision_safe_repeated_captures_do_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)

            first = store.capture(EvidenceCaptureRequest(image=self._image()))
            second = store.capture(EvidenceCaptureRequest(image=self._image()))

            self.assertTrue(first.is_valid, first.diagnostics)
            self.assertTrue(second.is_valid, second.diagnostics)
            self.assertNotEqual(first.reference.image_path, second.reference.image_path)
            self.assertTrue(second.reference.image_path.endswith("-001.png"))

    def test_existing_destination_is_not_silently_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            directory = root / "screenshots"
            directory.mkdir()
            base = directory / "2026-07-02T01-02-03Z-screenshot-fixed-id"
            (base.with_suffix(".png")).write_bytes(b"existing")
            (base.with_suffix(".json")).write_text("existing", encoding="utf-8")

            result = self._store(root).capture(EvidenceCaptureRequest(image=self._image()))

            self.assertTrue(result.is_valid, result.diagnostics)
            self.assertTrue(result.reference.image_path.endswith("-001.png"))
            self.assertEqual(b"existing", base.with_suffix(".png").read_bytes())
            self.assertEqual("existing", base.with_suffix(".json").read_text(encoding="utf-8"))

    def test_preexisting_image_only_and_metadata_only_destinations_are_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            directory = root / "screenshots"
            directory.mkdir()
            base = directory / "2026-07-02T01-02-03Z-screenshot-fixed-id"
            (base.with_suffix(".png")).write_bytes(b"existing-image")
            (directory / "2026-07-02T01-02-03Z-screenshot-fixed-id-001.json").write_text(
                "existing-metadata",
                encoding="utf-8",
            )

            result = self._store(root).capture(EvidenceCaptureRequest(image=self._image()))

            self.assertTrue(result.is_valid, result.diagnostics)
            self.assertTrue(result.reference.image_path.endswith("-002.png"))
            self.assertEqual(b"existing-image", base.with_suffix(".png").read_bytes())

    def test_collision_exhaustion_returns_structured_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            directory = root / "screenshots"
            directory.mkdir()
            base = directory / "2026-07-02T01-02-03Z-screenshot-fixed-id"
            base.with_suffix(".png").write_bytes(b"existing")
            (directory / "2026-07-02T01-02-03Z-screenshot-fixed-id-001.png").write_bytes(b"existing")

            with patch.object(evidence_capture, "_MAX_COLLISION_ATTEMPTS", 2):
                result = self._store(root).capture(EvidenceCaptureRequest(image=self._image()))

            self.assertFalse(result.is_valid)
            self.assertDiagnostic(result, "evidence.collision_exhausted")

    def test_preexisting_temp_filename_returns_structured_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            directory = root / "screenshots"
            directory.mkdir()
            temp_file = directory / ".2026-07-02T01-02-03Z-screenshot-fixed-id.png.tmp"
            temp_file.write_bytes(b"existing-temp")

            result = self._store(root).capture(EvidenceCaptureRequest(image=self._image()))

            self.assertFalse(result.is_valid)
            self.assertDiagnostic(result, "evidence.temp_collision")
            self.assertEqual(b"existing-temp", temp_file.read_bytes())

    def test_reference_rejects_absolute_paths(self) -> None:
        with self.assertRaises(ValueError):
            EvidenceReference(
                image_path=str(Path.cwd() / "evidence.png"),
                metadata_path="screenshots/evidence.json",
                content_hash="0" * 64,
            )

    def test_results_and_diagnostics_do_not_expose_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._store(temp_dir).capture(
                EvidenceCaptureRequest(
                    image=np.zeros((1, 1), dtype=np.float32),
                    relative_directory="../escape",
                )
            )

            combined = "\n".join(
                (
                    *(diagnostic.message for diagnostic in result.diagnostics),
                    result.reference.image_path if result.reference else "",
                    result.reference.metadata_path if result.reference else "",
                )
            )
            self.assertNotIn(str(temp_dir), combined)

    def test_parent_absolute_drive_relative_and_unc_paths_are_rejected(self) -> None:
        invalid_paths = (
            "../outside",
            str(Path.cwd()),
            "C:relative",
            "\\\\server\\share\\evidence",
            "screenshots\\..\\outside",
            "\\rooted",
            "\\\\?\\C:\\evidence",
            "screenshots//bad",
            "screenshots/trailing. ",
            "CON",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for relative_directory in invalid_paths:
                with self.subTest(relative_directory=relative_directory):
                    result = self._store(temp_dir).capture(
                        EvidenceCaptureRequest(
                            image=self._image(),
                            relative_directory=relative_directory,
                        )
                    )

                    self.assertFalse(result.is_valid)
                    self.assertDiagnostic(result, "evidence.invalid_relative_path")

    def test_symlink_escape_is_rejected_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "root"
            outside = Path(temp_dir) / "outside"
            root.mkdir()
            outside.mkdir()
            link = root / "link"
            try:
                os.symlink(outside, link, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")

            result = self._store(root).capture(
                EvidenceCaptureRequest(image=self._image(), relative_directory="link")
            )

            self.assertFalse(result.is_valid)
            self.assertDiagnostic(result, "evidence.path_escape")

    def test_symlink_evidence_root_is_rejected_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "target"
            link = Path(temp_dir) / "root-link"
            target.mkdir()
            try:
                os.symlink(target, link, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")

            result = self._store(link).capture(EvidenceCaptureRequest(image=self._image()))

            self.assertFalse(result.is_valid)
            self.assertDiagnostic(result, "evidence.path_escape")

    def test_invalid_crop_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._store(temp_dir).capture(
                EvidenceCaptureRequest(
                    image=self._image(),
                    crop=BoundingBox(5, 4, 3, 3),
                )
            )

            self.assertFalse(result.is_valid)
            self.assertDiagnostic(result, "evidence.invalid_crop")

    def test_contradictory_crop_and_detection_bounding_box_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            detection = DetectionResult(
                matched_semantic_key="city.collect.food.ready",
                confidence=0.9,
                bounding_box=BoundingBox(1, 1, 2, 2),
            )

            result = self._store(temp_dir).capture(
                EvidenceCaptureRequest(
                    image=self._image(),
                    crop=BoundingBox(2, 1, 3, 4),
                    detection_result=detection,
                )
            )

            self.assertFalse(result.is_valid)
            self.assertDiagnostic(result, "evidence.contradictory_context")

    def test_malformed_crop_object_returns_structured_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            request = EvidenceCaptureRequest(image=self._image())
            object.__setattr__(request, "crop", object())

            result = self._store(temp_dir).capture(request)

            self.assertFalse(result.is_valid)
            self.assertDiagnostic(result, "evidence.invalid_crop")

    def test_empty_or_malformed_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for image in (np.array([], dtype=np.uint8), Path(temp_dir) / "missing.png"):
                with self.subTest(image=repr(image)):
                    result = self._store(temp_dir).capture(EvidenceCaptureRequest(image=image))

                    self.assertFalse(result.is_valid)
                    self.assertDiagnostic(result, "evidence.invalid_image")

    def test_unsupported_dimensions_channels_and_dtype_are_rejected(self) -> None:
        images = (
            np.zeros((2, 2, 2, 2), dtype=np.uint8),
            np.zeros((2, 2, 2), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=bool),
            np.array([[object()]], dtype=object),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for image in images:
                with self.subTest(shape=image.shape, dtype=image.dtype):
                    result = self._store(temp_dir).capture(EvidenceCaptureRequest(image=image))

                    self.assertFalse(result.is_valid)
                    self.assertDiagnostic(result, "evidence.invalid_image")

    def test_image_encoding_failure_returns_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("cv2.imencode", return_value=(False, None)):
                result = self._store(temp_dir).capture(EvidenceCaptureRequest(image=self._image()))

            self.assertFalse(result.is_valid)
            self.assertDiagnostic(result, "evidence.encode_failed")

    def test_metadata_serialization_failure_returns_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(evidence_capture.json, "dumps", side_effect=TypeError("bad json")):
                result = self._store(temp_dir).capture(EvidenceCaptureRequest(image=self._image()))

            self.assertFalse(result.is_valid)
            self.assertDiagnostic(result, "evidence.metadata_serialization_failed")

    def test_filesystem_write_failure_returns_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(FileSystemEvidenceStore, "_write_temp", side_effect=PermissionError("denied")):
                result = self._store(temp_dir).capture(EvidenceCaptureRequest(image=self._image()))

            self.assertFalse(result.is_valid)
            self.assertDiagnostic(result, "evidence.write_failed")

    def test_failed_capture_followed_by_successful_capture_reuses_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            with patch.object(FileSystemEvidenceStore, "_write_temp", side_effect=PermissionError("denied")):
                failed = store.capture(EvidenceCaptureRequest(image=self._image()))

            recovered = store.capture(EvidenceCaptureRequest(image=self._image()))

            self.assertFalse(failed.is_valid)
            self.assertTrue(recovered.is_valid, recovered.diagnostics)

    def test_atomic_cleanup_after_metadata_commit_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _FailingMetadataCommitStore(
                temp_dir,
                clock=self._clock,
                identifier_factory=lambda: "fixed-id",
            )

            result = store.capture(EvidenceCaptureRequest(image=self._image()))

            self.assertFalse(result.is_valid)
            self.assertDiagnostic(result, "evidence.publish_failed")
            screenshots = Path(temp_dir) / "screenshots"
            self.assertFalse(any(screenshots.glob("*.png")))
            self.assertFalse(any(screenshots.glob("*.json")))
            self.assertFalse(any(screenshots.glob("*.tmp")))

    def test_rollback_failure_returns_structured_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _FailingMetadataCommitStore(
                temp_dir,
                clock=self._clock,
                identifier_factory=lambda: "fixed-id",
            )

            with patch.object(evidence_capture, "_unlink_if_exists", return_value=False):
                result = store.capture(EvidenceCaptureRequest(image=self._image()))

            self.assertFalse(result.is_valid)
            self.assertDiagnostic(result, "evidence.publish_failed")
            self.assertDiagnostic(result, "evidence.rollback_failed")

    def test_separate_store_instances_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = self._store(first_dir).capture(EvidenceCaptureRequest(image=self._image()))
            second = self._store(second_dir).capture(EvidenceCaptureRequest(image=self._image()))

            self.assertTrue(first.is_valid, first.diagnostics)
            self.assertTrue(second.is_valid, second.diagnostics)
            self.assertTrue((Path(first_dir) / first.reference.image_path).is_file())
            self.assertTrue((Path(second_dir) / second.reference.image_path).is_file())

    def test_two_store_instances_sharing_one_root_do_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first_store = self._store(temp_dir)
            second_store = self._store(temp_dir)

            first = first_store.capture(EvidenceCaptureRequest(image=self._image()))
            second = second_store.capture(EvidenceCaptureRequest(image=self._image()))

            self.assertTrue(first.is_valid, first.diagnostics)
            self.assertTrue(second.is_valid, second.diagnostics)
            self.assertNotEqual(first.reference.image_path, second.reference.image_path)

    def test_repeated_store_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FileSystemEvidenceStore(
                temp_dir,
                clock=self._clock,
                identifier_factory=_IncrementingIdentifier(),
            )

            first = store.capture(EvidenceCaptureRequest(image=self._image()))
            second = store.capture(EvidenceCaptureRequest(image=self._image()))

            self.assertTrue(first.is_valid, first.diagnostics)
            self.assertTrue(second.is_valid, second.diagnostics)
            self.assertNotEqual(first.reference.image_path, second.reference.image_path)

    def test_detection_result_metadata_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            detection = DetectionResult(
                matched_semantic_key="city.collect.food.ready",
                confidence=0.91,
                bounding_box=BoundingBox(1, 2, 3, 4),
                matched_scale=1.25,
                scene="city",
                template_pack_version="2026.07",
            )

            result = self._store(temp_dir).capture(
                EvidenceCaptureRequest(image=self._image(), detection_result=detection)
            )

            self.assertTrue(result.is_valid, result.diagnostics)
            metadata = self._read_metadata(temp_dir, result)
            self.assertEqual("city.collect.food.ready", metadata["semantic_template_key"])
            self.assertEqual("city", metadata["semantic_scene_key"])
            self.assertEqual(0.91, metadata["detection_confidence"])
            self.assertEqual(1.25, metadata["matched_scale"])
            self.assertEqual("2026.07", metadata["template_pack_version"])
            self.assertEqual({"x": 1, "y": 2, "width": 3, "height": 4}, metadata["bounding_box"])

    def test_scene_classification_result_metadata_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scene = SceneClassificationResult(
                status=SceneClassificationStatus.CLASSIFIED,
                scene_key="city",
                score=0.84,
            )

            result = self._store(temp_dir).capture(
                EvidenceCaptureRequest(image=self._image(), scene_result=scene)
            )

            self.assertTrue(result.is_valid, result.diagnostics)
            metadata = self._read_metadata(temp_dir, result)
            self.assertEqual("city", metadata["semantic_scene_key"])
            self.assertEqual("classified", metadata["classification_status"])
            self.assertEqual(0.84, metadata["classification_score"])

    def test_invalid_request_text_and_identifier_inputs_return_structured_diagnostics(self) -> None:
        cases = (
            EvidenceCaptureRequest(image=self._image(), evidence_kind=" "),
            EvidenceCaptureRequest(image=self._image(), semantic_template_key=" "),
            EvidenceCaptureRequest(image=self._image(), semantic_scene_key=" "),
            EvidenceCaptureRequest(image=self._image(), correlation_id=" "),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for request in cases:
                with self.subTest(request=request):
                    result = self._store(temp_dir).capture(request)

                    self.assertFalse(result.is_valid)
                    self.assertDiagnostic(result, "evidence.invalid_metadata")

    def test_identifier_containing_separators_is_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FileSystemEvidenceStore(
                temp_dir,
                clock=self._clock,
                identifier_factory=lambda: "../bad\\id",
            )

            result = store.capture(EvidenceCaptureRequest(image=self._image()))

            self.assertTrue(result.is_valid, result.diagnostics)
            self.assertIn("bad-id", result.reference.image_path)
            self.assertNotIn("..", result.reference.image_path)
            self.assertNotIn("\\", result.reference.image_path)

    def test_non_string_identifier_and_timezone_naive_clock_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_identifier = FileSystemEvidenceStore(
                temp_dir,
                clock=self._clock,
                identifier_factory=lambda: 123,
            ).capture(EvidenceCaptureRequest(image=self._image()))
            naive_clock = FileSystemEvidenceStore(
                temp_dir,
                clock=lambda: datetime(2026, 7, 2, 1, 2, 3),
                identifier_factory=lambda: "fixed-id",
            ).capture(EvidenceCaptureRequest(image=self._image()))

            self.assertFalse(bad_identifier.is_valid)
            self.assertDiagnostic(bad_identifier, "evidence.invalid_identifier")
            self.assertFalse(naive_clock.is_valid)
            self.assertDiagnostic(naive_clock, "evidence.invalid_timestamp")

    def test_malformed_context_returns_structured_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            request = EvidenceCaptureRequest(image=self._image())
            object.__setattr__(request, "detection_result", object())
            object.__setattr__(request, "scene_result", object())

            result = self._store(temp_dir).capture(request)

            self.assertFalse(result.is_valid)
            self.assertEqual(
                ("evidence.invalid_context", "evidence.invalid_context"),
                tuple(diagnostic.code for diagnostic in result.diagnostics),
            )

    def test_malformed_typed_context_returns_metadata_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            detection = object.__new__(DetectionResult)
            object.__setattr__(detection, "matched_semantic_key", "city.collect.food.ready")
            scene = object.__new__(SceneClassificationResult)
            object.__setattr__(scene, "status", "classified")
            request = EvidenceCaptureRequest(
                image=self._image(),
                detection_result=detection,
                scene_result=scene,
            )

            result = self._store(temp_dir).capture(request)

            self.assertFalse(result.is_valid)
            self.assertDiagnostic(result, "evidence.invalid_metadata")

    def test_evidence_result_and_reference_invariants_reject_contradictions(self) -> None:
        from rok_assistant.vision import EvidenceCaptureResult

        with self.assertRaises(ValueError):
            EvidenceCaptureResult()
        with self.assertRaises(ValueError):
            EvidenceReference(
                image_path="../bad.png",
                metadata_path="screenshots/evidence.json",
                content_hash="0" * 64,
            )

    def test_stable_diagnostic_codes_and_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            request = EvidenceCaptureRequest(image=np.array([], dtype=np.uint8), relative_directory="../bad")
            first = self._store(temp_dir).capture(request)
            second = self._store(temp_dir).capture(request)

            self.assertEqual(
                [(item.code, item.field) for item in first.diagnostics],
                [(item.code, item.field) for item in second.diagnostics],
            )

    def test_diagnostics_never_expose_evidence_root_or_temp_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._store(temp_dir).capture(
                EvidenceCaptureRequest(image=np.zeros((2, 2), dtype=np.float32))
            )

            text = "\n".join(diagnostic.message for diagnostic in result.diagnostics)
            self.assertNotIn(temp_dir, text)
            self.assertNotIn(str(Path(temp_dir).drive), text)

    def assertDiagnostic(self, result: object, code: str) -> None:
        self.assertIn(code, {diagnostic.code for diagnostic in result.diagnostics})

    @staticmethod
    def _read_metadata(root: str | Path, result: object) -> dict[str, object]:
        return json.loads((Path(root) / result.reference.metadata_path).read_text(encoding="utf-8"))

    @staticmethod
    def _image() -> np.ndarray:
        image = np.zeros((6, 7, 3), dtype=np.uint8)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                image[y, x] = [(x * 17) % 255, (y * 31) % 255, ((x + y) * 11) % 255]
        return image

    @staticmethod
    def _clock() -> datetime:
        return datetime(2026, 7, 2, 1, 2, 3, tzinfo=UTC)

    def _store(self, root: str | Path) -> FileSystemEvidenceStore:
        return FileSystemEvidenceStore(
            root,
            clock=self._clock,
            identifier_factory=lambda: "fixed-id",
        )


class _IncrementingIdentifier:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return f"id-{self.value}"


class _FailingMetadataCommitStore(FileSystemEvidenceStore):
    def _publish(self, source: Path, destination: Path) -> None:
        if destination.suffix == ".json":
            raise PermissionError("metadata write denied")
        super()._publish(source, destination)


if __name__ == "__main__":
    unittest.main()
