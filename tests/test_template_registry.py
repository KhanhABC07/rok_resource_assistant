from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rok_assistant.vision import (
    BoundingBox,
    DetectionResult,
    MatchingMetadata,
    SceneConstraints,
    TemplatePack,
    TemplateNotFoundError,
    TemplatePackValidationError,
    TemplateRegistry,
    validate_template_pack,
)


class TemplateRegistryTest(unittest.TestCase):
    def test_valid_template_pack_loading_and_semantic_lookup(self) -> None:
        with self._template_pack() as root:
            registry = TemplateRegistry.from_pack_root(root)

            definition = registry.get("city.collect.food.ready")

            self.assertEqual("city.collect.food.ready", definition.semantic_key)
            self.assertEqual("2026.07", definition.template_pack_version)
            self.assertEqual("en", definition.language)
            self.assertEqual("phone.720p", definition.resolution_profile)
            self.assertEqual(0.87, definition.confidence_threshold)
            self.assertEqual(0.75, definition.scale_range.minimum)
            self.assertEqual(1.25, definition.scale_range.maximum)
            self.assertEqual(("city", "home"), definition.scene_constraints.allowed)
            self.assertEqual(("home",), definition.scene_constraints.required)

    def test_malformed_json_returns_structured_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "template-pack.json").write_text("{bad json", encoding="utf-8")

            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "manifest.invalid_json")
            self.assertNotIn(str(root), report.diagnostics[0].message)

    def test_wrong_manifest_root_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "template-pack.json").write_text("[]", encoding="utf-8")

            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "manifest.invalid_type")

    def test_missing_manifest_returns_structured_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "manifest.unreadable")
            self.assertNotIn(str(root), report.diagnostics[0].message)

    def test_unsupported_manifest_version_is_rejected(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["manifest_version"] = 2

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "manifest.unsupported_version")

    def test_duplicate_semantic_keys_are_rejected(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"].append(dict(manifest["templates"][0]))

        with self._template_pack(mutate=mutate) as root:
            with self.assertRaises(TemplatePackValidationError) as caught:
                TemplateRegistry.from_pack_root(root)

            self.assertDiagnostic(caught.exception.diagnostics, "template.duplicate_key")

    def test_empty_template_pack_is_rejected(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"] = []

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "templates.invalid_type")

    def test_empty_semantic_key_is_rejected(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"][0]["key"] = ""

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "missing_key")

    def test_missing_source_reference_is_structured_diagnostic(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"][0].pop("source")

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "template.missing_source")

    def test_missing_template_file_is_structured_diagnostic(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"][0]["source"] = "templates/missing.png"

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "template.missing_file")

    def test_absolute_source_path_is_rejected(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"][0]["source"] = str(Path.cwd() / "template.png")

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "template.invalid_source")
            self.assertNotIn(str(Path.cwd()), self._diagnostic_messages(report.diagnostics))

    def test_parent_directory_traversal_is_rejected(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"][0]["source"] = "../outside.png"

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "template.invalid_source")

    def test_mixed_windows_separator_traversal_is_rejected(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"][0]["source"] = "templates\\..\\outside.png"

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "template.invalid_source")

    def test_windows_drive_relative_and_unc_paths_are_rejected(self) -> None:
        for source in ("C:templates\\template.png", "\\\\server\\share\\template.png"):
            with self.subTest(source=source):
                def mutate(manifest: dict[str, Any]) -> None:
                    manifest["templates"][0]["source"] = source

                with self._template_pack(mutate=mutate) as root:
                    report = validate_template_pack(root)

                    self.assertFalse(report.is_valid)
                    self.assertDiagnostic(report.diagnostics, "template.invalid_source")

    def test_symlink_escape_is_rejected_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "pack"
            outside = Path(temp_dir) / "outside.png"
            root.mkdir()
            outside.write_bytes(b"outside")
            link = root / "templates" / "escape.png"
            link.parent.mkdir()
            try:
                os.symlink(outside, link)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")
            self._write_manifest(root, source="templates/escape.png")

            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "template.invalid_source")

    def test_threshold_boundaries_are_inclusive(self) -> None:
        for threshold in (0.0, 1.0):
            with self.subTest(threshold=threshold):
                def mutate(manifest: dict[str, Any]) -> None:
                    manifest["templates"][0]["threshold"] = threshold

                with self._template_pack(mutate=mutate) as root:
                    report = validate_template_pack(root)

                    self.assertTrue(report.is_valid, report.diagnostics)

    def test_invalid_threshold_boundaries_are_rejected(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"][0]["threshold"] = 1.01

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "template.invalid_threshold")

    def test_nan_and_infinity_are_rejected(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"][0]["threshold"] = float("nan")
            manifest["templates"][0]["scale_range"] = {"min": 1.0, "max": float("inf")}

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "template.invalid_threshold")
            self.assertDiagnostic(report.diagnostics, "template.invalid_scale_range")

    def test_invalid_roi_boundaries_are_rejected(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"][0]["roi"] = {
                "x": 1270,
                "y": 700,
                "width": 20,
                "height": 30,
            }

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "template.invalid_roi")

    def test_zero_area_roi_is_rejected(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"][0]["roi"]["width"] = 0

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "template.invalid_roi")

    def test_invalid_scale_range_is_rejected(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"][0]["scale_range"] = {"min": 1.5, "max": 0.75}

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "template.invalid_scale_range")

    def test_optional_mask_reference_is_validated(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"][0]["mask"] = "masks/missing-mask.png"

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "template.invalid_mask")

    def test_source_and_mask_paths_are_confined_to_pack_root(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"][0]["mask"] = "masks/city_collect_food_ready_mask.png"

        with self._template_pack(
            extra_files=("masks/city_collect_food_ready_mask.png",),
            mutate=mutate,
        ) as root:
            registry = TemplateRegistry.from_pack_root(root)
            definition = registry.get("city.collect.food.ready")

            self.assertEqual(
                root.resolve() / "templates" / "city_collect_food_ready.png",
                registry.resolve_template_path(definition),
            )
            self.assertEqual(
                root.resolve() / "masks" / "city_collect_food_ready_mask.png",
                registry.resolve_mask_path(definition),
            )

    def test_malformed_scene_constraints_are_rejected(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"][0]["scene_constraints"] = {"allowed": "city"}

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "template.invalid_scene_constraints")

    def test_contradictory_scene_constraints_are_rejected(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"][0]["scene_constraints"] = {
                "allowed": ["city"],
                "required": ["home"],
            }

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "template.contradictory_scene_constraints")

    def test_unsupported_language_and_resolution_profile_are_rejected(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["templates"][0]["language"] = "fr"
            manifest["templates"][0]["resolution_profile"] = "tablet.1440p"

        with self._template_pack(mutate=mutate) as root:
            report = validate_template_pack(root)

            self.assertFalse(report.is_valid)
            self.assertDiagnostic(report.diagnostics, "template.unsupported_language")
            self.assertDiagnostic(
                report.diagnostics,
                "template.unsupported_resolution_profile",
            )

    def test_registry_ordering_is_deterministic_by_semantic_key(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            second = dict(manifest["templates"][0])
            second["key"] = "alliance.help.ready"
            second["source"] = "templates/alliance_help_ready.png"
            manifest["templates"].append(second)

        with self._template_pack(extra_files=("templates/alliance_help_ready.png",), mutate=mutate) as root:
            registry = TemplateRegistry.from_pack_root(root)

            self.assertEqual(
                ("alliance.help.ready", "city.collect.food.ready"),
                registry.keys(),
            )

    def test_missing_key_raises_structured_registry_error(self) -> None:
        with self._template_pack() as root:
            registry = TemplateRegistry.from_pack_root(root)

            with self.assertRaises(TemplateNotFoundError) as caught:
                registry.get("city.collect.wood.ready")

            self.assertDiagnostic(caught.exception.diagnostics, "template.missing_key")

    def test_validation_failures_have_deterministic_order(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["manifest_version"] = 2
            manifest["templates"][0]["threshold"] = -0.1
            manifest["templates"].append(dict(manifest["templates"][0]))

        with self._template_pack(mutate=mutate) as root:
            first = validate_template_pack(root)
            second = validate_template_pack(root)

            self.assertEqual(
                [(item.code, item.field) for item in first.diagnostics],
                [(item.code, item.field) for item in second.diagnostics],
            )

    def test_registry_instances_are_isolated(self) -> None:
        with self._template_pack() as first_root, self._template_pack(
            key="alliance.help.ready",
            source="templates/alliance_help_ready.png",
        ) as second_root:
            first = TemplateRegistry.from_pack_root(first_root)
            second = TemplateRegistry.from_pack_root(second_root)

            self.assertEqual(("city.collect.food.ready",), first.keys())
            self.assertEqual(("alliance.help.ready",), second.keys())
            with self.assertRaises(TemplateNotFoundError):
                first.get("alliance.help.ready")

    def test_paths_are_relative_and_loading_does_not_read_image_bytes(self) -> None:
        with self._template_pack(image_bytes=b"not a real image") as root:
            with patch.object(Path, "read_bytes", side_effect=AssertionError):
                registry = TemplateRegistry.from_pack_root(root)
            definition = registry.get("city.collect.food.ready")

            self.assertFalse(definition.source.is_absolute())
            self.assertEqual(
                root.resolve() / "templates" / "city_collect_food_ready.png",
                registry.resolve_template_path(definition),
            )

    def test_model_invariants_reject_invalid_bounding_box_and_matched_scale(self) -> None:
        with self.assertRaises(ValueError):
            BoundingBox(x=0, y=0, width=0, height=1)
        with self.assertRaises(ValueError):
            DetectionResult(matched_semantic_key="city.collect.food.ready", confidence=0.5, matched_scale=0.0)
        with self.assertRaises(ValueError):
            DetectionResult(matched_semantic_key="city.collect.food.ready", confidence=float("inf"))

    def test_collection_state_is_defensively_copied(self) -> None:
        allowed = ["city", "home"]
        required = ["home"]
        constraints = SceneConstraints(allowed=allowed, required=required)
        allowed.append("mutated")
        required.clear()

        self.assertEqual(("city", "home"), constraints.allowed)
        self.assertEqual(("home",), constraints.required)

        diagnostics = [self._diagnostic("test.code")]
        metadata = MatchingMetadata(diagnostics=diagnostics)
        diagnostics.append(self._diagnostic("test.other"))

        self.assertEqual(("test.code",), tuple(item.code for item in metadata.diagnostics))

        pack = TemplatePack(
            version="2026.07",
            languages=["en"],
            resolution_profiles=[],
            templates=[],
            root=Path("pack"),
        )
        self.assertEqual(("en",), pack.languages)

    def test_repeated_equivalent_pack_loading(self) -> None:
        with self._template_pack() as root:
            first = TemplateRegistry.from_pack_root(root)
            second = TemplateRegistry.from_pack_root(root)

            self.assertEqual(first.keys(), second.keys())
            self.assertEqual(first.templates(), second.templates())

    def assertDiagnostic(
        self,
        diagnostics: tuple[Any, ...],
        code: str,
    ) -> None:
        self.assertIn(code, {diagnostic.code for diagnostic in diagnostics})

    @staticmethod
    def _diagnostic(code: str) -> Any:
        from rok_assistant.vision import ValidationDiagnostic

        return ValidationDiagnostic(code=code, message="test diagnostic")

    @staticmethod
    def _diagnostic_messages(diagnostics: tuple[Any, ...]) -> str:
        return "\n".join(diagnostic.message for diagnostic in diagnostics)

    @staticmethod
    def _template_pack(
        *,
        key: str = "city.collect.food.ready",
        source: str = "templates/city_collect_food_ready.png",
        image_bytes: bytes = b"template-placeholder",
        extra_files: tuple[str, ...] = (),
        mutate: Callable[[dict[str, Any]], None] | None = None,
    ) -> _TemplatePackContext:
        return _TemplatePackContext(
            key=key,
            source=source,
            image_bytes=image_bytes,
            extra_files=extra_files,
            mutate=mutate,
        )

    @staticmethod
    def _write_manifest(root: Path, *, source: str) -> None:
        _TemplatePackContext.write_manifest(root, key="city.collect.food.ready", source=source)


class _TemplatePackContext:
    def __init__(
        self,
        *,
        key: str,
        source: str,
        image_bytes: bytes,
        extra_files: tuple[str, ...],
        mutate: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        self.key = key
        self.source = source
        self.image_bytes = image_bytes
        self.extra_files = extra_files
        self.mutate = mutate

    def __enter__(self) -> Path:
        for relative_path in (self.source, *self.extra_files):
            path = self.root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.image_bytes)

        manifest: dict[str, Any] = {
            "manifest_version": 1,
            "version": "2026.07",
            "languages": ["en"],
            "resolution_profiles": {
                "phone.720p": {
                    "width": 1280,
                    "height": 720,
                }
            },
            "templates": [
                {
                    "key": self.key,
                    "source": self.source,
                    "language": "en",
                    "resolution_profile": "phone.720p",
                    "roi": {
                        "x": 100,
                        "y": 200,
                        "width": 300,
                        "height": 120,
                    },
                    "threshold": 0.87,
                    "scale_range": {
                        "min": 0.75,
                        "max": 1.25,
                    },
                    "scene_constraints": {
                        "allowed": ["city", "home"],
                        "required": ["home"],
                    },
                    "source_reference": "synthetic test fixture",
                }
            ],
        }
        if self.mutate is not None:
            self.mutate(manifest)
        (self.root / "template-pack.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        return self.root

    def __exit__(self, *args: object) -> None:
        self._temp_dir.cleanup()

    @staticmethod
    def write_manifest(root: Path, *, key: str, source: str) -> None:
        manifest: dict[str, Any] = {
            "manifest_version": 1,
            "version": "2026.07",
            "languages": ["en"],
            "resolution_profiles": {
                "phone.720p": {
                    "width": 1280,
                    "height": 720,
                }
            },
            "templates": [
                {
                    "key": key,
                    "source": source,
                    "language": "en",
                    "resolution_profile": "phone.720p",
                    "roi": {
                        "x": 100,
                        "y": 200,
                        "width": 300,
                        "height": 120,
                    },
                    "threshold": 0.87,
                    "scale_range": {
                        "min": 0.75,
                        "max": 1.25,
                    },
                    "scene_constraints": {
                        "allowed": ["city", "home"],
                        "required": ["home"],
                    },
                    "source_reference": "synthetic test fixture",
                }
            ],
        }
        (root / "template-pack.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
