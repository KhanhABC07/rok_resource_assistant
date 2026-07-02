from __future__ import annotations

import json
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

import rok_assistant.vision.image_matching as image_matching
from rok_assistant.vision import (
    RegionOfInterest,
    ResolutionProfile,
    ScaleRange,
    TemplateDefinition,
    TemplateImageNormalizer,
    TemplatePack,
    TemplateRegistry,
    TemplateScreenAnalyzer,
    find_template,
)


class TemplateScreenAnalyzerTest(unittest.TestCase):
    def test_exact_scale_positive_match_populates_detection_result(self) -> None:
        with self._pack() as pack:
            template = self._template()
            screenshot = self._screenshot()
            screenshot[24:32, 18:28] = template
            pack.write_image("templates/collect.png", template)

            result = TemplateScreenAnalyzer().match(
                screenshot,
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertEqual("city.collect.food.ready", result.matched_semantic_key)
            self.assertGreaterEqual(result.confidence, 0.99)
            self.assertEqual(18, result.bounding_box.x)
            self.assertEqual(24, result.bounding_box.y)
            self.assertEqual(10, result.bounding_box.width)
            self.assertEqual(8, result.bounding_box.height)
            self.assertEqual(1.0, result.matched_scale)
            self.assertEqual("city", result.scene)
            self.assertEqual("2026.07", result.template_pack_version)
            self.assertEqual("TemplateScreenAnalyzer", result.metadata.matcher)
            self.assertTrue(result.metadata.normalized)
            self.assertEqual(1, result.metadata.candidate_count)

    def test_no_match_below_threshold_returns_structured_diagnostic(self) -> None:
        with self._pack(threshold=0.99) as pack:
            pack.write_image("templates/collect.png", self._template())
            screenshot = self._screenshot()
            screenshot[24:32, 18:28] = np.full((8, 10), 7, dtype=np.uint8)

            result = TemplateScreenAnalyzer().match(
                screenshot,
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertIsNone(result.matched_semantic_key)
            self.assertIsNone(result.bounding_box)
            self.assertDiagnostic(result, "match.below_threshold")

    def test_roi_local_match_is_translated_to_full_screen_coordinates(self) -> None:
        roi = {"x": 20, "y": 15, "width": 50, "height": 40}
        with self._pack(roi=roi) as pack:
            template = self._template()
            screenshot = self._screenshot()
            screenshot[29:37, 33:43] = template
            pack.write_image("templates/collect.png", template)

            result = TemplateScreenAnalyzer().match(
                screenshot,
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertEqual(33, result.bounding_box.x)
            self.assertEqual(29, result.bounding_box.y)

    def test_roi_outside_screenshot_is_rejected(self) -> None:
        with self._pack(
            roi={"x": 80, "y": 70, "width": 30, "height": 20},
            profile_width=120,
            profile_height=100,
        ) as pack:
            pack.write_image("templates/collect.png", self._template())

            result = TemplateScreenAnalyzer().match(
                self._screenshot(),
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertIsNone(result.matched_semantic_key)
            self.assertDiagnostic(result, "match.invalid_roi")

    def test_zero_area_runtime_roi_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template_path = root / "templates" / "collect.png"
            template_path.parent.mkdir()
            cv2.imwrite(str(template_path), self._template())
            roi = object.__new__(RegionOfInterest)
            object.__setattr__(roi, "x", 0)
            object.__setattr__(roi, "y", 0)
            object.__setattr__(roi, "width", 0)
            object.__setattr__(roi, "height", 20)
            definition = TemplateDefinition(
                semantic_key="city.collect.food.ready",
                template_pack_version="2026.07",
                language="en",
                resolution_profile="phone.720p",
                source=Path("templates/collect.png"),
                region_of_interest=roi,
                confidence_threshold=0.95,
            )
            registry = TemplateRegistry(
                TemplatePack(
                    version="2026.07",
                    languages=("en",),
                    resolution_profiles=(ResolutionProfile("phone.720p", 100, 80),),
                    templates=(definition,),
                    root=root,
                )
            )

            result = TemplateScreenAnalyzer().match(
                self._screenshot(),
                "city.collect.food.ready",
                registry,
            )

            self.assertDiagnostic(result, "match.invalid_roi")

    def test_deterministic_multi_scale_selection_uses_best_scale(self) -> None:
        with self._pack(scale_min=1.0, scale_max=1.5) as pack:
            template = self._template()
            scaled = cv2.resize(template, (15, 12), interpolation=cv2.INTER_AREA)
            screenshot = self._screenshot()
            screenshot[26:38, 40:55] = scaled
            pack.write_image("templates/collect.png", template)

            result = TemplateScreenAnalyzer(scale_step=0.25).match(
                screenshot,
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertEqual(1.5, result.matched_scale)
            self.assertEqual(40, result.bounding_box.x)
            self.assertEqual(26, result.bounding_box.y)

    def test_equal_confidence_tie_breaks_to_top_left_match(self) -> None:
        with self._pack() as pack:
            template = self._template()
            screenshot = self._screenshot()
            screenshot[20:28, 40:50] = template
            screenshot[20:28, 6:16] = template
            pack.write_image("templates/collect.png", template)

            result = TemplateScreenAnalyzer().match(
                screenshot,
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertEqual(6, result.bounding_box.x)
            self.assertEqual(20, result.bounding_box.y)

    def test_scales_larger_than_roi_are_skipped(self) -> None:
        with self._pack(
            roi={"x": 0, "y": 0, "width": 12, "height": 10},
            scale_min=1.0,
            scale_max=2.0,
        ) as pack:
            template = self._template()
            screenshot = self._screenshot()
            screenshot[1:9, 1:11] = template
            pack.write_image("templates/collect.png", template)

            result = TemplateScreenAnalyzer(scale_step=1.0).match(
                screenshot,
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertEqual(1, result.metadata.candidate_count)
            self.assertEqual(1.0, result.matched_scale)

    def test_optional_mask_is_accepted_when_compatible(self) -> None:
        with self._pack(mask="masks/collect-mask.png") as pack:
            template = self._template()
            mask = np.full(template.shape, 255, dtype=np.uint8)
            screenshot = self._screenshot()
            screenshot[24:32, 18:28] = template
            pack.write_image("templates/collect.png", template)
            pack.write_image("masks/collect-mask.png", mask)

            result = TemplateScreenAnalyzer().match(
                screenshot,
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertEqual("city.collect.food.ready", result.matched_semantic_key)

    def test_incompatible_mask_dimensions_are_rejected(self) -> None:
        with self._pack(mask="masks/collect-mask.png") as pack:
            pack.write_image("templates/collect.png", self._template())
            pack.write_image("masks/collect-mask.png", np.full((7, 10), 255, dtype=np.uint8))

            result = TemplateScreenAnalyzer().match(
                self._screenshot(),
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertDiagnostic(result, "template.mask_incompatible")

    def test_unreadable_source_image_produces_structured_diagnostic(self) -> None:
        with self._pack() as pack:
            (pack.root / "templates").mkdir(parents=True, exist_ok=True)
            (pack.root / "templates" / "collect.png").write_bytes(b"not an image")

            result = TemplateScreenAnalyzer().match(
                self._screenshot(),
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertDiagnostic(result, "template.invalid_source_image")
            self.assertNotIn(str(pack.root), self._diagnostic_text(result))

    def test_malformed_screenshot_input_is_rejected(self) -> None:
        with self._pack() as pack:
            pack.write_image("templates/collect.png", self._template())
            malformed = np.zeros((10, 10), dtype=np.float32)

            result = TemplateScreenAnalyzer().match(
                malformed,
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertDiagnostic(result, "image.invalid")

    def test_empty_screenshot_array_is_rejected(self) -> None:
        with self._pack() as pack:
            pack.write_image("templates/collect.png", self._template())

            result = TemplateScreenAnalyzer().match(
                np.array([], dtype=np.uint8),
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertDiagnostic(result, "image.invalid")

    def test_unsupported_dimensionality_is_rejected(self) -> None:
        result = TemplateImageNormalizer().normalize(
            np.zeros((2, 2, 2, 2), dtype=np.uint8),
            field="screenshot",
        )

        self.assertFalse(result.is_valid)
        self.assertEqual("image.invalid", result.diagnostics[0].code)

    def test_unsupported_channel_count_is_rejected(self) -> None:
        result = TemplateImageNormalizer().normalize(
            np.zeros((10, 10, 2), dtype=np.uint8),
            field="screenshot",
        )

        self.assertFalse(result.is_valid)
        self.assertEqual("image.invalid", result.diagnostics[0].code)

    def test_unsupported_data_type_is_rejected(self) -> None:
        result = TemplateImageNormalizer().normalize(
            np.zeros((10, 10), dtype=np.float32),
            field="screenshot",
        )

        self.assertFalse(result.is_valid)
        self.assertEqual("image.invalid", result.diagnostics[0].code)

    def test_non_contiguous_input_is_copied_to_contiguous_output(self) -> None:
        source = np.arange(100, dtype=np.uint8).reshape(10, 10)
        non_contiguous = source[:, ::2]

        result = TemplateImageNormalizer().normalize(non_contiguous, field="screenshot")

        self.assertTrue(result.is_valid)
        self.assertTrue(result.image.pixels.flags.c_contiguous)
        result.image.pixels[:, :] = 0
        self.assertFalse(np.array_equal(source[:, ::2], result.image.pixels))

    def test_alpha_channel_input_is_converted_to_grayscale_without_using_alpha_as_mask(self) -> None:
        bgra = np.zeros((4, 5, 4), dtype=np.uint8)
        bgra[:, :, 0] = 10
        bgra[:, :, 1] = 20
        bgra[:, :, 2] = 30
        bgra[:, :, 3] = 0

        result = TemplateImageNormalizer().normalize(bgra, field="screenshot")

        self.assertTrue(result.is_valid)
        self.assertEqual((4, 5), result.image.pixels.shape)
        self.assertEqual(np.uint8, result.image.pixels.dtype)

    def test_grayscale_and_color_normalization_are_supported(self) -> None:
        normalizer = TemplateImageNormalizer()
        grayscale = np.arange(25, dtype=np.uint8).reshape(5, 5)
        color = np.zeros((5, 5, 3), dtype=np.uint8)
        color[:, :, 1] = 128

        grayscale_result = normalizer.normalize(grayscale, field="gray")
        color_result = normalizer.normalize(color, field="color")

        self.assertTrue(grayscale_result.is_valid)
        self.assertTrue(color_result.is_valid)
        self.assertEqual((5, 5), grayscale_result.image.pixels.shape)
        self.assertEqual((5, 5), color_result.image.pixels.shape)

    def test_caller_image_is_not_mutated(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[5:10, 5:10] = [10, 20, 30]
        original = image.copy()

        result = TemplateImageNormalizer().normalize(image, field="screenshot")
        result.image.pixels[:, :] = 255

        self.assertTrue(np.array_equal(original, image))

    def test_caller_array_writeability_state_is_preserved(self) -> None:
        image = np.zeros((8, 8), dtype=np.uint8)
        image.setflags(write=False)

        result = TemplateImageNormalizer().normalize(image, field="screenshot")
        result.image.pixels[:, :] = 255

        self.assertFalse(image.flags.writeable)
        self.assertTrue(np.array_equal(image, np.zeros((8, 8), dtype=np.uint8)))

    def test_exact_roi_right_and_bottom_boundary_is_allowed(self) -> None:
        with self._pack(roi={"x": 90, "y": 72, "width": 10, "height": 8}) as pack:
            template = self._template()
            screenshot = self._screenshot()
            screenshot[72:80, 90:100] = template
            pack.write_image("templates/collect.png", template)

            result = TemplateScreenAnalyzer().match(
                screenshot,
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertEqual("city.collect.food.ready", result.matched_semantic_key)
            self.assertEqual(90, result.bounding_box.x)
            self.assertEqual(72, result.bounding_box.y)

    def test_partially_out_of_frame_roi_is_rejected(self) -> None:
        with self._pack(
            roi={"x": 95, "y": 20, "width": 10, "height": 8},
            profile_width=110,
        ) as pack:
            pack.write_image("templates/collect.png", self._template())

            result = TemplateScreenAnalyzer().match(
                self._screenshot(),
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertDiagnostic(result, "match.invalid_roi")

    def test_translated_bounding_box_remains_inside_screenshot(self) -> None:
        roi = {"x": 75, "y": 60, "width": 25, "height": 20}
        with self._pack(roi=roi) as pack:
            template = self._template()
            screenshot = self._screenshot()
            screenshot[66:74, 88:98] = template
            pack.write_image("templates/collect.png", template)

            result = TemplateScreenAnalyzer().match(
                screenshot,
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertLessEqual(result.bounding_box.x + result.bounding_box.width, screenshot.shape[1])
            self.assertLessEqual(result.bounding_box.y + result.bounding_box.height, screenshot.shape[0])

    def test_min_scale_equal_to_max_scale_is_supported(self) -> None:
        with self._pack(scale_min=1.25, scale_max=1.25) as pack:
            template = self._template()
            scaled = cv2.resize(template, (12, 10), interpolation=cv2.INTER_AREA)
            screenshot = self._screenshot()
            screenshot[10:20, 10:22] = scaled
            pack.write_image("templates/collect.png", template)

            result = TemplateScreenAnalyzer().match(
                screenshot,
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertEqual(1.25, result.matched_scale)
            self.assertEqual(1, result.metadata.candidate_count)

    def test_scale_rounding_duplicate_sizes_are_evaluated_once(self) -> None:
        with self._pack(scale_min=1.0, scale_max=1.04) as pack:
            template = self._template()
            screenshot = self._screenshot()
            screenshot[24:32, 18:28] = template
            pack.write_image("templates/collect.png", template)

            result = TemplateScreenAnalyzer(scale_step=0.01).match(
                screenshot,
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertEqual(1, result.metadata.candidate_count)

    def test_no_scale_fits_roi_returns_structured_no_match(self) -> None:
        with self._pack(roi={"x": 0, "y": 0, "width": 9, "height": 7}) as pack:
            pack.write_image("templates/collect.png", self._template())

            result = TemplateScreenAnalyzer().match(
                self._screenshot(),
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertIsNone(result.matched_semantic_key)
            self.assertDiagnostic(result, "match.no_eligible_scale")

    def test_threshold_equality_is_treated_as_match(self) -> None:
        with self._pack(threshold=0.75) as pack:
            template = self._template()
            pack.write_image("templates/collect.png", template)

            with patch.object(
                TemplateScreenAnalyzer,
                "_match_at_scale",
                return_value=image_matching._MatchCandidate(
                    confidence=0.75,
                    x=18,
                    y=24,
                    width=10,
                    height=8,
                    scale=1.0,
                ),
            ):
                result = TemplateScreenAnalyzer().match(
                    self._screenshot(),
                    "city.collect.food.ready",
                    pack.registry(),
                    scene="city",
                )

            self.assertEqual("city.collect.food.ready", result.matched_semantic_key)
            self.assertEqual(0.75, result.confidence)

    def test_non_finite_matching_scores_are_ignored_when_finite_candidate_exists(self) -> None:
        with self._pack(threshold=0.5) as pack:
            template = self._template()
            pack.write_image("templates/collect.png", template)
            scores = np.array([[np.nan, -np.inf], [0.75, np.inf]], dtype=np.float32)

            with patch("cv2.matchTemplate", return_value=scores):
                result = TemplateScreenAnalyzer().match(
                    self._screenshot(),
                    "city.collect.food.ready",
                    pack.registry(),
                    scene="city",
                )

            self.assertEqual("city.collect.food.ready", result.matched_semantic_key)
            self.assertEqual(0, result.bounding_box.x)
            self.assertEqual(1, result.bounding_box.y)
            self.assertEqual(0.75, result.confidence)

    def test_degenerate_constant_images_return_structured_result(self) -> None:
        with self._pack(threshold=0.99) as pack:
            pack.write_image("templates/collect.png", np.zeros((8, 10), dtype=np.uint8))

            result = TemplateScreenAnalyzer().match(
                self._screenshot(),
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertIsNone(result.matched_semantic_key)
            self.assertTrue(result.metadata.diagnostics)

    def test_invalid_runtime_scale_range_returns_structured_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template_path = root / "templates" / "collect.png"
            template_path.parent.mkdir()
            cv2.imwrite(str(template_path), self._template())
            invalid_scale_range = object.__new__(ScaleRange)
            object.__setattr__(invalid_scale_range, "minimum", float("nan"))
            object.__setattr__(invalid_scale_range, "maximum", 1.0)
            definition = TemplateDefinition(
                semantic_key="city.collect.food.ready",
                template_pack_version="2026.07",
                language="en",
                resolution_profile="phone.720p",
                source=Path("templates/collect.png"),
                region_of_interest=RegionOfInterest(0, 0, 100, 80),
                confidence_threshold=0.95,
                scale_range=invalid_scale_range,
            )
            registry = TemplateRegistry(
                TemplatePack(
                    version="2026.07",
                    languages=("en",),
                    resolution_profiles=(ResolutionProfile("phone.720p", 100, 80),),
                    templates=(definition,),
                    root=root,
                )
            )

            result = TemplateScreenAnalyzer().match(
                self._screenshot(),
                "city.collect.food.ready",
                registry,
            )

            self.assertDiagnostic(result, "match.invalid_scale_range")

    def test_unreadable_template_followed_by_successful_match_does_not_poison_analyzer(self) -> None:
        analyzer = TemplateScreenAnalyzer()
        with self._pack() as broken_pack:
            (broken_pack.root / "templates").mkdir(parents=True, exist_ok=True)
            (broken_pack.root / "templates" / "collect.png").write_bytes(b"not an image")
            broken = analyzer.match(
                self._screenshot(),
                "city.collect.food.ready",
                broken_pack.registry(),
                scene="city",
            )
        with self._pack() as valid_pack:
            template = self._template()
            screenshot = self._screenshot()
            screenshot[24:32, 18:28] = template
            valid_pack.write_image("templates/collect.png", template)
            recovered = analyzer.match(
                screenshot,
                "city.collect.food.ready",
                valid_pack.registry(),
                scene="city",
            )

        self.assertDiagnostic(broken, "template.invalid_source_image")
        self.assertEqual("city.collect.food.ready", recovered.matched_semantic_key)

    def test_failed_mask_load_followed_by_successful_unmasked_match(self) -> None:
        analyzer = TemplateScreenAnalyzer()
        with self._pack(mask="masks/collect-mask.png") as masked_pack:
            template = self._template()
            screenshot = self._screenshot()
            screenshot[24:32, 18:28] = template
            masked_pack.write_image("templates/collect.png", template)
            (masked_pack.root / "masks").mkdir(parents=True, exist_ok=True)
            (masked_pack.root / "masks" / "collect-mask.png").write_bytes(b"not an image")
            masked = analyzer.match(
                screenshot,
                "city.collect.food.ready",
                masked_pack.registry(),
                scene="city",
            )
        with self._pack() as plain_pack:
            template = self._template()
            screenshot = self._screenshot()
            screenshot[24:32, 18:28] = template
            plain_pack.write_image("templates/collect.png", template)
            unmasked = analyzer.match(
                screenshot,
                "city.collect.food.ready",
                plain_pack.registry(),
                scene="city",
            )

        self.assertDiagnostic(masked, "template.invalid_mask_image")
        self.assertEqual("city.collect.food.ready", unmasked.matched_semantic_key)

    def test_template_source_is_loaded_lazily_after_registry_construction(self) -> None:
        with self._pack() as pack:
            pack.write_image("templates/collect.png", self._template())
            with patch("cv2.imread", side_effect=AssertionError("should not load image")):
                registry = pack.registry()

            result = TemplateScreenAnalyzer().match(
                self._screenshot(),
                "city.collect.food.ready",
                registry,
                scene="city",
            )

            self.assertIsNone(result.matched_semantic_key)

    def test_registry_is_reusable_across_repeated_matches(self) -> None:
        with self._pack() as pack:
            template = self._template()
            screenshot = self._screenshot()
            screenshot[24:32, 18:28] = template
            pack.write_image("templates/collect.png", template)
            registry = pack.registry()
            analyzer = TemplateScreenAnalyzer()

            first = analyzer.match(screenshot, "city.collect.food.ready", registry, scene="city")
            second = analyzer.match(screenshot, "city.collect.food.ready", registry, scene="city")

            self.assertEqual(first.bounding_box, second.bounding_box)
            self.assertEqual(first.matched_scale, second.matched_scale)

    def test_independent_analyzers_do_not_share_mutable_state(self) -> None:
        with self._pack() as pack:
            template = self._template()
            screenshot = self._screenshot()
            screenshot[24:32, 18:28] = template
            pack.write_image("templates/collect.png", template)
            registry = pack.registry()

            first_analyzer = TemplateScreenAnalyzer()
            second_analyzer = TemplateScreenAnalyzer(scale_step=0.5)

            first = first_analyzer.match(screenshot, "city.collect.food.ready", registry, scene="city")
            second = second_analyzer.match(
                screenshot,
                "city.collect.food.ready",
                registry,
                scene="city",
            )

            self.assertEqual(first.bounding_box, second.bounding_box)
            self.assertIsNot(first_analyzer.normalizer, second_analyzer.normalizer)
            self.assertIsNot(first.metadata, second.metadata)

    def test_repeated_analyzer_use_returns_deterministic_result(self) -> None:
        with self._pack() as pack:
            template = self._template()
            screenshot = self._screenshot()
            screenshot[24:32, 18:28] = template
            pack.write_image("templates/collect.png", template)
            registry = pack.registry()
            analyzer = TemplateScreenAnalyzer(clock=lambda: 10.0)

            results = [
                analyzer.match(screenshot, "city.collect.food.ready", registry, scene="city")
                for _ in range(3)
            ]

            self.assertEqual(results[0], results[1])
            self.assertEqual(results[1], results[2])

    def test_two_analyzers_using_different_registries_remain_isolated(self) -> None:
        with self._pack() as first_pack, self._pack() as second_pack:
            template = self._template()
            first_screenshot = self._screenshot()
            second_screenshot = self._screenshot()
            first_screenshot[10:18, 10:20] = template
            second_screenshot[40:48, 50:60] = template
            first_pack.write_image("templates/collect.png", template)
            second_pack.write_image("templates/collect.png", template)

            first = TemplateScreenAnalyzer().match(
                first_screenshot,
                "city.collect.food.ready",
                first_pack.registry(),
                scene="city",
            )
            second = TemplateScreenAnalyzer().match(
                second_screenshot,
                "city.collect.food.ready",
                second_pack.registry(),
                scene="city",
            )

            self.assertEqual(10, first.bounding_box.x)
            self.assertEqual(50, second.bounding_box.x)

    def test_raw_opencv_exceptions_do_not_leak(self) -> None:
        with self._pack() as pack:
            template = self._template()
            pack.write_image("templates/collect.png", template)
            with patch("cv2.matchTemplate", side_effect=cv2.error("bad match")):
                result = TemplateScreenAnalyzer().match(
                    self._screenshot(),
                    "city.collect.food.ready",
                    pack.registry(),
                    scene="city",
                )

            self.assertDiagnostic(result, "match.failed")

    def test_legacy_find_template_contract_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            screenshot_path = temp_path / "screenshot.png"
            template_path = temp_path / "template.png"
            screenshot = np.zeros((80, 100, 3), dtype=np.uint8)
            template = np.dstack([self._template()] * 3)
            screenshot[24:32, 18:28] = template
            cv2.imwrite(str(screenshot_path), screenshot)
            cv2.imwrite(str(template_path), template)

            result = find_template(screenshot_path, template_path, threshold=0.95)

            self.assertEqual({"found", "confidence", "x", "y"}, set(result))
            self.assertTrue(result["found"])
            self.assertEqual(18, result["x"])
            self.assertEqual(24, result["y"])

    def test_structured_diagnostics_do_not_contain_absolute_local_paths(self) -> None:
        with self._pack() as pack:
            pack.write_image("templates/collect.png", self._template())

            result = TemplateScreenAnalyzer().match(
                pack.root / "missing-screenshot.png",
                "city.collect.food.ready",
                pack.registry(),
                scene="city",
            )

            self.assertDiagnostic(result, "image.invalid")
            self.assertNotIn(str(pack.root), self._diagnostic_text(result))

    def test_all_non_finite_matching_scores_return_structured_failure(self) -> None:
        with self._pack(threshold=0.5) as pack:
            pack.write_image("templates/collect.png", self._template())
            scores = np.array([[np.nan, -np.inf], [np.inf, np.nan]], dtype=np.float32)

            with patch("cv2.matchTemplate", return_value=scores):
                result = TemplateScreenAnalyzer().match(
                    self._screenshot(),
                    "city.collect.food.ready",
                    pack.registry(),
                    scene="city",
                )

            self.assertDiagnostic(result, "match.failed")

    def test_scale_generation_rejects_excessively_dense_ranges(self) -> None:
        with self._pack(scale_min=1.0, scale_max=1.2) as pack:
            pack.write_image("templates/collect.png", self._template())

            with patch.object(image_matching, "_MAX_SCALE_COUNT", 3):
                result = TemplateScreenAnalyzer(scale_step=0.01).match(
                    self._screenshot(),
                    "city.collect.food.ready",
                    pack.registry(),
                    scene="city",
                )

            self.assertDiagnostic(result, "match.scale_range_too_dense")

    def assertDiagnostic(self, result: object, code: str) -> None:
        diagnostics = result.metadata.diagnostics
        self.assertIn(code, {diagnostic.code for diagnostic in diagnostics})

    @staticmethod
    def _diagnostic_text(result: object) -> str:
        return "\n".join(diagnostic.message for diagnostic in result.metadata.diagnostics)

    @staticmethod
    def _template() -> np.ndarray:
        template = np.zeros((8, 10), dtype=np.uint8)
        for y in range(template.shape[0]):
            for x in range(template.shape[1]):
                template[y, x] = 40 + ((x * 17 + y * 23) % 200)
        return template

    @staticmethod
    def _screenshot() -> np.ndarray:
        return np.zeros((80, 100), dtype=np.uint8)

    @staticmethod
    def _pack(
        *,
        roi: dict[str, int] | None = None,
        threshold: float = 0.95,
        scale_min: float = 1.0,
        scale_max: float = 1.0,
        mask: str | None = None,
        profile_width: int = 100,
        profile_height: int = 80,
    ) -> _TemplatePackContext:
        return _TemplatePackContext(
            roi=roi or {"x": 0, "y": 0, "width": 100, "height": 80},
            threshold=threshold,
            scale_min=scale_min,
            scale_max=scale_max,
            mask=mask,
            profile_width=profile_width,
            profile_height=profile_height,
        )


class _TemplatePackContext:
    def __init__(
        self,
        *,
        roi: dict[str, int],
        threshold: float,
        scale_min: float,
        scale_max: float,
        mask: str | None,
        profile_width: int,
        profile_height: int,
    ) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        self.roi = roi
        self.threshold = threshold
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.mask = mask
        self.profile_width = profile_width
        self.profile_height = profile_height

    def __enter__(self) -> _TemplatePackContext:
        manifest = {
            "manifest_version": 1,
            "version": "2026.07",
            "languages": ["en"],
            "resolution_profiles": {
                "phone.720p": {
                    "width": self.profile_width,
                    "height": self.profile_height,
                }
            },
            "templates": [
                {
                    "key": "city.collect.food.ready",
                    "source": "templates/collect.png",
                    "language": "en",
                    "resolution_profile": "phone.720p",
                    "roi": self.roi,
                    "threshold": self.threshold,
                    "scale_range": {
                        "min": self.scale_min,
                        "max": self.scale_max,
                    },
                    "scene_constraints": {
                        "allowed": ["city"],
                        "required": [],
                    },
                    "source_reference": "synthetic test fixture",
                }
            ],
        }
        if self.mask is not None:
            manifest["templates"][0]["mask"] = self.mask
        (self.root / "template-pack.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        return self

    def __exit__(self, *args: object) -> None:
        self._temp_dir.cleanup()

    def registry(self) -> TemplateRegistry:
        return TemplateRegistry.from_pack_root(self.root)

    def write_image(self, relative_path: str, image: np.ndarray) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.assert_image_written(path, image)

    @staticmethod
    def assert_image_written(path: Path, image: np.ndarray) -> None:
        if not cv2.imwrite(str(path), image):
            raise AssertionError(f"Could not write synthetic image: {path.name}")


if __name__ == "__main__":
    unittest.main()
