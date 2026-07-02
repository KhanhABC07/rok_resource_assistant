from __future__ import annotations

import json
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
    RegionOfInterest,
    ResolutionProfile,
    ScaleRange,
    SceneCandidateResult,
    SceneClassificationResult,
    SceneClassificationStatus,
    SceneClassifier,
    SceneConstraints,
    SceneDefinition,
    SceneRule,
    TemplateDefinition,
    TemplatePack,
    TemplateRegistry,
    ValidationDiagnostic,
)


class SceneClassifierTest(unittest.TestCase):
    def test_single_scene_positive_classification(self) -> None:
        registry = self._registry(("city.hall",))
        analyzer = _RecordingAnalyzer({"city.hall": self._match("city.hall", 0.9)})

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.CLASSIFIED, result.status)
        self.assertEqual("city", result.scene_key)
        self.assertEqual(0.72, result.score)
        self.assertEqual(("city.hall",), analyzer.called_keys())

    def test_classifier_uses_real_template_screen_analyzer_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = self._template_image()
            screenshot = np.zeros((20, 20), dtype=np.uint8)
            screenshot[6:10, 7:12] = template
            template_path = root / "templates" / "city_hall.png"
            template_path.parent.mkdir(parents=True)
            self.assertTrue(cv2.imwrite(str(template_path), template))
            (root / "template-pack.json").write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "version": "2026.07",
                        "languages": ["en"],
                        "resolution_profiles": {
                            "phone.720p": {"width": 20, "height": 20}
                        },
                        "templates": [
                            {
                                "key": "city.hall",
                                "source": "templates/city_hall.png",
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

            result = SceneClassifier().classify(
                screenshot,
                (SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),),
                TemplateRegistry.from_pack_root(root),
            )

            self.assertEqual(SceneClassificationStatus.CLASSIFIED, result.status)
            self.assertEqual("city", result.scene_key)

    def test_required_evidence_missing_returns_unknown(self) -> None:
        registry = self._registry(("city.hall",))
        analyzer = _RecordingAnalyzer({"city.hall": self._no_match()})

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.UNKNOWN, result.status)
        self.assertEqual(("city.hall",), result.candidates[0].missing_required)

    def test_optional_evidence_improves_score(self) -> None:
        registry = self._registry(("city.hall", "city.collect"))
        analyzer = _RecordingAnalyzer(
            {
                "city.hall": self._match("city.hall", 0.8),
                "city.collect": self._match("city.collect", 1.0),
            }
        )

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition(
                    "city",
                    SceneRule(
                        required_template_keys=("city.hall",),
                        optional_template_keys=("city.collect",),
                    ),
                ),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.CLASSIFIED, result.status)
        self.assertEqual(0.84, result.score)
        self.assertEqual(("city.collect",), result.candidates[0].satisfied_optional)

    def test_additional_lower_confidence_optional_evidence_does_not_reduce_required_scene_score(self) -> None:
        registry = self._registry(("city.hall", "city.collect", "city.help"))
        analyzer = _RecordingAnalyzer(
            {
                "city.hall": self._match("city.hall", 0.8),
                "city.collect": self._match("city.collect", 1.0),
                "city.help": self._match("city.help", 0.25),
            }
        )

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition(
                    "city",
                    SceneRule(
                        required_template_keys=("city.hall",),
                        optional_template_keys=("city.collect", "city.help"),
                    ),
                ),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.CLASSIFIED, result.status)
        self.assertEqual(0.84, result.score)

    def test_missing_optional_evidence_does_not_invalidate_required_scene(self) -> None:
        registry = self._registry(("city.hall", "city.collect"))
        analyzer = _RecordingAnalyzer(
            {
                "city.hall": self._match("city.hall", 0.8),
                "city.collect": self._no_match(),
            }
        )

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition(
                    "city",
                    SceneRule(
                        required_template_keys=("city.hall",),
                        optional_template_keys=("city.collect",),
                    ),
                ),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.CLASSIFIED, result.status)
        self.assertEqual(0.64, result.score)

    def test_optional_only_scene_with_no_matches_is_unknown(self) -> None:
        registry = self._registry(("city.collect",))
        analyzer = _RecordingAnalyzer({"city.collect": self._no_match()})

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (SceneDefinition("city", SceneRule(optional_template_keys=("city.collect",))),),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.UNKNOWN, result.status)
        self.assertEqual(0.0, result.candidates[0].score)

    def test_optional_only_scene_with_one_match_classifies_using_optional_average(self) -> None:
        registry = self._registry(("city.collect", "city.help"))
        analyzer = _RecordingAnalyzer(
            {
                "city.collect": self._match("city.collect", 0.8),
                "city.help": self._no_match(),
            }
        )

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition(
                    "city",
                    SceneRule(optional_template_keys=("city.collect", "city.help")),
                ),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.CLASSIFIED, result.status)
        self.assertEqual(0.8, result.score)

    def test_optional_only_scene_averages_multiple_matched_optional_confidences(self) -> None:
        registry = self._registry(("city.collect", "city.help"))
        analyzer = _RecordingAnalyzer(
            {
                "city.collect": self._match("city.collect", 0.8),
                "city.help": self._match("city.help", 0.4),
            }
        )

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition(
                    "city",
                    SceneRule(optional_template_keys=("city.collect", "city.help")),
                ),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.CLASSIFIED, result.status)
        self.assertEqual(0.6, result.score)

    def test_forbidden_evidence_rejects_scene(self) -> None:
        registry = self._registry(("city.hall", "map.marker"))
        analyzer = _RecordingAnalyzer(
            {
                "city.hall": self._match("city.hall", 0.9),
                "map.marker": self._match("map.marker", 0.9),
            }
        )

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition(
                    "city",
                    SceneRule(
                        required_template_keys=("city.hall",),
                        forbidden_template_keys=("map.marker",),
                    ),
                ),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.UNKNOWN, result.status)
        self.assertEqual(("map.marker",), result.candidates[0].present_forbidden)

    def test_forbidden_match_overrides_high_required_and_optional_score(self) -> None:
        registry = self._registry(("city.hall", "city.collect", "map.marker"))
        analyzer = _RecordingAnalyzer(
            {
                "city.hall": self._match("city.hall", 1.0),
                "city.collect": self._match("city.collect", 1.0),
                "map.marker": self._match("map.marker", 1.0),
            }
        )

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition(
                    "city",
                    SceneRule(
                        required_template_keys=("city.hall",),
                        optional_template_keys=("city.collect",),
                        forbidden_template_keys=("map.marker",),
                    ),
                ),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.UNKNOWN, result.status)
        self.assertEqual(1.0, result.candidates[0].score)
        self.assertEqual(("map.marker",), result.candidates[0].present_forbidden)

    def test_unknown_result_when_no_scene_qualifies(self) -> None:
        registry = self._registry(("city.hall", "map.marker"))
        analyzer = _RecordingAnalyzer(
            {
                "city.hall": self._no_match(),
                "map.marker": self._no_match(),
            }
        )

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),
                SceneDefinition("map", SceneRule(required_template_keys=("map.marker",))),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.UNKNOWN, result.status)
        self.assertIsNone(result.scene_key)

    def test_ambiguous_result_for_equal_candidates(self) -> None:
        registry = self._registry(("alliance.panel", "city.hall"))
        analyzer = _RecordingAnalyzer(
            {
                "alliance.panel": self._match("alliance.panel", 0.9),
                "city.hall": self._match("city.hall", 0.9),
            }
        )

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition("city", SceneRule(required_template_keys=("city.hall",)), priority=5),
                SceneDefinition(
                    "alliance",
                    SceneRule(required_template_keys=("alliance.panel",)),
                    priority=5,
                ),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.AMBIGUOUS, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.ambiguous")
        self.assertEqual(("alliance", "city"), tuple(candidate.scene_key for candidate in result.candidates))

    def test_deterministic_priority_tie_breaking(self) -> None:
        registry = self._registry(("alliance.panel", "city.hall"))
        analyzer = _RecordingAnalyzer(
            {
                "alliance.panel": self._match("alliance.panel", 0.9),
                "city.hall": self._match("city.hall", 0.9),
            }
        )

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition("city", SceneRule(required_template_keys=("city.hall",)), priority=1),
                SceneDefinition(
                    "alliance",
                    SceneRule(required_template_keys=("alliance.panel",)),
                    priority=9,
                ),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.CLASSIFIED, result.status)
        self.assertEqual("city", result.scene_key)

    def test_required_evidence_count_ranks_before_score(self) -> None:
        registry = self._registry(("city.hall", "city.collect", "map.marker"))
        analyzer = _RecordingAnalyzer(
            {
                "city.hall": self._match("city.hall", 0.7),
                "city.collect": self._match("city.collect", 0.7),
                "map.marker": self._match("map.marker", 1.0),
            }
        )

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition(
                    "city",
                    SceneRule(required_template_keys=("city.hall", "city.collect")),
                    priority=5,
                ),
                SceneDefinition("map", SceneRule(required_template_keys=("map.marker",)), priority=1),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.CLASSIFIED, result.status)
        self.assertEqual("city", result.scene_key)

    def test_semantic_key_ordering_is_deterministic_for_ambiguous_candidates(self) -> None:
        registry = self._registry(("a.template", "z.template"))
        analyzer = _RecordingAnalyzer(
            {
                "a.template": self._match("a.template", 0.9),
                "z.template": self._match("z.template", 0.9),
            }
        )

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition("z.scene", SceneRule(required_template_keys=("z.template",))),
                SceneDefinition("a.scene", SceneRule(required_template_keys=("a.template",))),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.AMBIGUOUS, result.status)
        self.assertEqual(("a.scene", "z.scene"), tuple(candidate.scene_key for candidate in result.candidates))

    def test_three_way_ambiguity_is_not_resolved_by_semantic_key(self) -> None:
        registry = self._registry(("a.template", "b.template", "c.template"))
        analyzer = _RecordingAnalyzer(
            {
                "a.template": self._match("a.template", 0.9),
                "b.template": self._match("b.template", 0.9),
                "c.template": self._match("c.template", 0.9),
            }
        )

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            {
                SceneDefinition("c.scene", SceneRule(required_template_keys=("c.template",))),
                SceneDefinition("a.scene", SceneRule(required_template_keys=("a.template",))),
                SceneDefinition("b.scene", SceneRule(required_template_keys=("b.template",))),
            },
            registry,
        )

        self.assertEqual(SceneClassificationStatus.AMBIGUOUS, result.status)
        self.assertEqual(
            ("a.scene", "b.scene", "c.scene"),
            tuple(candidate.scene_key for candidate in result.candidates),
        )

    def test_candidate_order_is_independent_of_input_order(self) -> None:
        registry = self._registry(("a.template", "b.template"))
        analyzer = _RecordingAnalyzer(
            {
                "a.template": self._match("a.template", 0.8),
                "b.template": self._match("b.template", 0.9),
            }
        )
        definitions = [
            SceneDefinition("b.scene", SceneRule(required_template_keys=("b.template",)), priority=1),
            SceneDefinition("a.scene", SceneRule(required_template_keys=("a.template",)), priority=1),
        ]
        reversed_definitions = list(reversed(definitions))

        first = SceneClassifier(analyzer).classify(self._screenshot(), definitions, registry)
        second = SceneClassifier(analyzer).classify(self._screenshot(), reversed_definitions, registry)

        self.assertEqual(
            tuple(candidate.scene_key for candidate in first.candidates),
            tuple(candidate.scene_key for candidate in second.candidates),
        )

    def test_numerically_near_equal_scores_are_ambiguous(self) -> None:
        registry = self._registry(("a.template", "b.template"))
        analyzer = _RecordingAnalyzer(
            {
                "a.template": self._match("a.template", 0.9000000000001),
                "b.template": self._match("b.template", 0.9),
            }
        )

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition("a.scene", SceneRule(required_template_keys=("a.template",))),
                SceneDefinition("b.scene", SceneRule(required_template_keys=("b.template",))),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.AMBIGUOUS, result.status)

    def test_threshold_equality_is_accepted(self) -> None:
        registry = self._registry(("city.hall",))
        analyzer = _RecordingAnalyzer({"city.hall": self._match("city.hall", 1.0)})

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition(
                    "city",
                    SceneRule(required_template_keys=("city.hall",), minimum_score=0.8),
                ),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.CLASSIFIED, result.status)
        self.assertEqual(0.8, result.score)

    def test_duplicate_scene_key_rejection(self) -> None:
        registry = self._registry(("city.hall",))

        result = SceneClassifier(_RecordingAnalyzer({})).classify(
            self._screenshot(),
            (
                SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),
                SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.duplicate_key")

    def test_duplicate_scene_key_after_normalization_is_rejected(self) -> None:
        registry = self._registry(("city.hall",))

        result = SceneClassifier(_RecordingAnalyzer({})).classify(
            self._screenshot(),
            (
                SceneDefinition(" city ", SceneRule(required_template_keys=("city.hall",))),
                SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.duplicate_key")

    def test_whitespace_only_scene_key_is_rejected_without_raw_exception(self) -> None:
        registry = self._registry(("city.hall",))
        definition = object.__new__(SceneDefinition)
        object.__setattr__(definition, "semantic_key", "   ")
        object.__setattr__(definition, "rule", SceneRule(required_template_keys=("city.hall",)))
        object.__setattr__(definition, "priority", 1)
        object.__setattr__(definition, "description", "")

        result = SceneClassifier(_RecordingAnalyzer({})).classify(
            self._screenshot(),
            (definition,),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.invalid_key")

    def test_unknown_template_key_rejection(self) -> None:
        registry = self._registry(("city.hall",))

        result = SceneClassifier(_RecordingAnalyzer({})).classify(
            self._screenshot(),
            (SceneDefinition("city", SceneRule(required_template_keys=("city.missing",))),),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.unknown_template_key")

    def test_required_forbidden_contradiction_is_rejected(self) -> None:
        registry = self._registry(("city.hall",))

        result = SceneClassifier(_RecordingAnalyzer({})).classify(
            self._screenshot(),
            (
                SceneDefinition(
                    "city",
                    SceneRule(
                        required_template_keys=("city.hall",),
                        forbidden_template_keys=("city.hall",),
                    ),
                ),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.contradictory_definition")

    def test_required_optional_overlap_is_rejected(self) -> None:
        registry = self._registry(("city.hall",))

        result = SceneClassifier(_RecordingAnalyzer({})).classify(
            self._screenshot(),
            (
                SceneDefinition(
                    "city",
                    SceneRule(
                        required_template_keys=("city.hall",),
                        optional_template_keys=("city.hall",),
                    ),
                ),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.contradictory_definition")

    def test_optional_forbidden_overlap_is_rejected(self) -> None:
        registry = self._registry(("city.hall",))

        result = SceneClassifier(_RecordingAnalyzer({})).classify(
            self._screenshot(),
            (
                SceneDefinition(
                    "city",
                    SceneRule(
                        optional_template_keys=("city.hall",),
                        forbidden_template_keys=("city.hall",),
                    ),
                ),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.contradictory_definition")

    def test_non_finite_score_and_invalid_priority_are_rejected(self) -> None:
        registry = self._registry(("city.hall",))
        rule = object.__new__(SceneRule)
        object.__setattr__(rule, "required_template_keys", ("city.hall",))
        object.__setattr__(rule, "optional_template_keys", ())
        object.__setattr__(rule, "forbidden_template_keys", ())
        object.__setattr__(rule, "minimum_score", float("nan"))
        definition = object.__new__(SceneDefinition)
        object.__setattr__(definition, "semantic_key", "city")
        object.__setattr__(definition, "rule", rule)
        object.__setattr__(definition, "priority", -1)
        object.__setattr__(definition, "description", "")

        result = SceneClassifier(_RecordingAnalyzer({})).classify(
            self._screenshot(),
            (definition,),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.invalid_score")
        self.assertDiagnostic(result.diagnostics, "scene.invalid_priority")

    def test_bool_priority_is_rejected(self) -> None:
        registry = self._registry(("city.hall",))
        definition = object.__new__(SceneDefinition)
        object.__setattr__(definition, "semantic_key", "city")
        object.__setattr__(definition, "rule", SceneRule(required_template_keys=("city.hall",)))
        object.__setattr__(definition, "priority", True)
        object.__setattr__(definition, "description", "")

        result = SceneClassifier(_RecordingAnalyzer({})).classify(
            self._screenshot(),
            (definition,),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.invalid_priority")

    def test_empty_scene_key_is_rejected_without_raw_exception(self) -> None:
        registry = self._registry(("city.hall",))
        definition = object.__new__(SceneDefinition)
        object.__setattr__(definition, "semantic_key", "")
        object.__setattr__(definition, "rule", SceneRule(required_template_keys=("city.hall",)))
        object.__setattr__(definition, "priority", 1)
        object.__setattr__(definition, "description", "")

        result = SceneClassifier(_RecordingAnalyzer({})).classify(
            self._screenshot(),
            (definition,),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.invalid_key")

    def test_duplicate_template_keys_inside_rule_group_are_rejected(self) -> None:
        registry = self._registry(("city.hall",))

        result = SceneClassifier(_RecordingAnalyzer({})).classify(
            self._screenshot(),
            (
                SceneDefinition(
                    "city",
                    SceneRule(required_template_keys=("city.hall", "city.hall")),
                ),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.duplicate_template_key")

    def test_duplicate_keys_inside_optional_and_forbidden_groups_are_rejected(self) -> None:
        registry = self._registry(("city.collect", "map.marker"))

        result = SceneClassifier(_RecordingAnalyzer({})).classify(
            self._screenshot(),
            (
                SceneDefinition(
                    "city",
                    SceneRule(
                        optional_template_keys=("city.collect", "city.collect"),
                        forbidden_template_keys=("map.marker", "map.marker"),
                    ),
                ),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertEqual(
            2,
            sum(1 for diagnostic in result.diagnostics if diagnostic.code == "scene.duplicate_template_key"),
        )

    def test_empty_definition_without_evidence_is_rejected(self) -> None:
        registry = self._registry(("city.hall",))

        result = SceneClassifier(_RecordingAnalyzer({})).classify(
            self._screenshot(),
            (SceneDefinition("city"),),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.empty_evidence")

    def test_template_scene_constraints_are_enforced(self) -> None:
        registry = self._registry(
            ("city.hall",),
            constraints={"city.hall": SceneConstraints(allowed=("map",), required=())},
        )

        result = SceneClassifier(_RecordingAnalyzer({})).classify(
            self._screenshot(),
            (SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.template_disallowed")

    def test_validation_diagnostic_order_is_deterministic(self) -> None:
        registry = self._registry(("city.hall",))
        definitions = (
            SceneDefinition("city", SceneRule(required_template_keys=("missing", "missing"))),
            SceneDefinition("city", SceneRule()),
        )
        classifier = SceneClassifier(_RecordingAnalyzer({}))

        first = classifier.classify(self._screenshot(), definitions, registry)
        second = classifier.classify(self._screenshot(), definitions, registry)

        self.assertEqual(
            [(item.code, item.field) for item in first.diagnostics],
            [(item.code, item.field) for item in second.diagnostics],
        )

    def test_same_template_is_evaluated_once_per_request(self) -> None:
        registry = self._registry(("city.hall",))
        analyzer = _RecordingAnalyzer({"city.hall": self._match("city.hall", 0.9)})

        SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition("city", SceneRule(required_template_keys=("city.hall",)), priority=1),
                SceneDefinition("home", SceneRule(required_template_keys=("city.hall",)), priority=2),
            ),
            registry,
        )

        self.assertEqual(("city.hall",), analyzer.called_keys())

    def test_failed_match_is_cached_once_per_request(self) -> None:
        registry = self._registry(("city.hall",))
        analyzer = _RecordingAnalyzer({"city.hall": self._invalid_match()})

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition("city", SceneRule(required_template_keys=("city.hall",)), priority=1),
                SceneDefinition("home", SceneRule(optional_template_keys=("city.hall",)), priority=2),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertEqual(("city.hall",), analyzer.called_keys())

    def test_cache_is_cleared_between_requests(self) -> None:
        registry = self._registry(("city.hall",))
        analyzer = _RecordingAnalyzer({"city.hall": self._match("city.hall", 0.9)})
        classifier = SceneClassifier(analyzer)
        definitions = (SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),)

        classifier.classify(self._screenshot(), definitions, registry)
        classifier.classify(self._screenshot(), definitions, registry)

        self.assertEqual(("city.hall", "city.hall"), analyzer.called_keys())

    def test_one_matcher_exception_does_not_skip_unrelated_template(self) -> None:
        registry = self._registry(("city.hall", "map.marker"))
        analyzer = _RecordingAnalyzer(
            {
                "city.hall": cv2.error("bad match"),
                "map.marker": self._match("map.marker", 0.9),
            }
        )

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (
                SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),
                SceneDefinition("map", SceneRule(required_template_keys=("map.marker",))),
            ),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertEqual(("city.hall", "map.marker"), analyzer.called_keys())

    def test_lower_level_scene_argument_is_explicit_for_constrained_template(self) -> None:
        registry = self._registry(
            ("city.hall",),
            constraints={"city.hall": SceneConstraints(allowed=("city",), required=())},
        )
        analyzer = _RecordingAnalyzer({"city.hall": self._match("city.hall", 0.9)})

        SceneClassifier(analyzer).classify(
            self._screenshot(),
            (SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),),
            registry,
        )

        self.assertEqual((("city.hall", "city"),), tuple(analyzer.calls))

    def test_matching_failure_becomes_structured_diagnostic(self) -> None:
        registry = self._registry(("city.hall",))
        analyzer = _RecordingAnalyzer({"city.hall": self._invalid_match()})

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.match_failed")

    def test_raw_opencv_exception_does_not_leak(self) -> None:
        registry = self._registry(("city.hall",))
        analyzer = _RecordingAnalyzer({"city.hall": cv2.error("bad match")})

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.match_failed")

    def test_malformed_matcher_result_becomes_structured_diagnostic(self) -> None:
        registry = self._registry(("city.hall",))
        analyzer = _RecordingAnalyzer({"city.hall": object()})

        result = SceneClassifier(analyzer).classify(
            self._screenshot(),
            (SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, result.status)
        self.assertDiagnostic(result.diagnostics, "scene.match_failed")

    def test_caller_image_and_scene_definitions_remain_unchanged(self) -> None:
        registry = self._registry(("city.hall",))
        analyzer = _RecordingAnalyzer({"city.hall": self._match("city.hall", 0.9)})
        screenshot = self._screenshot()
        original = screenshot.copy()
        definitions = [SceneDefinition("city", SceneRule(required_template_keys=("city.hall",)))]
        original_definitions = tuple(definitions)

        SceneClassifier(analyzer).classify(screenshot, definitions, registry)

        self.assertTrue(np.array_equal(original, screenshot))
        self.assertEqual(original_definitions, tuple(definitions))

    def test_repeated_classification_produces_identical_results(self) -> None:
        registry = self._registry(("city.hall",))
        analyzer = _RecordingAnalyzer({"city.hall": self._match("city.hall", 0.9)})
        classifier = SceneClassifier(analyzer)
        definitions = (SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),)

        first = classifier.classify(self._screenshot(), definitions, registry)
        second = classifier.classify(self._screenshot(), definitions, registry)

        self.assertEqual(first, second)

    def test_independent_classifier_instances_do_not_share_state(self) -> None:
        registry = self._registry(("city.hall", "map.marker"))
        city_classifier = SceneClassifier(_RecordingAnalyzer({"city.hall": self._match("city.hall", 0.9)}))
        map_classifier = SceneClassifier(_RecordingAnalyzer({"map.marker": self._match("map.marker", 0.9)}))

        city = city_classifier.classify(
            self._screenshot(),
            (SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),),
            registry,
        )
        map_result = map_classifier.classify(
            self._screenshot(),
            (SceneDefinition("map", SceneRule(required_template_keys=("map.marker",))),),
            registry,
        )

        self.assertEqual("city", city.scene_key)
        self.assertEqual("map", map_result.scene_key)

    def test_registry_remains_reusable_after_failed_classification(self) -> None:
        registry = self._registry(("city.hall",))
        failed = SceneClassifier(_RecordingAnalyzer({"city.hall": self._invalid_match()})).classify(
            self._screenshot(),
            (SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),),
            registry,
        )
        recovered = SceneClassifier(_RecordingAnalyzer({"city.hall": self._match("city.hall", 0.9)})).classify(
            self._screenshot(),
            (SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, failed.status)
        self.assertEqual(SceneClassificationStatus.CLASSIFIED, recovered.status)

    def test_classifier_reuse_after_failed_request(self) -> None:
        registry = self._registry(("city.hall",))
        analyzer = _RecordingAnalyzer({"city.hall": self._invalid_match()})
        classifier = SceneClassifier(analyzer)

        failed = classifier.classify(
            self._screenshot(),
            (SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),),
            registry,
        )
        analyzer.results["city.hall"] = self._match("city.hall", 0.9)
        recovered = classifier.classify(
            self._screenshot(),
            (SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),),
            registry,
        )

        self.assertEqual(SceneClassificationStatus.INVALID, failed.status)
        self.assertEqual(SceneClassificationStatus.CLASSIFIED, recovered.status)

    def test_no_machine_specific_paths_appear_in_diagnostics(self) -> None:
        registry = self._registry(("city.hall",))
        result = SceneClassifier(_RecordingAnalyzer({"city.hall": self._invalid_match()})).classify(
            self._screenshot(),
            (SceneDefinition("city", SceneRule(required_template_keys=("city.hall",))),),
            registry,
        )

        text = "\n".join(diagnostic.message for diagnostic in result.diagnostics)
        self.assertNotIn(str(Path.cwd()), text)

    def test_result_model_rejects_contradictory_states(self) -> None:
        diagnostic = ValidationDiagnostic(code="scene.test", message="test diagnostic")
        candidate = SceneCandidateResult(scene_key="city", score=0.5, priority=1)
        with self.assertRaises(ValueError):
            SceneClassificationResult(status=SceneClassificationStatus.CLASSIFIED)
        with self.assertRaises(ValueError):
            SceneClassificationResult(status=SceneClassificationStatus.UNKNOWN, scene_key="city")
        with self.assertRaises(ValueError):
            SceneClassificationResult(
                status=SceneClassificationStatus.AMBIGUOUS,
                candidates=(candidate,),
                diagnostics=(diagnostic,),
            )
        with self.assertRaises(ValueError):
            SceneClassificationResult(status=SceneClassificationStatus.INVALID)

    def assertDiagnostic(self, diagnostics: tuple[ValidationDiagnostic, ...], code: str) -> None:
        self.assertIn(code, {diagnostic.code for diagnostic in diagnostics})

    @staticmethod
    def _match(key: str, confidence: float) -> DetectionResult:
        return DetectionResult(matched_semantic_key=key, confidence=confidence)

    @staticmethod
    def _no_match() -> DetectionResult:
        return DetectionResult(
            matched_semantic_key=None,
            confidence=0.0,
            metadata=MatchingMetadata(
                diagnostics=(
                    ValidationDiagnostic(
                        code="match.below_threshold",
                        field="confidence",
                        message="Best match confidence is below the template threshold.",
                    ),
                )
            ),
        )

    @staticmethod
    def _invalid_match() -> DetectionResult:
        return DetectionResult(
            matched_semantic_key=None,
            confidence=0.0,
            metadata=MatchingMetadata(
                diagnostics=(
                    ValidationDiagnostic(
                        code="match.failed",
                        field="template",
                        message="Template matching failed.",
                    ),
                )
            ),
        )

    @staticmethod
    def _screenshot() -> np.ndarray:
        return np.zeros((20, 20), dtype=np.uint8)

    @staticmethod
    def _template_image() -> np.ndarray:
        template = np.zeros((4, 5), dtype=np.uint8)
        for y in range(template.shape[0]):
            for x in range(template.shape[1]):
                template[y, x] = 50 + ((x * 31 + y * 17) % 180)
        return template

    @staticmethod
    def _registry(
        keys: tuple[str, ...],
        *,
        constraints: dict[str, SceneConstraints] | None = None,
    ) -> TemplateRegistry:
        constraints = constraints or {}
        templates = tuple(
            TemplateDefinition(
                semantic_key=key,
                template_pack_version="2026.07",
                language="en",
                resolution_profile="phone.720p",
                source=Path(f"templates/{key.replace('.', '_')}.png"),
                region_of_interest=RegionOfInterest(0, 0, 20, 20),
                confidence_threshold=0.9,
                scale_range=ScaleRange(),
                scene_constraints=constraints.get(key, SceneConstraints()),
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


class _RecordingAnalyzer:
    def __init__(self, results: dict[str, object]) -> None:
        self.results = dict(results)
        self.calls: list[tuple[str, str | None]] = []

    def match(
        self,
        screenshot: object,
        semantic_key: str,
        registry: TemplateRegistry,
        *,
        scene: str | None = None,
    ) -> DetectionResult:
        self.calls.append((semantic_key, scene))
        result = self.results.get(semantic_key)
        if isinstance(result, BaseException):
            raise result
        if result is None:
            return SceneClassifierTest._no_match()
        return result

    def called_keys(self) -> tuple[str, ...]:
        return tuple(key for key, _scene in self.calls)


if __name__ == "__main__":
    unittest.main()
