from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

import cv2

from .image_matching import ImageInput, NormalizedImage, TemplateScreenAnalyzer
from .template_models import DetectionResult, MatchingMetadata, ValidationDiagnostic
from .template_registry import TemplateRegistry, TemplateRegistryError

_EPSILON = 1e-12
_NO_MATCH_DIAGNOSTIC_CODES = {
    "match.below_threshold",
    "match.no_eligible_scale",
}


class SceneClassificationStatus(str, Enum):
    CLASSIFIED = "classified"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    INVALID = "invalid"


@dataclass(frozen=True)
class SceneRule:
    required_template_keys: tuple[str, ...] = ()
    optional_template_keys: tuple[str, ...] = ()
    forbidden_template_keys: tuple[str, ...] = ()
    minimum_score: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_template_keys",
            _normalize_keys(self.required_template_keys),
        )
        object.__setattr__(
            self,
            "optional_template_keys",
            _normalize_keys(self.optional_template_keys),
        )
        object.__setattr__(
            self,
            "forbidden_template_keys",
            _normalize_keys(self.forbidden_template_keys),
        )
        object.__setattr__(
            self,
            "minimum_score",
            _require_score(self.minimum_score, "minimum score"),
        )


@dataclass(frozen=True)
class SceneDefinition:
    semantic_key: str
    rule: SceneRule = field(default_factory=SceneRule)
    priority: int = 100
    description: str = ""

    def __post_init__(self) -> None:
        semantic_key = _normalize_key(self.semantic_key)
        if not semantic_key:
            raise ValueError("scene semantic key must be a non-empty string.")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool) or self.priority < 0:
            raise ValueError("scene priority must be a non-negative integer.")
        object.__setattr__(self, "semantic_key", semantic_key)
        object.__setattr__(self, "description", str(self.description))


@dataclass(frozen=True)
class SceneClassificationRequest:
    screenshot: ImageInput | NormalizedImage
    scene_definitions: tuple[SceneDefinition, ...]
    registry: TemplateRegistry
    analyzer: TemplateScreenAnalyzer | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_definitions", tuple(self.scene_definitions))


@dataclass(frozen=True)
class SceneCandidateResult:
    scene_key: str
    score: float
    priority: int
    satisfied_required: tuple[str, ...] = ()
    satisfied_optional: tuple[str, ...] = ()
    present_forbidden: tuple[str, ...] = ()
    missing_required: tuple[str, ...] = ()
    diagnostics: tuple[ValidationDiagnostic, ...] = ()
    detection_results: tuple[DetectionResult, ...] = ()

    def __post_init__(self) -> None:
        scene_key = _normalize_key(self.scene_key)
        if not scene_key:
            raise ValueError("candidate scene key must be a non-empty string.")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool) or self.priority < 0:
            raise ValueError("candidate priority must be a non-negative integer.")
        object.__setattr__(self, "scene_key", scene_key)
        object.__setattr__(self, "score", _require_score(self.score, "scene score"))
        object.__setattr__(self, "satisfied_required", tuple(self.satisfied_required))
        object.__setattr__(self, "satisfied_optional", tuple(self.satisfied_optional))
        object.__setattr__(self, "present_forbidden", tuple(self.present_forbidden))
        object.__setattr__(self, "missing_required", tuple(self.missing_required))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "detection_results", tuple(self.detection_results))

    @property
    def qualifies(self) -> bool:
        return (
            not self.missing_required
            and not self.present_forbidden
            and not self.diagnostics
        )


@dataclass(frozen=True)
class SceneClassificationResult:
    status: SceneClassificationStatus
    scene_key: str | None = None
    score: float = 0.0
    candidates: tuple[SceneCandidateResult, ...] = ()
    diagnostics: tuple[ValidationDiagnostic, ...] = ()
    metadata: MatchingMetadata = field(default_factory=MatchingMetadata)

    def __post_init__(self) -> None:
        status = SceneClassificationStatus(self.status)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "score", _require_score(self.score, "classification score"))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        scene_key = self.scene_key.strip() if isinstance(self.scene_key, str) else self.scene_key
        if scene_key == "":
            scene_key = None
        object.__setattr__(self, "scene_key", scene_key)
        if status == SceneClassificationStatus.CLASSIFIED:
            if not isinstance(scene_key, str) or not scene_key:
                raise ValueError("classified scene results require exactly one selected scene key.")
            if self.diagnostics:
                raise ValueError("classified scene results must not contain diagnostics.")
        elif status == SceneClassificationStatus.UNKNOWN:
            if scene_key is not None:
                raise ValueError("unknown scene results must not select a scene.")
        elif status == SceneClassificationStatus.AMBIGUOUS:
            if scene_key is not None:
                raise ValueError("ambiguous scene results must not select a scene.")
            if len(self.candidates) < 2:
                raise ValueError("ambiguous scene results require at least two candidates.")
            if not self.diagnostics:
                raise ValueError("ambiguous scene results require diagnostics.")
        elif status == SceneClassificationStatus.INVALID:
            if scene_key is not None:
                raise ValueError("invalid scene results must not select a scene.")
            if not self.diagnostics:
                raise ValueError("invalid scene results require diagnostics.")


class SceneClassifier:
    """Deterministic scene classifier built on semantic template matching.

    Scoring is intentionally simple and bounded. A scene is eligible only when
    all required templates match and no forbidden template matches. Required
    evidence contributes 80% of the score and optional supporting evidence can
    contribute up to the remaining 20% using the strongest optional match, so
    additional optional evidence never lowers a required scene's score.
    Optional-only scenes use the average confidence of matched optional
    evidence. Threshold equality is accepted. Candidate ordering is
    deterministic by satisfied required count, score, priority, then semantic
    scene key. If the best candidates still tie before the semantic-key
    ordering step, the result is reported as ambiguous instead of choosing a
    scene arbitrarily.
    """

    def __init__(self, analyzer: TemplateScreenAnalyzer | None = None) -> None:
        self.analyzer = analyzer or TemplateScreenAnalyzer()

    def classify(
        self,
        screenshot: ImageInput | NormalizedImage,
        scene_definitions: Sequence[SceneDefinition],
        registry: TemplateRegistry,
        *,
        analyzer: TemplateScreenAnalyzer | None = None,
    ) -> SceneClassificationResult:
        request = SceneClassificationRequest(
            screenshot=screenshot,
            scene_definitions=tuple(scene_definitions),
            registry=registry,
            analyzer=analyzer,
        )
        return self.classify_request(request)

    def classify_request(self, request: SceneClassificationRequest) -> SceneClassificationResult:
        definitions = tuple(request.scene_definitions)
        diagnostics = self.validate_definitions(definitions, request.registry)
        if diagnostics:
            return self._result(
                status=SceneClassificationStatus.INVALID,
                diagnostics=diagnostics,
            )

        analyzer = request.analyzer or self.analyzer
        frame = request.screenshot.pixels if isinstance(request.screenshot, NormalizedImage) else request.screenshot
        detection_cache: dict[str, DetectionResult] = {}
        matching_diagnostics: list[ValidationDiagnostic] = []

        for template_key in self._template_keys(definitions):
            detection = self._match_once(frame, template_key, request.registry, analyzer)
            detection_cache[template_key] = detection
            if self._is_invalid_match(detection):
                matching_diagnostics.extend(
                    self._scene_diagnostic_from_match(template_key, detection)
                )

        candidates = tuple(
            self._candidate_for(definition, detection_cache)
            for definition in sorted(definitions, key=lambda item: item.semantic_key)
        )
        if matching_diagnostics:
            return self._result(
                status=SceneClassificationStatus.INVALID,
                candidates=candidates,
                diagnostics=tuple(matching_diagnostics),
            )

        qualifying = tuple(
            candidate
            for candidate in candidates
            if candidate.qualifies
            and (candidate.satisfied_required or candidate.satisfied_optional)
            and candidate.score + _EPSILON >= self._definition_by_key(definitions, candidate.scene_key).rule.minimum_score
        )
        if not qualifying:
            return self._result(
                status=SceneClassificationStatus.UNKNOWN,
                candidates=candidates,
            )

        ranked = tuple(sorted(qualifying, key=self._ranking_key))
        best = ranked[0]
        tied = tuple(
            candidate
            for candidate in ranked
            if self._ambiguous_tie_key(candidate) == self._ambiguous_tie_key(best)
        )
        if len(tied) > 1:
            return self._result(
                status=SceneClassificationStatus.AMBIGUOUS,
                score=best.score,
                candidates=ranked,
                diagnostics=(
                    self._diagnostic(
                        "scene.ambiguous",
                        "scene_definitions",
                        "Multiple scene definitions satisfy the same deterministic ranking criteria.",
                    ),
                ),
            )

        return self._result(
            status=SceneClassificationStatus.CLASSIFIED,
            scene_key=best.scene_key,
            score=best.score,
            candidates=ranked,
        )

    def validate_definitions(
        self,
        scene_definitions: Sequence[SceneDefinition],
        registry: TemplateRegistry,
    ) -> tuple[ValidationDiagnostic, ...]:
        diagnostics: list[ValidationDiagnostic] = []
        definitions = tuple(scene_definitions)
        if not definitions:
            return (
                self._diagnostic(
                    "scene.empty_definitions",
                    "scene_definitions",
                    "At least one scene definition is required.",
                ),
            )

        seen_scene_keys: set[str] = set()
        for index, definition in enumerate(sorted(definitions, key=self._definition_sort_key)):
            field = f"scene_definitions[{index}]"
            semantic_key = getattr(definition, "semantic_key", None)
            if not isinstance(semantic_key, str) or not semantic_key.strip():
                diagnostics.append(
                    self._diagnostic(
                        "scene.invalid_key",
                        f"{field}.semantic_key",
                        "Scene semantic key must be a non-empty string.",
                    )
                )
                continue
            semantic_key = semantic_key.strip()
            priority = getattr(definition, "priority", None)
            if not isinstance(priority, int) or isinstance(priority, bool) or priority < 0:
                diagnostics.append(
                    self._diagnostic(
                        "scene.invalid_priority",
                        f"{field}.priority",
                        "Scene priority must be a non-negative integer.",
                    )
                )
            if semantic_key in seen_scene_keys:
                diagnostics.append(
                    self._diagnostic(
                        "scene.duplicate_key",
                        f"{field}.semantic_key",
                        f"Duplicate scene semantic key: {semantic_key}",
                    )
                )
            seen_scene_keys.add(semantic_key)
            diagnostics.extend(self._validate_rule(definition, registry, field))
        return tuple(sorted(diagnostics, key=lambda item: (item.field, item.code, item.message)))

    def _validate_rule(
        self,
        definition: SceneDefinition,
        registry: TemplateRegistry,
        field: str,
    ) -> tuple[ValidationDiagnostic, ...]:
        diagnostics: list[ValidationDiagnostic] = []
        rule = getattr(definition, "rule", None)
        if not isinstance(rule, SceneRule):
            return (
                self._diagnostic(
                    "scene.invalid_rule",
                    f"{field}.rule",
                    "Scene rule must be a SceneRule.",
                ),
            )
        if (
            not isinstance(rule.minimum_score, int | float)
            or isinstance(rule.minimum_score, bool)
            or not math.isfinite(float(rule.minimum_score))
            or float(rule.minimum_score) < 0.0
            or float(rule.minimum_score) > 1.0
        ):
            diagnostics.append(
                self._diagnostic(
                    "scene.invalid_score",
                    f"{field}.rule.minimum_score",
                    "Scene minimum score must be finite and between 0.0 and 1.0.",
                )
            )
        groups: dict[str, tuple[str, ...] | None] = {
            "required_template_keys": _safe_keys(rule.required_template_keys),
            "optional_template_keys": _safe_keys(rule.optional_template_keys),
            "forbidden_template_keys": _safe_keys(rule.forbidden_template_keys),
        }
        for group_name, keys in groups.items():
            if keys is None:
                diagnostics.append(
                    self._diagnostic(
                        "scene.invalid_template_key",
                        f"{field}.rule.{group_name}",
                        "Scene template key groups must contain only non-empty strings.",
                    )
                )
                groups[group_name] = ()
        if not any(groups.values()):
            diagnostics.append(
                self._diagnostic(
                    "scene.empty_evidence",
                    f"{field}.rule",
                    "Scene definitions require required, optional, or forbidden template evidence.",
                )
            )

        for group_name, keys in groups.items():
            duplicate_keys = _duplicates(keys)
            for template_key in duplicate_keys:
                diagnostics.append(
                    self._diagnostic(
                        "scene.duplicate_template_key",
                        f"{field}.rule.{group_name}",
                        f"Duplicate template key in scene rule: {template_key}",
                    )
                )

        required = set(groups["required_template_keys"] or ())
        optional = set(groups["optional_template_keys"] or ())
        forbidden = set(groups["forbidden_template_keys"] or ())
        overlaps = (
            (required & forbidden)
            | (optional & forbidden)
            | (required & optional)
        )
        for template_key in sorted(overlaps):
            diagnostics.append(
                self._diagnostic(
                    "scene.contradictory_definition",
                    f"{field}.rule",
                    f"Template key appears in contradictory scene rule groups: {template_key}",
                )
            )

        for template_key in sorted(required | optional | forbidden):
            try:
                template = registry.get(template_key)
            except TemplateRegistryError:
                diagnostics.append(
                    self._diagnostic(
                        "scene.unknown_template_key",
                        f"{field}.rule",
                        f"Scene references an unknown template semantic key: {template_key}",
                    )
                )
                continue
            constraints = template.scene_constraints
            if constraints.allowed and definition.semantic_key not in constraints.allowed:
                diagnostics.append(
                    self._diagnostic(
                        "scene.template_disallowed",
                        f"{field}.rule",
                        "Scene references a template that does not allow this scene.",
                    )
                )
            if constraints.required and definition.semantic_key not in constraints.required:
                diagnostics.append(
                    self._diagnostic(
                        "scene.template_required_mismatch",
                        f"{field}.rule",
                        "Scene references a template whose required scene constraint does not include this scene.",
                    )
                )
        return tuple(diagnostics)

    @staticmethod
    def _definition_sort_key(definition: Any) -> str:
        semantic_key = getattr(definition, "semantic_key", "")
        return semantic_key.strip() if isinstance(semantic_key, str) else ""

    def _match_once(
        self,
        screenshot: ImageInput,
        template_key: str,
        registry: TemplateRegistry,
        analyzer: TemplateScreenAnalyzer,
    ) -> DetectionResult:
        try:
            result = analyzer.match(
                screenshot,
                template_key,
                registry,
                scene=self._matching_scene_for_template(template_key, registry),
            )
            if not isinstance(result, DetectionResult):
                return self._invalid_match_result(analyzer, template_key)
            return result
        except (cv2.error, KeyError, OSError, TypeError, ValueError, AttributeError):
            return self._invalid_match_result(analyzer, template_key)

    def _invalid_match_result(
        self,
        analyzer: TemplateScreenAnalyzer,
        template_key: str,
    ) -> DetectionResult:
        return DetectionResult(
            matched_semantic_key=None,
            confidence=0.0,
            metadata=MatchingMetadata(
                matcher=analyzer.__class__.__name__,
                diagnostics=(
                    self._diagnostic(
                        "scene.match_failed",
                        f"templates.{template_key}",
                        "Template matcher failed while classifying a scene.",
                    ),
                ),
            ),
        )

    @staticmethod
    def _matching_scene_for_template(template_key: str, registry: TemplateRegistry) -> str | None:
        try:
            constraints = registry.get(template_key).scene_constraints
        except TemplateRegistryError:
            return None
        if constraints.required:
            return sorted(constraints.required)[0]
        if len(constraints.allowed) == 1:
            return constraints.allowed[0]
        return None

    @staticmethod
    def _template_keys(definitions: Sequence[SceneDefinition]) -> tuple[str, ...]:
        keys: set[str] = set()
        for definition in definitions:
            keys.update(definition.rule.required_template_keys)
            keys.update(definition.rule.optional_template_keys)
            keys.update(definition.rule.forbidden_template_keys)
        return tuple(sorted(keys))

    @staticmethod
    def _definition_by_key(
        definitions: Sequence[SceneDefinition],
        scene_key: str,
    ) -> SceneDefinition:
        return next(definition for definition in definitions if definition.semantic_key == scene_key)

    def _candidate_for(
        self,
        definition: SceneDefinition,
        detection_cache: dict[str, DetectionResult],
    ) -> SceneCandidateResult:
        rule = definition.rule
        required = tuple(
            key
            for key in rule.required_template_keys
            if self._is_positive(detection_cache[key], key)
        )
        optional = tuple(
            key
            for key in rule.optional_template_keys
            if self._is_positive(detection_cache[key], key)
        )
        forbidden = tuple(
            key
            for key in rule.forbidden_template_keys
            if self._is_positive(detection_cache[key], key)
        )
        missing_required = tuple(
            key
            for key in rule.required_template_keys
            if key not in required
        )
        score = self._score(
            required_results=tuple(detection_cache[key] for key in required),
            optional_results=tuple(detection_cache[key] for key in optional),
            has_required=bool(rule.required_template_keys),
        )
        detections = tuple(
            detection_cache[key]
            for key in sorted(
                set(rule.required_template_keys)
                | set(rule.optional_template_keys)
                | set(rule.forbidden_template_keys)
            )
        )
        return SceneCandidateResult(
            scene_key=definition.semantic_key,
            score=score,
            priority=definition.priority,
            satisfied_required=required,
            satisfied_optional=optional,
            present_forbidden=forbidden,
            missing_required=missing_required,
            detection_results=detections,
        )

    @staticmethod
    def _is_positive(detection: DetectionResult, template_key: str) -> bool:
        return detection.matched_semantic_key == template_key and detection.confidence >= 0.0

    @staticmethod
    def _score(
        *,
        required_results: tuple[DetectionResult, ...],
        optional_results: tuple[DetectionResult, ...],
        has_required: bool,
    ) -> float:
        required_score = _average_confidence(required_results)
        if has_required:
            optional_score = _maximum_confidence(optional_results)
            return _bounded_score((required_score * 0.8) + (optional_score * 0.2))
        optional_score = _average_confidence(optional_results)
        return _bounded_score(optional_score)

    @staticmethod
    def _is_invalid_match(detection: DetectionResult) -> bool:
        if detection.matched_semantic_key is not None:
            return False
        diagnostics = detection.metadata.diagnostics
        if not diagnostics:
            return False
        return any(diagnostic.code not in _NO_MATCH_DIAGNOSTIC_CODES for diagnostic in diagnostics)

    def _scene_diagnostic_from_match(
        self,
        template_key: str,
        detection: DetectionResult,
    ) -> tuple[ValidationDiagnostic, ...]:
        if not detection.metadata.diagnostics:
            return (
                self._diagnostic(
                    "scene.match_failed",
                    f"templates.{template_key}",
                    "Template matcher returned an invalid result while classifying a scene.",
                ),
            )
        return tuple(
            self._diagnostic(
                "scene.match_failed",
                f"templates.{template_key}",
                f"Template matcher diagnostic {diagnostic.code} prevented scene classification.",
            )
            for diagnostic in detection.metadata.diagnostics
        )

    @staticmethod
    def _ranking_key(candidate: SceneCandidateResult) -> tuple[int, float, int, str]:
        return (
            -len(candidate.satisfied_required),
            -candidate.score,
            candidate.priority,
            candidate.scene_key,
        )

    @staticmethod
    def _ambiguous_tie_key(candidate: SceneCandidateResult) -> tuple[int, float, int]:
        return (
            len(candidate.satisfied_required),
            round(candidate.score, 12),
            candidate.priority,
        )

    @staticmethod
    def _result(
        *,
        status: SceneClassificationStatus,
        scene_key: str | None = None,
        score: float = 0.0,
        candidates: tuple[SceneCandidateResult, ...] = (),
        diagnostics: tuple[ValidationDiagnostic, ...] = (),
    ) -> SceneClassificationResult:
        return SceneClassificationResult(
            status=status,
            scene_key=scene_key,
            score=score,
            candidates=candidates,
            diagnostics=diagnostics,
            metadata=MatchingMetadata(
                matcher="SceneClassifier",
                candidate_count=len(candidates),
                diagnostics=diagnostics,
            ),
        )

    @staticmethod
    def _diagnostic(code: str, field: str, message: str) -> ValidationDiagnostic:
        return ValidationDiagnostic(code=code, field=field, message=message)


def _normalize_keys(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(_normalize_key(value) for value in values)


def _normalize_key(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("semantic keys must be strings.")
    return value.strip()


def _safe_keys(values: Any) -> tuple[str, ...] | None:
    if not isinstance(values, Sequence) or isinstance(values, str):
        return None
    keys: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            return None
        keys.append(value.strip())
    return tuple(keys)


def _require_score(value: Any, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number.")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{field_name} must be finite.")
    if score < 0.0 or score > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    return score


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _average_confidence(results: tuple[DetectionResult, ...]) -> float:
    if not results:
        return 0.0
    return _bounded_score(
        sum(_bounded_score(result.confidence) for result in results) / len(results)
    )


def _maximum_confidence(results: tuple[DetectionResult, ...]) -> float:
    if not results:
        return 0.0
    return _bounded_score(max(_bounded_score(result.confidence) for result in results))


def _bounded_score(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(max(0.0, min(1.0, float(value))), 12)
