from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from pathlib import Path
from typing import Any

import cv2

from .image_matching import ImageInput, TemplateImageNormalizer, TemplateScreenAnalyzer
from .replay_dataset import (
    ReplayCaseDefinition,
    ReplayDatasetDefinition,
    ReplayDatasetLoader,
    ReplayDetectionExpectation,
    ReplayExpectation,
    ReplaySceneExpectation,
)
from .scene_classification import (
    SceneClassificationResult,
    SceneClassificationStatus,
    SceneClassifier,
    SceneDefinition,
    SceneRule,
)
from .template_models import DetectionResult, TemplateDefinition, TemplatePack, ValidationDiagnostic
from .template_registry import TemplateRegistry, TemplateRegistryError

_EPSILON = 1e-12


class ThresholdCalibrationTarget(str, Enum):
    TEMPLATE_DETECTION = "template_detection_threshold"
    SCENE_MINIMUM_SCORE = "scene_minimum_score"


class ThresholdObservationStatus(str, Enum):
    TRUE_POSITIVE = "true_positive"
    TRUE_NEGATIVE = "true_negative"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    INVALID_CASE = "invalid_case"
    INPUT_ERROR = "input_error"
    PIPELINE_ERROR = "pipeline_error"


@dataclass(frozen=True)
class ThresholdCandidate:
    identifier: str
    target: ThresholdCalibrationTarget | str
    value: float

    def __post_init__(self) -> None:
        identifier = _normalize_text(self.identifier, "candidate identifier")
        target = ThresholdCalibrationTarget(self.target)
        value = _require_threshold(self.value, "candidate threshold")
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "value", value)

    @property
    def semantic_id(self) -> str:
        return f"{self.target.value}:{self.identifier}:{self.value:.12g}"


@dataclass(frozen=True)
class ThresholdCalibrationCase:
    case_id: str
    target: ThresholdCalibrationTarget | str
    expected_positive: bool
    screenshot_path: str
    template_key: str | None = None
    scene_key: str | None = None
    label: str = ""
    metadata: tuple[tuple[str, str], ...] = ()
    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        target = ThresholdCalibrationTarget(self.target)
        case_id = _normalize_text(self.case_id, "calibration case identifier")
        screenshot_path = _normalize_text(self.screenshot_path, "screenshot path")
        if not isinstance(self.expected_positive, bool):
            raise ValueError("expected_positive must be boolean.")
        template_key = _normalize_optional_text(self.template_key, "template key")
        scene_key = _normalize_optional_text(self.scene_key, "scene key")
        if target == ThresholdCalibrationTarget.TEMPLATE_DETECTION and template_key is None:
            raise ValueError("template detection calibration cases require a template key.")
        if target == ThresholdCalibrationTarget.SCENE_MINIMUM_SCORE and scene_key is None:
            raise ValueError("scene calibration cases require a scene key.")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "screenshot_path", screenshot_path)
        object.__setattr__(self, "template_key", template_key)
        object.__setattr__(self, "scene_key", scene_key)
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))
        object.__setattr__(self, "diagnostics", _sorted_diagnostics(self.diagnostics))

    @classmethod
    def from_replay_case(
        cls,
        case: ReplayCaseDefinition,
        target: ThresholdCalibrationTarget | str,
    ) -> ThresholdCalibrationCase:
        target = ThresholdCalibrationTarget(target)
        if target == ThresholdCalibrationTarget.TEMPLATE_DETECTION:
            expectation = case.expectation.detection if case.expectation and case.expectation.detection else None
            if expectation is None or expectation.expected_match is None:
                raise ValueError("detection calibration requires explicit expected_match ground truth.")
            template_key = case.template_key or expectation.semantic_key
            return cls(
                case_id=case.case_id,
                target=target,
                expected_positive=expectation.expected_match,
                screenshot_path=case.screenshot_path,
                template_key=template_key,
                label=case.label,
                diagnostics=case.diagnostics,
            )
        expectation = case.expectation.scene if case.expectation and case.expectation.scene else None
        if expectation is None:
            raise ValueError("scene calibration requires explicit scene ground truth.")
        scene_key = expectation.semantic_scene_key
        if scene_key is None:
            raise ValueError("scene calibration requires a semantic scene key.")
        return cls(
            case_id=case.case_id,
            target=target,
            expected_positive=expectation.status == SceneClassificationStatus.CLASSIFIED,
            screenshot_path=case.screenshot_path,
            scene_key=scene_key,
            label=case.label,
            diagnostics=case.diagnostics,
        )


@dataclass(frozen=True)
class ThresholdCalibrationRequest:
    candidates: tuple[ThresholdCandidate, ...]
    cases: tuple[ThresholdCalibrationCase, ...] = ()
    replay_dataset: ReplayDatasetDefinition | None = None
    manifest_path: str | Path | None = None
    dataset_root: str | Path | None = None
    registry: TemplateRegistry | None = None
    analyzer: TemplateScreenAnalyzer | None = None
    scene_classifier: SceneClassifier | None = None
    scene_definitions: tuple[SceneDefinition, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", _sorted_candidates(self.candidates))
        object.__setattr__(self, "cases", tuple(sorted(tuple(self.cases), key=lambda item: item.case_id)))
        object.__setattr__(self, "scene_definitions", tuple(self.scene_definitions))


@dataclass(frozen=True)
class ThresholdCaseObservation:
    case_id: str
    candidate_id: str
    target: ThresholdCalibrationTarget | str
    expected_positive: bool
    actual_positive: bool | None
    status: ThresholdObservationStatus | str
    confidence: float | None = None
    score: float | None = None
    matched_identifier: str | None = None
    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        status = ThresholdObservationStatus(self.status)
        if not isinstance(self.expected_positive, bool):
            raise ValueError("expected_positive must be boolean.")
        if self.actual_positive is not None and not isinstance(self.actual_positive, bool):
            raise ValueError("actual_positive must be boolean when provided.")
        if status in _CONFUSION_STATUSES and self.actual_positive is None:
            raise ValueError("confusion observations require actual_positive.")
        if status not in _CONFUSION_STATUSES and self.actual_positive is not None:
            raise ValueError("non-confusion observations must not set actual_positive.")
        if self.confidence is not None:
            _require_threshold(self.confidence, "observation confidence")
        if self.score is not None:
            _require_threshold(self.score, "observation score")
        object.__setattr__(self, "case_id", _normalize_text(self.case_id, "case identifier"))
        object.__setattr__(self, "candidate_id", _normalize_text(self.candidate_id, "candidate identifier"))
        object.__setattr__(self, "target", ThresholdCalibrationTarget(self.target))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "matched_identifier", _normalize_optional_text(self.matched_identifier, "matched identifier"))
        object.__setattr__(self, "diagnostics", _sorted_diagnostics(self.diagnostics))


@dataclass(frozen=True)
class ThresholdConfusionMatrix:
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def __post_init__(self) -> None:
        for field_name in ("true_positives", "true_negatives", "false_positives", "false_negatives"):
            _require_non_negative_int(getattr(self, field_name), field_name)

    @classmethod
    def from_observations(cls, observations: tuple[ThresholdCaseObservation, ...]) -> ThresholdConfusionMatrix:
        return cls(
            true_positives=sum(1 for item in observations if item.status == ThresholdObservationStatus.TRUE_POSITIVE),
            true_negatives=sum(1 for item in observations if item.status == ThresholdObservationStatus.TRUE_NEGATIVE),
            false_positives=sum(1 for item in observations if item.status == ThresholdObservationStatus.FALSE_POSITIVE),
            false_negatives=sum(1 for item in observations if item.status == ThresholdObservationStatus.FALSE_NEGATIVE),
        )

    @property
    def evaluated_cases(self) -> int:
        return self.true_positives + self.true_negatives + self.false_positives + self.false_negatives


@dataclass(frozen=True)
class ThresholdMetrics:
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    evaluated_cases: int
    invalid_cases: int
    pipeline_error_cases: int
    precision: float
    recall: float
    specificity: float
    accuracy: float
    f1_score: float

    def __post_init__(self) -> None:
        for field_name in (
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "evaluated_cases",
            "invalid_cases",
            "pipeline_error_cases",
        ):
            _require_non_negative_int(getattr(self, field_name), field_name)
        for field_name in ("precision", "recall", "specificity", "accuracy", "f1_score"):
            value = getattr(self, field_name)
            if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite.")
        expected = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        if self.evaluated_cases != expected:
            raise ValueError("evaluated case count must match confusion-matrix counts.")

    @classmethod
    def from_observations(cls, observations: tuple[ThresholdCaseObservation, ...]) -> ThresholdMetrics:
        matrix = ThresholdConfusionMatrix.from_observations(observations)
        tp = matrix.true_positives
        tn = matrix.true_negatives
        fp = matrix.false_positives
        fn = matrix.false_negatives
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        specificity = _ratio(tn, tn + fp)
        accuracy = _ratio(tp + tn, matrix.evaluated_cases)
        f1_score = _ratio(2.0 * precision * recall, precision + recall)
        return cls(
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            evaluated_cases=matrix.evaluated_cases,
            invalid_cases=sum(1 for item in observations if item.status == ThresholdObservationStatus.INVALID_CASE),
            pipeline_error_cases=sum(
                1
                for item in observations
                if item.status in (ThresholdObservationStatus.PIPELINE_ERROR, ThresholdObservationStatus.INPUT_ERROR)
            ),
            precision=precision,
            recall=recall,
            specificity=specificity,
            accuracy=accuracy,
            f1_score=f1_score,
        )


@dataclass(frozen=True)
class ThresholdCandidateResult:
    candidate: ThresholdCandidate
    observations: tuple[ThresholdCaseObservation, ...]
    confusion_matrix: ThresholdConfusionMatrix = field(init=False)
    metrics: ThresholdMetrics = field(init=False)
    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        observations = tuple(sorted(tuple(self.observations), key=lambda item: item.case_id))
        if not observations:
            raise ValueError("candidate results require at least one observation.")
        if any(observation.candidate_id != self.candidate.identifier for observation in observations):
            raise ValueError("candidate result observations must match the candidate identifier.")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "confusion_matrix", ThresholdConfusionMatrix.from_observations(observations))
        object.__setattr__(self, "metrics", ThresholdMetrics.from_observations(observations))
        object.__setattr__(self, "diagnostics", _sorted_diagnostics(self.diagnostics))


@dataclass(frozen=True)
class ThresholdCalibrationSummary:
    candidate_count: int
    case_count: int
    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        _require_positive_int(self.candidate_count, "candidate count")
        _require_positive_int(self.case_count, "case count")
        object.__setattr__(self, "diagnostics", _sorted_diagnostics(self.diagnostics))


@dataclass(frozen=True)
class ThresholdCalibrationResult:
    candidate_results: tuple[ThresholdCandidateResult, ...] = ()
    summary: ThresholdCalibrationSummary | None = None
    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        candidate_results = tuple(self.candidate_results)
        object.__setattr__(self, "candidate_results", candidate_results)
        object.__setattr__(self, "diagnostics", _sorted_diagnostics(self.diagnostics))
        if self.summary is None and not self.diagnostics:
            raise ValueError("calibration result without a summary requires diagnostics.")
        if self.summary is not None:
            if not candidate_results:
                raise ValueError("successful calibration results require candidate results.")
            if self.summary.candidate_count != len(candidate_results):
                raise ValueError("summary candidate count must match candidate results.")
            identifiers = [result.candidate.identifier for result in candidate_results]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("candidate result identifiers must be unique.")

    @property
    def is_valid(self) -> bool:
        return self.summary is not None and not self.diagnostics


class ThresholdCalibrationRunner:
    def __init__(
        self,
        *,
        normalizer: TemplateImageNormalizer | None = None,
        loader_factory: type[ReplayDatasetLoader] = ReplayDatasetLoader,
    ) -> None:
        self.normalizer = normalizer or TemplateImageNormalizer()
        self.loader_factory = loader_factory

    def run(self, request: ThresholdCalibrationRequest) -> ThresholdCalibrationResult:
        validation = self._validate_request(request)
        if validation:
            return ThresholdCalibrationResult(diagnostics=validation)
        case_result = self._cases_for_request(request)
        if case_result[1]:
            return ThresholdCalibrationResult(diagnostics=case_result[1])
        cases = case_result[0]
        loader = self.loader_factory(request.dataset_root)
        candidate_results = tuple(
            self._evaluate_candidate(candidate, cases, request, loader)
            for candidate in request.candidates
        )
        return ThresholdCalibrationResult(
            candidate_results=candidate_results,
            summary=ThresholdCalibrationSummary(
                candidate_count=len(candidate_results),
                case_count=len(cases),
            ),
        )

    def _validate_request(self, request: ThresholdCalibrationRequest) -> tuple[ValidationDiagnostic, ...]:
        diagnostics: list[ValidationDiagnostic] = []
        if not request.candidates:
            diagnostics.append(_diagnostic("calibration.request.empty_candidates", "candidates", "Calibration requires at least one candidate."))
        if request.dataset_root is None:
            diagnostics.append(_diagnostic("calibration.request.dataset_root_missing", "dataset_root", "Calibration requires a replay dataset root."))
        if request.registry is None:
            diagnostics.append(_diagnostic("calibration.request.registry_missing", "registry", "Calibration requires a TemplateRegistry."))
        diagnostics.extend(_candidate_collection_diagnostics(request.candidates))
        if not request.cases and request.replay_dataset is None and request.manifest_path is None:
            diagnostics.append(_diagnostic("calibration.request.empty_cases", "cases", "Calibration requires replay-derived cases."))
        return _sorted_diagnostics(diagnostics)

    def _cases_for_request(
        self,
        request: ThresholdCalibrationRequest,
    ) -> tuple[tuple[ThresholdCalibrationCase, ...], tuple[ValidationDiagnostic, ...]]:
        if request.cases:
            cases = request.cases
        else:
            dataset_result = self._dataset_for_request(request)
            if dataset_result[1]:
                return (), dataset_result[1]
            cases = self._cases_from_dataset(dataset_result[0], request)
        diagnostics = _case_collection_diagnostics(cases, request.candidates)
        if diagnostics:
            return (), diagnostics
        return cases, ()

    def _dataset_for_request(
        self,
        request: ThresholdCalibrationRequest,
    ) -> tuple[ReplayDatasetDefinition | None, tuple[ValidationDiagnostic, ...]]:
        if request.replay_dataset is not None:
            return request.replay_dataset, ()
        try:
            load_result = self.loader_factory(request.dataset_root).load(request.manifest_path or "")
        except (OSError, PermissionError, TypeError, ValueError, AttributeError, cv2.error):
            return None, (_diagnostic("calibration.dataset.validation_failed", "manifest", "Calibration replay dataset could not be loaded."),)
        if load_result.dataset is None:
            return None, _map_replay_diagnostics(load_result.diagnostics, "calibration.dataset")
        if load_result.dataset.diagnostics:
            return None, _map_replay_diagnostics(load_result.dataset.diagnostics, "calibration.dataset")
        return load_result.dataset, ()

    def _cases_from_dataset(
        self,
        dataset: ReplayDatasetDefinition | None,
        request: ThresholdCalibrationRequest,
    ) -> tuple[ThresholdCalibrationCase, ...]:
        if dataset is None:
            return ()
        cases: list[ThresholdCalibrationCase] = []
        targets = tuple(dict.fromkeys(candidate.target for candidate in request.candidates))
        for replay_case in dataset.cases:
            for target in targets:
                try:
                    cases.append(ThresholdCalibrationCase.from_replay_case(replay_case, target))
                except ValueError:
                    continue
        return tuple(sorted(cases, key=lambda item: (item.case_id, item.target.value)))

    def _evaluate_candidate(
        self,
        candidate: ThresholdCandidate,
        cases: tuple[ThresholdCalibrationCase, ...],
        request: ThresholdCalibrationRequest,
        loader: ReplayDatasetLoader,
    ) -> ThresholdCandidateResult:
        applicable = tuple(case for case in cases if case.target == candidate.target)
        observations = tuple(
            self._evaluate_case(candidate, case, request, loader)
            for case in applicable
        )
        if not observations:
            observations = (
                ThresholdCaseObservation(
                    case_id=f"{candidate.identifier}.no_applicable_cases",
                    candidate_id=candidate.identifier,
                    target=candidate.target,
                    expected_positive=False,
                    actual_positive=None,
                    status=ThresholdObservationStatus.INVALID_CASE,
                    diagnostics=(
                        _diagnostic(
                            "calibration.candidate.no_applicable_cases",
                            candidate.identifier,
                            "Candidate target has no applicable calibration cases.",
                        ),
                    ),
                ),
            )
        return ThresholdCandidateResult(candidate=candidate, observations=observations)

    def _evaluate_case(
        self,
        candidate: ThresholdCandidate,
        case: ThresholdCalibrationCase,
        request: ThresholdCalibrationRequest,
        loader: ReplayDatasetLoader,
    ) -> ThresholdCaseObservation:
        if case.diagnostics:
            return self._observation(
                candidate,
                case,
                ThresholdObservationStatus.INVALID_CASE,
                diagnostics=_map_replay_diagnostics(case.diagnostics, "calibration.ground_truth"),
            )
        screenshot_result = self._load_screenshot(loader, case)
        if screenshot_result[0] is None:
            return self._observation(candidate, case, ThresholdObservationStatus.INPUT_ERROR, diagnostics=screenshot_result[1])
        if candidate.target == ThresholdCalibrationTarget.TEMPLATE_DETECTION:
            return self._evaluate_detection(candidate, case, request, screenshot_result[0])
        return self._evaluate_scene(candidate, case, request, screenshot_result[0])

    def _load_screenshot(
        self,
        loader: ReplayDatasetLoader,
        case: ThresholdCalibrationCase,
    ) -> tuple[ImageInput | None, tuple[ValidationDiagnostic, ...]]:
        try:
            replay_case = ReplayCaseDefinition(case.case_id, case.screenshot_path, case.template_key)
            screenshot_path = loader.screenshot_path(replay_case)
        except (OSError, PermissionError, TypeError, ValueError, AttributeError):
            return None, (_diagnostic("calibration.path.invalid", "screenshot", "Calibration screenshot path is invalid."),)
        normalized = self.normalizer.normalize(screenshot_path, field="screenshot", grayscale=True)
        if normalized.image is None:
            return None, (_diagnostic("calibration.image.invalid", "screenshot", "Calibration screenshot is not a supported readable image."),)
        return normalized.image.pixels.copy(), ()

    def _evaluate_detection(
        self,
        candidate: ThresholdCandidate,
        case: ThresholdCalibrationCase,
        request: ThresholdCalibrationRequest,
        screenshot: ImageInput,
    ) -> ThresholdCaseObservation:
        analyzer = request.analyzer or TemplateScreenAnalyzer()
        try:
            registry = _registry_with_threshold(request.registry, case.template_key, candidate.value)
            result = analyzer.match(screenshot, case.template_key, registry)
        except (cv2.error, OSError, PermissionError, KeyError, TypeError, ValueError, AttributeError, TemplateRegistryError):
            return self._observation(
                candidate,
                case,
                ThresholdObservationStatus.PIPELINE_ERROR,
                diagnostics=(_diagnostic("calibration.detection.pipeline_failed", case.case_id, "Template detection failed during calibration."),),
            )
        if not isinstance(result, DetectionResult):
            return self._observation(
                candidate,
                case,
                ThresholdObservationStatus.PIPELINE_ERROR,
                diagnostics=(_diagnostic("calibration.detection.malformed_result", case.case_id, "Template detection returned a malformed result."),),
            )
        confidence = getattr(result, "confidence", None)
        if not _is_threshold(confidence):
            return self._observation(
                candidate,
                case,
                ThresholdObservationStatus.PIPELINE_ERROR,
                diagnostics=(_diagnostic("calibration.detection.non_finite_confidence", case.case_id, "Template detection returned a non-finite confidence."),),
            )
        matched_key = getattr(result, "matched_semantic_key", None)
        actual_positive = matched_key == case.template_key and float(confidence) >= candidate.value
        diagnostics = ()
        if matched_key is not None and matched_key != case.template_key:
            diagnostics = (_diagnostic("calibration.detection.wrong_template", case.case_id, "Template detection matched a different semantic key."),)
        return self._confusion_observation(
            candidate,
            case,
            actual_positive=actual_positive,
            confidence=float(confidence),
            matched_identifier=matched_key,
            diagnostics=diagnostics,
        )

    def _evaluate_scene(
        self,
        candidate: ThresholdCandidate,
        case: ThresholdCalibrationCase,
        request: ThresholdCalibrationRequest,
        screenshot: ImageInput,
    ) -> ThresholdCaseObservation:
        classifier = request.scene_classifier or SceneClassifier(request.analyzer)
        try:
            definitions = _scene_definitions_with_minimum_score(request.scene_definitions, case.scene_key, candidate.value)
            result = classifier.classify(screenshot, definitions, request.registry, analyzer=request.analyzer)
        except (cv2.error, OSError, PermissionError, KeyError, TypeError, ValueError, AttributeError, TemplateRegistryError):
            return self._observation(
                candidate,
                case,
                ThresholdObservationStatus.PIPELINE_ERROR,
                diagnostics=(_diagnostic("calibration.scene.pipeline_failed", case.case_id, "Scene classification failed during calibration."),),
            )
        if not isinstance(result, SceneClassificationResult):
            return self._observation(
                candidate,
                case,
                ThresholdObservationStatus.PIPELINE_ERROR,
                diagnostics=(_diagnostic("calibration.scene.malformed_result", case.case_id, "Scene classification returned a malformed result."),),
            )
        score = getattr(result, "score", None)
        if not _is_threshold(score):
            return self._observation(
                candidate,
                case,
                ThresholdObservationStatus.PIPELINE_ERROR,
                diagnostics=(_diagnostic("calibration.scene.non_finite_score", case.case_id, "Scene classification returned a non-finite score."),),
            )
        actual_positive = (
            result.status == SceneClassificationStatus.CLASSIFIED
            and result.scene_key == case.scene_key
            and float(score) + _EPSILON >= candidate.value
        )
        return self._confusion_observation(
            candidate,
            case,
            actual_positive=actual_positive,
            score=float(score),
            matched_identifier=result.scene_key,
        )

    def _confusion_observation(
        self,
        candidate: ThresholdCandidate,
        case: ThresholdCalibrationCase,
        *,
        actual_positive: bool,
        confidence: float | None = None,
        score: float | None = None,
        matched_identifier: str | None = None,
        diagnostics: tuple[ValidationDiagnostic, ...] = (),
    ) -> ThresholdCaseObservation:
        if case.expected_positive and actual_positive:
            status = ThresholdObservationStatus.TRUE_POSITIVE
        elif not case.expected_positive and not actual_positive:
            status = ThresholdObservationStatus.TRUE_NEGATIVE
        elif not case.expected_positive and actual_positive:
            status = ThresholdObservationStatus.FALSE_POSITIVE
        else:
            status = ThresholdObservationStatus.FALSE_NEGATIVE
        return self._observation(
            candidate,
            case,
            status,
            actual_positive=actual_positive,
            confidence=confidence,
            score=score,
            matched_identifier=matched_identifier,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _observation(
        candidate: ThresholdCandidate,
        case: ThresholdCalibrationCase,
        status: ThresholdObservationStatus,
        *,
        actual_positive: bool | None = None,
        confidence: float | None = None,
        score: float | None = None,
        matched_identifier: str | None = None,
        diagnostics: tuple[ValidationDiagnostic, ...] = (),
    ) -> ThresholdCaseObservation:
        return ThresholdCaseObservation(
            case_id=case.case_id,
            candidate_id=candidate.identifier,
            target=candidate.target,
            expected_positive=case.expected_positive,
            actual_positive=actual_positive,
            status=status,
            confidence=confidence,
            score=score,
            matched_identifier=matched_identifier,
            diagnostics=diagnostics,
        )


_CONFUSION_STATUSES = frozenset(
    {
        ThresholdObservationStatus.TRUE_POSITIVE,
        ThresholdObservationStatus.TRUE_NEGATIVE,
        ThresholdObservationStatus.FALSE_POSITIVE,
        ThresholdObservationStatus.FALSE_NEGATIVE,
    }
)


def _registry_with_threshold(
    registry: TemplateRegistry | None,
    template_key: str | None,
    threshold: float,
) -> TemplateRegistry:
    if registry is None or template_key is None:
        raise ValueError("registry and template key are required")
    templates: list[TemplateDefinition] = []
    found = False
    for definition in registry.templates():
        if definition.semantic_key == template_key:
            templates.append(replace(definition, confidence_threshold=threshold))
            found = True
        else:
            templates.append(definition)
    if not found:
        registry.get(template_key)
    pack = registry.template_pack
    return TemplateRegistry(
        TemplatePack(
            version=pack.version,
            languages=pack.languages,
            resolution_profiles=pack.resolution_profiles,
            templates=tuple(templates),
            root=pack.root,
        )
    )


def _scene_definitions_with_minimum_score(
    definitions: tuple[SceneDefinition, ...],
    scene_key: str | None,
    threshold: float,
) -> tuple[SceneDefinition, ...]:
    if scene_key is None:
        raise ValueError("scene key is required")
    copied: list[SceneDefinition] = []
    found = False
    for definition in definitions:
        if definition.semantic_key == scene_key:
            rule = definition.rule
            copied.append(
                replace(
                    definition,
                    rule=SceneRule(
                        required_template_keys=rule.required_template_keys,
                        optional_template_keys=rule.optional_template_keys,
                        forbidden_template_keys=rule.forbidden_template_keys,
                        minimum_score=threshold,
                    ),
                )
            )
            found = True
        else:
            copied.append(definition)
    if not found:
        raise ValueError("scene definition is missing")
    return tuple(copied)


def _candidate_collection_diagnostics(candidates: tuple[ThresholdCandidate, ...]) -> tuple[ValidationDiagnostic, ...]:
    diagnostics: list[ValidationDiagnostic] = []
    seen_ids: set[str] = set()
    seen_values: set[tuple[ThresholdCalibrationTarget, float]] = set()
    for candidate in candidates:
        if candidate.identifier in seen_ids:
            diagnostics.append(_diagnostic("calibration.candidate.duplicate_identifier", "candidates", "Candidate identifiers must be unique."))
        seen_ids.add(candidate.identifier)
        value_key = (candidate.target, candidate.value)
        if value_key in seen_values:
            diagnostics.append(_diagnostic("calibration.candidate.duplicate_threshold", "candidates", "Candidate thresholds must be unique per target."))
        seen_values.add(value_key)
    return _sorted_diagnostics(diagnostics)


def _case_collection_diagnostics(
    cases: tuple[ThresholdCalibrationCase, ...],
    candidates: tuple[ThresholdCandidate, ...],
) -> tuple[ValidationDiagnostic, ...]:
    diagnostics: list[ValidationDiagnostic] = []
    if not cases:
        diagnostics.append(_diagnostic("calibration.request.empty_cases", "cases", "Calibration requires at least one executable case."))
        return tuple(diagnostics)
    seen_ids: set[str] = set()
    candidate_targets = {candidate.target for candidate in candidates}
    for case in cases:
        if case.case_id in seen_ids:
            diagnostics.append(_diagnostic("calibration.case.duplicate_identifier", "cases", "Calibration case identifiers must be unique."))
        seen_ids.add(case.case_id)
        if case.target not in candidate_targets:
            diagnostics.append(_diagnostic("calibration.case.unsupported_target", case.case_id, "Calibration case target has no matching candidate."))
    return _sorted_diagnostics(diagnostics)


def _sorted_candidates(candidates: tuple[ThresholdCandidate, ...]) -> tuple[ThresholdCandidate, ...]:
    return tuple(sorted(tuple(candidates), key=lambda item: (item.target.value, item.identifier, item.value)))


def _map_replay_diagnostics(
    diagnostics: tuple[ValidationDiagnostic, ...],
    prefix: str,
) -> tuple[ValidationDiagnostic, ...]:
    return _sorted_diagnostics(
        tuple(
            _diagnostic(
                f"{prefix}.{diagnostic.code}",
                diagnostic.field,
                "Replay diagnostic prevented calibration.",
            )
            for diagnostic in diagnostics
        )
    )


def _diagnostic(code: str, field: str, message: str) -> ValidationDiagnostic:
    return ValidationDiagnostic(code=code, field=field, message=message)


def _normalize_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _normalize_optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _normalize_metadata(values: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[str, str]] = []
    for key, value in tuple(values):
        normalized.append((_normalize_text(key, "metadata key"), _normalize_text(value, "metadata value")))
    return tuple(sorted(normalized))


def _require_threshold(value: Any, field_name: str) -> float:
    if not _is_threshold(value):
        raise ValueError(f"{field_name} must be a finite number between 0.0 and 1.0.")
    return float(value)


def _is_threshold(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


def _require_non_negative_int(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")


def _require_positive_int(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    value = float(numerator) / float(denominator)
    if not math.isfinite(value):
        return 0.0
    return round(value, 12)


def _sorted_diagnostics(diagnostics: tuple[ValidationDiagnostic, ...] | list[ValidationDiagnostic]) -> tuple[ValidationDiagnostic, ...]:
    return tuple(sorted(tuple(diagnostics), key=lambda item: (item.field, item.code, item.message)))
