from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import cv2

from .evidence_capture import EvidenceCaptureRequest, EvidenceReference, EvidenceStore
from .image_matching import ImageInput, TemplateImageNormalizer, TemplateScreenAnalyzer
from .scene_classification import (
    SceneClassificationResult,
    SceneClassificationStatus,
    SceneClassifier,
    SceneDefinition,
)
from .template_models import BoundingBox, DetectionResult, ValidationDiagnostic
from .template_registry import TemplateRegistry

REPLAY_DATASET_SCHEMA_VERSION = 1
_MANIFEST_FIELDS = frozenset({"schema_version", "dataset_id", "cases"})
_CASE_FIELDS = frozenset({"case_id", "screenshot", "template_key", "expectation", "label"})
_EXPECTATION_FIELDS = frozenset({"detection", "scene"})
_DETECTION_EXPECTATION_FIELDS = frozenset(
    {
        "expected_match",
        "semantic_key",
        "confidence_min",
        "confidence_max",
        "matched_scale_min",
        "matched_scale_max",
        "bounding_box",
        "bounding_box_tolerance",
        "diagnostic_code",
    }
)
_SCENE_EXPECTATION_FIELDS = frozenset(
    {"status", "semantic_scene_key", "candidate_count", "diagnostic_code"}
)
_BOUNDING_BOX_FIELDS = frozenset({"x", "y", "width", "height"})
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{index}" for index in range(1, 10)), *(f"LPT{index}" for index in range(1, 10))}
)


class ReplayCaseStatus(str, Enum):
    PASS = "pass"
    EXPECTATION_MISMATCH = "expectation_mismatch"
    INVALID_CASE = "invalid_case"
    INPUT_ERROR = "input_error"
    PIPELINE_ERROR = "pipeline_error"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ReplayDetectionExpectation:
    expected_match: bool | None = None
    semantic_key: str | None = None
    confidence_min: float | None = None
    confidence_max: float | None = None
    matched_scale_min: float | None = None
    matched_scale_max: float | None = None
    bounding_box: BoundingBox | None = None
    bounding_box_tolerance: int = 0
    diagnostic_code: str | None = None

    def __post_init__(self) -> None:
        if self.expected_match is not None and not isinstance(self.expected_match, bool):
            raise ValueError("expected_match must be boolean when provided.")
        if self.semantic_key is not None:
            object.__setattr__(self, "semantic_key", _normalize_optional_text(self.semantic_key, "semantic key"))
        for field_name in (
            "confidence_min",
            "confidence_max",
            "matched_scale_min",
            "matched_scale_max",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_finite_number(value, field_name)
        _require_non_negative_int(self.bounding_box_tolerance, "bounding box tolerance")
        if (
            self.confidence_min is not None
            and self.confidence_max is not None
            and self.confidence_min > self.confidence_max
        ):
            raise ValueError("confidence_min must not exceed confidence_max.")
        if (
            self.matched_scale_min is not None
            and self.matched_scale_max is not None
            and self.matched_scale_min > self.matched_scale_max
        ):
            raise ValueError("matched_scale_min must not exceed matched_scale_max.")
        if self.diagnostic_code is not None:
            object.__setattr__(self, "diagnostic_code", _normalize_optional_text(self.diagnostic_code, "diagnostic code"))


@dataclass(frozen=True)
class ReplaySceneExpectation:
    status: SceneClassificationStatus | str
    semantic_scene_key: str | None = None
    candidate_count: int | None = None
    diagnostic_code: str | None = None

    def __post_init__(self) -> None:
        status = SceneClassificationStatus(self.status)
        object.__setattr__(self, "status", status)
        if self.semantic_scene_key is not None:
            object.__setattr__(
                self,
                "semantic_scene_key",
                _normalize_optional_text(self.semantic_scene_key, "semantic scene key"),
            )
            if status != SceneClassificationStatus.CLASSIFIED:
                raise ValueError("scene key expectations are only valid for classified scenes.")
        if self.candidate_count is not None:
            _require_non_negative_int(self.candidate_count, "candidate count")
        if self.diagnostic_code is not None:
            object.__setattr__(self, "diagnostic_code", _normalize_optional_text(self.diagnostic_code, "diagnostic code"))


@dataclass(frozen=True)
class ReplayExpectation:
    detection: ReplayDetectionExpectation | None = None
    scene: ReplaySceneExpectation | None = None

    def __post_init__(self) -> None:
        if self.detection is None and self.scene is None:
            raise ValueError("replay expectation requires detection or scene expectations.")


@dataclass(frozen=True)
class ReplayCaseDefinition:
    case_id: str
    screenshot_path: str
    template_key: str | None = None
    expectation: ReplayExpectation | None = None
    label: str = ""
    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _normalize_text(self.case_id, "case identifier"))
        object.__setattr__(self, "screenshot_path", _normalize_text(self.screenshot_path, "screenshot path"))
        if self.template_key is not None:
            object.__setattr__(self, "template_key", _normalize_optional_text(self.template_key, "template key"))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def executes_detection(self) -> bool:
        return self.template_key is not None or (
            self.expectation is not None and self.expectation.detection is not None
        )

    @property
    def executes_scene(self) -> bool:
        return self.expectation is not None and self.expectation.scene is not None


@dataclass(frozen=True)
class ReplayDatasetDefinition:
    dataset_id: str
    cases: tuple[ReplayCaseDefinition, ...]
    schema_version: int = REPLAY_DATASET_SCHEMA_VERSION
    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_DATASET_SCHEMA_VERSION:
            raise ValueError("unsupported replay dataset schema version.")
        object.__setattr__(self, "dataset_id", _normalize_text(self.dataset_id, "dataset identifier"))
        object.__setattr__(self, "cases", tuple(sorted(self.cases, key=lambda case: case.case_id)))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


@dataclass(frozen=True)
class ReplayDatasetLoadResult:
    dataset: ReplayDatasetDefinition | None = None
    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", _sorted_diagnostics(self.diagnostics))
        if self.dataset is None and not self.diagnostics:
            raise ValueError("failed replay dataset load requires diagnostics.")

    @property
    def is_valid(self) -> bool:
        return self.dataset is not None and not self.diagnostics


@dataclass(frozen=True)
class ReplayRunRequest:
    dataset: ReplayDatasetDefinition | None = None
    manifest_path: str | Path | None = None
    dataset_root: str | Path | None = None
    registry: TemplateRegistry | None = None
    analyzer: TemplateScreenAnalyzer | None = None
    scene_classifier: SceneClassifier | None = None
    scene_definitions: tuple[SceneDefinition, ...] = ()
    evidence_store: EvidenceStore | None = None
    capture_evidence_on: tuple[ReplayCaseStatus | str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_definitions", tuple(self.scene_definitions))
        object.__setattr__(
            self,
            "capture_evidence_on",
            tuple(ReplayCaseStatus(status) for status in self.capture_evidence_on),
        )
        if self.dataset is None and self.manifest_path is None:
            raise ValueError("replay run requires a dataset definition or manifest path.")


@dataclass(frozen=True)
class ReplayCaseResult:
    case_id: str
    status: ReplayCaseStatus
    diagnostics: tuple[ValidationDiagnostic, ...] = ()
    detection_result: DetectionResult | None = None
    scene_result: SceneClassificationResult | None = None
    evidence_reference: EvidenceReference | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _normalize_text(self.case_id, "case identifier"))
        object.__setattr__(self, "status", ReplayCaseStatus(self.status))
        object.__setattr__(self, "diagnostics", _sorted_diagnostics(self.diagnostics))
        if self.status == ReplayCaseStatus.PASS and self.diagnostics:
            raise ValueError("passing replay case results must not contain diagnostics.")
        if self.status != ReplayCaseStatus.PASS and not self.diagnostics:
            raise ValueError("non-passing replay case results require diagnostics.")


@dataclass(frozen=True)
class ReplayRunSummary:
    total_cases: int
    passed_cases: int
    failed_cases: int
    invalid_cases: int
    skipped_cases: int
    detection_cases: int
    scene_classification_cases: int

    def __post_init__(self) -> None:
        for field_name in (
            "total_cases",
            "passed_cases",
            "failed_cases",
            "invalid_cases",
            "skipped_cases",
            "detection_cases",
            "scene_classification_cases",
        ):
            _require_non_negative_int(getattr(self, field_name), field_name)
        if self.total_cases <= 0:
            raise ValueError("replay summaries require at least one case.")
        if self.passed_cases + self.failed_cases + self.invalid_cases + self.skipped_cases != self.total_cases:
            raise ValueError("replay summary counts must match total cases.")
        if self.detection_cases > self.total_cases or self.scene_classification_cases > self.total_cases:
            raise ValueError("operation counts cannot exceed total cases.")

    @classmethod
    def from_results(
        cls,
        cases: Sequence[ReplayCaseDefinition],
        results: Sequence[ReplayCaseResult],
    ) -> ReplayRunSummary:
        return cls(
            total_cases=len(results),
            passed_cases=sum(1 for result in results if result.status == ReplayCaseStatus.PASS),
            failed_cases=sum(1 for result in results if result.status in (ReplayCaseStatus.EXPECTATION_MISMATCH, ReplayCaseStatus.PIPELINE_ERROR, ReplayCaseStatus.INPUT_ERROR)),
            invalid_cases=sum(1 for result in results if result.status == ReplayCaseStatus.INVALID_CASE),
            skipped_cases=sum(1 for result in results if result.status == ReplayCaseStatus.SKIPPED),
            detection_cases=sum(1 for case in cases if case.executes_detection),
            scene_classification_cases=sum(1 for case in cases if case.executes_scene),
        )


@dataclass(frozen=True)
class ReplayRunResult:
    dataset_id: str | None = None
    case_results: tuple[ReplayCaseResult, ...] = ()
    summary: ReplayRunSummary | None = None
    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.dataset_id is not None:
            object.__setattr__(self, "dataset_id", _normalize_optional_text(self.dataset_id, "dataset identifier"))
        object.__setattr__(self, "case_results", tuple(sorted(self.case_results, key=lambda result: result.case_id)))
        object.__setattr__(self, "diagnostics", _sorted_diagnostics(self.diagnostics))
        if self.summary is None and not self.diagnostics:
            raise ValueError("replay run result without a summary requires diagnostics.")
        if self.summary is not None and self.summary.total_cases != len(self.case_results):
            raise ValueError("replay summary total must match case results.")

    @property
    def is_valid(self) -> bool:
        return self.summary is not None and not self.diagnostics and all(
            result.status == ReplayCaseStatus.PASS for result in self.case_results
        )


class ReplayDatasetLoader:
    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root)

    def load(self, manifest_path: str | Path) -> ReplayDatasetLoadResult:
        try:
            root = self._resolved_root()
            manifest_file = self._resolve_manifest_path(root, manifest_path)
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return ReplayDatasetLoadResult(
                diagnostics=(self._diagnostic("replay.manifest.invalid_json", "manifest", "Replay dataset manifest is not valid JSON."),)
            )
        except (OSError, PermissionError, TypeError, ValueError):
            return ReplayDatasetLoadResult(
                diagnostics=(self._diagnostic("replay.manifest.unreadable", "manifest", "Replay dataset manifest cannot be read."),)
            )
        return self.load_mapping(manifest)

    def load_mapping(self, manifest: Mapping[str, Any]) -> ReplayDatasetLoadResult:
        diagnostics: list[ValidationDiagnostic] = []
        if not isinstance(manifest, Mapping):
            return ReplayDatasetLoadResult(
                diagnostics=(self._diagnostic("replay.manifest.invalid_type", "manifest", "Replay dataset manifest must be a JSON object."),)
            )
        diagnostics.extend(_unknown_key_diagnostics(manifest, _MANIFEST_FIELDS, "manifest"))
        if manifest.get("schema_version") != REPLAY_DATASET_SCHEMA_VERSION:
            diagnostics.append(
                self._diagnostic(
                    "replay.manifest.unsupported_version",
                    "schema_version",
                    f"Replay dataset schema_version must be {REPLAY_DATASET_SCHEMA_VERSION}.",
                )
            )
        dataset_id = _required_text(manifest, "dataset_id", diagnostics, "manifest")
        raw_cases = manifest.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            diagnostics.append(
                self._diagnostic("replay.dataset.empty", "cases", "Replay dataset requires a non-empty case list.")
            )
            return ReplayDatasetLoadResult(diagnostics=tuple(diagnostics))

        case_ids: set[str] = set()
        cases: list[ReplayCaseDefinition] = []
        for index, raw_case in enumerate(raw_cases):
            case = self._parse_case(raw_case, f"cases[{index}]")
            if case.case_id in case_ids:
                case = _case_with_diagnostic(
                    case,
                    self._diagnostic("replay.case.duplicate_id", f"cases[{index}].case_id", "Replay case identifiers must be unique."),
                )
            case_ids.add(case.case_id)
            cases.append(case)
        if not any(case.executes_detection or case.executes_scene for case in cases):
            diagnostics.append(
                self._diagnostic(
                    "replay.dataset.no_executable_cases",
                    "cases",
                    "Replay dataset contains no executable detection or scene cases.",
                )
            )
        if diagnostics:
            return ReplayDatasetLoadResult(diagnostics=tuple(diagnostics))
        return ReplayDatasetLoadResult(
            dataset=ReplayDatasetDefinition(
                dataset_id=dataset_id,
                cases=tuple(cases),
            )
        )

    def _parse_case(self, raw_case: Any, field: str) -> ReplayCaseDefinition:
        diagnostics: list[ValidationDiagnostic] = []
        if not isinstance(raw_case, Mapping):
            return ReplayCaseDefinition(
                case_id=f"invalid-{field}",
                screenshot_path="invalid.png",
                diagnostics=(self._diagnostic("replay.case.invalid_type", field, "Replay case must be an object."),),
            )
        diagnostics.extend(_unknown_key_diagnostics(raw_case, _CASE_FIELDS, field))
        case_id = _required_text(raw_case, "case_id", diagnostics, field) or f"invalid-{field}"
        screenshot_path = _required_text(raw_case, "screenshot", diagnostics, field) or "invalid.png"
        template_key = _optional_text(raw_case.get("template_key"), diagnostics, f"{field}.template_key")
        expectation = self._parse_expectation(raw_case.get("expectation"), diagnostics, field)
        label = str(raw_case.get("label", ""))
        screenshot_diagnostics = self._validate_screenshot_reference(screenshot_path, f"{field}.screenshot")
        diagnostics.extend(screenshot_diagnostics)
        if expectation is None and not template_key:
            diagnostics.append(
                self._diagnostic("replay.case.no_operation", field, "Replay case requires a template key or scene expectation.")
            )
        return ReplayCaseDefinition(
            case_id=case_id,
            screenshot_path=screenshot_path,
            template_key=template_key,
            expectation=expectation,
            label=label,
            diagnostics=tuple(diagnostics),
        )

    def _parse_expectation(
        self,
        raw_expectation: Any,
        diagnostics: list[ValidationDiagnostic],
        field: str,
    ) -> ReplayExpectation | None:
        if raw_expectation is None:
            return None
        if not isinstance(raw_expectation, Mapping):
            diagnostics.append(
                self._diagnostic("replay.expectation.invalid_type", f"{field}.expectation", "Replay expectation must be an object.")
            )
            return None
        diagnostics.extend(_unknown_key_diagnostics(raw_expectation, _EXPECTATION_FIELDS, f"{field}.expectation"))
        detection = self._parse_detection_expectation(raw_expectation.get("detection"), diagnostics, field)
        scene = self._parse_scene_expectation(raw_expectation.get("scene"), diagnostics, field)
        if detection is None and scene is None:
            diagnostics.append(
                self._diagnostic(
                    "replay.expectation.empty",
                    f"{field}.expectation",
                    "Replay expectation requires detection or scene expectations.",
                )
            )
            return None
        try:
            return ReplayExpectation(detection=detection, scene=scene)
        except ValueError:
            diagnostics.append(
                self._diagnostic("replay.expectation.invalid", f"{field}.expectation", "Replay expectation is malformed.")
            )
            return None

    def _parse_detection_expectation(
        self,
        raw_detection: Any,
        diagnostics: list[ValidationDiagnostic],
        field: str,
    ) -> ReplayDetectionExpectation | None:
        if raw_detection is None:
            return None
        if not isinstance(raw_detection, Mapping):
            diagnostics.append(
                self._diagnostic("replay.expectation.invalid_detection", f"{field}.expectation.detection", "Detection expectation must be an object.")
            )
            return None
        diagnostics.extend(
            _unknown_key_diagnostics(
                raw_detection,
                _DETECTION_EXPECTATION_FIELDS,
                f"{field}.expectation.detection",
            )
        )
        try:
            return ReplayDetectionExpectation(
                expected_match=raw_detection.get("expected_match"),
                semantic_key=raw_detection.get("semantic_key"),
                confidence_min=raw_detection.get("confidence_min"),
                confidence_max=raw_detection.get("confidence_max"),
                matched_scale_min=raw_detection.get("matched_scale_min"),
                matched_scale_max=raw_detection.get("matched_scale_max"),
                bounding_box=_parse_bounding_box(raw_detection.get("bounding_box")),
                bounding_box_tolerance=raw_detection.get("bounding_box_tolerance", 0),
                diagnostic_code=raw_detection.get("diagnostic_code"),
            )
        except (KeyError, TypeError, ValueError):
            diagnostics.append(
                self._diagnostic("replay.expectation.invalid_detection", f"{field}.expectation.detection", "Detection expectation is malformed.")
            )
            return None

    def _parse_scene_expectation(
        self,
        raw_scene: Any,
        diagnostics: list[ValidationDiagnostic],
        field: str,
    ) -> ReplaySceneExpectation | None:
        if raw_scene is None:
            return None
        if not isinstance(raw_scene, Mapping):
            diagnostics.append(
                self._diagnostic("replay.expectation.invalid_scene", f"{field}.expectation.scene", "Scene expectation must be an object.")
            )
            return None
        diagnostics.extend(
            _unknown_key_diagnostics(raw_scene, _SCENE_EXPECTATION_FIELDS, f"{field}.expectation.scene")
        )
        try:
            return ReplaySceneExpectation(
                status=raw_scene["status"],
                semantic_scene_key=raw_scene.get("semantic_scene_key"),
                candidate_count=raw_scene.get("candidate_count"),
                diagnostic_code=raw_scene.get("diagnostic_code"),
            )
        except (KeyError, TypeError, ValueError):
            diagnostics.append(
                self._diagnostic("replay.expectation.invalid_scene", f"{field}.expectation.scene", "Scene expectation is malformed.")
            )
            return None

    def _validate_screenshot_reference(
        self,
        value: str,
        field: str,
    ) -> tuple[ValidationDiagnostic, ...]:
        relative = _relative_path(value)
        if relative is None:
            return (self._diagnostic("replay.path.invalid", field, "Replay screenshot path must be relative and confined to the dataset root."),)
        try:
            root = self._resolved_root()
            resolved = (root / relative).resolve()
        except (OSError, PermissionError, ValueError):
            return (self._diagnostic("replay.path.invalid", field, "Replay screenshot path could not be resolved inside the dataset root."),)
        if not _is_relative_to(resolved, root):
            return (self._diagnostic("replay.path_escape", field, "Replay screenshot path escaped the dataset root."),)
        if resolved.exists() and not resolved.is_file():
            return (self._diagnostic("replay.screenshot.not_file", field, "Replay screenshot path must reference a file."),)
        if not resolved.is_file():
            return (self._diagnostic("replay.screenshot.missing", field, "Replay screenshot file does not exist."),)
        return ()

    def screenshot_path(self, case: ReplayCaseDefinition) -> Path:
        root = self._resolved_root()
        relative = _relative_path(case.screenshot_path)
        if relative is None:
            raise ValueError("invalid screenshot path")
        path = (root / relative).resolve()
        if not _is_relative_to(path, root):
            raise ValueError("screenshot path escaped dataset root")
        return path

    def _resolve_manifest_path(self, root: Path, manifest_path: str | Path) -> Path:
        path = Path(manifest_path)
        if path.is_absolute():
            resolved = path.resolve()
        else:
            relative = _relative_path(str(manifest_path))
            if relative is None:
                raise ValueError("invalid manifest path")
            resolved = (root / relative).resolve()
        if not _is_relative_to(resolved, root):
            raise ValueError("manifest path escaped dataset root")
        return resolved

    def _resolved_root(self) -> Path:
        if self.dataset_root.exists() and self.dataset_root.is_symlink():
            raise ValueError("dataset root must not be a symlink")
        return self.dataset_root.resolve()

    @staticmethod
    def _diagnostic(code: str, field: str, message: str) -> ValidationDiagnostic:
        return ValidationDiagnostic(code=code, field=field, message=message)


class ReplayRunner:
    def __init__(
        self,
        *,
        normalizer: TemplateImageNormalizer | None = None,
        loader_factory: type[ReplayDatasetLoader] = ReplayDatasetLoader,
    ) -> None:
        self.normalizer = normalizer or TemplateImageNormalizer()
        self.loader_factory = loader_factory

    def run(self, request: ReplayRunRequest) -> ReplayRunResult:
        load_result = self._dataset_for(request)
        if load_result.dataset is None:
            return ReplayRunResult(diagnostics=load_result.diagnostics)
        dataset = load_result.dataset
        if dataset.diagnostics:
            return ReplayRunResult(dataset_id=dataset.dataset_id, diagnostics=dataset.diagnostics)
        if request.registry is None:
            return ReplayRunResult(
                dataset_id=dataset.dataset_id,
                diagnostics=(self._diagnostic("replay.registry_missing", "registry", "Replay execution requires a TemplateRegistry."),),
            )
        loader = self._loader_for_request(request)
        case_results = tuple(
            self._run_case(case, dataset, request, loader)
            for case in dataset.cases
        )
        summary = ReplayRunSummary.from_results(dataset.cases, case_results)
        return ReplayRunResult(dataset_id=dataset.dataset_id, case_results=case_results, summary=summary)

    def _dataset_for(self, request: ReplayRunRequest) -> ReplayDatasetLoadResult:
        if request.dataset is not None:
            return ReplayDatasetLoadResult(dataset=request.dataset)
        if request.dataset_root is None:
            return ReplayDatasetLoadResult(
                diagnostics=(self._diagnostic("replay.dataset_root_missing", "dataset_root", "Replay manifest loading requires a dataset root."),)
            )
        return self.loader_factory(request.dataset_root).load(request.manifest_path or "")

    def _loader_for_request(self, request: ReplayRunRequest) -> ReplayDatasetLoader:
        root = request.dataset_root if request.dataset_root is not None else Path.cwd()
        return self.loader_factory(root)

    def _run_case(
        self,
        case: ReplayCaseDefinition,
        dataset: ReplayDatasetDefinition,
        request: ReplayRunRequest,
        loader: ReplayDatasetLoader,
    ) -> ReplayCaseResult:
        if case.diagnostics:
            return self._case_result(case, ReplayCaseStatus.INVALID_CASE, diagnostics=case.diagnostics, request=request)
        if not case.executes_detection and not case.executes_scene:
            return self._case_result(
                case,
                ReplayCaseStatus.INVALID_CASE,
                diagnostics=(self._diagnostic("replay.case.no_operation", "case", "Replay case has no executable operation."),),
                request=request,
            )
        if case.executes_scene and request.scene_classifier is None and not request.scene_definitions:
            return self._case_result(
                case,
                ReplayCaseStatus.INVALID_CASE,
                diagnostics=(self._diagnostic("replay.scene_definitions_missing", "scene_definitions", "Scene replay requires scene definitions."),),
                request=request,
            )

        screenshot_result = self._load_screenshot(loader, case)
        if screenshot_result[0] is None:
            return self._case_result(case, ReplayCaseStatus.INPUT_ERROR, diagnostics=screenshot_result[1], request=request)
        screenshot = screenshot_result[0]
        detection: DetectionResult | None = None
        scene: SceneClassificationResult | None = None
        diagnostics: list[ValidationDiagnostic] = []

        if case.executes_detection:
            detection = self._run_detection(case, screenshot, request)
            if detection is None:
                return self._case_result(
                    case,
                    ReplayCaseStatus.PIPELINE_ERROR,
                    diagnostics=(self._diagnostic("replay.pipeline.detection_failed", "template_key", "Template detection failed during replay."),),
                    request=request,
                )
            diagnostics.extend(self._evaluate_detection(case, detection))

        if case.executes_scene:
            scene = self._run_scene(screenshot, request)
            if scene is None:
                return self._case_result(
                    case,
                    ReplayCaseStatus.PIPELINE_ERROR,
                    diagnostics=(self._diagnostic("replay.pipeline.scene_failed", "scene", "Scene classification failed during replay."),),
                    request=request,
                    detection=detection,
                )
            diagnostics.extend(self._evaluate_scene(case, scene))

        status = ReplayCaseStatus.PASS if not diagnostics else ReplayCaseStatus.EXPECTATION_MISMATCH
        return self._case_result(
            case,
            status,
            diagnostics=tuple(diagnostics),
            request=request,
            screenshot=screenshot,
            detection=detection,
            scene=scene,
            dataset=dataset,
        )

    def _load_screenshot(
        self,
        loader: ReplayDatasetLoader,
        case: ReplayCaseDefinition,
    ) -> tuple[ImageInput | None, tuple[ValidationDiagnostic, ...]]:
        try:
            screenshot_path = loader.screenshot_path(case)
        except (OSError, PermissionError, TypeError, ValueError):
            return None, (self._diagnostic("replay.screenshot.invalid_path", "screenshot", "Replay screenshot path is invalid."),)
        normalized = self.normalizer.normalize(screenshot_path, field="screenshot", grayscale=True)
        if normalized.image is None:
            return None, tuple(
                self._diagnostic("replay.screenshot.invalid_image", "screenshot", "Replay screenshot is not a supported readable image.")
                for _diagnostic in (normalized.diagnostics or (None,))
            )
        return normalized.image.pixels.copy(), ()

    def _run_detection(
        self,
        case: ReplayCaseDefinition,
        screenshot: ImageInput,
        request: ReplayRunRequest,
    ) -> DetectionResult | None:
        template_key = case.template_key or (
            case.expectation.detection.semantic_key
            if case.expectation is not None and case.expectation.detection is not None
            else None
        )
        if template_key is None or request.registry is None:
            return None
        analyzer = request.analyzer or TemplateScreenAnalyzer()
        try:
            result = analyzer.match(screenshot, template_key, request.registry)
        except (cv2.error, OSError, TypeError, ValueError, AttributeError):
            return None
        return result if isinstance(result, DetectionResult) else None

    def _run_scene(
        self,
        screenshot: ImageInput,
        request: ReplayRunRequest,
    ) -> SceneClassificationResult | None:
        classifier = request.scene_classifier or SceneClassifier(request.analyzer)
        try:
            result = classifier.classify(
                screenshot,
                request.scene_definitions,
                request.registry,
                analyzer=request.analyzer,
            )
        except (cv2.error, OSError, TypeError, ValueError, AttributeError):
            return None
        return result if isinstance(result, SceneClassificationResult) else None

    def _evaluate_detection(
        self,
        case: ReplayCaseDefinition,
        detection: DetectionResult,
    ) -> tuple[ValidationDiagnostic, ...]:
        expectation = case.expectation.detection if case.expectation and case.expectation.detection else None
        if expectation is None:
            return ()
        diagnostics: list[ValidationDiagnostic] = []
        expected_key = expectation.semantic_key or case.template_key
        if expectation.expected_match is True:
            if detection.matched_semantic_key != expected_key:
                diagnostics.append(self._mismatch("replay.detection.unexpected_no_match", case.case_id, "Expected template detection did not match."))
        elif expectation.expected_match is False and detection.matched_semantic_key is not None:
            diagnostics.append(self._mismatch("replay.detection.unexpected_match", case.case_id, "Template detection matched unexpectedly."))
        if expectation.diagnostic_code is not None and expectation.diagnostic_code not in _diagnostic_codes(detection.metadata.diagnostics):
            diagnostics.append(self._mismatch("replay.detection.diagnostic_mismatch", case.case_id, "Expected detection diagnostic code was not returned."))
        if expectation.confidence_min is not None and detection.confidence + 1e-12 < expectation.confidence_min:
            diagnostics.append(self._mismatch("replay.detection.confidence_low", case.case_id, "Detection confidence is below the expected lower bound."))
        if expectation.confidence_max is not None and detection.confidence > expectation.confidence_max + 1e-12:
            diagnostics.append(self._mismatch("replay.detection.confidence_high", case.case_id, "Detection confidence is above the expected upper bound."))
        if expectation.matched_scale_min is not None:
            if detection.matched_scale is None or detection.matched_scale + 1e-12 < expectation.matched_scale_min:
                diagnostics.append(self._mismatch("replay.detection.scale_low", case.case_id, "Matched scale is below the expected lower bound."))
        if expectation.matched_scale_max is not None:
            if detection.matched_scale is None or detection.matched_scale > expectation.matched_scale_max + 1e-12:
                diagnostics.append(self._mismatch("replay.detection.scale_high", case.case_id, "Matched scale is above the expected upper bound."))
        if expectation.bounding_box is not None:
            if detection.bounding_box is None or not _box_matches(
                detection.bounding_box,
                expectation.bounding_box,
                expectation.bounding_box_tolerance,
            ):
                diagnostics.append(self._mismatch("replay.detection.bounding_box_mismatch", case.case_id, "Detection bounding box did not match expected constraints."))
        return tuple(diagnostics)

    def _evaluate_scene(
        self,
        case: ReplayCaseDefinition,
        scene: SceneClassificationResult,
    ) -> tuple[ValidationDiagnostic, ...]:
        expectation = case.expectation.scene if case.expectation and case.expectation.scene else None
        if expectation is None:
            return ()
        diagnostics: list[ValidationDiagnostic] = []
        if scene.status != expectation.status:
            diagnostics.append(self._mismatch("replay.scene.status_mismatch", case.case_id, "Scene classification status did not match."))
        if expectation.semantic_scene_key is not None and scene.scene_key != expectation.semantic_scene_key:
            diagnostics.append(self._mismatch("replay.scene.key_mismatch", case.case_id, "Scene classification selected the wrong scene."))
        if expectation.candidate_count is not None and len(scene.candidates) != expectation.candidate_count:
            diagnostics.append(self._mismatch("replay.scene.candidate_count_mismatch", case.case_id, "Scene candidate count did not match."))
        if expectation.diagnostic_code is not None and expectation.diagnostic_code not in _diagnostic_codes(scene.diagnostics):
            diagnostics.append(self._mismatch("replay.scene.diagnostic_mismatch", case.case_id, "Expected scene diagnostic code was not returned."))
        return tuple(diagnostics)

    def _case_result(
        self,
        case: ReplayCaseDefinition,
        status: ReplayCaseStatus,
        *,
        diagnostics: tuple[ValidationDiagnostic, ...],
        request: ReplayRunRequest,
        screenshot: ImageInput | None = None,
        detection: DetectionResult | None = None,
        scene: SceneClassificationResult | None = None,
        dataset: ReplayDatasetDefinition | None = None,
    ) -> ReplayCaseResult:
        evidence_reference = None
        all_diagnostics = list(diagnostics)
        final_status = status
        if request.evidence_store is not None and status in request.capture_evidence_on and screenshot is not None:
            try:
                evidence = request.evidence_store.capture(
                    EvidenceCaptureRequest(
                        image=screenshot,
                        evidence_kind=f"replay-{status.value}",
                        detection_result=detection,
                        scene_result=scene,
                        semantic_template_key=case.template_key,
                        semantic_scene_key=scene.scene_key if scene is not None else None,
                        correlation_id=f"{dataset.dataset_id}:{case.case_id}" if dataset is not None else case.case_id,
                    )
                )
                evidence_is_valid = bool(getattr(evidence, "is_valid", False))
                evidence_reference = getattr(evidence, "reference", None) if evidence_is_valid else None
            except (OSError, PermissionError, TypeError, ValueError, AttributeError, cv2.error):
                evidence_is_valid = False
                evidence_reference = None
            if evidence_is_valid and evidence_reference is not None:
                evidence_reference = evidence.reference
            else:
                all_diagnostics.append(
                    self._diagnostic(
                        "replay.evidence_failed",
                        case.case_id,
                        "Replay evidence capture failed.",
                    )
                )
                if status == ReplayCaseStatus.PASS:
                    final_status = ReplayCaseStatus.PIPELINE_ERROR
        return ReplayCaseResult(
            case_id=case.case_id,
            status=final_status,
            diagnostics=tuple(all_diagnostics),
            detection_result=detection,
            scene_result=scene,
            evidence_reference=evidence_reference,
        )

    @staticmethod
    def _diagnostic(code: str, field: str, message: str) -> ValidationDiagnostic:
        return ValidationDiagnostic(code=code, field=field, message=message)

    @staticmethod
    def _mismatch(code: str, case_id: str, message: str) -> ValidationDiagnostic:
        return ValidationDiagnostic(code=code, field=f"cases.{case_id}", message=message)


def _case_with_diagnostic(
    case: ReplayCaseDefinition,
    diagnostic: ValidationDiagnostic,
) -> ReplayCaseDefinition:
    return ReplayCaseDefinition(
        case_id=case.case_id,
        screenshot_path=case.screenshot_path,
        template_key=case.template_key,
        expectation=case.expectation,
        label=case.label,
        diagnostics=(*case.diagnostics, diagnostic),
    )


def _parse_bounding_box(raw_box: Any) -> BoundingBox | None:
    if raw_box is None:
        return None
    if not isinstance(raw_box, Mapping):
        raise ValueError("bounding_box must be an object")
    if set(raw_box) != _BOUNDING_BOX_FIELDS:
        raise ValueError("bounding_box must contain only x, y, width, and height.")
    return BoundingBox(
        x=_require_int(raw_box["x"], "bounding box x"),
        y=_require_int(raw_box["y"], "bounding box y"),
        width=_require_int(raw_box["width"], "bounding box width"),
        height=_require_int(raw_box["height"], "bounding box height"),
    )


def _box_matches(actual: BoundingBox, expected: BoundingBox, tolerance: int) -> bool:
    return (
        abs(actual.x - expected.x) <= tolerance
        and abs(actual.y - expected.y) <= tolerance
        and abs(actual.width - expected.width) <= tolerance
        and abs(actual.height - expected.height) <= tolerance
    )


def _diagnostic_codes(diagnostics: Sequence[ValidationDiagnostic]) -> set[str]:
    return {diagnostic.code for diagnostic in diagnostics}


def _required_text(
    mapping: Mapping[str, Any],
    key: str,
    diagnostics: list[ValidationDiagnostic],
    parent_field: str,
) -> str:
    value = mapping.get(key)
    field = f"{parent_field}.{key}"
    if not isinstance(value, str) or not value.strip():
        diagnostics.append(
            ValidationDiagnostic(
                code=f"replay.missing_{key}",
                field=field,
                message=f"{field} is required and must be a non-empty string.",
            )
        )
        return ""
    return value.strip()


def _optional_text(
    value: Any,
    diagnostics: list[ValidationDiagnostic],
    field: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        diagnostics.append(
            ValidationDiagnostic(
                code="replay.invalid_text",
                field=field,
                message=f"{field} must be a non-empty string when provided.",
            )
        )
        return None
    return value.strip()


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


def _require_finite_number(value: Any, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be a finite number.")
    return numeric


def _require_non_negative_int(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")


def _require_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer.")
    return value


def _relative_path(value: str) -> Path | None:
    if not isinstance(value, str):
        return None
    normalized_value = value.strip()
    if not normalized_value or normalized_value != value:
        return None
    raw_parts = normalized_value.replace("\\", "/").split("/")
    if any(part in ("", ".", "..") for part in raw_parts):
        return None
    for part in raw_parts:
        if part != part.strip() or part.endswith((".", " ")) or ":" in part:
            return None
        stem = part.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED_NAMES:
            return None
    windows_path = PureWindowsPath(normalized_value)
    posix_path = PurePosixPath(normalized_value.replace("\\", "/"))
    if windows_path.is_absolute() or windows_path.drive or posix_path.is_absolute():
        return None
    return Path(*posix_path.parts)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sorted_diagnostics(
    diagnostics: Sequence[ValidationDiagnostic],
) -> tuple[ValidationDiagnostic, ...]:
    return tuple(sorted(tuple(diagnostics), key=lambda item: (item.field, item.code, item.message)))


def _unknown_key_diagnostics(
    mapping: Mapping[str, Any],
    allowed_keys: frozenset[str],
    field: str,
) -> tuple[ValidationDiagnostic, ...]:
    return tuple(
        ValidationDiagnostic(
            code="replay.unknown_field",
            field=f"{field}.{key_text}",
            message="Replay dataset contains an unsupported field.",
        )
        for key_text in sorted(str(key) for key in mapping if key not in allowed_keys)
    )
