from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationDiagnostic:
    code: str
    message: str
    field: str = ""
    severity: ValidationSeverity = ValidationSeverity.ERROR

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("diagnostic code must be a non-empty string.")
        if not self.message.strip():
            raise ValueError("diagnostic message must be a non-empty string.")


@dataclass(frozen=True)
class ValidationReport:
    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def is_valid(self) -> bool:
        return not any(
            diagnostic.severity == ValidationSeverity.ERROR
            for diagnostic in self.diagnostics
        )


@dataclass(frozen=True)
class ResolutionProfile:
    key: str
    width: int
    height: int

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("resolution profile key must be a non-empty string.")
        _require_positive_int(self.width, "resolution profile width")
        _require_positive_int(self.height, "resolution profile height")


@dataclass(frozen=True)
class RegionOfInterest:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        _require_non_negative_int(self.x, "ROI x")
        _require_non_negative_int(self.y, "ROI y")
        _require_positive_int(self.width, "ROI width")
        _require_positive_int(self.height, "ROI height")


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        _require_non_negative_int(self.x, "bounding box x")
        _require_non_negative_int(self.y, "bounding box y")
        _require_positive_int(self.width, "bounding box width")
        _require_positive_int(self.height, "bounding box height")


@dataclass(frozen=True)
class ScaleRange:
    minimum: float = 1.0
    maximum: float = 1.0

    def __post_init__(self) -> None:
        minimum = _require_finite_number(self.minimum, "scale minimum")
        maximum = _require_finite_number(self.maximum, "scale maximum")
        if minimum <= 0.0 or maximum <= 0.0:
            raise ValueError("scale range values must be positive.")
        if minimum > maximum:
            raise ValueError("scale minimum must not exceed scale maximum.")


@dataclass(frozen=True)
class SceneConstraints:
    allowed: tuple[str, ...] = ()
    required: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        allowed = _normalize_text_tuple(self.allowed, "allowed scenes")
        required = _normalize_text_tuple(self.required, "required scenes")
        if allowed and not set(required).issubset(set(allowed)):
            raise ValueError("required scenes must be included in allowed scenes.")
        object.__setattr__(self, "allowed", allowed)
        object.__setattr__(self, "required", required)


@dataclass(frozen=True)
class TemplateDefinition:
    semantic_key: str
    template_pack_version: str
    language: str
    resolution_profile: str
    source: Path
    region_of_interest: RegionOfInterest
    confidence_threshold: float
    scale_range: ScaleRange = field(default_factory=ScaleRange)
    mask: Path | None = None
    scene_constraints: SceneConstraints = field(default_factory=SceneConstraints)
    source_reference: str = ""

    def __post_init__(self) -> None:
        if not self.semantic_key.strip():
            raise ValueError("semantic key must be a non-empty string.")
        if not self.template_pack_version.strip():
            raise ValueError("template pack version must be a non-empty string.")
        if not self.language.strip():
            raise ValueError("language must be a non-empty string.")
        if not self.resolution_profile.strip():
            raise ValueError("resolution profile must be a non-empty string.")
        _require_confidence(self.confidence_threshold, "confidence threshold")
        object.__setattr__(self, "source", Path(self.source))
        if self.mask is not None:
            object.__setattr__(self, "mask", Path(self.mask))


@dataclass(frozen=True)
class TemplatePack:
    version: str
    languages: tuple[str, ...]
    resolution_profiles: tuple[ResolutionProfile, ...]
    templates: tuple[TemplateDefinition, ...]
    root: Path

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("template pack version must be a non-empty string.")
        object.__setattr__(self, "languages", _normalize_text_tuple(self.languages, "languages"))
        object.__setattr__(self, "resolution_profiles", tuple(self.resolution_profiles))
        object.__setattr__(self, "templates", tuple(self.templates))
        object.__setattr__(self, "root", Path(self.root))


@dataclass(frozen=True)
class MatchingMetadata:
    matcher: str = ""
    normalized: bool = False
    region_of_interest: RegionOfInterest | None = None
    elapsed_ms: float | None = None
    candidate_count: int = 0
    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.elapsed_ms is not None:
            elapsed_ms = _require_finite_number(self.elapsed_ms, "elapsed milliseconds")
            if elapsed_ms < 0.0:
                raise ValueError("elapsed milliseconds must not be negative.")
        _require_non_negative_int(self.candidate_count, "candidate count")
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


@dataclass(frozen=True)
class DetectionResult:
    matched_semantic_key: str | None
    confidence: float
    bounding_box: BoundingBox | None = None
    matched_scale: float | None = None
    scene: str | None = None
    template_pack_version: str | None = None
    evidence_reference: str | None = None
    metadata: MatchingMetadata = field(default_factory=MatchingMetadata)

    def __post_init__(self) -> None:
        _require_confidence(self.confidence, "detection confidence")
        if self.matched_scale is not None:
            matched_scale = _require_finite_number(self.matched_scale, "matched scale")
            if matched_scale <= 0.0:
                raise ValueError("matched scale must be positive.")


def _normalize_text_tuple(values: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must contain only non-empty strings.")
        normalized.append(value.strip())
    return tuple(normalized)


def _require_confidence(value: Any, field_name: str) -> float:
    confidence = _require_finite_number(value, field_name)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    return confidence


def _require_finite_number(value: Any, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be a finite number.")
    return numeric


def _require_positive_int(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


def _require_non_negative_int(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
