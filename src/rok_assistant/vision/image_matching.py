from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from .template_models import (
    BoundingBox,
    DetectionResult,
    MatchingMetadata,
    RegionOfInterest,
    TemplateDefinition,
    ValidationDiagnostic,
)
from .template_registry import TemplateRegistry, TemplateRegistryError

ImageInput = str | Path | np.ndarray
_MAX_SCALE_COUNT = 10_000


def find_template(
    screenshot_path: str | Path,
    template_path: str | Path,
    threshold: float = 0.8,
) -> dict[str, Any]:
    logger = logging.getLogger("ImageMatching")
    screenshot = Path(screenshot_path)
    template = Path(template_path)
    if not screenshot.exists():
        raise FileNotFoundError(f"Screenshot not found: {screenshot}")
    if not template.exists():
        raise FileNotFoundError(f"Template not found: {template}")

    screenshot_image = cv2.imread(str(screenshot), cv2.IMREAD_COLOR)
    template_image = cv2.imread(str(template), cv2.IMREAD_COLOR)
    if screenshot_image is None:
        raise ValueError(f"Screenshot is not a readable image: {screenshot}")
    if template_image is None:
        raise ValueError(f"Template is not a readable image: {template}")

    screenshot_height, screenshot_width = screenshot_image.shape[:2]
    template_height, template_width = template_image.shape[:2]
    if template_width > screenshot_width or template_height > screenshot_height:
        logger.info(
            "[ImageMatch] Template larger than screenshot: template=%s screenshot=%s",
            template,
            screenshot,
        )
        return {"found": False, "confidence": 0.0, "x": -1, "y": -1}

    result = cv2.matchTemplate(screenshot_image, template_image, cv2.TM_CCOEFF_NORMED)
    _minimum_value, maximum_value, _minimum_location, maximum_location = cv2.minMaxLoc(result)
    confidence = float(maximum_value)
    found = confidence >= threshold
    x, y = maximum_location if found else (-1, -1)
    logger.info(
        "[ImageMatch] template=%s screenshot=%s confidence=%.4f found=%s x=%s y=%s",
        template,
        screenshot,
        confidence,
        found,
        x,
        y,
    )
    return {
        "found": found,
        "confidence": confidence,
        "x": int(x),
        "y": int(y),
    }


@dataclass(frozen=True)
class NormalizedImage:
    pixels: np.ndarray
    width: int
    height: int
    grayscale: bool = True


@dataclass(frozen=True)
class ImageNormalizationResult:
    image: NormalizedImage | None = None
    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    @property
    def is_valid(self) -> bool:
        return self.image is not None and not self.diagnostics


@dataclass(frozen=True)
class TemplateMatchRequest:
    screenshot: ImageInput
    semantic_key: str
    registry: TemplateRegistry
    scene: str | None = None


@dataclass(frozen=True)
class _MatchCandidate:
    confidence: float
    x: int
    y: int
    width: int
    height: int
    scale: float

    @property
    def tie_breaker(self) -> tuple[float, int, int]:
        return (self.scale, self.y, self.x)


@dataclass(frozen=True)
class _PreparedTemplate:
    definition: TemplateDefinition
    image: NormalizedImage
    mask: NormalizedImage | None = None


class TemplateImageNormalizer:
    """Normalize image inputs into deterministic uint8 grayscale arrays.

    The matching path intentionally converts BGR/BGRA inputs to grayscale.
    Alpha channels are not used as masks; template masks come from explicit
    template metadata and are normalized separately.
    """

    def normalize(
        self,
        image_input: ImageInput,
        *,
        field: str,
        grayscale: bool = True,
    ) -> ImageNormalizationResult:
        try:
            image = self._load_input(image_input)
            normalized = self._normalize_array(image, grayscale=grayscale)
        except (cv2.error, OSError, TypeError, ValueError):
            return ImageNormalizationResult(
                diagnostics=(
                    ValidationDiagnostic(
                        code="image.invalid",
                        field=field,
                        message=f"{field} is not a supported readable image.",
                    ),
                )
            )
        height, width = normalized.shape[:2]
        return ImageNormalizationResult(
            image=NormalizedImage(
                pixels=normalized,
                width=int(width),
                height=int(height),
                grayscale=grayscale,
            )
        )

    @staticmethod
    def _load_input(image_input: ImageInput) -> np.ndarray:
        if isinstance(image_input, np.ndarray):
            return image_input.copy()
        path = Path(image_input)
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("unreadable image")
        return image

    @staticmethod
    def _normalize_array(image: np.ndarray, *, grayscale: bool) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image must be a non-empty numpy array")
        if image.dtype != np.uint8:
            raise ValueError("image must use uint8 pixels")
        if image.ndim == 2:
            normalized = image.copy()
        elif image.ndim == 3 and image.shape[2] == 1:
            normalized = image[:, :, 0].copy()
        elif image.ndim == 3 and image.shape[2] == 3:
            normalized = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if grayscale else image.copy()
        elif image.ndim == 3 and image.shape[2] == 4:
            normalized = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY) if grayscale else image[:, :, :3].copy()
        else:
            raise ValueError("unsupported image shape")
        if normalized.size == 0:
            raise ValueError("normalized image is empty")
        return np.ascontiguousarray(normalized.copy())


class TemplateScreenAnalyzer:
    def __init__(
        self,
        *,
        normalizer: TemplateImageNormalizer | None = None,
        scale_step: float = 0.05,
        clock: Any = time.monotonic,
    ) -> None:
        if not isinstance(scale_step, int | float) or not math.isfinite(float(scale_step)) or scale_step <= 0:
            raise ValueError("scale_step must be a positive finite number.")
        self.normalizer = normalizer or TemplateImageNormalizer()
        self.scale_step = float(scale_step)
        self.clock = clock

    def match(
        self,
        screenshot: ImageInput,
        semantic_key: str,
        registry: TemplateRegistry,
        *,
        scene: str | None = None,
    ) -> DetectionResult:
        started_at = self.clock()
        try:
            definition = registry.get(semantic_key)
        except TemplateRegistryError as exc:
            return self._result(
                semantic_key=None,
                confidence=0.0,
                started_at=started_at,
                diagnostics=exc.diagnostics,
            )

        scene_diagnostics = self._validate_scene(definition, scene)
        if scene_diagnostics:
            return self._result(
                semantic_key=None,
                confidence=0.0,
                started_at=started_at,
                scene=scene,
                pack_version=definition.template_pack_version,
                roi=definition.region_of_interest,
                diagnostics=scene_diagnostics,
            )

        screenshot_result = self.normalizer.normalize(
            screenshot,
            field="screenshot",
            grayscale=True,
        )
        if screenshot_result.image is None:
            return self._result(
                semantic_key=None,
                confidence=0.0,
                started_at=started_at,
                scene=scene,
                pack_version=definition.template_pack_version,
                roi=definition.region_of_interest,
                diagnostics=screenshot_result.diagnostics,
            )

        roi_result = self._extract_roi(screenshot_result.image, definition.region_of_interest)
        if roi_result[0] is None:
            return self._result(
                semantic_key=None,
                confidence=0.0,
                started_at=started_at,
                scene=scene,
                pack_version=definition.template_pack_version,
                roi=definition.region_of_interest,
                diagnostics=roi_result[1],
            )
        roi_pixels = roi_result[0]

        prepared = self._load_template(registry, definition)
        if prepared[0] is None:
            return self._result(
                semantic_key=None,
                confidence=0.0,
                started_at=started_at,
                scene=scene,
                pack_version=definition.template_pack_version,
                roi=definition.region_of_interest,
                diagnostics=prepared[1],
            )
        template = prepared[0]

        candidate_count = 0
        best: _MatchCandidate | None = None
        diagnostics: list[ValidationDiagnostic] = []
        scales, scale_diagnostics = self._scale_sequence(definition)
        if scale_diagnostics:
            return self._result(
                semantic_key=None,
                confidence=0.0,
                started_at=started_at,
                scene=scene,
                pack_version=definition.template_pack_version,
                roi=definition.region_of_interest,
                diagnostics=scale_diagnostics,
            )
        seen_sizes: set[tuple[int, int]] = set()
        for scale in scales:
            scaled = self._resize_template(template.image.pixels, scale)
            if scaled is None:
                diagnostics.append(self._diagnostic("match.invalid_scale", "scale_range", "Scale produced an invalid template size."))
                continue
            scaled_height, scaled_width = scaled.shape[:2]
            size_key = (int(scaled_width), int(scaled_height))
            if size_key in seen_sizes:
                continue
            seen_sizes.add(size_key)
            if scaled_width > roi_pixels.shape[1] or scaled_height > roi_pixels.shape[0]:
                continue
            scaled_mask = None
            if template.mask is not None:
                scaled_mask = self._resize_mask(template.mask.pixels, scale)
                if scaled_mask is None or scaled_mask.shape[:2] != scaled.shape[:2]:
                    return self._result(
                        semantic_key=None,
                        confidence=0.0,
                        started_at=started_at,
                        scene=scene,
                        pack_version=definition.template_pack_version,
                        roi=definition.region_of_interest,
                        diagnostics=(
                            self._diagnostic(
                                "template.mask_incompatible",
                                "template.mask",
                                "Template mask dimensions are not compatible with the template image.",
                            ),
                        ),
                    )
            match = self._match_at_scale(roi_pixels, scaled, scaled_mask, scale)
            candidate_count += 1
            if match is None:
                return self._result(
                    semantic_key=None,
                    confidence=0.0,
                    started_at=started_at,
                    scene=scene,
                    pack_version=definition.template_pack_version,
                    roi=definition.region_of_interest,
                    diagnostics=(
                        self._diagnostic(
                            "match.failed",
                            "template",
                            "Template matching failed for the normalized images.",
                        ),
                    ),
                )
            if not self._candidate_fits_roi(match, roi_pixels):
                return self._result(
                    semantic_key=None,
                    confidence=0.0,
                    started_at=started_at,
                    scene=scene,
                    pack_version=definition.template_pack_version,
                    roi=definition.region_of_interest,
                    diagnostics=(
                        self._diagnostic(
                            "match.invalid_candidate",
                            "template",
                            "Template match coordinates are outside the selected ROI.",
                        ),
                    ),
                )
            if best is None or self._is_better_candidate(match, best):
                best = match

        if best is None:
            return self._result(
                semantic_key=None,
                confidence=0.0,
                started_at=started_at,
                scene=scene,
                pack_version=definition.template_pack_version,
                roi=definition.region_of_interest,
                candidate_count=candidate_count,
                diagnostics=(
                    self._diagnostic(
                        "match.no_eligible_scale",
                        "scale_range",
                        "No configured template scale fits inside the selected ROI.",
                    ),
                    *diagnostics,
                ),
            )

        full_x = definition.region_of_interest.x + best.x
        full_y = definition.region_of_interest.y + best.y
        if best.confidence < definition.confidence_threshold:
            return self._result(
                semantic_key=None,
                confidence=best.confidence,
                started_at=started_at,
                scene=scene,
                pack_version=definition.template_pack_version,
                roi=definition.region_of_interest,
                matched_scale=best.scale,
                candidate_count=candidate_count,
                diagnostics=(
                    self._diagnostic(
                        "match.below_threshold",
                        "confidence",
                        "Best match confidence is below the template threshold.",
                    ),
                ),
            )

        return self._result(
            semantic_key=definition.semantic_key,
            confidence=best.confidence,
            started_at=started_at,
            bounding_box=BoundingBox(
                x=full_x,
                y=full_y,
                width=best.width,
                height=best.height,
            ),
            matched_scale=best.scale,
            scene=scene,
            pack_version=definition.template_pack_version,
            roi=definition.region_of_interest,
            candidate_count=candidate_count,
        )

    def _load_template(
        self,
        registry: TemplateRegistry,
        definition: TemplateDefinition,
    ) -> tuple[_PreparedTemplate | None, tuple[ValidationDiagnostic, ...]]:
        source_result = self.normalizer.normalize(
            registry.resolve_template_path(definition),
            field="template.source",
            grayscale=True,
        )
        if source_result.image is None:
            return None, (
                self._diagnostic(
                    "template.invalid_source_image",
                    "template.source",
                    "Template source is not a supported readable image.",
                ),
            )
        mask_image = None
        mask_path = registry.resolve_mask_path(definition)
        if mask_path is not None:
            mask_result = self.normalizer.normalize(mask_path, field="template.mask", grayscale=True)
            if mask_result.image is None:
                return None, (
                    self._diagnostic(
                        "template.invalid_mask_image",
                        "template.mask",
                        "Template mask is not a supported readable image.",
                    ),
                )
            mask_image = mask_result.image
            if (
                mask_image.width != source_result.image.width
                or mask_image.height != source_result.image.height
            ):
                return None, (
                    self._diagnostic(
                        "template.mask_incompatible",
                        "template.mask",
                        "Template mask dimensions are not compatible with the template image.",
                    ),
                )
        return _PreparedTemplate(definition=definition, image=source_result.image, mask=mask_image), ()

    @staticmethod
    def _extract_roi(
        screenshot: NormalizedImage,
        roi: RegionOfInterest,
    ) -> tuple[np.ndarray | None, tuple[ValidationDiagnostic, ...]]:
        if roi.width <= 0 or roi.height <= 0:
            return None, (
                TemplateScreenAnalyzer._diagnostic(
                    "match.invalid_roi",
                    "template.roi",
                    "Template ROI must have positive dimensions.",
                ),
            )
        if roi.x < 0 or roi.y < 0 or roi.x + roi.width > screenshot.width or roi.y + roi.height > screenshot.height:
            return None, (
                TemplateScreenAnalyzer._diagnostic(
                    "match.invalid_roi",
                    "template.roi",
                    "Template ROI must be contained by the screenshot dimensions.",
                ),
            )
        return screenshot.pixels[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width].copy(), ()

    def _scale_sequence(
        self,
        definition: TemplateDefinition,
    ) -> tuple[tuple[float, ...], tuple[ValidationDiagnostic, ...]]:
        try:
            minimum = float(definition.scale_range.minimum)
            maximum = float(definition.scale_range.maximum)
        except (TypeError, ValueError):
            return (), (
                self._diagnostic(
                    "match.invalid_scale_range",
                    "scale_range",
                    "Template scale range must contain finite positive numbers.",
                ),
            )
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum <= 0.0
            or maximum <= 0.0
            or minimum > maximum
        ):
            return (), (
                self._diagnostic(
                    "match.invalid_scale_range",
                    "scale_range",
                    "Template scale range must be finite, positive, and ordered.",
                ),
            )
        values: list[float] = []
        current = minimum
        scale_count = 0
        while current <= maximum + 1e-9:
            if scale_count >= _MAX_SCALE_COUNT:
                return (), (
                    self._diagnostic(
                        "match.scale_range_too_dense",
                        "scale_range",
                        "Template scale range produces too many candidate scales.",
                    ),
                )
            values.append(round(current, 6))
            current += self.scale_step
            scale_count += 1
        if not values or not math.isclose(values[-1], maximum, rel_tol=0.0, abs_tol=1e-9):
            values.append(round(maximum, 6))
        return tuple(dict.fromkeys(values)), ()

    @staticmethod
    def _resize_template(image: np.ndarray, scale: float) -> np.ndarray | None:
        height, width = image.shape[:2]
        scaled_width = int(round(width * scale))
        scaled_height = int(round(height * scale))
        if scaled_width <= 0 or scaled_height <= 0:
            return None
        if scaled_width == width and scaled_height == height:
            return image.copy()
        return cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _resize_mask(mask: np.ndarray, scale: float) -> np.ndarray | None:
        height, width = mask.shape[:2]
        scaled_width = int(round(width * scale))
        scaled_height = int(round(height * scale))
        if scaled_width <= 0 or scaled_height <= 0:
            return None
        if scaled_width == width and scaled_height == height:
            return mask.copy()
        return cv2.resize(mask, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _match_at_scale(
        roi_pixels: np.ndarray,
        template_pixels: np.ndarray,
        mask_pixels: np.ndarray | None,
        scale: float,
    ) -> _MatchCandidate | None:
        try:
            if mask_pixels is None:
                result = cv2.matchTemplate(
                    roi_pixels,
                    template_pixels,
                    cv2.TM_CCORR_NORMED,
                )
            else:
                result = cv2.matchTemplate(
                    roi_pixels,
                    template_pixels,
                    cv2.TM_CCORR_NORMED,
                    None,
                    mask_pixels,
                )
            finite_result = np.where(np.isfinite(result), result, -np.inf)
            if not np.isfinite(finite_result).any():
                return None
            max_index = int(np.argmax(finite_result))
            max_y, max_x = np.unravel_index(max_index, finite_result.shape)
            max_value = float(finite_result[max_y, max_x])
            max_location = (int(max_x), int(max_y))
        except cv2.error:
            return None
        confidence = float(max_value)
        if not math.isfinite(confidence):
            return None
        height, width = template_pixels.shape[:2]
        return _MatchCandidate(
            confidence=max(0.0, min(1.0, confidence)),
            x=int(max_location[0]),
            y=int(max_location[1]),
            width=int(width),
            height=int(height),
            scale=float(scale),
        )

    @staticmethod
    def _is_better_candidate(candidate: _MatchCandidate, current: _MatchCandidate) -> bool:
        if candidate.confidence > current.confidence + 1e-12:
            return True
        if math.isclose(candidate.confidence, current.confidence, rel_tol=0.0, abs_tol=1e-12):
            return candidate.tie_breaker < current.tie_breaker
        return False

    @staticmethod
    def _candidate_fits_roi(candidate: _MatchCandidate, roi_pixels: np.ndarray) -> bool:
        roi_height, roi_width = roi_pixels.shape[:2]
        return (
            candidate.x >= 0
            and candidate.y >= 0
            and candidate.width > 0
            and candidate.height > 0
            and candidate.x + candidate.width <= roi_width
            and candidate.y + candidate.height <= roi_height
        )

    @staticmethod
    def _validate_scene(
        definition: TemplateDefinition,
        scene: str | None,
    ) -> tuple[ValidationDiagnostic, ...]:
        constraints = definition.scene_constraints
        if constraints.required and scene is None:
            return (
                TemplateScreenAnalyzer._diagnostic(
                    "match.scene_required",
                    "scene",
                    "Template requires a caller-supplied scene.",
                ),
            )
        if scene is not None and constraints.allowed and scene not in constraints.allowed:
            return (
                TemplateScreenAnalyzer._diagnostic(
                    "match.scene_disallowed",
                    "scene",
                    "Caller-supplied scene is not allowed for this template.",
                ),
            )
        if scene is not None and constraints.required and scene not in constraints.required:
            return (
                TemplateScreenAnalyzer._diagnostic(
                    "match.scene_required",
                    "scene",
                    "Caller-supplied scene does not satisfy required template scenes.",
                ),
            )
        return ()

    def _result(
        self,
        *,
        semantic_key: str | None,
        confidence: float,
        started_at: float,
        bounding_box: BoundingBox | None = None,
        matched_scale: float | None = None,
        scene: str | None = None,
        pack_version: str | None = None,
        roi: RegionOfInterest | None = None,
        candidate_count: int = 0,
        diagnostics: tuple[ValidationDiagnostic, ...] = (),
    ) -> DetectionResult:
        elapsed_ms = max(0.0, (self.clock() - started_at) * 1000.0)
        return DetectionResult(
            matched_semantic_key=semantic_key,
            confidence=confidence,
            bounding_box=bounding_box,
            matched_scale=matched_scale,
            scene=scene,
            template_pack_version=pack_version,
            metadata=MatchingMetadata(
                matcher=self.__class__.__name__,
                normalized=True,
                region_of_interest=roi,
                elapsed_ms=elapsed_ms,
                candidate_count=candidate_count,
                diagnostics=diagnostics,
            ),
        )

    @staticmethod
    def _diagnostic(code: str, field: str, message: str) -> ValidationDiagnostic:
        return ValidationDiagnostic(code=code, field=field, message=message)
