from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Protocol
import hashlib
import json
import math
import os
import re
import uuid

import cv2
import numpy as np

from .image_matching import ImageInput
from .scene_classification import SceneClassificationResult
from .template_models import (
    BoundingBox,
    DetectionResult,
    RegionOfInterest,
    ValidationDiagnostic,
)

EVIDENCE_SCHEMA_VERSION = 1
_MAX_COLLISION_ATTEMPTS = 10_000
_TOKEN_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class _EvidenceWriteError(Exception):
    def __init__(self, diagnostics: tuple[ValidationDiagnostic, ...]):
        super().__init__("Evidence persistence failed.")
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class EvidenceReference:
    image_path: str
    metadata_path: str
    content_hash: str

    def __post_init__(self) -> None:
        _require_relative_reference(self.image_path, "image evidence path")
        _require_relative_reference(self.metadata_path, "metadata evidence path")
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_hash):
            raise ValueError("content hash must be a SHA-256 hex digest.")


@dataclass(frozen=True)
class EvidenceMetadata:
    schema_version: int
    captured_at: str
    evidence_kind: str
    screenshot_width: int
    screenshot_height: int
    evidence_width: int
    evidence_height: int
    content_hash: str
    semantic_template_key: str | None = None
    semantic_scene_key: str | None = None
    detection_confidence: float | None = None
    matched_scale: float | None = None
    bounding_box: BoundingBox | None = None
    template_pack_version: str | None = None
    classification_status: str | None = None
    classification_score: float | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported evidence schema version.")
        if not self.captured_at.strip():
            raise ValueError("capture timestamp must be a non-empty string.")
        if not self.evidence_kind.strip():
            raise ValueError("evidence kind must be a non-empty string.")
        _require_positive_int(self.screenshot_width, "screenshot width")
        _require_positive_int(self.screenshot_height, "screenshot height")
        _require_positive_int(self.evidence_width, "evidence width")
        _require_positive_int(self.evidence_height, "evidence height")
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_hash):
            raise ValueError("content hash must be a SHA-256 hex digest.")
        if self.detection_confidence is not None:
            _require_score(self.detection_confidence, "detection confidence")
        if self.matched_scale is not None:
            scale = _require_finite_number(self.matched_scale, "matched scale")
            if scale <= 0.0:
                raise ValueError("matched scale must be positive.")
        if self.classification_score is not None:
            _require_score(self.classification_score, "classification score")

    def to_json_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "captured_at": self.captured_at,
            "content_hash": self.content_hash,
            "evidence_height": self.evidence_height,
            "evidence_kind": self.evidence_kind,
            "evidence_width": self.evidence_width,
            "schema_version": self.schema_version,
            "screenshot_height": self.screenshot_height,
            "screenshot_width": self.screenshot_width,
        }
        optional: dict[str, Any] = {
            "classification_score": self.classification_score,
            "classification_status": self.classification_status,
            "correlation_id": self.correlation_id,
            "detection_confidence": self.detection_confidence,
            "matched_scale": self.matched_scale,
            "semantic_scene_key": self.semantic_scene_key,
            "semantic_template_key": self.semantic_template_key,
            "template_pack_version": self.template_pack_version,
        }
        payload.update(
            {
                key: value
                for key, value in optional.items()
                if value is not None
            }
        )
        if self.bounding_box is not None:
            payload["bounding_box"] = {
                "height": self.bounding_box.height,
                "width": self.bounding_box.width,
                "x": self.bounding_box.x,
                "y": self.bounding_box.y,
            }
        return payload


@dataclass(frozen=True)
class EvidenceCaptureRequest:
    image: ImageInput
    evidence_kind: str = "screenshot"
    relative_directory: str = "screenshots"
    crop: BoundingBox | RegionOfInterest | None = None
    detection_result: DetectionResult | None = None
    scene_result: SceneClassificationResult | None = None
    semantic_template_key: str | None = None
    semantic_scene_key: str | None = None
    correlation_id: str | None = None


@dataclass(frozen=True)
class EvidenceCaptureResult:
    reference: EvidenceReference | None = None
    metadata: EvidenceMetadata | None = None
    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if self.reference is not None and self.metadata is None:
            raise ValueError("successful evidence capture requires metadata.")
        if self.reference is None and not self.diagnostics:
            raise ValueError("failed evidence capture requires diagnostics.")

    @property
    def is_valid(self) -> bool:
        return self.reference is not None and self.metadata is not None and not self.diagnostics


class EvidenceStore(Protocol):
    def capture(self, request: EvidenceCaptureRequest) -> EvidenceCaptureResult:
        ...


class FileSystemEvidenceStore:
    """Filesystem-backed screenshot evidence store.

    Evidence is encoded as PNG and metadata as deterministic JSON. Public
    references are repository-relative strings rooted below the configured
    evidence root; absolute machine paths never leave this boundary. The
    content hash is SHA-256 over the exact PNG bytes persisted for the evidence
    image, not over the pre-encoded pixel array.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        identifier_factory: Callable[[], str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.clock = clock or (lambda: datetime.now(UTC))
        self.identifier_factory = identifier_factory or (lambda: uuid.uuid4().hex)

    def capture(self, request: EvidenceCaptureRequest) -> EvidenceCaptureResult:
        diagnostics = self._validate_context(request)
        if diagnostics:
            return self._failure(diagnostics)
        diagnostics = self._validate_request_text(request)
        if diagnostics:
            return self._failure(diagnostics)

        directory_result = self._evidence_directory(request.relative_directory)
        if directory_result[0] is None:
            return self._failure(directory_result[1])
        directory = directory_result[0]

        normalized = self._normalize_image(request.image)
        if normalized[0] is None:
            return self._failure(normalized[1])
        image, screenshot_width, screenshot_height = normalized[0]

        try:
            crop_box = self._crop_box(request.crop)
        except (TypeError, ValueError):
            return self._failure(
                (
                    self._diagnostic(
                        "evidence.invalid_crop",
                        "crop",
                        "Evidence crop must be a bounding box or ROI inside the screenshot.",
                    ),
                )
            )
        if crop_box is not None:
            crop_result = self._crop(image, crop_box)
            if crop_result[0] is None:
                return self._failure(crop_result[1])
            image = crop_result[0]

        timestamp = self._capture_timestamp()
        if timestamp[0] is None:
            return self._failure(timestamp[1])
        captured_at = timestamp[0]

        name_parts = self._name_parts(request, captured_at)
        if name_parts[0] is None:
            return self._failure(name_parts[1])
        stem = name_parts[0]

        encoded = self._encode_png(image)
        if encoded[0] is None:
            return self._failure(encoded[1])
        image_bytes = encoded[0]
        content_hash = hashlib.sha256(image_bytes).hexdigest()

        metadata_result = self._metadata(
            request,
            captured_at=captured_at,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            evidence_width=int(image.shape[1]),
            evidence_height=int(image.shape[0]),
            content_hash=content_hash,
            crop_box=crop_box,
        )
        if metadata_result[0] is None:
            return self._failure(metadata_result[1])
        metadata = metadata_result[0]

        serialized = self._serialize_metadata(metadata)
        if serialized[0] is None:
            return self._failure(serialized[1])
        metadata_bytes = serialized[0]

        try:
            image_path, metadata_path = self._unique_paths(directory, stem)
            self._write_atomically(image_path, metadata_path, image_bytes, metadata_bytes)
        except _EvidenceWriteError as exc:
            return self._failure(exc.diagnostics)
        except (OSError, PermissionError, TypeError, ValueError):
            return self._failure(
                (
                    self._diagnostic(
                        "evidence.write_failed",
                        "evidence_store",
                        "Evidence files could not be written.",
                    ),
                )
            )

        root = self._resolved_root()
        try:
            reference = EvidenceReference(
                image_path=_relative_posix(image_path, root),
                metadata_path=_relative_posix(metadata_path, root),
                content_hash=content_hash,
            )
        except ValueError:
            return self._failure(
                (
                    self._diagnostic(
                        "evidence.path_escape",
                        "evidence_store",
                        "Evidence path escaped the configured root.",
                    ),
                )
            )
        return EvidenceCaptureResult(reference=reference, metadata=metadata)

    def _evidence_directory(
        self,
        relative_directory: str,
    ) -> tuple[Path | None, tuple[ValidationDiagnostic, ...]]:
        relative = _relative_path(relative_directory)
        if relative is None:
            return None, (
                self._diagnostic(
                    "evidence.invalid_relative_path",
                    "relative_directory",
                    "Evidence directory must be a relative path inside the evidence root.",
                ),
            )
        try:
            root = self._resolved_root()
            directory = root / relative
            directory.mkdir(parents=True, exist_ok=True)
            resolved = directory.resolve()
        except (OSError, PermissionError):
            return None, (
                self._diagnostic(
                    "evidence.write_failed",
                    "relative_directory",
                    "Evidence directory could not be created.",
                ),
            )
        except ValueError:
            return None, (
                self._diagnostic(
                    "evidence.path_escape",
                    "relative_directory",
                    "Evidence root must not be a symlink or path escape.",
                ),
            )
        if not _is_relative_to(resolved, root):
            return None, (
                self._diagnostic(
                    "evidence.path_escape",
                    "relative_directory",
                    "Evidence directory must remain inside the configured root.",
                ),
            )
        return resolved, ()

    def _normalize_image(
        self,
        image_input: ImageInput,
    ) -> tuple[tuple[np.ndarray, int, int] | None, tuple[ValidationDiagnostic, ...]]:
        try:
            if isinstance(image_input, np.ndarray):
                image = image_input.copy()
            else:
                image = cv2.imread(str(Path(image_input)), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError("unreadable image")
            image = _normalize_array(image)
        except (cv2.error, OSError, TypeError, ValueError):
            return None, (
                self._diagnostic(
                    "evidence.invalid_image",
                    "image",
                    "Evidence image is not a supported readable image.",
                ),
            )
        height, width = image.shape[:2]
        return (image, int(width), int(height)), ()

    def _metadata(
        self,
        request: EvidenceCaptureRequest,
        *,
        captured_at: str,
        screenshot_width: int,
        screenshot_height: int,
        evidence_width: int,
        evidence_height: int,
        content_hash: str,
        crop_box: BoundingBox | None,
    ) -> tuple[EvidenceMetadata | None, tuple[ValidationDiagnostic, ...]]:
        detection = request.detection_result
        scene = request.scene_result
        try:
            explicit_template = _normalize_optional_text(
                request.semantic_template_key,
                "semantic template key",
            )
            explicit_scene = _normalize_optional_text(
                request.semantic_scene_key,
                "semantic scene key",
            )
            metadata = EvidenceMetadata(
                schema_version=EVIDENCE_SCHEMA_VERSION,
                captured_at=captured_at,
                evidence_kind=_normalize_text(request.evidence_kind, "evidence kind"),
                screenshot_width=screenshot_width,
                screenshot_height=screenshot_height,
                evidence_width=evidence_width,
                evidence_height=evidence_height,
                content_hash=content_hash,
                semantic_template_key=(
                    explicit_template
                    or (detection.matched_semantic_key if detection is not None else None)
                ),
                semantic_scene_key=(
                    explicit_scene
                    or (detection.scene if detection is not None else None)
                    or (scene.scene_key if scene is not None else None)
                ),
                detection_confidence=(detection.confidence if detection is not None else None),
                matched_scale=(detection.matched_scale if detection is not None else None),
                bounding_box=(
                    detection.bounding_box
                    if detection is not None and detection.bounding_box is not None
                    else crop_box
                ),
                template_pack_version=(
                    detection.template_pack_version if detection is not None else None
                ),
                classification_status=(scene.status.value if scene is not None else None),
                classification_score=(scene.score if scene is not None else None),
                correlation_id=_normalize_optional_text(request.correlation_id, "correlation identifier"),
            )
        except (TypeError, ValueError, AttributeError):
            return None, (
                self._diagnostic(
                    "evidence.invalid_metadata",
                    "metadata",
                    "Evidence metadata could not be constructed.",
                ),
            )
        return metadata, ()

    def _validate_context(
        self,
        request: EvidenceCaptureRequest,
    ) -> tuple[ValidationDiagnostic, ...]:
        diagnostics: list[ValidationDiagnostic] = []
        if request.detection_result is not None and not isinstance(request.detection_result, DetectionResult):
            diagnostics.append(
                self._diagnostic(
                    "evidence.invalid_context",
                    "detection_result",
                    "Detection context must be a DetectionResult.",
                )
            )
        if request.scene_result is not None and not isinstance(request.scene_result, SceneClassificationResult):
            diagnostics.append(
                self._diagnostic(
                    "evidence.invalid_context",
                    "scene_result",
                    "Scene context must be a SceneClassificationResult.",
                )
            )
        if (
            isinstance(request.detection_result, DetectionResult)
            and request.crop is not None
        ):
            try:
                detection_box = request.detection_result.bounding_box
            except AttributeError:
                detection_box = None
            if detection_box is None:
                return tuple(sorted(diagnostics, key=lambda item: (item.field, item.code, item.message)))
            try:
                crop_box = self._crop_box(request.crop)
            except (TypeError, ValueError):
                crop_box = None
            if crop_box is not None and crop_box != detection_box:
                diagnostics.append(
                    self._diagnostic(
                        "evidence.contradictory_context",
                        "crop",
                        "Evidence crop conflicts with the detection bounding box.",
                    )
                )
        return tuple(sorted(diagnostics, key=lambda item: (item.field, item.code, item.message)))

    def _validate_request_text(
        self,
        request: EvidenceCaptureRequest,
    ) -> tuple[ValidationDiagnostic, ...]:
        try:
            _normalize_text(request.evidence_kind, "evidence kind")
            _normalize_optional_text(request.semantic_template_key, "semantic template key")
            _normalize_optional_text(request.semantic_scene_key, "semantic scene key")
            _normalize_optional_text(request.correlation_id, "correlation identifier")
        except (TypeError, ValueError):
            return (
                self._diagnostic(
                    "evidence.invalid_metadata",
                    "metadata",
                    "Evidence metadata could not be constructed.",
                ),
            )
        return ()

    def _capture_timestamp(
        self,
    ) -> tuple[str | None, tuple[ValidationDiagnostic, ...]]:
        try:
            timestamp = self.clock()
            if not isinstance(timestamp, datetime):
                raise TypeError("clock must return datetime")
            if timestamp.tzinfo is None:
                raise ValueError("clock must return a timezone-aware datetime")
            return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"), ()
        except (TypeError, ValueError, OSError):
            return None, (
                self._diagnostic(
                    "evidence.invalid_timestamp",
                    "clock",
                    "Evidence timestamp could not be generated.",
                ),
            )

    def _name_parts(
        self,
        request: EvidenceCaptureRequest,
        captured_at: str,
    ) -> tuple[str | None, tuple[ValidationDiagnostic, ...]]:
        try:
            raw_identifier = self.identifier_factory()
            if not isinstance(raw_identifier, str):
                raise TypeError("identifier must be a string")
            identifier = _safe_token(raw_identifier)
            evidence_kind = _safe_token(request.evidence_kind)
            timestamp = _safe_token(captured_at)
        except (TypeError, ValueError, OSError):
            return None, (
                self._diagnostic(
                    "evidence.invalid_identifier",
                    "identifier",
                    "Evidence identifier could not be generated.",
                ),
            )
        if not identifier or not evidence_kind or not timestamp:
            return None, (
                self._diagnostic(
                    "evidence.invalid_identifier",
                    "identifier",
                    "Evidence identifier must be non-empty.",
                ),
            )
        return f"{timestamp}-{evidence_kind}-{identifier}", ()

    def _encode_png(
        self,
        image: np.ndarray,
    ) -> tuple[bytes | None, tuple[ValidationDiagnostic, ...]]:
        try:
            ok, encoded = cv2.imencode(".png", image)
            if not ok or encoded is None:
                raise ValueError("PNG encoding failed")
            return bytes(encoded), ()
        except (cv2.error, TypeError, ValueError):
            return None, (
                self._diagnostic(
                    "evidence.encode_failed",
                    "image",
                    "Evidence image could not be encoded as PNG.",
                ),
            )

    def _serialize_metadata(
        self,
        metadata: EvidenceMetadata,
    ) -> tuple[bytes | None, tuple[ValidationDiagnostic, ...]]:
        try:
            return (
                (
                    json.dumps(
                        metadata.to_json_dict(),
                        sort_keys=True,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
                (),
            )
        except (TypeError, ValueError):
            return None, (
                self._diagnostic(
                    "evidence.metadata_serialization_failed",
                    "metadata",
                    "Evidence metadata could not be serialized.",
                ),
            )

    @staticmethod
    def _crop_box(crop: BoundingBox | RegionOfInterest | None) -> BoundingBox | None:
        if crop is None:
            return None
        if isinstance(crop, BoundingBox):
            return crop
        if isinstance(crop, RegionOfInterest):
            return BoundingBox(crop.x, crop.y, crop.width, crop.height)
        raise TypeError("crop must be a bounding box or ROI")

    def _crop(
        self,
        image: np.ndarray,
        crop: BoundingBox,
    ) -> tuple[np.ndarray | None, tuple[ValidationDiagnostic, ...]]:
        height, width = image.shape[:2]
        if crop.x + crop.width > width or crop.y + crop.height > height:
            return None, (
                self._diagnostic(
                    "evidence.invalid_crop",
                    "crop",
                    "Evidence crop must be contained by the screenshot dimensions.",
                ),
            )
        return image[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width].copy(), ()

    def _unique_paths(self, directory: Path, stem: str) -> tuple[Path, Path]:
        for index in range(_MAX_COLLISION_ATTEMPTS):
            suffix = "" if index == 0 else f"-{index:03d}"
            image_path = directory / f"{stem}{suffix}.png"
            metadata_path = directory / f"{stem}{suffix}.json"
            if not image_path.exists() and not metadata_path.exists():
                return image_path, metadata_path
        raise _EvidenceWriteError(
            (
                self._diagnostic(
                    "evidence.collision_exhausted",
                    "evidence_store",
                    "Evidence destination could not be allocated without collision.",
                ),
            )
        )

    def _write_atomically(
        self,
        image_path: Path,
        metadata_path: Path,
        image_bytes: bytes,
        metadata_bytes: bytes,
    ) -> None:
        image_tmp = image_path.with_name(f".{image_path.name}.tmp")
        metadata_tmp = metadata_path.with_name(f".{metadata_path.name}.tmp")
        committed_image = False
        committed_metadata = False
        image_tmp_created = False
        metadata_tmp_created = False
        diagnostics: list[ValidationDiagnostic] = []
        try:
            if image_path.exists() or metadata_path.exists():
                raise _EvidenceWriteError(
                    (
                        self._diagnostic(
                            "evidence.collision",
                            "evidence_store",
                            "Evidence destination already exists.",
                        ),
                    )
                )
            try:
                self._write_temp(image_tmp, image_bytes)
                image_tmp_created = True
                self._write_temp(metadata_tmp, metadata_bytes)
                metadata_tmp_created = True
            except FileExistsError as exc:
                raise _EvidenceWriteError(
                    (
                        self._diagnostic(
                            "evidence.temp_collision",
                            "evidence_store",
                            "Evidence temporary file already exists.",
                        ),
                    )
                ) from exc
            except (OSError, PermissionError, TypeError, ValueError) as exc:
                raise _EvidenceWriteError(
                    (
                        self._diagnostic(
                            "evidence.write_failed",
                            "evidence_store",
                            "Evidence temporary files could not be written.",
                        ),
                    )
                ) from exc
            if image_path.exists() or metadata_path.exists():
                raise _EvidenceWriteError(
                    (
                        self._diagnostic(
                            "evidence.collision",
                            "evidence_store",
                            "Evidence destination already exists.",
                        ),
                    )
                )
            try:
                self._publish(image_tmp, image_path)
            except (OSError, PermissionError, TypeError, ValueError) as exc:
                raise _EvidenceWriteError(
                    (
                        self._diagnostic(
                            "evidence.publish_failed",
                            "evidence_store",
                            "Evidence image file could not be published.",
                        ),
                    )
                ) from exc
            committed_image = True
            try:
                self._publish(metadata_tmp, metadata_path)
            except (OSError, PermissionError, TypeError, ValueError) as exc:
                diagnostics.append(
                    self._diagnostic(
                        "evidence.publish_failed",
                        "evidence_store",
                        "Evidence metadata file could not be published.",
                    )
                )
                if not _unlink_if_exists(image_path):
                    diagnostics.append(
                        self._diagnostic(
                            "evidence.rollback_failed",
                            "evidence_store",
                            "Evidence image rollback failed after metadata publish failure.",
                        )
                    )
                raise _EvidenceWriteError(tuple(diagnostics)) from exc
            committed_metadata = True
        except _EvidenceWriteError:
            if committed_image and not committed_metadata and image_path.exists():
                _unlink_if_exists(image_path)
            if committed_metadata and not committed_image and metadata_path.exists():
                _unlink_if_exists(metadata_path)
            raise
        finally:
            cleanup_failed = False
            if image_tmp_created and image_tmp.exists() and not _unlink_if_exists(image_tmp):
                cleanup_failed = True
            if metadata_tmp_created and metadata_tmp.exists() and not _unlink_if_exists(metadata_tmp):
                cleanup_failed = True
            if cleanup_failed and not diagnostics:
                diagnostics.append(
                    self._diagnostic(
                        "evidence.cleanup_failed",
                        "evidence_store",
                        "Evidence temporary files could not be cleaned up.",
                    )
                )
        if diagnostics:
            raise _EvidenceWriteError(tuple(diagnostics))

    @staticmethod
    def _write_temp(path: Path, content: bytes) -> None:
        with path.open("xb") as handle:
            handle.write(content)

    @staticmethod
    def _publish(source: Path, destination: Path) -> None:
        try:
            with destination.open("xb") as handle:
                handle.write(source.read_bytes())
            source.unlink()
        except FileExistsError:
            raise
        except (OSError, PermissionError, TypeError, ValueError):
            _unlink_if_exists(destination)
            raise

    def _resolved_root(self) -> Path:
        if self.root.exists() and self.root.is_symlink():
            raise ValueError("evidence root must not be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root.resolve()

    @staticmethod
    def _diagnostic(code: str, field: str, message: str) -> ValidationDiagnostic:
        return ValidationDiagnostic(code=code, field=field, message=message)

    @staticmethod
    def _failure(diagnostics: tuple[ValidationDiagnostic, ...]) -> EvidenceCaptureResult:
        return EvidenceCaptureResult(
            diagnostics=tuple(sorted(diagnostics, key=lambda item: (item.field, item.code, item.message)))
        )


def _normalize_array(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty array")
    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 pixels")
    if image.ndim == 2:
        normalized = image.copy()
    elif image.ndim == 3 and image.shape[2] in (1, 3, 4):
        normalized = image.copy()
    else:
        raise ValueError("unsupported image shape")
    if normalized.shape[0] <= 0 or normalized.shape[1] <= 0:
        raise ValueError("image dimensions must be positive")
    return np.ascontiguousarray(normalized.copy())


def _relative_path(value: str) -> Path | None:
    if not isinstance(value, str):
        return None
    normalized_value = value.strip()
    if not normalized_value:
        return None
    if normalized_value != value:
        return None
    raw_parts = re.split(r"[\\/]", normalized_value)
    if any(part == "" for part in raw_parts):
        return None
    windows_path = PureWindowsPath(normalized_value)
    posix_path = PurePosixPath(normalized_value.replace("\\", "/"))
    if windows_path.is_absolute() or windows_path.drive or posix_path.is_absolute():
        return None
    if any(part in ("", ".", "..") or _is_unsafe_windows_part(part) for part in posix_path.parts):
        return None
    return Path(*posix_path.parts)


def _require_relative_reference(value: str, field_name: str) -> None:
    if _relative_path(value) is None:
        raise ValueError(f"{field_name} must be a relative path.")


def _relative_posix(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not _is_relative_to(resolved, root):
        raise ValueError("path is not relative to root")
    return resolved.relative_to(root).as_posix()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_token(value: str) -> str:
    token = _TOKEN_PATTERN.sub("-", value.strip()).strip(".-_")
    if not token:
        raise ValueError("token must be non-empty")
    return token[:96]


def _normalize_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _normalize_optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _require_positive_int(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


def _require_score(value: Any, field_name: str) -> float:
    score = _require_finite_number(value, field_name)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    return score


def _require_finite_number(value: Any, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be a finite number.")
    return numeric


def _is_unsafe_windows_part(part: str) -> bool:
    if part.rstrip(" .") != part:
        return True
    name = part.split(".", 1)[0].upper()
    return name in _WINDOWS_RESERVED_NAMES


def _unlink_if_exists(path: Path) -> bool:
    try:
        path.unlink(missing_ok=True)
        return True
    except OSError:
        return False
