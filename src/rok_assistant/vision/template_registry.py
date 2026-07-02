from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .template_models import (
    RegionOfInterest,
    ResolutionProfile,
    ScaleRange,
    SceneConstraints,
    TemplateDefinition,
    TemplatePack,
    ValidationDiagnostic,
    ValidationReport,
)

MANIFEST_FILE_NAME = "template-pack.json"
MANIFEST_SCHEMA_VERSION = 1
SEMANTIC_KEY_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")


class TemplateRegistryError(Exception):
    def __init__(self, message: str, diagnostics: Sequence[ValidationDiagnostic]):
        super().__init__(message)
        self.diagnostics = tuple(diagnostics)


class TemplatePackValidationError(TemplateRegistryError):
    pass


class TemplateNotFoundError(TemplateRegistryError):
    def __init__(self, semantic_key: str):
        diagnostic = ValidationDiagnostic(
            code="template.missing_key",
            field="semantic_key",
            message=f"Template semantic key is not registered: {semantic_key}",
        )
        super().__init__(diagnostic.message, (diagnostic,))
        self.semantic_key = semantic_key


class TemplateRegistry:
    def __init__(self, template_pack: TemplatePack):
        self._pack = template_pack
        self._templates = {
            template.semantic_key: template
            for template in sorted(
                template_pack.templates,
                key=lambda definition: definition.semantic_key,
            )
        }

    @classmethod
    def from_pack_root(
        cls,
        pack_root: str | Path,
        manifest_name: str = MANIFEST_FILE_NAME,
    ) -> TemplateRegistry:
        report = validate_template_pack(pack_root, manifest_name)
        if not report.is_valid:
            raise TemplatePackValidationError("Template pack validation failed.", report.diagnostics)
        manifest = _load_manifest(Path(pack_root), manifest_name)
        return cls(_parse_template_pack(_resolved_root(Path(pack_root)), manifest))

    @property
    def template_pack(self) -> TemplatePack:
        return self._pack

    def keys(self) -> tuple[str, ...]:
        return tuple(self._templates.keys())

    def templates(self) -> tuple[TemplateDefinition, ...]:
        return tuple(self._templates.values())

    def get(self, semantic_key: str) -> TemplateDefinition:
        try:
            return self._templates[semantic_key]
        except KeyError as exc:
            raise TemplateNotFoundError(semantic_key) from exc

    def resolve_template_path(self, definition: TemplateDefinition) -> Path:
        return _resolve_relative(self._pack.root, definition.source)

    def resolve_mask_path(self, definition: TemplateDefinition) -> Path | None:
        if definition.mask is None:
            return None
        return _resolve_relative(self._pack.root, definition.mask)


def validate_template_pack(
    pack_root: str | Path,
    manifest_name: str = MANIFEST_FILE_NAME,
) -> ValidationReport:
    root = Path(pack_root)
    diagnostics: list[ValidationDiagnostic] = []
    try:
        manifest = _load_manifest(root, manifest_name)
    except OSError:
        return ValidationReport(
            (
                ValidationDiagnostic(
                    code="manifest.unreadable",
                    field=manifest_name,
                    message="Template pack manifest cannot be read.",
                ),
            )
        )
    except json.JSONDecodeError:
        return ValidationReport(
            (
                ValidationDiagnostic(
                    code="manifest.invalid_json",
                    field=manifest_name,
                    message="Template pack manifest is not valid JSON.",
                ),
            )
        )

    if not isinstance(manifest, Mapping):
        return ValidationReport(
            (
                ValidationDiagnostic(
                    code="manifest.invalid_type",
                    field=manifest_name,
                    message="Template pack manifest must be a JSON object.",
                ),
            )
        )

    _validate_manifest_version(manifest.get("manifest_version", MANIFEST_SCHEMA_VERSION), diagnostics)
    version = _required_text(manifest, "version", diagnostics)
    languages = _validate_languages(manifest.get("languages"), diagnostics)
    profiles = _validate_resolution_profiles(manifest.get("resolution_profiles"), diagnostics)
    templates = manifest.get("templates")
    seen_keys: set[str] = set()

    if not isinstance(templates, list) or not templates:
        diagnostics.append(
            ValidationDiagnostic(
                code="templates.invalid_type",
                field="templates",
                message="templates must be a non-empty list.",
            )
        )
        return ValidationReport(tuple(diagnostics))

    for index, raw_template in enumerate(templates):
        field = f"templates[{index}]"
        if not isinstance(raw_template, Mapping):
            diagnostics.append(
                ValidationDiagnostic(
                    code="template.invalid_type",
                    field=field,
                    message="Template definition must be an object.",
                )
            )
            continue
        semantic_key = _required_text(raw_template, "key", diagnostics, field)
        if semantic_key:
            if not SEMANTIC_KEY_PATTERN.fullmatch(semantic_key):
                diagnostics.append(
                    ValidationDiagnostic(
                        code="template.invalid_key",
                        field=f"{field}.key",
                        message="Template semantic key must be lowercase dot, dash, or underscore separated text.",
                    )
                )
            if semantic_key in seen_keys:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="template.duplicate_key",
                        field=f"{field}.key",
                        message=f"Duplicate template semantic key: {semantic_key}",
                    )
                )
            seen_keys.add(semantic_key)

        language = _required_text(raw_template, "language", diagnostics, field)
        if language and language not in languages:
            diagnostics.append(
                ValidationDiagnostic(
                    code="template.unsupported_language",
                    field=f"{field}.language",
                    message=f"Unsupported template language reference: {language}",
                )
            )

        profile_key = _required_text(raw_template, "resolution_profile", diagnostics, field)
        profile = profiles.get(profile_key) if profile_key else None
        if profile_key and profile is None:
            diagnostics.append(
                ValidationDiagnostic(
                    code="template.unsupported_resolution_profile",
                    field=f"{field}.resolution_profile",
                    message=f"Unsupported resolution profile reference: {profile_key}",
                )
            )

        _validate_relative_file_reference(
            root,
            raw_template,
            "source",
            diagnostics,
            field,
            missing_code="template.missing_file",
        )
        _validate_relative_file_reference(
            root,
            raw_template,
            "mask",
            diagnostics,
            field,
            required=False,
            missing_code="template.invalid_mask",
        )
        _validate_threshold(raw_template.get("threshold"), diagnostics, field)
        _validate_roi(raw_template.get("roi"), profile, diagnostics, field)
        _validate_scale_range(raw_template.get("scale_range"), diagnostics, field)
        _validate_scene_constraints(raw_template.get("scene_constraints"), diagnostics, field)

    if not version:
        diagnostics.append(
            ValidationDiagnostic(
                code="pack.missing_version",
                field="version",
                message="Template pack version is required.",
            )
        )

    return ValidationReport(tuple(diagnostics))


def _load_manifest(root: Path, manifest_name: str) -> Any:
    return json.loads((root / manifest_name).read_text(encoding="utf-8"))


def _parse_template_pack(root: Path, manifest: Mapping[str, Any]) -> TemplatePack:
    version = str(manifest["version"]).strip()
    languages = tuple(str(language).strip() for language in manifest["languages"])
    resolution_profiles = tuple(
        ResolutionProfile(
            key=str(key),
            width=int(raw_profile["width"]),
            height=int(raw_profile["height"]),
        )
        for key, raw_profile in sorted(manifest["resolution_profiles"].items())
    )
    templates = tuple(
        _parse_template_definition(version, raw_template)
        for raw_template in sorted(manifest["templates"], key=lambda item: item["key"])
    )
    return TemplatePack(
        version=version,
        languages=languages,
        resolution_profiles=resolution_profiles,
        templates=templates,
        root=root,
    )


def _parse_template_definition(
    version: str,
    raw_template: Mapping[str, Any],
) -> TemplateDefinition:
    scale_range = raw_template.get("scale_range") or {}
    constraints = raw_template.get("scene_constraints") or {}
    mask = raw_template.get("mask")
    return TemplateDefinition(
        semantic_key=str(raw_template["key"]).strip(),
        template_pack_version=version,
        language=str(raw_template["language"]).strip(),
        resolution_profile=str(raw_template["resolution_profile"]).strip(),
        source=Path(str(raw_template["source"]).strip()),
        region_of_interest=_parse_roi(raw_template["roi"]),
        confidence_threshold=float(raw_template["threshold"]),
        scale_range=ScaleRange(
            minimum=float(scale_range.get("min", 1.0)),
            maximum=float(scale_range.get("max", 1.0)),
        ),
        mask=Path(str(mask).strip()) if mask else None,
        scene_constraints=SceneConstraints(
            allowed=tuple(str(scene).strip() for scene in constraints.get("allowed", ())),
            required=tuple(str(scene).strip() for scene in constraints.get("required", ())),
        ),
        source_reference=str(raw_template.get("source_reference", "")).strip(),
    )


def _parse_roi(raw_roi: Mapping[str, Any]) -> RegionOfInterest:
    return RegionOfInterest(
        x=int(raw_roi["x"]),
        y=int(raw_roi["y"]),
        width=int(raw_roi["width"]),
        height=int(raw_roi["height"]),
    )


def _validate_languages(
    raw_languages: Any,
    diagnostics: list[ValidationDiagnostic],
) -> set[str]:
    if not isinstance(raw_languages, list) or not raw_languages:
        diagnostics.append(
            ValidationDiagnostic(
                code="pack.invalid_languages",
                field="languages",
                message="languages must be a non-empty list of language codes.",
            )
        )
        return set()
    languages: set[str] = set()
    for index, language in enumerate(raw_languages):
        if not isinstance(language, str) or not language.strip():
            diagnostics.append(
                ValidationDiagnostic(
                    code="pack.invalid_language",
                    field=f"languages[{index}]",
                    message="Language entries must be non-empty strings.",
                )
            )
            continue
        languages.add(language.strip())
    return languages


def _validate_manifest_version(
    raw_version: Any,
    diagnostics: list[ValidationDiagnostic],
) -> None:
    if raw_version != MANIFEST_SCHEMA_VERSION:
        diagnostics.append(
            ValidationDiagnostic(
                code="manifest.unsupported_version",
                field="manifest_version",
                message=f"Template pack manifest_version must be {MANIFEST_SCHEMA_VERSION}.",
            )
        )


def _validate_resolution_profiles(
    raw_profiles: Any,
    diagnostics: list[ValidationDiagnostic],
) -> dict[str, ResolutionProfile]:
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        diagnostics.append(
            ValidationDiagnostic(
                code="pack.invalid_resolution_profiles",
                field="resolution_profiles",
                message="resolution_profiles must be a non-empty object.",
            )
        )
        return {}
    profiles: dict[str, ResolutionProfile] = {}
    for key, raw_profile in raw_profiles.items():
        field = f"resolution_profiles.{key}"
        if not isinstance(key, str) or not key.strip():
            diagnostics.append(
                ValidationDiagnostic(
                    code="profile.invalid_key",
                    field=field,
                    message="Resolution profile keys must be non-empty strings.",
                )
            )
            continue
        if not isinstance(raw_profile, Mapping):
            diagnostics.append(
                ValidationDiagnostic(
                    code="profile.invalid_type",
                    field=field,
                    message="Resolution profile must be an object.",
                )
            )
            continue
        width = _positive_int(raw_profile.get("width"))
        height = _positive_int(raw_profile.get("height"))
        if width is None or height is None:
            diagnostics.append(
                ValidationDiagnostic(
                    code="profile.invalid_dimensions",
                    field=field,
                    message="Resolution profile width and height must be positive integers.",
                )
            )
            continue
        profiles[key] = ResolutionProfile(key=key, width=width, height=height)
    return profiles


def _validate_relative_file_reference(
    root: Path,
    raw_template: Mapping[str, Any],
    name: str,
    diagnostics: list[ValidationDiagnostic],
    template_field: str,
    *,
    required: bool = True,
    missing_code: str,
) -> None:
    field = f"{template_field}.{name}"
    value = raw_template.get(name)
    if value is None or value == "":
        if required:
            diagnostics.append(
                ValidationDiagnostic(
                    code=f"template.missing_{name}",
                    field=field,
                    message=f"Template {name} reference is required.",
                )
            )
        return
    if not isinstance(value, str):
        diagnostics.append(
            ValidationDiagnostic(
                code=f"template.invalid_{name}",
                field=field,
                message=f"Template {name} reference must be a string.",
            )
        )
        return
    path = _relative_pack_path(value)
    if path is None:
        diagnostics.append(
            ValidationDiagnostic(
                code=f"template.invalid_{name}",
                field=field,
                message=f"Template {name} reference must be a relative path inside the template pack.",
            )
        )
        return
    resolved = _resolve_relative(root, path)
    if not resolved.is_file():
        diagnostics.append(
            ValidationDiagnostic(
                code=missing_code,
                field=field,
                message=f"Template {name} file does not exist: {value}",
            )
        )
        return
    if not _is_relative_to(resolved, _resolved_root(root)):
        diagnostics.append(
            ValidationDiagnostic(
                code=f"template.invalid_{name}",
                field=field,
                message=f"Template {name} reference must resolve inside the template pack.",
            )
        )


def _validate_threshold(
    raw_threshold: Any,
    diagnostics: list[ValidationDiagnostic],
    template_field: str,
) -> None:
    field = f"{template_field}.threshold"
    if not _is_finite_number(raw_threshold):
        diagnostics.append(
            ValidationDiagnostic(
                code="template.invalid_threshold",
                field=field,
                message="Template threshold must be a finite number between 0.0 and 1.0.",
            )
        )
        return
    if float(raw_threshold) < 0.0 or float(raw_threshold) > 1.0:
        diagnostics.append(
            ValidationDiagnostic(
                code="template.invalid_threshold",
                field=field,
                message="Template threshold must be between 0.0 and 1.0.",
            )
        )


def _validate_roi(
    raw_roi: Any,
    profile: ResolutionProfile | None,
    diagnostics: list[ValidationDiagnostic],
    template_field: str,
) -> None:
    field = f"{template_field}.roi"
    if not isinstance(raw_roi, Mapping):
        diagnostics.append(
            ValidationDiagnostic(
                code="template.invalid_roi",
                field=field,
                message="Template roi must be an object.",
            )
        )
        return
    x = _non_negative_int(raw_roi.get("x"))
    y = _non_negative_int(raw_roi.get("y"))
    width = _positive_int(raw_roi.get("width"))
    height = _positive_int(raw_roi.get("height"))
    if x is None or y is None or width is None or height is None:
        diagnostics.append(
            ValidationDiagnostic(
                code="template.invalid_roi",
                field=field,
                message="ROI x/y must be non-negative integers and width/height must be positive integers.",
            )
        )
        return
    if profile is not None and (x + width > profile.width or y + height > profile.height):
        diagnostics.append(
            ValidationDiagnostic(
                code="template.invalid_roi",
                field=field,
                message="ROI must be contained by its resolution profile.",
            )
        )


def _validate_scale_range(
    raw_scale_range: Any,
    diagnostics: list[ValidationDiagnostic],
    template_field: str,
) -> None:
    field = f"{template_field}.scale_range"
    if raw_scale_range is None:
        return
    if not isinstance(raw_scale_range, Mapping):
        diagnostics.append(
            ValidationDiagnostic(
                code="template.invalid_scale_range",
                field=field,
                message="scale_range must be an object.",
            )
        )
        return
    minimum = raw_scale_range.get("min", 1.0)
    maximum = raw_scale_range.get("max", 1.0)
    if not _is_finite_number(minimum) or not _is_finite_number(maximum):
        diagnostics.append(
            ValidationDiagnostic(
                code="template.invalid_scale_range",
                field=field,
                message="scale_range min and max must be finite numbers.",
            )
        )
        return
    if float(minimum) <= 0.0 or float(maximum) <= 0.0 or float(minimum) > float(maximum):
        diagnostics.append(
            ValidationDiagnostic(
                code="template.invalid_scale_range",
                field=field,
                message="scale_range min and max must be positive and min must not exceed max.",
            )
        )


def _validate_scene_constraints(
    raw_constraints: Any,
    diagnostics: list[ValidationDiagnostic],
    template_field: str,
) -> None:
    field = f"{template_field}.scene_constraints"
    if raw_constraints is None:
        return
    if not isinstance(raw_constraints, Mapping):
        diagnostics.append(
            ValidationDiagnostic(
                code="template.invalid_scene_constraints",
                field=field,
                message="scene_constraints must be an object.",
            )
        )
        return
    unknown_keys = set(raw_constraints) - {"allowed", "required"}
    if unknown_keys:
        diagnostics.append(
            ValidationDiagnostic(
                code="template.invalid_scene_constraints",
                field=field,
                message="scene_constraints may only contain allowed and required keys.",
            )
        )
    for key in ("allowed", "required"):
        value = raw_constraints.get(key, [])
        if not isinstance(value, list) or any(
            not isinstance(scene, str) or not scene.strip() for scene in value
        ):
            diagnostics.append(
                ValidationDiagnostic(
                    code="template.invalid_scene_constraints",
                    field=f"{field}.{key}",
                    message="Scene constraints must be lists of non-empty scene names.",
                )
            )
    allowed = raw_constraints.get("allowed", [])
    required = raw_constraints.get("required", [])
    if isinstance(allowed, list) and isinstance(required, list):
        allowed_scenes = {scene.strip() for scene in allowed if isinstance(scene, str)}
        required_scenes = {scene.strip() for scene in required if isinstance(scene, str)}
        if allowed_scenes and not required_scenes.issubset(allowed_scenes):
            diagnostics.append(
                ValidationDiagnostic(
                    code="template.contradictory_scene_constraints",
                    field=field,
                    message="Required scenes must be included in allowed scenes.",
                )
            )


def _required_text(
    mapping: Mapping[str, Any],
    key: str,
    diagnostics: list[ValidationDiagnostic],
    parent_field: str = "",
) -> str:
    field = f"{parent_field}.{key}" if parent_field else key
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        diagnostics.append(
            ValidationDiagnostic(
                code=f"missing_{key}",
                field=field,
                message=f"{field} is required and must be a non-empty string.",
            )
        )
        return ""
    return value.strip()


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _non_negative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _relative_pack_path(value: str) -> Path | None:
    normalized_value = value.strip()
    if not normalized_value:
        return None
    windows_path = PureWindowsPath(normalized_value)
    posix_path = PurePosixPath(normalized_value.replace("\\", "/"))
    if windows_path.is_absolute() or windows_path.drive or posix_path.is_absolute():
        return None
    if any(part in ("", ".", "..") for part in posix_path.parts):
        return None
    return Path(*posix_path.parts)


def _resolve_relative(root: Path, path: Path) -> Path:
    return (_resolved_root(root) / path).resolve()


def _resolved_root(root: Path) -> Path:
    return root.resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
