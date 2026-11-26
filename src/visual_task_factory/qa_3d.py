"""Isolated 3D Ground-Truth QA schemas and independent validator (Phase 4)."""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any, Literal, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from visual_task_factory.models.scene3d import SceneEntity3D, SceneSpec3D
from visual_task_factory.solver_3d import (
    GroundTruthSolver3D,
    UnsolvableTask3DError,
    UnsupportedTask3DError,
    VerticalRelation,
    VisibilityStatus,
)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_COORD_LEAK_PATTERN = re.compile(
    r"\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\)"
)
_AT_LEAST_NUMBER_PATTERN = re.compile(
    r"(?:at\s+least|minimum\s+of|greater\s+than\s+or\s+equal\s+to)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:m|meters?)\b",
    re.IGNORECASE,
)


class QuestionFamily3D(str, Enum):
    """Supported 3D visual spatial-reasoning question families."""

    OBJECT_OVERLAP_3D = "object_overlap_3d"
    VERTICAL_RELATION_3D = "vertical_relation_3d"
    VERTICAL_CLEARANCE_3D = "vertical_clearance_3d"
    CAMERA_VISIBILITY_3D = "camera_visibility_3d"


class AnswerType3D(str, Enum):
    """Canonical answer formats for 3D visual-reasoning tasks."""

    BOOLEAN = "boolean"
    NUMERIC = "numeric"


ALLOWED_ANSWER_TYPES_3D_BY_FAMILY: dict[QuestionFamily3D, tuple[AnswerType3D, ...]] = {
    QuestionFamily3D.OBJECT_OVERLAP_3D: (AnswerType3D.BOOLEAN,),
    QuestionFamily3D.VERTICAL_RELATION_3D: (AnswerType3D.BOOLEAN,),
    QuestionFamily3D.VERTICAL_CLEARANCE_3D: (AnswerType3D.NUMERIC, AnswerType3D.BOOLEAN),
    QuestionFamily3D.CAMERA_VISIBILITY_3D: (AnswerType3D.BOOLEAN,),
}


class OverlapQuery3D(BaseModel):
    """Structured query verifying 3D bounding-box contact/overlap vs disjointness."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["object_overlap_3d"] = "object_overlap_3d"
    source_id: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    mode: Literal["any_contact_or_overlap", "interior_overlap", "disjoint"] = (
        "any_contact_or_overlap"
    )

    @model_validator(mode="after")
    def _validate_distinct_ids(self) -> OverlapQuery3D:
        if self.source_id == self.target_id:
            raise ValueError("source_id and target_id must refer to distinct objects.")
        return self


class VerticalRelationQuery3D(BaseModel):
    """Structured query verifying 3D Z-axis ordering between two objects."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["vertical_relation_3d"] = "vertical_relation_3d"
    source_id: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    predicate: Literal["above", "below", "touching_above", "equal_height"]

    @model_validator(mode="after")
    def _validate_distinct_ids(self) -> VerticalRelationQuery3D:
        if self.source_id == self.target_id:
            raise ValueError("source_id and target_id must refer to distinct objects.")
        return self


class VerticalClearanceQuery3D(BaseModel):
    """Structured query verifying defined vertical clearance between objects or to ceiling."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["vertical_clearance_3d"] = "vertical_clearance_3d"
    source_id: str = Field(..., min_length=1)
    target_id: str | None = None
    target_kind: Literal["entity", "ceiling"] = "entity"
    answer_type: AnswerType3D = AnswerType3D.NUMERIC
    min_required_meters: float | None = Field(default=None, ge=0.0)
    numeric_tolerance_meters: float = Field(default=0.05, ge=0.0)

    @field_validator("min_required_meters", "numeric_tolerance_meters")
    @classmethod
    def _require_finite_values(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Clearance distances and tolerances must be finite.")
        return value

    @model_validator(mode="after")
    def _validate_clearance_fields(self) -> VerticalClearanceQuery3D:
        if self.target_kind == "entity":
            if not self.target_id:
                raise ValueError("target_id is required when target_kind='entity'.")
            if self.source_id == self.target_id:
                raise ValueError("source_id and target_id must refer to distinct objects.")
        else:
            if self.target_id is not None:
                raise ValueError("target_id must be None when target_kind='ceiling'.")

        if self.answer_type == AnswerType3D.BOOLEAN and self.min_required_meters is None:
            raise ValueError("min_required_meters is required when answer_type='boolean'.")
        if self.answer_type == AnswerType3D.NUMERIC and self.min_required_meters is not None:
            raise ValueError("min_required_meters must be None when answer_type='numeric'.")
        return self


class CameraVisibilityQuery3D(BaseModel):
    """Structured query verifying unambiguous camera line-of-sight visibility."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["camera_visibility_3d"] = "camera_visibility_3d"
    target_id: str = Field(..., min_length=1)
    require_clear: bool = True


SolverQuery3D = Union[
    OverlapQuery3D,
    VerticalRelationQuery3D,
    VerticalClearanceQuery3D,
    CameraVisibilityQuery3D,
]


class ReferenceAnswer3D(BaseModel):
    """Ground-truth answer payload for a 3D visual reasoning question."""

    model_config = ConfigDict(extra="forbid")

    answer_type: AnswerType3D
    value: bool | float
    unit: Literal["m"] | None = None
    numeric_tolerance: float | None = Field(default=None, ge=0.0)
    solver_name: str = Field(default="ground_truth_solver_3d", min_length=1)
    reasoning_trace: list[str] = Field(default_factory=list)

    @field_validator("numeric_tolerance")
    @classmethod
    def _require_finite_tolerance(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("numeric_tolerance must be finite.")
        return value

    @model_validator(mode="after")
    def _validate_value_matches_type(self) -> ReferenceAnswer3D:
        if self.answer_type == AnswerType3D.BOOLEAN:
            if type(self.value) is not bool:
                raise ValueError("BOOLEAN reference answer requires strict bool (True or False).")
            if self.unit is not None or self.numeric_tolerance is not None:
                raise ValueError("BOOLEAN answers must not specify unit or numeric_tolerance.")
        elif self.answer_type == AnswerType3D.NUMERIC:
            if type(self.value) not in (int, float):
                raise ValueError("NUMERIC reference answer requires int or float (not bool).")
            num = float(self.value)
            if not math.isfinite(num) or num < 0.0:
                raise ValueError("NUMERIC reference answer must be finite and >= 0.0.")
            if self.unit != "m":
                raise ValueError("NUMERIC vertical clearance answers must specify unit='m'.")
            self.value = num
        return self


class VisualQuestion3D(BaseModel):
    """Normalized 3D visual-reasoning question paired with its reference answer."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    question_id: str = Field(..., min_length=1)
    scene_id: str = Field(..., min_length=1)
    family: QuestionFamily3D
    template_id: str = Field(..., min_length=1)
    prompt_text: str = Field(..., min_length=10)
    expected_answer_type: AnswerType3D
    reference_answer: ReferenceAnswer3D
    solver_query: dict[str, Any] | None = None
    requires_visual_reasoning: bool = True

    @field_validator("question_id", "scene_id", "template_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or not _ID_PATTERN.match(cleaned):
            raise ValueError(f"Identifier {value!r} must match {_ID_PATTERN.pattern}.")
        return cleaned

    @field_validator("prompt_text")
    @classmethod
    def _clean_prompt(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 10:
            raise ValueError("prompt_text must have at least 10 non-whitespace characters.")
        return cleaned

    @model_validator(mode="after")
    def _validate_consistency(self) -> VisualQuestion3D:
        if not self.requires_visual_reasoning:
            raise ValueError("3D benchmark questions must require visual reasoning.")

        allowed = ALLOWED_ANSWER_TYPES_3D_BY_FAMILY[self.family]
        if self.expected_answer_type not in allowed:
            raise ValueError(
                f"Question family {self.family.value!r} requires expected_answer_type in "
                f"{[a.value for a in allowed]}, got {self.expected_answer_type.value!r}."
            )

        if self.reference_answer.answer_type != self.expected_answer_type:
            raise ValueError(
                f"reference_answer.answer_type ({self.reference_answer.answer_type.value!r}) "
                f"must match expected_answer_type ({self.expected_answer_type.value!r})."
            )

        if self.solver_query is not None:
            if not self.solver_query:
                raise ValueError("solver_query, when provided, must be a non-empty dictionary.")
            query_family = self.solver_query.get("family")
            if query_family is not None and query_family != self.family.value:
                raise ValueError(
                    f"solver_query.family ({query_family!r}) must match question.family ({self.family.value!r})."
                )

        return self


class BenchmarkItem3D(BaseModel):
    """Joint record binding SceneSpec3D and VisualQuestion3D."""

    model_config = ConfigDict(extra="forbid")

    scene: SceneSpec3D
    question: VisualQuestion3D

    @model_validator(mode="after")
    def _validate_scene_link(self) -> BenchmarkItem3D:
        if self.question.scene_id != self.scene.scene_id:
            raise ValueError(
                f"question.scene_id ({self.question.scene_id!r}) does not match "
                f"scene.scene_id ({self.scene.scene_id!r})."
            )
        return self


class ValidationSeverity3D(str, Enum):
    """Severity levels for 3D QA validation findings."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationFinding3D(BaseModel):
    """Single deterministic validation finding for a 3D QA item."""

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: ValidationSeverity3D
    message: str


class ValidationReport3D(BaseModel):
    """Structured QA validation outcome for a single BenchmarkItem3D."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    errors: list[ValidationFinding3D]
    warnings: list[ValidationFinding3D]
    info: list[ValidationFinding3D]
    item_id: str
    scene_id: str
    question_id: str
    validation_version: str = "1.0"


class BatchValidationReport3D(BaseModel):
    """Aggregated QA validation outcome across multiple BenchmarkItem3D objects."""

    model_config = ConfigDict(extra="forbid")

    total_items: int
    valid_items: int
    invalid_items: int
    warning_count: int
    duplicate_question_ids: list[str]
    duplicate_prompts: list[str]
    reports: list[ValidationReport3D]


def is_camera_isometric_aligned(scene: SceneSpec3D) -> bool:
    """Return True if SceneSpec3D.camera sits inside the +X/+Y/+Z isometric octant."""
    return (
        15.0 <= scene.camera.azimuth_deg <= 75.0
        and 20.0 <= scene.camera.elevation_deg <= 55.0
    )


def evaluate_overlap_query(
    solver: GroundTruthSolver3D,
    scene: SceneSpec3D,
    query: OverlapQuery3D,
) -> ReferenceAnswer3D:
    """Compute canonical ReferenceAnswer3D for an OverlapQuery3D via Phase 3 solver."""
    res = solver.solve_box_overlap(scene, query.source_id, query.target_id)
    if query.mode == "any_contact_or_overlap":
        ans = bool(res.overlaps_3d or res.touches_3d)
    elif query.mode == "interior_overlap":
        ans = bool(res.overlaps_3d)
    else:
        ans = bool(not res.overlaps_3d and not res.touches_3d)

    return ReferenceAnswer3D(
        answer_type=AnswerType3D.BOOLEAN,
        value=ans,
        solver_name="ground_truth_solver_3d.solve_box_overlap",
        reasoning_trace=[
            f"3D relation({query.source_id}, {query.target_id}) = {res.relation.value}",
            f"mode={query.mode} -> {ans}",
        ],
    )


def evaluate_vertical_relation_query(
    solver: GroundTruthSolver3D,
    scene: SceneSpec3D,
    query: VerticalRelationQuery3D,
) -> ReferenceAnswer3D:
    """Compute canonical ReferenceAnswer3D for a VerticalRelationQuery3D."""
    res = solver.solve_vertical_relation(scene, query.source_id, query.target_id)
    if query.predicate == "above":
        ans = bool(res.is_above)
    elif query.predicate == "below":
        ans = bool(res.is_below)
    elif query.predicate == "touching_above":
        ans = bool(res.relation == VerticalRelation.FLUSH_ABOVE and res.xy_overlap)
    else:
        ans = bool(res.relation == VerticalRelation.EQUAL_HEIGHT)

    return ReferenceAnswer3D(
        answer_type=AnswerType3D.BOOLEAN,
        value=ans,
        solver_name="ground_truth_solver_3d.solve_vertical_relation",
        reasoning_trace=[
            f"Vertical relation({query.source_id}, {query.target_id}) = {res.relation.value}",
            f"predicate={query.predicate} -> {ans}",
        ],
    )


def evaluate_vertical_clearance_query(
    solver: GroundTruthSolver3D,
    scene: SceneSpec3D,
    query: VerticalClearanceQuery3D,
) -> ReferenceAnswer3D:
    """Compute canonical ReferenceAnswer3D for a defined VerticalClearanceQuery3D."""
    if query.target_kind == "ceiling":
        res = solver.solve_ceiling_clearance(scene, query.source_id)
    else:
        assert query.target_id is not None
        res = solver.solve_vertical_clearance(scene, query.source_id, query.target_id)

    if not res.is_defined or res.clearance_meters is None:
        raise UnsolvableTask3DError(
            f"Undefined vertical clearance ({res.status.value}): {res.reason}"
        )

    if query.answer_type == AnswerType3D.NUMERIC:
        return ReferenceAnswer3D(
            answer_type=AnswerType3D.NUMERIC,
            value=res.clearance_meters,
            unit="m",
            numeric_tolerance=query.numeric_tolerance_meters,
            solver_name="ground_truth_solver_3d.solve_vertical_clearance",
            reasoning_trace=[
                f"status={res.status.value}, clearance={res.clearance_meters:.6f} m",
            ],
        )

    assert query.min_required_meters is not None
    meets_min = bool(res.clearance_meters + 1e-9 >= query.min_required_meters)
    return ReferenceAnswer3D(
        answer_type=AnswerType3D.BOOLEAN,
        value=meets_min,
        solver_name="ground_truth_solver_3d.solve_vertical_clearance",
        reasoning_trace=[
            f"clearance={res.clearance_meters:.6f} m vs threshold={query.min_required_meters:.6f} m -> {meets_min}",
        ],
    )


def evaluate_camera_visibility_query(
    solver: GroundTruthSolver3D,
    scene: SceneSpec3D,
    query: CameraVisibilityQuery3D,
) -> ReferenceAnswer3D:
    """Compute canonical ReferenceAnswer3D only for unambiguous visibility states."""
    if not is_camera_isometric_aligned(scene):
        raise UnsolvableTask3DError(
            f"Scene camera (az={scene.camera.azimuth_deg}, el={scene.camera.elevation_deg}) "
            "does not align with the fixed-angle isometric renderer."
        )

    res = solver.solve_camera_visibility(scene, query.target_id)
    if not res.within_fov_cone:
        raise UnsolvableTask3DError(
            f"Target {query.target_id!r} lies outside the camera FOV cone."
        )
    if res.status == VisibilityStatus.PARTIALLY_OCCLUDED:
        raise UnsolvableTask3DError(
            f"Target {query.target_id!r} is partially occluded ({res.visible_sample_count}/"
            f"{res.total_sample_count} samples); visibility QA is ambiguous."
        )

    is_clear = res.status == VisibilityStatus.CLEAR
    ans = bool(is_clear if query.require_clear else not is_clear)
    return ReferenceAnswer3D(
        answer_type=AnswerType3D.BOOLEAN,
        value=ans,
        solver_name="ground_truth_solver_3d.solve_camera_visibility",
        reasoning_trace=[
            f"visibility({query.target_id}) = {res.status.value}, blockers={res.blocking_object_ids}",
            f"require_clear={query.require_clear} -> {ans}",
        ],
    )


def _build_report(
    findings: list[ValidationFinding3D],
    *,
    item_id: str,
    scene_id: str,
    question_id: str,
) -> ValidationReport3D:
    errors = [f for f in findings if f.severity == ValidationSeverity3D.ERROR]
    warnings = [f for f in findings if f.severity == ValidationSeverity3D.WARNING]
    info = [f for f in findings if f.severity == ValidationSeverity3D.INFO]
    return ValidationReport3D(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        info=info,
        item_id=item_id,
        scene_id=scene_id,
        question_id=question_id,
    )


class GroundTruthQAValidator3D:
    """Independent validator verifying 3D QA structure, references, and ground truth."""

    def __init__(self, solver: GroundTruthSolver3D | None = None) -> None:
        self.solver = solver or GroundTruthSolver3D()

    def validate(self, item: BenchmarkItem3D) -> ValidationReport3D:
        """Validate a single BenchmarkItem3D without modifying or repairing it."""
        if not isinstance(item, BenchmarkItem3D):
            raise TypeError(f"Expected BenchmarkItem3D instance, got {type(item).__name__}.")

        scene = item.scene
        question = item.question
        findings: list[ValidationFinding3D] = []

        query = self._reconstruct_query(question, findings)
        if query is not None:
            self._check_scene_references(scene, question, query, findings)
            if not any(f.severity == ValidationSeverity3D.ERROR for f in findings):
                self._verify_ground_truth(scene, question, query, findings)

        if _COORD_LEAK_PATTERN.search(question.prompt_text):
            findings.append(
                ValidationFinding3D(
                    code="COORDINATE_LEAKAGE",
                    severity=ValidationSeverity3D.ERROR,
                    message="Prompt text leaks raw 3D coordinates.",
                )
            )

        return _build_report(
            findings,
            item_id=f"{scene.scene_id}::{question.question_id}",
            scene_id=scene.scene_id,
            question_id=question.question_id,
        )

    def validate_batch(self, items: Sequence[BenchmarkItem3D]) -> BatchValidationReport3D:
        """Validate a batch of BenchmarkItem3D records and reject duplicates."""
        seen_qids: dict[str, int] = {}
        seen_prompts: dict[tuple[str, str], int] = {}
        duplicate_qids: list[str] = []
        duplicate_prompts: list[str] = []
        reports: list[ValidationReport3D] = []

        for item in items:
            report = self.validate(item)
            errors = list(report.errors)
            warnings = list(report.warnings)
            info = list(report.info)

            qid = report.question_id
            seen_qids[qid] = seen_qids.get(qid, 0) + 1
            if seen_qids[qid] > 1:
                if qid not in duplicate_qids:
                    duplicate_qids.append(qid)
                errors.append(
                    ValidationFinding3D(
                        code="DUPLICATE_QUESTION_ID",
                        severity=ValidationSeverity3D.ERROR,
                        message=f"Duplicate question_id {qid!r} detected in batch.",
                    )
                )

            norm_prompt = " ".join(item.question.prompt_text.strip().lower().split())
            pkey = (report.scene_id, norm_prompt)
            seen_prompts[pkey] = seen_prompts.get(pkey, 0) + 1
            if seen_prompts[pkey] > 1:
                if item.question.prompt_text not in duplicate_prompts:
                    duplicate_prompts.append(item.question.prompt_text)
                errors.append(
                    ValidationFinding3D(
                        code="DUPLICATE_PROMPT",
                        severity=ValidationSeverity3D.ERROR,
                        message="Duplicate normalized prompt_text detected for the same scene.",
                    )
                )

            reports.append(
                ValidationReport3D(
                    is_valid=len(errors) == 0,
                    errors=errors,
                    warnings=warnings,
                    info=info,
                    item_id=report.item_id,
                    scene_id=report.scene_id,
                    question_id=report.question_id,
                    validation_version=report.validation_version,
                )
            )

        valid_count = sum(1 for r in reports if r.is_valid)
        total_count = len(reports)
        return BatchValidationReport3D(
            total_items=total_count,
            valid_items=valid_count,
            invalid_items=total_count - valid_count,
            warning_count=sum(len(r.warnings) for r in reports),
            duplicate_question_ids=duplicate_qids,
            duplicate_prompts=duplicate_prompts,
            reports=reports,
        )

    def _reconstruct_query(
        self,
        question: VisualQuestion3D,
        findings: list[ValidationFinding3D],
    ) -> SolverQuery3D | None:
        raw = question.solver_query
        if raw is None:
            findings.append(
                ValidationFinding3D(
                    code="MISSING_SOLVER_QUERY",
                    severity=ValidationSeverity3D.ERROR,
                    message="VisualQuestion3D.solver_query is required for ground-truth QA validation.",
                )
            )
            return None

        if not isinstance(raw, dict) or not raw:
            findings.append(
                ValidationFinding3D(
                    code="MALFORMED_SOLVER_QUERY",
                    severity=ValidationSeverity3D.ERROR,
                    message="VisualQuestion3D.solver_query must be a non-empty dictionary.",
                )
            )
            return None

        raw_family = raw.get("family")
        if raw_family != question.family.value:
            findings.append(
                ValidationFinding3D(
                    code="QUERY_FAMILY_MISMATCH",
                    severity=ValidationSeverity3D.ERROR,
                    message=(
                        f"solver_query.family ({raw_family!r}) must match "
                        f"question.family ({question.family.value!r})."
                    ),
                )
            )
            return None

        cls_map: dict[QuestionFamily3D, type[OverlapQuery3D | VerticalRelationQuery3D | VerticalClearanceQuery3D | CameraVisibilityQuery3D]] = {
            QuestionFamily3D.OBJECT_OVERLAP_3D: OverlapQuery3D,
            QuestionFamily3D.VERTICAL_RELATION_3D: VerticalRelationQuery3D,
            QuestionFamily3D.VERTICAL_CLEARANCE_3D: VerticalClearanceQuery3D,
            QuestionFamily3D.CAMERA_VISIBILITY_3D: CameraVisibilityQuery3D,
        }
        query_cls = cls_map.get(question.family)
        if query_cls is None:
            findings.append(
                ValidationFinding3D(
                    code="UNSUPPORTED_QUESTION_TYPE",
                    severity=ValidationSeverity3D.ERROR,
                    message=f"Unsupported 3D question family: {question.family!r}.",
                )
            )
            return None

        try:
            parsed = query_cls.model_validate(raw)
        except ValidationError as exc:
            findings.append(
                ValidationFinding3D(
                    code="MALFORMED_SOLVER_QUERY",
                    severity=ValidationSeverity3D.ERROR,
                    message=f"Failed to parse solver_query as {query_cls.__name__}: {exc}",
                )
            )
            return None

        if isinstance(parsed, VerticalClearanceQuery3D) and parsed.answer_type != question.expected_answer_type:
            findings.append(
                ValidationFinding3D(
                    code="QUERY_ANSWER_TYPE_MISMATCH",
                    severity=ValidationSeverity3D.ERROR,
                    message=(
                        f"VerticalClearanceQuery3D.answer_type ({parsed.answer_type.value!r}) "
                        f"must match expected_answer_type ({question.expected_answer_type.value!r})."
                    ),
                )
            )
            return None

        return parsed

    def _check_scene_references(
        self,
        scene: SceneSpec3D,
        question: VisualQuestion3D,
        query: SolverQuery3D,
        findings: list[ValidationFinding3D],
    ) -> None:
        entities_by_id: dict[str, SceneEntity3D] = {ent.id: ent for ent in scene.entities}
        label_counts: dict[str, int] = {}
        for ent in scene.entities:
            label_counts[ent.label] = label_counts.get(ent.label, 0) + 1

        referenced_ids: list[str] = []
        if isinstance(query, (OverlapQuery3D, VerticalRelationQuery3D)):
            referenced_ids = [query.source_id, query.target_id]
        elif isinstance(query, VerticalClearanceQuery3D):
            referenced_ids = [query.source_id]
            if query.target_id is not None:
                referenced_ids.append(query.target_id)
        elif isinstance(query, CameraVisibilityQuery3D):
            referenced_ids = [query.target_id]

        for ref_id in referenced_ids:
            ent = entities_by_id.get(ref_id)
            if ent is None:
                findings.append(
                    ValidationFinding3D(
                        code="UNKNOWN_ENTITY_ID",
                        severity=ValidationSeverity3D.ERROR,
                        message=f"Referenced 3D entity ID {ref_id!r} does not exist in scene.entities.",
                    )
                )
                continue
            if label_counts.get(ent.label, 0) > 1:
                findings.append(
                    ValidationFinding3D(
                        code="AMBIGUOUS_ENTITY_LABEL",
                        severity=ValidationSeverity3D.ERROR,
                        message=(
                            f"Entity {ref_id!r} shares duplicate label {ent.label!r} in scene "
                            "and cannot be referenced unambiguously."
                        ),
                    )
                )

        if isinstance(query, VerticalClearanceQuery3D) and query.answer_type == AnswerType3D.BOOLEAN:
            match = _AT_LEAST_NUMBER_PATTERN.search(question.prompt_text)
            if match is not None and query.min_required_meters is not None:
                prompt_val = float(match.group(1))
                if not math.isclose(prompt_val, query.min_required_meters, abs_tol=1e-6):
                    findings.append(
                        ValidationFinding3D(
                            code="THRESHOLD_PROMPT_MISMATCH",
                            severity=ValidationSeverity3D.ERROR,
                            message=(
                                f"Prompt clearance threshold ({prompt_val} m) does not match "
                                f"solver_query.min_required_meters ({query.min_required_meters} m)."
                            ),
                        )
                    )

    def _verify_ground_truth(
        self,
        scene: SceneSpec3D,
        question: VisualQuestion3D,
        query: SolverQuery3D,
        findings: list[ValidationFinding3D],
    ) -> None:
        stored = question.reference_answer
        try:
            if isinstance(query, OverlapQuery3D):
                expected = evaluate_overlap_query(self.solver, scene, query)
            elif isinstance(query, VerticalRelationQuery3D):
                expected = evaluate_vertical_relation_query(self.solver, scene, query)
            elif isinstance(query, VerticalClearanceQuery3D):
                expected = evaluate_vertical_clearance_query(self.solver, scene, query)
            elif isinstance(query, CameraVisibilityQuery3D):
                expected = evaluate_camera_visibility_query(self.solver, scene, query)
            else:
                findings.append(
                    ValidationFinding3D(
                        code="UNSUPPORTED_QUESTION_TYPE",
                        severity=ValidationSeverity3D.ERROR,
                        message=f"Unsupported query object: {type(query).__name__}",
                    )
                )
                return
        except (UnsolvableTask3DError, UnsupportedTask3DError) as exc:
            findings.append(
                ValidationFinding3D(
                    code="AMBIGUOUS_OR_UNSUPPORTED_GROUND_TRUTH",
                    severity=ValidationSeverity3D.ERROR,
                    message=str(exc),
                )
            )
            return

        if stored.answer_type != expected.answer_type:
            findings.append(
                ValidationFinding3D(
                    code="ANSWER_TYPE_MISMATCH",
                    severity=ValidationSeverity3D.ERROR,
                    message=(
                        f"Stored answer_type ({stored.answer_type.value!r}) does not match "
                        f"solver expected type ({expected.answer_type.value!r})."
                    ),
                )
            )
            return

        if stored.answer_type == AnswerType3D.BOOLEAN:
            if stored.value != expected.value:
                findings.append(
                    ValidationFinding3D(
                        code="REFERENCE_ANSWER_MISMATCH",
                        severity=ValidationSeverity3D.ERROR,
                        message=(
                            f"Stored boolean answer {stored.value!r} does not match "
                            f"ground truth {expected.value!r}."
                        ),
                    )
                )
        elif stored.answer_type == AnswerType3D.NUMERIC:
            tol = (
                expected.numeric_tolerance
                if expected.numeric_tolerance is not None
                else (stored.numeric_tolerance or 0.05)
            )
            if abs(float(stored.value) - float(expected.value)) > tol + 1e-9:
                findings.append(
                    ValidationFinding3D(
                        code="REFERENCE_ANSWER_MISMATCH",
                        severity=ValidationSeverity3D.ERROR,
                        message=(
                            f"Stored numeric answer {stored.value!r} m differs from "
                            f"ground truth {expected.value!r} m beyond tolerance {tol}."
                        ),
                    )
                )
