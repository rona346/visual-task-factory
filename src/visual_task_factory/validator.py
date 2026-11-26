"""Independent BenchmarkValidator gate for generated BenchmarkItem records."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from visual_task_factory.ground_truth_solver import (
    ClearanceQuery,
    ContainmentMode,
    GroundTruthSolver,
    ReachabilityQuery,
    RegionCountingQuery,
    SolverQuery,
    UnsolvableTaskError,
    UnsupportedTaskError,
)
from visual_task_factory.models import (
    AnswerType,
    BenchmarkItem,
    EntityType,
    QuestionFamily,
    ReferenceAnswer,
    SceneEntity,
    SceneSpec,
    VisualQuestion,
    Zone,
)

_COORD_PATTERN = re.compile(r"\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\)")
_INTERNAL_ID_PATTERN = re.compile(r"\b(?:Z_[A-Z0-9_]+|ENT_\d+)\b")
_SOLVER_NAME_LEAKS = ("grid_dijkstra", "rect_containment", "euclidean_clearance")
_AT_LEAST_NUMBER_PATTERN = re.compile(r"\bat\s+least\s+(\d+(?:\.\d+)?)\s*meters?\b", re.IGNORECASE)

_CONTAINMENT_KEYWORDS: dict[ContainmentMode, tuple[str, ...]] = {
    ContainmentMode.STRICT_INSIDE: ("completely inside", "strictly inside", "fully inside"),
    ContainmentMode.CENTER_INSIDE: ("centered inside", "center lies inside", "center is inside"),
    ContainmentMode.ANY_OVERLAP: ("overlapping any part of", "overlapping", "overlap"),
}

_ENTITY_KEYWORDS: dict[EntityType, tuple[str, ...]] = {
    EntityType.PALLET_RACK: ("pallet rack", "pallet racks"),
    EntityType.FORKLIFT: ("forklift", "forklifts"),
    EntityType.CRATE: ("crate", "crates"),
    EntityType.PILLAR: ("pillar", "pillars"),
    EntityType.CONVEYOR: ("conveyor", "conveyors"),
    EntityType.ROBOT: ("robot", "robots", "amr"),
}


class ValidationSeverity(str, Enum):
    """Severity levels for validation findings."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationFinding(BaseModel):
    """Single deterministic finding produced by BenchmarkValidator."""

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: ValidationSeverity
    message: str


class ValidationReport(BaseModel):
    """Structured validation outcome for a single BenchmarkItem."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    errors: list[ValidationFinding]
    warnings: list[ValidationFinding]
    info: list[ValidationFinding]
    item_id: str
    scene_id: str
    question_id: str
    validation_version: str = "1.0"


class BatchValidationReport(BaseModel):
    """Aggregated validation outcome across a batch of BenchmarkItems."""

    model_config = ConfigDict(extra="forbid")

    total_items: int
    valid_items: int
    invalid_items: int
    warning_count: int
    duplicate_question_ids: list[str]
    duplicate_prompts: list[str]
    reports: list[ValidationReport]


def _is_entity_visibly_labeled(entity: SceneEntity) -> bool:
    if entity.entity_type == EntityType.PILLAR:
        return False
    if entity.entity_type == EntityType.FORKLIFT:
        return max(entity.bounds.width, entity.bounds.height) >= 3.0
    return min(entity.bounds.width, entity.bounds.height) >= 1.4


def _build_report(
    *,
    findings: list[ValidationFinding],
    item_id: str,
    scene_id: str,
    question_id: str,
    validation_version: str,
) -> ValidationReport:
    errors = [f for f in findings if f.severity == ValidationSeverity.ERROR]
    warnings = [f for f in findings if f.severity == ValidationSeverity.WARNING]
    info = [f for f in findings if f.severity == ValidationSeverity.INFO]
    return ValidationReport(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        info=info,
        item_id=item_id,
        scene_id=scene_id,
        question_id=question_id,
        validation_version=validation_version,
    )


class BenchmarkValidator:
    """Independent validation gate for BenchmarkItem structural, semantic, and solver integrity."""

    def __init__(
        self,
        solver: GroundTruthSolver | None = None,
        validation_version: str = "1.0",
    ) -> None:
        self.solver = solver or GroundTruthSolver()
        self.validation_version = validation_version

    def validate(self, item: BenchmarkItem) -> ValidationReport:
        """Validate a single BenchmarkItem independently without mutating inputs."""
        findings: list[ValidationFinding] = []

        if not isinstance(item, BenchmarkItem):
            findings.append(
                ValidationFinding(
                    code="INVALID_ITEM_TYPE",
                    severity=ValidationSeverity.ERROR,
                    message=f"Expected BenchmarkItem instance, got {type(item).__name__}.",
                )
            )
            return _build_report(
                findings=findings,
                item_id="unknown::unknown",
                scene_id="unknown",
                question_id="unknown",
                validation_version=self.validation_version,
            )

        raw_scene_id = getattr(getattr(item, "scene", None), "scene_id", "unknown") or "unknown"
        raw_question_id = (
            getattr(getattr(item, "question", None), "question_id", "unknown") or "unknown"
        )
        item_id = f"{raw_scene_id}::{raw_question_id}"

        # 1. Structural & Pydantic re-validation (catches model_construct or tampered data).
        try:
            validated_item = BenchmarkItem.model_validate(item.model_dump(mode="python"))
        except ValidationError as exc:
            findings.append(
                ValidationFinding(
                    code="SCHEMA_VALIDATION_ERROR",
                    severity=ValidationSeverity.ERROR,
                    message=f"Pydantic schema re-validation failed: {exc}",
                )
            )
            return _build_report(
                findings=findings,
                item_id=item_id,
                scene_id=str(raw_scene_id),
                question_id=str(raw_question_id),
                validation_version=self.validation_version,
            )
        except Exception as exc:  # Defensive catch if sub-objects were malformed
            findings.append(
                ValidationFinding(
                    code="SCHEMA_VALIDATION_ERROR",
                    severity=ValidationSeverity.ERROR,
                    message=f"Failed to serialize/validate BenchmarkItem: {exc}",
                )
            )
            return _build_report(
                findings=findings,
                item_id=item_id,
                scene_id=str(raw_scene_id),
                question_id=str(raw_question_id),
                validation_version=self.validation_version,
            )

        scene = validated_item.scene
        question = validated_item.question

        # 2. Check supported family and strict solver_query presence/reconstruction.
        parsed_query = self._reconstruct_solver_query(question, findings)

        # 3. Run leakage checks independent of solver reconstruction.
        self._check_prompt_leakage(question, findings)

        if parsed_query is not None:
            # 4. Question-to-scene consistency & visual identifiability.
            self._check_scene_consistency(scene, question, parsed_query, findings)

            # 5. Independent solver recomputation & reference answer comparison.
            self._verify_with_solver(scene, question, parsed_query, findings)

        return _build_report(
            findings=findings,
            item_id=f"{scene.scene_id}::{question.question_id}",
            scene_id=scene.scene_id,
            question_id=question.question_id,
            validation_version=self.validation_version,
        )

    def validate_batch(self, items: Sequence[BenchmarkItem]) -> BatchValidationReport:
        """Validate a sequence of BenchmarkItems and check batch-level duplicates."""
        seen_qids: dict[str, int] = {}
        seen_prompts: dict[tuple[str, str], int] = {}
        duplicate_qids: list[str] = []
        duplicate_prompts: list[str] = []
        reports: list[ValidationReport] = []

        for item in items:
            report = self.validate(item)
            errors = list(report.errors)
            warnings = list(report.warnings)
            info = list(report.info)

            qid = report.question_id
            if qid != "unknown":
                count = seen_qids.get(qid, 0) + 1
                seen_qids[qid] = count
                if count > 1:
                    if qid not in duplicate_qids:
                        duplicate_qids.append(qid)
                    errors.append(
                        ValidationFinding(
                            code="DUPLICATE_QUESTION_ID",
                            severity=ValidationSeverity.ERROR,
                            message=f"Duplicate question_id {qid!r} detected in batch.",
                        )
                    )

            prompt_text = getattr(getattr(item, "question", None), "prompt_text", None)
            if isinstance(prompt_text, str) and prompt_text.strip():
                norm_prompt = " ".join(prompt_text.strip().lower().split())
                prompt_key = (report.scene_id, norm_prompt)
                p_count = seen_prompts.get(prompt_key, 0) + 1
                seen_prompts[prompt_key] = p_count
                if p_count > 1:
                    cleaned_prompt = prompt_text.strip()
                    if cleaned_prompt not in duplicate_prompts:
                        duplicate_prompts.append(cleaned_prompt)
                    warnings.append(
                        ValidationFinding(
                            code="DUPLICATE_PROMPT",
                            severity=ValidationSeverity.WARNING,
                            message="Duplicate normalized prompt_text detected for the same scene.",
                        )
                    )

            reports.append(
                ValidationReport(
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

        valid_items = sum(1 for r in reports if r.is_valid)
        total_items = len(reports)
        warning_count = sum(len(r.warnings) for r in reports)
        return BatchValidationReport(
            total_items=total_items,
            valid_items=valid_items,
            invalid_items=total_items - valid_items,
            warning_count=warning_count,
            duplicate_question_ids=duplicate_qids,
            duplicate_prompts=duplicate_prompts,
            reports=reports,
        )

    def _reconstruct_solver_query(
        self,
        question: VisualQuestion,
        findings: list[ValidationFinding],
    ) -> SolverQuery | None:
        raw_query: dict[str, Any] | None = question.solver_query
        if raw_query is None:
            findings.append(
                ValidationFinding(
                    code="MISSING_SOLVER_QUERY",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        "VisualQuestion.solver_query is required for independent solver verification."
                    ),
                )
            )
            return None

        if not isinstance(raw_query, dict) or not raw_query:
            findings.append(
                ValidationFinding(
                    code="MALFORMED_SOLVER_QUERY",
                    severity=ValidationSeverity.ERROR,
                    message="VisualQuestion.solver_query must be a non-empty dictionary.",
                )
            )
            return None

        raw_family = raw_query.get("family")
        if raw_family != question.family.value:
            findings.append(
                ValidationFinding(
                    code="QUERY_FAMILY_MISMATCH",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"solver_query.family ({raw_family!r}) must explicitly match "
                        f"question.family ({question.family.value!r})."
                    ),
                )
            )
            return None

        query_cls_map: dict[QuestionFamily, type[ReachabilityQuery | RegionCountingQuery | ClearanceQuery]] = {
            QuestionFamily.CONSTRAINED_REACHABILITY: ReachabilityQuery,
            QuestionFamily.REGION_CONTAINMENT_COUNTING: RegionCountingQuery,
            QuestionFamily.SCALE_GROUNDED_CLEARANCE: ClearanceQuery,
        }
        query_cls = query_cls_map.get(question.family)
        if query_cls is None:
            findings.append(
                ValidationFinding(
                    code="UNSUPPORTED_QUESTION_FAMILY",
                    severity=ValidationSeverity.ERROR,
                    message=f"Unsupported question family: {question.family!r}.",
                )
            )
            return None

        try:
            parsed = query_cls.model_validate(raw_query)
        except ValidationError as exc:
            findings.append(
                ValidationFinding(
                    code="MALFORMED_SOLVER_QUERY",
                    severity=ValidationSeverity.ERROR,
                    message=f"Failed to parse solver_query as {query_cls.__name__}: {exc}",
                )
            )
            return None

        if isinstance(parsed, ClearanceQuery) and parsed.answer_type != question.expected_answer_type:
            findings.append(
                ValidationFinding(
                    code="QUERY_ANSWER_TYPE_MISMATCH",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"ClearanceQuery.answer_type ({parsed.answer_type.value!r}) does not match "
                        f"question.expected_answer_type ({question.expected_answer_type.value!r})."
                    ),
                )
            )
            return None

        return parsed

    def _check_scene_consistency(
        self,
        scene: SceneSpec,
        question: VisualQuestion,
        query: SolverQuery,
        findings: list[ValidationFinding],
    ) -> None:
        zones_by_id: dict[str, Zone] = {z.id: z for z in scene.zones}
        entities_by_id: dict[str, SceneEntity] = {e.id: e for e in scene.entities}
        exits_by_id = {e.id: e for e in scene.exits}
        doors_by_id = {d.id: d for d in scene.doors}
        walls_by_id = {w.id: w for w in scene.walls}

        zone_label_counts: dict[str, int] = {}
        for z in scene.zones:
            zone_label_counts[z.label] = zone_label_counts.get(z.label, 0) + 1

        entity_label_counts: dict[str, int] = {}
        entity_type_counts: dict[EntityType, int] = {}
        for ent in scene.entities:
            entity_label_counts[ent.label] = entity_label_counts.get(ent.label, 0) + 1
            entity_type_counts[ent.entity_type] = entity_type_counts.get(ent.entity_type, 0) + 1

        prompt_lower = question.prompt_text.lower()

        if isinstance(query, ReachabilityQuery):
            if query.start_zone_id is not None:
                zone = zones_by_id.get(query.start_zone_id)
                if zone is None:
                    findings.append(
                        ValidationFinding(
                            code="UNKNOWN_START_ZONE",
                            severity=ValidationSeverity.ERROR,
                            message=f"start_zone_id {query.start_zone_id!r} not found in scene.zones.",
                        )
                    )
                else:
                    if zone.zone_type in query.forbidden_zone_types:
                        findings.append(
                            ValidationFinding(
                                code="FORBIDDEN_START_ZONE",
                                severity=ValidationSeverity.ERROR,
                                message=(
                                    f"start_zone_id {zone.id!r} has forbidden type {zone.zone_type.value!r}."
                                ),
                            )
                        )
                    if zone_label_counts.get(zone.label, 0) > 1:
                        findings.append(
                            ValidationFinding(
                                code="AMBIGUOUS_ZONE_LABEL",
                                severity=ValidationSeverity.ERROR,
                                message=f"Zone label {zone.label!r} is duplicated in scene.",
                            )
                        )
                    if zone.label.lower() not in prompt_lower:
                        findings.append(
                            ValidationFinding(
                                code="PROMPT_ZONE_MISMATCH",
                                severity=ValidationSeverity.ERROR,
                                message=(
                                    f"Prompt does not reference start zone label {zone.label!r}."
                                ),
                            )
                        )

            if query.candidate_exit_ids is not None:
                missing_exits = [eid for eid in query.candidate_exit_ids if eid not in exits_by_id]
                if missing_exits:
                    findings.append(
                        ValidationFinding(
                            code="UNKNOWN_CANDIDATE_EXIT",
                            severity=ValidationSeverity.ERROR,
                            message=f"candidate_exit_ids contain unknown exit IDs: {missing_exits}.",
                        )
                    )
                if question.candidate_choices is not None:
                    if set(question.candidate_choices) != set(query.candidate_exit_ids):
                        findings.append(
                            ValidationFinding(
                                code="CANDIDATE_CHOICES_MISMATCH",
                                severity=ValidationSeverity.ERROR,
                                message=(
                                    "question.candidate_choices does not match solver_query.candidate_exit_ids."
                                ),
                            )
                        )

        elif isinstance(query, RegionCountingQuery):
            zone = zones_by_id.get(query.zone_id)
            if zone is None:
                findings.append(
                    ValidationFinding(
                        code="UNKNOWN_TARGET_ZONE",
                        severity=ValidationSeverity.ERROR,
                        message=f"zone_id {query.zone_id!r} not found in scene.zones.",
                    )
                )
            else:
                if zone_label_counts.get(zone.label, 0) > 1:
                    findings.append(
                        ValidationFinding(
                            code="AMBIGUOUS_ZONE_LABEL",
                            severity=ValidationSeverity.ERROR,
                            message=f"Zone label {zone.label!r} is duplicated in scene.",
                        )
                    )
                if zone.label.lower() not in prompt_lower:
                    findings.append(
                        ValidationFinding(
                            code="PROMPT_ZONE_MISMATCH",
                            severity=ValidationSeverity.ERROR,
                            message=f"Prompt does not reference target zone label {zone.label!r}.",
                        )
                    )

            expected_phrases = _CONTAINMENT_KEYWORDS[query.containment_mode]
            if not any(phrase in prompt_lower for phrase in expected_phrases):
                findings.append(
                    ValidationFinding(
                        code="CONTAINMENT_WORDING_MISMATCH",
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Prompt wording does not match containment_mode={query.containment_mode.value!r}."
                        ),
                    )
                )

            if query.entity_types is not None:
                for etype in query.entity_types:
                    kws = _ENTITY_KEYWORDS[etype]
                    if not any(kw in prompt_lower for kw in kws):
                        findings.append(
                            ValidationFinding(
                                code="ENTITY_TYPE_WORDING_MISMATCH",
                                severity=ValidationSeverity.ERROR,
                                message=(
                                    f"Prompt does not mention entity_type={etype.value!r} specified in solver_query."
                                ),
                            )
                        )

        elif isinstance(query, ClearanceQuery):
            for elem_id in (query.source_id, query.target_id):
                if elem_id in entities_by_id:
                    ent = entities_by_id[elem_id]
                    if entity_label_counts.get(ent.label, 0) > 1:
                        findings.append(
                            ValidationFinding(
                                code="AMBIGUOUS_ELEMENT_LABEL",
                                severity=ValidationSeverity.ERROR,
                                message=f"Entity label {ent.label!r} is duplicated in scene.",
                            )
                        )
                    elif not _is_entity_visibly_labeled(ent):
                        if entity_type_counts.get(ent.entity_type, 0) > 1:
                            findings.append(
                                ValidationFinding(
                                    code="NON_IDENTIFIABLE_ELEMENT",
                                    severity=ValidationSeverity.ERROR,
                                    message=(
                                        f"Entity {ent.id!r} ({ent.label!r}) is too small to render a label "
                                        f"and multiple {ent.entity_type.value} items exist."
                                    ),
                                )
                            )
                        else:
                            findings.append(
                                ValidationFinding(
                                    code="UNLABELED_ELEMENT_IN_RENDER",
                                    severity=ValidationSeverity.WARNING,
                                    message=(
                                        f"Entity {ent.id!r} ({ent.label!r}) omits inline text label in standard rendering."
                                    ),
                                )
                            )
                    if ent.label.lower() not in prompt_lower and ent.id.lower() not in prompt_lower:
                        findings.append(
                            ValidationFinding(
                                code="PROMPT_TARGET_MISMATCH",
                                severity=ValidationSeverity.ERROR,
                                message=f"Prompt does not reference clearance target {ent.label!r}.",
                            )
                        )
                elif elem_id in zones_by_id:
                    zone = zones_by_id[elem_id]
                    if zone_label_counts.get(zone.label, 0) > 1:
                        findings.append(
                            ValidationFinding(
                                code="AMBIGUOUS_ZONE_LABEL",
                                severity=ValidationSeverity.ERROR,
                                message=f"Zone label {zone.label!r} is duplicated in scene.",
                            )
                        )
                    if zone.label.lower() not in prompt_lower:
                        findings.append(
                            ValidationFinding(
                                code="PROMPT_TARGET_MISMATCH",
                                severity=ValidationSeverity.ERROR,
                                message=f"Prompt does not reference clearance zone {zone.label!r}.",
                            )
                        )
                elif elem_id in exits_by_id:
                    if elem_id.lower() not in prompt_lower:
                        findings.append(
                            ValidationFinding(
                                code="PROMPT_TARGET_MISMATCH",
                                severity=ValidationSeverity.ERROR,
                                message=f"Prompt does not reference exit {elem_id!r}.",
                            )
                        )
                elif elem_id in doors_by_id or elem_id in walls_by_id:
                    findings.append(
                        ValidationFinding(
                            code="UNLABELED_ELEMENT_IN_RENDER",
                            severity=ValidationSeverity.WARNING,
                            message=(
                                f"Element {elem_id!r} has no inline label unless audit_overlay=True."
                            ),
                        )
                    )
                else:
                    findings.append(
                        ValidationFinding(
                            code="UNKNOWN_ELEMENT_ID",
                            severity=ValidationSeverity.ERROR,
                            message=f"Clearance element ID {elem_id!r} not found in scene.",
                        )
                    )

            if query.answer_type == AnswerType.BOOLEAN and query.min_required_meters is not None:
                match = _AT_LEAST_NUMBER_PATTERN.search(question.prompt_text)
                if match is not None:
                    prompt_threshold = float(match.group(1))
                    if not math.isclose(prompt_threshold, query.min_required_meters, abs_tol=1e-6):
                        findings.append(
                            ValidationFinding(
                                code="THRESHOLD_PROMPT_MISMATCH",
                                severity=ValidationSeverity.ERROR,
                                message=(
                                    f"Prompt threshold ({prompt_threshold} m) does not match "
                                    f"solver_query.min_required_meters ({query.min_required_meters} m)."
                                ),
                            )
                        )

    def _verify_with_solver(
        self,
        scene: SceneSpec,
        question: VisualQuestion,
        query: SolverQuery,
        findings: list[ValidationFinding],
    ) -> None:
        stored = question.reference_answer

        try:
            if isinstance(query, ReachabilityQuery):
                reach_res = self.solver.solve_reachability(scene, query)
                if not reach_res.is_reachable or reach_res.reference_answer is None:
                    findings.append(
                        ValidationFinding(
                            code="UNSOLVABLE_TASK",
                            severity=ValidationSeverity.ERROR,
                            message="Recomputed reachability found no reachable candidate exit.",
                        )
                    )
                    return
                expected = reach_res.reference_answer
                sorted_dists = sorted(reach_res.reachable_exit_distances_m.values())
                if len(sorted_dists) >= 2 and (sorted_dists[1] - sorted_dists[0]) < 0.25:
                    findings.append(
                        ValidationFinding(
                            code="TIGHT_REACHABILITY_MARGIN",
                            severity=ValidationSeverity.WARNING,
                            message=(
                                f"Top two exits differ by only {sorted_dists[1] - sorted_dists[0]:.3f} m."
                            ),
                        )
                    )

            elif isinstance(query, RegionCountingQuery):
                count_res = self.solver.solve_region_counting(scene, query)
                expected = count_res.reference_answer
                if count_res.count == 0:
                    findings.append(
                        ValidationFinding(
                            code="ZERO_COUNT_TASK",
                            severity=ValidationSeverity.INFO,
                            message=f"Zero matching entities found in zone {query.zone_id!r}.",
                        )
                    )

            elif isinstance(query, ClearanceQuery):
                clear_res = self.solver.solve_clearance(scene, query)
                expected = clear_res.reference_answer
                if query.answer_type == AnswerType.BOOLEAN and query.min_required_meters is not None:
                    margin = abs(clear_res.distance_meters - query.min_required_meters)
                    if margin < 0.25:
                        findings.append(
                            ValidationFinding(
                                code="TIGHT_CLEARANCE_MARGIN",
                                severity=ValidationSeverity.WARNING,
                                message=(
                                    f"Clearance {clear_res.distance_meters:.3f} m is within "
                                    f"{margin:.3f} m of threshold {query.min_required_meters:.3f} m."
                                ),
                            )
                        )
            else:
                findings.append(
                    ValidationFinding(
                        code="UNSUPPORTED_SOLVER_QUERY",
                        severity=ValidationSeverity.ERROR,
                        message=f"Unsupported solver query type: {type(query).__name__}.",
                    )
                )
                return

        except UnsolvableTaskError as exc:
            findings.append(
                ValidationFinding(
                    code="UNSOLVABLE_TASK",
                    severity=ValidationSeverity.ERROR,
                    message=f"GroundTruthSolver could not solve query: {exc}",
                )
            )
            return
        except UnsupportedTaskError as exc:
            findings.append(
                ValidationFinding(
                    code="UNSUPPORTED_TASK",
                    severity=ValidationSeverity.ERROR,
                    message=f"GroundTruthSolver rejected unsupported query: {exc}",
                )
            )
            return

        self._compare_reference_answers(stored, expected, findings)

    def _compare_reference_answers(
        self,
        stored: ReferenceAnswer,
        expected: ReferenceAnswer,
        findings: list[ValidationFinding],
    ) -> None:
        if stored.answer_type != expected.answer_type:
            findings.append(
                ValidationFinding(
                    code="ANSWER_TYPE_MISMATCH",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"Stored answer_type {stored.answer_type.value!r} != "
                        f"solver answer_type {expected.answer_type.value!r}."
                    ),
                )
            )
            return

        if stored.unit != expected.unit:
            findings.append(
                ValidationFinding(
                    code="ANSWER_UNIT_MISMATCH",
                    severity=ValidationSeverity.ERROR,
                    message=f"Stored unit {stored.unit!r} != solver unit {expected.unit!r}.",
                )
            )
            return

        if expected.answer_type == AnswerType.NUMERIC:
            if type(stored.value) not in (int, float):
                findings.append(
                    ValidationFinding(
                        code="REFERENCE_ANSWER_MISMATCH",
                        severity=ValidationSeverity.ERROR,
                        message=f"Stored numeric answer has invalid type {type(stored.value).__name__}.",
                    )
                )
                return

            stored_val = float(stored.value)
            expected_val = float(expected.value)
            tol = stored.numeric_tolerance if stored.numeric_tolerance is not None else (
                expected.numeric_tolerance if expected.numeric_tolerance is not None else 0.0
            )
            diff = abs(stored_val - expected_val)
            if diff > tol + 1e-9:
                findings.append(
                    ValidationFinding(
                        code="NUMERIC_ANSWER_MISMATCH",
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Stored numeric answer {stored_val} m differs from solver "
                            f"{expected_val} m by {diff:.6f} m (> tolerance {tol} m)."
                        ),
                    )
                )
                return

            if diff > 1e-6:
                findings.append(
                    ValidationFinding(
                        code="NUMERIC_WITHIN_TOLERANCE_DRIFT",
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Stored numeric answer {stored_val} m differs slightly from solver "
                            f"{expected_val} m by {diff:.6f} m (within tolerance {tol} m)."
                        ),
                    )
                )
        else:
            if type(stored.value) is not type(expected.value) or stored.value != expected.value:
                findings.append(
                    ValidationFinding(
                        code="REFERENCE_ANSWER_MISMATCH",
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Stored answer {stored.value!r} != solver answer {expected.value!r}."
                        ),
                    )
                )
                return

        findings.append(
            ValidationFinding(
                code="SOLVER_VERIFIED",
                severity=ValidationSeverity.INFO,
                message=f"Independent {expected.solver_name} verification matched.",
            )
        )

    def _check_prompt_leakage(
        self,
        question: VisualQuestion,
        findings: list[ValidationFinding],
    ) -> None:
        prompt = question.prompt_text

        if _COORD_PATTERN.search(prompt):
            findings.append(
                ValidationFinding(
                    code="COORDINATE_LEAKAGE",
                    severity=ValidationSeverity.ERROR,
                    message="Question prompt appears to contain raw (x, y) coordinates.",
                )
            )

        prompt_lower = prompt.lower()
        if any(leak in prompt_lower for leak in _SOLVER_NAME_LEAKS) or _INTERNAL_ID_PATTERN.search(prompt):
            findings.append(
                ValidationFinding(
                    code="INTERNAL_METADATA_LEAKAGE",
                    severity=ValidationSeverity.ERROR,
                    message="Question prompt exposes internal solver names or internal element IDs.",
                )
            )

        if question.expected_answer_type == AnswerType.EXIT_ID:
            ans_id = str(question.reference_answer.value)
            if re.search(rf"\b{re.escape(ans_id)}\b", prompt):
                findings.append(
                    ValidationFinding(
                        code="ANSWER_LEAKAGE",
                        severity=ValidationSeverity.ERROR,
                        message=f"Prompt directly contains reference exit ID {ans_id!r}.",
                    )
                )

        if question.expected_answer_type == AnswerType.NUMERIC:
            ans_str = f"{float(question.reference_answer.value):g}"
            if re.search(rf"(?<!\d\.){re.escape(ans_str)}(?!\.\d|\d)", prompt):
                findings.append(
                    ValidationFinding(
                        code="ANSWER_LEAKAGE",
                        severity=ValidationSeverity.ERROR,
                        message=f"Prompt directly contains numeric reference value {ans_str!r}.",
                    )
                )


def validate_benchmark_item(
    item: BenchmarkItem,
    *,
    solver: GroundTruthSolver | None = None,
) -> ValidationReport:
    """Convenience entry point to independently validate a single BenchmarkItem."""
    return BenchmarkValidator(solver=solver).validate(item)


def validate_benchmark_batch(
    items: Sequence[BenchmarkItem],
    *,
    solver: GroundTruthSolver | None = None,
) -> BatchValidationReport:
    """Convenience entry point to validate a batch of BenchmarkItems."""
    return BenchmarkValidator(solver=solver).validate_batch(items)
