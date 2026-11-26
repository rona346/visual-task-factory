"""Deterministic isolated 3D Question Generator backed by GroundTruthSolver3D."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Literal

from visual_task_factory.models.scene import EntityType
from visual_task_factory.models.scene3d import SceneEntity3D, SceneSpec3D
from visual_task_factory.qa_3d import (
    AnswerType3D,
    BenchmarkItem3D,
    CameraVisibilityQuery3D,
    OverlapQuery3D,
    QuestionFamily3D,
    VerticalClearanceQuery3D,
    VerticalRelationQuery3D,
    VisualQuestion3D,
    evaluate_camera_visibility_query,
    evaluate_overlap_query,
    evaluate_vertical_clearance_query,
    evaluate_vertical_relation_query,
    is_camera_isometric_aligned,
)
from visual_task_factory.solver_3d import (
    GroundTruthSolver3D,
    UnsolvableTask3DError,
    UnsupportedTask3DError,
    VisibilityStatus,
)

_DEFAULT_FAMILIES_3D: tuple[QuestionFamily3D, ...] = (
    QuestionFamily3D.OBJECT_OVERLAP_3D,
    QuestionFamily3D.VERTICAL_RELATION_3D,
    QuestionFamily3D.VERTICAL_CLEARANCE_3D,
    QuestionFamily3D.CAMERA_VISIBILITY_3D,
)


@dataclass(frozen=True)
class QuestionGenerator3DConfig:
    """Configuration for deterministic 3D question generation."""

    families: tuple[QuestionFamily3D, ...] = _DEFAULT_FAMILIES_3D
    max_questions_per_family: int = 2
    clearance_variant: Literal["numeric", "boolean", "both"] = "both"
    numeric_tolerance_m: float = 0.05
    include_ceiling_clearance: bool = True
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if not self.families:
            raise ValueError("families must contain at least one QuestionFamily3D.")
        for fam in self.families:
            if not isinstance(fam, QuestionFamily3D):
                raise TypeError(f"Unsupported question family type: {fam!r}.")
        if self.max_questions_per_family < 1 or self.max_questions_per_family > 10:
            raise ValueError("max_questions_per_family must be between 1 and 10.")
        if self.clearance_variant not in ("numeric", "boolean", "both"):
            raise ValueError("clearance_variant must be 'numeric', 'boolean', or 'both'.")
        if self.numeric_tolerance_m < 0.0:
            raise ValueError("numeric_tolerance_m must be >= 0.0.")


def _stable_rng_3d(scene: SceneSpec3D, seed_offset: int) -> random.Random:
    digest = hashlib.sha256(
        f"{scene.scene_id}:{scene.variation.seed}:{seed_offset}:3d".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _eligible_labeled_entities(scene: SceneSpec3D) -> list[SceneEntity3D]:
    """Return entities with unique, render-referencable labels (excluding pillars)."""
    label_counts: dict[str, int] = {}
    for ent in scene.entities:
        label_counts[ent.label] = label_counts.get(ent.label, 0) + 1

    return [
        ent
        for ent in scene.entities
        if ent.entity_type != EntityType.PILLAR and label_counts.get(ent.label, 0) == 1
    ]


def _round_half(value: float) -> float:
    return round(value * 2.0) / 2.0


class QuestionGenerator3D:
    """Generates deterministic, deduplicated BenchmarkItem3D questions."""

    def __init__(
        self,
        config: QuestionGenerator3DConfig | None = None,
        solver: GroundTruthSolver3D | None = None,
    ) -> None:
        self.config = config or QuestionGenerator3DConfig()
        self.solver = solver or GroundTruthSolver3D()

    def generate(self, scene: SceneSpec3D) -> list[BenchmarkItem3D]:
        """Generate grounded 3D questions for supported families on ``scene``."""
        if not isinstance(scene, SceneSpec3D):
            raise TypeError(f"Expected SceneSpec3D instance, got {type(scene).__name__}.")

        entities = _eligible_labeled_entities(scene)
        if not entities:
            return []

        # Exclude completely hidden entities from non-visibility questions so items
        # remain answerable from the rendered isometric image.
        if is_camera_isometric_aligned(scene):
            visible_entities = [
                ent
                for ent in entities
                if self.solver.solve_camera_visibility(scene, ent.id).status
                != VisibilityStatus.FULLY_OCCLUDED
            ]
        else:
            visible_entities = entities

        rng = _stable_rng_3d(scene, self.config.seed_offset)
        seen_prompts: set[str] = set()
        items: list[BenchmarkItem3D] = []

        for family in self.config.families:
            if family == QuestionFamily3D.OBJECT_OVERLAP_3D:
                generated = self._generate_overlap_items(scene, visible_entities, rng)
            elif family == QuestionFamily3D.VERTICAL_RELATION_3D:
                generated = self._generate_vertical_relation_items(scene, visible_entities, rng)
            elif family == QuestionFamily3D.VERTICAL_CLEARANCE_3D:
                generated = self._generate_vertical_clearance_items(scene, visible_entities, rng)
            elif family == QuestionFamily3D.CAMERA_VISIBILITY_3D:
                generated = self._generate_visibility_items(scene, entities, rng)
            else:
                raise UnsupportedTask3DError(f"Unsupported 3D question family: {family!r}.")

            for item in generated:
                norm_prompt = " ".join(item.question.prompt_text.strip().lower().split())
                if norm_prompt in seen_prompts:
                    continue
                seen_prompts.add(norm_prompt)
                items.append(item)

        return items

    def _ordered_pairs(
        self,
        entities: list[SceneEntity3D],
        rng: random.Random,
    ) -> list[tuple[SceneEntity3D, SceneEntity3D]]:
        pairs: list[tuple[SceneEntity3D, SceneEntity3D]] = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                pairs.append((entities[i], entities[j]))
        rng.shuffle(pairs)
        return pairs

    def _generate_overlap_items(
        self,
        scene: SceneSpec3D,
        entities: list[SceneEntity3D],
        rng: random.Random,
    ) -> list[BenchmarkItem3D]:
        if len(entities) < 2:
            return []

        candidates: list[tuple[SceneEntity3D, SceneEntity3D, Literal["any_contact_or_overlap", "disjoint"], bool]] = []
        for src, tgt in self._ordered_pairs(entities, rng):
            res = self.solver.solve_box_overlap(scene, src.id, tgt.id)
            mode: Literal["any_contact_or_overlap", "disjoint"] = (
                "any_contact_or_overlap"
                if (len(candidates) % 2 == 0 or res.overlaps_3d or res.touches_3d)
                else "disjoint"
            )
            has_contact_or_overlap = bool(res.overlaps_3d or res.touches_3d)
            candidates.append((src, tgt, mode, has_contact_or_overlap))

        # Prioritize balanced coverage: touching/overlapping cases alongside disjoint cases.
        positive = [c for c in candidates if c[3]]
        negative = [c for c in candidates if not c[3]]
        balanced: list[tuple[SceneEntity3D, SceneEntity3D, Literal["any_contact_or_overlap", "disjoint"]]] = []
        while positive or negative:
            if positive:
                s, t, m, _ = positive.pop(0)
                balanced.append((s, t, m))
            if negative:
                s, t, m, _ = negative.pop(0)
                balanced.append((s, t, m))

        items: list[BenchmarkItem3D] = []
        for src, tgt, mode in balanced:
            if len(items) >= self.config.max_questions_per_family:
                break

            query = OverlapQuery3D(source_id=src.id, target_id=tgt.id, mode=mode)
            ref = evaluate_overlap_query(self.solver, scene, query)

            if mode == "any_contact_or_overlap":
                prompt_text = (
                    f"Do {src.label} and {tgt.label} touch or overlap in 3D space?"
                )
            else:
                prompt_text = (
                    f"Are {src.label} and {tgt.label} completely separated (disjoint) in 3D space?"
                )

            idx = len(items) + 1
            question = VisualQuestion3D(
                question_id=f"{scene.scene_id}__ov3d__{idx:02d}",
                scene_id=scene.scene_id,
                family=QuestionFamily3D.OBJECT_OVERLAP_3D,
                template_id=f"overlap_3d_{mode}_v1",
                prompt_text=prompt_text,
                expected_answer_type=AnswerType3D.BOOLEAN,
                reference_answer=ref,
                solver_query=query.model_dump(mode="json"),
            )
            items.append(BenchmarkItem3D(scene=scene, question=question))

        return items

    def _generate_vertical_relation_items(
        self,
        scene: SceneSpec3D,
        entities: list[SceneEntity3D],
        rng: random.Random,
    ) -> list[BenchmarkItem3D]:
        if len(entities) < 2:
            return []

        pairs = self._ordered_pairs(entities, rng)
        predicates: list[Literal["above", "below", "touching_above", "equal_height"]] = [
            "above",
            "touching_above",
            "below",
            "equal_height",
        ]

        candidates: list[
            tuple[SceneEntity3D, SceneEntity3D, Literal["above", "below", "touching_above", "equal_height"], bool]
        ] = []
        for idx, (src, tgt) in enumerate(pairs):
            res = self.solver.solve_vertical_relation(scene, src.id, tgt.id)
            if res.is_above:
                pred: Literal["above", "below", "touching_above", "equal_height"] = (
                    "touching_above" if res.is_flush_contact and idx % 2 == 1 else "above"
                )
            elif res.is_below:
                pred = "below"
            elif res.relation.value == "equal_height":
                pred = "equal_height"
            else:
                pred = predicates[idx % len(predicates)]
            query = VerticalRelationQuery3D(source_id=src.id, target_id=tgt.id, predicate=pred)
            ref = evaluate_vertical_relation_query(self.solver, scene, query)
            candidates.append((src, tgt, pred, bool(ref.value)))

        true_cases = [c for c in candidates if c[3]]
        false_cases = [c for c in candidates if not c[3]]
        ordered: list[tuple[SceneEntity3D, SceneEntity3D, Literal["above", "below", "touching_above", "equal_height"]]] = []
        while true_cases or false_cases:
            if true_cases:
                s, t, p, _ = true_cases.pop(0)
                ordered.append((s, t, p))
            if false_cases:
                s, t, p, _ = false_cases.pop(0)
                ordered.append((s, t, p))

        items: list[BenchmarkItem3D] = []
        for src, tgt, pred in ordered:
            if len(items) >= self.config.max_questions_per_family:
                break

            query = VerticalRelationQuery3D(source_id=src.id, target_id=tgt.id, predicate=pred)
            ref = evaluate_vertical_relation_query(self.solver, scene, query)

            if pred == "above":
                prompt_text = f"In 3D space, is {src.label} positioned entirely above {tgt.label}?"
            elif pred == "below":
                prompt_text = f"In 3D space, is {src.label} positioned entirely below {tgt.label}?"
            elif pred == "touching_above":
                prompt_text = f"Does {src.label} rest directly on top of {tgt.label}?"
            else:
                prompt_text = f"Do {src.label} and {tgt.label} share the exact same vertical height range?"

            idx = len(items) + 1
            question = VisualQuestion3D(
                question_id=f"{scene.scene_id}__vrel3d__{idx:02d}",
                scene_id=scene.scene_id,
                family=QuestionFamily3D.VERTICAL_RELATION_3D,
                template_id=f"vertical_relation_{pred}_v1",
                prompt_text=prompt_text,
                expected_answer_type=AnswerType3D.BOOLEAN,
                reference_answer=ref,
                solver_query=query.model_dump(mode="json"),
            )
            items.append(BenchmarkItem3D(scene=scene, question=question))

        return items

    def _generate_vertical_clearance_items(
        self,
        scene: SceneSpec3D,
        entities: list[SceneEntity3D],
        rng: random.Random,
    ) -> list[BenchmarkItem3D]:
        want_num = self.config.clearance_variant in ("numeric", "both")
        want_bool = self.config.clearance_variant in ("boolean", "both")

        items: list[BenchmarkItem3D] = []

        # 1. Defined entity-to-entity vertical clearances (only where Z disjoint AND XY overlaps).
        for pair_idx, (src, tgt) in enumerate(self._ordered_pairs(entities, rng)):
            if len(items) >= self.config.max_questions_per_family:
                break

            try:
                num_query = VerticalClearanceQuery3D(
                    source_id=src.id,
                    target_id=tgt.id,
                    target_kind="entity",
                    answer_type=AnswerType3D.NUMERIC,
                    numeric_tolerance_meters=self.config.numeric_tolerance_m,
                )
                num_ref = evaluate_vertical_clearance_query(self.solver, scene, num_query)
            except UnsolvableTask3DError:
                continue

            clearance_m = float(num_ref.value)
            emit_num = want_num and (not want_bool or pair_idx % 2 == 0)
            if emit_num:
                idx = len(items) + 1
                question = VisualQuestion3D(
                    question_id=f"{scene.scene_id}__vclr3d__{idx:02d}",
                    scene_id=scene.scene_id,
                    family=QuestionFamily3D.VERTICAL_CLEARANCE_3D,
                    template_id="vertical_clearance_entity_numeric_v1",
                    prompt_text=(
                        f"Using the {scene.scale_bar.display_meters:g} m scale reference, what is the "
                        f"vertical clearance in meters between {src.label} and {tgt.label}?"
                    ),
                    expected_answer_type=AnswerType3D.NUMERIC,
                    reference_answer=num_ref,
                    solver_query=num_query.model_dump(mode="json"),
                )
                items.append(BenchmarkItem3D(scene=scene, question=question))
            elif want_bool:
                delta = 0.5 if clearance_m >= 0.5 else 0.5
                threshold = max(0.5, _round_half(clearance_m + (delta if pair_idx % 2 == 1 else -delta)))
                bool_query = VerticalClearanceQuery3D(
                    source_id=src.id,
                    target_id=tgt.id,
                    target_kind="entity",
                    answer_type=AnswerType3D.BOOLEAN,
                    min_required_meters=threshold,
                )
                bool_ref = evaluate_vertical_clearance_query(self.solver, scene, bool_query)
                idx = len(items) + 1
                question = VisualQuestion3D(
                    question_id=f"{scene.scene_id}__vclr3d__{idx:02d}",
                    scene_id=scene.scene_id,
                    family=QuestionFamily3D.VERTICAL_CLEARANCE_3D,
                    template_id="vertical_clearance_entity_boolean_v1",
                    prompt_text=(
                        f"Is there at least {threshold:g} m of vertical clearance between "
                        f"{src.label} and {tgt.label}?"
                    ),
                    expected_answer_type=AnswerType3D.BOOLEAN,
                    reference_answer=bool_ref,
                    solver_query=bool_query.model_dump(mode="json"),
                )
                items.append(BenchmarkItem3D(scene=scene, question=question))

        # 2. Ceiling clearance (if room remains and entity doesn't span full ceiling).
        if self.config.include_ceiling_clearance and len(items) < self.config.max_questions_per_family:
            shuffled = list(entities)
            rng.shuffle(shuffled)
            for idx_ent, ent in enumerate(shuffled):
                if len(items) >= self.config.max_questions_per_family:
                    break
                num_query = VerticalClearanceQuery3D(
                    source_id=ent.id,
                    target_id=None,
                    target_kind="ceiling",
                    answer_type=AnswerType3D.NUMERIC,
                    numeric_tolerance_meters=self.config.numeric_tolerance_m,
                )
                num_ref = evaluate_vertical_clearance_query(self.solver, scene, num_query)
                gap_m = float(num_ref.value)
                if gap_m <= 0.05:
                    continue

                emit_num = want_num and (not want_bool or (len(items) + idx_ent) % 2 == 0)
                if emit_num:
                    idx = len(items) + 1
                    question = VisualQuestion3D(
                        question_id=f"{scene.scene_id}__vclr3d__{idx:02d}",
                        scene_id=scene.scene_id,
                        family=QuestionFamily3D.VERTICAL_CLEARANCE_3D,
                        template_id="vertical_clearance_ceiling_numeric_v1",
                        prompt_text=(
                            f"Using the {scene.scale_bar.display_meters:g} m scale reference, what is the "
                            f"vertical clearance in meters between the top of {ent.label} and the ceiling?"
                        ),
                        expected_answer_type=AnswerType3D.NUMERIC,
                        reference_answer=num_ref,
                        solver_query=num_query.model_dump(mode="json"),
                    )
                    items.append(BenchmarkItem3D(scene=scene, question=question))
                elif want_bool:
                    threshold = max(0.5, _round_half(gap_m + (0.5 if idx_ent % 2 == 1 else -0.5)))
                    bool_query = VerticalClearanceQuery3D(
                        source_id=ent.id,
                        target_id=None,
                        target_kind="ceiling",
                        answer_type=AnswerType3D.BOOLEAN,
                        min_required_meters=threshold,
                    )
                    bool_ref = evaluate_vertical_clearance_query(self.solver, scene, bool_query)
                    idx = len(items) + 1
                    question = VisualQuestion3D(
                        question_id=f"{scene.scene_id}__vclr3d__{idx:02d}",
                        scene_id=scene.scene_id,
                        family=QuestionFamily3D.VERTICAL_CLEARANCE_3D,
                        template_id="vertical_clearance_ceiling_boolean_v1",
                        prompt_text=(
                            f"Is there at least {threshold:g} m of vertical clearance between the top of "
                            f"{ent.label} and the ceiling?"
                        ),
                        expected_answer_type=AnswerType3D.BOOLEAN,
                        reference_answer=bool_ref,
                        solver_query=bool_query.model_dump(mode="json"),
                    )
                    items.append(BenchmarkItem3D(scene=scene, question=question))

        return items

    def _generate_visibility_items(
        self,
        scene: SceneSpec3D,
        entities: list[SceneEntity3D],
        rng: random.Random,
    ) -> list[BenchmarkItem3D]:
        if not is_camera_isometric_aligned(scene):
            return []

        shuffled = list(entities)
        rng.shuffle(shuffled)

        # Balance CLEAR vs FULLY_OCCLUDED entities when both exist; skip PARTIALLY_OCCLUDED.
        clear_targets: list[tuple[SceneEntity3D, CameraVisibilityQuery3D]] = []
        occluded_targets: list[tuple[SceneEntity3D, CameraVisibilityQuery3D]] = []

        for ent in shuffled:
            query = CameraVisibilityQuery3D(target_id=ent.id, require_clear=True)
            try:
                ref = evaluate_camera_visibility_query(self.solver, scene, query)
            except UnsolvableTask3DError:
                continue
            if ref.value is True:
                clear_targets.append((ent, query))
            else:
                occluded_targets.append((ent, query))

        ordered: list[tuple[SceneEntity3D, CameraVisibilityQuery3D]] = []
        while clear_targets or occluded_targets:
            if occluded_targets:
                ordered.append(occluded_targets.pop(0))
            if clear_targets:
                ordered.append(clear_targets.pop(0))

        items: list[BenchmarkItem3D] = []
        for ent, query in ordered:
            if len(items) >= self.config.max_questions_per_family:
                break
            ref = evaluate_camera_visibility_query(self.solver, scene, query)
            idx = len(items) + 1
            question = VisualQuestion3D(
                question_id=f"{scene.scene_id}__vis3d__{idx:02d}",
                scene_id=scene.scene_id,
                family=QuestionFamily3D.CAMERA_VISIBILITY_3D,
                template_id="camera_visibility_clear_v1",
                prompt_text=(
                    f"Is {ent.label} visibly present and unobstructed from the camera viewpoint?"
                ),
                expected_answer_type=AnswerType3D.BOOLEAN,
                reference_answer=ref,
                solver_query=query.model_dump(mode="json"),
            )
            items.append(BenchmarkItem3D(scene=scene, question=question))

        return items
