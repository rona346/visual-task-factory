"""Deterministic template-based Question Generator backed by GroundTruthSolver."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

from visual_task_factory.ground_truth_solver import (
    ClearanceQuery,
    ContainmentMode,
    GroundTruthSolver,
    ReachabilityQuery,
    RegionCountingQuery,
    UnsolvableTaskError,
    UnsupportedTaskError,
)
from visual_task_factory.models import (
    AnswerType,
    BenchmarkItem,
    EntityType,
    ExitMarker,
    QuestionFamily,
    SceneEntity,
    SceneSpec,
    VisualQuestion,
    Zone,
    ZoneType,
)

_ENTITY_PLURALS: dict[EntityType, str] = {
    EntityType.PALLET_RACK: "pallet racks",
    EntityType.FORKLIFT: "forklifts",
    EntityType.CRATE: "crates",
    EntityType.PILLAR: "pillars",
    EntityType.CONVEYOR: "conveyors",
    EntityType.ROBOT: "autonomous mobile robots",
}

_CONTAINMENT_PHRASES: dict[ContainmentMode, str] = {
    ContainmentMode.STRICT_INSIDE: "completely inside",
    ContainmentMode.CENTER_INSIDE: "centered inside",
    ContainmentMode.ANY_OVERLAP: "overlapping any part of",
}

_DEFAULT_FAMILIES: tuple[QuestionFamily, ...] = (
    QuestionFamily.CONSTRAINED_REACHABILITY,
    QuestionFamily.REGION_CONTAINMENT_COUNTING,
    QuestionFamily.SCALE_GROUNDED_CLEARANCE,
)


@dataclass(frozen=True)
class QuestionGeneratorConfig:
    """Configuration for deterministic template-based question generation."""

    families: tuple[QuestionFamily, ...] = _DEFAULT_FAMILIES
    max_questions_per_family: int = 1
    containment_mode: ContainmentMode = ContainmentMode.STRICT_INSIDE
    clearance_variant: str = "both"  # "boolean", "numeric", or "both"
    min_reachability_margin_m: float = 0.5
    numeric_tolerance_m: float = 0.05
    include_candidate_choices: bool = True
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if not self.families:
            raise ValueError("families must contain at least one QuestionFamily.")
        if self.max_questions_per_family < 1 or self.max_questions_per_family > 10:
            raise ValueError("max_questions_per_family must be between 1 and 10.")
        if self.clearance_variant not in ("boolean", "numeric", "both"):
            raise ValueError("clearance_variant must be 'boolean', 'numeric', or 'both'.")
        if self.min_reachability_margin_m < 0.0:
            raise ValueError("min_reachability_margin_m must be >= 0.0.")
        if self.numeric_tolerance_m < 0.0:
            raise ValueError("numeric_tolerance_m must be >= 0.0.")


def _stable_rng(scene: SceneSpec, seed_offset: int) -> random.Random:
    digest = hashlib.sha256(
        f"{scene.scene_id}:{scene.variation.seed}:{seed_offset}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _is_entity_visibly_labeled(entity: SceneEntity) -> bool:
    """Match renderer visibility heuristic so questions only reference labeled items."""
    if entity.entity_type == EntityType.PILLAR:
        return False
    if entity.entity_type == EntityType.FORKLIFT:
        return max(entity.bounds.width, entity.bounds.height) >= 3.0
    return min(entity.bounds.width, entity.bounds.height) >= 1.4


def _round_half(value: float) -> float:
    return round(value * 2.0) / 2.0


class QuestionGenerator:
    """Generates deterministic BenchmarkItem questions using GroundTruthSolver."""

    def __init__(
        self,
        config: QuestionGeneratorConfig | None = None,
        solver: GroundTruthSolver | None = None,
    ) -> None:
        self.config = config or QuestionGeneratorConfig()
        self.solver = solver or GroundTruthSolver()

    def generate(self, scene: SceneSpec) -> list[BenchmarkItem]:
        """Generate validated BenchmarkItems for supported families on scene."""
        if not isinstance(scene, SceneSpec):
            raise TypeError(f"Expected SceneSpec instance, got {type(scene).__name__}.")

        rng = _stable_rng(scene, self.config.seed_offset)
        items: list[BenchmarkItem] = []

        for family in self.config.families:
            if family == QuestionFamily.CONSTRAINED_REACHABILITY:
                items.extend(self._generate_reachability_items(scene, rng))
            elif family == QuestionFamily.REGION_CONTAINMENT_COUNTING:
                items.extend(self._generate_counting_items(scene, rng))
            elif family == QuestionFamily.SCALE_GROUNDED_CLEARANCE:
                items.extend(self._generate_clearance_items(scene, rng))
            else:
                raise UnsupportedTaskError(f"Unsupported question family: {family!r}.")

        return items

    def _generate_reachability_items(
        self,
        scene: SceneSpec,
        rng: random.Random,
    ) -> list[BenchmarkItem]:
        eligible_zones = [
            z
            for z in scene.zones
            if z.zone_type not in (ZoneType.HAZARD, ZoneType.RESTRICTED)
        ]
        if not eligible_zones or len(scene.exits) < 1:
            return []

        # Ensure zone labels are unambiguous visually.
        label_counts: dict[str, int] = {}
        for z in scene.zones:
            label_counts[z.label] = label_counts.get(z.label, 0) + 1
        eligible_zones = [z for z in eligible_zones if label_counts[z.label] == 1]
        if not eligible_zones:
            return []

        ordered_zones = list(eligible_zones)
        rng.shuffle(ordered_zones)

        candidate_ids = [e.id for e in scene.exits]
        items: list[BenchmarkItem] = []

        for zone in ordered_zones:
            if len(items) >= self.config.max_questions_per_family:
                break

            query = ReachabilityQuery(
                start_zone_id=zone.id,
                candidate_exit_ids=candidate_ids,
                forbidden_zone_types=(ZoneType.HAZARD, ZoneType.RESTRICTED),
            )
            try:
                res = self.solver.solve_reachability(scene, query)
            except UnsolvableTaskError:
                continue

            if not res.is_reachable or res.reference_answer is None:
                continue

            sorted_dists = sorted(res.reachable_exit_distances_m.values())
            if len(sorted_dists) >= 2:
                gap = sorted_dists[1] - sorted_dists[0]
                if gap < self.config.min_reachability_margin_m:
                    continue

            idx = len(items) + 1
            if len(res.reachable_exit_distances_m) == 1 and len(scene.exits) > 1:
                template_id = "reach_only_valid_exit_v1"
                prompt_text = (
                    f"Starting from the {zone.label} zone, which exit is reachable "
                    "without passing through walls, blocked doors, obstacles, hazard zones, "
                    "or restricted zones?"
                )
            else:
                template_id = "reach_shortest_valid_exit_v1"
                prompt_text = (
                    f"Starting from the {zone.label} zone, which exit has the shortest "
                    "valid path without crossing walls, blocked doors, obstacles, hazard zones, "
                    "or restricted zones?"
                )

            choices = candidate_ids if (self.config.include_candidate_choices and len(candidate_ids) >= 2) else None
            question = VisualQuestion(
                question_id=f"{scene.scene_id}__reach__{idx:02d}",
                scene_id=scene.scene_id,
                family=QuestionFamily.CONSTRAINED_REACHABILITY,
                template_id=template_id,
                prompt_text=prompt_text,
                expected_answer_type=AnswerType.EXIT_ID,
                candidate_choices=choices,
                reference_answer=res.reference_answer,
                solver_query=query.model_dump(mode="json"),
            )
            items.append(BenchmarkItem(scene=scene, question=question))

        return items

    def _generate_counting_items(
        self,
        scene: SceneSpec,
        rng: random.Random,
    ) -> list[BenchmarkItem]:
        if not scene.zones:
            return []

        label_counts: dict[str, int] = {}
        for z in scene.zones:
            label_counts[z.label] = label_counts.get(z.label, 0) + 1

        # Prefer storage zones first (typically contain racks), then staging/loading.
        priority = {
            ZoneType.STORAGE: 0,
            ZoneType.STAGING: 1,
            ZoneType.LOADING: 2,
            ZoneType.HAZARD: 3,
            ZoneType.RESTRICTED: 4,
        }
        eligible_zones = [z for z in scene.zones if label_counts[z.label] == 1]
        eligible_zones.sort(key=lambda z: (priority[z.zone_type], z.id))
        if not eligible_zones:
            return []

        mode = self.config.containment_mode
        mode_phrase = _CONTAINMENT_PHRASES[mode]

        items: list[BenchmarkItem] = []
        # Build candidate (zone, entity_types_filter) tasks deterministically.
        candidate_specs: list[tuple[Zone, tuple[EntityType, ...] | None]] = []
        for zone in eligible_zones:
            if zone.zone_type == ZoneType.STORAGE:
                candidate_specs.append((zone, (EntityType.PALLET_RACK,)))
                candidate_specs.append((zone, None))
            else:
                candidate_specs.append((zone, None))

        # Keep first storage pallet_rack task prioritized when available, shuffle extras.
        if len(candidate_specs) > 2:
            tail = candidate_specs[1:]
            rng.shuffle(tail)
            candidate_specs = [candidate_specs[0], *tail]

        for zone, entity_types in candidate_specs:
            if len(items) >= self.config.max_questions_per_family:
                break

            query = RegionCountingQuery(
                zone_id=zone.id,
                entity_types=entity_types,
                containment_mode=mode,
            )
            try:
                res = self.solver.solve_region_counting(scene, query)
            except UnsolvableTaskError:
                continue

            idx = len(items) + 1
            if entity_types is not None and len(entity_types) == 1:
                noun = _ENTITY_PLURALS[entity_types[0]]
                template_id = f"count_{entity_types[0].value}_{mode.value}_v1"
            else:
                noun = "equipment items or obstacles"
                template_id = f"count_all_entities_{mode.value}_v1"

            prompt_text = (
                f"How many {noun} are {mode_phrase} the {zone.label} zone?"
            )

            question = VisualQuestion(
                question_id=f"{scene.scene_id}__count__{idx:02d}",
                scene_id=scene.scene_id,
                family=QuestionFamily.REGION_CONTAINMENT_COUNTING,
                template_id=template_id,
                prompt_text=prompt_text,
                expected_answer_type=AnswerType.INTEGER,
                reference_answer=res.reference_answer,
                solver_query=query.model_dump(mode="json"),
            )
            items.append(BenchmarkItem(scene=scene, question=question))

        return items

    def _generate_clearance_items(
        self,
        scene: SceneSpec,
        rng: random.Random,
    ) -> list[BenchmarkItem]:
        # Select elements with unique, visibly rendered labels/IDs.
        ent_label_counts: dict[str, int] = {}
        for ent in scene.entities:
            ent_label_counts[ent.label] = ent_label_counts.get(ent.label, 0) + 1

        labeled_entities = [
            ent
            for ent in scene.entities
            if _is_entity_visibly_labeled(ent) and ent_label_counts[ent.label] == 1
        ]
        # Fall back to zones if fewer than 2 visibly labeled entities exist.
        zone_label_counts: dict[str, int] = {}
        for z in scene.zones:
            zone_label_counts[z.label] = zone_label_counts.get(z.label, 0) + 1
        labeled_zones = [z for z in scene.zones if zone_label_counts[z.label] == 1]

        visual_targets: list[tuple[str, str]] = [
            (ent.id, ent.label) for ent in labeled_entities
        ]
        if len(visual_targets) < 2:
            visual_targets.extend((z.id, f"the {z.label} zone") for z in labeled_zones)

        if len(visual_targets) < 2:
            return []

        pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
        for i in range(len(visual_targets)):
            for j in range(i + 1, len(visual_targets)):
                pairs.append((visual_targets[i], visual_targets[j]))
        rng.shuffle(pairs)

        items: list[BenchmarkItem] = []
        want_bool = self.config.clearance_variant in ("boolean", "both")
        want_num = self.config.clearance_variant in ("numeric", "both")

        for pair_idx, ((src_id, src_label), (tgt_id, tgt_label)) in enumerate(pairs):
            if len(items) >= self.config.max_questions_per_family:
                break

            num_query = ClearanceQuery(
                source_id=src_id,
                target_id=tgt_id,
                answer_type=AnswerType.NUMERIC,
                numeric_tolerance_meters=self.config.numeric_tolerance_m,
            )
            try:
                num_res = self.solver.solve_clearance(scene, num_query)
            except UnsolvableTaskError:
                continue

            # Decide whether this item uses BOOLEAN or NUMERIC format.
            if want_bool and want_num:
                use_boolean = (pair_idx + scene.variation.seed) % 2 == 0
            else:
                use_boolean = want_bool

            idx = len(items) + 1
            if use_boolean:
                dist_m = num_res.distance_meters
                margin_m = max(0.5, dist_m * 0.15)
                # Pick a clean 0.5m threshold separated from dist_m to avoid visual ruler ambiguity.
                prefer_true = ((pair_idx + scene.variation.seed) % 2 == 0) and dist_m >= 1.0
                if prefer_true:
                    threshold = max(0.5, math.floor((dist_m - margin_m) * 2.0) / 2.0)
                    if dist_m - threshold < 0.45:
                        threshold = max(0.5, math.ceil((dist_m + margin_m) * 2.0) / 2.0)
                else:
                    threshold = max(0.5, math.ceil((dist_m + margin_m) * 2.0) / 2.0)

                bool_query = ClearanceQuery(
                    source_id=src_id,
                    target_id=tgt_id,
                    answer_type=AnswerType.BOOLEAN,
                    min_required_meters=threshold,
                )
                bool_res = self.solver.solve_clearance(scene, bool_query)
                question = VisualQuestion(
                    question_id=f"{scene.scene_id}__clear__{idx:02d}",
                    scene_id=scene.scene_id,
                    family=QuestionFamily.SCALE_GROUNDED_CLEARANCE,
                    template_id="clearance_threshold_bool_v1",
                    prompt_text=(
                        f"Using the scale bar, is the edge-to-edge clearance between "
                        f"{src_label} and {tgt_label} at least {threshold:g} meters?"
                    ),
                    expected_answer_type=AnswerType.BOOLEAN,
                    reference_answer=bool_res.reference_answer,
                    solver_query=bool_query.model_dump(mode="json"),
                )
            else:
                question = VisualQuestion(
                    question_id=f"{scene.scene_id}__clear__{idx:02d}",
                    scene_id=scene.scene_id,
                    family=QuestionFamily.SCALE_GROUNDED_CLEARANCE,
                    template_id="clearance_distance_numeric_v1",
                    prompt_text=(
                        f"Using the scale bar, what is the minimum edge-to-edge clearance "
                        f"in meters between {src_label} and {tgt_label}?"
                    ),
                    expected_answer_type=AnswerType.NUMERIC,
                    reference_answer=num_res.reference_answer,
                    solver_query=num_query.model_dump(mode="json"),
                )

            items.append(BenchmarkItem(scene=scene, question=question))

        return items


def generate_questions(
    scene: SceneSpec,
    *,
    config: QuestionGeneratorConfig | None = None,
    solver: GroundTruthSolver | None = None,
) -> list[BenchmarkItem]:
    """Convenience entry point to generate deterministic BenchmarkItems for a SceneSpec."""
    return QuestionGenerator(config=config, solver=solver).generate(scene)
