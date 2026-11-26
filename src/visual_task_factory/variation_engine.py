"""Deterministic Meaningful Variation Engine for spatial warehouse benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from visual_task_factory.ground_truth_solver import (
    GroundTruthSolver,
    ReachabilityQuery,
    UnsolvableTaskError,
)
from visual_task_factory.models import (
    ClutterLevel,
    DifficultyTier,
    SceneSpec,
    VariationConfig,
    VisualTheme,
    ZoneType,
)
from visual_task_factory.scene_generator import (
    SUPPORTED_TOPOLOGIES,
    WarehouseSceneGenerator,
)


class SceneSignature(BaseModel):
    """Canonical structural signature computed strictly from physical scene layout."""

    model_config = ConfigDict(extra="forbid")

    structural_hash: str
    canvas_size: tuple[float, float]
    wall_count: int
    interior_wall_count: int
    open_door_count: int
    blocked_door_count: int
    exit_count: int
    zone_type_counts: dict[str, int]
    entity_count: int
    entity_type_counts: dict[str, int]
    scale_meters_per_unit: float


class SceneDifferenceReport(BaseModel):
    """Comparison summarizing structural differences between two scenes."""

    model_config = ConfigDict(extra="forbid")

    is_meaningfully_different: bool
    changed_dimensions: list[str]
    signature_a: SceneSignature
    signature_b: SceneSignature


class VariationPlanConfig(BaseModel):
    """Controlled multi-axis plan for generating meaningfully varied scenes."""

    model_config = ConfigDict(extra="forbid")

    base_seed: int = Field(default=100, ge=0)
    count: int = Field(default=6, ge=1, le=200)
    topologies: tuple[str, ...] = SUPPORTED_TOPOLOGIES
    difficulties: tuple[DifficultyTier, ...] = (
        DifficultyTier.EASY,
        DifficultyTier.MEDIUM,
        DifficultyTier.HARD,
    )
    orientations: tuple[Literal[0, 90, 180, 270], ...] = (0, 90, 180, 270)
    scales_meters_per_unit: tuple[float, ...] = (0.5, 1.0, 0.25)
    themes: tuple[VisualTheme, ...] = (
        VisualTheme.SAFETY_STANDARD,
        VisualTheme.BLUEPRINT,
        VisualTheme.HIGH_CONTRAST,
        VisualTheme.CLUTTERED_OPS,
    )
    clutter_levels: tuple[ClutterLevel, ...] = (
        ClutterLevel.NONE,
        ClutterLevel.LOW,
        ClutterLevel.MEDIUM,
        ClutterLevel.HIGH,
    )
    min_hazard_zones: int = Field(default=1, ge=0, le=20)
    min_obstacles: int = Field(default=2, ge=0, le=100)
    max_obstacles: int = Field(default=10, ge=0, le=100)
    require_reachable_scene: bool = True
    deduplicate_layouts: bool = True
    max_retries_per_slot: int = Field(default=5, ge=0, le=25)

    @field_validator(
        "topologies",
        "difficulties",
        "orientations",
        "scales_meters_per_unit",
        "themes",
        "clutter_levels",
    )
    @classmethod
    def _require_non_empty_axis(cls, value: tuple) -> tuple:
        if not value:
            raise ValueError("Every variation axis tuple must contain at least one entry.")
        return value

    @field_validator("topologies")
    @classmethod
    def _validate_topologies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value if item.strip())
        if len(cleaned) != len(value):
            raise ValueError("topologies must not contain empty strings.")
        return cleaned

    @field_validator("scales_meters_per_unit")
    @classmethod
    def _validate_scales(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        for scale in value:
            if not math.isfinite(scale) or scale <= 0.0 or scale > 50.0:
                raise ValueError(
                    f"Each scale in scales_meters_per_unit must be finite and in (0, 50]; got {scale!r}."
                )
        return value

    @model_validator(mode="after")
    def _validate_obstacle_bounds(self) -> VariationPlanConfig:
        if self.max_obstacles < self.min_obstacles:
            raise ValueError(
                f"max_obstacles ({self.max_obstacles}) must be >= min_obstacles ({self.min_obstacles})."
            )
        return self


class GeneratedVariationItem(BaseModel):
    """Single generated scene paired with its VariationConfig and structural signature."""

    model_config = ConfigDict(extra="forbid")

    slot_index: int
    variation: VariationConfig
    scene: SceneSpec
    signature: SceneSignature
    retries_used: int = 0


class VariationBatchResult(BaseModel):
    """Output and measurable diversity summary for a variation sweep."""

    model_config = ConfigDict(extra="forbid")

    items: list[GeneratedVariationItem]
    unique_structural_signatures: int
    duplicate_rejections: int
    unsolvable_rejections: int
    topology_coverage: dict[str, int]
    difficulty_coverage: dict[str, int]
    orientation_coverage: dict[int, int]
    theme_coverage: dict[str, int]
    clutter_coverage: dict[str, int]
    entity_types_observed: list[str]


def _round2(value: float) -> float:
    return round(float(value), 2)


def compute_scene_signature(scene: SceneSpec) -> SceneSignature:
    """Compute a deterministic structural signature ignoring IDs, seeds, labels, and theme names."""
    if not isinstance(scene, SceneSpec):
        raise TypeError(f"Expected SceneSpec instance, got {type(scene).__name__}.")

    walls_geom = sorted(
        (
            _round2(min(w.start.x, w.end.x)),
            _round2(min(w.start.y, w.end.y)),
            _round2(max(w.start.x, w.end.x)),
            _round2(max(w.start.y, w.end.y)),
            _round2(w.thickness),
        )
        for w in scene.walls
    )
    doors_geom = sorted(
        (
            _round2(d.position.x),
            _round2(d.position.y),
            _round2(d.width),
            d.orientation,
            d.is_blocked,
        )
        for d in scene.doors
    )
    exits_geom = sorted(
        (
            _round2(e.position.x),
            _round2(e.position.y),
            _round2(e.width),
            e.is_emergency_exit,
        )
        for e in scene.exits
    )
    zones_geom = sorted(
        (
            z.zone_type.value,
            _round2(z.bounds.x),
            _round2(z.bounds.y),
            _round2(z.bounds.width),
            _round2(z.bounds.height),
        )
        for z in scene.zones
    )
    entities_geom = sorted(
        (
            ent.entity_type.value,
            _round2(ent.bounds.x),
            _round2(ent.bounds.y),
            _round2(ent.bounds.width),
            _round2(ent.bounds.height),
            ent.passable,
        )
        for ent in scene.entities
    )

    canonical_payload = {
        "canvas": (_round2(scene.width), _round2(scene.height)),
        "scale_m_per_unit": round(scene.scale_bar.meters_per_world_unit, 4),
        "walls": walls_geom,
        "doors": doors_geom,
        "exits": exits_geom,
        "zones": zones_geom,
        "entities": entities_geom,
    }
    structural_hash = hashlib.sha256(
        json.dumps(canonical_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    zone_type_counts: dict[str, int] = {}
    for z in scene.zones:
        zone_type_counts[z.zone_type.value] = zone_type_counts.get(z.zone_type.value, 0) + 1

    entity_type_counts: dict[str, int] = {}
    for ent in scene.entities:
        entity_type_counts[ent.entity_type.value] = (
            entity_type_counts.get(ent.entity_type.value, 0) + 1
        )

    interior_walls = sum(1 for w in scene.walls if not w.id.startswith("W_BOUND_"))
    open_doors = sum(1 for d in scene.doors if not d.is_blocked)
    blocked_doors = sum(1 for d in scene.doors if d.is_blocked)

    return SceneSignature(
        structural_hash=structural_hash,
        canvas_size=(_round2(scene.width), _round2(scene.height)),
        wall_count=len(scene.walls),
        interior_wall_count=interior_walls,
        open_door_count=open_doors,
        blocked_door_count=blocked_doors,
        exit_count=len(scene.exits),
        zone_type_counts=dict(sorted(zone_type_counts.items())),
        entity_count=len(scene.entities),
        entity_type_counts=dict(sorted(entity_type_counts.items())),
        scale_meters_per_unit=round(scene.scale_bar.meters_per_world_unit, 4),
    )


def compare_scenes(scene_a: SceneSpec, scene_b: SceneSpec) -> SceneDifferenceReport:
    """Assess whether two SceneSpecs differ meaningfully beyond IDs, seeds, or themes."""
    sig_a = compute_scene_signature(scene_a)
    sig_b = compute_scene_signature(scene_b)

    changed: list[str] = []
    if sig_a.canvas_size != sig_b.canvas_size:
        changed.append("canvas_size")
    if (sig_a.wall_count, sig_a.interior_wall_count) != (
        sig_b.wall_count,
        sig_b.interior_wall_count,
    ):
        changed.append("wall_topology")
    elif _wall_geom_digest(scene_a) != _wall_geom_digest(scene_b):
        changed.append("wall_geometry")

    if (sig_a.open_door_count, sig_a.blocked_door_count) != (
        sig_b.open_door_count,
        sig_b.blocked_door_count,
    ) or _door_geom_digest(scene_a) != _door_geom_digest(scene_b):
        changed.append("doorways")

    if _exit_geom_digest(scene_a) != _exit_geom_digest(scene_b):
        changed.append("exits")

    if sig_a.zone_type_counts != sig_b.zone_type_counts or _zone_geom_digest(
        scene_a
    ) != _zone_geom_digest(scene_b):
        changed.append("zones")

    if sig_a.entity_type_counts != sig_b.entity_type_counts:
        changed.append("entity_composition")
    elif _entity_geom_digest(scene_a) != _entity_geom_digest(scene_b):
        changed.append("entity_positions")

    if not math.isclose(sig_a.scale_meters_per_unit, sig_b.scale_meters_per_unit, abs_tol=1e-6):
        changed.append("scale")

    return SceneDifferenceReport(
        is_meaningfully_different=sig_a.structural_hash != sig_b.structural_hash,
        changed_dimensions=changed,
        signature_a=sig_a,
        signature_b=sig_b,
    )


def _wall_geom_digest(scene: SceneSpec) -> list[tuple[float, float, float, float]]:
    return sorted(
        (
            _round2(min(w.start.x, w.end.x)),
            _round2(min(w.start.y, w.end.y)),
            _round2(max(w.start.x, w.end.x)),
            _round2(max(w.start.y, w.end.y)),
        )
        for w in scene.walls
    )


def _door_geom_digest(scene: SceneSpec) -> list[tuple[float, float, str, bool]]:
    return sorted(
        (_round2(d.position.x), _round2(d.position.y), d.orientation, d.is_blocked)
        for d in scene.doors
    )


def _exit_geom_digest(scene: SceneSpec) -> list[tuple[float, float]]:
    return sorted((_round2(e.position.x), _round2(e.position.y)) for e in scene.exits)


def _zone_geom_digest(scene: SceneSpec) -> list[tuple[str, float, float, float, float]]:
    return sorted(
        (
            z.zone_type.value,
            _round2(z.bounds.x),
            _round2(z.bounds.y),
            _round2(z.bounds.width),
            _round2(z.bounds.height),
        )
        for z in scene.zones
    )


def _entity_geom_digest(scene: SceneSpec) -> list[tuple[str, float, float, float, float]]:
    return sorted(
        (
            ent.entity_type.value,
            _round2(ent.bounds.x),
            _round2(ent.bounds.y),
            _round2(ent.bounds.width),
            _round2(ent.bounds.height),
        )
        for ent in scene.entities
    )


class VariationEngine:
    """Generates deterministic, multi-axis varied warehouse scenes with deduplication."""

    def __init__(
        self,
        generator: WarehouseSceneGenerator | None = None,
        solver: GroundTruthSolver | None = None,
    ) -> None:
        self.generator = generator or WarehouseSceneGenerator()
        self.solver = solver or GroundTruthSolver()

    def plan_configs(self, plan: VariationPlanConfig) -> list[VariationConfig]:
        """Build a balanced, deterministic VariationConfig schedule across all axes."""
        if not isinstance(plan, VariationPlanConfig):
            raise TypeError(
                f"Expected VariationPlanConfig instance, got {type(plan).__name__}."
            )

        configs: list[VariationConfig] = []
        n_topo = len(plan.topologies)
        n_diff = len(plan.difficulties)
        n_orient = len(plan.orientations)
        n_scale = len(plan.scales_meters_per_unit)
        n_theme = len(plan.themes)
        n_clutter = len(plan.clutter_levels)

        for idx in range(plan.count):
            topology_id = plan.topologies[idx % n_topo]
            difficulty = plan.difficulties[(idx + (idx // n_diff)) % n_diff]
            orientation = plan.orientations[(idx + (idx // n_orient)) % n_orient]
            scale = plan.scales_meters_per_unit[(idx + 2 * (idx // n_scale)) % n_scale]
            theme = plan.themes[(idx + (idx // n_theme)) % n_theme]
            clutter = plan.clutter_levels[(idx + 3 * (idx // n_clutter)) % n_clutter]

            configs.append(
                VariationConfig(
                    seed=plan.base_seed + idx * 17,
                    topology_id=topology_id,
                    difficulty=difficulty,
                    orientation_deg=orientation,
                    scale_meters_per_unit=scale,
                    theme=theme,
                    clutter_level=clutter,
                    min_hazard_zones=plan.min_hazard_zones,
                    min_obstacles=plan.min_obstacles,
                    max_obstacles=plan.max_obstacles,
                )
            )
        return configs

    def generate_batch(self, plan: VariationPlanConfig) -> VariationBatchResult:
        """Generate a varied batch of scenes, rejecting duplicate or unsolvable layouts."""
        base_configs = self.plan_configs(plan)
        seen_hashes: set[str] = set()
        items: list[GeneratedVariationItem] = []
        duplicate_rejections = 0
        unsolvable_rejections = 0

        for slot_idx, base_cfg in enumerate(base_configs):
            chosen_item: GeneratedVariationItem | None = None

            for attempt in range(plan.max_retries_per_slot + 1):
                candidate_cfg = base_cfg.model_copy(
                    update={"seed": base_cfg.seed + attempt * 1009}
                )
                scene_id = f"scene_var_{slot_idx:03d}_s{candidate_cfg.seed}"
                scene = self.generator.generate(candidate_cfg, scene_id=scene_id)
                signature = compute_scene_signature(scene)

                if plan.deduplicate_layouts and signature.structural_hash in seen_hashes:
                    duplicate_rejections += 1
                    continue

                if plan.require_reachable_scene and not self._has_reachable_zone(scene):
                    unsolvable_rejections += 1
                    continue

                seen_hashes.add(signature.structural_hash)
                chosen_item = GeneratedVariationItem(
                    slot_index=slot_idx,
                    variation=candidate_cfg,
                    scene=scene,
                    signature=signature,
                    retries_used=attempt,
                )
                break

            if chosen_item is None:
                raise RuntimeError(
                    f"Failed to generate a valid unique scene for slot {slot_idx} "
                    f"after {plan.max_retries_per_slot + 1} attempts."
                )
            items.append(chosen_item)

        topology_cov: dict[str, int] = {}
        difficulty_cov: dict[str, int] = {}
        orientation_cov: dict[int, int] = {}
        theme_cov: dict[str, int] = {}
        clutter_cov: dict[str, int] = {}
        entity_types: set[str] = set()

        for item in items:
            v = item.variation
            topology_cov[v.topology_id] = topology_cov.get(v.topology_id, 0) + 1
            difficulty_cov[v.difficulty.value] = difficulty_cov.get(v.difficulty.value, 0) + 1
            orientation_cov[v.orientation_deg] = orientation_cov.get(v.orientation_deg, 0) + 1
            theme_cov[v.theme.value] = theme_cov.get(v.theme.value, 0) + 1
            clutter_cov[v.clutter_level.value] = clutter_cov.get(v.clutter_level.value, 0) + 1
            entity_types.update(item.signature.entity_type_counts.keys())

        return VariationBatchResult(
            items=items,
            unique_structural_signatures=len({it.signature.structural_hash for it in items}),
            duplicate_rejections=duplicate_rejections,
            unsolvable_rejections=unsolvable_rejections,
            topology_coverage=dict(sorted(topology_cov.items())),
            difficulty_coverage=dict(sorted(difficulty_cov.items())),
            orientation_coverage=dict(sorted(orientation_cov.items())),
            theme_coverage=dict(sorted(theme_cov.items())),
            clutter_coverage=dict(sorted(clutter_cov.items())),
            entity_types_observed=sorted(entity_types),
        )

    def _has_reachable_zone(self, scene: SceneSpec) -> bool:
        for zone in scene.zones:
            if zone.zone_type in (ZoneType.HAZARD, ZoneType.RESTRICTED):
                continue
            try:
                res = self.solver.solve_reachability(
                    scene,
                    ReachabilityQuery(start_zone_id=zone.id),
                )
                if res.is_reachable:
                    return True
            except UnsolvableTaskError:
                continue
        return False


def generate_varied_scenes(plan: VariationPlanConfig) -> VariationBatchResult:
    """Convenience entry point to generate a multi-axis varied batch of scenes."""
    return VariationEngine().generate_batch(plan)
