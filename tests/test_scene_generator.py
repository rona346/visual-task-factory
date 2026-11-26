"""Unit tests for the procedural warehouse SceneGenerator."""

from __future__ import annotations

import math

import pytest

from visual_task_factory.models import (
    ClutterLevel,
    DifficultyTier,
    Doorway,
    SceneSpec,
    VariationConfig,
    VisualTheme,
    WallSegment,
    ZoneType,
)
from visual_task_factory.scene_generator import generate_scene


def _make_config(
    *,
    seed: int = 101,
    difficulty: DifficultyTier = DifficultyTier.MEDIUM,
    orientation_deg: int = 0,
    theme: VisualTheme = VisualTheme.SAFETY_STANDARD,
    clutter_level: ClutterLevel = ClutterLevel.LOW,
    min_obstacles: int = 3,
    max_obstacles: int = 8,
    min_hazard_zones: int = 1,
    scale_meters_per_unit: float = 0.5,
) -> VariationConfig:
    return VariationConfig(
        seed=seed,
        topology_id="warehouse_split_aisle_v1",
        difficulty=difficulty,
        orientation_deg=orientation_deg,  # type: ignore[arg-type]
        scale_meters_per_unit=scale_meters_per_unit,
        theme=theme,
        clutter_level=clutter_level,
        min_hazard_zones=min_hazard_zones,
        min_obstacles=min_obstacles,
        max_obstacles=max_obstacles,
    )


def _door_lies_on_wall(door: Doorway, wall: WallSegment) -> bool:
    tol = 1e-6
    half_door = door.width / 2.0
    if door.orientation == "vertical" and math.isclose(wall.start.x, wall.end.x, abs_tol=tol):
        if not math.isclose(door.position.x, wall.start.x, abs_tol=tol):
            return False
        y_min = min(wall.start.y, wall.end.y)
        y_max = max(wall.start.y, wall.end.y)
        return door.position.y - half_door >= y_min and door.position.y + half_door <= y_max

    if door.orientation == "horizontal" and math.isclose(wall.start.y, wall.end.y, abs_tol=tol):
        if not math.isclose(door.position.y, wall.start.y, abs_tol=tol):
            return False
        x_min = min(wall.start.x, wall.end.x)
        x_max = max(wall.start.x, wall.end.x)
        return door.position.x - half_door >= x_min and door.position.x + half_door <= x_max

    return False


def test_generate_valid_scene_spec_and_required_elements() -> None:
    cfg = _make_config(seed=7, difficulty=DifficultyTier.MEDIUM)
    scene = generate_scene(cfg)

    assert isinstance(scene, SceneSpec)
    # Round-trip through Pydantic JSON validation to confirm all invariants hold.
    assert SceneSpec.model_validate_json(scene.model_dump_json()) == scene

    # Required structural features: boundary + interior walls, doors, >=2 exits, hazard/restricted zone, entities, scale bar.
    wall_ids = {w.id for w in scene.walls}
    assert {"W_BOUND_N", "W_BOUND_S", "W_BOUND_W", "W_BOUND_E"}.issubset(wall_ids)
    assert any(w_id.startswith("W_INT_") for w_id in wall_ids)

    assert len(scene.exits) >= 2
    assert len(scene.doors) >= 1
    assert any(not door.is_blocked for door in scene.doors)

    for door in scene.doors:
        assert any(_door_lies_on_wall(door, wall) for wall in scene.walls), (
            f"Doorway {door.id} does not lie cleanly on any wall segment."
        )

    assert any(
        zone.zone_type in (ZoneType.HAZARD, ZoneType.RESTRICTED) for zone in scene.zones
    )
    assert cfg.min_obstacles <= len(scene.entities) <= cfg.max_obstacles
    assert scene.scale_bar.meters_per_world_unit == pytest.approx(cfg.scale_meters_per_unit)


def test_same_seed_reproducibility() -> None:
    cfg_a = _make_config(seed=2026, difficulty=DifficultyTier.HARD, orientation_deg=90)
    cfg_b = _make_config(seed=2026, difficulty=DifficultyTier.HARD, orientation_deg=90)

    scene_a = generate_scene(cfg_a)
    scene_b = generate_scene(cfg_b)

    assert scene_a == scene_b
    assert scene_a.model_dump_json() == scene_b.model_dump_json()


def test_different_seeds_produce_meaningful_layout_differences() -> None:
    scene_1 = generate_scene(_make_config(seed=10))
    scene_2 = generate_scene(_make_config(seed=99))

    assert scene_1.walls != scene_2.walls
    assert scene_1.entities != scene_2.entities
    assert scene_1.zones != scene_2.zones


@pytest.mark.parametrize("difficulty", [DifficultyTier.EASY, DifficultyTier.MEDIUM, DifficultyTier.HARD])
@pytest.mark.parametrize("orientation", [0, 90, 180, 270])
def test_variation_axes_difficulty_and_orientation(
    difficulty: DifficultyTier,
    orientation: int,
) -> None:
    cfg = _make_config(
        seed=55,
        difficulty=difficulty,
        orientation_deg=orientation,
        theme=VisualTheme.BLUEPRINT,
        clutter_level=ClutterLevel.HIGH,
        min_obstacles=2,
        max_obstacles=10,
    )
    scene = generate_scene(cfg)

    assert scene.variation.difficulty == difficulty
    assert scene.variation.orientation_deg == orientation
    assert scene.variation.theme == VisualTheme.BLUEPRINT
    assert scene.variation.clutter_level == ClutterLevel.HIGH
    assert 2 <= len(scene.entities) <= 10

    if orientation in (90, 270):
        assert scene.width < scene.height
    else:
        assert scene.width > scene.height

    for door in scene.doors:
        assert any(_door_lies_on_wall(door, wall) for wall in scene.walls)
