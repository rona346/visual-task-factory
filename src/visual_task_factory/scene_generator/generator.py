"""Deterministic procedural 2D warehouse layout generator."""

from __future__ import annotations

import hashlib
import random
from typing import Literal

from visual_task_factory.models import (
    ClutterLevel,
    DifficultyTier,
    Doorway,
    EntityType,
    ExitMarker,
    Point2D,
    Rect2D,
    ScaleBarSpec,
    SceneEntity,
    SceneSpec,
    VariationConfig,
    WallSegment,
    Zone,
    ZoneType,
)

_CANVAS_BY_DIFFICULTY: dict[DifficultyTier, tuple[float, float]] = {
    DifficultyTier.EASY: (36.0, 26.0),
    DifficultyTier.MEDIUM: (40.0, 30.0),
    DifficultyTier.HARD: (46.0, 34.0),
}

_CLUTTER_EXTRA: dict[ClutterLevel, int] = {
    ClutterLevel.NONE: 0,
    ClutterLevel.LOW: 1,
    ClutterLevel.MEDIUM: 2,
    ClutterLevel.HIGH: 4,
}

_WALL_MARGIN = 0.5
_WALL_THICKNESS = 0.4
_DOOR_WIDTH = 2.5
_EXIT_WIDTH = 2.0

SUPPORTED_TOPOLOGIES: tuple[str, ...] = (
    "warehouse_split_aisle_v1",
    "warehouse_corridor_spine_v1",
    "warehouse_cross_aisle_v1",
)


def _snap_half(value: float) -> float:
    return round(value * 2.0) / 2.0


def _stable_rng(variation: VariationConfig) -> random.Random:
    digest = hashlib.sha256(
        f"{variation.seed}:{variation.topology_id}:{variation.difficulty.value}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _rect_padded(rect: Rect2D, pad: float) -> Rect2D:
    return Rect2D(
        x=max(0.0, rect.x - pad),
        y=max(0.0, rect.y - pad),
        width=rect.width + 2.0 * pad,
        height=rect.height + 2.0 * pad,
    )


def _wall_clearance_box(wall: WallSegment, pad: float) -> Rect2D:
    min_x = min(wall.start.x, wall.end.x) - wall.thickness / 2.0 - pad
    min_y = min(wall.start.y, wall.end.y) - wall.thickness / 2.0 - pad
    max_x = max(wall.start.x, wall.end.x) + wall.thickness / 2.0 + pad
    max_y = max(wall.start.y, wall.end.y) + wall.thickness / 2.0 + pad
    return Rect2D(
        x=max(0.0, min_x),
        y=max(0.0, min_y),
        width=max(0.1, max_x - max(0.0, min_x)),
        height=max(0.1, max_y - max(0.0, min_y)),
    )


def _point_clearance_box(point: Point2D, half_span: float) -> Rect2D:
    x0 = max(0.0, point.x - half_span)
    y0 = max(0.0, point.y - half_span)
    return Rect2D(
        x=x0,
        y=y0,
        width=max(0.1, (point.x + half_span) - x0),
        height=max(0.1, (point.y + half_span) - y0),
    )


def _rotate_point(point: Point2D, width: float, height: float, angle: Literal[0, 90, 180, 270]) -> Point2D:
    if angle == 0:
        return point
    if angle == 90:
        return Point2D(x=_snap_half(height - point.y), y=_snap_half(point.x))
    if angle == 180:
        return Point2D(x=_snap_half(width - point.x), y=_snap_half(height - point.y))
    return Point2D(x=_snap_half(point.y), y=_snap_half(width - point.x))


def _rotate_rect(rect: Rect2D, width: float, height: float, angle: Literal[0, 90, 180, 270]) -> Rect2D:
    if angle == 0:
        return rect
    if angle == 90:
        return Rect2D(
            x=_snap_half(height - rect.y2),
            y=_snap_half(rect.x),
            width=rect.height,
            height=rect.width,
        )
    if angle == 180:
        return Rect2D(
            x=_snap_half(width - rect.x2),
            y=_snap_half(height - rect.y2),
            width=rect.width,
            height=rect.height,
        )
    return Rect2D(
        x=_snap_half(rect.y),
        y=_snap_half(width - rect.x2),
        width=rect.height,
        height=rect.width,
    )


def _rotate_door_orientation(
    orientation: Literal["horizontal", "vertical"],
    angle: Literal[0, 90, 180, 270],
) -> Literal["horizontal", "vertical"]:
    if angle in (90, 270):
        return "vertical" if orientation == "horizontal" else "horizontal"
    return orientation


class WarehouseSceneGenerator:
    """Generates valid, reproducible 2D warehouse SceneSpec instances."""

    def generate(
        self,
        variation: VariationConfig,
        scene_id: str | None = None,
    ) -> SceneSpec:
        """Build a complete, validated SceneSpec from a VariationConfig."""
        rng = _stable_rng(variation)
        base_w, base_h = _CANVAS_BY_DIFFICULTY[variation.difficulty]
        resolved_scene_id = scene_id or f"scene_{variation.topology_id}_{variation.seed:04d}"

        inner_x0 = _WALL_MARGIN
        inner_y0 = _WALL_MARGIN
        inner_x1 = base_w - _WALL_MARGIN
        inner_y1 = base_h - _WALL_MARGIN

        walls: list[WallSegment] = [
            WallSegment(
                id="W_BOUND_N",
                start=Point2D(x=inner_x0, y=inner_y0),
                end=Point2D(x=inner_x1, y=inner_y0),
                thickness=_WALL_THICKNESS,
            ),
            WallSegment(
                id="W_BOUND_S",
                start=Point2D(x=inner_x0, y=inner_y1),
                end=Point2D(x=inner_x1, y=inner_y1),
                thickness=_WALL_THICKNESS,
            ),
            WallSegment(
                id="W_BOUND_W",
                start=Point2D(x=inner_x0, y=inner_y0),
                end=Point2D(x=inner_x0, y=inner_y1),
                thickness=_WALL_THICKNESS,
            ),
            WallSegment(
                id="W_BOUND_E",
                start=Point2D(x=inner_x1, y=inner_y0),
                end=Point2D(x=inner_x1, y=inner_y1),
                thickness=_WALL_THICKNESS,
            ),
        ]

        # 1. Partition interior walls and consistent doorways by topology.
        if variation.topology_id == "warehouse_corridor_spine_v1":
            split_x, split_y, interior_walls, doors, zone_regions = self._build_corridor_spine_layout(
                rng=rng,
                variation=variation,
                base_w=base_w,
                base_h=base_h,
            )
            walls.extend(interior_walls)
        elif variation.topology_id == "warehouse_cross_aisle_v1":
            split_x, split_y, interior_walls, doors, zone_regions = self._build_cross_aisle_layout(
                rng=rng,
                variation=variation,
                base_w=base_w,
                base_h=base_h,
            )
            walls.extend(interior_walls)
        else:
            split_x, split_y, interior_walls, doors, zone_regions = self._build_split_aisle_layout(
                rng=rng,
                variation=variation,
                base_w=base_w,
                base_h=base_h,
            )
            walls.extend(interior_walls)

        # 2. Place perimeter exits (at least 2; 3 for medium/hard) avoiding partition joints.
        exits = self._generate_exits(
            rng=rng,
            variation=variation,
            base_w=base_w,
            base_h=base_h,
            split_x=split_x,
            split_y=split_y,
        )

        # 3. Place semantic zones (storage, loading/staging, and >=1 hazard/restricted zone).
        zones = self._generate_zones(
            rng=rng,
            variation=variation,
            base_w=base_w,
            base_h=base_h,
            split_x=split_x,
            split_y=split_y,
            walls=walls,
            doors=doors,
            exits=exits,
            candidate_regions=zone_regions,
        )

        # 4. Place storage racks and obstacles without colliding with walls, doors, exits, or each other.
        entities = self._generate_entities(
            rng=rng,
            variation=variation,
            base_w=base_w,
            base_h=base_h,
            walls=walls,
            doors=doors,
            exits=exits,
            zones=zones,
        )

        # 5. Apply geometric orientation rotation across all coordinates.
        angle = variation.orientation_deg
        out_w = base_h if angle in (90, 270) else base_w
        out_h = base_w if angle in (90, 270) else base_h

        rotated_walls = [
            WallSegment(
                id=w.id,
                start=_rotate_point(w.start, base_w, base_h, angle),
                end=_rotate_point(w.end, base_w, base_h, angle),
                thickness=w.thickness,
            )
            for w in walls
        ]
        rotated_doors = [
            Doorway(
                id=d.id,
                position=_rotate_point(d.position, base_w, base_h, angle),
                width=d.width,
                orientation=_rotate_door_orientation(d.orientation, angle),
                is_blocked=d.is_blocked,
            )
            for d in doors
        ]
        rotated_exits = [
            ExitMarker(
                id=e.id,
                label=e.label,
                position=_rotate_point(e.position, base_w, base_h, angle),
                width=e.width,
                is_emergency_exit=e.is_emergency_exit,
            )
            for e in exits
        ]
        rotated_zones = [
            Zone(
                id=z.id,
                label=z.label,
                zone_type=z.zone_type,
                bounds=_rotate_rect(z.bounds, base_w, base_h, angle),
            )
            for z in zones
        ]
        rotated_entities = [
            SceneEntity(
                id=ent.id,
                label=ent.label,
                entity_type=ent.entity_type,
                bounds=_rotate_rect(ent.bounds, base_w, base_h, angle),
                passable=ent.passable,
            )
            for ent in entities
        ]

        scale_world_units = 4.0
        scale_bar = ScaleBarSpec(
            world_units=scale_world_units,
            display_meters=round(scale_world_units * variation.scale_meters_per_unit, 3),
        )

        return SceneSpec(
            scene_id=resolved_scene_id,
            width=out_w,
            height=out_h,
            grid_resolution=0.5,
            scale_bar=scale_bar,
            variation=variation,
            walls=rotated_walls,
            doors=rotated_doors,
            zones=rotated_zones,
            exits=rotated_exits,
            entities=rotated_entities,
        )

    def _build_split_aisle_layout(
        self,
        rng: random.Random,
        variation: VariationConfig,
        base_w: float,
        base_h: float,
    ) -> tuple[
        float,
        float,
        list[WallSegment],
        list[Doorway],
        list[tuple[float, float, float, float]],
    ]:
        inner_x0 = _WALL_MARGIN
        inner_y0 = _WALL_MARGIN
        inner_x1 = base_w - _WALL_MARGIN
        inner_y1 = base_h - _WALL_MARGIN

        split_x = _snap_half(rng.uniform(base_w * 0.36, base_w * 0.62))
        split_y = _snap_half(rng.uniform(base_h * 0.38, base_h * 0.64))

        interior_walls: list[WallSegment] = [
            WallSegment(
                id="W_INT_V1",
                start=Point2D(x=split_x, y=inner_y0),
                end=Point2D(x=split_x, y=inner_y1),
                thickness=_WALL_THICKNESS,
            )
        ]
        door_v1_y = _snap_half(rng.uniform(inner_y0 + 3.5, split_y - 2.5))
        doors: list[Doorway] = [
            Doorway(
                id="D1",
                position=Point2D(x=split_x, y=door_v1_y),
                width=_DOOR_WIDTH,
                orientation="vertical",
                is_blocked=False,
            )
        ]

        if variation.difficulty in (DifficultyTier.MEDIUM, DifficultyTier.HARD):
            interior_walls.append(
                WallSegment(
                    id="W_INT_H1",
                    start=Point2D(x=split_x, y=split_y),
                    end=Point2D(x=inner_x1, y=split_y),
                    thickness=_WALL_THICKNESS,
                )
            )
            door_h1_x = _snap_half(rng.uniform(split_x + 3.5, inner_x1 - 3.5))
            doors.append(
                Doorway(
                    id="D2",
                    position=Point2D(x=door_h1_x, y=split_y),
                    width=_DOOR_WIDTH,
                    orientation="horizontal",
                    is_blocked=False,
                )
            )
            door_v2_y = _snap_half(rng.uniform(split_y + 2.5, inner_y1 - 3.5))
            doors.append(
                Doorway(
                    id="D3",
                    position=Point2D(x=split_x, y=door_v2_y),
                    width=_DOOR_WIDTH,
                    orientation="vertical",
                    is_blocked=(
                        variation.difficulty == DifficultyTier.HARD and rng.random() < 0.5
                    ),
                )
            )

        if variation.difficulty == DifficultyTier.HARD:
            left_split_y = _snap_half(rng.uniform(base_h * 0.35, base_h * 0.65))
            interior_walls.append(
                WallSegment(
                    id="W_INT_H2",
                    start=Point2D(x=inner_x0, y=left_split_y),
                    end=Point2D(x=split_x, y=left_split_y),
                    thickness=_WALL_THICKNESS,
                )
            )
            door_h2_x = _snap_half(rng.uniform(inner_x0 + 3.5, split_x - 3.5))
            doors.append(
                Doorway(
                    id="D4",
                    position=Point2D(x=door_h2_x, y=left_split_y),
                    width=_DOOR_WIDTH,
                    orientation="horizontal",
                    is_blocked=False,
                )
            )

        regions = [
            (_WALL_MARGIN + 1.5, _WALL_MARGIN + 1.5, split_x - 1.5, base_h - _WALL_MARGIN - 1.5),
            (split_x + 1.5, _WALL_MARGIN + 1.5, base_w - _WALL_MARGIN - 1.5, split_y - 1.5),
            (split_x + 1.5, split_y + 1.5, base_w - _WALL_MARGIN - 1.5, base_h - _WALL_MARGIN - 1.5),
        ]
        return split_x, split_y, interior_walls, doors, regions

    def _build_corridor_spine_layout(
        self,
        rng: random.Random,
        variation: VariationConfig,
        base_w: float,
        base_h: float,
    ) -> tuple[
        float,
        float,
        list[WallSegment],
        list[Doorway],
        list[tuple[float, float, float, float]],
    ]:
        """Central horizontal spine corridor separating North and South facility wings."""
        inner_x0 = _WALL_MARGIN
        inner_y0 = _WALL_MARGIN
        inner_x1 = base_w - _WALL_MARGIN
        inner_y1 = base_h - _WALL_MARGIN

        corridor_h = _snap_half(rng.uniform(3.5, 4.5))
        cor_y0 = _snap_half(rng.uniform(base_h * 0.36, base_h * 0.46))
        cor_y1 = _snap_half(cor_y0 + corridor_h)
        split_x = _snap_half(rng.uniform(base_w * 0.38, base_w * 0.62))
        split_y = _snap_half((cor_y0 + cor_y1) / 2.0)

        interior_walls: list[WallSegment] = [
            WallSegment(
                id="W_INT_SPINE_N",
                start=Point2D(x=inner_x0 + 3.0, y=cor_y0),
                end=Point2D(x=inner_x1 - 3.0, y=cor_y0),
                thickness=_WALL_THICKNESS,
            ),
            WallSegment(
                id="W_INT_SPINE_S",
                start=Point2D(x=inner_x0 + 3.0, y=cor_y1),
                end=Point2D(x=inner_x1 - 3.0, y=cor_y1),
                thickness=_WALL_THICKNESS,
            ),
        ]

        door_n_x = _snap_half(rng.uniform(inner_x0 + 5.5, split_x - 2.0))
        door_s_x = _snap_half(rng.uniform(split_x + 2.0, inner_x1 - 5.5))
        doors: list[Doorway] = [
            Doorway(
                id="D1",
                position=Point2D(x=door_n_x, y=cor_y0),
                width=_DOOR_WIDTH,
                orientation="horizontal",
                is_blocked=False,
            ),
            Doorway(
                id="D2",
                position=Point2D(x=door_s_x, y=cor_y1),
                width=_DOOR_WIDTH,
                orientation="horizontal",
                is_blocked=False,
            ),
        ]

        if variation.difficulty in (DifficultyTier.MEDIUM, DifficultyTier.HARD):
            interior_walls.append(
                WallSegment(
                    id="W_INT_WING_N",
                    start=Point2D(x=split_x, y=inner_y0),
                    end=Point2D(x=split_x, y=cor_y0),
                    thickness=_WALL_THICKNESS,
                )
            )
            door_wn_y = _snap_half(rng.uniform(inner_y0 + 2.5, cor_y0 - 2.5))
            doors.append(
                Doorway(
                    id="D3",
                    position=Point2D(x=split_x, y=door_wn_y),
                    width=_DOOR_WIDTH,
                    orientation="vertical",
                    is_blocked=False,
                )
            )
            # Second door into North wing right half.
            door_n2_x = _snap_half(rng.uniform(split_x + 2.5, inner_x1 - 5.5))
            doors.append(
                Doorway(
                    id="D4",
                    position=Point2D(x=door_n2_x, y=cor_y0),
                    width=_DOOR_WIDTH,
                    orientation="horizontal",
                    is_blocked=(
                        variation.difficulty == DifficultyTier.HARD and rng.random() < 0.4
                    ),
                )
            )

        if variation.difficulty == DifficultyTier.HARD:
            interior_walls.append(
                WallSegment(
                    id="W_INT_WING_S",
                    start=Point2D(x=split_x, y=cor_y1),
                    end=Point2D(x=split_x, y=inner_y1),
                    thickness=_WALL_THICKNESS,
                )
            )
            door_ws_y = _snap_half(rng.uniform(cor_y1 + 2.5, inner_y1 - 2.5))
            doors.append(
                Doorway(
                    id="D5",
                    position=Point2D(x=split_x, y=door_ws_y),
                    width=_DOOR_WIDTH,
                    orientation="vertical",
                    is_blocked=False,
                )
            )

        regions = [
            (inner_x0 + 1.5, inner_y0 + 1.5, split_x - 1.5, cor_y0 - 1.5),
            (split_x + 1.5, inner_y0 + 1.5, inner_x1 - 1.5, cor_y0 - 1.5),
            (inner_x0 + 1.5, cor_y1 + 1.5, inner_x1 - 1.5, inner_y1 - 1.5),
        ]
        return split_x, split_y, interior_walls, doors, regions

    def _build_cross_aisle_layout(
        self,
        rng: random.Random,
        variation: VariationConfig,
        base_w: float,
        base_h: float,
    ) -> tuple[
        float,
        float,
        list[WallSegment],
        list[Doorway],
        list[tuple[float, float, float, float]],
    ]:
        """Four-quadrant cross-partition layout with spanning-tree doorway connectivity."""
        inner_x0 = _WALL_MARGIN
        inner_y0 = _WALL_MARGIN
        inner_x1 = base_w - _WALL_MARGIN
        inner_y1 = base_h - _WALL_MARGIN

        split_x = _snap_half(rng.uniform(base_w * 0.40, base_w * 0.60))
        split_y = _snap_half(rng.uniform(base_h * 0.40, base_h * 0.60))

        interior_walls: list[WallSegment] = [
            WallSegment(
                id="W_INT_V1",
                start=Point2D(x=split_x, y=inner_y0),
                end=Point2D(x=split_x, y=inner_y1),
                thickness=_WALL_THICKNESS,
            ),
            WallSegment(
                id="W_INT_H1",
                start=Point2D(x=inner_x0, y=split_y),
                end=Point2D(x=inner_x1, y=split_y),
                thickness=_WALL_THICKNESS,
            ),
        ]

        # Four doorways (N, S, W, E arms); keep at least 3 open to guarantee quadrant connectivity.
        door_n_y = _snap_half(rng.uniform(inner_y0 + 2.5, split_y - 2.5))
        door_s_y = _snap_half(rng.uniform(split_y + 2.5, inner_y1 - 2.5))
        door_w_x = _snap_half(rng.uniform(inner_x0 + 2.5, split_x - 2.5))
        door_e_x = _snap_half(rng.uniform(split_x + 2.5, inner_x1 - 2.5))

        block_fourth = variation.difficulty in (DifficultyTier.MEDIUM, DifficultyTier.HARD) and rng.random() < 0.6
        blocked_arm = rng.randint(0, 3) if block_fourth else -1

        arm_specs: list[tuple[Point2D, Literal["horizontal", "vertical"], bool]] = [
            (Point2D(x=split_x, y=door_n_y), "vertical", blocked_arm == 0),
            (Point2D(x=split_x, y=door_s_y), "vertical", blocked_arm == 1),
            (Point2D(x=door_w_x, y=split_y), "horizontal", blocked_arm == 2),
            (Point2D(x=door_e_x, y=split_y), "horizontal", blocked_arm == 3),
        ]
        doors = [
            Doorway(
                id=f"D{idx + 1}",
                position=pos,
                width=_DOOR_WIDTH,
                orientation=orient,
                is_blocked=is_blk,
            )
            for idx, (pos, orient, is_blk) in enumerate(arm_specs)
        ]

        regions = [
            (inner_x0 + 1.5, inner_y0 + 1.5, split_x - 1.5, split_y - 1.5),
            (split_x + 1.5, inner_y0 + 1.5, inner_x1 - 1.5, split_y - 1.5),
            (inner_x0 + 1.5, split_y + 1.5, split_x - 1.5, inner_y1 - 1.5),
            (split_x + 1.5, split_y + 1.5, inner_x1 - 1.5, inner_y1 - 1.5),
        ]
        return split_x, split_y, interior_walls, doors, regions

    def _generate_exits(
        self,
        rng: random.Random,
        variation: VariationConfig,
        base_w: float,
        base_h: float,
        split_x: float,
        split_y: float,
    ) -> list[ExitMarker]:
        inner_x0 = _WALL_MARGIN
        inner_y0 = _WALL_MARGIN
        inner_x1 = base_w - _WALL_MARGIN
        inner_y1 = base_h - _WALL_MARGIN

        candidates: list[tuple[str, Point2D]] = [
            (
                "N_LEFT",
                Point2D(
                    x=_snap_half(rng.uniform(inner_x0 + 3.0, split_x - 3.0)),
                    y=inner_y0,
                ),
            ),
            (
                "E_TOP",
                Point2D(
                    x=inner_x1,
                    y=_snap_half(rng.uniform(inner_y0 + 3.0, split_y - 2.5)),
                ),
            ),
            (
                "S_RIGHT",
                Point2D(
                    x=_snap_half(rng.uniform(split_x + 3.0, inner_x1 - 3.0)),
                    y=inner_y1,
                ),
            ),
            (
                "W_BOTTOM",
                Point2D(
                    x=inner_x0,
                    y=_snap_half(rng.uniform(split_y + 2.5, inner_y1 - 3.0)),
                ),
            ),
        ]

        rng.shuffle(candidates)
        exit_count = 2 if variation.difficulty == DifficultyTier.EASY else 3
        if variation.difficulty == DifficultyTier.HARD and rng.random() < 0.5:
            exit_count = 4

        exits: list[ExitMarker] = []
        for idx in range(exit_count):
            _, pos = candidates[idx]
            exit_id = f"E{idx + 1}"
            exits.append(
                ExitMarker(
                    id=exit_id,
                    label=f"Exit {exit_id}",
                    position=pos,
                    width=_EXIT_WIDTH,
                    is_emergency_exit=True,
                )
            )
        return exits

    def _generate_zones(
        self,
        rng: random.Random,
        variation: VariationConfig,
        base_w: float,
        base_h: float,
        split_x: float,
        split_y: float,
        walls: list[WallSegment],
        doors: list[Doorway],
        exits: list[ExitMarker],
        candidate_regions: list[tuple[float, float, float, float]] | None = None,
    ) -> list[Zone]:
        zones: list[Zone] = []
        protected: list[Rect2D] = [
            _wall_clearance_box(w, 0.8) for w in walls
        ] + [
            _point_clearance_box(d.position, 2.2) for d in doors
        ] + [
            _point_clearance_box(e.position, 2.5) for e in exits
        ]

        def _place_zone(
            zone_id: str,
            label: str,
            zone_type: ZoneType,
            region: tuple[float, float, float, float],
            min_w: float,
            max_w: float,
            min_h: float,
            max_h: float,
        ) -> Zone | None:
            rx0, ry0, rx1, ry1 = region
            avail_w = rx1 - rx0
            avail_h = ry1 - ry0
            if avail_w < min_w or avail_h < min_h:
                return None

            for _ in range(40):
                zw = _snap_half(rng.uniform(min_w, min(max_w, avail_w)))
                zh = _snap_half(rng.uniform(min_h, min(max_h, avail_h)))
                zx = _snap_half(rng.uniform(rx0, rx1 - zw))
                zy = _snap_half(rng.uniform(ry0, ry1 - zh))
                rect = Rect2D(x=zx, y=zy, width=max(1.0, zw), height=max(1.0, zh))
                if any(rect.overlaps_interior(p) for p in protected):
                    continue
                if any(rect.overlaps_interior(_rect_padded(z.bounds, 0.5)) for z in zones):
                    continue
                return Zone(id=zone_id, label=label, zone_type=zone_type, bounds=rect)
            return None

        if candidate_regions and len(candidate_regions) >= 3:
            left_region = candidate_regions[0]
            right_top_region = candidate_regions[1]
            right_bot_region = candidate_regions[2]
        else:
            left_region = (_WALL_MARGIN + 1.5, _WALL_MARGIN + 1.5, split_x - 1.5, base_h - _WALL_MARGIN - 1.5)
            right_top_region = (split_x + 1.5, _WALL_MARGIN + 1.5, base_w - _WALL_MARGIN - 1.5, split_y - 1.5)
            right_bot_region = (split_x + 1.5, split_y + 1.5, base_w - _WALL_MARGIN - 1.5, base_h - _WALL_MARGIN - 1.5)

        # Storage zone (always present so counting tasks have a primary storage region).
        storage_region = right_top_region if rng.random() < 0.5 else left_region
        storage_zone = _place_zone(
            "Z_STORAGE_1",
            "Storage Bay A",
            ZoneType.STORAGE,
            storage_region,
            min_w=6.0,
            max_w=12.0,
            min_h=4.5,
            max_h=9.0,
        )
        if storage_zone is None and candidate_regions:
            for alt_reg in candidate_regions:
                storage_zone = _place_zone(
                    "Z_STORAGE_1",
                    "Storage Bay A",
                    ZoneType.STORAGE,
                    alt_reg,
                    min_w=5.0,
                    max_w=11.0,
                    min_h=4.0,
                    max_h=8.0,
                )
                if storage_zone is not None:
                    break
        if storage_zone is None:
            storage_zone = Zone(
                id="Z_STORAGE_1",
                label="Storage Bay A",
                zone_type=ZoneType.STORAGE,
                bounds=Rect2D(
                    x=_snap_half(split_x + 2.0),
                    y=_snap_half(_WALL_MARGIN + 2.0),
                    width=_snap_half(max(5.0, (base_w - split_x) - 5.0)),
                    height=_snap_half(max(4.0, (split_y - _WALL_MARGIN) - 4.0)),
                ),
            )
        zones.append(storage_zone)

        # At least one hazard or restricted zone.
        target_hazards = max(1, variation.min_hazard_zones)
        hazard_regions = list(candidate_regions) if candidate_regions else [right_bot_region, left_region, right_top_region]
        rng.shuffle(hazard_regions)

        hazard_idx = 0
        for reg in hazard_regions:
            if hazard_idx >= target_hazards:
                break
            z_type = ZoneType.HAZARD if hazard_idx % 2 == 0 else ZoneType.RESTRICTED
            z_label = f"Hazard Zone {hazard_idx + 1}" if z_type == ZoneType.HAZARD else f"Restricted Zone {hazard_idx + 1}"
            placed = _place_zone(
                f"Z_HAZARD_{hazard_idx + 1}",
                z_label,
                z_type,
                reg,
                min_w=3.5,
                max_w=7.5,
                min_h=3.0,
                max_h=6.0,
            )
            if placed is not None:
                zones.append(placed)
                hazard_idx += 1

        # Deterministic fallback if random attempts could not place the mandatory first hazard zone.
        if hazard_idx == 0:
            fallback_x = _snap_half(max(_WALL_MARGIN + 2.0, split_x - 7.0))
            fallback_y = _snap_half(max(_WALL_MARGIN + 2.0, base_h - _WALL_MARGIN - 6.0))
            fallback_rect = Rect2D(x=fallback_x, y=fallback_y, width=4.5, height=3.5)
            zones.append(
                Zone(
                    id="Z_HAZARD_1",
                    label="Hazard Zone 1",
                    zone_type=ZoneType.HAZARD,
                    bounds=fallback_rect,
                )
            )

        # Optional staging/loading zone on medium/hard scenes.
        if variation.difficulty != DifficultyTier.EASY:
            staging = _place_zone(
                "Z_STAGING_1",
                "Staging Area",
                ZoneType.STAGING,
                left_region,
                min_w=4.0,
                max_w=7.0,
                min_h=3.5,
                max_h=5.5,
            )
            if staging is not None:
                zones.append(staging)

        return zones

    def _generate_entities(
        self,
        rng: random.Random,
        variation: VariationConfig,
        base_w: float,
        base_h: float,
        walls: list[WallSegment],
        doors: list[Doorway],
        exits: list[ExitMarker],
        zones: list[Zone],
    ) -> list[SceneEntity]:
        min_obs = variation.min_obstacles
        max_obs = variation.max_obstacles
        if max_obs == 0:
            return []

        span = max_obs - min_obs
        if variation.difficulty == DifficultyTier.EASY:
            base_target = min_obs + round(span * 0.25)
        elif variation.difficulty == DifficultyTier.MEDIUM:
            base_target = min_obs + round(span * 0.55)
        else:
            base_target = min_obs + round(span * 0.8)

        target_count = min(
            max_obs,
            max(min_obs, base_target + _CLUTTER_EXTRA[variation.clutter_level] - 1),
        )
        if span > 0:
            target_count = min(max_obs, max(min_obs, target_count + rng.randint(-1, 1)))

        blocked_regions: list[Rect2D] = (
            [_wall_clearance_box(w, 1.0) for w in walls]
            + [_point_clearance_box(d.position, 2.3) for d in doors]
            + [_point_clearance_box(e.position, 2.5) for e in exits]
            + [
                _rect_padded(z.bounds, 0.3)
                for z in zones
                if z.zone_type in (ZoneType.HAZARD, ZoneType.RESTRICTED)
            ]
        )

        entities: list[SceneEntity] = []
        storage_zones = [z for z in zones if z.zone_type == ZoneType.STORAGE]

        def _can_place(rect: Rect2D) -> bool:
            if (
                rect.x < _WALL_MARGIN + 1.0
                or rect.y < _WALL_MARGIN + 1.0
                or rect.x2 > base_w - _WALL_MARGIN - 1.0
                or rect.y2 > base_h - _WALL_MARGIN - 1.0
            ):
                return False
            if any(rect.overlaps_interior(b) for b in blocked_regions):
                return False
            if any(rect.overlaps_interior(_rect_padded(ent.bounds, 0.5)) for ent in entities):
                return False
            return True

        # Place pallet racks inside the storage zone first when space allows.
        rack_goal = min(target_count, max(1, target_count // 2)) if storage_zones else 0
        if storage_zones and rack_goal > 0:
            sz = storage_zones[0].bounds
            for _ in range(rack_goal * 20):
                if len(entities) >= rack_goal:
                    break
                rw, rh = rng.choice([(3.5, 1.5), (1.5, 3.5), (3.0, 2.0)])
                if sz.width < rw + 0.5 or sz.height < rh + 0.5:
                    continue
                rx = _snap_half(rng.uniform(sz.x + 0.25, sz.x2 - rw - 0.25))
                ry = _snap_half(rng.uniform(sz.y + 0.25, sz.y2 - rh - 0.25))
                rect = Rect2D(x=rx, y=ry, width=rw, height=rh)
                if _can_place(rect):
                    idx = len(entities) + 1
                    entities.append(
                        SceneEntity(
                            id=f"ENT_{idx}",
                            label=f"Rack {idx}",
                            entity_type=EntityType.PALLET_RACK,
                            bounds=rect,
                            passable=False,
                        )
                    )

        obstacle_specs: list[tuple[EntityType, str, float, float]] = [
            (EntityType.PALLET_RACK, "Rack", 3.5, 1.5),
            (EntityType.CRATE, "Crate", 1.5, 1.5),
            (EntityType.FORKLIFT, "Forklift", 2.0, 1.5),
            (EntityType.PILLAR, "Pillar", 1.0, 1.0),
            (EntityType.CONVEYOR, "Conveyor", 4.0, 1.5),
            (EntityType.ROBOT, "AMR", 1.5, 1.5),
        ]

        # Randomized floor placement up to target_count.
        for _ in range(target_count * 40):
            if len(entities) >= target_count:
                break
            e_type, prefix, ew, eh = rng.choice(obstacle_specs)
            if rng.random() < 0.5:
                ew, eh = eh, ew
            ex = _snap_half(rng.uniform(_WALL_MARGIN + 1.5, base_w - _WALL_MARGIN - ew - 1.5))
            ey = _snap_half(rng.uniform(_WALL_MARGIN + 1.5, base_h - _WALL_MARGIN - eh - 1.5))
            rect = Rect2D(x=ex, y=ey, width=ew, height=eh)
            if _can_place(rect):
                idx = len(entities) + 1
                entities.append(
                    SceneEntity(
                        id=f"ENT_{idx}",
                        label=f"{prefix} {idx}",
                        entity_type=e_type,
                        bounds=rect,
                        passable=False,
                    )
                )

        # Deterministic grid-scan fallback if min_obstacles has not yet been reached.
        if len(entities) < min_obs:
            step = 2.0
            y_cursor = _WALL_MARGIN + 1.5
            while y_cursor + 1.5 <= base_h - _WALL_MARGIN - 1.5 and len(entities) < min_obs:
                x_cursor = _WALL_MARGIN + 1.5
                while x_cursor + 1.5 <= base_w - _WALL_MARGIN - 1.5 and len(entities) < min_obs:
                    rect = Rect2D(x=_snap_half(x_cursor), y=_snap_half(y_cursor), width=1.5, height=1.5)
                    if _can_place(rect):
                        idx = len(entities) + 1
                        entities.append(
                            SceneEntity(
                                id=f"ENT_{idx}",
                                label=f"Crate {idx}",
                                entity_type=EntityType.CRATE,
                                bounds=rect,
                                passable=False,
                            )
                        )
                    x_cursor += step
                y_cursor += step

        return entities


def generate_scene(
    variation: VariationConfig,
    scene_id: str | None = None,
) -> SceneSpec:
    """Convenience entry point to generate a SceneSpec for a VariationConfig."""
    return WarehouseSceneGenerator().generate(variation=variation, scene_id=scene_id)
