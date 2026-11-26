"""Deterministic CPU-only 3D geometric solver for SceneSpec3D."""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from visual_task_factory.models.scene3d import (
    Box3D,
    Doorway3D,
    SceneEntity3D,
    SceneSpec3D,
    WallSegment3D,
)

_EPS = 1e-6


class Solver3DError(ValueError):
    """Base error raised by the deterministic 3D geometric solver."""


class UnsupportedTask3DError(Solver3DError):
    """Raised when a query references unsupported 3D geometry."""


class UnsolvableTask3DError(Solver3DError):
    """Raised when a queried scene object cannot be resolved."""


class BoxOverlapRelation(str, Enum):
    """Exact 3D relationship between two axis-aligned boxes."""

    INTERIOR_OVERLAP = "interior_overlap"
    BOUNDARY_TOUCHING = "boundary_touching"
    DISJOINT = "disjoint"


class VerticalRelation(str, Enum):
    """Vertical Z-axis relation between a source box and target box."""

    STRICTLY_ABOVE = "strictly_above"
    FLUSH_ABOVE = "flush_above"
    STRICTLY_BELOW = "strictly_below"
    FLUSH_BELOW = "flush_below"
    EQUAL_HEIGHT = "equal_height"
    OVERLAPPING_HEIGHT = "overlapping_height"


class VerticalClearanceStatus(str, Enum):
    """Status describing whether vertical clearance is geometrically well-defined."""

    DEFINED = "defined"
    OVERLAPPING_HEIGHTS = "overlapping_heights"
    NO_XY_OVERLAP = "no_xy_overlap"
    SAME_OBJECT = "same_object"


class VisibilityStatus(str, Enum):
    """Line-of-sight status from SceneSpec3D.camera.eye toward a target 3D box."""

    CLEAR = "clear"
    PARTIALLY_OCCLUDED = "partially_occluded"
    FULLY_OCCLUDED = "fully_occluded"


class BoxOverlapResult(BaseModel):
    """Result of evaluating 3D bounding-box overlap and boundary contact."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    relation: BoxOverlapRelation
    overlaps_3d: bool
    touches_3d: bool
    overlaps_xy: bool
    intersection_dimensions: tuple[float, float, float]
    overlap_volume: float = Field(..., ge=0.0)
    surface_distance_units: float = Field(..., ge=0.0)
    surface_distance_meters: float = Field(..., ge=0.0)
    touching_axes: tuple[Literal["x", "y", "z"], ...]


class VerticalRelationResult(BaseModel):
    """Result of evaluating vertical placement between two 3D boxes."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    relation: VerticalRelation
    is_above: bool
    is_below: bool
    is_flush_contact: bool
    heights_overlap: bool
    xy_overlap: bool
    source_z_range: tuple[float, float]
    target_z_range: tuple[float, float]
    vertical_gap_units: float | None = Field(default=None, ge=0.0)
    vertical_overlap_units: float = Field(..., ge=0.0)
    is_directly_supported: bool = False


class VerticalClearanceResult(BaseModel):
    """Result of evaluating meaningful vertical clearance between two 3D objects."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    is_defined: bool
    status: VerticalClearanceStatus
    clearance_units: float | None = Field(default=None, ge=0.0)
    clearance_meters: float | None = Field(default=None, ge=0.0)
    reason: str


class CameraVisibilityResult(BaseModel):
    """Deterministic sampled ray-AABB line-of-sight result from CameraPose3D.eye."""

    model_config = ConfigDict(extra="forbid")

    target_id: str
    camera_eye: tuple[float, float, float]
    status: VisibilityStatus
    center_visible: bool
    visible_sample_count: int = Field(..., ge=0)
    total_sample_count: int = Field(..., ge=1)
    visible_fraction: float = Field(..., ge=0.0, le=1.0)
    within_fov_cone: bool
    blocking_object_ids: list[str]
    visibility_model: Literal[
        "orthographic_isometric_sampled_raycast",
        "perspective_camera_sampled_raycast",
    ] = "orthographic_isometric_sampled_raycast"
    limitation_note: str


def _validate_box3d(box: Box3D, name: str) -> None:
    if not isinstance(box, Box3D):
        raise TypeError(f"Expected Box3D for {name}, got {type(box).__name__}.")


def check_box_overlap_3d(
    a: Box3D,
    b: Box3D,
    *,
    source_id: str = "box_a",
    target_id: str = "box_b",
    meters_per_world_unit: float = 1.0,
    eps: float = _EPS,
) -> BoxOverlapResult:
    """Classify 3D overlap between two axis-aligned boxes in X, Y, and Z.

    Semantics:
    - ``INTERIOR_OVERLAP``: positive overlap strictly > ``eps`` along X, Y, and Z.
    - ``BOUNDARY_TOUCHING``: closed-box intervals meet along all 3 axes (within ``eps``),
      with at least one axis touching only at the boundary (zero overlap volume).
    - ``DISJOINT``: strictly separated along at least one axis (> ``eps`` gap).
    2D XY overlap alone never produces ``INTERIOR_OVERLAP`` or ``BOUNDARY_TOUCHING``
    when the boxes are separated along Z.
    """
    _validate_box3d(a, "a")
    _validate_box3d(b, "b")
    if not math.isfinite(meters_per_world_unit) or meters_per_world_unit <= 0.0:
        raise ValueError("meters_per_world_unit must be finite and > 0.")

    dx = min(a.x2, b.x2) - max(a.x, b.x)
    dy = min(a.y2, b.y2) - max(a.y, b.y)
    dz = min(a.z2, b.z2) - max(a.z, b.z)

    overlaps_xy = dx > eps and dy > eps

    gap_x = max(0.0, max(a.x - b.x2, b.x - a.x2))
    gap_y = max(0.0, max(a.y - b.y2, b.y - a.y2))
    gap_z = max(0.0, max(a.z - b.z2, b.z - a.z2))
    dist_units = math.sqrt(gap_x * gap_x + gap_y * gap_y + gap_z * gap_z)

    if dx > eps and dy > eps and dz > eps:
        relation = BoxOverlapRelation.INTERIOR_OVERLAP
        touching_axes: tuple[Literal["x", "y", "z"], ...] = ()
        ix, iy, iz = dx, dy, dz
        volume = ix * iy * iz
        dist_units = 0.0
    elif dx >= -eps and dy >= -eps and dz >= -eps:
        relation = BoxOverlapRelation.BOUNDARY_TOUCHING
        axes: list[Literal["x", "y", "z"]] = []
        if abs(dx) <= eps:
            axes.append("x")
        if abs(dy) <= eps:
            axes.append("y")
        if abs(dz) <= eps:
            axes.append("z")
        touching_axes = tuple(axes)
        ix, iy, iz = max(0.0, dx), max(0.0, dy), max(0.0, dz)
        volume = 0.0
        dist_units = 0.0
    else:
        relation = BoxOverlapRelation.DISJOINT
        touching_axes = ()
        ix, iy, iz = max(0.0, dx), max(0.0, dy), max(0.0, dz)
        volume = 0.0

    dist_rounded = round(dist_units, 6)
    return BoxOverlapResult(
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        overlaps_3d=(relation == BoxOverlapRelation.INTERIOR_OVERLAP),
        touches_3d=(relation == BoxOverlapRelation.BOUNDARY_TOUCHING),
        overlaps_xy=overlaps_xy,
        intersection_dimensions=(round(ix, 6), round(iy, 6), round(iz, 6)),
        overlap_volume=round(volume, 9),
        surface_distance_units=dist_rounded,
        surface_distance_meters=round(dist_units * meters_per_world_unit, 6),
        touching_axes=touching_axes,
    )


def analyze_vertical_relation(
    a: Box3D,
    b: Box3D,
    *,
    source_id: str = "box_a",
    target_id: str = "box_b",
    is_directly_supported: bool = False,
    eps: float = _EPS,
) -> VerticalRelationResult:
    """Classify the 3D Z-axis relationship of source ``a`` relative to target ``b``."""
    _validate_box3d(a, "a")
    _validate_box3d(b, "b")

    xy_overlap = a.overlaps_xy(b, eps=eps)
    z_overlap = max(0.0, min(a.z2, b.z2) - max(a.z, b.z))

    if a.z >= b.z2 - eps:
        gap = max(0.0, a.z - b.z2)
        if gap <= eps:
            relation = VerticalRelation.FLUSH_ABOVE
            gap = 0.0
        else:
            relation = VerticalRelation.STRICTLY_ABOVE
        return VerticalRelationResult(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            is_above=True,
            is_below=False,
            is_flush_contact=(relation == VerticalRelation.FLUSH_ABOVE and xy_overlap),
            heights_overlap=False,
            xy_overlap=xy_overlap,
            source_z_range=(a.z, a.z2),
            target_z_range=(b.z, b.z2),
            vertical_gap_units=round(gap, 6),
            vertical_overlap_units=0.0,
            is_directly_supported=bool(is_directly_supported and relation == VerticalRelation.FLUSH_ABOVE and xy_overlap),
        )

    if b.z >= a.z2 - eps:
        gap = max(0.0, b.z - a.z2)
        if gap <= eps:
            relation = VerticalRelation.FLUSH_BELOW
            gap = 0.0
        else:
            relation = VerticalRelation.STRICTLY_BELOW
        return VerticalRelationResult(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            is_above=False,
            is_below=True,
            is_flush_contact=(relation == VerticalRelation.FLUSH_BELOW and xy_overlap),
            heights_overlap=False,
            xy_overlap=xy_overlap,
            source_z_range=(a.z, a.z2),
            target_z_range=(b.z, b.z2),
            vertical_gap_units=round(gap, 6),
            vertical_overlap_units=0.0,
            is_directly_supported=False,
        )

    if abs(a.z - b.z) <= eps and abs(a.z2 - b.z2) <= eps:
        relation = VerticalRelation.EQUAL_HEIGHT
    else:
        relation = VerticalRelation.OVERLAPPING_HEIGHT

    return VerticalRelationResult(
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        is_above=False,
        is_below=False,
        is_flush_contact=False,
        heights_overlap=True,
        xy_overlap=xy_overlap,
        source_z_range=(a.z, a.z2),
        target_z_range=(b.z, b.z2),
        vertical_gap_units=None,
        vertical_overlap_units=round(z_overlap, 6),
        is_directly_supported=False,
    )


def compute_vertical_clearance(
    a: Box3D,
    b: Box3D,
    *,
    source_id: str = "box_a",
    target_id: str = "box_b",
    meters_per_world_unit: float = 1.0,
    eps: float = _EPS,
) -> VerticalClearanceResult:
    """Compute vertical clearance between ``a`` and ``b`` only when well-defined.

    Vertical clearance is defined iff:
    1. ``a`` and ``b`` do NOT overlap along the Z axis, and
    2. Their XY footprints overlap (so one spans directly above the other).
    """
    _validate_box3d(a, "a")
    _validate_box3d(b, "b")
    if not math.isfinite(meters_per_world_unit) or meters_per_world_unit <= 0.0:
        raise ValueError("meters_per_world_unit must be finite and > 0.")

    if source_id == target_id and a == b:
        return VerticalClearanceResult(
            source_id=source_id,
            target_id=target_id,
            is_defined=False,
            status=VerticalClearanceStatus.SAME_OBJECT,
            reason="Vertical clearance is undefined between an object and itself.",
        )

    v_rel = analyze_vertical_relation(
        a, b, source_id=source_id, target_id=target_id, eps=eps
    )
    if v_rel.heights_overlap:
        return VerticalClearanceResult(
            source_id=source_id,
            target_id=target_id,
            is_defined=False,
            status=VerticalClearanceStatus.OVERLAPPING_HEIGHTS,
            reason=(
                f"Objects {source_id!r} and {target_id!r} have overlapping Z intervals "
                f"({v_rel.source_z_range} vs {v_rel.target_z_range}); vertical clearance is undefined."
            ),
        )

    if not v_rel.xy_overlap:
        return VerticalClearanceResult(
            source_id=source_id,
            target_id=target_id,
            is_defined=False,
            status=VerticalClearanceStatus.NO_XY_OVERLAP,
            reason=(
                f"Objects {source_id!r} and {target_id!r} have disjoint XY footprints; "
                "no direct vertical column exists between them."
            ),
        )

    assert v_rel.vertical_gap_units is not None
    clearance_units = round(v_rel.vertical_gap_units, 6)
    clearance_meters = round(clearance_units * meters_per_world_unit, 6)
    return VerticalClearanceResult(
        source_id=source_id,
        target_id=target_id,
        is_defined=True,
        status=VerticalClearanceStatus.DEFINED,
        clearance_units=clearance_units,
        clearance_meters=clearance_meters,
        reason="One object sits cleanly above the other over a shared XY footprint.",
    )


def ray_intersects_box3d(
    origin: tuple[float, float, float],
    target: tuple[float, float, float],
    box: Box3D,
    *,
    t_min: float = 1e-6,
    t_max: float = 1.0 - 1e-4,
) -> float | None:
    """Return earliest ray parameter ``t`` in [t_min, t_max] where segment hits interior of ``box``.

    Uses the 3D slab intersection algorithm on p(t) = origin + t * (target - origin).
    Purely grazing intersections where t_enter == t_exit do not count as occluding hits.
    """
    _validate_box3d(box, "box")
    t_low = t_min
    t_high = t_max

    bounds = (
        (origin[0], target[0] - origin[0], box.x, box.x2),
        (origin[1], target[1] - origin[1], box.y, box.y2),
        (origin[2], target[2] - origin[2], box.z, box.z2),
    )

    for orig, delta, b_min, b_max in bounds:
        if abs(delta) < 1e-12:
            if orig <= b_min + 1e-9 or orig >= b_max - 1e-9:
                return None
            continue
        inv = 1.0 / delta
        t0 = (b_min - orig) * inv
        t1 = (b_max - orig) * inv
        if t0 > t1:
            t0, t1 = t1, t0
        if t0 > t_low:
            t_low = t0
        if t1 < t_high:
            t_high = t1
        if t_low >= t_high - 1e-7:
            return None

    return t_low


def _wall_to_boxes_geometric(
    scene: SceneSpec3D,
    wall: WallSegment3D,
    doors: list[Doorway3D],
) -> list[Box3D]:
    """Convert an orthogonal WallSegment3D into physical Box3D spans cut by open doors."""
    half_t = wall.thickness / 2.0
    x1 = min(wall.start.x, wall.end.x)
    x2 = max(wall.start.x, wall.end.x)
    y1 = min(wall.start.y, wall.end.y)
    y2 = max(wall.start.y, wall.end.y)

    is_vertical = math.isclose(x1, x2, abs_tol=_EPS)
    is_horizontal = math.isclose(y1, y2, abs_tol=_EPS)

    if not is_vertical and not is_horizontal:
        raise UnsupportedTask3DError(
            f"Diagonal WallSegment3D {wall.id!r} cannot be represented exactly as an axis-aligned Box3D."
        )

    if is_horizontal:
        wall_const = y1
        span_start, span_end = x1, x2
        aligned = [
            d
            for d in doors
            if not d.is_blocked
            and d.orientation == "horizontal"
            and math.isclose(d.position.y, wall_const, abs_tol=max(half_t, 0.25))
        ]
        spans = [
            (
                max(span_start, d.position.x - d.width / 2.0),
                min(span_end, d.position.x + d.width / 2.0),
                d,
            )
            for d in aligned
        ]
    else:
        wall_const = x1
        span_start, span_end = y1, y2
        aligned = [
            d
            for d in doors
            if not d.is_blocked
            and d.orientation == "vertical"
            and math.isclose(d.position.x, wall_const, abs_tol=max(half_t, 0.25))
        ]
        spans = [
            (
                max(span_start, d.position.y - d.width / 2.0),
                min(span_end, d.position.y + d.width / 2.0),
                d,
            )
            for d in aligned
        ]

    cuts = sorted([(s, e, d) for s, e, d in spans if e - s > 0.05], key=lambda item: (item[0], item[1]))
    boxes: list[Box3D] = []
    cursor = span_start

    def _span_box(s0: float, s1: float, z0: float, z_h: float) -> Box3D:
        if is_horizontal:
            by1 = max(0.0, wall_const - half_t)
            by2 = min(scene.depth, wall_const + half_t)
            return Box3D(x=s0, y=by1, z=z0, width=s1 - s0, depth=max(0.05, by2 - by1), height=z_h)
        bx1 = max(0.0, wall_const - half_t)
        bx2 = min(scene.width, wall_const + half_t)
        return Box3D(x=bx1, y=s0, z=z0, width=max(0.05, bx2 - bx1), depth=s1 - s0, height=z_h)

    for s0, s1, door in cuts:
        if s0 - cursor > 0.05:
            boxes.append(_span_box(cursor, s0, wall.z_min, wall.height))
        lintel_z = door.position.z + door.height
        lintel_h = wall.z_max - lintel_z
        if lintel_h > 0.05:
            boxes.append(_span_box(s0, s1, lintel_z, lintel_h))
        cursor = max(cursor, s1)

    if span_end - cursor > 0.05:
        boxes.append(_span_box(cursor, span_end, wall.z_min, wall.height))

    return boxes


def _sample_visible_box_points(
    eye: tuple[float, float, float],
    box: Box3D,
) -> list[tuple[float, float, float]]:
    """Return center plus face-centers on faces turned toward ``eye``."""
    cx = box.x + box.width / 2.0
    cy = box.y + box.depth / 2.0
    cz = box.z + box.height / 2.0
    points: list[tuple[float, float, float]] = [(cx, cy, cz)]

    if eye[0] < box.x - _EPS:
        points.append((box.x, cy, cz))
    elif eye[0] > box.x2 + _EPS:
        points.append((box.x2, cy, cz))

    if eye[1] < box.y - _EPS:
        points.append((cx, box.y, cz))
    elif eye[1] > box.y2 + _EPS:
        points.append((cx, box.y2, cz))

    if eye[2] < box.z - _EPS:
        points.append((cx, cy, box.z))
    elif eye[2] > box.z2 + _EPS:
        points.append((cx, cy, box.z2))

    return points


class GroundTruthSolver3D:
    """Deterministic CPU-only 3D geometric solver operating on SceneSpec3D."""

    def solve_box_overlap(
        self,
        scene: SceneSpec3D,
        source_id: str,
        target_id: str,
    ) -> BoxOverlapResult:
        """Evaluate 3D overlap/touching/disjoint status between two entities or walls."""
        self._require_valid_scene(scene)
        src_box = self._resolve_object_box(scene, source_id)
        tgt_box = self._resolve_object_box(scene, target_id)
        return check_box_overlap_3d(
            src_box,
            tgt_box,
            source_id=source_id,
            target_id=target_id,
            meters_per_world_unit=scene.scale_bar.meters_per_world_unit,
        )

    def solve_vertical_relation(
        self,
        scene: SceneSpec3D,
        source_id: str,
        target_id: str,
    ) -> VerticalRelationResult:
        """Evaluate whether ``source_id`` sits above, below, or overlaps heights with ``target_id``."""
        self._require_valid_scene(scene)
        src_entity = self._find_entity(scene, source_id)
        src_box = src_entity.bounds if src_entity is not None else self._resolve_object_box(scene, source_id)
        tgt_box = self._resolve_object_box(scene, target_id)
        is_supported = src_entity is not None and src_entity.supported_by == target_id
        return analyze_vertical_relation(
            src_box,
            tgt_box,
            source_id=source_id,
            target_id=target_id,
            is_directly_supported=is_supported,
        )

    def solve_vertical_clearance(
        self,
        scene: SceneSpec3D,
        source_id: str,
        target_id: str,
    ) -> VerticalClearanceResult:
        """Compute vertical clearance between ``source_id`` and ``target_id`` when valid."""
        self._require_valid_scene(scene)
        if source_id == target_id:
            # Verify target exists before returning SAME_OBJECT.
            self._resolve_object_box(scene, source_id)
            return VerticalClearanceResult(
                source_id=source_id,
                target_id=target_id,
                is_defined=False,
                status=VerticalClearanceStatus.SAME_OBJECT,
                reason="Vertical clearance is undefined between an object and itself.",
            )

        src_box = self._resolve_object_box(scene, source_id)
        tgt_box = self._resolve_object_box(scene, target_id)
        return compute_vertical_clearance(
            src_box,
            tgt_box,
            source_id=source_id,
            target_id=target_id,
            meters_per_world_unit=scene.scale_bar.meters_per_world_unit,
        )

    def solve_ceiling_clearance(
        self,
        scene: SceneSpec3D,
        object_id: str,
    ) -> VerticalClearanceResult:
        """Compute vertical clearance from top of ``object_id`` to ``scene.ceiling_height``."""
        self._require_valid_scene(scene)
        box = self._resolve_object_box(scene, object_id)
        gap_units = max(0.0, scene.ceiling_height - box.z2)
        gap_units_rounded = round(gap_units, 6)
        gap_meters = round(gap_units * scene.scale_bar.meters_per_world_unit, 6)
        return VerticalClearanceResult(
            source_id=object_id,
            target_id="__ceiling__",
            is_defined=True,
            status=VerticalClearanceStatus.DEFINED,
            clearance_units=gap_units_rounded,
            clearance_meters=gap_meters,
            reason="Vertical gap between top of object and room ceiling.",
        )

    def solve_camera_visibility(
        self,
        scene: SceneSpec3D,
        target_id: str,
        *,
        projection: Literal["orthographic_isometric", "perspective_camera"] = "orthographic_isometric",
    ) -> CameraVisibilityResult:
        """Perform a sampled 3D line-of-sight check toward ``target_id``.

        When ``projection="orthographic_isometric"`` (default), casts parallel 3D rays
        along the fixed (+1, +1, +1) isometric direction matching Warehouse3DRenderer
        so solver visibility aligns with the rendered orthographic PNG.
        """
        self._require_valid_scene(scene)
        target_box = self._resolve_object_box(scene, target_id)
        eye = scene.camera.eye

        occluders: list[tuple[str, Box3D]] = []
        for ent in scene.entities:
            if ent.id == target_id or ent.passable:
                continue
            occluders.append((ent.id, ent.bounds))
        for wall in scene.walls:
            if wall.id == target_id:
                continue
            for wall_box in _wall_to_boxes_geometric(scene, wall, scene.doors):
                occluders.append((wall.id, wall_box))

        visible_count = 0
        center_visible = False
        blocking_ids: set[str] = set()

        if projection == "orthographic_isometric":
            # Orthographic isometric view frames the full room along (+1, +1, +1).
            within_fov = True
            iso_span = max(scene.width, scene.depth, scene.ceiling_height) * 2.0 + 10.0
            cx = target_box.x + target_box.width / 2.0
            cy = target_box.y + target_box.depth / 2.0
            cz = target_box.z + target_box.height / 2.0
            sample_points = [
                (cx, cy, cz),
                (target_box.x2, cy, cz),
                (cx, target_box.y2, cz),
                (cx, cy, target_box.z2),
            ]
            for idx, pt in enumerate(sample_points):
                ray_origin = (pt[0] + iso_span, pt[1] + iso_span, pt[2] + iso_span)
                blocked = False
                for occ_id, occ_box in occluders:
                    hit_t = ray_intersects_box3d(ray_origin, pt, occ_box)
                    if hit_t is not None:
                        blocked = True
                        blocking_ids.add(occ_id)
                if not blocked:
                    visible_count += 1
                    if idx == 0:
                        center_visible = True
            vis_model: Literal[
                "orthographic_isometric_sampled_raycast",
                "perspective_camera_sampled_raycast",
            ] = "orthographic_isometric_sampled_raycast"
            note = (
                "Evaluates sampled parallel 3D ray-AABB intersections along (+1, +1, +1) matching "
                "the fixed-angle orthographic isometric renderer; not pixel-exact raster visibility "
                "and not equivalent to full cutaway/label rasterization."
            )
        else:
            # Angular check against perspective CameraPose3D cone.
            look = scene.camera.look_at
            fwd = (look.x - eye[0], look.y - eye[1], look.z - eye[2])
            to_tgt = (
                target_box.center.x - eye[0],
                target_box.center.y - eye[1],
                target_box.center.z - eye[2],
            )
            fwd_len = math.sqrt(fwd[0] ** 2 + fwd[1] ** 2 + fwd[2] ** 2)
            tgt_len = math.sqrt(to_tgt[0] ** 2 + to_tgt[1] ** 2 + to_tgt[2] ** 2)
            if fwd_len <= _EPS or tgt_len <= _EPS:
                within_fov = True
            else:
                cos_theta = max(
                    -1.0,
                    min(
                        1.0,
                        (fwd[0] * to_tgt[0] + fwd[1] * to_tgt[1] + fwd[2] * to_tgt[2])
                        / (fwd_len * tgt_len),
                    ),
                )
                theta_deg = math.degrees(math.acos(cos_theta))
                within_fov = theta_deg <= (scene.camera.fov_deg / 2.0) + 1e-6

            sample_points = _sample_visible_box_points(eye, target_box)
            for idx, pt in enumerate(sample_points):
                blocked = False
                for occ_id, occ_box in occluders:
                    hit_t = ray_intersects_box3d(eye, pt, occ_box)
                    if hit_t is not None:
                        blocked = True
                        blocking_ids.add(occ_id)
                if not blocked:
                    visible_count += 1
                    if idx == 0:
                        center_visible = True
            vis_model = "perspective_camera_sampled_raycast"
            note = (
                "Evaluates sampled 3D ray-AABB intersections from SceneSpec3D.camera.eye; "
                "not pixel-exact raster visibility and not equivalent to Phase 2's fixed-angle "
                "orthographic isometric projection."
            )

        total = len(sample_points)
        fraction = visible_count / total
        if visible_count == total:
            status = VisibilityStatus.CLEAR
        elif visible_count == 0:
            status = VisibilityStatus.FULLY_OCCLUDED
        else:
            status = VisibilityStatus.PARTIALLY_OCCLUDED

        return CameraVisibilityResult(
            target_id=target_id,
            camera_eye=(round(eye[0], 6), round(eye[1], 6), round(eye[2], 6)),
            status=status,
            center_visible=center_visible,
            visible_sample_count=visible_count,
            total_sample_count=total,
            visible_fraction=round(fraction, 6),
            within_fov_cone=within_fov,
            blocking_object_ids=sorted(blocking_ids),
            visibility_model=vis_model,
            limitation_note=note,
        )

    def _find_entity(self, scene: SceneSpec3D, object_id: str) -> SceneEntity3D | None:
        for ent in scene.entities:
            if ent.id == object_id:
                return ent
        return None

    def _resolve_object_box(self, scene: SceneSpec3D, object_id: str) -> Box3D:
        ent = self._find_entity(scene, object_id)
        if ent is not None:
            return ent.bounds
        for wall in scene.walls:
            if wall.id == object_id:
                boxes = _wall_to_boxes_geometric(scene, wall, [])
                return boxes[0]
        raise UnsolvableTask3DError(
            f"3D object ID {object_id!r} not found in scene {scene.scene_id!r}."
        )

    def _require_valid_scene(self, scene: SceneSpec3D) -> None:
        if not isinstance(scene, SceneSpec3D):
            raise TypeError(f"Expected SceneSpec3D instance, got {type(scene).__name__}.")
