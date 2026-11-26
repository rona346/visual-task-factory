"""Deterministic fixed-angle isometric 3D PNG renderer for SceneSpec3D."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from visual_task_factory.models.scene import EntityType, ZoneType
from visual_task_factory.models.scene3d import (
    Box3D,
    Doorway3D,
    ExitMarker3D,
    Point3D,
    SceneSpec3D,
    WallSegment3D,
)
from visual_task_factory.models.variation import VisualTheme

ColorRGB = tuple[int, int, int]
ColorRGBA = tuple[int, int, int, int]

_COS30 = math.cos(math.radians(30.0))
_SIN30 = 0.5
_EPS = 1e-6

_DEFAULT_FONT_CANDIDATES: tuple[Path, ...] = (
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\consola.ttf"),
)


@dataclass(frozen=True)
class Theme3DPalette:
    """Color palette for fixed-angle isometric 3D warehouse rendering."""

    canvas_bg: ColorRGB
    floor_top: ColorRGB
    floor_edge_x: ColorRGB
    floor_edge_y: ColorRGB
    grid_line: ColorRGB
    wall_base: ColorRGB
    wall_border: ColorRGB
    door_open: ColorRGB
    door_blocked: ColorRGB
    exit_fill: ColorRGB
    header_bg: ColorRGB
    header_text: ColorRGB
    footer_bg: ColorRGB
    footer_text: ColorRGB
    zone_fills: dict[ZoneType, ColorRGBA]
    zone_borders: dict[ZoneType, ColorRGB]
    entity_fills: dict[EntityType, ColorRGB]
    entity_borders: dict[EntityType, ColorRGB]


_PALETTES_3D: dict[VisualTheme, Theme3DPalette] = {
    VisualTheme.SAFETY_STANDARD: Theme3DPalette(
        canvas_bg=(238, 242, 247),
        floor_top=(249, 250, 252),
        floor_edge_x=(196, 204, 216),
        floor_edge_y=(165, 175, 190),
        grid_line=(224, 230, 238),
        wall_base=(72, 84, 102),
        wall_border=(30, 41, 59),
        door_open=(34, 150, 96),
        door_blocked=(205, 50, 50),
        exit_fill=(22, 163, 74),
        header_bg=(30, 41, 59),
        header_text=(248, 250, 252),
        footer_bg=(228, 234, 242),
        footer_text=(30, 41, 59),
        zone_fills={
            ZoneType.STORAGE: (59, 130, 246, 52),
            ZoneType.LOADING: (14, 165, 233, 52),
            ZoneType.HAZARD: (245, 158, 11, 78),
            ZoneType.RESTRICTED: (220, 38, 38, 72),
            ZoneType.STAGING: (16, 185, 129, 52),
        },
        zone_borders={
            ZoneType.STORAGE: (37, 99, 235),
            ZoneType.LOADING: (2, 132, 199),
            ZoneType.HAZARD: (217, 119, 6),
            ZoneType.RESTRICTED: (185, 28, 28),
            ZoneType.STAGING: (5, 150, 105),
        },
        entity_fills={
            EntityType.PALLET_RACK: (88, 126, 164),
            EntityType.CRATE: (204, 152, 92),
            EntityType.FORKLIFT: (234, 179, 8),
            EntityType.PILLAR: (100, 116, 139),
            EntityType.CONVEYOR: (132, 144, 160),
            EntityType.ROBOT: (14, 132, 162),
        },
        entity_borders={
            EntityType.PALLET_RACK: (36, 58, 82),
            EntityType.CRATE: (116, 76, 34),
            EntityType.FORKLIFT: (132, 86, 8),
            EntityType.PILLAR: (51, 65, 85),
            EntityType.CONVEYOR: (64, 74, 88),
            EntityType.ROBOT: (12, 74, 96),
        },
    ),
    VisualTheme.BLUEPRINT: Theme3DPalette(
        canvas_bg=(17, 34, 64),
        floor_top=(23, 47, 86),
        floor_edge_x=(14, 30, 58),
        floor_edge_y=(10, 22, 44),
        grid_line=(40, 72, 122),
        wall_base=(120, 168, 224),
        wall_border=(208, 230, 255),
        door_open=(52, 211, 153),
        door_blocked=(248, 113, 113),
        exit_fill=(34, 197, 94),
        header_bg=(12, 25, 48),
        header_text=(224, 240, 255),
        footer_bg=(14, 29, 55),
        footer_text=(196, 220, 248),
        zone_fills={
            ZoneType.STORAGE: (96, 165, 250, 48),
            ZoneType.LOADING: (56, 189, 248, 48),
            ZoneType.HAZARD: (251, 191, 36, 64),
            ZoneType.RESTRICTED: (248, 113, 113, 64),
            ZoneType.STAGING: (52, 211, 153, 48),
        },
        zone_borders={
            ZoneType.STORAGE: (147, 197, 253),
            ZoneType.LOADING: (125, 211, 252),
            ZoneType.HAZARD: (252, 211, 77),
            ZoneType.RESTRICTED: (252, 165, 165),
            ZoneType.STAGING: (110, 231, 183),
        },
        entity_fills={
            EntityType.PALLET_RACK: (59, 130, 206),
            EntityType.CRATE: (217, 158, 86),
            EntityType.FORKLIFT: (234, 179, 8),
            EntityType.PILLAR: (148, 163, 184),
            EntityType.CONVEYOR: (94, 148, 194),
            EntityType.ROBOT: (34, 184, 207),
        },
        entity_borders={
            EntityType.PALLET_RACK: (191, 219, 254),
            EntityType.CRATE: (253, 230, 138),
            EntityType.FORKLIFT: (254, 240, 138),
            EntityType.PILLAR: (226, 232, 240),
            EntityType.CONVEYOR: (186, 230, 253),
            EntityType.ROBOT: (165, 243, 252),
        },
    ),
    VisualTheme.HIGH_CONTRAST: Theme3DPalette(
        canvas_bg=(245, 245, 245),
        floor_top=(255, 255, 255),
        floor_edge_x=(180, 180, 180),
        floor_edge_y=(140, 140, 140),
        grid_line=(210, 210, 210),
        wall_base=(40, 40, 40),
        wall_border=(0, 0, 0),
        door_open=(0, 150, 70),
        door_blocked=(210, 20, 20),
        exit_fill=(0, 155, 65),
        header_bg=(0, 0, 0),
        header_text=(255, 255, 255),
        footer_bg=(230, 230, 230),
        footer_text=(0, 0, 0),
        zone_fills={
            ZoneType.STORAGE: (0, 102, 204, 52),
            ZoneType.LOADING: (0, 145, 195, 52),
            ZoneType.HAZARD: (245, 158, 11, 80),
            ZoneType.RESTRICTED: (210, 20, 20, 76),
            ZoneType.STAGING: (0, 150, 90, 52),
        },
        zone_borders={
            ZoneType.STORAGE: (0, 72, 155),
            ZoneType.LOADING: (0, 105, 155),
            ZoneType.HAZARD: (180, 95, 0),
            ZoneType.RESTRICTED: (150, 0, 0),
            ZoneType.STAGING: (0, 105, 60),
        },
        entity_fills={
            EntityType.PALLET_RACK: (56, 98, 148),
            EntityType.CRATE: (210, 138, 42),
            EntityType.FORKLIFT: (238, 182, 0),
            EntityType.PILLAR: (75, 75, 75),
            EntityType.CONVEYOR: (115, 120, 128),
            EntityType.ROBOT: (0, 130, 160),
        },
        entity_borders={
            EntityType.PALLET_RACK: (0, 0, 0),
            EntityType.CRATE: (0, 0, 0),
            EntityType.FORKLIFT: (0, 0, 0),
            EntityType.PILLAR: (0, 0, 0),
            EntityType.CONVEYOR: (0, 0, 0),
            EntityType.ROBOT: (0, 0, 0),
        },
    ),
    VisualTheme.CLUTTERED_OPS: Theme3DPalette(
        canvas_bg=(230, 226, 218),
        floor_top=(242, 239, 232),
        floor_edge_x=(188, 182, 170),
        floor_edge_y=(160, 154, 142),
        grid_line=(214, 209, 198),
        wall_base=(76, 70, 62),
        wall_border=(36, 32, 28),
        door_open=(42, 136, 90),
        door_blocked=(192, 52, 42),
        exit_fill=(32, 142, 74),
        header_bg=(54, 49, 44),
        header_text=(248, 245, 238),
        footer_bg=(220, 215, 206),
        footer_text=(42, 38, 34),
        zone_fills={
            ZoneType.STORAGE: (78, 121, 167, 52),
            ZoneType.LOADING: (89, 161, 169, 52),
            ZoneType.HAZARD: (237, 137, 54, 76),
            ZoneType.RESTRICTED: (197, 48, 48, 72),
            ZoneType.STAGING: (72, 152, 112, 52),
        },
        zone_borders={
            ZoneType.STORAGE: (52, 92, 136),
            ZoneType.LOADING: (56, 122, 130),
            ZoneType.HAZARD: (192, 86, 33),
            ZoneType.RESTRICTED: (155, 28, 28),
            ZoneType.STAGING: (47, 118, 84),
        },
        entity_fills={
            EntityType.PALLET_RACK: (102, 114, 128),
            EntityType.CRATE: (184, 134, 76),
            EntityType.FORKLIFT: (218, 165, 24),
            EntityType.PILLAR: (88, 82, 76),
            EntityType.CONVEYOR: (124, 118, 112),
            EntityType.ROBOT: (36, 122, 138),
        },
        entity_borders={
            EntityType.PALLET_RACK: (48, 56, 66),
            EntityType.CRATE: (110, 74, 34),
            EntityType.FORKLIFT: (128, 82, 12),
            EntityType.PILLAR: (36, 34, 32),
            EntityType.CONVEYOR: (58, 54, 50),
            EntityType.ROBOT: (20, 72, 84),
        },
    ),
}


@dataclass(frozen=True)
class Render3DConfig:
    """Configuration controlling deterministic fixed-angle isometric 3D rendering."""

    width_px: int = 800
    height_px: int = 540
    supersample: int = 2
    margin_px: int = 28
    header_height: int = 44
    footer_height: int = 38
    theme: VisualTheme | None = None
    show_labels: bool = True
    show_grid: bool = True
    show_zones: bool = True
    cutaway_front_walls: bool = True

    def __post_init__(self) -> None:
        if self.width_px < 240 or self.width_px > 4096:
            raise ValueError("width_px must be between 240 and 4096.")
        if self.height_px < 180 or self.height_px > 4096:
            raise ValueError("height_px must be between 180 and 4096.")
        if self.supersample not in (1, 2, 3):
            raise ValueError("supersample must be 1, 2, or 3.")
        if self.margin_px < 8:
            raise ValueError("margin_px must be >= 8.")
        if self.header_height < 0 or self.footer_height < 0:
            raise ValueError("header_height and footer_height must be >= 0.")
        usable_h = self.height_px - self.header_height - self.footer_height - 2 * self.margin_px
        usable_w = self.width_px - 2 * self.margin_px
        if usable_h < 60 or usable_w < 80:
            raise ValueError(
                "Configured margins, header, and footer leave insufficient viewport area."
            )
        if self.theme is not None and not isinstance(self.theme, VisualTheme):
            raise ValueError("theme must be a VisualTheme or None.")


@dataclass(frozen=True)
class Render3DResult:
    """Rendered isometric 3D image alongside deterministic draw metadata."""

    image: Image.Image
    output_path: Path | None
    width_px: int
    height_px: int
    draw_order: tuple[str, ...]
    cycle_detected: bool
    placed_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class _BoxPrimitive:
    """Renderable axis-aligned 3D box primitive placed in the painter queue."""

    prim_id: str
    bounds: Box3D
    base_color: ColorRGB
    border_color: ColorRGB
    label: str | None = None
    sort_priority: int = 1  # Higher priority draws later when unconstrained.


@dataclass(frozen=True)
class _IsoProjector:
    """Fixed-angle orthographic isometric projection (+X, +Y, +Z camera)."""

    scale: float
    offset_x: float
    offset_y: float

    @staticmethod
    def _raw_uv(x: float, y: float, z: float) -> tuple[float, float]:
        return ((x - y) * _COS30, (x + y) * _SIN30 - z)

    def project(self, x: float, y: float, z: float) -> tuple[float, float]:
        u, v = self._raw_uv(x, y, z)
        return (self.offset_x + u * self.scale, self.offset_y + v * self.scale)

    def project_point(self, pt: Point3D) -> tuple[float, float]:
        return self.project(pt.x, pt.y, pt.z)

    @classmethod
    def fit_scene(
        cls,
        scene: SceneSpec3D,
        viewport: tuple[float, float, float, float],
    ) -> _IsoProjector:
        vx0, vy0, vx1, vy1 = viewport
        sample_pts = [
            (0.0, 0.0, 0.0),
            (scene.width, 0.0, 0.0),
            (0.0, scene.depth, 0.0),
            (scene.width, scene.depth, 0.0),
            (0.0, 0.0, scene.ceiling_height),
            (scene.width, 0.0, scene.ceiling_height),
            (0.0, scene.depth, scene.ceiling_height),
            (scene.width, scene.depth, scene.ceiling_height),
        ]
        uvs = [cls._raw_uv(x, y, z) for x, y, z in sample_pts]
        min_u = min(u for u, _ in uvs)
        max_u = max(u for u, _ in uvs)
        min_v = min(v for _, v in uvs)
        max_v = max(v for _, v in uvs)

        span_u = max(max_u - min_u, 1e-6)
        span_v = max(max_v - min_v, 1e-6)
        vp_w = max(vx1 - vx0, 1.0)
        vp_h = max(vy1 - vy0, 1.0)
        scale = min(vp_w / span_u, vp_h / span_v)

        center_u = (min_u + max_u) / 2.0
        center_v = (min_v + max_v) / 2.0
        vp_cx = (vx0 + vx1) / 2.0
        vp_cy = (vy0 + vy1) / 2.0
        return cls(
            scale=scale,
            offset_x=vp_cx - center_u * scale,
            offset_y=vp_cy - center_v * scale,
        )


def _shade_color(rgb: ColorRGB, factor: float) -> ColorRGB:
    return tuple(max(0, min(255, int(round(channel * factor)))) for channel in rgb)  # type: ignore[return-value]


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for candidate in _DEFAULT_FONT_CANDIDATES:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _box_a_behind_box_b(a: Box3D, b: Box3D, eps: float = _EPS) -> bool:
    """Return True if camera rays from (+1, +1, +1) through B hit A strictly behind B.

    Equivalently, there exists t > 0, p_a in interior(A), p_b in interior(B) with
    p_b - p_a = (t, t, t), which holds iff (I_x n I_y n I_z) contains some t > 0.
    """
    low = max(0.0, b.x - a.x2, b.y - a.y2, b.z - a.z2)
    high = min(b.x2 - a.x, b.y2 - a.y, b.z2 - a.z)
    return (low + eps) < high


def sort_primitives_back_to_front(
    primitives: list[_BoxPrimitive],
) -> tuple[list[_BoxPrimitive], bool]:
    """Deterministically order non-intersecting 3D boxes back-to-front.

    Uses separating-ray topological sorting along the (+1, +1, +1) isometric view ray.
    If an interlocking cyclic overlap (e.g. a 3-box pinwheel) creates a topological
    cycle, falls back deterministically for the remaining cycle nodes using centroid
    view-depth ordering and reports ``cycle_detected=True``.
    """
    n = len(primitives)
    if n <= 1:
        return list(primitives), False

    adj: list[set[int]] = [set() for _ in range(n)]
    in_degree = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            a = primitives[i].bounds
            b = primitives[j].bounds
            a_behind_b = _box_a_behind_box_b(a, b)
            b_behind_a = _box_a_behind_box_b(b, a)
            if a_behind_b and not b_behind_a:
                adj[i].add(j)
                in_degree[j] += 1
            elif b_behind_a and not a_behind_b:
                adj[j].add(i)
                in_degree[i] += 1

    def _tie_key(idx: int) -> tuple[float, int, float, float, float, str]:
        b = primitives[idx].bounds
        cx = b.x + b.width / 2.0
        cy = b.y + b.depth / 2.0
        cz = b.z + b.height / 2.0
        return (
            round(cx + cy + cz, 6),
            primitives[idx].sort_priority,
            round(b.z, 6),
            round(b.x, 6),
            round(b.y, 6),
            primitives[idx].prim_id,
        )

    remaining = set(range(n))
    ordered: list[_BoxPrimitive] = []
    cycle_detected = False

    while remaining:
        ready = [idx for idx in remaining if in_degree[idx] == 0]
        if ready:
            chosen = min(ready, key=_tie_key)
        else:
            # Classic painter's algorithm cycle (e.g. 3-bar pinwheel); break deterministically.
            cycle_detected = True
            chosen = min(remaining, key=_tie_key)

        remaining.remove(chosen)
        ordered.append(primitives[chosen])
        for nxt in adj[chosen]:
            if nxt in remaining:
                in_degree[nxt] -= 1

    return ordered, cycle_detected


def _wall_to_boxes_with_doors(
    scene: SceneSpec3D,
    wall: WallSegment3D,
    doors: list[Doorway3D],
) -> list[Box3D]:
    """Convert a WallSegment3D into non-overlapping Box3D slabs cut around aligned doors."""
    half_t = wall.thickness / 2.0
    x1 = min(wall.start.x, wall.end.x)
    x2 = max(wall.start.x, wall.end.x)
    y1 = min(wall.start.y, wall.end.y)
    y2 = max(wall.start.y, wall.end.y)

    is_vertical = math.isclose(x1, x2, abs_tol=1e-6)
    is_horizontal = math.isclose(y1, y2, abs_tol=1e-6)

    if not is_vertical and not is_horizontal:
        bx1 = max(0.0, x1 - half_t)
        bx2 = min(scene.width, x2 + half_t)
        by1 = max(0.0, y1 - half_t)
        by2 = min(scene.depth, y2 + half_t)
        return [
            Box3D(
                x=bx1,
                y=by1,
                z=wall.z_min,
                width=max(0.05, bx2 - bx1),
                depth=max(0.05, by2 - by1),
                height=wall.height,
            )
        ]

    if is_horizontal:
        wall_const = y1
        span_start, span_end = x1, x2
        aligned = [
            d
            for d in doors
            if d.orientation == "horizontal"
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
            if d.orientation == "vertical"
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

    # Filter positive-length doorway overlaps and sort along wall span.
    cuts = sorted(
        [(s, e, d) for s, e, d in spans if e - s > 0.05],
        key=lambda item: (item[0], item[1]),
    )

    boxes: list[Box3D] = []
    cursor = span_start

    def _make_span_box(s0: float, s1: float, z0: float, z_h: float) -> Box3D:
        if is_horizontal:
            by1 = max(0.0, wall_const - half_t)
            by2 = min(scene.depth, wall_const + half_t)
            return Box3D(
                x=s0,
                y=by1,
                z=z0,
                width=max(0.05, s1 - s0),
                depth=max(0.05, by2 - by1),
                height=z_h,
            )
        bx1 = max(0.0, wall_const - half_t)
        bx2 = min(scene.width, wall_const + half_t)
        return Box3D(
            x=bx1,
            y=s0,
            z=z0,
            width=max(0.05, bx2 - bx1),
            depth=max(0.05, s1 - s0),
            height=z_h,
        )

    for s0, s1, door in cuts:
        if s0 - cursor > 0.05:
            boxes.append(_make_span_box(cursor, s0, wall.z_min, wall.height))
        lintel_z = door.position.z + door.height
        lintel_h = wall.z_max - lintel_z
        if lintel_h > 0.05:
            boxes.append(_make_span_box(s0, s1, lintel_z, lintel_h))
        cursor = max(cursor, s1)

    if span_end - cursor > 0.05:
        boxes.append(_make_span_box(cursor, span_end, wall.z_min, wall.height))

    return boxes


def _door_to_box3d(scene: SceneSpec3D, door: Doorway3D) -> Box3D:
    """Represent a doorway threshold (open) or barrier (blocked) in the wall gap."""
    half_t = 0.14 if not door.is_blocked else 0.16
    door_h = 0.12 if not door.is_blocked else min(door.height, 1.8)
    if door.orientation == "horizontal":
        x1 = max(0.0, door.position.x - door.width / 2.0)
        x2 = min(scene.width, x1 + door.width)
        y1 = max(0.0, door.position.y - half_t)
        y2 = min(scene.depth, door.position.y + half_t)
    else:
        x1 = max(0.0, door.position.x - half_t)
        x2 = min(scene.width, door.position.x + half_t)
        y1 = max(0.0, door.position.y - door.width / 2.0)
        y2 = min(scene.depth, y1 + door.width)

    return Box3D(
        x=x1,
        y=y1,
        z=door.position.z,
        width=max(0.05, x2 - x1),
        depth=max(0.05, y2 - y1),
        height=door_h,
    )


def _exit_to_box3d(scene: SceneSpec3D, exit_marker: ExitMarker3D) -> Box3D:
    """Represent an exit marker as an upright portal prism clamped inside scene bounds."""
    half_w = min(exit_marker.width / 2.0, 1.0)
    thickness = 0.35
    x_on_vertical = math.isclose(exit_marker.position.x, 0.0, abs_tol=0.25) or math.isclose(
        exit_marker.position.x, scene.width, abs_tol=0.25
    )
    if x_on_vertical:
        x1 = max(0.0, min(scene.width - thickness, exit_marker.position.x - thickness / 2.0))
        x2 = min(scene.width, x1 + thickness)
        y1 = max(0.0, min(scene.depth - 0.5, exit_marker.position.y - half_w))
        y2 = min(scene.depth, y1 + exit_marker.width)
    else:
        x1 = max(0.0, min(scene.width - 0.5, exit_marker.position.x - half_w))
        x2 = min(scene.width, x1 + exit_marker.width)
        y1 = max(0.0, min(scene.depth - thickness, exit_marker.position.y - thickness / 2.0))
        y2 = min(scene.depth, y1 + thickness)

    return Box3D(
        x=x1,
        y=y1,
        z=exit_marker.position.z,
        width=max(0.1, x2 - x1),
        depth=max(0.1, y2 - y1),
        height=exit_marker.height,
    )


def _is_front_perimeter_wall(scene: SceneSpec3D, box: Box3D) -> bool:
    """Identify walls sitting directly on front-most (+X or +Y) edges."""
    on_front_x = math.isclose(box.x2, scene.width, abs_tol=0.4) and box.depth > box.width * 2.0
    on_front_y = math.isclose(box.y2, scene.depth, abs_tol=0.4) and box.width > box.depth * 2.0
    return on_front_x or on_front_y


def _point_in_polygon(
    px: float,
    py: float,
    poly: list[tuple[float, float]],
) -> bool:
    """Ray-casting point-in-polygon test in 2D screen space."""
    inside = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _box_projected_hull_faces(
    b: Box3D,
    proj: _IsoProjector,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
    x1, y1, z1 = b.x, b.y, b.z
    x2, y2, z2 = b.x2, b.y2, b.z2
    top_face = [
        proj.project(x1, y1, z2),
        proj.project(x2, y1, z2),
        proj.project(x2, y2, z2),
        proj.project(x1, y2, z2),
    ]
    right_face = [
        proj.project(x2, y1, z1),
        proj.project(x2, y2, z1),
        proj.project(x2, y2, z2),
        proj.project(x2, y1, z2),
    ]
    left_face = [
        proj.project(x1, y2, z1),
        proj.project(x2, y2, z1),
        proj.project(x2, y2, z2),
        proj.project(x1, y2, z2),
    ]
    return top_face, right_face, left_face


def _rects_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    pad: float = 4.0,
) -> bool:
    return not (
        a[2] + pad <= b[0]
        or b[2] + pad <= a[0]
        or a[3] + pad <= b[1]
        or b[3] + pad <= a[1]
    )


class Warehouse3DRenderer:
    """Deterministic CPU-only isometric renderer for SceneSpec3D."""

    def __init__(self, config: Render3DConfig | None = None) -> None:
        self.config = config or Render3DConfig()

    def render(
        self,
        scene: SceneSpec3D,
        output_path: str | Path | None = None,
    ) -> Render3DResult:
        """Render ``scene`` into an RGB PNG image and optionally save to ``output_path``."""
        if not isinstance(scene, SceneSpec3D):
            raise TypeError(f"Expected SceneSpec3D instance, got {type(scene).__name__}.")

        resolved_path: Path | None = None
        if output_path is not None:
            resolved_path = Path(output_path)
            if str(output_path).strip() == "":
                raise ValueError("output_path must not be empty.")
            if resolved_path.exists() and resolved_path.is_dir():
                raise IsADirectoryError(f"output_path points to a directory: {resolved_path}")

        cfg = self.config
        ss = cfg.supersample
        canvas_w = cfg.width_px * ss
        canvas_h = cfg.height_px * ss
        header_h = cfg.header_height * ss
        footer_h = cfg.footer_height * ss
        margin = cfg.margin_px * ss

        theme = cfg.theme or scene.variation.theme
        palette = _PALETTES_3D[theme]

        img = Image.new("RGBA", (canvas_w, canvas_h), (*palette.canvas_bg, 255))
        draw = ImageDraw.Draw(img, "RGBA")

        viewport = (
            float(margin),
            float(header_h + margin),
            float(canvas_w - margin),
            float(canvas_h - footer_h - margin),
        )
        projector = _IsoProjector.fit_scene(scene, viewport)

        self._draw_floor_and_zones(draw, scene, projector, palette, ss)

        primitives = self._collect_primitives(scene, palette)
        ordered, cycle_detected = sort_primitives_back_to_front(primitives)

        self._draw_primitives(draw, ordered, projector, ss)

        placed_labels: tuple[str, ...] = ()
        if cfg.show_labels:
            placed_labels = self._draw_labels(draw, ordered, projector, ss)

        self._draw_header_and_footer(draw, scene, palette, canvas_w, canvas_h, header_h, footer_h, ss)

        rgb_img = img.convert("RGB")
        if ss > 1:
            rgb_img = rgb_img.resize((cfg.width_px, cfg.height_px), Image.Resampling.LANCZOS)

        if resolved_path is not None:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            rgb_img.save(resolved_path, format="PNG", optimize=False, compress_level=6)

        return Render3DResult(
            image=rgb_img,
            output_path=resolved_path,
            width_px=cfg.width_px,
            height_px=cfg.height_px,
            draw_order=tuple(item.prim_id for item in ordered),
            cycle_detected=cycle_detected,
            placed_labels=placed_labels,
        )

    def _draw_floor_and_zones(
        self,
        draw: ImageDraw.ImageDraw,
        scene: SceneSpec3D,
        proj: _IsoProjector,
        palette: Theme3DPalette,
        ss: int,
    ) -> None:
        plinth_depth = 0.35
        floor_corners = [
            proj.project(0.0, 0.0, 0.0),
            proj.project(scene.width, 0.0, 0.0),
            proj.project(scene.width, scene.depth, 0.0),
            proj.project(0.0, scene.depth, 0.0),
        ]
        # Subtle 3D slab edge under front +X and +Y sides so platform depth reads immediately.
        edge_x = [
            proj.project(scene.width, 0.0, 0.0),
            proj.project(scene.width, scene.depth, 0.0),
            proj.project(scene.width, scene.depth, -plinth_depth),
            proj.project(scene.width, 0.0, -plinth_depth),
        ]
        edge_y = [
            proj.project(0.0, scene.depth, 0.0),
            proj.project(scene.width, scene.depth, 0.0),
            proj.project(scene.width, scene.depth, -plinth_depth),
            proj.project(0.0, scene.depth, -plinth_depth),
        ]
        draw.polygon(edge_x, fill=palette.floor_edge_x, outline=palette.wall_border)
        draw.polygon(edge_y, fill=palette.floor_edge_y, outline=palette.wall_border)
        draw.polygon(floor_corners, fill=palette.floor_top, outline=palette.wall_border)

        if self.config.show_grid:
            step = 2.0 if max(scene.width, scene.depth) >= 12.0 else 1.0
            grid_w = max(1, ss)
            x = step
            while x < scene.width - 1e-6:
                draw.line(
                    [proj.project(x, 0.0, 0.0), proj.project(x, scene.depth, 0.0)],
                    fill=palette.grid_line,
                    width=grid_w,
                )
                x += step
            y = step
            while y < scene.depth - 1e-6:
                draw.line(
                    [proj.project(0.0, y, 0.0), proj.project(scene.width, y, 0.0)],
                    fill=palette.grid_line,
                    width=grid_w,
                )
                y += step

        if self.config.show_zones:
            zone_font = _load_font(11 * ss)
            for zone in scene.zones:
                zb = zone.bounds
                z_pts = [
                    proj.project(zb.x, zb.y, zone.z_floor),
                    proj.project(zb.x2, zb.y, zone.z_floor),
                    proj.project(zb.x2, zb.y2, zone.z_floor),
                    proj.project(zb.x, zb.y2, zone.z_floor),
                ]
                draw.polygon(
                    z_pts,
                    fill=palette.zone_fills[zone.zone_type],
                    outline=palette.zone_borders[zone.zone_type],
                )
                if self.config.show_labels:
                    cx, cy = proj.project(
                        zb.x + zb.width / 2.0,
                        zb.y + zb.height / 2.0,
                        zone.z_floor,
                    )
                    draw.text(
                        (cx, cy),
                        zone.label,
                        fill=palette.zone_borders[zone.zone_type],
                        font=zone_font,
                        anchor="mm",
                    )

    def _collect_primitives(
        self,
        scene: SceneSpec3D,
        palette: Theme3DPalette,
    ) -> list[_BoxPrimitive]:
        primitives: list[_BoxPrimitive] = []

        for wall in scene.walls:
            wall_boxes = _wall_to_boxes_with_doors(scene, wall, scene.doors)
            for seg_idx, box in enumerate(wall_boxes):
                if self.config.cutaway_front_walls and _is_front_perimeter_wall(scene, box):
                    box = Box3D(
                        x=box.x,
                        y=box.y,
                        z=box.z,
                        width=box.width,
                        depth=box.depth,
                        height=min(box.height, 0.5),
                    )
                seg_id = wall.id if len(wall_boxes) == 1 else f"{wall.id}__seg{seg_idx}"
                primitives.append(
                    _BoxPrimitive(
                        prim_id=seg_id,
                        bounds=box,
                        base_color=palette.wall_base,
                        border_color=palette.wall_border,
                        label=None,
                        sort_priority=0,
                    )
                )

        for door in scene.doors:
            box = _door_to_box3d(scene, door)
            door_color = palette.door_blocked if door.is_blocked else palette.door_open
            primitives.append(
                _BoxPrimitive(
                    prim_id=door.id,
                    bounds=box,
                    base_color=door_color,
                    border_color=_shade_color(door_color, 0.45),
                    label="BLOCKED" if door.is_blocked else None,
                    sort_priority=1,
                )
            )

        for exit_marker in scene.exits:
            box = _exit_to_box3d(scene, exit_marker)
            primitives.append(
                _BoxPrimitive(
                    prim_id=exit_marker.id,
                    bounds=box,
                    base_color=palette.exit_fill,
                    border_color=_shade_color(palette.exit_fill, 0.45),
                    label=exit_marker.label,
                    sort_priority=1,
                )
            )

        for entity in scene.entities:
            primitives.append(
                _BoxPrimitive(
                    prim_id=entity.id,
                    bounds=entity.bounds,
                    base_color=palette.entity_fills[entity.entity_type],
                    border_color=palette.entity_borders[entity.entity_type],
                    label=None if entity.entity_type == EntityType.PILLAR else entity.label,
                    sort_priority=2,
                )
            )

        return primitives

    def _draw_primitives(
        self,
        draw: ImageDraw.ImageDraw,
        ordered: list[_BoxPrimitive],
        proj: _IsoProjector,
        ss: int,
    ) -> None:
        line_w = max(1, ss)
        for item in ordered:
            # Distinct lighting across top (+Z), right-front (+X), and left-front (+Y) faces.
            top_color = _shade_color(item.base_color, 1.12)
            right_color = _shade_color(item.base_color, 0.86)
            left_color = _shade_color(item.base_color, 0.64)

            top_face, right_face, left_face = _box_projected_hull_faces(item.bounds, proj)

            draw.polygon(left_face, fill=left_color)
            draw.polygon(right_face, fill=right_color)
            draw.polygon(top_face, fill=top_color)

            for poly in (left_face, right_face, top_face):
                draw.line(poly + [poly[0]], fill=item.border_color, width=line_w)

    def _draw_labels(
        self,
        draw: ImageDraw.ImageDraw,
        ordered: list[_BoxPrimitive],
        proj: _IsoProjector,
        ss: int,
    ) -> tuple[str, ...]:
        font = _load_font(10 * ss)
        occupied: list[tuple[float, float, float, float]] = []
        front_faces: list[list[tuple[float, float]]] = []
        placed: list[str] = []

        # Iterate front-to-back so occluded objects do not draw labels over foreground boxes.
        for item in reversed(ordered):
            faces = _box_projected_hull_faces(item.bounds, proj)
            if not item.label:
                front_faces.extend(faces)
                continue

            b = item.bounds
            top_corners = faces[0]
            min_sx = min(pt[0] for pt in top_corners)
            max_sx = max(pt[0] for pt in top_corners)
            min_top_span = 8.0 * ss if item.sort_priority == 1 else 22.0 * ss
            if (max_sx - min_sx) < min_top_span:
                front_faces.extend(faces)
                continue

            base_cx, base_cy = proj.project(b.x + b.width / 2.0, b.y + b.depth / 2.0, b.z2)
            if any(_point_in_polygon(base_cx, base_cy, poly) for poly in front_faces):
                front_faces.extend(faces)
                continue

            pad_x = 4 * ss
            pad_y = 2 * ss
            offsets = (
                (0.0, 0.0),
                (0.0, -16.0 * ss),
                (0.0, 16.0 * ss),
                (-22.0 * ss, 0.0),
                (22.0 * ss, 0.0),
                (-16.0 * ss, -14.0 * ss),
                (16.0 * ss, -14.0 * ss),
                (-16.0 * ss, 14.0 * ss),
                (16.0 * ss, 14.0 * ss),
            )
            chosen: tuple[float, float, tuple[float, float, float, float]] | None = None
            for dx, dy in offsets:
                cand_x = base_cx + dx
                cand_y = base_cy + dy
                bbox = draw.textbbox((cand_x, cand_y), item.label, font=font, anchor="mm")
                pill = (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y)
                if not any(_rects_overlap(pill, existing, pad=2.0 * ss) for existing in occupied):
                    chosen = (cand_x, cand_y, pill)
                    break

            if chosen is None:
                front_faces.extend(faces)
                continue

            cx, cy, pill = chosen
            occupied.append(pill)
            placed.append(item.label)
            draw.rounded_rectangle(
                pill,
                radius=3 * ss,
                fill=(15, 23, 42, 205),
                outline=(255, 255, 255, 180),
                width=max(1, ss),
            )
            draw.text((cx, cy), item.label, fill=(255, 255, 255, 255), font=font, anchor="mm")
            front_faces.extend(faces)

        return tuple(placed)

    def _draw_header_and_footer(
        self,
        draw: ImageDraw.ImageDraw,
        scene: SceneSpec3D,
        palette: Theme3DPalette,
        canvas_w: int,
        canvas_h: int,
        header_h: int,
        footer_h: int,
        ss: int,
    ) -> None:
        if header_h > 0:
            draw.rectangle([0, 0, canvas_w, header_h], fill=palette.header_bg)
            title_font = _load_font(15 * ss)
            sub_font = _load_font(11 * ss)
            draw.text(
                (16 * ss, header_h / 2.0),
                f"Warehouse 3D Isometric | {scene.scene_id}",
                fill=palette.header_text,
                font=title_font,
                anchor="lm",
            )
            draw.text(
                (canvas_w - 16 * ss, header_h / 2.0),
                f"{scene.width:.1f}m x {scene.depth:.1f}m x {scene.ceiling_height:.1f}m | seed={scene.variation.seed}",
                fill=palette.header_text,
                font=sub_font,
                anchor="rm",
            )

        if footer_h > 0:
            fy0 = canvas_h - footer_h
            draw.rectangle([0, fy0, canvas_w, canvas_h], fill=palette.footer_bg)
            footer_font = _load_font(11 * ss)
            draw.text(
                (16 * ss, fy0 + footer_h / 2.0),
                f"Scale: {scene.scale_bar.world_units:g} units = {scene.scale_bar.display_meters:g}{scene.scale_bar.unit_label} | Projection: Fixed Isometric (+X, +Y, +Z)",
                fill=palette.footer_text,
                font=footer_font,
                anchor="lm",
            )


def render_scene_3d(
    scene: SceneSpec3D,
    output_path: str | Path | None = None,
    *,
    config: Render3DConfig | None = None,
) -> Render3DResult:
    """Render a SceneSpec3D using Warehouse3DRenderer with optional PNG output."""
    return Warehouse3DRenderer(config=config).render(scene, output_path=output_path)
