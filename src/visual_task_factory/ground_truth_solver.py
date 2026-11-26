"""Independent deterministic Ground Truth Solver for Warehouse SceneSpec instances."""

from __future__ import annotations

import heapq
import math
from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from visual_task_factory.models import (
    AnswerType,
    Doorway,
    EntityType,
    ExitMarker,
    Point2D,
    QuestionFamily,
    Rect2D,
    ReferenceAnswer,
    SceneEntity,
    SceneSpec,
    WallSegment,
    Zone,
    ZoneType,
)


class SolverError(ValueError):
    """Base error raised by the deterministic ground truth solver."""


class UnsupportedTaskError(SolverError):
    """Raised when a query specifies an unsupported family or configuration."""


class UnsolvableTaskError(SolverError):
    """Raised when a query cannot be solved unambiguously on the given scene."""


class ContainmentMode(str, Enum):
    """Geometric rule used to decide whether an entity is inside a zone."""

    STRICT_INSIDE = "strict_inside"
    CENTER_INSIDE = "center_inside"
    ANY_OVERLAP = "any_overlap"


class ReachabilityQuery(BaseModel):
    """Specification for a constrained reachability / exit-selection query."""

    model_config = ConfigDict(extra="forbid")

    family: Literal[QuestionFamily.CONSTRAINED_REACHABILITY] = (
        QuestionFamily.CONSTRAINED_REACHABILITY
    )
    start_point: Point2D | None = None
    start_zone_id: str | None = None
    candidate_exit_ids: list[str] | None = None
    forbidden_zone_types: tuple[ZoneType, ...] = (ZoneType.HAZARD, ZoneType.RESTRICTED)
    criterion: Literal["shortest_path"] = "shortest_path"

    @model_validator(mode="after")
    def _validate_start_spec(self) -> ReachabilityQuery:
        if (self.start_point is None) == (self.start_zone_id is None):
            raise ValueError("Provide exactly one of start_point or start_zone_id.")
        if self.candidate_exit_ids is not None:
            cleaned = [eid.strip() for eid in self.candidate_exit_ids if eid.strip()]
            if not cleaned or len(set(cleaned)) != len(cleaned):
                raise ValueError("candidate_exit_ids must contain unique non-empty exit IDs.")
            self.candidate_exit_ids = cleaned
        return self


class RegionCountingQuery(BaseModel):
    """Specification for a region containment and counting query."""

    model_config = ConfigDict(extra="forbid")

    family: Literal[QuestionFamily.REGION_CONTAINMENT_COUNTING] = (
        QuestionFamily.REGION_CONTAINMENT_COUNTING
    )
    zone_id: str = Field(..., min_length=1)
    entity_types: tuple[EntityType, ...] | None = None
    containment_mode: ContainmentMode = ContainmentMode.STRICT_INSIDE
    include_passable: bool = True

    @field_validator("zone_id")
    @classmethod
    def _clean_zone_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("zone_id must not be empty.")
        return cleaned


class ClearanceQuery(BaseModel):
    """Specification for a scale-grounded spatial clearance query."""

    model_config = ConfigDict(extra="forbid")

    family: Literal[QuestionFamily.SCALE_GROUNDED_CLEARANCE] = (
        QuestionFamily.SCALE_GROUNDED_CLEARANCE
    )
    source_id: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    answer_type: Literal[AnswerType.BOOLEAN, AnswerType.NUMERIC] = AnswerType.NUMERIC
    min_required_meters: float | None = Field(default=None, ge=0.0)
    numeric_tolerance_meters: float = Field(default=0.05, ge=0.0)

    @field_validator("source_id", "target_id")
    @classmethod
    def _clean_ids(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("source_id and target_id must not be empty.")
        return cleaned

    @field_validator("min_required_meters", "numeric_tolerance_meters")
    @classmethod
    def _require_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Clearance distances and tolerances must be finite.")
        return value

    @model_validator(mode="after")
    def _validate_clearance_mode(self) -> ClearanceQuery:
        if self.source_id == self.target_id:
            raise ValueError("source_id and target_id must refer to distinct scene elements.")
        if self.answer_type == AnswerType.BOOLEAN and self.min_required_meters is None:
            raise ValueError("min_required_meters is required when answer_type is BOOLEAN.")
        if self.answer_type == AnswerType.NUMERIC and self.min_required_meters is not None:
            raise ValueError("min_required_meters must be None when answer_type is NUMERIC.")
        return self


SolverQuery = Union[ReachabilityQuery, RegionCountingQuery, ClearanceQuery]


class ReachabilityResult(BaseModel):
    """Detailed deterministic reachability evaluation output."""

    model_config = ConfigDict(extra="forbid")

    is_reachable: bool
    shortest_exit_id: str | None = None
    reachable_exit_distances_m: dict[str, float]
    unreachable_exit_ids: list[str]
    grid_resolution: float
    start_world_point: Point2D
    path_world_points: list[Point2D]
    reference_answer: ReferenceAnswer | None = None


class RegionCountingResult(BaseModel):
    """Detailed deterministic region counting output."""

    model_config = ConfigDict(extra="forbid")

    zone_id: str
    count: int = Field(..., ge=0)
    matched_entity_ids: list[str]
    containment_mode: ContainmentMode
    reference_answer: ReferenceAnswer


class ClearanceResult(BaseModel):
    """Detailed deterministic clearance evaluation output."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    distance_world_units: float = Field(..., ge=0.0)
    distance_meters: float = Field(..., ge=0.0)
    touching_or_overlapping: bool
    reference_answer: ReferenceAnswer


_EPS = 1e-9


def _point_to_rect_distance(px: float, py: float, rect: Rect2D) -> float:
    dx = max(0.0, rect.x - px, px - rect.x2)
    dy = max(0.0, rect.y - py, py - rect.y2)
    return math.hypot(dx, dy)


def _point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    vx = bx - ax
    vy = by - ay
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= _EPS:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / seg_len_sq))
    proj_x = ax + t * vx
    proj_y = ay + t * vy
    return math.hypot(px - proj_x, py - proj_y)


def _segments_intersect(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> bool:
    def _orient(p1x: float, p1y: float, p2x: float, p2y: float, p3x: float, p3y: float) -> float:
        return (p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x)

    def _on_seg(p1x: float, p1y: float, p2x: float, p2y: float, qx: float, qy: float) -> bool:
        return (
            min(p1x, p2x) - _EPS <= qx <= max(p1x, p2x) + _EPS
            and min(p1y, p2y) - _EPS <= qy <= max(p1y, p2y) + _EPS
        )

    o1 = _orient(ax, ay, bx, by, cx, cy)
    o2 = _orient(ax, ay, bx, by, dx, dy)
    o3 = _orient(cx, cy, dx, dy, ax, ay)
    o4 = _orient(cx, cy, dx, dy, bx, by)

    if ((o1 > _EPS and o2 < -_EPS) or (o1 < -_EPS and o2 > _EPS)) and (
        (o3 > _EPS and o4 < -_EPS) or (o3 < -_EPS and o4 > _EPS)
    ):
        return True
    if abs(o1) <= _EPS and _on_seg(ax, ay, bx, by, cx, cy):
        return True
    if abs(o2) <= _EPS and _on_seg(ax, ay, bx, by, dx, dy):
        return True
    if abs(o3) <= _EPS and _on_seg(cx, cy, dx, dy, ax, ay):
        return True
    if abs(o4) <= _EPS and _on_seg(cx, cy, dx, dy, bx, by):
        return True
    return False


def _segment_to_rect_distance(start: Point2D, end: Point2D, rect: Rect2D) -> float:
    if (
        rect.x - _EPS <= start.x <= rect.x2 + _EPS
        and rect.y - _EPS <= start.y <= rect.y2 + _EPS
    ) or (
        rect.x - _EPS <= end.x <= rect.x2 + _EPS
        and rect.y - _EPS <= end.y <= rect.y2 + _EPS
    ):
        return 0.0

    rect_edges = (
        (rect.x, rect.y, rect.x2, rect.y),
        (rect.x2, rect.y, rect.x2, rect.y2),
        (rect.x2, rect.y2, rect.x, rect.y2),
        (rect.x, rect.y2, rect.x, rect.y),
    )
    for ex0, ey0, ex1, ey1 in rect_edges:
        if _segments_intersect(start.x, start.y, end.x, end.y, ex0, ey0, ex1, ey1):
            return 0.0

    corners = (
        (rect.x, rect.y),
        (rect.x2, rect.y),
        (rect.x2, rect.y2),
        (rect.x, rect.y2),
    )
    return min(
        _point_to_rect_distance(start.x, start.y, rect),
        _point_to_rect_distance(end.x, end.y, rect),
        *(
            _point_to_segment_distance(cx, cy, start.x, start.y, end.x, end.y)
            for cx, cy in corners
        ),
    )


def _rect_to_rect_distance(a: Rect2D, b: Rect2D) -> float:
    dx = max(0.0, a.x - b.x2, b.x - a.x2)
    dy = max(0.0, a.y - b.y2, b.y - a.y2)
    return math.hypot(dx, dy)


def _segment_to_segment_distance(a0: Point2D, a1: Point2D, b0: Point2D, b1: Point2D) -> float:
    if _segments_intersect(a0.x, a0.y, a1.x, a1.y, b0.x, b0.y, b1.x, b1.y):
        return 0.0
    return min(
        _point_to_segment_distance(a0.x, a0.y, b0.x, b0.y, b1.x, b1.y),
        _point_to_segment_distance(a1.x, a1.y, b0.x, b0.y, b1.x, b1.y),
        _point_to_segment_distance(b0.x, b0.y, a0.x, a0.y, a1.x, a1.y),
        _point_to_segment_distance(b1.x, b1.y, a0.x, a0.y, a1.x, a1.y),
    )


def _door_carve_box(scene: SceneSpec, door: Doorway) -> Rect2D:
    half_len = door.width / 2.0
    cross_half = max(0.4, scene.grid_resolution * 0.75)
    if door.orientation == "vertical":
        x0 = max(0.0, door.position.x - cross_half)
        y0 = max(0.0, door.position.y - half_len)
        x1 = min(scene.width, door.position.x + cross_half)
        y1 = min(scene.height, door.position.y + half_len)
    else:
        x0 = max(0.0, door.position.x - half_len)
        y0 = max(0.0, door.position.y - cross_half)
        x1 = min(scene.width, door.position.x + half_len)
        y1 = min(scene.height, door.position.y + cross_half)
    return Rect2D(x=x0, y=y0, width=max(0.05, x1 - x0), height=max(0.05, y1 - y0))


def _exit_opening_box(scene: SceneSpec, exit_marker: ExitMarker) -> Rect2D:
    dist_w = exit_marker.position.x
    dist_e = scene.width - exit_marker.position.x
    dist_n = exit_marker.position.y
    dist_s = scene.height - exit_marker.position.y
    nearest = min(
        [("W", dist_w), ("E", dist_e), ("N", dist_n), ("S", dist_s)],
        key=lambda item: item[1],
    )[0]

    along_half = max(scene.grid_resolution * 0.5, exit_marker.width / 2.0)
    cross_half = max(0.75, scene.grid_resolution * 1.2)
    if nearest in ("N", "S"):
        x0 = max(0.0, exit_marker.position.x - along_half)
        x1 = min(scene.width, exit_marker.position.x + along_half)
        y0 = max(0.0, exit_marker.position.y - cross_half)
        y1 = min(scene.height, exit_marker.position.y + cross_half)
    else:
        x0 = max(0.0, exit_marker.position.x - cross_half)
        x1 = min(scene.width, exit_marker.position.x + cross_half)
        y0 = max(0.0, exit_marker.position.y - along_half)
        y1 = min(scene.height, exit_marker.position.y + along_half)

    return Rect2D(x=x0, y=y0, width=max(0.05, x1 - x0), height=max(0.05, y1 - y0))


class GroundTruthSolver:
    """Computes independent, deterministic answers from SceneSpec geometry."""

    REACHABILITY_SOLVER_NAME = "grid_dijkstra_v1"
    COUNTING_SOLVER_NAME = "rect_containment_v1"
    CLEARANCE_SOLVER_NAME = "euclidean_clearance_v1"

    def solve(self, scene: SceneSpec, query: SolverQuery) -> ReferenceAnswer:
        """Solve a supported spatial query and return a validated ReferenceAnswer."""
        self._require_valid_scene(scene)

        if isinstance(query, ReachabilityQuery):
            res = self.solve_reachability(scene, query)
            if not res.is_reachable or res.reference_answer is None:
                raise UnsolvableTaskError(
                    f"No candidate exit is reachable from {res.start_world_point} in scene {scene.scene_id!r}."
                )
            return res.reference_answer

        if isinstance(query, RegionCountingQuery):
            return self.solve_region_counting(scene, query).reference_answer

        if isinstance(query, ClearanceQuery):
            return self.solve_clearance(scene, query).reference_answer

        raise UnsupportedTaskError(f"Unsupported solver query type: {type(query).__name__}.")

    def solve_reachability(
        self,
        scene: SceneSpec,
        query: ReachabilityQuery,
    ) -> ReachabilityResult:
        """Evaluate constrained grid reachability from start to candidate exits."""
        self._require_valid_scene(scene)

        res = scene.grid_resolution
        nx = max(1, math.ceil(scene.width / res))
        ny = max(1, math.ceil(scene.height / res))

        exits_by_id = {exit_marker.id: exit_marker for exit_marker in scene.exits}
        if query.candidate_exit_ids is not None:
            unknown = [eid for eid in query.candidate_exit_ids if eid not in exits_by_id]
            if unknown:
                raise UnsolvableTaskError(
                    f"Unknown candidate exit IDs {unknown} in scene {scene.scene_id!r}."
                )
            candidate_ids = list(query.candidate_exit_ids)
        else:
            candidate_ids = [exit_marker.id for exit_marker in scene.exits]

        open_door_boxes = [_door_carve_box(scene, d) for d in scene.doors if not d.is_blocked]
        blocked_door_boxes = [_door_carve_box(scene, d) for d in scene.doors if d.is_blocked]
        exit_boxes = {eid: _exit_opening_box(scene, exits_by_id[eid]) for eid in candidate_ids}
        all_exit_boxes = [_exit_opening_box(scene, e) for e in scene.exits]

        solid_entity_boxes = [ent.bounds for ent in scene.entities if not ent.passable]
        forbidden_zone_boxes = [
            z.bounds for z in scene.zones if z.zone_type in query.forbidden_zone_types
        ]

        blocked = [[False] * nx for _ in range(ny)]
        exit_cells: dict[str, list[tuple[int, int]]] = {eid: [] for eid in candidate_ids}

        for row in range(ny):
            cy0 = row * res
            ch = min(res, scene.height - cy0)
            if ch <= _EPS:
                continue
            for col in range(nx):
                cx0 = col * res
                cw = min(res, scene.width - cx0)
                if cw <= _EPS:
                    continue
                cell_rect = Rect2D(x=cx0, y=cy0, width=cw, height=ch)

                # Walls block intersecting cells unless carved by an open doorway or perimeter exit opening.
                in_wall = any(
                    _segment_to_rect_distance(w.start, w.end, cell_rect) <= w.thickness / 2.0 + _EPS
                    for w in scene.walls
                )
                in_open_door = any(cell_rect.overlaps_interior(box) for box in open_door_boxes)
                in_exit_opening = any(cell_rect.overlaps_interior(box) for box in all_exit_boxes)
                in_blocked_door = any(cell_rect.overlaps_interior(box) for box in blocked_door_boxes)

                if (in_wall and not in_open_door and not in_exit_opening) or in_blocked_door:
                    blocked[row][col] = True
                    continue

                if any(cell_rect.overlaps_interior(box) for box in solid_entity_boxes):
                    blocked[row][col] = True
                    continue

                if any(cell_rect.overlaps_interior(box) for box in forbidden_zone_boxes):
                    blocked[row][col] = True
                    continue

                for eid, ebox in exit_boxes.items():
                    if cell_rect.overlaps_interior(ebox):
                        exit_cells[eid].append((col, row))

        start_col, start_row, start_world = self._resolve_start_cell(
            scene=scene,
            query=query,
            blocked=blocked,
            nx=nx,
            ny=ny,
            res=res,
        )

        # 8-connected Dijkstra over passable grid cells with corner-cutting prevention.
        dist_grid: list[list[float]] = [[math.inf] * nx for _ in range(ny)]
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        dist_grid[start_row][start_col] = 0.0
        pq: list[tuple[float, int, int]] = [(0.0, start_col, start_row)]

        neighbors = (
            (1, 0, res),
            (-1, 0, res),
            (0, 1, res),
            (0, -1, res),
            (1, 1, res * math.sqrt(2.0)),
            (1, -1, res * math.sqrt(2.0)),
            (-1, 1, res * math.sqrt(2.0)),
            (-1, -1, res * math.sqrt(2.0)),
        )

        while pq:
            cur_d, cx, cy = heapq.heappop(pq)
            if cur_d > dist_grid[cy][cx] + 1e-12:
                continue
            for dx, dy, step_cost in neighbors:
                nx_col = cx + dx
                ny_row = cy + dy
                if not (0 <= nx_col < nx and 0 <= ny_row < ny):
                    continue
                if blocked[ny_row][nx_col]:
                    continue
                if dx != 0 and dy != 0:
                    # Prevent diagonal corner-cutting across blocked cardinal neighbors.
                    if blocked[cy][nx_col] or blocked[ny_row][cx]:
                        continue
                cand_d = cur_d + step_cost
                if cand_d + 1e-12 < dist_grid[ny_row][nx_col]:
                    dist_grid[ny_row][nx_col] = cand_d
                    parent[(nx_col, ny_row)] = (cx, cy)
                    heapq.heappush(pq, (cand_d, nx_col, ny_row))

        m_per_unit = scene.scale_bar.meters_per_world_unit
        reachable_distances_m: dict[str, float] = {}
        best_exit_cell: dict[str, tuple[int, int]] = {}
        unreachable_ids: list[str] = []

        for eid in candidate_ids:
            best_d = math.inf
            best_cell: tuple[int, int] | None = None
            for ex_col, ex_row in exit_cells[eid]:
                d = dist_grid[ex_row][ex_col]
                if d < best_d:
                    best_d = d
                    best_cell = (ex_col, ex_row)
            if best_cell is not None and math.isfinite(best_d):
                reachable_distances_m[eid] = round(best_d * m_per_unit, 4)
                best_exit_cell[eid] = best_cell
            else:
                unreachable_ids.append(eid)

        if not reachable_distances_m:
            return ReachabilityResult(
                is_reachable=False,
                shortest_exit_id=None,
                reachable_exit_distances_m={},
                unreachable_exit_ids=unreachable_ids,
                grid_resolution=res,
                start_world_point=start_world,
                path_world_points=[],
                reference_answer=None,
            )

        # Deterministic selection: minimum distance in meters, alphabetical exit ID tie-break.
        shortest_exit_id = min(
            reachable_distances_m.keys(),
            key=lambda eid: (round(reachable_distances_m[eid], 6), eid),
        )
        goal_cell = best_exit_cell[shortest_exit_id]
        path_cells: list[tuple[int, int]] = [goal_cell]
        cur = goal_cell
        while cur in parent:
            cur = parent[cur]
            path_cells.append(cur)
        path_cells.reverse()

        path_points = [
            Point2D(
                x=min(scene.width, (col + 0.5) * res),
                y=min(scene.height, (row + 0.5) * res),
            )
            for col, row in path_cells
        ]

        ref_answer = ReferenceAnswer(
            answer_type=AnswerType.EXIT_ID,
            value=shortest_exit_id,
            solver_name=self.REACHABILITY_SOLVER_NAME,
            reasoning_trace=[
                f"Grid resolution={res} world units ({nx}x{ny} cells).",
                f"Start cell world center=({start_world.x:.2f}, {start_world.y:.2f}).",
                f"Reachable exits (m): {reachable_distances_m}; unreachable={unreachable_ids}.",
                f"Selected {shortest_exit_id} ({reachable_distances_m[shortest_exit_id]:.4f} m).",
            ],
        )

        return ReachabilityResult(
            is_reachable=True,
            shortest_exit_id=shortest_exit_id,
            reachable_exit_distances_m=reachable_distances_m,
            unreachable_exit_ids=unreachable_ids,
            grid_resolution=res,
            start_world_point=start_world,
            path_world_points=path_points,
            reference_answer=ref_answer,
        )

    def _resolve_start_cell(
        self,
        scene: SceneSpec,
        query: ReachabilityQuery,
        blocked: list[list[bool]],
        nx: int,
        ny: int,
        res: float,
    ) -> tuple[int, int, Point2D]:
        if query.start_point is not None:
            pt = query.start_point
            if pt.x > scene.width or pt.y > scene.height:
                raise UnsolvableTaskError(
                    f"start_point ({pt.x}, {pt.y}) lies outside scene bounds ({scene.width}, {scene.height})."
                )
            col = min(nx - 1, max(0, int(pt.x / res)))
            row = min(ny - 1, max(0, int(pt.y / res)))
            if blocked[row][col]:
                raise UnsolvableTaskError(
                    f"start_point ({pt.x}, {pt.y}) falls inside a blocked cell."
                )
            return col, row, Point2D(
                x=min(scene.width, (col + 0.5) * res),
                y=min(scene.height, (row + 0.5) * res),
            )

        assert query.start_zone_id is not None
        zones_by_id = {z.id: z for z in scene.zones}
        if query.start_zone_id not in zones_by_id:
            raise UnsolvableTaskError(
                f"Unknown start_zone_id {query.start_zone_id!r} in scene {scene.scene_id!r}."
            )
        zone = zones_by_id[query.start_zone_id]
        if zone.zone_type in query.forbidden_zone_types:
            raise UnsolvableTaskError(
                f"start_zone_id {zone.id!r} has forbidden zone_type {zone.zone_type.value!r}."
            )

        zc = zone.bounds.center
        best_cell: tuple[int, int] | None = None
        best_dist = math.inf
        for row in range(ny):
            cy = (row + 0.5) * res
            if not (zone.bounds.y - _EPS <= cy <= zone.bounds.y2 + _EPS):
                continue
            for col in range(nx):
                cx = (col + 0.5) * res
                if not (zone.bounds.x - _EPS <= cx <= zone.bounds.x2 + _EPS):
                    continue
                if blocked[row][col]:
                    continue
                d = math.hypot(cx - zc.x, cy - zc.y)
                if d < best_dist - 1e-12:
                    best_dist = d
                    best_cell = (col, row)

        if best_cell is None:
            raise UnsolvableTaskError(
                f"No passable starting cell found inside zone {zone.id!r}."
            )
        col, row = best_cell
        return col, row, Point2D(
            x=min(scene.width, (col + 0.5) * res),
            y=min(scene.height, (row + 0.5) * res),
        )

    def solve_region_counting(
        self,
        scene: SceneSpec,
        query: RegionCountingQuery,
    ) -> RegionCountingResult:
        """Count unique eligible entities inside a named zone under containment_mode."""
        self._require_valid_scene(scene)

        zones_by_id = {z.id: z for z in scene.zones}
        if query.zone_id not in zones_by_id:
            raise UnsolvableTaskError(
                f"Unknown zone_id {query.zone_id!r} in scene {scene.scene_id!r}."
            )
        zone = zones_by_id[query.zone_id]

        matched_ids: list[str] = []
        seen_ids: set[str] = set()
        for ent in scene.entities:
            if ent.id in seen_ids:
                continue
            seen_ids.add(ent.id)

            if not query.include_passable and ent.passable:
                continue
            if query.entity_types is not None and ent.entity_type not in query.entity_types:
                continue

            if self._entity_matches_zone(ent, zone, query.containment_mode):
                matched_ids.append(ent.id)

        count = len(matched_ids)
        filter_desc = (
            [et.value for et in query.entity_types] if query.entity_types is not None else "all"
        )
        ref_answer = ReferenceAnswer(
            answer_type=AnswerType.INTEGER,
            value=count,
            solver_name=self.COUNTING_SOLVER_NAME,
            reasoning_trace=[
                f"Zone {zone.id} bounds=({zone.bounds.x}, {zone.bounds.y}, {zone.bounds.x2}, {zone.bounds.y2}).",
                f"Containment mode={query.containment_mode.value}; entity_types={filter_desc}.",
                f"Matched {count} unique entities: {matched_ids}.",
            ],
        )

        return RegionCountingResult(
            zone_id=zone.id,
            count=count,
            matched_entity_ids=matched_ids,
            containment_mode=query.containment_mode,
            reference_answer=ref_answer,
        )

    def _entity_matches_zone(
        self,
        entity: SceneEntity,
        zone: Zone,
        mode: ContainmentMode,
    ) -> bool:
        eb = entity.bounds
        zb = zone.bounds
        if mode == ContainmentMode.STRICT_INSIDE:
            return (
                eb.x >= zb.x - _EPS
                and eb.y >= zb.y - _EPS
                and eb.x2 <= zb.x2 + _EPS
                and eb.y2 <= zb.y2 + _EPS
            )
        if mode == ContainmentMode.CENTER_INSIDE:
            center = eb.center
            return (
                zb.x - _EPS <= center.x <= zb.x2 + _EPS
                and zb.y - _EPS <= center.y <= zb.y2 + _EPS
            )
        if mode == ContainmentMode.ANY_OVERLAP:
            return eb.overlaps_interior(zb)
        raise UnsupportedTaskError(f"Unsupported containment_mode: {mode!r}.")

    def solve_clearance(
        self,
        scene: SceneSpec,
        query: ClearanceQuery,
    ) -> ClearanceResult:
        """Compute exact world-coordinate clearance converted to meters via ScaleBarSpec."""
        self._require_valid_scene(scene)

        geom_a = self._lookup_geometry(scene, query.source_id)
        geom_b = self._lookup_geometry(scene, query.target_id)

        dist_world = self._geometry_distance(geom_a, geom_b)
        dist_meters = round(dist_world * scene.scale_bar.meters_per_world_unit, 6)
        touching_or_overlapping = dist_world <= _EPS

        if query.answer_type == AnswerType.BOOLEAN:
            assert query.min_required_meters is not None
            passed = dist_meters + _EPS >= query.min_required_meters
            ref_answer = ReferenceAnswer(
                answer_type=AnswerType.BOOLEAN,
                value=passed,
                solver_name=self.CLEARANCE_SOLVER_NAME,
                reasoning_trace=[
                    f"Distance({query.source_id}, {query.target_id}) = {dist_world:.6f} world units.",
                    f"Scale = {scene.scale_bar.meters_per_world_unit:.6f} m/unit -> {dist_meters:.6f} m.",
                    f"Threshold check: {dist_meters:.6f} m >= {query.min_required_meters:.6f} m -> {passed}.",
                ],
            )
        else:
            ref_answer = ReferenceAnswer(
                answer_type=AnswerType.NUMERIC,
                value=dist_meters,
                unit="m",
                numeric_tolerance=query.numeric_tolerance_meters,
                solver_name=self.CLEARANCE_SOLVER_NAME,
                reasoning_trace=[
                    f"Distance({query.source_id}, {query.target_id}) = {dist_world:.6f} world units.",
                    f"Scale = {scene.scale_bar.meters_per_world_unit:.6f} m/unit -> {dist_meters:.6f} m.",
                ],
            )

        return ClearanceResult(
            source_id=query.source_id,
            target_id=query.target_id,
            distance_world_units=round(dist_world, 6),
            distance_meters=dist_meters,
            touching_or_overlapping=touching_or_overlapping,
            reference_answer=ref_answer,
        )

    def _lookup_geometry(
        self,
        scene: SceneSpec,
        element_id: str,
    ) -> Rect2D | Point2D | WallSegment:
        for ent in scene.entities:
            if ent.id == element_id:
                return ent.bounds
        for zone in scene.zones:
            if zone.id == element_id:
                return zone.bounds
        for exit_marker in scene.exits:
            if exit_marker.id == element_id:
                return exit_marker.position
        for door in scene.doors:
            if door.id == element_id:
                return door.position
        for wall in scene.walls:
            if wall.id == element_id:
                return wall
        raise UnsolvableTaskError(
            f"Scene element ID {element_id!r} not found in scene {scene.scene_id!r}."
        )

    def _geometry_distance(
        self,
        a: Rect2D | Point2D | WallSegment,
        b: Rect2D | Point2D | WallSegment,
    ) -> float:
        if isinstance(a, Rect2D) and isinstance(b, Rect2D):
            return _rect_to_rect_distance(a, b)
        if isinstance(a, Point2D) and isinstance(b, Point2D):
            return math.hypot(a.x - b.x, a.y - b.y)
        if isinstance(a, Rect2D) and isinstance(b, Point2D):
            return _point_to_rect_distance(b.x, b.y, a)
        if isinstance(a, Point2D) and isinstance(b, Rect2D):
            return _point_to_rect_distance(a.x, a.y, b)
        if isinstance(a, WallSegment):
            return self._wall_to_geometry_distance(a, b)
        if isinstance(b, WallSegment):
            return self._wall_to_geometry_distance(b, a)
        raise UnsupportedTaskError("Unsupported geometry combination for clearance calculation.")

    def _wall_to_geometry_distance(
        self,
        wall: WallSegment,
        other: Rect2D | Point2D | WallSegment,
    ) -> float:
        half_t = wall.thickness / 2.0
        if isinstance(other, Rect2D):
            raw = _segment_to_rect_distance(wall.start, wall.end, other)
            return max(0.0, raw - half_t)
        if isinstance(other, Point2D):
            raw = _point_to_segment_distance(
                other.x, other.y, wall.start.x, wall.start.y, wall.end.x, wall.end.y
            )
            return max(0.0, raw - half_t)
        if isinstance(other, WallSegment):
            raw = _segment_to_segment_distance(wall.start, wall.end, other.start, other.end)
            return max(0.0, raw - half_t - other.thickness / 2.0)
        raise UnsupportedTaskError("Unsupported geometry combination for wall clearance.")

    def _require_valid_scene(self, scene: SceneSpec) -> None:
        if not isinstance(scene, SceneSpec):
            raise TypeError(f"Expected SceneSpec instance, got {type(scene).__name__}.")


def solve_scene(scene: SceneSpec, query: SolverQuery) -> ReferenceAnswer:
    """Convenience entry point to solve a spatial reasoning query on a SceneSpec."""
    return GroundTruthSolver().solve(scene=scene, query=query)
