"""Additive 3D warehouse scene data models (Pydantic v2)."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from visual_task_factory.models.scene import (
    EntityType,
    Point2D,
    Rect2D,
    ScaleBarSpec,
    SceneSpec,
    ZoneType,
    _validate_identifier,
    _validate_label,
)
from visual_task_factory.models.variation import VariationConfig

_EPS = 1e-6

_DEFAULT_ENTITY_HEIGHTS_M: dict[EntityType, float] = {
    EntityType.PALLET_RACK: 3.0,
    EntityType.FORKLIFT: 2.1,
    EntityType.CRATE: 1.2,
    EntityType.PILLAR: 4.5,
    EntityType.CONVEYOR: 1.1,
    EntityType.ROBOT: 0.8,
}


def _require_finite(value: float, field_name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite.")
    return value


class Point3D(BaseModel):
    """3D coordinate in right-handed room units (X=width, Y=depth, Z=up)."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(..., ge=0.0)
    y: float = Field(..., ge=0.0)
    z: float = Field(default=0.0, ge=0.0)

    @field_validator("x", "y", "z")
    @classmethod
    def _validate_finite_coords(cls, value: float, info) -> float:
        return _require_finite(value, f"Point3D.{info.field_name}")


class Box3D(BaseModel):
    """Axis-aligned 3D bounding box anchored at minimum corner (x, y, z)."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(..., ge=0.0)
    y: float = Field(..., ge=0.0)
    z: float = Field(default=0.0, ge=0.0)
    width: float = Field(..., gt=0.0)
    depth: float = Field(..., gt=0.0)
    height: float = Field(..., gt=0.0)

    @field_validator("x", "y", "z", "width", "depth", "height")
    @classmethod
    def _validate_finite_fields(cls, value: float, info) -> float:
        return _require_finite(value, f"Box3D.{info.field_name}")

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.depth

    @property
    def z2(self) -> float:
        return self.z + self.height

    @property
    def center(self) -> Point3D:
        return Point3D(
            x=self.x + self.width / 2.0,
            y=self.y + self.depth / 2.0,
            z=self.z + self.height / 2.0,
        )

    @property
    def footprint(self) -> Rect2D:
        return Rect2D(x=self.x, y=self.y, width=self.width, height=self.depth)

    @property
    def volume(self) -> float:
        return self.width * self.depth * self.height

    def overlaps_xy(self, other: Box3D, eps: float = _EPS) -> bool:
        """Return True if XY footprints have positive interior overlap."""
        return (
            self.x < other.x2 - eps
            and self.x2 > other.x + eps
            and self.y < other.y2 - eps
            and self.y2 > other.y + eps
        )

    def overlaps_interior(self, other: Box3D, eps: float = _EPS) -> bool:
        """Return True if 3D volumes have positive interior overlap (touching allowed)."""
        return (
            self.overlaps_xy(other, eps=eps)
            and self.z < other.z2 - eps
            and self.z2 > other.z + eps
        )


class WallSegment3D(BaseModel):
    """Vertical 3D wall slab extruded upward from (start, end) on floor plane."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    start: Point2D
    end: Point2D
    z_min: float = Field(default=0.0, ge=0.0)
    height: float = Field(default=4.0, gt=0.0)
    thickness: float = Field(default=0.35, gt=0.0)
    is_exterior: bool = False

    @field_validator("id")
    @classmethod
    def _clean_id(cls, value: str) -> str:
        return _validate_identifier(value, "WallSegment3D.id")

    @field_validator("z_min", "height", "thickness")
    @classmethod
    def _validate_finite_scalars(cls, value: float, info) -> float:
        return _require_finite(value, f"WallSegment3D.{info.field_name}")

    @model_validator(mode="after")
    def _validate_non_zero_length(self) -> WallSegment3D:
        if math.isclose(self.start.x, self.end.x, abs_tol=1e-9) and math.isclose(
            self.start.y, self.end.y, abs_tol=1e-9
        ):
            raise ValueError("WallSegment3D start and end points must not coincide.")
        return self

    @property
    def z_max(self) -> float:
        return self.z_min + self.height

    @property
    def length(self) -> float:
        return math.hypot(self.end.x - self.start.x, self.end.y - self.start.y)


class Doorway3D(BaseModel):
    """3D doorway opening positioned along a horizontal or vertical wall span."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    position: Point3D
    width: float = Field(..., gt=0.0)
    height: float = Field(default=2.4, gt=0.0)
    orientation: Literal["horizontal", "vertical"] = "horizontal"
    is_blocked: bool = False

    @field_validator("id")
    @classmethod
    def _clean_id(cls, value: str) -> str:
        return _validate_identifier(value, "Doorway3D.id")

    @field_validator("width", "height")
    @classmethod
    def _validate_finite_scalars(cls, value: float, info) -> float:
        return _require_finite(value, f"Doorway3D.{info.field_name}")


class Zone3D(BaseModel):
    """Named floor region inside the 3D warehouse."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    zone_type: ZoneType
    bounds: Rect2D
    z_floor: float = Field(default=0.0, ge=0.0)

    @field_validator("id")
    @classmethod
    def _clean_id(cls, value: str) -> str:
        return _validate_identifier(value, "Zone3D.id")

    @field_validator("label")
    @classmethod
    def _clean_label(cls, value: str) -> str:
        return _validate_label(value, "Zone3D.label")

    @field_validator("z_floor")
    @classmethod
    def _validate_finite_floor(cls, value: float) -> float:
        return _require_finite(value, "Zone3D.z_floor")


class ExitMarker3D(BaseModel):
    """3D exit marker/portal with width and height at a room boundary or opening."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    position: Point3D
    width: float = Field(default=2.0, gt=0.0)
    height: float = Field(default=2.5, gt=0.0)
    is_emergency_exit: bool = True

    @field_validator("id")
    @classmethod
    def _clean_id(cls, value: str) -> str:
        return _validate_identifier(value, "ExitMarker3D.id")

    @field_validator("label")
    @classmethod
    def _clean_label(cls, value: str) -> str:
        return _validate_label(value, "ExitMarker3D.label")

    @field_validator("width", "height")
    @classmethod
    def _validate_finite_dims(cls, value: float, info) -> float:
        return _require_finite(value, f"ExitMarker3D.{info.field_name}")


class SceneEntity3D(BaseModel):
    """Axis-aligned 3D warehouse object with optional stacking support reference."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    entity_type: EntityType
    bounds: Box3D
    passable: bool = False
    supported_by: str | None = None

    @field_validator("id")
    @classmethod
    def _clean_id(cls, value: str) -> str:
        return _validate_identifier(value, "SceneEntity3D.id")

    @field_validator("label")
    @classmethod
    def _clean_label(cls, value: str) -> str:
        return _validate_label(value, "SceneEntity3D.label")

    @field_validator("supported_by")
    @classmethod
    def _clean_supported_by(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_identifier(value, "SceneEntity3D.supported_by")

    @model_validator(mode="after")
    def _validate_support_self_reference(self) -> SceneEntity3D:
        if self.supported_by is not None and self.supported_by == self.id:
            raise ValueError("SceneEntity3D.supported_by cannot reference itself.")
        return self


class CameraPose3D(BaseModel):
    """Orbit-based perspective camera specification for 3D scene viewpoints."""

    model_config = ConfigDict(extra="forbid")

    look_at: Point3D
    distance: float = Field(default=28.0, gt=0.0)
    azimuth_deg: float = Field(default=45.0, ge=0.0, lt=360.0)
    elevation_deg: float = Field(default=35.0, gt=0.0, le=85.0)
    fov_deg: float = Field(default=45.0, ge=20.0, le=120.0)

    @field_validator("distance", "azimuth_deg", "elevation_deg", "fov_deg")
    @classmethod
    def _validate_finite_camera_params(cls, value: float, info) -> float:
        return _require_finite(value, f"CameraPose3D.{info.field_name}")

    @property
    def eye(self) -> tuple[float, float, float]:
        """Compute camera world eye coordinates (x, y, z) from orbit pose."""
        az = math.radians(self.azimuth_deg)
        el = math.radians(self.elevation_deg)
        horiz = self.distance * math.cos(el)
        return (
            self.look_at.x + horiz * math.cos(az),
            self.look_at.y + horiz * math.sin(az),
            self.look_at.z + self.distance * math.sin(el),
        )


class SceneSpec3D(BaseModel):
    """Complete additive 3D specification of a generated warehouse environment."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    scene_id: str = Field(..., min_length=1)
    width: float = Field(..., gt=0.0)
    depth: float = Field(..., gt=0.0)
    ceiling_height: float = Field(default=4.5, gt=0.0)
    grid_resolution: float = Field(default=0.5, gt=0.0)
    scale_bar: ScaleBarSpec
    variation: VariationConfig
    camera: CameraPose3D
    walls: list[WallSegment3D] = Field(default_factory=list)
    doors: list[Doorway3D] = Field(default_factory=list)
    zones: list[Zone3D] = Field(default_factory=list)
    exits: list[ExitMarker3D] = Field(..., min_length=1)
    entities: list[SceneEntity3D] = Field(default_factory=list)

    @field_validator("scene_id")
    @classmethod
    def _clean_scene_id(cls, value: str) -> str:
        return _validate_identifier(value, "SceneSpec3D.scene_id")

    @field_validator("width", "depth", "ceiling_height", "grid_resolution")
    @classmethod
    def _validate_finite_room_dims(cls, value: float, info) -> float:
        return _require_finite(value, f"SceneSpec3D.{info.field_name}")

    @model_validator(mode="after")
    def _validate_scene_3d_integrity(self) -> SceneSpec3D:
        if self.grid_resolution > min(self.width, self.depth, self.ceiling_height):
            raise ValueError("grid_resolution must not exceed min(width, depth, ceiling_height).")

        if self.scale_bar.world_units > self.width:
            raise ValueError("scale_bar.world_units must not exceed scene width.")

        seen_ids: set[str] = set()
        for group in (self.walls, self.doors, self.zones, self.exits, self.entities):
            for item in group:
                if item.id in seen_ids:
                    raise ValueError(f"Duplicate element id detected in SceneSpec3D: {item.id!r}")
                seen_ids.add(item.id)

        if (
            self.camera.look_at.x > self.width + _EPS
            or self.camera.look_at.y > self.depth + _EPS
            or self.camera.look_at.z > self.ceiling_height + _EPS
        ):
            raise ValueError("CameraPose3D.look_at exceeds room bounds.")

        for wall in self.walls:
            for pt_name, pt in (("start", wall.start), ("end", wall.end)):
                if pt.x > self.width + _EPS or pt.y > self.depth + _EPS:
                    raise ValueError(
                        f"Wall {wall.id!r} {pt_name} point ({pt.x}, {pt.y}) exceeds room footprint."
                    )
            if wall.z_max > self.ceiling_height + _EPS:
                raise ValueError(
                    f"Wall {wall.id!r} top z_max={wall.z_max} exceeds ceiling_height={self.ceiling_height}."
                )

        for door in self.doors:
            if door.position.x > self.width + _EPS or door.position.y > self.depth + _EPS:
                raise ValueError(f"Door {door.id!r} position exceeds room footprint.")
            if door.position.z + door.height > self.ceiling_height + _EPS:
                raise ValueError(
                    f"Door {door.id!r} top exceeds ceiling_height={self.ceiling_height}."
                )

        for zone in self.zones:
            if zone.bounds.x2 > self.width + _EPS or zone.bounds.y2 > self.depth + _EPS:
                raise ValueError(f"Zone {zone.id!r} bounds exceed room footprint.")
            if zone.z_floor > self.ceiling_height + _EPS:
                raise ValueError(f"Zone {zone.id!r} z_floor exceeds ceiling_height.")

        for exit_marker in self.exits:
            if (
                exit_marker.position.x > self.width + _EPS
                or exit_marker.position.y > self.depth + _EPS
                or exit_marker.position.z + exit_marker.height > self.ceiling_height + _EPS
            ):
                raise ValueError(f"Exit {exit_marker.id!r} exceeds 3D room bounds.")

        entities_by_id: dict[str, SceneEntity3D] = {}
        for entity in self.entities:
            b = entity.bounds
            if (
                b.x2 > self.width + _EPS
                or b.y2 > self.depth + _EPS
                or b.z2 > self.ceiling_height + _EPS
            ):
                raise ValueError(f"Entity {entity.id!r} bounds exceed 3D room dimensions.")
            entities_by_id[entity.id] = entity

        # Check elevated / stacked entity support rules.
        for entity in self.entities:
            if entity.bounds.z > _EPS:
                if entity.supported_by is None:
                    raise ValueError(
                        f"Elevated entity {entity.id!r} (z={entity.bounds.z}) must specify supported_by."
                    )
                supporter = entities_by_id.get(entity.supported_by)
                if supporter is None or supporter.passable:
                    raise ValueError(
                        f"Entity {entity.id!r} references missing or passable supported_by={entity.supported_by!r}."
                    )
                if not math.isclose(supporter.bounds.z2, entity.bounds.z, abs_tol=_EPS):
                    raise ValueError(
                        f"Entity {entity.id!r} z={entity.bounds.z} does not rest flush on "
                        f"supporter {supporter.id!r} top z2={supporter.bounds.z2}."
                    )
                if not entity.bounds.overlaps_xy(supporter.bounds):
                    raise ValueError(
                        f"Entity {entity.id!r} XY footprint does not overlap supporter {supporter.id!r}."
                    )
            elif entity.supported_by is not None:
                raise ValueError(
                    f"Floor-level entity {entity.id!r} (z=0) must not specify supported_by."
                )

        # Pairwise non-overlap check for solid entities.
        solid_entities = [entity for entity in self.entities if not entity.passable]
        for idx, left in enumerate(solid_entities):
            for right in solid_entities[idx + 1 :]:
                if left.bounds.overlaps_interior(right.bounds):
                    raise ValueError(
                        f"Solid 3D entities {left.id!r} and {right.id!r} overlap in 3D space."
                    )

        return self


def extrude_scene_2d_to_3d(
    scene_2d: SceneSpec,
    *,
    ceiling_height: float = 4.5,
    camera: CameraPose3D | None = None,
    wall_height: float | None = None,
) -> SceneSpec3D:
    """Safely extrude an immutable V1 2D SceneSpec into a floor-grounded SceneSpec3D."""
    if not isinstance(scene_2d, SceneSpec):
        raise TypeError(f"Expected SceneSpec instance, got {type(scene_2d).__name__}.")
    _require_finite(ceiling_height, "ceiling_height")
    if ceiling_height <= 0.0:
        raise ValueError("ceiling_height must be > 0.0.")

    resolved_wall_height = ceiling_height if wall_height is None else wall_height
    _require_finite(resolved_wall_height, "wall_height")
    if resolved_wall_height <= 0.0 or resolved_wall_height > ceiling_height + _EPS:
        raise ValueError("wall_height must be in (0, ceiling_height].")

    resolved_door_height = min(2.5, resolved_wall_height)
    resolved_exit_height = min(2.5, ceiling_height)

    resolved_camera = camera or CameraPose3D(
        look_at=Point3D(
            x=scene_2d.width / 2.0,
            y=scene_2d.height / 2.0,
            z=min(1.0, ceiling_height * 0.25),
        ),
        distance=max(scene_2d.width, scene_2d.height) * 1.4,
        azimuth_deg=45.0,
        elevation_deg=35.0,
        fov_deg=45.0,
    )

    walls_3d = [
        WallSegment3D(
            id=w.id,
            start=w.start.model_copy(),
            end=w.end.model_copy(),
            z_min=0.0,
            height=resolved_wall_height,
            thickness=w.thickness,
            is_exterior=False,
        )
        for w in scene_2d.walls
    ]

    doors_3d = [
        Doorway3D(
            id=d.id,
            position=Point3D(x=d.position.x, y=d.position.y, z=0.0),
            width=d.width,
            height=resolved_door_height,
            orientation=d.orientation,
            is_blocked=d.is_blocked,
        )
        for d in scene_2d.doors
    ]

    zones_3d = [
        Zone3D(
            id=z.id,
            label=z.label,
            zone_type=z.zone_type,
            bounds=z.bounds.model_copy(),
            z_floor=0.0,
        )
        for z in scene_2d.zones
    ]

    exits_3d = [
        ExitMarker3D(
            id=e.id,
            label=e.label,
            position=Point3D(x=e.position.x, y=e.position.y, z=0.0),
            width=e.width,
            height=resolved_exit_height,
            is_emergency_exit=e.is_emergency_exit,
        )
        for e in scene_2d.exits
    ]

    entities_3d = [
        SceneEntity3D(
            id=ent.id,
            label=ent.label,
            entity_type=ent.entity_type,
            bounds=Box3D(
                x=ent.bounds.x,
                y=ent.bounds.y,
                z=0.0,
                width=ent.bounds.width,
                depth=ent.bounds.height,
                height=min(_DEFAULT_ENTITY_HEIGHTS_M[ent.entity_type], ceiling_height),
            ),
            passable=ent.passable,
            supported_by=None,
        )
        for ent in scene_2d.entities
    ]

    return SceneSpec3D(
        scene_id=scene_2d.scene_id,
        width=scene_2d.width,
        depth=scene_2d.height,
        ceiling_height=ceiling_height,
        grid_resolution=scene_2d.grid_resolution,
        scale_bar=scene_2d.scale_bar.model_copy(),
        variation=scene_2d.variation.model_copy(),
        camera=resolved_camera,
        walls=walls_3d,
        doors=doors_3d,
        zones=zones_3d,
        exits=exits_3d,
        entities=entities_3d,
    )
