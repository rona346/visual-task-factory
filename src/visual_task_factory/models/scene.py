"""Scene specification models for 2D facility spatial layouts."""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from visual_task_factory.models.variation import VariationConfig

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_identifier(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned or not _IDENTIFIER_PATTERN.match(cleaned):
        raise ValueError(
            f"{field_name} must be a non-empty alphanumeric/underscore/hyphen identifier; got {value!r}."
        )
    return cleaned


def _validate_label(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty or whitespace.")
    return cleaned


class Point2D(BaseModel):
    """2D point in world coordinates."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(..., ge=0.0)
    y: float = Field(..., ge=0.0)

    @field_validator("x", "y")
    @classmethod
    def _require_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Point2D coordinates must be finite.")
        return value


class Rect2D(BaseModel):
    """Axis-aligned 2D rectangle in world coordinates (origin at top-left)."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(..., ge=0.0)
    y: float = Field(..., ge=0.0)
    width: float = Field(..., gt=0.0)
    height: float = Field(..., gt=0.0)

    @field_validator("x", "y", "width", "height")
    @classmethod
    def _require_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Rect2D dimensions must be finite.")
        return value

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> Point2D:
        return Point2D(x=self.x + self.width / 2.0, y=self.y + self.height / 2.0)

    def overlaps_interior(self, other: Rect2D) -> bool:
        """Return True if two rectangles overlap with positive area."""
        return (
            self.x < other.x2
            and self.x2 > other.x
            and self.y < other.y2
            and self.y2 > other.y
        )


class WallSegment(BaseModel):
    """Impassable wall segment defined by two distinct endpoints."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    start: Point2D
    end: Point2D
    thickness: float = Field(default=0.3, gt=0.0)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return _validate_identifier(value, "WallSegment.id")

    @field_validator("thickness")
    @classmethod
    def _require_finite_thickness(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("WallSegment.thickness must be finite.")
        return value

    @model_validator(mode="after")
    def _check_non_zero_length(self) -> WallSegment:
        if self.start.x == self.end.x and self.start.y == self.end.y:
            raise ValueError(f"WallSegment '{self.id}' must have non-zero length.")
        return self


class Doorway(BaseModel):
    """Doorway opening that may be passable or blocked."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    position: Point2D
    width: float = Field(..., gt=0.0)
    orientation: Literal["horizontal", "vertical"]
    is_blocked: bool = False

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return _validate_identifier(value, "Doorway.id")

    @field_validator("width")
    @classmethod
    def _require_finite_width(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Doorway.width must be finite.")
        return value


class ZoneType(str, Enum):
    """Semantic zone classifications within the facility."""

    STORAGE = "storage"
    LOADING = "loading"
    HAZARD = "hazard"
    RESTRICTED = "restricted"
    STAGING = "staging"


class Zone(BaseModel):
    """Named rectangular zone within the facility floorplan."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    zone_type: ZoneType
    bounds: Rect2D

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return _validate_identifier(value, "Zone.id")

    @field_validator("label")
    @classmethod
    def _check_label(cls, value: str) -> str:
        return _validate_label(value, "Zone.label")


class ExitMarker(BaseModel):
    """Facility exit marker used as a target in reachability/evacuation tasks."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    position: Point2D
    width: float = Field(default=1.5, gt=0.0)
    is_emergency_exit: bool = True

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return _validate_identifier(value, "ExitMarker.id")

    @field_validator("label")
    @classmethod
    def _check_label(cls, value: str) -> str:
        return _validate_label(value, "ExitMarker.label")

    @field_validator("width")
    @classmethod
    def _require_finite_width(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("ExitMarker.width must be finite.")
        return value


class EntityType(str, Enum):
    """Physical objects and obstacles inside the facility."""

    PALLET_RACK = "pallet_rack"
    FORKLIFT = "forklift"
    CRATE = "crate"
    PILLAR = "pillar"
    CONVEYOR = "conveyor"
    ROBOT = "robot"


class SceneEntity(BaseModel):
    """Physical equipment, rack, or obstacle in the scene."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    entity_type: EntityType
    bounds: Rect2D
    passable: bool = False

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return _validate_identifier(value, "SceneEntity.id")

    @field_validator("label")
    @classmethod
    def _check_label(cls, value: str) -> str:
        return _validate_label(value, "SceneEntity.label")


class ScaleBarSpec(BaseModel):
    """Scale bar specification linking world coordinates to physical meters."""

    model_config = ConfigDict(extra="forbid")

    world_units: float = Field(..., gt=0.0)
    display_meters: float = Field(..., gt=0.0)
    unit_label: Literal["m"] = "m"

    @field_validator("world_units", "display_meters")
    @classmethod
    def _require_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("ScaleBarSpec measurements must be finite.")
        return value

    @property
    def meters_per_world_unit(self) -> float:
        return self.display_meters / self.world_units


class SceneSpec(BaseModel):
    """Complete deterministic specification of a 2D facility spatial scene."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    scene_id: str = Field(..., min_length=1)
    width: float = Field(..., gt=0.0)
    height: float = Field(..., gt=0.0)
    grid_resolution: float = Field(default=0.5, gt=0.0)
    scale_bar: ScaleBarSpec
    variation: VariationConfig
    walls: list[WallSegment] = Field(default_factory=list)
    doors: list[Doorway] = Field(default_factory=list)
    zones: list[Zone] = Field(default_factory=list)
    exits: list[ExitMarker] = Field(..., min_length=1)
    entities: list[SceneEntity] = Field(default_factory=list)

    @field_validator("scene_id")
    @classmethod
    def _check_scene_id(cls, value: str) -> str:
        return _validate_identifier(value, "SceneSpec.scene_id")

    @field_validator("width", "height", "grid_resolution")
    @classmethod
    def _require_finite_dims(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("SceneSpec dimensions must be finite.")
        return value

    def _point_in_bounds(self, point: Point2D, label: str) -> None:
        if point.x > self.width or point.y > self.height:
            raise ValueError(
                f"{label} ({point.x}, {point.y}) exceeds scene bounds ({self.width}, {self.height})."
            )

    def _rect_in_bounds(self, rect: Rect2D, label: str) -> None:
        if rect.x2 > self.width or rect.y2 > self.height:
            raise ValueError(
                f"{label} bounds [({rect.x}, {rect.y}) -> ({rect.x2}, {rect.y2})] "
                f"exceed scene bounds ({self.width}, {self.height})."
            )

    @model_validator(mode="after")
    def _validate_scene_integrity(self) -> SceneSpec:
        if self.grid_resolution > min(self.width, self.height):
            raise ValueError("grid_resolution must not exceed min(width, height).")

        if self.scale_bar.world_units > self.width:
            raise ValueError("scale_bar.world_units must not exceed scene width.")

        seen_ids: set[str] = set()
        for group_name, items in (
            ("wall", self.walls),
            ("door", self.doors),
            ("zone", self.zones),
            ("exit", self.exits),
            ("entity", self.entities),
        ):
            for item in items:
                if item.id in seen_ids:
                    raise ValueError(
                        f"Duplicate scene element id {item.id!r} detected in {group_name}s."
                    )
                seen_ids.add(item.id)

        for wall in self.walls:
            self._point_in_bounds(wall.start, f"Wall {wall.id} start")
            self._point_in_bounds(wall.end, f"Wall {wall.id} end")

        for door in self.doors:
            self._point_in_bounds(door.position, f"Door {door.id}")

        for exit_marker in self.exits:
            self._point_in_bounds(exit_marker.position, f"Exit {exit_marker.id}")

        for zone in self.zones:
            self._rect_in_bounds(zone.bounds, f"Zone {zone.id}")

        solid_entities: list[SceneEntity] = []
        for entity in self.entities:
            self._rect_in_bounds(entity.bounds, f"Entity {entity.id}")
            if not entity.passable:
                for other in solid_entities:
                    if entity.bounds.overlaps_interior(other.bounds):
                        raise ValueError(
                            f"Solid entities {entity.id!r} and {other.id!r} overlap."
                        )
                solid_entities.append(entity)

        return self
