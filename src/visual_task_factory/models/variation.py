"""Variation configuration models controlling structural and visual diversity."""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DifficultyTier(str, Enum):
    """Difficulty tiers governing spatial complexity and obstacle density."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class VisualTheme(str, Enum):
    """Visual rendering styles for procedural facility floorplans."""

    BLUEPRINT = "blueprint"
    SAFETY_STANDARD = "safety_standard"
    HIGH_CONTRAST = "high_contrast"
    CLUTTERED_OPS = "cluttered_ops"


class ClutterLevel(str, Enum):
    """Degree of non-blocking visual clutter / distractor details."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VariationConfig(BaseModel):
    """Explicit parameterization of structural, geometric, and visual variations."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(..., ge=0, description="Master deterministic RNG seed.")
    topology_id: str = Field(
        ...,
        min_length=1,
        description="Identifier for the structural room/corridor topology family.",
    )
    difficulty: DifficultyTier = DifficultyTier.MEDIUM
    orientation_deg: Literal[0, 90, 180, 270] = 0
    scale_meters_per_unit: float = Field(default=1.0, gt=0.0, le=50.0)
    theme: VisualTheme = VisualTheme.SAFETY_STANDARD
    clutter_level: ClutterLevel = ClutterLevel.LOW
    min_hazard_zones: int = Field(default=1, ge=0, le=20)
    min_obstacles: int = Field(default=2, ge=0, le=100)
    max_obstacles: int = Field(default=10, ge=0, le=100)
    include_distractor_labels: bool = False

    @field_validator("topology_id")
    @classmethod
    def _strip_non_empty_topology_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("topology_id must not be empty or whitespace.")
        return cleaned

    @field_validator("scale_meters_per_unit")
    @classmethod
    def _require_finite_scale(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("scale_meters_per_unit must be finite.")
        return value

    @model_validator(mode="after")
    def _validate_obstacle_bounds(self) -> VariationConfig:
        if self.max_obstacles < self.min_obstacles:
            raise ValueError(
                f"max_obstacles ({self.max_obstacles}) must be >= min_obstacles ({self.min_obstacles})."
            )
        return self
