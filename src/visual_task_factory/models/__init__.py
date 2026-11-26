"""Core Pydantic data models for scenes, variations, questions, and answers."""

from visual_task_factory.models.qa import (
    AnswerType,
    BenchmarkItem,
    QuestionFamily,
    ReferenceAnswer,
    VisualQuestion,
)
from visual_task_factory.models.scene import (
    Doorway,
    EntityType,
    ExitMarker,
    Point2D,
    Rect2D,
    ScaleBarSpec,
    SceneEntity,
    SceneSpec,
    WallSegment,
    Zone,
    ZoneType,
)
from visual_task_factory.models.variation import (
    ClutterLevel,
    DifficultyTier,
    VariationConfig,
    VisualTheme,
)

__all__ = [
    "AnswerType",
    "BenchmarkItem",
    "ClutterLevel",
    "DifficultyTier",
    "Doorway",
    "EntityType",
    "ExitMarker",
    "Point2D",
    "QuestionFamily",
    "Rect2D",
    "ReferenceAnswer",
    "ScaleBarSpec",
    "SceneEntity",
    "SceneSpec",
    "VariationConfig",
    "VisualQuestion",
    "VisualTheme",
    "WallSegment",
    "Zone",
    "ZoneType",
]
