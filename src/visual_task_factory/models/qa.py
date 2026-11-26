"""Normalized question, reference-answer, and benchmark-item data models."""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from visual_task_factory.models.scene import SceneSpec

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class QuestionFamily(str, Enum):
    """Supported visual-reasoning task families."""

    CONSTRAINED_REACHABILITY = "constrained_reachability"
    REGION_CONTAINMENT_COUNTING = "region_containment_counting"
    SCALE_GROUNDED_CLEARANCE = "scale_grounded_clearance"


class AnswerType(str, Enum):
    """Machine-readable normalized answer types."""

    EXIT_ID = "exit_id"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    NUMERIC = "numeric"


ALLOWED_ANSWER_TYPES_BY_FAMILY: dict[QuestionFamily, tuple[AnswerType, ...]] = {
    QuestionFamily.CONSTRAINED_REACHABILITY: (AnswerType.EXIT_ID,),
    QuestionFamily.REGION_CONTAINMENT_COUNTING: (AnswerType.INTEGER,),
    QuestionFamily.SCALE_GROUNDED_CLEARANCE: (AnswerType.BOOLEAN, AnswerType.NUMERIC),
}


class ReferenceAnswer(BaseModel):
    """Independently computed reference answer with explicit type and provenance."""

    model_config = ConfigDict(extra="forbid")

    answer_type: AnswerType
    value: bool | int | float | str
    unit: Literal["m"] | None = None
    numeric_tolerance: float | None = Field(default=None, ge=0.0)
    solver_name: str = Field(..., min_length=1)
    reasoning_trace: list[str] = Field(default_factory=list)

    @field_validator("solver_name")
    @classmethod
    def _clean_solver_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("solver_name must not be empty or whitespace.")
        return cleaned

    @field_validator("numeric_tolerance")
    @classmethod
    def _require_finite_tolerance(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("numeric_tolerance must be finite.")
        return value

    @model_validator(mode="after")
    def _validate_value_matches_answer_type(self) -> ReferenceAnswer:
        val = self.value

        if self.answer_type == AnswerType.EXIT_ID:
            if type(val) is not str or not _ID_PATTERN.match(val.strip()):
                raise ValueError(
                    "EXIT_ID answer must be a non-empty alphanumeric/underscore/hyphen string."
                )
            self.value = val.strip()
            if self.unit is not None or self.numeric_tolerance is not None:
                raise ValueError("EXIT_ID answers must not specify unit or numeric_tolerance.")

        elif self.answer_type == AnswerType.INTEGER:
            if type(val) is not int or val < 0:
                raise ValueError("INTEGER answer must be a non-negative integer (not bool or float).")
            if self.unit is not None or self.numeric_tolerance is not None:
                raise ValueError("INTEGER answers must not specify unit or numeric_tolerance.")

        elif self.answer_type == AnswerType.BOOLEAN:
            if type(val) is not bool:
                raise ValueError("BOOLEAN answer must be a strict bool (True or False).")
            if self.unit is not None or self.numeric_tolerance is not None:
                raise ValueError("BOOLEAN answers must not specify unit or numeric_tolerance.")

        elif self.answer_type == AnswerType.NUMERIC:
            if type(val) not in (int, float):
                raise ValueError("NUMERIC answer must be a finite non-negative number (not bool).")
            numeric_val = float(val)
            if not math.isfinite(numeric_val) or numeric_val < 0.0:
                raise ValueError("NUMERIC answer must be finite and >= 0.0.")
            if self.unit is None:
                raise ValueError("NUMERIC clearance answers must explicitly specify unit='m'.")
            self.value = numeric_val

        return self


class VisualQuestion(BaseModel):
    """Normalized visual-reasoning question paired with its reference answer."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    question_id: str = Field(..., min_length=1)
    scene_id: str = Field(..., min_length=1)
    family: QuestionFamily
    template_id: str = Field(..., min_length=1)
    prompt_text: str = Field(..., min_length=10)
    expected_answer_type: AnswerType
    candidate_choices: list[str] | None = None
    reference_answer: ReferenceAnswer
    solver_query: dict[str, Any] | None = None
    requires_visual_reasoning: bool = True

    @field_validator("question_id", "scene_id", "template_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or not _ID_PATTERN.match(cleaned):
            raise ValueError(f"Identifier {value!r} must match {_ID_PATTERN.pattern}.")
        return cleaned

    @field_validator("prompt_text")
    @classmethod
    def _clean_prompt_text(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 10:
            raise ValueError("prompt_text must be at least 10 non-whitespace characters.")
        return cleaned

    @model_validator(mode="after")
    def _validate_question_consistency(self) -> VisualQuestion:
        if not self.requires_visual_reasoning:
            raise ValueError("Benchmark questions must require visual reasoning.")

        allowed_types = ALLOWED_ANSWER_TYPES_BY_FAMILY[self.family]
        if self.expected_answer_type not in allowed_types:
            allowed_names = [item.value for item in allowed_types]
            raise ValueError(
                f"Question family {self.family.value!r} requires expected_answer_type in "
                f"{allowed_names}, got {self.expected_answer_type.value!r}."
            )

        if self.reference_answer.answer_type != self.expected_answer_type:
            raise ValueError(
                f"reference_answer.answer_type ({self.reference_answer.answer_type.value!r}) "
                f"must match expected_answer_type ({self.expected_answer_type.value!r})."
            )

        if self.candidate_choices is not None:
            cleaned_choices = [choice.strip() for choice in self.candidate_choices if choice.strip()]
            if len(cleaned_choices) < 2 or len(set(cleaned_choices)) != len(cleaned_choices):
                raise ValueError(
                    "candidate_choices must contain at least 2 unique non-empty choices."
                )
            self.candidate_choices = cleaned_choices
            if self.expected_answer_type == AnswerType.EXIT_ID:
                if str(self.reference_answer.value) not in self.candidate_choices:
                    raise ValueError(
                        "reference_answer.value must be present in candidate_choices."
                    )

        if self.solver_query is not None:
            if not self.solver_query:
                raise ValueError("solver_query, when provided, must be a non-empty dictionary.")
            query_family = self.solver_query.get("family")
            if query_family is not None and query_family != self.family.value:
                raise ValueError(
                    f"solver_query.family ({query_family!r}) must match question family "
                    f"({self.family.value!r})."
                )

        return self


class BenchmarkItem(BaseModel):
    """Joint record binding a SceneSpec and a VisualQuestion with cross-entity validation."""

    model_config = ConfigDict(extra="forbid")

    scene: SceneSpec
    question: VisualQuestion

    @model_validator(mode="after")
    def _validate_cross_references(self) -> BenchmarkItem:
        if self.question.scene_id != self.scene.scene_id:
            raise ValueError(
                f"question.scene_id ({self.question.scene_id!r}) does not match "
                f"scene.scene_id ({self.scene.scene_id!r})."
            )

        if self.question.expected_answer_type == AnswerType.EXIT_ID:
            valid_exit_ids = {exit_marker.id for exit_marker in self.scene.exits}
            answer_exit = str(self.question.reference_answer.value)
            if answer_exit not in valid_exit_ids:
                raise ValueError(
                    f"Reference exit ID {answer_exit!r} does not exist in scene.exits ({sorted(valid_exit_ids)})."
                )
            if self.question.candidate_choices is not None:
                unknown = set(self.question.candidate_choices) - valid_exit_ids
                if unknown:
                    raise ValueError(
                        f"candidate_choices contain unknown exit IDs not in scene.exits: {sorted(unknown)}."
                    )

        return self
