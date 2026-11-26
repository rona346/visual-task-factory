"""Provider-agnostic model evaluation framework with deterministic mock adapter."""

from __future__ import annotations

import hashlib
import json
import math
import re
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from visual_task_factory.dataset_pipeline import DatasetManifest, DatasetRecord
from visual_task_factory.models import AnswerType, QuestionFamily, ReferenceAnswer

_INT_STRICT_PATTERN = re.compile(r"^\s*\+?(\d+)\s*\.?\s*$")
_INT_LABELED_PATTERN = re.compile(
    r"^\s*(?:answer|count|total)\s*[:=]\s*\+?(\d+)\s*\.?\s*$",
    re.IGNORECASE,
)
_FLOAT_WITH_UNIT_PATTERN = re.compile(
    r"^\s*(?:(?:answer|clearance|distance)\s*[:=]\s*)?([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*\.?\s*$",
    re.IGNORECASE,
)
_SIMPLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class PredictionStatus(str, Enum):
    """Execution and parsing status for a single evaluated dataset record."""

    OK = "ok"
    UNPARSEABLE = "unparseable"
    ADAPTER_ERROR = "adapter_error"
    MISSING_IMAGE = "missing_image"


class ModelRequest(BaseModel):
    """Sanitized input payload sent to a MultimodalModelAdapter (never contains reference_answer)."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    scene_id: str
    question_id: str
    image_path: Path
    prompt_text: str
    expected_answer_type: AnswerType
    candidate_choices: list[str] | None = None
    unit: str | None = None
    include_image: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelPrediction(BaseModel):
    """Raw prediction output returned by a MultimodalModelAdapter."""

    model_config = ConfigDict(extra="forbid")

    raw_output: str
    model_name: str
    provider_name: str
    is_mock: bool = False
    latency_ms: float | None = Field(default=None, ge=0.0)
    token_usage: dict[str, int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MultimodalModelAdapter(ABC):
    """Abstract provider-agnostic interface for multimodal model evaluation."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the evaluated model."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider or backend name."""

    @property
    @abstractmethod
    def is_mock(self) -> bool:
        """True if predictions are synthetic/mock rather than from a real model."""

    @abstractmethod
    def predict(self, request: ModelRequest) -> ModelPrediction:
        """Generate a prediction for the given visual reasoning request."""


class MockAdapterMode(str, Enum):
    """Behavior modes for the deterministic MockModelAdapter."""

    BLIND_HASH = "blind_hash"
    SCRIPTED = "scripted"


class MockModelAdapter(MultimodalModelAdapter):
    """Deterministic synthetic adapter for pipeline testing (never inspects reference answers)."""

    def __init__(
        self,
        *,
        seed: int = 0,
        mode: MockAdapterMode = MockAdapterMode.BLIND_HASH,
        model_id: str = "mock_vlm_v1",
        scripted_outputs: dict[str, str] | None = None,
        default_scripted_output: str = "UNPARSEABLE_OUTPUT",
        raise_on_question_ids: set[str] | None = None,
    ) -> None:
        if seed < 0:
            raise ValueError("seed must be >= 0.")
        self._seed = seed
        self._mode = mode
        self._model_id = model_id
        self._scripted_outputs = dict(scripted_outputs or {})
        self._default_scripted_output = default_scripted_output
        self._raise_on_question_ids = set(raise_on_question_ids or set())

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def provider_name(self) -> str:
        return "mock_synthetic"

    @property
    def is_mock(self) -> bool:
        return True

    def predict(self, request: ModelRequest) -> ModelPrediction:
        if request.question_id in self._raise_on_question_ids:
            raise RuntimeError(
                f"Simulated adapter failure for question_id={request.question_id!r}."
            )

        if self._mode == MockAdapterMode.SCRIPTED:
            raw_output = self._scripted_outputs.get(
                request.question_id,
                self._scripted_outputs.get(request.record_id, self._default_scripted_output),
            )
        else:
            raw_output = self._deterministic_blind_guess(request)

        return ModelPrediction(
            raw_output=raw_output,
            model_name=self.model_name,
            provider_name=self.provider_name,
            is_mock=True,
            latency_ms=1.0,
            metadata={
                "evaluation_nature": "mock_synthetic_pipeline_verification_only",
                "mode": self._mode.value,
                "seed": self._seed,
            },
        )

    def _deterministic_blind_guess(self, request: ModelRequest) -> str:
        digest = hashlib.sha256(
            f"{self._seed}:{request.question_id}:{request.expected_answer_type.value}:{request.prompt_text}".encode(
                "utf-8"
            )
        ).digest()
        val_int = int.from_bytes(digest[:8], "big")

        if request.expected_answer_type == AnswerType.EXIT_ID:
            if request.candidate_choices:
                return request.candidate_choices[val_int % len(request.candidate_choices)]
            return f"E{(val_int % 3) + 1}"

        if request.expected_answer_type == AnswerType.INTEGER:
            return str(val_int % 5)

        if request.expected_answer_type == AnswerType.BOOLEAN:
            return "true" if (val_int % 2 == 0) else "false"

        if request.expected_answer_type == AnswerType.NUMERIC:
            sample_meters = round(0.5 + (val_int % 10) * 0.5, 2)
            return f"{sample_meters:g} m"

        return "UNSUPPORTED"


class NormalizedPrediction(BaseModel):
    """Result of conservative prediction parsing."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    parsed_value: bool | int | float | str | None = None
    parsed_unit: str | None = None
    error_reason: str | None = None


def _unwrap_json_scalar(raw_output: str) -> str:
    text = raw_output.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                for key in ("answer", "value", "prediction"):
                    if key in loaded and len(loaded) <= 2:
                        inner = loaded[key]
                        if isinstance(inner, bool):
                            return "true" if inner else "false"
                        if isinstance(inner, (int, float, str)):
                            return str(inner)
        except json.JSONDecodeError:
            pass
    return text


def normalize_prediction(
    raw_output: str,
    expected_type: AnswerType,
    *,
    candidate_choices: list[str] | None = None,
    expected_unit: str | None = None,
) -> NormalizedPrediction:
    """Conservatively parse a raw model string into expected_type without guessing."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        return NormalizedPrediction(is_valid=False, error_reason="Empty model output.")

    text = _unwrap_json_scalar(raw_output).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()

    if expected_type == AnswerType.BOOLEAN:
        cleaned = text.rstrip(".").strip().lower()
        if cleaned in ("true", "yes"):
            return NormalizedPrediction(is_valid=True, parsed_value=True)
        if cleaned in ("false", "no"):
            return NormalizedPrediction(is_valid=True, parsed_value=False)
        return NormalizedPrediction(
            is_valid=False,
            error_reason=f"Could not unambiguously parse boolean from {raw_output!r}.",
        )

    if expected_type == AnswerType.INTEGER:
        m = _INT_STRICT_PATTERN.match(text) or _INT_LABELED_PATTERN.match(text)
        if m is None:
            return NormalizedPrediction(
                is_valid=False,
                error_reason=f"Could not unambiguously parse non-negative integer from {raw_output!r}.",
            )
        return NormalizedPrediction(is_valid=True, parsed_value=int(m.group(1)))

    if expected_type == AnswerType.NUMERIC:
        m = _FLOAT_WITH_UNIT_PATTERN.match(text)
        if m is None:
            return NormalizedPrediction(
                is_valid=False,
                error_reason=f"Could not unambiguously parse numeric clearance from {raw_output!r}.",
            )
        val = float(m.group(1))
        if not math.isfinite(val) or val < 0.0:
            return NormalizedPrediction(
                is_valid=False,
                error_reason=f"Numeric prediction must be finite and >= 0; got {val}.",
            )
        unit_tok = m.group(2).lower()
        parsed_unit: str | None = None
        if unit_tok:
            if unit_tok in ("m", "meter", "meters"):
                parsed_unit = "m"
            else:
                return NormalizedPrediction(
                    is_valid=False,
                    error_reason=f"Unsupported or mismatched unit {unit_tok!r} (expected {expected_unit!r}).",
                )
        if expected_unit is not None and parsed_unit is not None and parsed_unit != expected_unit:
            return NormalizedPrediction(
                is_valid=False,
                error_reason=f"Parsed unit {parsed_unit!r} != expected {expected_unit!r}.",
            )
        return NormalizedPrediction(is_valid=True, parsed_value=val, parsed_unit=parsed_unit)

    if expected_type == AnswerType.EXIT_ID:
        cleaned = text.rstrip(".").strip()
        if candidate_choices:
            choice_map = {c.casefold(): c for c in candidate_choices}
            if cleaned.casefold() in choice_map:
                return NormalizedPrediction(
                    is_valid=True,
                    parsed_value=choice_map[cleaned.casefold()],
                )
            # Match whole tokens unambiguously against candidate_choices.
            matched = [
                c
                for c in candidate_choices
                if re.search(rf"\b{re.escape(c)}\b", cleaned, re.IGNORECASE)
            ]
            if len(matched) == 1:
                return NormalizedPrediction(is_valid=True, parsed_value=matched[0])
            return NormalizedPrediction(
                is_valid=False,
                error_reason=f"Output {raw_output!r} did not map uniquely to candidate_choices={candidate_choices}.",
            )

        if _SIMPLE_ID_PATTERN.match(cleaned):
            return NormalizedPrediction(is_valid=True, parsed_value=cleaned)
        return NormalizedPrediction(
            is_valid=False,
            error_reason=f"Could not parse exit identifier from {raw_output!r}.",
        )

    return NormalizedPrediction(
        is_valid=False,
        error_reason=f"Unsupported expected_type={expected_type!r}.",
    )


def compare_prediction_to_reference(
    parsed_value: bool | int | float | str,
    reference: ReferenceAnswer,
) -> bool:
    """Compare a normalized prediction against ReferenceAnswer according to answer_type."""
    if reference.answer_type == AnswerType.BOOLEAN:
        return type(parsed_value) is bool and parsed_value == reference.value

    if reference.answer_type == AnswerType.INTEGER:
        return type(parsed_value) is int and parsed_value == reference.value

    if reference.answer_type == AnswerType.EXIT_ID:
        return (
            isinstance(parsed_value, str)
            and parsed_value.casefold() == str(reference.value).casefold()
        )

    if reference.answer_type == AnswerType.NUMERIC:
        if type(parsed_value) not in (int, float):
            return False
        tol = reference.numeric_tolerance if reference.numeric_tolerance is not None else 0.0
        return abs(float(parsed_value) - float(reference.value)) <= tol + 1e-9

    return False


class EvaluationConfig(BaseModel):
    """Configuration controlling a single model evaluation run."""

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str = Field(default="eval_run_v1", min_length=1)
    max_records: int | None = Field(default=None, ge=1)
    families: tuple[QuestionFamily, ...] | None = None
    require_images_exist: bool = True
    fail_fast: bool = False

    @field_validator("evaluation_id")
    @classmethod
    def _clean_eval_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("evaluation_id must not be empty.")
        return cleaned


class EvaluationRecord(BaseModel):
    """Per-item evaluation output comparing a model prediction with ReferenceAnswer."""

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str
    dataset_id: str
    record_id: str
    scene_id: str
    question_id: str
    family: QuestionFamily
    expected_answer_type: AnswerType
    image_rel_path: str
    status: PredictionStatus
    model_name: str
    provider_name: str
    is_mock_evaluation: bool
    raw_prediction: str | None = None
    normalized_prediction: bool | int | float | str | None = None
    reference_answer: ReferenceAnswer
    is_correct: bool
    error_message: str | None = None
    latency_ms: float | None = None


class EvaluationSummary(BaseModel):
    """Machine-readable run summary for an evaluation pass."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    evaluation_id: str
    dataset_id: str
    dataset_logical_sha256: str | None = None
    model_name: str
    provider_name: str
    is_mock_evaluation: bool
    evaluation_notice: str
    total_records: int
    completed_predictions: int
    correct_count: int
    incorrect_count: int
    unparseable_count: int
    adapter_error_count: int
    missing_image_count: int
    raw_accuracy: float
    artifacts: dict[str, str]


class EvaluationResult(BaseModel):
    """Full in-memory and on-disk result returned by ModelEvaluator.evaluate()."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    summary: EvaluationSummary
    records: list[EvaluationRecord]


class ModelEvaluator:
    """Loads a validated Phase 8 dataset, runs a model adapter, and saves evaluation outputs."""

    def __init__(self, adapter: MultimodalModelAdapter) -> None:
        if not isinstance(adapter, MultimodalModelAdapter):
            raise TypeError(
                f"Expected MultimodalModelAdapter instance, got {type(adapter).__name__}."
            )
        self.adapter = adapter

    def load_dataset(
        self,
        dataset_dir: Path | str,
        config: EvaluationConfig | None = None,
    ) -> tuple[list[DatasetRecord], DatasetManifest | None]:
        """Load valid DatasetRecords and optional manifest from a Phase 8 dataset directory."""
        ds_dir = Path(dataset_dir)
        if not ds_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: '{ds_dir}'.")

        dataset_jsonl = ds_dir / "dataset.jsonl"
        if not dataset_jsonl.is_file():
            raise FileNotFoundError(f"Missing dataset.jsonl in '{ds_dir}'.")

        manifest: DatasetManifest | None = None
        manifest_path = ds_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = DatasetManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )

        records: list[DatasetRecord] = []
        for raw_line in dataset_jsonl.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            rec = DatasetRecord.model_validate_json(raw_line)
            if config and config.families is not None and rec.family not in config.families:
                continue
            records.append(rec)
            if config and config.max_records is not None and len(records) >= config.max_records:
                break

        return records, manifest

    def evaluate(
        self,
        dataset_dir: Path | str,
        output_dir: Path | str,
        config: EvaluationConfig | None = None,
    ) -> EvaluationResult:
        """Evaluate all selected dataset records and write predictions.jsonl + summary JSON."""
        cfg = config or EvaluationConfig()
        ds_dir = Path(dataset_dir).resolve()
        out_dir = Path(output_dir)
        if out_dir.exists() and not out_dir.is_dir():
            raise FileExistsError(f"output_dir points to an existing file: '{out_dir}'.")
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset_records, manifest = self.load_dataset(ds_dir, cfg)
        eval_records: list[EvaluationRecord] = []

        for rec in dataset_records:
            resolved_img = (ds_dir / rec.image_rel_path).resolve()
            # Prevent path traversal outside dataset_dir.
            if ds_dir not in resolved_img.parents and resolved_img != ds_dir:
                eval_records.append(
                    EvaluationRecord(
                        evaluation_id=cfg.evaluation_id,
                        dataset_id=rec.dataset_id,
                        record_id=rec.record_id,
                        scene_id=rec.scene_id,
                        question_id=rec.question_id,
                        family=rec.family,
                        expected_answer_type=rec.expected_answer_type,
                        image_rel_path=rec.image_rel_path,
                        status=PredictionStatus.MISSING_IMAGE,
                        model_name=self.adapter.model_name,
                        provider_name=self.adapter.provider_name,
                        is_mock_evaluation=self.adapter.is_mock,
                        reference_answer=rec.reference_answer,
                        is_correct=False,
                        error_message=f"Image path escapes dataset directory: {rec.image_rel_path!r}.",
                    )
                )
                continue

            if cfg.require_images_exist and (
                not resolved_img.is_file() or resolved_img.stat().st_size == 0
            ):
                eval_records.append(
                    EvaluationRecord(
                        evaluation_id=cfg.evaluation_id,
                        dataset_id=rec.dataset_id,
                        record_id=rec.record_id,
                        scene_id=rec.scene_id,
                        question_id=rec.question_id,
                        family=rec.family,
                        expected_answer_type=rec.expected_answer_type,
                        image_rel_path=rec.image_rel_path,
                        status=PredictionStatus.MISSING_IMAGE,
                        model_name=self.adapter.model_name,
                        provider_name=self.adapter.provider_name,
                        is_mock_evaluation=self.adapter.is_mock,
                        reference_answer=rec.reference_answer,
                        is_correct=False,
                        error_message=f"Image file missing or empty: '{resolved_img}'.",
                    )
                )
                continue

            request = ModelRequest(
                record_id=rec.record_id,
                scene_id=rec.scene_id,
                question_id=rec.question_id,
                image_path=resolved_img,
                prompt_text=rec.prompt_text,
                expected_answer_type=rec.expected_answer_type,
                candidate_choices=rec.candidate_choices,
                unit=rec.reference_answer.unit,
                metadata={
                    "family": rec.family.value,
                    "template_id": rec.template_id,
                },
            )

            try:
                prediction = self.adapter.predict(request)
            except Exception as exc:
                if cfg.fail_fast:
                    raise
                eval_records.append(
                    EvaluationRecord(
                        evaluation_id=cfg.evaluation_id,
                        dataset_id=rec.dataset_id,
                        record_id=rec.record_id,
                        scene_id=rec.scene_id,
                        question_id=rec.question_id,
                        family=rec.family,
                        expected_answer_type=rec.expected_answer_type,
                        image_rel_path=rec.image_rel_path,
                        status=PredictionStatus.ADAPTER_ERROR,
                        model_name=self.adapter.model_name,
                        provider_name=self.adapter.provider_name,
                        is_mock_evaluation=self.adapter.is_mock,
                        reference_answer=rec.reference_answer,
                        is_correct=False,
                        error_message=str(exc),
                    )
                )
                continue

            norm = normalize_prediction(
                prediction.raw_output,
                rec.expected_answer_type,
                candidate_choices=rec.candidate_choices,
                expected_unit=rec.reference_answer.unit,
            )

            if not norm.is_valid or norm.parsed_value is None:
                eval_records.append(
                    EvaluationRecord(
                        evaluation_id=cfg.evaluation_id,
                        dataset_id=rec.dataset_id,
                        record_id=rec.record_id,
                        scene_id=rec.scene_id,
                        question_id=rec.question_id,
                        family=rec.family,
                        expected_answer_type=rec.expected_answer_type,
                        image_rel_path=rec.image_rel_path,
                        status=PredictionStatus.UNPARSEABLE,
                        model_name=prediction.model_name,
                        provider_name=prediction.provider_name,
                        is_mock_evaluation=prediction.is_mock or self.adapter.is_mock,
                        raw_prediction=prediction.raw_output,
                        normalized_prediction=None,
                        reference_answer=rec.reference_answer,
                        is_correct=False,
                        error_message=norm.error_reason,
                        latency_ms=prediction.latency_ms,
                    )
                )
                continue

            is_correct = compare_prediction_to_reference(
                norm.parsed_value,
                rec.reference_answer,
            )
            eval_records.append(
                EvaluationRecord(
                    evaluation_id=cfg.evaluation_id,
                    dataset_id=rec.dataset_id,
                    record_id=rec.record_id,
                    scene_id=rec.scene_id,
                    question_id=rec.question_id,
                    family=rec.family,
                    expected_answer_type=rec.expected_answer_type,
                    image_rel_path=rec.image_rel_path,
                    status=PredictionStatus.OK,
                    model_name=prediction.model_name,
                    provider_name=prediction.provider_name,
                    is_mock_evaluation=prediction.is_mock or self.adapter.is_mock,
                    raw_prediction=prediction.raw_output,
                    normalized_prediction=norm.parsed_value,
                    reference_answer=rec.reference_answer,
                    is_correct=is_correct,
                    latency_ms=prediction.latency_ms,
                )
            )

        predictions_path = out_dir / "predictions.jsonl"
        summary_path = out_dir / "evaluation_summary.json"

        pred_lines = [r.model_dump_json() for r in eval_records]
        predictions_path.write_text(
            ("\n".join(pred_lines) + "\n") if pred_lines else "",
            encoding="utf-8",
        )

        total = len(eval_records)
        completed = sum(1 for r in eval_records if r.status == PredictionStatus.OK)
        correct = sum(1 for r in eval_records if r.status == PredictionStatus.OK and r.is_correct)
        incorrect = sum(
            1 for r in eval_records if r.status == PredictionStatus.OK and not r.is_correct
        )
        unparseable = sum(1 for r in eval_records if r.status == PredictionStatus.UNPARSEABLE)
        adapter_errors = sum(1 for r in eval_records if r.status == PredictionStatus.ADAPTER_ERROR)
        missing_images = sum(1 for r in eval_records if r.status == PredictionStatus.MISSING_IMAGE)

        dataset_id = (
            manifest.dataset_id
            if manifest is not None
            else (dataset_records[0].dataset_id if dataset_records else "unknown")
        )
        notice = (
            "SYNTHETIC MOCK EVALUATION - NOT REAL MODEL CAPABILITY"
            if self.adapter.is_mock
            else "REAL MODEL EVALUATION"
        )

        summary = EvaluationSummary(
            evaluation_id=cfg.evaluation_id,
            dataset_id=dataset_id,
            dataset_logical_sha256=manifest.logical_dataset_sha256 if manifest else None,
            model_name=self.adapter.model_name,
            provider_name=self.adapter.provider_name,
            is_mock_evaluation=self.adapter.is_mock,
            evaluation_notice=notice,
            total_records=total,
            completed_predictions=completed,
            correct_count=correct,
            incorrect_count=incorrect,
            unparseable_count=unparseable,
            adapter_error_count=adapter_errors,
            missing_image_count=missing_images,
            raw_accuracy=round(correct / total, 4) if total > 0 else 0.0,
            artifacts={
                "predictions_jsonl": "predictions.jsonl",
                "summary_json": "evaluation_summary.json",
            },
        )
        summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")

        return EvaluationResult(
            output_dir=out_dir,
            summary=summary,
            records=eval_records,
        )


def evaluate_dataset(
    dataset_dir: Path | str,
    output_dir: Path | str,
    adapter: MultimodalModelAdapter,
    *,
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    """Convenience entry point to evaluate a Phase 8 dataset with a model adapter."""
    return ModelEvaluator(adapter=adapter).evaluate(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        config=config,
    )
