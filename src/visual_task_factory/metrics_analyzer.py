"""Metrics and failure-analysis module for Phase 9 model evaluation outputs."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from visual_task_factory.model_evaluator import (
    EvaluationRecord,
    EvaluationSummary,
    PredictionStatus,
)
from visual_task_factory.models import AnswerType, QuestionFamily, ReferenceAnswer

MOCK_EVALUATION_NOTICE = "SYNTHETIC MOCK EVALUATION — NOT REAL MODEL CAPABILITY"
REAL_EVALUATION_NOTICE = "REAL MODEL EVALUATION"
UNKNOWN_EVALUATION_NOTICE = (
    "UNKNOWN EVALUATION TYPE — SOURCE METADATA MISSING OR AMBIGUOUS"
)


class EvaluationNature(str, Enum):
    """Provenance classification of an evaluation run."""

    MOCK = "mock"
    REAL = "real"
    UNKNOWN = "unknown"


class FailureCategory(str, Enum):
    """Failure categories grounded strictly in PredictionStatus and is_correct."""

    INCORRECT_PREDICTION = "incorrect_prediction"
    UNPARSEABLE_PREDICTION = "unparseable_prediction"
    ADAPTER_ERROR = "adapter_error"
    MISSING_IMAGE = "missing_image"


class MixedEvaluationError(ValueError):
    """Raised when mock and real evaluation records are combined in one analysis."""


class GroupedMetrics(BaseModel):
    """Accuracy and status breakdown for a single slice (family or answer type)."""

    model_config = ConfigDict(extra="forbid")

    total: int
    completed: int
    correct: int
    incorrect: int
    unparseable: int
    adapter_errors: int
    missing_images: int
    accuracy: float
    completed_accuracy: float


class MetricsSummary(BaseModel):
    """Aggregate benchmark performance metrics written to metrics_summary.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    evaluation_id: str | None = None
    dataset_id: str | None = None
    model_name: str | None = None
    provider_name: str | None = None
    evaluation_nature: EvaluationNature
    is_mock: bool | None = None
    evaluation_notice: str
    total_items: int
    completed_predictions: int
    correct_predictions: int
    incorrect_predictions: int
    accuracy: float
    completed_accuracy: float
    unparseable_count: int
    unparseable_rate: float
    adapter_error_count: int
    adapter_error_rate: float
    missing_image_count: int
    missing_image_rate: float
    failure_count: int
    failure_rate: float
    by_family: dict[str, GroupedMetrics]
    by_answer_type: dict[str, GroupedMetrics]
    failure_category_counts: dict[str, int]
    failure_category_rates: dict[str, float]


class FailureExample(BaseModel):
    """Single failed or incorrect record exported to failure_examples.jsonl."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    scene_id: str
    question_id: str
    family: QuestionFamily
    expected_answer_type: AnswerType
    image_rel_path: str
    status: PredictionStatus
    failure_category: FailureCategory
    reference_answer: ReferenceAnswer
    raw_prediction: str | None = None
    normalized_prediction: bool | int | float | str | None = None
    error_message: str | None = None
    model_name: str
    provider_name: str
    is_mock_evaluation: bool | None = None


class FailureAnalysisReport(BaseModel):
    """Structured failure breakdown written to failure_analysis.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    evaluation_id: str | None = None
    dataset_id: str | None = None
    model_name: str | None = None
    evaluation_nature: EvaluationNature
    evaluation_notice: str
    total_items: int
    total_failures: int
    failure_rate: float
    category_counts: dict[str, int]
    category_rates: dict[str, float]
    failures_by_family: dict[str, dict[str, int]]
    failures_by_answer_type: dict[str, dict[str, int]]
    sample_failure_ids_by_category: dict[str, list[str]]


class MetricsAnalysisResult(BaseModel):
    """Combined metrics and failure-analysis outputs."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path | None = None
    metrics_summary: MetricsSummary
    failure_analysis: FailureAnalysisReport
    failure_examples: list[FailureExample]
    artifacts: dict[str, str]


class _NormalizedInputRecord(BaseModel):
    """Internal representation preserving unknown mock metadata if absent in raw dicts."""

    model_config = ConfigDict(extra="forbid")

    record: EvaluationRecord
    raw_is_mock: bool | None


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _classify_failure(record: EvaluationRecord) -> FailureCategory | None:
    if record.status == PredictionStatus.UNPARSEABLE:
        return FailureCategory.UNPARSEABLE_PREDICTION
    if record.status == PredictionStatus.ADAPTER_ERROR:
        return FailureCategory.ADAPTER_ERROR
    if record.status == PredictionStatus.MISSING_IMAGE:
        return FailureCategory.MISSING_IMAGE
    if record.status == PredictionStatus.OK:
        return None if record.is_correct else FailureCategory.INCORRECT_PREDICTION
    return FailureCategory.INCORRECT_PREDICTION


def _compute_grouped_metrics(records: list[EvaluationRecord]) -> GroupedMetrics:
    total = len(records)
    completed = sum(1 for r in records if r.status == PredictionStatus.OK)
    correct = sum(1 for r in records if r.status == PredictionStatus.OK and r.is_correct)
    incorrect = sum(1 for r in records if r.status == PredictionStatus.OK and not r.is_correct)
    unparseable = sum(1 for r in records if r.status == PredictionStatus.UNPARSEABLE)
    adapter_errors = sum(1 for r in records if r.status == PredictionStatus.ADAPTER_ERROR)
    missing_images = sum(1 for r in records if r.status == PredictionStatus.MISSING_IMAGE)
    return GroupedMetrics(
        total=total,
        completed=completed,
        correct=correct,
        incorrect=incorrect,
        unparseable=unparseable,
        adapter_errors=adapter_errors,
        missing_images=missing_images,
        accuracy=_rate(correct, total),
        completed_accuracy=_rate(correct, completed),
    )


def _resolve_evaluation_nature(
    items: list[_NormalizedInputRecord],
    summary_raw: dict[str, Any] | None,
) -> tuple[EvaluationNature, bool | None, str]:
    mock_flags: set[bool] = set()
    has_missing_flag = False
    providers: set[str] = set()

    for item in items:
        providers.add(item.record.provider_name)
        if item.raw_is_mock is None:
            has_missing_flag = True
        else:
            mock_flags.add(item.raw_is_mock)

    if summary_raw is not None:
        if "is_mock_evaluation" in summary_raw and isinstance(
            summary_raw["is_mock_evaluation"], bool
        ):
            mock_flags.add(summary_raw["is_mock_evaluation"])
        else:
            has_missing_flag = True
        if isinstance(summary_raw.get("provider_name"), str):
            providers.add(summary_raw["provider_name"])

    if len(mock_flags) > 1:
        raise MixedEvaluationError(
            "Cannot combine mock and real evaluation records into a single metrics report."
        )

    if not mock_flags or has_missing_flag:
        return EvaluationNature.UNKNOWN, None, UNKNOWN_EVALUATION_NOTICE

    is_mock = next(iter(mock_flags))
    if not is_mock and "mock_synthetic" in providers:
        # Conflicting provider vs is_mock metadata -> treat as ambiguous/unknown rather than real.
        return EvaluationNature.UNKNOWN, None, UNKNOWN_EVALUATION_NOTICE

    if is_mock:
        return EvaluationNature.MOCK, True, MOCK_EVALUATION_NOTICE
    return EvaluationNature.REAL, False, REAL_EVALUATION_NOTICE


def _coerce_input_record(raw: EvaluationRecord | dict[str, Any]) -> _NormalizedInputRecord:
    if isinstance(raw, EvaluationRecord):
        return _NormalizedInputRecord(record=raw, raw_is_mock=raw.is_mock_evaluation)
    if isinstance(raw, dict):
        raw_copy = dict(raw)
        raw_flag = raw_copy.get("is_mock_evaluation")
        parsed_flag = raw_flag if isinstance(raw_flag, bool) else None
        if parsed_flag is None:
            # Parse strict EvaluationRecord with temporary False, but keep raw_is_mock=None.
            raw_copy["is_mock_evaluation"] = False
        record = EvaluationRecord.model_validate(raw_copy)
        return _NormalizedInputRecord(record=record, raw_is_mock=parsed_flag)
    raise TypeError(
        f"Expected EvaluationRecord or dict, got {type(raw).__name__}."
    )


class MetricsAnalyzer:
    """Computes benchmark accuracy, slices, and grounded failure reports from Phase 9 outputs."""

    def analyze_directory(
        self,
        eval_dir: Path | str,
        *,
        output_dir: Path | str | None = None,
    ) -> MetricsAnalysisResult:
        """Load predictions.jsonl (and optional evaluation_summary.json) and write reports."""
        source_dir = Path(eval_dir)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Evaluation directory does not exist: '{source_dir}'.")

        predictions_path = source_dir / "predictions.jsonl"
        if not predictions_path.is_file():
            raise FileNotFoundError(f"Missing predictions.jsonl in '{source_dir}'.")

        summary_raw: dict[str, Any] | None = None
        summary_path = source_dir / "evaluation_summary.json"
        if summary_path.is_file():
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                summary_raw = loaded

        raw_records: list[dict[str, Any]] = []
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                loaded_line = json.loads(line)
                if not isinstance(loaded_line, dict):
                    raise ValueError("Each line in predictions.jsonl must be a JSON object.")
                raw_records.append(loaded_line)

        target_dir = Path(output_dir) if output_dir is not None else source_dir
        return self.analyze_records(
            raw_records,
            summary=summary_raw,
            output_dir=target_dir,
        )

    def analyze_records(
        self,
        records: Sequence[EvaluationRecord | dict[str, Any]],
        *,
        summary: EvaluationSummary | dict[str, Any] | None = None,
        output_dir: Path | str | None = None,
    ) -> MetricsAnalysisResult:
        """Analyze evaluation records in memory and optionally write Phase 10 report files."""
        normalized_items = [_coerce_input_record(r) for r in records]
        eval_records = [item.record for item in normalized_items]

        summary_raw: dict[str, Any] | None = None
        if isinstance(summary, EvaluationSummary):
            summary_raw = summary.model_dump(mode="python")
        elif isinstance(summary, dict):
            summary_raw = dict(summary)
        elif summary is not None:
            raise TypeError(
                f"Expected EvaluationSummary, dict, or None; got {type(summary).__name__}."
            )

        nature, is_mock, notice = _resolve_evaluation_nature(normalized_items, summary_raw)

        total = len(eval_records)
        overall = _compute_grouped_metrics(eval_records)

        # Group by QuestionFamily and AnswerType in deterministic sorted key order.
        family_buckets: dict[str, list[EvaluationRecord]] = {}
        type_buckets: dict[str, list[EvaluationRecord]] = {}
        for rec in eval_records:
            family_buckets.setdefault(rec.family.value, []).append(rec)
            type_buckets.setdefault(rec.expected_answer_type.value, []).append(rec)

        by_family = {
            key: _compute_grouped_metrics(family_buckets[key])
            for key in sorted(family_buckets.keys())
        }
        by_answer_type = {
            key: _compute_grouped_metrics(type_buckets[key])
            for key in sorted(type_buckets.keys())
        }

        # Collect grounded failures.
        failure_examples: list[FailureExample] = []
        category_counts: dict[str, int] = {}
        failures_by_family: dict[str, dict[str, int]] = {}
        failures_by_answer_type: dict[str, dict[str, int]] = {}

        for item in normalized_items:
            rec = item.record
            cat = _classify_failure(rec)
            if cat is None:
                continue

            cat_key = cat.value
            category_counts[cat_key] = category_counts.get(cat_key, 0) + 1

            fam_map = failures_by_family.setdefault(rec.family.value, {})
            fam_map[cat_key] = fam_map.get(cat_key, 0) + 1

            type_map = failures_by_answer_type.setdefault(rec.expected_answer_type.value, {})
            type_map[cat_key] = type_map.get(cat_key, 0) + 1

            failure_examples.append(
                FailureExample(
                    record_id=rec.record_id,
                    scene_id=rec.scene_id,
                    question_id=rec.question_id,
                    family=rec.family,
                    expected_answer_type=rec.expected_answer_type,
                    image_rel_path=rec.image_rel_path,
                    status=rec.status,
                    failure_category=cat,
                    reference_answer=rec.reference_answer,
                    raw_prediction=rec.raw_prediction,
                    normalized_prediction=rec.normalized_prediction,
                    error_message=rec.error_message,
                    model_name=rec.model_name,
                    provider_name=rec.provider_name,
                    is_mock_evaluation=item.raw_is_mock,
                )
            )

        # Deterministic ordering of failure examples and nested dictionaries.
        failure_examples.sort(
            key=lambda ex: (
                ex.failure_category.value,
                ex.family.value,
                ex.scene_id,
                ex.question_id,
                ex.record_id,
            )
        )
        sorted_category_counts = dict(sorted(category_counts.items()))
        sorted_category_rates = {
            key: _rate(val, total) for key, val in sorted_category_counts.items()
        }
        sorted_failures_by_family = {
            fam: dict(sorted(cat_map.items()))
            for fam, cat_map in sorted(failures_by_family.items())
        }
        sorted_failures_by_type = {
            atype: dict(sorted(cat_map.items()))
            for atype, cat_map in sorted(failures_by_answer_type.items())
        }

        sample_ids_by_cat: dict[str, list[str]] = {}
        for ex in failure_examples:
            sample_ids_by_cat.setdefault(ex.failure_category.value, []).append(ex.question_id)

        eval_id = (
            str(summary_raw.get("evaluation_id"))
            if summary_raw and summary_raw.get("evaluation_id") is not None
            else (eval_records[0].evaluation_id if eval_records else None)
        )
        dataset_id = (
            str(summary_raw.get("dataset_id"))
            if summary_raw and summary_raw.get("dataset_id") is not None
            else (eval_records[0].dataset_id if eval_records else None)
        )
        model_name = (
            str(summary_raw.get("model_name"))
            if summary_raw and summary_raw.get("model_name") is not None
            else (eval_records[0].model_name if eval_records else None)
        )
        provider_name = (
            str(summary_raw.get("provider_name"))
            if summary_raw and summary_raw.get("provider_name") is not None
            else (eval_records[0].provider_name if eval_records else None)
        )

        failure_count = len(failure_examples)

        metrics_summary = MetricsSummary(
            evaluation_id=eval_id,
            dataset_id=dataset_id,
            model_name=model_name,
            provider_name=provider_name,
            evaluation_nature=nature,
            is_mock=is_mock,
            evaluation_notice=notice,
            total_items=total,
            completed_predictions=overall.completed,
            correct_predictions=overall.correct,
            incorrect_predictions=overall.incorrect,
            accuracy=overall.accuracy,
            completed_accuracy=overall.completed_accuracy,
            unparseable_count=overall.unparseable,
            unparseable_rate=_rate(overall.unparseable, total),
            adapter_error_count=overall.adapter_errors,
            adapter_error_rate=_rate(overall.adapter_errors, total),
            missing_image_count=overall.missing_images,
            missing_image_rate=_rate(overall.missing_images, total),
            failure_count=failure_count,
            failure_rate=_rate(failure_count, total),
            by_family=by_family,
            by_answer_type=by_answer_type,
            failure_category_counts=sorted_category_counts,
            failure_category_rates=sorted_category_rates,
        )

        failure_analysis = FailureAnalysisReport(
            evaluation_id=eval_id,
            dataset_id=dataset_id,
            model_name=model_name,
            evaluation_nature=nature,
            evaluation_notice=notice,
            total_items=total,
            total_failures=failure_count,
            failure_rate=_rate(failure_count, total),
            category_counts=sorted_category_counts,
            category_rates=sorted_category_rates,
            failures_by_family=sorted_failures_by_family,
            failures_by_answer_type=sorted_failures_by_type,
            sample_failure_ids_by_category=dict(sorted(sample_ids_by_cat.items())),
        )

        artifacts: dict[str, str] = {}
        resolved_out: Path | None = None
        if output_dir is not None:
            resolved_out = Path(output_dir)
            if resolved_out.exists() and not resolved_out.is_dir():
                raise FileExistsError(
                    f"output_dir points to an existing file: '{resolved_out}'."
                )
            resolved_out.mkdir(parents=True, exist_ok=True)

            metrics_path = resolved_out / "metrics_summary.json"
            failure_path = resolved_out / "failure_analysis.json"
            examples_path = resolved_out / "failure_examples.jsonl"

            metrics_path.write_text(
                metrics_summary.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            failure_path.write_text(
                failure_analysis.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            example_lines = [ex.model_dump_json() for ex in failure_examples]
            examples_path.write_text(
                ("\n".join(example_lines) + "\n") if example_lines else "",
                encoding="utf-8",
            )
            artifacts = {
                "metrics_summary_json": "metrics_summary.json",
                "failure_analysis_json": "failure_analysis.json",
                "failure_examples_jsonl": "failure_examples.jsonl",
            }

        return MetricsAnalysisResult(
            output_dir=resolved_out,
            metrics_summary=metrics_summary,
            failure_analysis=failure_analysis,
            failure_examples=failure_examples,
            artifacts=artifacts,
        )


def analyze_evaluation(
    eval_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
) -> MetricsAnalysisResult:
    """Convenience entry point to analyze a Phase 9 evaluation directory."""
    return MetricsAnalyzer().analyze_directory(eval_dir=eval_dir, output_dir=output_dir)


def analyze_records(
    records: Sequence[EvaluationRecord | dict[str, Any]],
    *,
    summary: EvaluationSummary | dict[str, Any] | None = None,
    output_dir: Path | str | None = None,
) -> MetricsAnalysisResult:
    """Convenience entry point to analyze evaluation records in memory."""
    return MetricsAnalyzer().analyze_records(
        records=records,
        summary=summary,
        output_dir=output_dir,
    )
