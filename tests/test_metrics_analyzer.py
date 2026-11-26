"""Unit tests for Phase 10 MetricsAnalyzer and failure reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visual_task_factory.metrics_analyzer import (
    MOCK_EVALUATION_NOTICE,
    UNKNOWN_EVALUATION_NOTICE,
    EvaluationNature,
    FailureAnalysisReport,
    FailureCategory,
    FailureExample,
    MetricsAnalyzer,
    MetricsSummary,
    MixedEvaluationError,
    analyze_evaluation,
    analyze_records,
)
from visual_task_factory.model_evaluator import EvaluationRecord, PredictionStatus
from visual_task_factory.models import AnswerType, QuestionFamily, ReferenceAnswer


def _make_record(
    idx: int,
    *,
    family: QuestionFamily = QuestionFamily.CONSTRAINED_REACHABILITY,
    answer_type: AnswerType = AnswerType.EXIT_ID,
    status: PredictionStatus = PredictionStatus.OK,
    is_correct: bool = True,
    is_mock: bool = True,
    provider_name: str = "mock_synthetic",
    error_message: str | None = None,
) -> EvaluationRecord:
    if answer_type == AnswerType.EXIT_ID:
        ref = ReferenceAnswer(answer_type=AnswerType.EXIT_ID, value="E1", solver_name="s")
        raw_pred = "E1" if is_correct else "E2"
        norm_pred: bool | int | float | str | None = raw_pred if status == PredictionStatus.OK else None
    elif answer_type == AnswerType.INTEGER:
        ref = ReferenceAnswer(answer_type=AnswerType.INTEGER, value=2, solver_name="s")
        raw_pred = "2" if is_correct else "4"
        norm_pred = (2 if is_correct else 4) if status == PredictionStatus.OK else None
    elif answer_type == AnswerType.BOOLEAN:
        ref = ReferenceAnswer(answer_type=AnswerType.BOOLEAN, value=True, solver_name="s")
        raw_pred = "true" if is_correct else "false"
        norm_pred = (True if is_correct else False) if status == PredictionStatus.OK else None
    else:
        ref = ReferenceAnswer(
            answer_type=AnswerType.NUMERIC,
            value=1.5,
            unit="m",
            numeric_tolerance=0.05,
            solver_name="s",
        )
        raw_pred = "1.5 m" if is_correct else "3.0 m"
        norm_pred = (1.5 if is_correct else 3.0) if status == PredictionStatus.OK else None

    if status != PredictionStatus.OK:
        is_correct = False
        if status == PredictionStatus.UNPARSEABLE:
            raw_pred = "unclear"
        else:
            raw_pred = None

    return EvaluationRecord(
        evaluation_id="eval_test_1",
        dataset_id="ds_test_1",
        record_id=f"rec_{idx:03d}",
        scene_id=f"scene_{idx:03d}",
        question_id=f"q_{idx:03d}",
        family=family,
        expected_answer_type=answer_type,
        image_rel_path=f"images/scene_{idx:03d}.png",
        status=status,
        model_name="test_model",
        provider_name=provider_name,
        is_mock_evaluation=is_mock,
        raw_prediction=raw_pred,
        normalized_prediction=norm_pred,
        reference_answer=ref,
        is_correct=is_correct,
        error_message=error_message,
    )


def test_perfect_accuracy_and_zero_accuracy() -> None:
    perfect_recs = [
        _make_record(1, is_correct=True),
        _make_record(
            2,
            family=QuestionFamily.REGION_CONTAINMENT_COUNTING,
            answer_type=AnswerType.INTEGER,
            is_correct=True,
        ),
    ]
    perfect_res = analyze_records(perfect_recs)
    assert perfect_res.metrics_summary.total_items == 2
    assert perfect_res.metrics_summary.correct_predictions == 2
    assert perfect_res.metrics_summary.incorrect_predictions == 0
    assert perfect_res.metrics_summary.accuracy == pytest.approx(1.0)
    assert perfect_res.metrics_summary.failure_count == 0
    assert perfect_res.failure_examples == []

    zero_recs = [
        _make_record(1, is_correct=False),
        _make_record(2, status=PredictionStatus.UNPARSEABLE),
    ]
    zero_res = analyze_records(zero_recs)
    assert zero_res.metrics_summary.correct_predictions == 0
    assert zero_res.metrics_summary.accuracy == pytest.approx(0.0)
    assert zero_res.metrics_summary.failure_count == 2
    assert zero_res.metrics_summary.failure_rate == pytest.approx(1.0)


def test_mixed_statuses_grouped_by_family_and_answer_type() -> None:
    records = [
        # Constrained reachability (exit_id): 1 correct, 1 incorrect
        _make_record(
            1,
            family=QuestionFamily.CONSTRAINED_REACHABILITY,
            answer_type=AnswerType.EXIT_ID,
            status=PredictionStatus.OK,
            is_correct=True,
        ),
        _make_record(
            2,
            family=QuestionFamily.CONSTRAINED_REACHABILITY,
            answer_type=AnswerType.EXIT_ID,
            status=PredictionStatus.OK,
            is_correct=False,
        ),
        # Region counting (integer): 1 unparseable
        _make_record(
            3,
            family=QuestionFamily.REGION_CONTAINMENT_COUNTING,
            answer_type=AnswerType.INTEGER,
            status=PredictionStatus.UNPARSEABLE,
            error_message="Ambiguous count",
        ),
        # Clearance (boolean): 1 adapter error; (numeric): 1 missing image, 1 correct
        _make_record(
            4,
            family=QuestionFamily.SCALE_GROUNDED_CLEARANCE,
            answer_type=AnswerType.BOOLEAN,
            status=PredictionStatus.ADAPTER_ERROR,
            error_message="Timeout",
        ),
        _make_record(
            5,
            family=QuestionFamily.SCALE_GROUNDED_CLEARANCE,
            answer_type=AnswerType.NUMERIC,
            status=PredictionStatus.MISSING_IMAGE,
            error_message="Image missing",
        ),
        _make_record(
            6,
            family=QuestionFamily.SCALE_GROUNDED_CLEARANCE,
            answer_type=AnswerType.NUMERIC,
            status=PredictionStatus.OK,
            is_correct=True,
        ),
    ]

    res = analyze_records(records)
    ms = res.metrics_summary
    assert ms.total_items == 6
    assert ms.completed_predictions == 3
    assert ms.correct_predictions == 2
    assert ms.incorrect_predictions == 1
    assert ms.unparseable_count == 1
    assert ms.adapter_error_count == 1
    assert ms.missing_image_count == 1
    assert ms.accuracy == pytest.approx(round(2 / 6, 4))
    assert ms.completed_accuracy == pytest.approx(round(2 / 3, 4))
    assert ms.unparseable_rate == pytest.approx(round(1 / 6, 4))
    assert ms.adapter_error_rate == pytest.approx(round(1 / 6, 4))
    assert ms.missing_image_rate == pytest.approx(round(1 / 6, 4))

    assert ms.by_family["constrained_reachability"].accuracy == pytest.approx(0.5)
    assert ms.by_family["region_containment_counting"].unparseable == 1
    assert ms.by_family["scale_grounded_clearance"].total == 3
    assert ms.by_family["scale_grounded_clearance"].correct == 1

    assert ms.by_answer_type["exit_id"].accuracy == pytest.approx(0.5)
    assert ms.by_answer_type["integer"].unparseable == 1
    assert ms.by_answer_type["boolean"].adapter_errors == 1
    assert ms.by_answer_type["numeric"].missing_images == 1
    assert ms.by_answer_type["numeric"].correct == 1

    assert res.failure_analysis.total_failures == 4
    assert res.failure_analysis.category_counts == {
        FailureCategory.ADAPTER_ERROR.value: 1,
        FailureCategory.INCORRECT_PREDICTION.value: 1,
        FailureCategory.MISSING_IMAGE.value: 1,
        FailureCategory.UNPARSEABLE_PREDICTION.value: 1,
    }


def test_mock_real_mixed_and_unknown_evaluation_handling(tmp_path: Path) -> None:
    # 1. Mock evaluation prominently labeled
    mock_res = analyze_records([_make_record(1, is_mock=True)])
    assert mock_res.metrics_summary.evaluation_nature == EvaluationNature.MOCK
    assert mock_res.metrics_summary.is_mock is True
    assert mock_res.metrics_summary.evaluation_notice == MOCK_EVALUATION_NOTICE

    # 2. Mixing mock and real records raises MixedEvaluationError
    with pytest.raises(MixedEvaluationError, match="Cannot combine mock and real"):
        analyze_records(
            [
                _make_record(1, is_mock=True, provider_name="mock_synthetic"),
                _make_record(2, is_mock=False, provider_name="live_provider"),
            ]
        )

    # 3. Missing is_mock_evaluation metadata in raw dict -> UNKNOWN (never assumes real)
    raw_missing_mock = _make_record(1, is_mock=True, provider_name="custom").model_dump(mode="json")
    del raw_missing_mock["is_mock_evaluation"]
    unknown_res = analyze_records([raw_missing_mock])
    assert unknown_res.metrics_summary.evaluation_nature == EvaluationNature.UNKNOWN
    assert unknown_res.metrics_summary.is_mock is None
    assert unknown_res.metrics_summary.evaluation_notice == UNKNOWN_EVALUATION_NOTICE

    # 4. Empty directory without evaluation_summary.json -> UNKNOWN and zero counts
    empty_dir = tmp_path / "empty_eval"
    empty_dir.mkdir()
    (empty_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    empty_res = analyze_evaluation(empty_dir)
    assert empty_res.metrics_summary.total_items == 0
    assert empty_res.metrics_summary.accuracy == 0.0
    assert empty_res.metrics_summary.evaluation_nature == EvaluationNature.UNKNOWN


def test_deterministic_directory_artifacts_and_serialization(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval_src"
    eval_dir.mkdir()
    recs = [
        _make_record(3, is_correct=False),
        _make_record(1, is_correct=True),
        _make_record(2, status=PredictionStatus.UNPARSEABLE, error_message="bad"),
    ]
    (eval_dir / "predictions.jsonl").write_text(
        "\n".join(r.model_dump_json() for r in recs) + "\n",
        encoding="utf-8",
    )
    (eval_dir / "evaluation_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "evaluation_id": "eval_test_1",
                "dataset_id": "ds_test_1",
                "model_name": "test_model",
                "provider_name": "mock_synthetic",
                "is_mock_evaluation": True,
                "evaluation_notice": "SYNTHETIC MOCK EVALUATION - NOT REAL MODEL CAPABILITY",
                "total_records": 3,
                "completed_predictions": 2,
                "correct_count": 1,
                "incorrect_count": 1,
                "unparseable_count": 1,
                "adapter_error_count": 0,
                "missing_image_count": 0,
                "raw_accuracy": 0.3333,
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    analyzer = MetricsAnalyzer()
    res_a = analyzer.analyze_directory(eval_dir, output_dir=out_a)
    res_b = analyzer.analyze_directory(eval_dir, output_dir=out_b)

    for fname in ("metrics_summary.json", "failure_analysis.json", "failure_examples.jsonl"):
        assert (out_a / fname).is_file()
        assert (out_a / fname).read_text(encoding="utf-8") == (
            out_b / fname
        ).read_text(encoding="utf-8")

    loaded_metrics = MetricsSummary.model_validate_json(
        (out_a / "metrics_summary.json").read_text(encoding="utf-8")
    )
    loaded_failure = FailureAnalysisReport.model_validate_json(
        (out_a / "failure_analysis.json").read_text(encoding="utf-8")
    )
    loaded_examples = [
        FailureExample.model_validate_json(ln)
        for ln in (out_a / "failure_examples.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert loaded_metrics == res_a.metrics_summary
    assert loaded_failure == res_a.failure_analysis
    assert loaded_examples == res_a.failure_examples
    # Deterministic ordering: incorrect_prediction before unparseable_prediction
    assert [e.question_id for e in loaded_examples] == ["q_003", "q_002"]
