"""Unit tests for the provider-agnostic ModelEvaluator and MockModelAdapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from visual_task_factory.dataset_pipeline import DatasetPipelineConfig, build_dataset
from visual_task_factory.model_evaluator import (
    EvaluationConfig,
    EvaluationRecord,
    EvaluationSummary,
    MockAdapterMode,
    MockModelAdapter,
    ModelEvaluator,
    ModelPrediction,
    ModelRequest,
    MultimodalModelAdapter,
    PredictionStatus,
    compare_prediction_to_reference,
    evaluate_dataset,
    normalize_prediction,
)
from visual_task_factory.models import AnswerType, ReferenceAnswer
from visual_task_factory.variation_engine import VariationPlanConfig


def _build_fixture_dataset(tmp_path: Path) -> Path:
    ds_dir = tmp_path / "fixture_ds"
    cfg = DatasetPipelineConfig(
        dataset_id="eval_fixture_v1",
        variation_plan=VariationPlanConfig(
            base_seed=77,
            count=2,
            min_obstacles=3,
            max_obstacles=7,
        ),
    )
    build_dataset(cfg, ds_dir)
    return ds_dir


def test_adapter_interface_and_deterministic_mock_predictions(tmp_path: Path) -> None:
    ds_dir = _build_fixture_dataset(tmp_path)
    before_dataset = (ds_dir / "dataset.jsonl").read_text(encoding="utf-8")

    adapter_a = MockModelAdapter(seed=42, mode=MockAdapterMode.BLIND_HASH)
    adapter_b = MockModelAdapter(seed=42, mode=MockAdapterMode.BLIND_HASH)

    assert isinstance(adapter_a, MultimodalModelAdapter)
    assert adapter_a.is_mock is True
    assert adapter_a.provider_name == "mock_synthetic"

    res_a = evaluate_dataset(ds_dir, tmp_path / "eval_a", adapter_a)
    res_b = evaluate_dataset(ds_dir, tmp_path / "eval_b", adapter_b)

    assert res_a.summary.is_mock_evaluation is True
    assert "MOCK" in res_a.summary.evaluation_notice
    assert res_a.records == res_b.records
    assert (tmp_path / "eval_a" / "predictions.jsonl").read_text(encoding="utf-8") == (
        tmp_path / "eval_b" / "predictions.jsonl"
    ).read_text(encoding="utf-8")

    # Source dataset must remain untouched.
    assert (ds_dir / "dataset.jsonl").read_text(encoding="utf-8") == before_dataset


def test_scripted_correct_incorrect_unparseable_and_adapter_error_handling(
    tmp_path: Path,
) -> None:
    ds_dir = _build_fixture_dataset(tmp_path)
    evaluator = ModelEvaluator(MockModelAdapter())
    ds_records, _ = evaluator.load_dataset(ds_dir)
    assert len(ds_records) >= 4

    # Script exact right answer on record 0, exact wrong answer on record 1, unparseable on 2, error on 3.
    r0, r1, r2, r3 = ds_records[0], ds_records[1], ds_records[2], ds_records[3]
    r0_val = r0.reference_answer.value
    r0_str = f"{r0_val} m" if r0.expected_answer_type == AnswerType.NUMERIC else str(r0_val)

    # Pick an intentionally wrong value for r1.
    if r1.expected_answer_type == AnswerType.EXIT_ID:
        r1_wrong = "E99"
    elif r1.expected_answer_type == AnswerType.INTEGER:
        r1_wrong = str(int(r1.reference_answer.value) + 5)
    elif r1.expected_answer_type == AnswerType.BOOLEAN:
        r1_wrong = str(not bool(r1.reference_answer.value)).lower()
    else:
        r1_wrong = f"{float(r1.reference_answer.value) + 3.0} m"

    adapter = MockModelAdapter(
        mode=MockAdapterMode.SCRIPTED,
        scripted_outputs={
            r0.question_id: r0_str,
            r1.question_id: r1_wrong,
            r2.question_id: "cannot decide maybe 2 or 3",
        },
        raise_on_question_ids={r3.question_id},
    )

    eval_out = tmp_path / "scripted_eval"
    result = evaluate_dataset(
        ds_dir,
        eval_out,
        adapter,
        config=EvaluationConfig(evaluation_id="scripted_test", max_records=4),
    )

    assert result.summary.total_records == 4
    assert result.summary.completed_predictions == 2
    assert result.summary.correct_count == 1
    assert result.summary.incorrect_count == 1
    assert result.summary.unparseable_count == 1
    assert result.summary.adapter_error_count == 1
    assert result.summary.missing_image_count == 0
    assert result.summary.is_mock_evaluation is True

    by_qid = {r.question_id: r for r in result.records}
    assert by_qid[r0.question_id].status == PredictionStatus.OK
    assert by_qid[r0.question_id].is_correct is True
    assert by_qid[r1.question_id].status == PredictionStatus.OK
    assert by_qid[r1.question_id].is_correct is False
    assert by_qid[r2.question_id].status == PredictionStatus.UNPARSEABLE
    assert by_qid[r2.question_id].is_correct is False
    assert by_qid[r3.question_id].status == PredictionStatus.ADAPTER_ERROR
    assert "Simulated adapter failure" in (by_qid[r3.question_id].error_message or "")

    # Verify JSONL & summary JSON roundtrip cleanly.
    loaded_summary = EvaluationSummary.model_validate_json(
        (eval_out / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert loaded_summary == result.summary
    loaded_records = [
        EvaluationRecord.model_validate_json(ln)
        for ln in (eval_out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert loaded_records == result.records


def test_missing_image_handling(tmp_path: Path) -> None:
    ds_dir = _build_fixture_dataset(tmp_path)
    evaluator = ModelEvaluator(MockModelAdapter())
    ds_records, _ = evaluator.load_dataset(ds_dir)
    first_img = ds_dir / ds_records[0].image_rel_path
    first_img.unlink()

    res = evaluator.evaluate(
        ds_dir,
        tmp_path / "missing_img_eval",
        config=EvaluationConfig(max_records=3),
    )
    assert res.summary.missing_image_count >= 1
    assert res.records[0].status == PredictionStatus.MISSING_IMAGE
    assert res.records[0].is_correct is False


def test_conservative_prediction_normalization_across_answer_types() -> None:
    # Boolean
    assert normalize_prediction("True.", AnswerType.BOOLEAN).parsed_value is True
    assert normalize_prediction("no", AnswerType.BOOLEAN).parsed_value is False
    assert normalize_prediction("1", AnswerType.BOOLEAN).is_valid is False
    assert normalize_prediction("true or false", AnswerType.BOOLEAN).is_valid is False

    # Integer
    assert normalize_prediction("3", AnswerType.INTEGER).parsed_value == 3
    assert normalize_prediction("Answer: 2", AnswerType.INTEGER).parsed_value == 2
    assert normalize_prediction("3.5", AnswerType.INTEGER).is_valid is False
    assert normalize_prediction("between 2 and 4", AnswerType.INTEGER).is_valid is False

    # Numeric + unit + tolerance comparison
    num_ok = normalize_prediction("1.52 meters", AnswerType.NUMERIC, expected_unit="m")
    assert num_ok.is_valid is True and num_ok.parsed_value == pytest.approx(1.52)
    assert (
        normalize_prediction("150 cm", AnswerType.NUMERIC, expected_unit="m").is_valid
        is False
    )
    ref_num = ReferenceAnswer(
        answer_type=AnswerType.NUMERIC,
        value=1.5,
        unit="m",
        numeric_tolerance=0.05,
        solver_name="euclidean_clearance_v1",
    )
    assert compare_prediction_to_reference(1.54, ref_num) is True
    assert compare_prediction_to_reference(1.60, ref_num) is False

    # Exit ID + candidate choices
    exit_ok = normalize_prediction(
        "The shortest route is e2.",
        AnswerType.EXIT_ID,
        candidate_choices=["E1", "E2", "E3"],
    )
    assert exit_ok.is_valid is True and exit_ok.parsed_value == "E2"
    assert (
        normalize_prediction(
            "Either E1 or E2",
            AnswerType.EXIT_ID,
            candidate_choices=["E1", "E2", "E3"],
        ).is_valid
        is False
    )


def test_custom_adapter_and_invalid_inputs() -> None:
    class _FailingTypeAdapter:
        pass

    with pytest.raises(TypeError, match="MultimodalModelAdapter"):
        ModelEvaluator(_FailingTypeAdapter())  # type: ignore[arg-type]

    class _EchoAdapter(MultimodalModelAdapter):
        @property
        def model_name(self) -> str:
            return "echo_custom"

        @property
        def provider_name(self) -> str:
            return "custom_test"

        @property
        def is_mock(self) -> bool:
            return True

        def predict(self, request: ModelRequest) -> ModelPrediction:
            return ModelPrediction(
                raw_output="E1",
                model_name=self.model_name,
                provider_name=self.provider_name,
                is_mock=self.is_mock,
            )

    req = ModelRequest(
        record_id="rec1",
        scene_id="sc1",
        question_id="q1",
        image_path=Path("dummy.png"),
        prompt_text="Which exit?",
        expected_answer_type=AnswerType.EXIT_ID,
        candidate_choices=["E1", "E2"],
    )
    pred = _EchoAdapter().predict(req)
    assert pred.raw_output == "E1"
