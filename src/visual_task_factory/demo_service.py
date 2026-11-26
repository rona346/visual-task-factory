"""Framework-agnostic backend service for the Phase 11 Visual Task Factory Demo UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from visual_task_factory.dataset_builder_3d import (
    Dataset3DError,
    load_and_validate_dataset_3d,
)
from visual_task_factory.dataset_pipeline import (
    DatasetManifest,
    DatasetPipelineConfig,
    DatasetRecord,
    build_dataset,
)
from visual_task_factory.demo_3d import _answers_match, _recompute_solver_answer
from visual_task_factory.metrics_analyzer import (
    MOCK_EVALUATION_NOTICE,
    EvaluationNature,
    FailureAnalysisReport,
    FailureExample,
    MetricsSummary,
    analyze_evaluation,
)
from visual_task_factory.model_evaluator import (
    EvaluationConfig,
    EvaluationSummary,
    MockAdapterMode,
    MockModelAdapter,
    evaluate_dataset,
)
from visual_task_factory.models import DifficultyTier, VisualTheme
from visual_task_factory.qa_3d import (
    AnswerType3D,
    BenchmarkItem3D,
    QuestionFamily3D,
    ReferenceAnswer3D,
    VisualQuestion3D,
)
from visual_task_factory.scene_generator import SUPPORTED_TOPOLOGIES
from visual_task_factory.solver_3d import GroundTruthSolver3D
from visual_task_factory.variation_engine import VariationPlanConfig

BENCHMARK_3D_NOTICE = (
    "Offline 3D Benchmark Ground Truth & CPU Solver Verification — Not Live VLM Capability"
)


class DemoSampleRequest(BaseModel):
    """Bounded configuration for generating a small demo sample."""

    model_config = ConfigDict(extra="forbid")

    base_seed: int = Field(default=101, ge=0)
    scene_count: int = Field(default=2, ge=1, le=4)
    topology_id: str | None = None
    difficulty: DifficultyTier | None = None
    theme: VisualTheme | None = None
    run_mock_eval: bool = False


class DemoDatasetView(BaseModel):
    """Safe view-model for a generated or loaded Phase 8 dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_dir: str
    dataset_id: str | None = None
    manifest: DatasetManifest | None = None
    records: list[DatasetRecord] = Field(default_factory=list)
    missing_files: list[str] = Field(default_factory=list)
    error_message: str | None = None


class DemoEvaluationView(BaseModel):
    """Safe view-model for Phase 9 evaluation and Phase 10 metrics/failure reports."""

    model_config = ConfigDict(extra="forbid")

    eval_dir: str
    evaluation_summary: EvaluationSummary | None = None
    metrics_summary: MetricsSummary | None = None
    failure_analysis: FailureAnalysisReport | None = None
    failure_examples: list[FailureExample] = Field(default_factory=list)
    mock_warning_banner: str | None = None
    missing_files: list[str] = Field(default_factory=list)
    error_message: str | None = None


class DemoSampleResult(BaseModel):
    """Result returned after generating a small demo dataset (and optional mock eval)."""

    model_config = ConfigDict(extra="forbid")

    dataset: DemoDatasetView
    evaluation: DemoEvaluationView | None = None


class Demo3DRecordView(BaseModel):
    """Safe view-model pairing a 3D dataset record with recomputed CPU solver output."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    scene_id: str
    question_id: str
    family: QuestionFamily3D
    expected_answer_type: AnswerType3D
    image_rel_path: str
    image_width_px: int
    image_height_px: int
    prompt_text: str
    reference_answer: ReferenceAnswer3D
    solver_prediction: ReferenceAnswer3D
    solver_matches_ground_truth: bool
    validation_is_valid: bool


class Demo3DDatasetView(BaseModel):
    """Safe view-model for a validated offline 3D benchmark dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_dir: str
    dataset_id: str | None = None
    scene_count: int = 0
    record_count: int = 0
    matched_count: int = 0
    valid_count: int = 0
    family_counts: dict[str, int] = Field(default_factory=dict)
    benchmark_notice: str = BENCHMARK_3D_NOTICE
    records: list[Demo3DRecordView] = Field(default_factory=list)
    error_message: str | None = None


class DemoService:
    """Connects existing Phase 8–10 APIs cleanly for the lightweight demo interface."""

    def resolve_safe_image_path(
        self,
        dataset_dir: Path | str,
        image_rel_path: str,
    ) -> Path:
        """Resolve an image path inside dataset_dir while blocking directory traversal."""
        if not str(dataset_dir).strip():
            raise ValueError("dataset_dir must not be empty.")
        if not image_rel_path.strip():
            raise ValueError("image_rel_path must not be empty.")

        ds_root = Path(dataset_dir).resolve()
        if not ds_root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: '{ds_root}'.")

        target = (ds_root / image_rel_path).resolve()
        if ds_root not in target.parents:
            raise PermissionError("Image path escapes dataset directory.")
        if target.suffix.lower() != ".png":
            raise ValueError("Only PNG image files may be served.")
        if not target.is_file():
            raise FileNotFoundError(f"Image file not found: '{image_rel_path}'.")
        return target

    def load_dataset_view(self, dataset_dir: Path | str) -> DemoDatasetView:
        """Load dataset.jsonl and manifest.json gracefully without raising on missing artifacts."""
        raw_dir = str(dataset_dir).strip()
        if not raw_dir:
            return DemoDatasetView(
                dataset_dir="",
                error_message="Dataset directory path must not be empty.",
            )

        ds_path = Path(raw_dir)
        if not ds_path.exists() or not ds_path.is_dir():
            return DemoDatasetView(
                dataset_dir=raw_dir,
                missing_files=["dataset.jsonl", "manifest.json"],
                error_message=f"Dataset directory does not exist: '{raw_dir}'.",
            )

        missing: list[str] = []
        manifest: DatasetManifest | None = None
        manifest_path = ds_path / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = DatasetManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                return DemoDatasetView(
                    dataset_dir=raw_dir,
                    error_message=f"Failed to parse manifest.json: {exc}",
                )
        else:
            missing.append("manifest.json")

        records: list[DatasetRecord] = []
        dataset_path = ds_path / "dataset.jsonl"
        if dataset_path.is_file():
            try:
                for line in dataset_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        records.append(DatasetRecord.model_validate_json(line))
            except Exception as exc:
                return DemoDatasetView(
                    dataset_dir=raw_dir,
                    manifest=manifest,
                    missing_files=missing,
                    error_message=f"Failed to parse dataset.jsonl: {exc}",
                )
        else:
            missing.append("dataset.jsonl")

        dataset_id = (
            manifest.dataset_id
            if manifest is not None
            else (records[0].dataset_id if records else None)
        )
        return DemoDatasetView(
            dataset_dir=raw_dir,
            dataset_id=dataset_id,
            manifest=manifest,
            records=records,
            missing_files=missing,
        )

    def load_evaluation_view(self, eval_dir: Path | str) -> DemoEvaluationView:
        """Load Phase 9 evaluation summary and Phase 10 metrics/failure reports gracefully."""
        raw_dir = str(eval_dir).strip()
        if not raw_dir:
            return DemoEvaluationView(
                eval_dir="",
                error_message="Evaluation directory path must not be empty.",
            )

        ev_path = Path(raw_dir)
        if not ev_path.exists() or not ev_path.is_dir():
            return DemoEvaluationView(
                eval_dir=raw_dir,
                missing_files=[
                    "evaluation_summary.json",
                    "metrics_summary.json",
                    "failure_analysis.json",
                    "failure_examples.jsonl",
                ],
                error_message=f"Evaluation directory does not exist: '{raw_dir}'.",
            )

        missing: list[str] = []
        eval_summary: EvaluationSummary | None = None
        metrics_summary: MetricsSummary | None = None
        failure_analysis: FailureAnalysisReport | None = None
        failure_examples: list[FailureExample] = []

        eval_sum_path = ev_path / "evaluation_summary.json"
        if eval_sum_path.is_file():
            try:
                eval_summary = EvaluationSummary.model_validate_json(
                    eval_sum_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                return DemoEvaluationView(
                    eval_dir=raw_dir,
                    error_message=f"Failed to parse evaluation_summary.json: {exc}",
                )
        else:
            missing.append("evaluation_summary.json")

        metrics_path = ev_path / "metrics_summary.json"
        if metrics_path.is_file():
            try:
                metrics_summary = MetricsSummary.model_validate_json(
                    metrics_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                return DemoEvaluationView(
                    eval_dir=raw_dir,
                    evaluation_summary=eval_summary,
                    error_message=f"Failed to parse metrics_summary.json: {exc}",
                )
        else:
            missing.append("metrics_summary.json")

        failure_path = ev_path / "failure_analysis.json"
        if failure_path.is_file():
            try:
                failure_analysis = FailureAnalysisReport.model_validate_json(
                    failure_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                return DemoEvaluationView(
                    eval_dir=raw_dir,
                    evaluation_summary=eval_summary,
                    metrics_summary=metrics_summary,
                    error_message=f"Failed to parse failure_analysis.json: {exc}",
                )
        else:
            missing.append("failure_analysis.json")

        examples_path = ev_path / "failure_examples.jsonl"
        if examples_path.is_file():
            try:
                for line in examples_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        failure_examples.append(FailureExample.model_validate_json(line))
            except Exception as exc:
                return DemoEvaluationView(
                    eval_dir=raw_dir,
                    evaluation_summary=eval_summary,
                    metrics_summary=metrics_summary,
                    failure_analysis=failure_analysis,
                    error_message=f"Failed to parse failure_examples.jsonl: {exc}",
                )
        else:
            missing.append("failure_examples.jsonl")

        is_mock = (
            (eval_summary is not None and eval_summary.is_mock_evaluation)
            or (
                metrics_summary is not None
                and metrics_summary.evaluation_nature == EvaluationNature.MOCK
            )
            or (
                failure_analysis is not None
                and failure_analysis.evaluation_nature == EvaluationNature.MOCK
            )
        )
        banner = MOCK_EVALUATION_NOTICE if is_mock else None

        return DemoEvaluationView(
            eval_dir=raw_dir,
            evaluation_summary=eval_summary,
            metrics_summary=metrics_summary,
            failure_analysis=failure_analysis,
            failure_examples=failure_examples,
            mock_warning_banner=banner,
            missing_files=missing,
        )

    def generate_demo_sample(
        self,
        request: DemoSampleRequest,
        output_dir: Path | str,
    ) -> DemoSampleResult:
        """Generate a small benchmark dataset sample (and optional mock evaluation)."""
        if not isinstance(request, DemoSampleRequest):
            raise TypeError(
                f"Expected DemoSampleRequest, got {type(request).__name__}."
            )

        out_path = Path(output_dir)
        topologies = (
            (request.topology_id,)
            if request.topology_id is not None
            else SUPPORTED_TOPOLOGIES
        )
        difficulties = (
            (request.difficulty,)
            if request.difficulty is not None
            else (DifficultyTier.EASY, DifficultyTier.MEDIUM, DifficultyTier.HARD)
        )
        themes = (
            (request.theme,)
            if request.theme is not None
            else (
                VisualTheme.SAFETY_STANDARD,
                VisualTheme.BLUEPRINT,
                VisualTheme.HIGH_CONTRAST,
                VisualTheme.CLUTTERED_OPS,
            )
        )

        pipeline_cfg = DatasetPipelineConfig(
            dataset_id=f"demo_s{request.base_seed}_n{request.scene_count}",
            variation_plan=VariationPlanConfig(
                base_seed=request.base_seed,
                count=request.scene_count,
                topologies=topologies,
                difficulties=difficulties,
                themes=themes,
                min_obstacles=3,
                max_obstacles=8,
            ),
            max_questions_per_family=1,
        )
        build_dataset(pipeline_cfg, out_path)
        dataset_view = self.load_dataset_view(out_path)

        eval_view: DemoEvaluationView | None = None
        if request.run_mock_eval:
            eval_dir = out_path / "mock_eval"
            adapter = MockModelAdapter(
                seed=request.base_seed,
                mode=MockAdapterMode.BLIND_HASH,
            )
            evaluate_dataset(
                out_path,
                eval_dir,
                adapter,
                config=EvaluationConfig(evaluation_id=f"demo_mock_s{request.base_seed}"),
            )
            analyze_evaluation(eval_dir)
            eval_view = self.load_evaluation_view(eval_dir)

        return DemoSampleResult(dataset=dataset_view, evaluation=eval_view)

    def load_dataset_3d_view(self, dataset_dir: Path | str = "data/3d") -> Demo3DDatasetView:
        """Load the validated 3D dataset and recompute CPU solver answers offline."""
        raw_dir = str(dataset_dir).strip()
        if not raw_dir:
            return Demo3DDatasetView(
                dataset_dir="",
                error_message="3D dataset directory path must not be empty.",
            )

        ds_path = Path(raw_dir)
        if not ds_path.exists():
            return Demo3DDatasetView(
                dataset_dir=raw_dir,
                error_message=f"3D dataset path does not exist: '{raw_dir}'.",
            )

        try:
            bundle = load_and_validate_dataset_3d(
                ds_path, verify_images=True, revalidate_qa=True
            )
            solver = GroundTruthSolver3D()
            record_views: list[Demo3DRecordView] = []
            matched_count = 0
            valid_count = 0
            for rec in bundle.records:
                item = BenchmarkItem3D(
                    scene=bundle.scenes[rec.scene_id],
                    question=VisualQuestion3D(
                        question_id=rec.question_id,
                        scene_id=rec.scene_id,
                        family=rec.family,
                        template_id=rec.template_id,
                        prompt_text=rec.prompt_text,
                        expected_answer_type=rec.expected_answer_type,
                        reference_answer=rec.reference_answer,
                        solver_query=rec.solver_query,
                    ),
                )
                solver_pred = _recompute_solver_answer(solver, item)
                matched = _answers_match(solver_pred, rec.reference_answer)
                if matched:
                    matched_count += 1
                if rec.validation.is_valid:
                    valid_count += 1
                record_views.append(
                    Demo3DRecordView(
                        record_id=rec.record_id,
                        scene_id=rec.scene_id,
                        question_id=rec.question_id,
                        family=rec.family,
                        expected_answer_type=rec.expected_answer_type,
                        image_rel_path=rec.image_rel_path,
                        image_width_px=rec.image_width_px,
                        image_height_px=rec.image_height_px,
                        prompt_text=rec.prompt_text,
                        reference_answer=rec.reference_answer,
                        solver_prediction=solver_pred,
                        solver_matches_ground_truth=matched,
                        validation_is_valid=rec.validation.is_valid,
                    )
                )
        except (Dataset3DError, Exception) as exc:
            return Demo3DDatasetView(
                dataset_dir=raw_dir,
                error_message=f"Failed to load 3D benchmark dataset: {exc}",
            )

        return Demo3DDatasetView(
            dataset_dir=raw_dir,
            dataset_id=bundle.dataset_id,
            scene_count=bundle.scene_count,
            record_count=bundle.record_count,
            matched_count=matched_count,
            valid_count=valid_count,
            family_counts=bundle.family_counts,
            records=record_views,
        )
