"""Isolated Phase 7 end-to-end 3D demo runner and CLI orchestrating Phases 1-5."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Literal, Sequence

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from visual_task_factory.dataset_builder_3d import (
    build_default_3d_scenes,
    load_and_validate_dataset_3d,
)
from visual_task_factory.qa_3d import (
    AnswerType3D,
    BenchmarkItem3D,
    CameraVisibilityQuery3D,
    GroundTruthQAValidator3D,
    OverlapQuery3D,
    QuestionFamily3D,
    ReferenceAnswer3D,
    VerticalClearanceQuery3D,
    VerticalRelationQuery3D,
    VisualQuestion3D,
    evaluate_camera_visibility_query,
    evaluate_overlap_query,
    evaluate_vertical_clearance_query,
    evaluate_vertical_relation_query,
)
from visual_task_factory.question_generator_3d import (
    QuestionGenerator3D,
    QuestionGenerator3DConfig,
)
from visual_task_factory.renderer_3d import Render3DConfig, render_scene_3d
from visual_task_factory.solver_3d import GroundTruthSolver3D

DEFAULT_3D_DEMO_OUTPUT_DIR = Path("artifacts/demo_3d")


class Demo3DConfig(BaseModel):
    """Configuration controlling the deterministic 3D end-to-end demo flow."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["fresh_scene", "existing_dataset"] = "fresh_scene"
    dataset_path: str | None = None
    scene_index: int = Field(default=0, ge=0)
    max_questions: int = Field(default=6, ge=1, le=20)
    max_questions_per_family: int = Field(default=2, ge=1, le=10)
    image_width_px: int = Field(default=800, ge=240, le=4096)
    image_height_px: int = Field(default=540, ge=180, le=4096)
    supersample: Literal[1, 2, 3] = 2

    @model_validator(mode="after")
    def _validate_mode_requirements(self) -> Demo3DConfig:
        if self.mode == "existing_dataset":
            if self.dataset_path is None or not self.dataset_path.strip():
                raise ValueError(
                    "dataset_path is required when Demo3DConfig.mode == 'existing_dataset'."
                )
        return self


class Demo3DQuestionStep(BaseModel):
    """Single 3D question paired with ground truth, solver prediction, and QA status."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    scene_id: str
    family: QuestionFamily3D
    expected_answer_type: AnswerType3D
    prompt_text: str
    solver_query: dict[str, Any]
    ground_truth_answer: ReferenceAnswer3D
    solver_prediction: ReferenceAnswer3D
    matches_ground_truth: bool
    validation_is_valid: bool
    validation_errors: list[str] = Field(default_factory=list)


class Demo3DReport(BaseModel):
    """Deterministic machine-readable report exported to demo_3d_report.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    mode: Literal["fresh_scene", "existing_dataset"]
    dataset_id: str | None = None
    scene_id: str
    scene_dimensions_m: tuple[float, float, float]
    entity_count: int = Field(..., ge=0)
    image_rel_path: str
    image_width_px: int = Field(..., gt=0)
    image_height_px: int = Field(..., gt=0)
    total_questions: int = Field(..., ge=0)
    matched_count: int = Field(..., ge=0)
    valid_count: int = Field(..., ge=0)
    all_steps_matched_and_valid: bool
    steps: list[Demo3DQuestionStep]
    artifacts: dict[str, str]


class Demo3DRunResult(BaseModel):
    """Output paths alongside the validated Demo3DReport."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    report_json_path: Path
    summary_md_path: Path
    image_path: Path
    report: Demo3DReport


def _recompute_solver_answer(
    solver: GroundTruthSolver3D,
    item: BenchmarkItem3D,
) -> ReferenceAnswer3D:
    raw = item.question.solver_query
    if raw is None:
        raise ValueError(f"Question {item.question.question_id!r} missing solver_query.")

    fam = item.question.family
    if fam == QuestionFamily3D.OBJECT_OVERLAP_3D:
        return evaluate_overlap_query(solver, item.scene, OverlapQuery3D.model_validate(raw))
    if fam == QuestionFamily3D.VERTICAL_RELATION_3D:
        return evaluate_vertical_relation_query(
            solver, item.scene, VerticalRelationQuery3D.model_validate(raw)
        )
    if fam == QuestionFamily3D.VERTICAL_CLEARANCE_3D:
        return evaluate_vertical_clearance_query(
            solver, item.scene, VerticalClearanceQuery3D.model_validate(raw)
        )
    if fam == QuestionFamily3D.CAMERA_VISIBILITY_3D:
        return evaluate_camera_visibility_query(
            solver, item.scene, CameraVisibilityQuery3D.model_validate(raw)
        )
    raise ValueError(f"Unsupported 3D question family: {fam!r}.")


def _answers_match(
    prediction: ReferenceAnswer3D,
    ground_truth: ReferenceAnswer3D,
) -> bool:
    if prediction.answer_type != ground_truth.answer_type:
        return False
    if ground_truth.answer_type == AnswerType3D.BOOLEAN:
        return type(prediction.value) is bool and prediction.value == ground_truth.value
    tol = ground_truth.numeric_tolerance if ground_truth.numeric_tolerance is not None else 0.05
    return abs(float(prediction.value) - float(ground_truth.value)) <= tol + 1e-9


def _render_demo_summary_markdown(report: Demo3DReport) -> str:
    lines = [
        "# Visual Task Factory — 3D Demo Summary",
        "",
        f"- **Mode:** `{report.mode}`",
        f"- **Dataset ID:** `{report.dataset_id or 'N/A (fresh scene)'}`",
        f"- **Scene ID:** `{report.scene_id}`",
        f"- **Room Size (W x D x H):** {report.scene_dimensions_m[0]:g}m x "
        f"{report.scene_dimensions_m[1]:g}m x {report.scene_dimensions_m[2]:g}m",
        f"- **Entities:** {report.entity_count}",
        f"- **Rendered PNG:** `{report.image_rel_path}` ({report.image_width_px}x{report.image_height_px})",
        f"- **Questions Evaluated:** {report.total_questions}",
        f"- **Solver Matches Ground Truth:** {report.matched_count}/{report.total_questions}",
        f"- **QA Validated:** {report.valid_count}/{report.total_questions}",
        "",
        "## Question & Solver Comparison",
        "",
        "| Question ID | Family | Prompt | Ground Truth | Solver Prediction | Match | QA Valid |",
        "| --- | --- | --- | --- | --- | :---: | :---: |",
    ]
    for step in report.steps:
        gt_val = (
            f"{step.ground_truth_answer.value} {step.ground_truth_answer.unit}"
            if step.ground_truth_answer.unit
            else str(step.ground_truth_answer.value)
        )
        pred_val = (
            f"{step.solver_prediction.value} {step.solver_prediction.unit}"
            if step.solver_prediction.unit
            else str(step.solver_prediction.value)
        )
        prompt_clean = step.prompt_text.replace("|", "\\|")
        lines.append(
            f"| `{step.question_id}` | `{step.family.value}` | {prompt_clean} "
            f"| `{gt_val}` | `{pred_val}` | `{step.matches_ground_truth}` | `{step.validation_is_valid}` |"
        )
    return "\n".join(lines) + "\n"


def run_demo_3d(
    config: Demo3DConfig | None = None,
    output_dir: Path | str = DEFAULT_3D_DEMO_OUTPUT_DIR,
) -> Demo3DRunResult:
    """Execute the full deterministic 7-step 3D demo flow and export demo artifacts."""
    cfg = config or Demo3DConfig()
    out_dir = Path(output_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise FileExistsError(f"output_dir points to an existing non-directory file: '{out_dir}'.")
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    solver = GroundTruthSolver3D()
    validator = GroundTruthQAValidator3D(solver=solver)

    dataset_id: str | None = None

    if cfg.mode == "fresh_scene":
        scenes = build_default_3d_scenes()
        if cfg.scene_index >= len(scenes):
            raise IndexError(
                f"scene_index={cfg.scene_index} out of range (available: 0..{len(scenes) - 1})."
            )
        scene = scenes[cfg.scene_index]
        image_rel_path = f"images/{scene.scene_id}.png"
        image_abs_path = out_dir / image_rel_path
        render_res = render_scene_3d(
            scene,
            output_path=image_abs_path,
            config=Render3DConfig(
                width_px=cfg.image_width_px,
                height_px=cfg.image_height_px,
                supersample=cfg.supersample,
            ),
        )
        img_w, img_h = render_res.width_px, render_res.height_px

        qgen = QuestionGenerator3D(
            QuestionGenerator3DConfig(max_questions_per_family=cfg.max_questions_per_family),
            solver=solver,
        )
        items = qgen.generate(scene)[: cfg.max_questions]
    else:
        assert cfg.dataset_path is not None
        src_path = Path(cfg.dataset_path)
        src_root = src_path if src_path.is_dir() else src_path.parent
        bundle = load_and_validate_dataset_3d(src_path, verify_images=True, revalidate_qa=True)
        dataset_id = bundle.dataset_id

        scene_ids = sorted(bundle.scenes)
        if cfg.scene_index >= len(scene_ids):
            raise IndexError(
                f"scene_index={cfg.scene_index} out of range for dataset (available: 0..{len(scene_ids) - 1})."
            )
        target_scene_id = scene_ids[cfg.scene_index]
        scene = bundle.scenes[target_scene_id]
        matching_records = [r for r in bundle.records if r.scene_id == target_scene_id][
            : cfg.max_questions
        ]
        if not matching_records:
            raise ValueError(
                f"No dataset records found for scene_id={target_scene_id!r} in '{src_path}'."
            )

        src_img_path = (src_root / matching_records[0].image_rel_path).resolve()
        image_rel_path = f"images/{scene.scene_id}.png"
        image_abs_path = out_dir / image_rel_path
        if src_img_path != image_abs_path.resolve():
            shutil.copyfile(src_img_path, image_abs_path)
        with Image.open(image_abs_path) as img:
            img_w, img_h = img.size

        items = [
            BenchmarkItem3D(
                scene=scene,
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
            for rec in matching_records
        ]

    steps: list[Demo3DQuestionStep] = []
    for item in items:
        solver_pred = _recompute_solver_answer(solver, item)
        qa_report = validator.validate(item)
        matched = _answers_match(solver_pred, item.question.reference_answer)
        steps.append(
            Demo3DQuestionStep(
                question_id=item.question.question_id,
                scene_id=item.scene.scene_id,
                family=item.question.family,
                expected_answer_type=item.question.expected_answer_type,
                prompt_text=item.question.prompt_text,
                solver_query=dict(item.question.solver_query or {}),
                ground_truth_answer=item.question.reference_answer,
                solver_prediction=solver_pred,
                matches_ground_truth=matched,
                validation_is_valid=qa_report.is_valid,
                validation_errors=[f"{err.code}: {err.message}" for err in qa_report.errors],
            )
        )

    matched_count = sum(1 for s in steps if s.matches_ground_truth)
    valid_count = sum(1 for s in steps if s.validation_is_valid)
    report = Demo3DReport(
        mode=cfg.mode,
        dataset_id=dataset_id,
        scene_id=scene.scene_id,
        scene_dimensions_m=(scene.width, scene.depth, scene.ceiling_height),
        entity_count=len(scene.entities),
        image_rel_path=image_rel_path,
        image_width_px=img_w,
        image_height_px=img_h,
        total_questions=len(steps),
        matched_count=matched_count,
        valid_count=valid_count,
        all_steps_matched_and_valid=(
            len(steps) > 0 and matched_count == len(steps) and valid_count == len(steps)
        ),
        steps=steps,
        artifacts={
            "report_json": "demo_3d_report.json",
            "summary_md": "demo_3d_summary.md",
            "image_png": image_rel_path,
        },
    )

    report_json_path = out_dir / "demo_3d_report.json"
    summary_md_path = out_dir / "demo_3d_summary.md"
    summary_md = _render_demo_summary_markdown(report)
    report_json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    summary_md_path.write_text(summary_md, encoding="utf-8")

    return Demo3DRunResult(
        output_dir=out_dir,
        report_json_path=report_json_path,
        summary_md_path=summary_md_path,
        image_path=image_abs_path,
        report=report,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python -m visual_task_factory.demo_3d``."""
    parser = argparse.ArgumentParser(
        description="Run the deterministic Visual Task Factory 3D end-to-end demo."
    )
    parser.add_argument(
        "--mode",
        choices=("fresh_scene", "existing_dataset"),
        default="fresh_scene",
    )
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=6)
    parser.add_argument("--output-dir", default=str(DEFAULT_3D_DEMO_OUTPUT_DIR))
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = Demo3DConfig(
        mode=args.mode,
        dataset_path=args.dataset_dir,
        scene_index=args.scene_index,
        max_questions=args.max_questions,
    )
    res = run_demo_3d(cfg, output_dir=args.output_dir)
    rep = res.report
    print(f"3D Demo complete | mode={rep.mode} | scene={rep.scene_id}")
    print(f"Rendered image : {res.image_path}")
    print(f"Report JSON    : {res.report_json_path}")
    print(f"Summary MD     : {res.summary_md_path}")
    print(
        f"Solver/QA check: matched={rep.matched_count}/{rep.total_questions}, "
        f"valid={rep.valid_count}/{rep.total_questions}"
    )
    for step in rep.steps:
        print(
            f"  - [{step.family.value}] {step.question_id}: {step.prompt_text} "
            f"=> gt={step.ground_truth_answer.value}, solver={step.solver_prediction.value}, "
            f"match={step.matches_ground_truth}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
