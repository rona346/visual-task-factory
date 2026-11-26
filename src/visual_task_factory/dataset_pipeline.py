"""Reproducible end-to-end dataset generation and export pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from visual_task_factory.ground_truth_solver import ContainmentMode, GroundTruthSolver
from visual_task_factory.models import (
    AnswerType,
    BenchmarkItem,
    QuestionFamily,
    ReferenceAnswer,
    SceneSpec,
    VariationConfig,
    VisualQuestion,
)
from visual_task_factory.question_generator import (
    QuestionGenerator,
    QuestionGeneratorConfig,
)
from visual_task_factory.renderer import RenderConfig, WarehouseSceneRenderer
from visual_task_factory.validator import (
    BatchValidationReport,
    BenchmarkValidator,
    ValidationReport,
)
from visual_task_factory.variation_engine import (
    GeneratedVariationItem,
    SceneSignature,
    VariationEngine,
    VariationPlanConfig,
)

_DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class DatasetPipelineConfig(BaseModel):
    """Configuration controlling end-to-end dataset generation and export."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(default="warehouse_spatial_v1", min_length=1)
    variation_plan: VariationPlanConfig = Field(default_factory=VariationPlanConfig)
    question_families: tuple[QuestionFamily, ...] = (
        QuestionFamily.CONSTRAINED_REACHABILITY,
        QuestionFamily.REGION_CONTAINMENT_COUNTING,
        QuestionFamily.SCALE_GROUNDED_CLEARANCE,
    )
    max_questions_per_family: int = Field(default=1, ge=1, le=10)
    containment_mode: ContainmentMode = ContainmentMode.STRICT_INSIDE
    clearance_variant: Literal["boolean", "numeric", "both"] = "both"
    min_reachability_margin_m: float = Field(default=0.5, ge=0.0)
    numeric_tolerance_m: float = Field(default=0.05, ge=0.0)
    include_candidate_choices: bool = True
    pixels_per_unit: int = Field(default=24, ge=8, le=96)
    supersample: Literal[1, 2, 3] = 2
    render_audit_overlays: bool = False
    export_scenes_jsonl: bool = True

    @field_validator("dataset_id")
    @classmethod
    def _validate_dataset_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or not _DATASET_ID_PATTERN.match(cleaned):
            raise ValueError(
                f"dataset_id must match {_DATASET_ID_PATTERN.pattern}; got {value!r}."
            )
        return cleaned

    @field_validator("question_families")
    @classmethod
    def _require_non_empty_families(
        cls, value: tuple[QuestionFamily, ...]
    ) -> tuple[QuestionFamily, ...]:
        if not value:
            raise ValueError("question_families must contain at least one QuestionFamily.")
        return value


class DatasetRecord(BaseModel):
    """Single validated benchmark record exported to dataset.jsonl."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str
    record_id: str
    scene_id: str
    question_id: str
    image_rel_path: str
    image_width_px: int = Field(..., gt=0)
    image_height_px: int = Field(..., gt=0)
    prompt_text: str
    family: QuestionFamily
    template_id: str
    expected_answer_type: AnswerType
    candidate_choices: list[str] | None = None
    reference_answer: ReferenceAnswer
    solver_query: dict[str, Any]
    variation: VariationConfig
    scene_signature: SceneSignature
    validation: ValidationReport


class RejectedDatasetRecord(BaseModel):
    """Invalid or failed benchmark item excluded from dataset.jsonl."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str
    question_id: str
    image_rel_path: str
    question: VisualQuestion
    validation: ValidationReport


class DatasetManifest(BaseModel):
    """Machine-readable manifest describing dataset generation and validation metrics."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str
    generation_seed: int
    config: dict[str, Any]
    requested_scene_count: int
    generated_scene_count: int
    requested_question_count: int
    generated_question_count: int
    valid_item_count: int
    invalid_item_count: int
    warning_count: int
    duplicate_question_ids: list[str]
    duplicate_prompts: list[str]
    unique_structural_signatures: int
    duplicate_layout_rejections: int
    unsolvable_layout_rejections: int
    render_failures: list[str]
    validation_summary: dict[str, Any]
    artifacts: dict[str, str]
    logical_dataset_sha256: str
    image_sha256_by_scene: dict[str, str]


class DatasetPipelineResult(BaseModel):
    """In-memory and on-disk result returned by DatasetPipeline.build()."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    manifest: DatasetManifest
    valid_records: list[DatasetRecord]
    rejected_records: list[RejectedDatasetRecord]
    batch_validation: BatchValidationReport


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_output_dir(output_dir: Path | str) -> Path:
    if isinstance(output_dir, str):
        if not output_dir.strip():
            raise ValueError("output_dir must not be empty.")
        resolved = Path(output_dir)
    elif isinstance(output_dir, Path):
        if not str(output_dir).strip():
            raise ValueError("output_dir must not be empty.")
        resolved = output_dir
    else:
        raise TypeError(f"output_dir must be Path or str, got {type(output_dir).__name__}.")

    if resolved.exists() and not resolved.is_dir():
        raise FileExistsError(f"output_dir points to an existing non-directory file: '{resolved}'.")
    return resolved


class DatasetPipeline:
    """Orchestrates variation, rendering, QA generation, validation, and export."""

    def __init__(
        self,
        variation_engine: VariationEngine | None = None,
        solver: GroundTruthSolver | None = None,
        validator: BenchmarkValidator | None = None,
        question_generator: QuestionGenerator | None = None,
    ) -> None:
        self.solver = solver or GroundTruthSolver()
        self.variation_engine = variation_engine or VariationEngine(solver=self.solver)
        self.validator = validator or BenchmarkValidator(solver=self.solver)
        self._custom_question_generator = question_generator

    def build(
        self,
        config: DatasetPipelineConfig,
        output_dir: Path | str,
    ) -> DatasetPipelineResult:
        """Execute the full benchmark pipeline and export dataset artifacts to output_dir."""
        if not isinstance(config, DatasetPipelineConfig):
            raise TypeError(
                f"Expected DatasetPipelineConfig instance, got {type(config).__name__}."
            )
        root_dir = _resolve_output_dir(output_dir)
        images_dir = root_dir / "images"
        root_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate meaningfully varied scenes using VariationEngine.
        var_batch = self.variation_engine.generate_batch(config.variation_plan)

        # 2. Configure renderer and deterministic question generator.
        renderer = WarehouseSceneRenderer(
            config=RenderConfig(
                pixels_per_unit=config.pixels_per_unit,
                supersample=config.supersample,
                audit_overlay=False,
            )
        )
        audit_renderer = (
            WarehouseSceneRenderer(
                config=RenderConfig(
                    pixels_per_unit=config.pixels_per_unit,
                    supersample=config.supersample,
                    audit_overlay=True,
                )
            )
            if config.render_audit_overlays
            else None
        )
        qgen = self._custom_question_generator or QuestionGenerator(
            config=QuestionGeneratorConfig(
                families=config.question_families,
                max_questions_per_family=config.max_questions_per_family,
                containment_mode=config.containment_mode,
                clearance_variant=config.clearance_variant,
                min_reachability_margin_m=config.min_reachability_margin_m,
                numeric_tolerance_m=config.numeric_tolerance_m,
                include_candidate_choices=config.include_candidate_choices,
            ),
            solver=self.solver,
        )

        # 3. Render images and generate questions per scene.
        render_failures: list[str] = []
        image_meta_by_scene: dict[str, tuple[str, int, int]] = {}
        image_sha256_by_scene: dict[str, str] = {}
        var_item_by_scene: dict[str, GeneratedVariationItem] = {}
        candidate_items: list[BenchmarkItem] = []

        for var_item in var_batch.items:
            scene: SceneSpec = var_item.scene
            var_item_by_scene[scene.scene_id] = var_item
            rel_img = f"images/{scene.scene_id}.png"
            abs_img = root_dir / rel_img

            try:
                render_res = renderer.render(scene, abs_img)
                if not abs_img.is_file() or abs_img.stat().st_size == 0:
                    raise OSError(f"Rendered image missing or empty: '{abs_img}'.")
                if audit_renderer is not None:
                    audit_renderer.render(scene, root_dir / "audit_images" / f"{scene.scene_id}_audit.png")
            except Exception as exc:
                render_failures.append(f"{scene.scene_id}: {exc}")
                continue

            image_meta_by_scene[scene.scene_id] = (
                rel_img,
                render_res.width_px,
                render_res.height_px,
            )
            image_sha256_by_scene[scene.scene_id] = _sha256_bytes(abs_img.read_bytes())

            scene_items = qgen.generate(scene)
            candidate_items.extend(scene_items)

        # 4. Run independent batch validation (detects item errors + batch duplicates).
        batch_report = self.validator.validate_batch(candidate_items)

        valid_records: list[DatasetRecord] = []
        rejected_records: list[RejectedDatasetRecord] = []

        for item, report in zip(candidate_items, batch_report.reports):
            scene_id = report.scene_id
            img_info = image_meta_by_scene.get(scene_id)
            var_item = var_item_by_scene.get(scene_id)

            if img_info is None or var_item is None:
                # Defensive guard if scene rendering or metadata lookup failed.
                rejected_records.append(
                    RejectedDatasetRecord(
                        scene_id=scene_id,
                        question_id=report.question_id,
                        image_rel_path=f"images/{scene_id}.png",
                        question=item.question,
                        validation=report,
                    )
                )
                continue

            rel_img, img_w, img_h = img_info
            if not report.is_valid or item.question.solver_query is None:
                rejected_records.append(
                    RejectedDatasetRecord(
                        scene_id=scene_id,
                        question_id=report.question_id,
                        image_rel_path=rel_img,
                        question=item.question,
                        validation=report,
                    )
                )
                continue

            valid_records.append(
                DatasetRecord(
                    dataset_id=config.dataset_id,
                    record_id=f"{config.dataset_id}__{item.question.question_id}",
                    scene_id=scene_id,
                    question_id=item.question.question_id,
                    image_rel_path=rel_img,
                    image_width_px=img_w,
                    image_height_px=img_h,
                    prompt_text=item.question.prompt_text,
                    family=item.question.family,
                    template_id=item.question.template_id,
                    expected_answer_type=item.question.expected_answer_type,
                    candidate_choices=item.question.candidate_choices,
                    reference_answer=item.question.reference_answer,
                    solver_query=item.question.solver_query,
                    variation=var_item.variation,
                    scene_signature=var_item.signature,
                    validation=report,
                )
            )

        # 5. Export dataset.jsonl, rejected_items.jsonl, optional scenes.jsonl, and manifest.json.
        dataset_path = root_dir / "dataset.jsonl"
        rejected_path = root_dir / "rejected_items.jsonl"
        scenes_path = root_dir / "scenes.jsonl"
        manifest_path = root_dir / "manifest.json"

        dataset_lines = [rec.model_dump_json() for rec in valid_records]
        dataset_content = ("\n".join(dataset_lines) + "\n") if dataset_lines else ""
        dataset_path.write_text(dataset_content, encoding="utf-8")

        rejected_lines = [rec.model_dump_json() for rec in rejected_records]
        rejected_content = ("\n".join(rejected_lines) + "\n") if rejected_lines else ""
        rejected_path.write_text(rejected_content, encoding="utf-8")

        artifacts: dict[str, str] = {
            "dataset_jsonl": "dataset.jsonl",
            "rejected_jsonl": "rejected_items.jsonl",
            "images_dir": "images",
            "manifest_json": "manifest.json",
        }

        if config.export_scenes_jsonl:
            scene_lines = [
                json.dumps(
                    {
                        "scene_id": it.scene.scene_id,
                        "image_rel_path": image_meta_by_scene[it.scene.scene_id][0]
                        if it.scene.scene_id in image_meta_by_scene
                        else None,
                        "variation": it.variation.model_dump(mode="json"),
                        "signature": it.signature.model_dump(mode="json"),
                        "scene": it.scene.model_dump(mode="json"),
                    },
                    sort_keys=True,
                )
                for it in var_batch.items
            ]
            scenes_path.write_text(
                ("\n".join(scene_lines) + "\n") if scene_lines else "",
                encoding="utf-8",
            )
            artifacts["scenes_jsonl"] = "scenes.jsonl"

        if config.render_audit_overlays:
            artifacts["audit_images_dir"] = "audit_images"

        # Compute deterministic logical dataset SHA-256 across valid records.
        logical_sha256 = _sha256_bytes(dataset_content.encode("utf-8"))

        family_counts: dict[str, int] = {}
        answer_type_counts: dict[str, int] = {}
        error_code_counts: dict[str, int] = {}
        warning_code_counts: dict[str, int] = {}

        for rec in valid_records:
            family_counts[rec.family.value] = family_counts.get(rec.family.value, 0) + 1
            answer_type_counts[rec.expected_answer_type.value] = (
                answer_type_counts.get(rec.expected_answer_type.value, 0) + 1
            )
        for rep in batch_report.reports:
            for err in rep.errors:
                error_code_counts[err.code] = error_code_counts.get(err.code, 0) + 1
            for warn in rep.warnings:
                warning_code_counts[warn.code] = warning_code_counts.get(warn.code, 0) + 1

        requested_questions = (
            config.variation_plan.count
            * len(config.question_families)
            * config.max_questions_per_family
        )

        manifest = DatasetManifest(
            dataset_id=config.dataset_id,
            generation_seed=config.variation_plan.base_seed,
            config=config.model_dump(mode="json"),
            requested_scene_count=config.variation_plan.count,
            generated_scene_count=len(var_batch.items),
            requested_question_count=requested_questions,
            generated_question_count=len(candidate_items),
            valid_item_count=len(valid_records),
            invalid_item_count=len(rejected_records),
            warning_count=batch_report.warning_count,
            duplicate_question_ids=batch_report.duplicate_question_ids,
            duplicate_prompts=batch_report.duplicate_prompts,
            unique_structural_signatures=var_batch.unique_structural_signatures,
            duplicate_layout_rejections=var_batch.duplicate_rejections,
            unsolvable_layout_rejections=var_batch.unsolvable_rejections,
            render_failures=render_failures,
            validation_summary={
                "family_counts": dict(sorted(family_counts.items())),
                "answer_type_counts": dict(sorted(answer_type_counts.items())),
                "topology_coverage": var_batch.topology_coverage,
                "difficulty_coverage": var_batch.difficulty_coverage,
                "orientation_coverage": {
                    str(k): v for k, v in var_batch.orientation_coverage.items()
                },
                "theme_coverage": var_batch.theme_coverage,
                "error_code_counts": dict(sorted(error_code_counts.items())),
                "warning_code_counts": dict(sorted(warning_code_counts.items())),
            },
            artifacts=artifacts,
            logical_dataset_sha256=logical_sha256,
            image_sha256_by_scene=dict(sorted(image_sha256_by_scene.items())),
        )
        manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")

        return DatasetPipelineResult(
            output_dir=root_dir,
            manifest=manifest,
            valid_records=valid_records,
            rejected_records=rejected_records,
            batch_validation=batch_report,
        )


def build_dataset(
    config: DatasetPipelineConfig,
    output_dir: Path | str,
) -> DatasetPipelineResult:
    """Convenience entry point to build and export a validated visual benchmark dataset."""
    return DatasetPipeline().build(config=config, output_dir=output_dir)
