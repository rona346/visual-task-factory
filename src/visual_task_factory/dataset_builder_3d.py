"""Isolated reproducible 3D dataset builder, validator, and loader (Phase 5)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Sequence

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from visual_task_factory.models.scene import EntityType, Point2D, Rect2D, ScaleBarSpec, ZoneType
from visual_task_factory.models.scene3d import (
    Box3D,
    CameraPose3D,
    Doorway3D,
    ExitMarker3D,
    Point3D,
    SceneEntity3D,
    SceneSpec3D,
    WallSegment3D,
    Zone3D,
    extrude_scene_2d_to_3d,
)
from visual_task_factory.models.variation import VariationConfig, VisualTheme
from visual_task_factory.qa_3d import (
    AnswerType3D,
    BenchmarkItem3D,
    GroundTruthQAValidator3D,
    QuestionFamily3D,
    ReferenceAnswer3D,
    ValidationReport3D,
    VisualQuestion3D,
)
from visual_task_factory.question_generator_3d import (
    QuestionGenerator3D,
    QuestionGenerator3DConfig,
)
from visual_task_factory.renderer_3d import Render3DConfig, Warehouse3DRenderer
from visual_task_factory.scene_generator import generate_scene

DEFAULT_3D_DATASET_DIR = Path("data/3d")
_DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_DRIVE_OR_BACKSLASH_PATTERN = re.compile(r"(:|\\)")


class Dataset3DError(ValueError):
    """Base error raised during 3D dataset construction, validation, or loading."""


class InvalidQAItem3DError(Dataset3DError):
    """Raised when one or more 3D benchmark items fail QA validation."""


class MissingImage3DError(Dataset3DError):
    """Raised when a referenced rendered scene PNG is missing, invalid, or unsafe."""


class DuplicateID3DError(Dataset3DError):
    """Raised when duplicate scene, question, or record IDs are encountered."""


class Dataset3DConfig(BaseModel):
    """Configuration for deterministic 3D dataset generation and export."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(default="warehouse_spatial_3d_v1", min_length=1)
    question_families: tuple[QuestionFamily3D, ...] = (
        QuestionFamily3D.OBJECT_OVERLAP_3D,
        QuestionFamily3D.VERTICAL_RELATION_3D,
        QuestionFamily3D.VERTICAL_CLEARANCE_3D,
        QuestionFamily3D.CAMERA_VISIBILITY_3D,
    )
    max_questions_per_family: int = Field(default=2, ge=1, le=10)
    clearance_variant: Literal["numeric", "boolean", "both"] = "both"
    numeric_tolerance_m: float = Field(default=0.05, ge=0.0)
    include_ceiling_clearance: bool = True
    seed_offset: int = 0
    image_width_px: int = Field(default=800, ge=240, le=4096)
    image_height_px: int = Field(default=540, ge=180, le=4096)
    supersample: Literal[1, 2, 3] = 2
    show_labels: bool = True
    show_grid: bool = True
    show_zones: bool = True
    cutaway_front_walls: bool = False

    @field_validator("dataset_id")
    @classmethod
    def _validate_dataset_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or not _DATASET_ID_PATTERN.match(cleaned):
            raise ValueError(f"dataset_id {value!r} must match {_DATASET_ID_PATTERN.pattern}.")
        return cleaned


def _validate_safe_rel_image_path(rel_path: str) -> str:
    cleaned = rel_path.strip().replace("\\", "/")
    if not cleaned or cleaned.startswith("/") or ":" in cleaned:
        raise ValueError(f"image_rel_path must be relative without drive letters: {rel_path!r}")
    parts = PurePosixPath(cleaned).parts
    if any(part in ("..", "") for part in parts):
        raise ValueError(f"image_rel_path must not contain traversal segments: {rel_path!r}")
    if not cleaned.lower().endswith(".png"):
        raise ValueError(f"image_rel_path must reference a .png image: {rel_path!r}")
    return cleaned


class Dataset3DRecord(BaseModel):
    """Single validated 3D benchmark record serialized inside dataset.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str
    record_id: str
    scene_id: str
    question_id: str
    image_rel_path: str
    image_width_px: int = Field(..., gt=0)
    image_height_px: int = Field(..., gt=0)
    image_sha256: str = Field(..., min_length=64, max_length=64)
    prompt_text: str
    family: QuestionFamily3D
    template_id: str
    expected_answer_type: AnswerType3D
    reference_answer: ReferenceAnswer3D
    solver_query: dict[str, Any]
    variation: VariationConfig
    validation: ValidationReport3D

    @field_validator("image_rel_path")
    @classmethod
    def _check_rel_image_path(cls, value: str) -> str:
        return _validate_safe_rel_image_path(value)


class Dataset3DBundle(BaseModel):
    """Complete self-contained 3D dataset archive written to dataset.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str
    config: Dataset3DConfig
    scene_count: int = Field(..., ge=1)
    record_count: int = Field(..., ge=1)
    family_counts: dict[str, int]
    answer_type_counts: dict[str, int]
    logical_dataset_sha256: str = Field(..., min_length=64, max_length=64)
    scenes: dict[str, SceneSpec3D]
    records: list[Dataset3DRecord]

    @model_validator(mode="after")
    def _validate_bundle_consistency(self) -> Dataset3DBundle:
        if self.scene_count != len(self.scenes):
            raise ValueError(
                f"scene_count ({self.scene_count}) does not match len(scenes) ({len(self.scenes)})."
            )
        if self.record_count != len(self.records):
            raise ValueError(
                f"record_count ({self.record_count}) does not match len(records) ({len(self.records)})."
            )
        seen_record_ids: set[str] = set()
        seen_question_ids: set[str] = set()
        for rec in self.records:
            if rec.scene_id not in self.scenes:
                raise ValueError(f"Record {rec.record_id!r} references missing scene {rec.scene_id!r}.")
            if rec.record_id in seen_record_ids:
                raise ValueError(f"Duplicate record_id in dataset bundle: {rec.record_id!r}.")
            if rec.question_id in seen_question_ids:
                raise ValueError(f"Duplicate question_id in dataset bundle: {rec.question_id!r}.")
            seen_record_ids.add(rec.record_id)
            seen_question_ids.add(rec.question_id)
        return self


class Dataset3DBuildResult(BaseModel):
    """In-memory handle returned after building a 3D dataset directory."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    output_dir: Path
    dataset_json_path: Path
    images_dir: Path
    bundle: Dataset3DBundle


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_png_file(image_path: Path) -> tuple[int, int, str]:
    if not image_path.is_file() or image_path.stat().st_size == 0:
        raise MissingImage3DError(f"Referenced 3D PNG image does not exist or is empty: '{image_path}'.")
    try:
        raw_bytes = image_path.read_bytes()
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img2:
            if img2.format != "PNG":
                raise MissingImage3DError(f"Expected PNG image at '{image_path}', got {img2.format!r}.")
            width_px, height_px = img2.size
    except MissingImage3DError:
        raise
    except Exception as exc:
        raise MissingImage3DError(f"Failed to load valid PNG image at '{image_path}': {exc}") from exc
    return width_px, height_px, _sha256_bytes(raw_bytes)


def build_default_3d_scenes() -> list[SceneSpec3D]:
    """Create a deterministic, multi-scene 3D starter suite combining native & extruded scenes."""
    native_stacked = SceneSpec3D(
        scene_id="wh3d_stacked_01",
        width=20.0,
        depth=16.0,
        ceiling_height=5.0,
        scale_bar=ScaleBarSpec(world_units=2.0, display_meters=2.0),
        variation=VariationConfig(seed=101, topology_id="stacked_bay"),
        camera=CameraPose3D(
            look_at=Point3D(x=10.0, y=8.0, z=1.0),
            distance=26.0,
            azimuth_deg=45.0,
            elevation_deg=35.0,
            fov_deg=48.0,
        ),
        walls=[
            WallSegment3D(
                id="wall_north",
                start=Point2D(x=0.0, y=0.0),
                end=Point2D(x=20.0, y=0.0),
                height=4.5,
            ),
            WallSegment3D(
                id="wall_west",
                start=Point2D(x=0.0, y=0.0),
                end=Point2D(x=0.0, y=16.0),
                height=4.5,
            ),
            WallSegment3D(
                id="wall_partition",
                start=Point2D(x=8.0, y=1.0),
                end=Point2D(x=8.0, y=11.0),
                height=3.6,
            ),
        ],
        doors=[
            Doorway3D(
                id="door_mid",
                position=Point3D(x=8.0, y=6.0, z=0.0),
                width=2.0,
                height=2.4,
                orientation="vertical",
                is_blocked=False,
            )
        ],
        zones=[
            Zone3D(
                id="zone_storage",
                label="Zone A",
                zone_type=ZoneType.STORAGE,
                bounds=Rect2D(x=1.5, y=1.5, width=5.5, height=5.5),
            ),
            Zone3D(
                id="zone_hazard",
                label="Hazard B",
                zone_type=ZoneType.HAZARD,
                bounds=Rect2D(x=13.0, y=9.0, width=5.0, height=5.0),
            ),
        ],
        exits=[
            ExitMarker3D(
                id="exit_east",
                label="Exit E",
                position=Point3D(x=20.0, y=8.0, z=0.0),
            )
        ],
        entities=[
            SceneEntity3D(
                id="crate_base",
                label="Base Crate",
                entity_type=EntityType.CRATE,
                bounds=Box3D(x=2.5, y=2.5, z=0.0, width=2.4, depth=2.4, height=1.2),
            ),
            SceneEntity3D(
                id="crate_top",
                label="Top Crate",
                entity_type=EntityType.CRATE,
                bounds=Box3D(x=2.9, y=2.9, z=1.2, width=1.6, depth=1.6, height=0.9),
                supported_by="crate_base",
            ),
            SceneEntity3D(
                id="forklift_1",
                label="Forklift 1",
                entity_type=EntityType.FORKLIFT,
                bounds=Box3D(x=9.5, y=9.5, z=0.0, width=3.0, depth=2.0, height=2.2),
            ),
            SceneEntity3D(
                id="crate_hidden",
                label="Hidden Crate",
                entity_type=EntityType.CRATE,
                bounds=Box3D(x=10.5, y=1.5, z=0.0, width=1.6, depth=1.6, height=1.0),
            ),
            SceneEntity3D(
                id="rack_occluder",
                label="Front Rack",
                entity_type=EntityType.PALLET_RACK,
                bounds=Box3D(x=12.5, y=3.5, z=0.0, width=4.8, depth=4.4, height=4.4),
            ),
        ],
    )

    extruded_blueprint = extrude_scene_2d_to_3d(
        generate_scene(
            VariationConfig(
                seed=202,
                topology_id="two_zone_corridor",
                theme=VisualTheme.BLUEPRINT,
            )
        ),
        ceiling_height=4.5,
    )

    return [native_stacked, extruded_blueprint]


class DatasetBuilder3D:
    """Builds, validates, exports, and reloads isolated 3D benchmark datasets."""

    def __init__(self, validator: GroundTruthQAValidator3D | None = None) -> None:
        self.validator = validator or GroundTruthQAValidator3D()

    def build_from_scenes(
        self,
        scenes: Sequence[SceneSpec3D],
        output_dir: Path | str = DEFAULT_3D_DATASET_DIR,
        *,
        config: Dataset3DConfig | None = None,
    ) -> Dataset3DBuildResult:
        """Render scenes, generate grounded 3D QA items, validate, and export dataset.json."""
        cfg = config or Dataset3DConfig()
        if not scenes:
            raise Dataset3DError("scenes must contain at least one SceneSpec3D.")

        root_dir = Path(output_dir)
        images_dir = root_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        seen_scenes: set[str] = set()
        for scene in scenes:
            if not isinstance(scene, SceneSpec3D):
                raise TypeError(f"Expected SceneSpec3D instance, got {type(scene).__name__}.")
            if scene.scene_id in seen_scenes:
                raise DuplicateID3DError(f"Duplicate scene_id provided: {scene.scene_id!r}.")
            seen_scenes.add(scene.scene_id)

        renderer = Warehouse3DRenderer(
            Render3DConfig(
                width_px=cfg.image_width_px,
                height_px=cfg.image_height_px,
                supersample=cfg.supersample,
                show_labels=cfg.show_labels,
                show_grid=cfg.show_grid,
                show_zones=cfg.show_zones,
                cutaway_front_walls=cfg.cutaway_front_walls,
            )
        )
        qgen = QuestionGenerator3D(
            QuestionGenerator3DConfig(
                families=cfg.question_families,
                max_questions_per_family=cfg.max_questions_per_family,
                clearance_variant=cfg.clearance_variant,
                numeric_tolerance_m=cfg.numeric_tolerance_m,
                include_ceiling_clearance=cfg.include_ceiling_clearance,
                seed_offset=cfg.seed_offset,
            )
        )

        image_rel_by_scene: dict[str, str] = {}
        all_items: list[BenchmarkItem3D] = []

        for scene in scenes:
            rel_path = f"images/{scene.scene_id}.png"
            abs_path = root_dir / rel_path
            renderer.render(scene, output_path=abs_path)
            image_rel_by_scene[scene.scene_id] = rel_path
            all_items.extend(qgen.generate(scene))

        return self.build_from_items(
            all_items,
            output_dir=root_dir,
            image_rel_paths_by_scene=image_rel_by_scene,
            config=cfg,
        )

    def build_from_items(
        self,
        items: Sequence[BenchmarkItem3D],
        output_dir: Path | str = DEFAULT_3D_DATASET_DIR,
        *,
        image_rel_paths_by_scene: dict[str, str] | None = None,
        config: Dataset3DConfig | None = None,
    ) -> Dataset3DBuildResult:
        """Validate provided BenchmarkItem3D records + rendered PNGs and write dataset.json."""
        cfg = config or Dataset3DConfig()
        if not items:
            raise Dataset3DError("items must contain at least one validated BenchmarkItem3D.")

        root_dir = Path(output_dir)
        images_dir = root_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Verify distinct scenes per scene_id and unique question IDs upfront.
        scenes_by_id: dict[str, SceneSpec3D] = {}
        seen_qids: set[str] = set()
        for item in items:
            if not isinstance(item, BenchmarkItem3D):
                raise TypeError(f"Expected BenchmarkItem3D instance, got {type(item).__name__}.")
            existing_scene = scenes_by_id.get(item.scene.scene_id)
            if existing_scene is not None and existing_scene != item.scene:
                raise DuplicateID3DError(
                    f"Conflicting SceneSpec3D payloads share scene_id={item.scene.scene_id!r}."
                )
            scenes_by_id[item.scene.scene_id] = item.scene

            if item.question.question_id in seen_qids:
                raise DuplicateID3DError(
                    f"Duplicate question_id detected: {item.question.question_id!r}."
                )
            seen_qids.add(item.question.question_id)

        batch_report = self.validator.validate_batch(items)
        if batch_report.invalid_items > 0:
            failures = [
                f"{rep.question_id}: {[f'{err.code}: {err.message}' for err in rep.errors]}"
                for rep in batch_report.reports
                if not rep.is_valid
            ]
            raise InvalidQAItem3DError(
                f"3D QA validation failed for {batch_report.invalid_items} item(s): {failures}"
            )

        image_meta_by_scene: dict[str, tuple[str, int, int, str]] = {}
        for scene_id in sorted(scenes_by_id):
            raw_rel = (
                image_rel_paths_by_scene[scene_id]
                if image_rel_paths_by_scene is not None and scene_id in image_rel_paths_by_scene
                else f"images/{scene_id}.png"
            )
            rel_clean = _validate_safe_rel_image_path(raw_rel)
            img_path = (root_dir / rel_clean).resolve()
            if not img_path.is_relative_to(root_dir.resolve()):
                raise MissingImage3DError(
                    f"Image path {rel_clean!r} escapes dataset output root '{root_dir}'."
                )
            w_px, h_px, sha = _verify_png_file(img_path)
            image_meta_by_scene[scene_id] = (rel_clean, w_px, h_px, sha)

        records: list[Dataset3DRecord] = []
        family_counts: dict[str, int] = {}
        answer_type_counts: dict[str, int] = {}

        for item, report in zip(items, batch_report.reports):
            if item.question.solver_query is None:
                raise InvalidQAItem3DError(
                    f"Question {item.question.question_id!r} missing solver_query."
                )
            rel_img, w_px, h_px, img_sha = image_meta_by_scene[item.scene.scene_id]
            rec = Dataset3DRecord(
                dataset_id=cfg.dataset_id,
                record_id=f"{cfg.dataset_id}__{item.question.question_id}",
                scene_id=item.scene.scene_id,
                question_id=item.question.question_id,
                image_rel_path=rel_img,
                image_width_px=w_px,
                image_height_px=h_px,
                image_sha256=img_sha,
                prompt_text=item.question.prompt_text,
                family=item.question.family,
                template_id=item.question.template_id,
                expected_answer_type=item.question.expected_answer_type,
                reference_answer=item.question.reference_answer,
                solver_query=item.question.solver_query,
                variation=item.scene.variation,
                validation=report,
            )
            records.append(rec)
            family_counts[rec.family.value] = family_counts.get(rec.family.value, 0) + 1
            answer_type_counts[rec.expected_answer_type.value] = (
                answer_type_counts.get(rec.expected_answer_type.value, 0) + 1
            )

        canonical_records_json = json.dumps(
            [r.model_dump(mode="json") for r in records],
            sort_keys=True,
            separators=(",", ":"),
        )
        logical_sha = _sha256_bytes(canonical_records_json.encode("utf-8"))

        bundle = Dataset3DBundle(
            dataset_id=cfg.dataset_id,
            config=cfg,
            scene_count=len(scenes_by_id),
            record_count=len(records),
            family_counts=dict(sorted(family_counts.items())),
            answer_type_counts=dict(sorted(answer_type_counts.items())),
            logical_dataset_sha256=logical_sha,
            scenes={k: scenes_by_id[k] for k in sorted(scenes_by_id)},
            records=records,
        )

        dataset_json_path = root_dir / "dataset.json"
        dataset_json_path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")

        return Dataset3DBuildResult(
            output_dir=root_dir,
            dataset_json_path=dataset_json_path,
            images_dir=images_dir,
            bundle=bundle,
        )


def load_and_validate_dataset_3d(
    dataset_path_or_dir: Path | str,
    *,
    verify_images: bool = True,
    revalidate_qa: bool = True,
    validator: GroundTruthQAValidator3D | None = None,
) -> Dataset3DBundle:
    """Load dataset.json from disk and validate schema, image links, and ground-truth QA."""
    target = Path(dataset_path_or_dir)
    if target.is_dir():
        root_dir = target
        json_path = target / "dataset.json"
    else:
        root_dir = target.parent
        json_path = target

    if not json_path.is_file():
        raise Dataset3DError(f"3D dataset file not found: '{json_path}'.")

    try:
        bundle = Dataset3DBundle.model_validate_json(json_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise Dataset3DError(f"Invalid 3D dataset schema in '{json_path}': {exc}") from exc

    if verify_images:
        checked_images: dict[str, tuple[int, int, str]] = {}
        for rec in bundle.records:
            if rec.image_rel_path not in checked_images:
                img_path = (root_dir / rec.image_rel_path).resolve()
                if not img_path.is_relative_to(root_dir.resolve()):
                    raise MissingImage3DError(
                        f"Relative image path {rec.image_rel_path!r} escapes dataset root '{root_dir}'."
                    )
                checked_images[rec.image_rel_path] = _verify_png_file(img_path)
            w_px, h_px, sha = checked_images[rec.image_rel_path]
            if (w_px, h_px) != (rec.image_width_px, rec.image_height_px):
                raise MissingImage3DError(
                    f"Image dimensions {(w_px, h_px)} do not match record ({rec.image_width_px}, {rec.image_height_px})."
                )
            if sha != rec.image_sha256:
                raise MissingImage3DError(
                    f"Image SHA-256 mismatch for {rec.image_rel_path!r}."
                )

    if revalidate_qa:
        qa_validator = validator or GroundTruthQAValidator3D()
        items = [
            BenchmarkItem3D(
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
            for rec in bundle.records
        ]
        batch_report = qa_validator.validate_batch(items)
        if batch_report.invalid_items > 0:
            raise InvalidQAItem3DError(
                f"Reloaded 3D dataset failed QA revalidation on {batch_report.invalid_items} item(s)."
            )

    return bundle
