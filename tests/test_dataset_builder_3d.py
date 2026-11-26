"""Tests for the isolated reproducible 3D dataset builder (Phase 5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from visual_task_factory.dataset_builder_3d import (
    Dataset3DBundle,
    Dataset3DConfig,
    Dataset3DError,
    DatasetBuilder3D,
    DuplicateID3DError,
    InvalidQAItem3DError,
    MissingImage3DError,
    build_default_3d_scenes,
    load_and_validate_dataset_3d,
)
from visual_task_factory.dataset_pipeline import (
    DatasetPipelineConfig,
    DatasetRecord,
    build_dataset,
)
from visual_task_factory.qa_3d import BenchmarkItem3D
from visual_task_factory.question_generator_3d import QuestionGenerator3D
from visual_task_factory.renderer_3d import Render3DConfig, render_scene_3d
from visual_task_factory.variation_engine import VariationPlanConfig


def test_build_3d_dataset_end_to_end_structure_and_roundtrip(tmp_path: Path) -> None:
    out_dir = tmp_path / "data" / "3d"
    scenes = build_default_3d_scenes()
    cfg = Dataset3DConfig(
        dataset_id="wh3d_smoke",
        image_width_px=640,
        image_height_px=420,
        supersample=1,
    )

    result = DatasetBuilder3D().build_from_scenes(scenes, output_dir=out_dir, config=cfg)

    assert result.dataset_json_path == out_dir / "dataset.json"
    assert result.images_dir == out_dir / "images"
    assert result.dataset_json_path.is_file()
    assert result.bundle.scene_count == 2
    assert result.bundle.record_count >= 6

    raw_text = result.dataset_json_path.read_text(encoding="utf-8")
    # Ensure zero absolute local machine paths leak into dataset.json.
    assert str(tmp_path) not in raw_text
    assert ":\\" not in raw_text

    parsed_json = json.loads(raw_text)
    assert parsed_json["dataset_id"] == "wh3d_smoke"
    for rec in result.bundle.records:
        assert not Path(rec.image_rel_path).is_absolute()
        assert rec.image_rel_path.startswith("images/")
        img_file = out_dir / rec.image_rel_path
        assert img_file.is_file()
        with Image.open(img_file) as img:
            assert img.format == "PNG"
            assert img.size == (640, 420)

    loaded = load_and_validate_dataset_3d(out_dir)
    assert loaded == result.bundle


def test_3d_dataset_determinism_and_stable_ids(tmp_path: Path) -> None:
    scenes = build_default_3d_scenes()
    cfg = Dataset3DConfig(dataset_id="wh3d_repro", supersample=1, image_width_px=480, image_height_px=360)

    dir_a = tmp_path / "run_a" / "3d"
    dir_b = tmp_path / "run_b" / "3d"

    res_a = DatasetBuilder3D().build_from_scenes(scenes, output_dir=dir_a, config=cfg)
    res_b = DatasetBuilder3D().build_from_scenes(scenes, output_dir=dir_b, config=cfg)

    assert res_a.bundle.logical_dataset_sha256 == res_b.bundle.logical_dataset_sha256
    assert (dir_a / "dataset.json").read_text(encoding="utf-8") == (
        dir_b / "dataset.json"
    ).read_text(encoding="utf-8")
    for scene in scenes:
        assert (dir_a / "images" / f"{scene.scene_id}.png").read_bytes() == (
            dir_b / "images" / f"{scene.scene_id}.png"
        ).read_bytes()


def test_rejects_invalid_qa_missing_images_and_duplicate_ids(tmp_path: Path) -> None:
    scene = build_default_3d_scenes()[0]
    items = QuestionGenerator3D().generate(scene)
    assert len(items) >= 2

    out_dir = tmp_path / "bad_cases"
    img_path = out_dir / "images" / f"{scene.scene_id}.png"
    render_scene_3d(scene, output_path=img_path, config=Render3DConfig(supersample=1))

    builder = DatasetBuilder3D()

    # 1. Invalid QA item (corrupted reference answer) rejected.
    bad_val = not bool(items[0].question.reference_answer.value)
    corrupted_item = BenchmarkItem3D(
        scene=scene,
        question=items[0].question.model_copy(
            update={
                "reference_answer": items[0].question.reference_answer.model_copy(
                    update={"value": bad_val}
                )
            }
        ),
    )
    with pytest.raises(InvalidQAItem3DError):
        builder.build_from_items([corrupted_item], output_dir=out_dir)

    # 2. Duplicate question_id rejected.
    with pytest.raises(DuplicateID3DError):
        builder.build_from_items([items[0], items[0]], output_dir=out_dir)

    # 3. Duplicate scene_id across build_from_scenes rejected.
    with pytest.raises(DuplicateID3DError):
        builder.build_from_scenes([scene, scene], output_dir=out_dir)

    # 4. Missing image rejected cleanly during build and load.
    missing_dir = tmp_path / "missing_img_dir"
    with pytest.raises(MissingImage3DError):
        builder.build_from_items([items[0]], output_dir=missing_dir)

    # Build a valid bundle, delete its PNG, and verify load_and_validate_dataset_3d rejects it.
    valid_dir = tmp_path / "delete_png_check"
    builder.build_from_scenes([scene], output_dir=valid_dir, config=Dataset3DConfig(supersample=1))
    (valid_dir / "images" / f"{scene.scene_id}.png").unlink()
    with pytest.raises(MissingImage3DError):
        load_and_validate_dataset_3d(valid_dir)

    # 5. Missing dataset.json raises clear Dataset3DError.
    with pytest.raises(Dataset3DError):
        load_and_validate_dataset_3d(tmp_path / "does_not_exist")


def test_v1_dataset_pipeline_remains_independent_and_compatible(tmp_path: Path) -> None:
    v1_dir = tmp_path / "data" / "v1"
    v3d_dir = tmp_path / "data" / "3d"

    v1_cfg = DatasetPipelineConfig(
        dataset_id="v1_isolated",
        variation_plan=VariationPlanConfig(base_seed=91, count=2, min_obstacles=2, max_obstacles=6),
        supersample=1,
    )
    v1_res = build_dataset(v1_cfg, v1_dir)
    v3d_res = DatasetBuilder3D().build_from_scenes(
        build_default_3d_scenes()[:1],
        output_dir=v3d_dir,
        config=Dataset3DConfig(supersample=1),
    )

    assert (v1_dir / "dataset.jsonl").is_file()
    assert (v1_dir / "manifest.json").is_file()
    assert not (v1_dir / "dataset.json").exists()

    assert (v3d_dir / "dataset.json").is_file()
    assert v1_res.valid_records[0].schema_version == "1.0"
    assert isinstance(v1_res.valid_records[0], DatasetRecord)
    assert isinstance(v3d_res.bundle, Dataset3DBundle)
    assert load_and_validate_dataset_3d(v3d_dir).record_count == v3d_res.bundle.record_count
