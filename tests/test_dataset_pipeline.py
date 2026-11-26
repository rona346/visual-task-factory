"""Unit and integration tests for the end-to-end DatasetPipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from visual_task_factory.dataset_pipeline import (
    DatasetManifest,
    DatasetPipeline,
    DatasetPipelineConfig,
    DatasetRecord,
    build_dataset,
)
from visual_task_factory.models import BenchmarkItem, QuestionFamily, SceneSpec
from visual_task_factory.question_generator import QuestionGenerator
from visual_task_factory.variation_engine import VariationPlanConfig


def _small_pipeline_config(
    *,
    base_seed: int = 500,
    count: int = 3,
    dataset_id: str = "test_ds_v1",
) -> DatasetPipelineConfig:
    return DatasetPipelineConfig(
        dataset_id=dataset_id,
        variation_plan=VariationPlanConfig(
            base_seed=base_seed,
            count=count,
            min_obstacles=3,
            max_obstacles=8,
        ),
        max_questions_per_family=1,
    )


def test_end_to_end_dataset_generation_and_artifacts(tmp_path: Path) -> None:
    cfg = _small_pipeline_config(base_seed=123, count=3)
    cfg_snapshot = cfg.model_dump_json()
    out_dir = tmp_path / "bench_out"

    result = build_dataset(cfg, out_dir)

    # Input config must remain immutable.
    assert cfg.model_dump_json() == cfg_snapshot

    # Artifacts exist on disk.
    manifest_path = out_dir / "manifest.json"
    dataset_path = out_dir / "dataset.jsonl"
    rejected_path = out_dir / "rejected_items.jsonl"
    scenes_path = out_dir / "scenes.jsonl"

    assert manifest_path.is_file()
    assert dataset_path.is_file()
    assert rejected_path.is_file()
    assert scenes_path.is_file()

    # Manifest validates cleanly and counts align.
    loaded_manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert loaded_manifest == result.manifest
    assert loaded_manifest.requested_scene_count == 3
    assert loaded_manifest.generated_scene_count == 3
    assert loaded_manifest.valid_item_count == len(result.valid_records)
    assert loaded_manifest.invalid_item_count == len(result.rejected_records)
    assert loaded_manifest.valid_item_count >= 6
    assert loaded_manifest.invalid_item_count == 0
    assert loaded_manifest.unique_structural_signatures == 3
    assert loaded_manifest.render_failures == []

    # Every exported dataset.jsonl record deserializes, passed validation, and points to a real PNG.
    lines = [ln for ln in dataset_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == loaded_manifest.valid_item_count
    seen_record_ids: set[str] = set()
    for line in lines:
        rec = DatasetRecord.model_validate_json(line)
        assert rec.validation.is_valid is True
        assert rec.record_id not in seen_record_ids
        seen_record_ids.add(rec.record_id)
        assert not Path(rec.image_rel_path).is_absolute()
        img_file = out_dir / rec.image_rel_path
        assert img_file.is_file()
        with Image.open(img_file) as img:
            assert img.format == "PNG"
            assert img.size == (rec.image_width_px, rec.image_height_px)


def test_deterministic_logical_records_and_reproducible_ids(tmp_path: Path) -> None:
    cfg = _small_pipeline_config(base_seed=777, count=3, dataset_id="repro_ds")
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    res_a = build_dataset(cfg, dir_a)
    res_b = build_dataset(cfg, dir_b)

    assert res_a.manifest.logical_dataset_sha256 == res_b.manifest.logical_dataset_sha256
    assert (dir_a / "dataset.jsonl").read_text(encoding="utf-8") == (
        dir_b / "dataset.jsonl"
    ).read_text(encoding="utf-8")
    assert [r.record_id for r in res_a.valid_records] == [
        r.record_id for r in res_b.valid_records
    ]
    assert res_a.manifest.image_sha256_by_scene == res_b.manifest.image_sha256_by_scene


def test_invalid_and_duplicate_items_are_excluded_and_reported(tmp_path: Path) -> None:
    class _CorruptingQuestionGenerator(QuestionGenerator):
        """Test double injecting 1 invalid answer + 1 duplicate question_id alongside valid items."""

        def generate(self, scene: SceneSpec) -> list[BenchmarkItem]:
            valid_items = super().generate(scene)
            if scene.scene_id.endswith("_000_s500") and len(valid_items) >= 2:
                # 1. Corrupt reference answer on first item's copy
                wrong_exit = next(
                    (e.id for e in scene.exits if e.id != str(valid_items[0].question.reference_answer.value)),
                    "E1",
                )
                bad_ref = valid_items[0].question.reference_answer.model_copy(
                    update={"value": wrong_exit}
                )
                corrupted = BenchmarkItem(
                    scene=scene,
                    question=valid_items[0].question.model_copy(
                        update={
                            "question_id": f"{scene.scene_id}__corrupted",
                            "reference_answer": bad_ref,
                        }
                    ),
                )
                # 2. Exact duplicate question_id of valid_items[1]
                duplicate = valid_items[1].model_copy(deep=True)
                return [*valid_items, corrupted, duplicate]
            return valid_items

    cfg = _small_pipeline_config(base_seed=500, count=2, dataset_id="filter_test")
    pipeline = DatasetPipeline(question_generator=_CorruptingQuestionGenerator())
    out_dir = tmp_path / "filtered_out"
    result = pipeline.build(cfg, out_dir)

    assert result.manifest.invalid_item_count == 2
    assert len(result.rejected_records) == 2
    assert result.manifest.valid_item_count + result.manifest.invalid_item_count == (
        result.manifest.generated_question_count
    )
    assert len(result.manifest.duplicate_question_ids) == 1

    # dataset.jsonl must only contain valid items and zero rejected IDs.
    exported_ids = {
        json.loads(ln)["question_id"]
        for ln in (out_dir / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }
    assert f"{result.valid_records[0].scene_id}__corrupted" not in exported_ids

    rejected_lines = [
        json.loads(ln)
        for ln in (out_dir / "rejected_items.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rejected_lines) == 2


def test_clear_failure_handling_for_invalid_output_dir_and_config(tmp_path: Path) -> None:
    cfg = _small_pipeline_config(base_seed=1, count=1)

    existing_file = tmp_path / "not_a_dir.txt"
    existing_file.write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_dataset(cfg, existing_file)

    with pytest.raises(ValueError, match="empty"):
        build_dataset(cfg, "   ")

    with pytest.raises(ValueError, match="question_families"):
        DatasetPipelineConfig(question_families=())
