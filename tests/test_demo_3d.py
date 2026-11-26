"""Tests for isolated Phase 7 3D end-to-end demo integration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from visual_task_factory.dataset_builder_3d import (
    Dataset3DConfig,
    DatasetBuilder3D,
    build_default_3d_scenes,
)
from visual_task_factory.demo_3d import (
    Demo3DConfig,
    Demo3DReport,
    main as demo_3d_main,
    run_demo_3d,
)
from visual_task_factory.demo_service import DemoService


def test_fresh_scene_mode_artifacts_determinism_and_solver_match(tmp_path: Path) -> None:
    out_a = tmp_path / "demo_a"
    out_b = tmp_path / "demo_b"
    cfg = Demo3DConfig(
        mode="fresh_scene",
        scene_index=0,
        max_questions=5,
        image_width_px=640,
        image_height_px=420,
        supersample=1,
    )

    res_a = run_demo_3d(cfg, output_dir=out_a)
    res_b = run_demo_3d(cfg, output_dir=out_b)

    assert res_a.report_json_path == out_a / "demo_3d_report.json"
    assert res_a.summary_md_path == out_a / "demo_3d_summary.md"
    assert res_a.image_path == out_a / "images" / f"{res_a.report.scene_id}.png"
    assert res_a.report_json_path.is_file()
    assert res_a.summary_md_path.is_file()
    assert res_a.image_path.is_file()

    with Image.open(res_a.image_path) as img:
        assert img.format == "PNG"
        assert img.size == (640, 420)

    rep = res_a.report
    assert rep.mode == "fresh_scene"
    assert rep.total_questions == 5
    assert rep.matched_count == 5
    assert rep.valid_count == 5
    assert rep.all_steps_matched_and_valid is True
    assert all(s.matches_ground_truth and s.validation_is_valid for s in rep.steps)

    # Byte-identical deterministic outputs across runs.
    assert res_a.report_json_path.read_text(encoding="utf-8") == res_b.report_json_path.read_text(
        encoding="utf-8"
    )
    assert res_a.summary_md_path.read_text(encoding="utf-8") == res_b.summary_md_path.read_text(
        encoding="utf-8"
    )
    assert res_a.image_path.read_bytes() == res_b.image_path.read_bytes()

    reloaded = Demo3DReport.model_validate_json(
        res_a.report_json_path.read_text(encoding="utf-8")
    )
    assert reloaded == rep


def test_existing_dataset_mode_and_invalid_config_handling(tmp_path: Path) -> None:
    ds_dir = tmp_path / "data_3d"
    demo_dir = tmp_path / "demo_from_ds"
    build_res = DatasetBuilder3D().build_from_scenes(
        build_default_3d_scenes()[:1],
        output_dir=ds_dir,
        config=Dataset3DConfig(
            dataset_id="demo_src_3d",
            image_width_px=480,
            image_height_px=360,
            supersample=1,
        ),
    )

    cfg = Demo3DConfig(
        mode="existing_dataset",
        dataset_path=str(ds_dir),
        max_questions=4,
    )
    res = run_demo_3d(cfg, output_dir=demo_dir)

    assert res.report.mode == "existing_dataset"
    assert res.report.dataset_id == build_res.bundle.dataset_id
    assert res.report.total_questions == 4
    assert res.report.all_steps_matched_and_valid is True
    assert res.image_path.is_file()
    with Image.open(res.image_path) as img:
        assert img.size == (480, 360)

    # existing_dataset mode requires dataset_path.
    with pytest.raises(ValidationError):
        Demo3DConfig(mode="existing_dataset", dataset_path=None)

    # Out-of-range scene index raises clear IndexError.
    with pytest.raises(IndexError):
        run_demo_3d(Demo3DConfig(mode="fresh_scene", scene_index=99), output_dir=tmp_path / "bad_idx")


def test_demo_3d_cli_module_execution_and_v1_non_interference(tmp_path: Path) -> None:
    cli_out = tmp_path / "cli_demo_out"
    rc = demo_3d_main(
        [
            "--mode",
            "fresh_scene",
            "--max-questions",
            "3",
            "--output-dir",
            str(cli_out),
        ]
    )
    assert rc == 0
    assert (cli_out / "demo_3d_report.json").is_file()
    assert (cli_out / "demo_3d_summary.md").is_file()

    # Subprocess execution of `python -m visual_task_factory.demo_3d` works cleanly.
    sub_out = tmp_path / "sub_cli_out"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "visual_task_factory.demo_3d",
            "--max-questions",
            "2",
            "--output-dir",
            str(sub_out),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "3D Demo complete" in proc.stdout
    assert (sub_out / "demo_3d_report.json").is_file()

    # Confirm V1 DemoService remains cleanly callable and unaffected.
    v1_view = DemoService().load_dataset_view(tmp_path / "nonexistent_v1")
    assert v1_view.error_message is not None
