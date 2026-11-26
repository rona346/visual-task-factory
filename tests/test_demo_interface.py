"""Unit and HTTP integration tests for the Phase 11 DemoService and demo_app."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest
from pydantic import ValidationError

from visual_task_factory.demo_app import create_demo_server
from visual_task_factory.demo_service import DemoSampleRequest, DemoService
from visual_task_factory.metrics_analyzer import MOCK_EVALUATION_NOTICE


def test_demo_service_generates_small_sample_and_mock_eval_with_banner(
    tmp_path: Path,
) -> None:
    svc = DemoService()
    out_dir = tmp_path / "demo_run"
    req = DemoSampleRequest(base_seed=88, scene_count=2, run_mock_eval=True)

    result = svc.generate_demo_sample(req, out_dir)

    assert result.dataset.error_message is None
    assert result.dataset.manifest is not None
    assert result.dataset.manifest.generated_scene_count == 2
    assert result.dataset.manifest.valid_item_count == len(result.dataset.records)
    assert len(result.dataset.records) >= 4

    # Reference answer is stored in ReferenceAnswer, never inside prompt_text.
    for rec in result.dataset.records:
        assert rec.validation.is_valid is True
        img_path = svc.resolve_safe_image_path(out_dir, rec.image_rel_path)
        assert img_path.is_file()

    assert result.evaluation is not None
    assert result.evaluation.mock_warning_banner == MOCK_EVALUATION_NOTICE
    assert result.evaluation.evaluation_summary is not None
    assert result.evaluation.evaluation_summary.is_mock_evaluation is True
    assert result.evaluation.metrics_summary is not None
    assert result.evaluation.failure_analysis is not None


def test_demo_service_graceful_missing_paths_and_traversal_block(
    tmp_path: Path,
) -> None:
    svc = DemoService()

    missing_ds = svc.load_dataset_view(tmp_path / "does_not_exist")
    assert missing_ds.error_message is not None
    assert "dataset.jsonl" in missing_ds.missing_files

    missing_ev = svc.load_evaluation_view(tmp_path / "no_eval")
    assert missing_ev.error_message is not None
    assert "metrics_summary.json" in missing_ev.missing_files

    # Partial evaluation directory (only evaluation_summary.json missing metrics)
    partial_ev_dir = tmp_path / "partial_ev"
    partial_ev_dir.mkdir()
    partial_view = svc.load_evaluation_view(partial_ev_dir)
    assert partial_view.error_message is None
    assert "evaluation_summary.json" in partial_view.missing_files
    assert "metrics_summary.json" in partial_view.missing_files

    # Image path traversal blocked
    ds_dir = tmp_path / "safe_ds"
    ds_dir.mkdir()
    outside_png = tmp_path / "secret.png"
    outside_png.write_bytes(b"PNG")
    with pytest.raises(PermissionError, match="escapes"):
        svc.resolve_safe_image_path(ds_dir, "../secret.png")

    # Small sample cap enforced
    with pytest.raises(ValidationError):
        DemoSampleRequest(scene_count=20)


def test_demo_http_endpoints_end_to_end(tmp_path: Path) -> None:
    demo_out = tmp_path / "http_demo_out"
    server = create_demo_server(
        host="127.0.0.1",
        port=0,
        default_dataset_dir=str(tmp_path / "missing_ds"),
        default_eval_dir=str(tmp_path / "missing_ev"),
        demo_output_dir=demo_out,
    )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base = f"http://127.0.0.1:{port}"

        # 1. HTML home page loads cleanly
        with urllib.request.urlopen(f"{base}/") as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "Visual Task Factory" in html

        # 2. Missing dataset/eval directories return graceful 200 view-models with error_message
        with urllib.request.urlopen(f"{base}/api/dataset") as resp:
            ds_missing = json.loads(resp.read().decode("utf-8"))
            assert ds_missing["error_message"] is not None

        with urllib.request.urlopen(f"{base}/api/evaluation") as resp:
            ev_missing = json.loads(resp.read().decode("utf-8"))
            assert ev_missing["error_message"] is not None

        # 3. POST /api/generate creates small sample + mock eval
        req = urllib.request.Request(
            f"{base}/api/generate",
            data=json.dumps({"base_seed": 12, "scene_count": 1, "run_mock_eval": True}).encode(
                "utf-8"
            ),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            gen_data = json.loads(resp.read().decode("utf-8"))

        ds_dir = gen_data["dataset"]["dataset_dir"]
        rel_img = gen_data["dataset"]["records"][0]["image_rel_path"]
        assert gen_data["evaluation"]["mock_warning_banner"] == MOCK_EVALUATION_NOTICE

        # 4. GET /api/image returns actual PNG bytes
        img_q = urllib.parse.urlencode({"dataset_dir": ds_dir, "image_rel_path": rel_img})
        with urllib.request.urlopen(f"{base}/api/image?{img_q}") as resp:
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "image/png"
            png_bytes = resp.read()
            assert png_bytes.startswith(b"\x89PNG")

        # 5. Path traversal on /api/image returns HTTP 403
        bad_q = urllib.parse.urlencode({"dataset_dir": ds_dir, "image_rel_path": "../evil.png"})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{base}/api/image?{bad_q}")
        assert exc_info.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
