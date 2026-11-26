"""Unit tests for the deterministic 2D warehouse PNG renderer."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from visual_task_factory.models import (
    ClutterLevel,
    DifficultyTier,
    VariationConfig,
    VisualTheme,
)
from visual_task_factory.renderer import RenderConfig, RenderResult, render_scene
from visual_task_factory.scene_generator import generate_scene


def _make_scene(
    *,
    seed: int = 12,
    difficulty: DifficultyTier = DifficultyTier.MEDIUM,
    orientation_deg: int = 0,
    theme: VisualTheme = VisualTheme.SAFETY_STANDARD,
):
    cfg = VariationConfig(
        seed=seed,
        topology_id="warehouse_split_aisle_v1",
        difficulty=difficulty,
        orientation_deg=orientation_deg,  # type: ignore[arg-type]
        scale_meters_per_unit=0.5,
        theme=theme,
        clutter_level=ClutterLevel.MEDIUM,
        min_hazard_zones=1,
        min_obstacles=3,
        max_obstacles=8,
    )
    return generate_scene(cfg)


def test_render_scene_creates_valid_png_and_creates_parent_dir(tmp_path: Path) -> None:
    scene = _make_scene(seed=101, difficulty=DifficultyTier.HARD)
    out_file = tmp_path / "nested" / "floorplans" / "scene_101.png"

    result = render_scene(scene, out_file)

    assert isinstance(result, RenderResult)
    assert result.output_path == out_file
    assert out_file.is_file()
    assert out_file.stat().st_size > 1000
    assert result.width_px > 0
    assert result.height_px > 0
    assert result.audit_overlay is False

    with Image.open(out_file) as img:
        assert img.format == "PNG"
        assert img.size == (result.width_px, result.height_px)
        assert img.mode == "RGB"


def test_render_does_not_mutate_input_scene_spec(tmp_path: Path) -> None:
    scene = _make_scene(seed=77)
    before_json = scene.model_dump_json()

    render_scene(scene, tmp_path / "immutable.png")
    after_json = scene.model_dump_json()

    assert after_json == before_json


def test_render_determinism_byte_identical_png(tmp_path: Path) -> None:
    scene = _make_scene(seed=33, difficulty=DifficultyTier.MEDIUM, orientation_deg=90)
    out_a = tmp_path / "run_a.png"
    out_b = tmp_path / "run_b.png"

    res_a = render_scene(scene, out_a)
    res_b = render_scene(scene, out_b)

    assert res_a.width_px == res_b.width_px
    assert res_a.height_px == res_b.height_px
    assert res_a.font_name == res_b.font_name
    assert out_a.read_bytes() == out_b.read_bytes()


@pytest.mark.parametrize("orientation", [0, 90, 180, 270])
def test_render_all_supported_orientations(tmp_path: Path, orientation: int) -> None:
    scene = _make_scene(seed=42, orientation_deg=orientation)
    out_file = tmp_path / f"orient_{orientation}.png"

    result = render_scene(scene, out_file)

    assert result.orientation_deg == orientation
    with Image.open(out_file) as img:
        assert img.format == "PNG"
        w, h = img.size
        assert w > 0 and h > 0


@pytest.mark.parametrize(
    "theme",
    [
        VisualTheme.SAFETY_STANDARD,
        VisualTheme.BLUEPRINT,
        VisualTheme.HIGH_CONTRAST,
        VisualTheme.CLUTTERED_OPS,
    ],
)
def test_render_all_visual_themes_and_audit_overlay(tmp_path: Path, theme: VisualTheme) -> None:
    scene = _make_scene(seed=88, difficulty=DifficultyTier.HARD, theme=theme)
    normal_path = tmp_path / f"{theme.value}.png"
    audit_path = tmp_path / f"{theme.value}_audit.png"

    res_normal = render_scene(scene, normal_path)
    res_audit = render_scene(scene, audit_path, audit_overlay=True)

    assert res_normal.theme == theme
    assert res_audit.audit_overlay is True
    assert normal_path.read_bytes() != audit_path.read_bytes()


def test_render_invalid_inputs_and_paths_raise_clear_errors(tmp_path: Path) -> None:
    scene = _make_scene(seed=5)

    with pytest.raises(TypeError, match="Expected SceneSpec"):
        render_scene({"not": "a scene"}, tmp_path / "bad.png")  # type: ignore[arg-type]

    with pytest.raises(IsADirectoryError):
        render_scene(scene, tmp_path)

    with pytest.raises(ValueError, match=r"\.png"):
        render_scene(scene, tmp_path / "wrong_ext.jpg")

    with pytest.raises(ValueError, match="empty"):
        render_scene(scene, "   ")

    with pytest.raises(ValueError, match="pixels_per_unit"):
        RenderConfig(pixels_per_unit=2)
