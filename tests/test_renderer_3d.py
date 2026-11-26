"""Tests for the additive fixed-angle isometric 3D warehouse renderer."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from visual_task_factory.models.scene import (
    EntityType,
    Point2D,
    Rect2D,
    ScaleBarSpec,
    ZoneType,
)
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
from visual_task_factory.renderer_3d import (
    Render3DConfig,
    Warehouse3DRenderer,
    _BoxPrimitive,
    render_scene_3d,
    sort_primitives_back_to_front,
)
from visual_task_factory.scene_generator import generate_scene


def _sample_scene_3d() -> SceneSpec3D:
    crate_base = SceneEntity3D(
        id="crate_base",
        label="Base Crate",
        entity_type=EntityType.CRATE,
        bounds=Box3D(x=3.0, y=3.0, z=0.0, width=2.4, depth=2.4, height=1.3),
    )
    crate_top = SceneEntity3D(
        id="crate_top",
        label="Top Crate",
        entity_type=EntityType.CRATE,
        bounds=Box3D(x=3.3, y=3.3, z=1.3, width=1.8, depth=1.8, height=1.1),
        supported_by="crate_base",
    )
    back_rack = SceneEntity3D(
        id="rack_back",
        label="Back Rack",
        entity_type=EntityType.PALLET_RACK,
        bounds=Box3D(x=8.5, y=1.5, z=0.0, width=4.0, depth=2.0, height=3.2),
    )
    front_forklift = SceneEntity3D(
        id="forklift_front",
        label="Forklift 1",
        entity_type=EntityType.FORKLIFT,
        bounds=Box3D(x=9.5, y=8.0, z=0.0, width=3.0, depth=2.0, height=2.2),
    )
    return SceneSpec3D(
        scene_id="wh3d_demo",
        width=18.0,
        depth=14.0,
        ceiling_height=5.0,
        scale_bar=ScaleBarSpec(world_units=2.0, display_meters=2.0),
        variation=VariationConfig(seed=101, topology_id="iso_demo"),
        camera=CameraPose3D(
            look_at=Point3D(x=9.0, y=7.0, z=1.0),
            distance=24.0,
            azimuth_deg=45.0,
            elevation_deg=35.0,
            fov_deg=45.0,
        ),
        walls=[
            WallSegment3D(
                id="wall_north",
                start=Point2D(x=0.0, y=0.0),
                end=Point2D(x=18.0, y=0.0),
                height=4.0,
                is_exterior=True,
            ),
            WallSegment3D(
                id="wall_west",
                start=Point2D(x=0.0, y=0.0),
                end=Point2D(x=0.0, y=14.0),
                height=4.0,
                is_exterior=True,
            ),
            WallSegment3D(
                id="wall_mid",
                start=Point2D(x=7.0, y=1.0),
                end=Point2D(x=7.0, y=10.0),
                height=3.4,
            ),
        ],
        doors=[
            Doorway3D(
                id="door_mid",
                position=Point3D(x=7.0, y=5.5, z=0.0),
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
                bounds=Rect2D(x=1.5, y=1.5, width=4.5, height=4.5),
            ),
            Zone3D(
                id="zone_hazard",
                label="Hazard B",
                zone_type=ZoneType.HAZARD,
                bounds=Rect2D(x=12.0, y=7.5, width=4.5, height=4.5),
            ),
        ],
        exits=[
            ExitMarker3D(
                id="exit_east",
                label="Exit E",
                position=Point3D(x=18.0, y=7.0, z=0.0),
                width=2.0,
                height=2.6,
            )
        ],
        entities=[crate_base, crate_top, back_rack, front_forklift],
    )


def test_render_scene_3d_writes_valid_png_with_expected_dimensions(tmp_path: Path) -> None:
    scene = _sample_scene_3d()
    out_file = tmp_path / "nested" / "scene_3d.png"
    cfg = Render3DConfig(width_px=640, height_px=420, supersample=1)

    result = render_scene_3d(scene, output_path=out_file, config=cfg)

    assert out_file.exists()
    assert result.output_path == out_file
    assert result.width_px == 640
    assert result.height_px == 420
    assert result.image.size == (640, 420)
    assert result.cycle_detected is False

    with Image.open(out_file) as loaded:
        assert loaded.format == "PNG"
        assert loaded.size == (640, 420)
        assert loaded.mode == "RGB"


def test_render_scene_3d_determinism_and_no_mutation(tmp_path: Path) -> None:
    scene = _sample_scene_3d()
    snapshot = scene.model_dump()

    p1 = tmp_path / "run1.png"
    p2 = tmp_path / "run2.png"
    renderer = Warehouse3DRenderer(Render3DConfig(supersample=1))

    res1 = renderer.render(scene, output_path=p1)
    res2 = renderer.render(scene, output_path=p2)

    assert scene.model_dump() == snapshot
    assert res1.draw_order == res2.draw_order
    assert p1.read_bytes() == p2.read_bytes()


def test_invalid_render_3d_config_and_inputs_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Render3DConfig(width_px=120)
    with pytest.raises(ValueError):
        Render3DConfig(height_px=100)
    with pytest.raises(ValueError):
        Render3DConfig(supersample=4)
    with pytest.raises(ValueError):
        Render3DConfig(margin_px=2)
    with pytest.raises(ValueError):
        Render3DConfig(width_px=260, height_px=200, header_height=100, footer_height=100)

    scene = _sample_scene_3d()
    renderer = Warehouse3DRenderer()
    with pytest.raises(TypeError):
        renderer.render("not_a_scene")  # type: ignore[arg-type]
    with pytest.raises(IsADirectoryError):
        renderer.render(scene, output_path=tmp_path)


def test_isometric_face_shading_and_depth_occlusion() -> None:
    """Verify 3D box renders 3 distinct face shades and foreground box occludes back box."""
    config = Render3DConfig(
        width_px=480,
        height_px=360,
        supersample=1,
        show_labels=False,
        show_grid=False,
        show_zones=False,
        theme=VisualTheme.HIGH_CONTRAST,
    )

    # Single large centered crate: top (+Z), right (+X), left (+Y) faces must have distinct colors.
    single_crate = SceneEntity3D(
        id="single_crate",
        label="Single",
        entity_type=EntityType.CRATE,
        bounds=Box3D(x=3.0, y=3.0, z=0.0, width=4.0, depth=4.0, height=4.0),
    )
    scene_single = SceneSpec3D(
        scene_id="shade_check",
        width=10.0,
        depth=10.0,
        ceiling_height=6.0,
        scale_bar=ScaleBarSpec(world_units=2.0, display_meters=2.0),
        variation=VariationConfig(seed=1, topology_id="shade"),
        camera=CameraPose3D(look_at=Point3D(x=5.0, y=5.0, z=1.0)),
        exits=[ExitMarker3D(id="ex1", label="Ex1", position=Point3D(x=0.0, y=0.0, z=0.0))],
        entities=[single_crate],
    )
    res_single = render_scene_3d(scene_single, config=config)
    colors = res_single.image.getcolors(maxcolors=200_000)
    assert colors is not None
    # Floor + walls/exits + 3 distinct shaded crate faces + outlines produce > 5 distinct RGB colors.
    assert len(colors) >= 6

    # Occlusion check: tall foreground pallet rack in front (+X/+Y/+Z) occludes back crate.
    back_crate = SceneEntity3D(
        id="crate_back",
        label="Back",
        entity_type=EntityType.CRATE,
        bounds=Box3D(x=1.0, y=1.0, z=0.0, width=1.5, depth=1.5, height=1.2),
    )
    front_rack = SceneEntity3D(
        id="rack_front",
        label="Front",
        entity_type=EntityType.PALLET_RACK,
        bounds=Box3D(x=3.0, y=3.0, z=0.0, width=4.5, depth=4.5, height=5.0),
    )
    # Deliberately supply front_rack BEFORE back_crate in entity list to test depth sorting.
    scene_occluded = SceneSpec3D(
        scene_id="occlusion_check",
        width=10.0,
        depth=10.0,
        ceiling_height=6.0,
        scale_bar=ScaleBarSpec(world_units=2.0, display_meters=2.0),
        variation=VariationConfig(seed=2, topology_id="occlusion"),
        camera=CameraPose3D(look_at=Point3D(x=5.0, y=5.0, z=1.0)),
        exits=[ExitMarker3D(id="ex1", label="Ex1", position=Point3D(x=0.0, y=0.0, z=0.0))],
        entities=[front_rack, back_crate],
    )
    res_occluded = render_scene_3d(scene_occluded, config=config)
    assert res_occluded.draw_order.index("crate_back") < res_occluded.draw_order.index("rack_front")

    # Stacked crate order: base must render before supported top crate.
    stacked_scene = _sample_scene_3d()
    res_stacked = render_scene_3d(stacked_scene, config=config)
    assert res_stacked.draw_order.index("crate_base") < res_stacked.draw_order.index("crate_top")


def test_painter_algorithm_detects_cyclic_pinwheel_limitation() -> None:
    """Documented limitation: 3-bar cyclic pinwheel overlap cannot have an acyclic whole-box order."""
    r0 = _BoxPrimitive(
        prim_id="bar_0",
        bounds=Box3D(x=0.0, y=2.0, z=4.0, width=8.0, depth=1.0, height=1.0),
        base_color=(200, 80, 80),
        border_color=(40, 20, 20),
    )
    r1 = _BoxPrimitive(
        prim_id="bar_1",
        bounds=Box3D(x=4.0, y=0.0, z=2.0, width=1.0, depth=8.0, height=1.0),
        base_color=(80, 200, 80),
        border_color=(20, 40, 20),
    )
    r2 = _BoxPrimitive(
        prim_id="bar_2",
        bounds=Box3D(x=2.0, y=4.0, z=0.0, width=1.0, depth=1.0, height=8.0),
        base_color=(80, 80, 200),
        border_color=(20, 20, 40),
    )
    ordered, cycle_detected = sort_primitives_back_to_front([r0, r1, r2])
    assert cycle_detected is True
    assert len(ordered) == 3


def test_extruded_v1_scene_renders_crowded_rack_and_perimeter_exit_labels(tmp_path: Path) -> None:
    """Verify dense adjacent racks (Rack 2) and perimeter exits remain visibly labeled."""
    v1_scene = generate_scene(
        VariationConfig(
            seed=202,
            topology_id="two_zone_corridor",
            theme=VisualTheme.BLUEPRINT,
        )
    )
    scene_3d = extrude_scene_2d_to_3d(v1_scene, ceiling_height=4.5)
    out_file = tmp_path / f"{scene_3d.scene_id}.png"

    res = render_scene_3d(scene_3d, output_path=out_file)

    assert out_file.is_file()
    assert "Rack 1" in res.placed_labels
    assert "Rack 2" in res.placed_labels
    assert "Rack 3" in res.placed_labels
    assert "Exit E1" in res.placed_labels
    assert "Exit E2" in res.placed_labels
    assert "Exit E3" in res.placed_labels

