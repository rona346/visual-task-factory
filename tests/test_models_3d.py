"""Tests for additive 3D warehouse scene data models."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from visual_task_factory.models.scene import (
    Doorway,
    EntityType,
    ExitMarker,
    Point2D,
    Rect2D,
    ScaleBarSpec,
    SceneEntity,
    SceneSpec,
    WallSegment,
    Zone,
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
from visual_task_factory.models.variation import VariationConfig


def _build_valid_scene_3d() -> SceneSpec3D:
    crate_base = SceneEntity3D(
        id="crate_base",
        label="Base Crate",
        entity_type=EntityType.CRATE,
        bounds=Box3D(x=4.0, y=4.0, z=0.0, width=2.0, depth=2.0, height=1.2),
    )
    crate_top = SceneEntity3D(
        id="crate_top",
        label="Stacked Crate",
        entity_type=EntityType.CRATE,
        bounds=Box3D(x=4.2, y=4.2, z=1.2, width=1.4, depth=1.4, height=1.0),
        supported_by="crate_base",
    )
    rack = SceneEntity3D(
        id="rack_1",
        label="Rack 1",
        entity_type=EntityType.PALLET_RACK,
        bounds=Box3D(x=10.0, y=3.0, z=0.0, width=4.0, depth=2.0, height=3.0),
    )
    return SceneSpec3D(
        scene_id="wh3d_001",
        width=20.0,
        depth=16.0,
        ceiling_height=5.0,
        scale_bar=ScaleBarSpec(world_units=2.0, display_meters=2.0),
        variation=VariationConfig(seed=42, topology_id="wh3d_basic"),
        camera=CameraPose3D(
            look_at=Point3D(x=10.0, y=8.0, z=1.0),
            distance=24.0,
            azimuth_deg=45.0,
            elevation_deg=35.0,
            fov_deg=50.0,
        ),
        walls=[
            WallSegment3D(
                id="wall_north",
                start=Point2D(x=0.0, y=0.0),
                end=Point2D(x=20.0, y=0.0),
                height=5.0,
                is_exterior=True,
            ),
            WallSegment3D(
                id="wall_inner",
                start=Point2D(x=8.0, y=2.0),
                end=Point2D(x=8.0, y=10.0),
                height=3.5,
            ),
        ],
        doors=[
            Doorway3D(
                id="door_1",
                position=Point3D(x=8.0, y=5.0, z=0.0),
                width=2.0,
                height=2.4,
                orientation="vertical",
            )
        ],
        zones=[
            Zone3D(
                id="zone_a",
                label="Zone A",
                zone_type=ZoneType.STORAGE,
                bounds=Rect2D(x=2.0, y=2.0, width=6.0, height=6.0),
            )
        ],
        exits=[
            ExitMarker3D(
                id="exit_east",
                label="Exit E",
                position=Point3D(x=20.0, y=8.0, z=0.0),
                width=2.0,
                height=2.5,
            )
        ],
        entities=[crate_base, crate_top, rack],
    )


def test_scene_spec_3d_valid_and_json_roundtrip() -> None:
    scene = _build_valid_scene_3d()
    eye_x, eye_y, eye_z = scene.camera.eye
    assert eye_z > scene.camera.look_at.z
    assert math.isfinite(eye_x) and math.isfinite(eye_y)

    payload = scene.model_dump_json()
    restored = SceneSpec3D.model_validate_json(payload)
    assert restored == scene
    assert restored.entities[1].supported_by == "crate_base"
    assert restored.entities[0].bounds.z2 == pytest.approx(1.2)


def test_negative_zero_dimensions_and_nonfinite_geometry_rejected() -> None:
    with pytest.raises(ValidationError):
        Box3D(x=0.0, y=0.0, z=0.0, width=0.0, depth=1.0, height=1.0)

    with pytest.raises(ValidationError):
        Box3D(x=0.0, y=0.0, z=0.0, width=1.0, depth=-1.0, height=1.0)

    with pytest.raises(ValidationError):
        Box3D(x=0.0, y=0.0, z=0.0, width=1.0, depth=1.0, height=0.0)

    with pytest.raises(ValidationError):
        Point3D(x=math.nan, y=1.0, z=0.0)

    with pytest.raises(ValidationError):
        Box3D(x=0.0, y=0.0, z=0.0, width=math.inf, depth=1.0, height=1.0)

    with pytest.raises(ValidationError):
        WallSegment3D(
            id="bad_wall",
            start=Point2D(x=2.0, y=2.0),
            end=Point2D(x=2.0, y=2.0),
            height=3.0,
        )

    base = _build_valid_scene_3d()
    for bad_ceiling in (0.0, -2.0):
        with pytest.raises(ValidationError):
            SceneSpec3D.model_validate({**base.model_dump(), "ceiling_height": bad_ceiling})


def test_invalid_camera_elevation_and_fov_rejected() -> None:
    look_at = Point3D(x=5.0, y=5.0, z=1.0)

    for bad_el in (-15.0, 0.0, 89.0, 90.0, math.nan):
        with pytest.raises(ValidationError):
            CameraPose3D(look_at=look_at, elevation_deg=bad_el)

    for bad_fov in (-10.0, 0.0, 10.0, 150.0, math.inf):
        with pytest.raises(ValidationError):
            CameraPose3D(look_at=look_at, fov_deg=bad_fov)

    with pytest.raises(ValidationError):
        CameraPose3D(look_at=look_at, distance=0.0)

    with pytest.raises(ValidationError):
        CameraPose3D(look_at=look_at, azimuth_deg=360.0)


def test_3d_bounds_overlap_and_support_rules() -> None:
    base = _build_valid_scene_3d()
    base_data = base.model_dump()

    # Elevated entity without supported_by is rejected.
    no_support = dict(base_data)
    no_support["entities"] = [
        {
            "id": "floating_crate",
            "label": "Floating Crate",
            "entity_type": "crate",
            "bounds": {"x": 2.0, "y": 2.0, "z": 1.0, "width": 1.0, "depth": 1.0, "height": 1.0},
        }
    ]
    with pytest.raises(ValidationError):
        SceneSpec3D.model_validate(no_support)

    # Stacked entity floating above supporter is rejected.
    bad_gap = dict(base_data)
    bad_gap["entities"] = [
        base_data["entities"][0],
        {
            "id": "crate_top",
            "label": "Gap Crate",
            "entity_type": "crate",
            "bounds": {"x": 4.2, "y": 4.2, "z": 1.5, "width": 1.0, "depth": 1.0, "height": 1.0},
            "supported_by": "crate_base",
        },
    ]
    with pytest.raises(ValidationError):
        SceneSpec3D.model_validate(bad_gap)

    # Overlapping solid entities in 3D space rejected.
    overlapping = dict(base_data)
    overlapping["entities"] = [
        base_data["entities"][0],
        {
            "id": "crate_collide",
            "label": "Collide Crate",
            "entity_type": "crate",
            "bounds": {"x": 5.0, "y": 5.0, "z": 0.0, "width": 2.0, "depth": 2.0, "height": 1.2},
        },
    ]
    with pytest.raises(ValidationError):
        SceneSpec3D.model_validate(overlapping)

    # Exceeding ceiling height rejected.
    too_tall = dict(base_data)
    too_tall["entities"] = [
        {
            "id": "tall_rack",
            "label": "Tall Rack",
            "entity_type": "pallet_rack",
            "bounds": {"x": 1.0, "y": 1.0, "z": 0.0, "width": 2.0, "depth": 1.0, "height": 6.0},
        }
    ]
    with pytest.raises(ValidationError):
        SceneSpec3D.model_validate(too_tall)


def test_extrude_v1_scene_2d_to_3d_preserves_v1_and_validates() -> None:
    scene_2d = SceneSpec(
        scene_id="wh2d_sample",
        width=20.0,
        height=15.0,
        scale_bar=ScaleBarSpec(world_units=2.0, display_meters=2.0),
        variation=VariationConfig(seed=7, topology_id="wh2d_basic"),
        walls=[
            WallSegment(
                id="wall_top",
                start=Point2D(x=0.0, y=0.0),
                end=Point2D(x=20.0, y=0.0),
            )
        ],
        doors=[
            Doorway(
                id="door_main",
                position=Point2D(x=10.0, y=0.0),
                width=2.0,
                orientation="horizontal",
            )
        ],
        zones=[
            Zone(
                id="zone_1",
                label="Zone 1",
                zone_type=ZoneType.LOADING,
                bounds=Rect2D(x=1.0, y=1.0, width=5.0, height=4.0),
            )
        ],
        exits=[
            ExitMarker(
                id="exit_1",
                label="Exit 1",
                position=Point2D(x=20.0, y=7.5),
            )
        ],
        entities=[
            SceneEntity(
                id="rack_a",
                label="Rack A",
                entity_type=EntityType.PALLET_RACK,
                bounds=Rect2D(x=8.0, y=3.0, width=4.0, height=2.0),
            ),
            SceneEntity(
                id="pillar_1",
                label="Pillar 1",
                entity_type=EntityType.PILLAR,
                bounds=Rect2D(x=15.0, y=10.0, width=1.0, height=1.0),
            ),
        ],
    )
    snapshot = scene_2d.model_dump()

    scene_3d = extrude_scene_2d_to_3d(scene_2d, ceiling_height=4.5)
    assert scene_2d.model_dump() == snapshot
    assert scene_3d.scene_id == "wh2d_sample"
    assert scene_3d.width == 20.0
    assert scene_3d.depth == 15.0
    assert scene_3d.ceiling_height == 4.5
    assert scene_3d.entities[0].bounds.height == pytest.approx(3.0)
    assert scene_3d.entities[1].bounds.height == pytest.approx(4.5)
    assert SceneSpec3D.model_validate_json(scene_3d.model_dump_json()) == scene_3d
