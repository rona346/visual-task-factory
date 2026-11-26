"""Tests for the deterministic CPU-only 3D geometric solver."""

from __future__ import annotations

import pytest

from visual_task_factory.dataset_builder_3d import build_default_3d_scenes
from visual_task_factory.models.scene import EntityType, Point2D, ScaleBarSpec
from visual_task_factory.models.scene3d import (
    Box3D,
    CameraPose3D,
    Doorway3D,
    ExitMarker3D,
    Point3D,
    SceneEntity3D,
    SceneSpec3D,
    WallSegment3D,
)
from visual_task_factory.models.variation import VariationConfig
from visual_task_factory.qa_3d import (
    AnswerType3D,
    BenchmarkItem3D,
    CameraVisibilityQuery3D,
    GroundTruthQAValidator3D,
    QuestionFamily3D,
    ReferenceAnswer3D,
    VisualQuestion3D,
    evaluate_camera_visibility_query,
)
from visual_task_factory.solver_3d import (
    BoxOverlapRelation,
    GroundTruthSolver3D,
    UnsolvableTask3DError,
    UnsupportedTask3DError,
    VerticalClearanceStatus,
    VerticalRelation,
    VisibilityStatus,
    analyze_vertical_relation,
    check_box_overlap_3d,
    compute_vertical_clearance,
    ray_intersects_box3d,
)


def _scene_for_solver_tests() -> SceneSpec3D:
    floor_crate = SceneEntity3D(
        id="crate_floor",
        label="Floor Crate",
        entity_type=EntityType.CRATE,
        bounds=Box3D(x=2.0, y=2.0, z=0.0, width=2.0, depth=2.0, height=1.2),
    )
    stacked_crate = SceneEntity3D(
        id="crate_stacked",
        label="Stacked Crate",
        entity_type=EntityType.CRATE,
        bounds=Box3D(x=2.2, y=2.2, z=1.2, width=1.6, depth=1.6, height=0.8),
        supported_by="crate_floor",
    )
    side_rack = SceneEntity3D(
        id="rack_side",
        label="Side Rack",
        entity_type=EntityType.PALLET_RACK,
        bounds=Box3D(x=6.0, y=2.0, z=0.0, width=2.0, depth=2.0, height=3.0),
    )
    # Tall solid wall standing directly between camera (at +X, +Y, +Z) and hidden_crate.
    hidden_crate = SceneEntity3D(
        id="crate_hidden",
        label="Hidden Crate",
        entity_type=EntityType.CRATE,
        bounds=Box3D(x=10.0, y=2.0, z=0.0, width=1.5, depth=1.5, height=1.0),
    )
    tall_rack_occluder = SceneEntity3D(
        id="rack_occluder",
        label="Occluder Rack",
        entity_type=EntityType.PALLET_RACK,
        bounds=Box3D(x=12.0, y=4.0, z=0.0, width=5.0, depth=5.0, height=4.5),
    )
    passable_marker = SceneEntity3D(
        id="passable_pad",
        label="Pad",
        entity_type=EntityType.CRATE,
        bounds=Box3D(x=9.5, y=1.5, z=0.0, width=2.5, depth=2.5, height=1.5),
        passable=True,
    )
    return SceneSpec3D(
        scene_id="wh3d_solver_test",
        width=20.0,
        depth=16.0,
        ceiling_height=5.0,
        scale_bar=ScaleBarSpec(world_units=2.0, display_meters=4.0),  # 2.0 m/world unit
        variation=VariationConfig(seed=11, topology_id="solver_3d"),
        camera=CameraPose3D(
            look_at=Point3D(x=10.0, y=8.0, z=1.0),
            distance=20.0,
            azimuth_deg=45.0,
            elevation_deg=25.0,
            fov_deg=50.0,
        ),
        walls=[
            WallSegment3D(
                id="wall_with_door",
                start=Point2D(x=1.0, y=10.0),
                end=Point2D(x=15.0, y=10.0),
                height=4.0,
            )
        ],
        doors=[
            Doorway3D(
                id="door_open",
                position=Point3D(x=8.0, y=10.0, z=0.0),
                width=2.0,
                height=2.5,
                orientation="horizontal",
                is_blocked=False,
            )
        ],
        exits=[
            ExitMarker3D(
                id="exit_1",
                label="Exit 1",
                position=Point3D(x=20.0, y=8.0, z=0.0),
            )
        ],
        entities=[
            floor_crate,
            stacked_crate,
            side_rack,
            hidden_crate,
            tall_rack_occluder,
            passable_marker,
        ],
    )


def test_box_overlap_3d_handles_overlap_touching_disjoint_and_xy_vs_3d() -> None:
    a = Box3D(x=1.0, y=1.0, z=0.0, width=2.0, depth=2.0, height=2.0)

    # Interior 3D overlap (scale 2.0 m/unit)
    b_overlap = Box3D(x=2.0, y=2.0, z=1.0, width=2.0, depth=2.0, height=2.0)
    res_overlap = check_box_overlap_3d(a, b_overlap, meters_per_world_unit=2.0)
    assert res_overlap.relation == BoxOverlapRelation.INTERIOR_OVERLAP
    assert res_overlap.overlaps_3d is True
    assert res_overlap.touches_3d is False
    assert res_overlap.overlaps_xy is True
    assert res_overlap.intersection_dimensions == (1.0, 1.0, 1.0)
    assert res_overlap.overlap_volume == pytest.approx(1.0)
    assert res_overlap.surface_distance_units == 0.0

    # Boundary touching flush in Z
    b_touch_z = Box3D(x=1.0, y=1.0, z=2.0, width=2.0, depth=2.0, height=1.0)
    res_touch = check_box_overlap_3d(a, b_touch_z)
    assert res_touch.relation == BoxOverlapRelation.BOUNDARY_TOUCHING
    assert res_touch.overlaps_3d is False
    assert res_touch.touches_3d is True
    assert res_touch.touching_axes == ("z",)
    assert res_touch.overlap_volume == 0.0

    # Same XY footprint, separated in Z -> MUST be DISJOINT in 3D!
    b_above_gap = Box3D(x=1.0, y=1.0, z=3.5, width=2.0, depth=2.0, height=1.0)
    res_above = check_box_overlap_3d(a, b_above_gap, meters_per_world_unit=2.0)
    assert res_above.relation == BoxOverlapRelation.DISJOINT
    assert res_above.overlaps_3d is False
    assert res_above.touches_3d is False
    assert res_above.overlaps_xy is True
    assert res_above.surface_distance_units == pytest.approx(1.5)
    assert res_above.surface_distance_meters == pytest.approx(3.0)


def test_vertical_relations_above_below_flush_equal_and_overlapping_heights() -> None:
    low = Box3D(x=0.0, y=0.0, z=0.0, width=2.0, depth=2.0, height=1.0)
    high = Box3D(x=0.5, y=0.5, z=2.5, width=1.0, depth=1.0, height=1.0)
    flush = Box3D(x=0.0, y=0.0, z=1.0, width=2.0, depth=2.0, height=1.0)
    equal_side = Box3D(x=4.0, y=0.0, z=0.0, width=2.0, depth=2.0, height=1.0)
    partial_side = Box3D(x=4.0, y=0.0, z=0.5, width=2.0, depth=2.0, height=1.5)

    r_above = analyze_vertical_relation(high, low)
    assert r_above.relation == VerticalRelation.STRICTLY_ABOVE
    assert r_above.is_above is True
    assert r_above.is_below is False
    assert r_above.vertical_gap_units == pytest.approx(1.5)

    r_below = analyze_vertical_relation(low, high)
    assert r_below.relation == VerticalRelation.STRICTLY_BELOW
    assert r_below.is_below is True
    assert r_below.vertical_gap_units == pytest.approx(1.5)

    r_flush = analyze_vertical_relation(flush, low, is_directly_supported=True)
    assert r_flush.relation == VerticalRelation.FLUSH_ABOVE
    assert r_flush.is_above is True
    assert r_flush.is_flush_contact is True
    assert r_flush.is_directly_supported is True
    assert r_flush.vertical_gap_units == pytest.approx(0.0)

    r_equal = analyze_vertical_relation(low, equal_side)
    assert r_equal.relation == VerticalRelation.EQUAL_HEIGHT
    assert r_equal.heights_overlap is True
    assert r_equal.vertical_gap_units is None

    r_partial = analyze_vertical_relation(low, partial_side)
    assert r_partial.relation == VerticalRelation.OVERLAPPING_HEIGHT
    assert r_partial.heights_overlap is True
    assert r_partial.vertical_overlap_units == pytest.approx(0.5)
    assert r_partial.vertical_gap_units is None


def test_vertical_clearance_defined_and_ambiguous_cases() -> None:
    solver = GroundTruthSolver3D()
    scene = _scene_for_solver_tests()

    # Flush stacked pair has valid 0.0 vertical clearance.
    res_flush = solver.solve_vertical_clearance(scene, "crate_stacked", "crate_floor")
    assert res_flush.is_defined is True
    assert res_flush.status == VerticalClearanceStatus.DEFINED
    assert res_flush.clearance_units == pytest.approx(0.0)
    assert res_flush.clearance_meters == pytest.approx(0.0)

    # Separated vertically with XY overlap -> scaled meters (2.0 m / unit).
    base = Box3D(x=1.0, y=1.0, z=0.0, width=2.0, depth=2.0, height=1.2)
    beam = Box3D(x=1.5, y=1.5, z=2.7, width=2.0, depth=2.0, height=0.5)
    res_gap = compute_vertical_clearance(beam, base, meters_per_world_unit=2.0)
    assert res_gap.is_defined is True
    assert res_gap.status == VerticalClearanceStatus.DEFINED
    assert res_gap.clearance_units == pytest.approx(1.5)
    assert res_gap.clearance_meters == pytest.approx(3.0)

    # Ambiguous: side-by-side objects with overlapping Z heights must NOT invent 0.0 clearance.
    res_overlap_h = solver.solve_vertical_clearance(scene, "crate_floor", "rack_side")
    assert res_overlap_h.is_defined is False
    assert res_overlap_h.status == VerticalClearanceStatus.OVERLAPPING_HEIGHTS
    assert res_overlap_h.clearance_units is None
    assert res_overlap_h.clearance_meters is None

    # Ambiguous: higher in Z, but disjoint XY footprint -> NO_XY_OVERLAP, NOT invented.
    disjoint_high = Box3D(x=8.0, y=8.0, z=2.5, width=1.0, depth=1.0, height=1.0)
    res_no_xy = compute_vertical_clearance(disjoint_high, base)
    assert res_no_xy.is_defined is False
    assert res_no_xy.status == VerticalClearanceStatus.NO_XY_OVERLAP
    assert res_no_xy.clearance_units is None

    # Same object pair returns explicit undefined SAME_OBJECT status.
    res_same = solver.solve_vertical_clearance(scene, "crate_floor", "crate_floor")
    assert res_same.is_defined is False
    assert res_same.status == VerticalClearanceStatus.SAME_OBJECT
    assert res_same.clearance_units is None

    # Ceiling clearance for crate_stacked (z2 = 2.0, ceiling = 5.0, scale = 2.0 -> 3.0 units = 6.0 m).
    res_ceil = solver.solve_ceiling_clearance(scene, "crate_stacked")
    assert res_ceil.is_defined is True
    assert res_ceil.clearance_units == pytest.approx(3.0)
    assert res_ceil.clearance_meters == pytest.approx(6.0)


def test_camera_visibility_raycast_clear_and_occluded_plus_errors() -> None:
    solver = GroundTruthSolver3D()
    scene = _scene_for_solver_tests()

    # Low-level ray slab test.
    blocker = Box3D(x=4.0, y=4.0, z=0.0, width=2.0, depth=2.0, height=3.0)
    assert ray_intersects_box3d((10.0, 10.0, 1.5), (0.0, 0.0, 1.5), blocker) is not None
    assert ray_intersects_box3d((10.0, 10.0, 5.0), (0.0, 0.0, 5.0), blocker) is None

    # Occluder rack stands directly in front (+X/+Y/+Z) of crate_hidden -> FULLY_OCCLUDED.
    vis_hidden = solver.solve_camera_visibility(scene, "crate_hidden")
    assert vis_hidden.status == VisibilityStatus.FULLY_OCCLUDED
    assert vis_hidden.center_visible is False
    assert vis_hidden.visible_sample_count == 0
    assert "rack_occluder" in vis_hidden.blocking_object_ids
    assert "not equivalent" in vis_hidden.limitation_note.lower()

    # Foreground rack_occluder itself is unobstructed from the camera.
    vis_front = solver.solve_camera_visibility(scene, "rack_occluder")
    assert vis_front.status == VisibilityStatus.CLEAR
    assert vis_front.center_visible is True
    assert vis_front.visible_fraction == pytest.approx(1.0)
    assert vis_front.blocking_object_ids == []

    # Determinism & unknown ID handling.
    assert solver.solve_camera_visibility(scene, "crate_hidden") == vis_hidden
    with pytest.raises(UnsolvableTask3DError):
        solver.solve_vertical_relation(scene, "missing_id", "crate_floor")
    with pytest.raises(TypeError):
        solver.solve_box_overlap("not_a_scene", "crate_floor", "rack_side")  # type: ignore[arg-type]

    # Diagonal wall triggers explicit UnsupportedTask3DError when queried as an AABB.
    diag_scene = SceneSpec3D(
        scene_id="diag_wall_scene",
        width=10.0,
        depth=10.0,
        ceiling_height=4.0,
        scale_bar=ScaleBarSpec(world_units=1.0, display_meters=1.0),
        variation=VariationConfig(seed=1, topology_id="diag"),
        camera=CameraPose3D(look_at=Point3D(x=5.0, y=5.0, z=1.0)),
        walls=[
            WallSegment3D(
                id="diag_w",
                start=Point2D(x=1.0, y=1.0),
                end=Point2D(x=5.0, y=5.0),
                height=3.0,
            )
        ],
        exits=[ExitMarker3D(id="e1", label="E1", position=Point3D(x=10.0, y=5.0, z=0.0))],
    )
    with pytest.raises(UnsupportedTask3DError):
        solver.solve_ceiling_clearance(diag_scene, "diag_w")


def test_orthographic_visibility_aligns_with_wh3d_stacked_01_render() -> None:
    """Base Crate peeks out visibly beneath Top Crate and above wall_partition in isometric view."""
    solver = GroundTruthSolver3D()
    stacked_scene = build_default_3d_scenes()[0]
    assert stacked_scene.scene_id == "wh3d_stacked_01"

    # Orthographic isometric (+1, +1, +1) detects Base Crate's +Y face as open/visible,
    # while its top is covered by Top Crate and wall_partition clips the +X edge -> PARTIALLY_OCCLUDED.
    vis_base = solver.solve_camera_visibility(stacked_scene, "crate_base")
    assert vis_base.visibility_model == "orthographic_isometric_sampled_raycast"
    assert vis_base.status == VisibilityStatus.PARTIALLY_OCCLUDED
    assert vis_base.visible_sample_count == 1

    # Genuinely occluded object behind Front Rack remains FULLY_OCCLUDED.
    vis_hidden = solver.solve_camera_visibility(stacked_scene, "crate_hidden")
    assert vis_hidden.status == VisibilityStatus.FULLY_OCCLUDED
    assert vis_hidden.blocking_object_ids == ["rack_occluder"]

    # Foreground rack remains CLEAR.
    vis_front = solver.solve_camera_visibility(stacked_scene, "rack_occluder")
    assert vis_front.status == VisibilityStatus.CLEAR

    # QA evaluator refuses a False visibility claim on visibly rendered Base Crate,
    # while accepting False for genuinely hidden crate_hidden and True for Front Rack.
    with pytest.raises(UnsolvableTask3DError):
        evaluate_camera_visibility_query(
            solver,
            stacked_scene,
            CameraVisibilityQuery3D(target_id="crate_base", require_clear=True),
        )

    false_claim_item = BenchmarkItem3D(
        scene=stacked_scene,
        question=VisualQuestion3D(
            question_id="wh3d_stacked_01__vis3d__01",
            scene_id=stacked_scene.scene_id,
            family=QuestionFamily3D.CAMERA_VISIBILITY_3D,
            template_id="camera_visibility_clear_v1",
            prompt_text="Is Base Crate visibly present and unobstructed from the camera viewpoint?",
            expected_answer_type=AnswerType3D.BOOLEAN,
            reference_answer=ReferenceAnswer3D(answer_type=AnswerType3D.BOOLEAN, value=False),
            solver_query={"family": "camera_visibility_3d", "target_id": "crate_base", "require_clear": True},
        ),
    )
    rep = GroundTruthQAValidator3D(solver=solver).validate(false_claim_item)
    assert rep.is_valid is False
    assert any(f.code == "AMBIGUOUS_OR_UNSUPPORTED_GROUND_TRUTH" for f in rep.errors)

    hidden_ref = evaluate_camera_visibility_query(
        solver,
        stacked_scene,
        CameraVisibilityQuery3D(target_id="crate_hidden", require_clear=True),
    )
    assert hidden_ref.value is False

