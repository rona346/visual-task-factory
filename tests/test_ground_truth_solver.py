"""Unit tests for the independent deterministic GroundTruthSolver."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from visual_task_factory.ground_truth_solver import (
    ClearanceQuery,
    ContainmentMode,
    GroundTruthSolver,
    ReachabilityQuery,
    RegionCountingQuery,
    UnsolvableTaskError,
    UnsupportedTaskError,
    solve_scene,
)
from visual_task_factory.models import (
    AnswerType,
    BenchmarkItem,
    Doorway,
    EntityType,
    ExitMarker,
    Point2D,
    QuestionFamily,
    Rect2D,
    ReferenceAnswer,
    ScaleBarSpec,
    SceneEntity,
    SceneSpec,
    VariationConfig,
    VisualQuestion,
    WallSegment,
    Zone,
    ZoneType,
)
from visual_task_factory.scene_generator import generate_scene


def _base_variation(seed: int = 1) -> VariationConfig:
    return VariationConfig(
        seed=seed,
        topology_id="handcrafted_test_v1",
        scale_meters_per_unit=0.5,
    )


def _handcrafted_reachability_scene(
    *,
    block_upper_door: bool = False,
    block_lower_door: bool = False,
    add_obstacle_blocking_e1_corridor: bool = False,
    seal_start_with_hazard: bool = False,
) -> SceneSpec:
    """20x14 warehouse split vertically at x=10 with doors D1 (y=3) and D2 (y=11)."""
    walls = [
        WallSegment(id="W_N", start=Point2D(x=0.5, y=0.5), end=Point2D(x=19.5, y=0.5), thickness=0.4),
        WallSegment(id="W_S", start=Point2D(x=0.5, y=13.5), end=Point2D(x=19.5, y=13.5), thickness=0.4),
        WallSegment(id="W_W", start=Point2D(x=0.5, y=0.5), end=Point2D(x=0.5, y=13.5), thickness=0.4),
        WallSegment(id="W_E", start=Point2D(x=19.5, y=0.5), end=Point2D(x=19.5, y=13.5), thickness=0.4),
        WallSegment(id="W_DIV", start=Point2D(x=10.0, y=0.5), end=Point2D(x=10.0, y=13.5), thickness=0.4),
    ]
    doors = [
        Doorway(
            id="D_UP",
            position=Point2D(x=10.0, y=3.0),
            width=2.0,
            orientation="vertical",
            is_blocked=block_upper_door,
        ),
        Doorway(
            id="D_DOWN",
            position=Point2D(x=10.0, y=11.0),
            width=2.0,
            orientation="vertical",
            is_blocked=block_lower_door,
        ),
    ]
    zones = [
        Zone(
            id="Z_START",
            label="Start Staging",
            zone_type=ZoneType.STAGING,
            bounds=Rect2D(x=2.0, y=1.5, width=4.0, height=3.0),
        ),
    ]
    if seal_start_with_hazard:
        zones.append(
            Zone(
                id="Z_HAZARD_SEAL",
                label="Hazard Seal",
                zone_type=ZoneType.HAZARD,
                bounds=Rect2D(x=6.5, y=0.5, width=3.0, height=13.0),
            )
        )

    entities: list[SceneEntity] = []
    if add_obstacle_blocking_e1_corridor:
        # Solid wall-to-wall barrier across the right-upper corridor before E1.
        entities.append(
            SceneEntity(
                id="ENT_BLOCK",
                label="Barrier",
                entity_type=EntityType.CONVEYOR,
                bounds=Rect2D(x=14.0, y=0.8, width=1.5, height=12.4),
                passable=False,
            )
        )

    exits = [
        ExitMarker(id="E1", label="Exit E1", position=Point2D(x=19.5, y=3.0), width=2.0),
        ExitMarker(id="E2", label="Exit E2", position=Point2D(x=19.5, y=11.0), width=2.0),
    ]

    return SceneSpec(
        scene_id="scene_reach_test",
        width=20.0,
        height=14.0,
        grid_resolution=0.5,
        scale_bar=ScaleBarSpec(world_units=4.0, display_meters=2.0),
        variation=_base_variation(),
        walls=walls,
        doors=doors,
        zones=zones,
        exits=exits,
        entities=entities,
    )


def test_reachability_multiple_exits_and_blocked_door_reroute() -> None:
    solver = GroundTruthSolver()

    # With D_UP open, E1 (y=3.0) is directly aligned with Z_START (center (4, 3)) and closer than E2 (y=11.0).
    scene_open = _handcrafted_reachability_scene(block_upper_door=False)
    res_open = solver.solve_reachability(
        scene_open, ReachabilityQuery(start_zone_id="Z_START")
    )
    assert res_open.is_reachable is True
    assert res_open.shortest_exit_id == "E1"
    assert res_open.reachable_exit_distances_m["E1"] < res_open.reachable_exit_distances_m["E2"]
    assert isinstance(res_open.reference_answer, ReferenceAnswer)
    assert res_open.reference_answer.answer_type == AnswerType.EXIT_ID
    assert res_open.reference_answer.value == "E1"

    # Blocking D_UP forces path down through D_DOWN (y=11.0), making E2 strictly closer than detouring back up to E1.
    scene_blocked_up = _handcrafted_reachability_scene(block_upper_door=True)
    res_blocked_up = solver.solve_reachability(
        scene_blocked_up, ReachabilityQuery(start_zone_id="Z_START")
    )
    assert res_blocked_up.is_reachable is True
    assert res_blocked_up.shortest_exit_id == "E2"
    assert res_blocked_up.reachable_exit_distances_m["E2"] < res_blocked_up.reachable_exit_distances_m["E1"]


def test_reachability_unreachable_when_doors_or_obstacles_or_hazards_block() -> None:
    solver = GroundTruthSolver()
    query = ReachabilityQuery(start_zone_id="Z_START")

    # Case 1: Both doors blocked.
    scene_no_doors = _handcrafted_reachability_scene(block_upper_door=True, block_lower_door=True)
    res_doors = solver.solve_reachability(scene_no_doors, query)
    assert res_doors.is_reachable is False
    assert res_doors.shortest_exit_id is None
    assert res_doors.reference_answer is None
    assert set(res_doors.unreachable_exit_ids) == {"E1", "E2"}
    with pytest.raises(UnsolvableTaskError, match="No candidate exit is reachable"):
        solver.solve(scene_no_doors, query)

    # Case 2: Solid obstacle spans floor blocking access to exits.
    scene_obstacle = _handcrafted_reachability_scene(add_obstacle_blocking_e1_corridor=True)
    res_obs = solver.solve_reachability(scene_obstacle, query)
    assert res_obs.is_reachable is False
    assert set(res_obs.unreachable_exit_ids) == {"E1", "E2"}

    # Case 3: Hazard zone seals off all exits.
    scene_hazard = _handcrafted_reachability_scene(seal_start_with_hazard=True)
    res_haz = solver.solve_reachability(scene_hazard, query)
    assert res_haz.is_reachable is False


def test_region_containment_counting_strict_center_and_overlap_modes() -> None:
    scene = SceneSpec(
        scene_id="scene_count_test",
        width=30.0,
        height=20.0,
        grid_resolution=0.5,
        scale_bar=ScaleBarSpec(world_units=4.0, display_meters=2.0),
        variation=_base_variation(),
        zones=[
            Zone(
                id="Z_STORAGE",
                label="Storage",
                zone_type=ZoneType.STORAGE,
                bounds=Rect2D(x=5.0, y=5.0, width=10.0, height=10.0),  # [5, 15] x [5, 15]
            )
        ],
        exits=[ExitMarker(id="E1", label="Exit E1", position=Point2D(x=29.5, y=10.0))],
        entities=[
            # 1. Strictly inside (touching boundary x=5.0, y=5.0):
            SceneEntity(
                id="ENT_IN_1",
                label="Rack 1",
                entity_type=EntityType.PALLET_RACK,
                bounds=Rect2D(x=5.0, y=5.0, width=3.0, height=2.0),
            ),
            # 2. Strictly inside (interior):
            SceneEntity(
                id="ENT_IN_2",
                label="Rack 2",
                entity_type=EntityType.PALLET_RACK,
                bounds=Rect2D(x=10.0, y=8.0, width=3.0, height=2.0),
            ),
            # 3. Different entity type strictly inside:
            SceneEntity(
                id="ENT_CRATE",
                label="Crate 1",
                entity_type=EntityType.CRATE,
                bounds=Rect2D(x=6.0, y=10.0, width=2.0, height=2.0),
            ),
            # 4. Partial overlap with center INSIDE ([13.5, 16.5], center x=15.0 on boundary):
            SceneEntity(
                id="ENT_PARTIAL_CENTER_IN",
                label="Rack 3",
                entity_type=EntityType.PALLET_RACK,
                bounds=Rect2D(x=13.5, y=11.0, width=3.0, height=2.0),
            ),
            # 5. Partial overlap with center OUTSIDE ([14.5, 18.5], center x=16.5):
            SceneEntity(
                id="ENT_PARTIAL_CENTER_OUT",
                label="Rack 4",
                entity_type=EntityType.PALLET_RACK,
                bounds=Rect2D(x=14.5, y=13.5, width=4.0, height=1.5),
            ),
            # 6. Completely outside:
            SceneEntity(
                id="ENT_OUT",
                label="Rack 5",
                entity_type=EntityType.PALLET_RACK,
                bounds=Rect2D(x=20.0, y=5.0, width=3.0, height=2.0),
            ),
        ],
    )

    solver = GroundTruthSolver()

    # Strict inside, all entity types -> ENT_IN_1, ENT_IN_2, ENT_CRATE = 3
    res_all_strict = solver.solve_region_counting(
        scene, RegionCountingQuery(zone_id="Z_STORAGE", containment_mode=ContainmentMode.STRICT_INSIDE)
    )
    assert res_all_strict.count == 3
    assert res_all_strict.matched_entity_ids == ["ENT_IN_1", "ENT_IN_2", "ENT_CRATE"]
    assert res_all_strict.reference_answer.answer_type == AnswerType.INTEGER
    assert res_all_strict.reference_answer.value == 3

    # Strict inside, pallet_rack only -> ENT_IN_1, ENT_IN_2 = 2
    res_racks_strict = solver.solve_region_counting(
        scene,
        RegionCountingQuery(
            zone_id="Z_STORAGE",
            entity_types=(EntityType.PALLET_RACK,),
            containment_mode=ContainmentMode.STRICT_INSIDE,
        ),
    )
    assert res_racks_strict.count == 2

    # Center inside, pallet_rack only -> adds ENT_PARTIAL_CENTER_IN = 3
    res_racks_center = solver.solve_region_counting(
        scene,
        RegionCountingQuery(
            zone_id="Z_STORAGE",
            entity_types=(EntityType.PALLET_RACK,),
            containment_mode=ContainmentMode.CENTER_INSIDE,
        ),
    )
    assert res_racks_center.count == 3

    # Any overlap, pallet_rack only -> adds ENT_PARTIAL_CENTER_OUT = 4
    res_racks_overlap = solver.solve_region_counting(
        scene,
        RegionCountingQuery(
            zone_id="Z_STORAGE",
            entity_types=(EntityType.PALLET_RACK,),
            containment_mode=ContainmentMode.ANY_OVERLAP,
        ),
    )
    assert res_racks_overlap.count == 4


def test_scale_grounded_clearance_numeric_boolean_touching_and_tolerance() -> None:
    # Scale: 4 world units = 2.0 meters => 0.5 meters per world unit.
    scene = SceneSpec(
        scene_id="scene_clearance_test",
        width=30.0,
        height=20.0,
        grid_resolution=0.5,
        scale_bar=ScaleBarSpec(world_units=4.0, display_meters=2.0),
        variation=_base_variation(),
        exits=[ExitMarker(id="E1", label="Exit E1", position=Point2D(x=29.5, y=10.0))],
        entities=[
            # R1 spans x=[2, 6], y=[4, 6]
            SceneEntity(
                id="R1",
                label="Rack 1",
                entity_type=EntityType.PALLET_RACK,
                bounds=Rect2D(x=2.0, y=4.0, width=4.0, height=2.0),
            ),
            # R2 spans x=[9, 13], y=[4, 6] -> horizontal gap = 3.0 world units = 1.5 meters
            SceneEntity(
                id="R2",
                label="Rack 2",
                entity_type=EntityType.PALLET_RACK,
                bounds=Rect2D(x=9.0, y=4.0, width=4.0, height=2.0),
            ),
            # R3 touches R1 at x=6.0, y=[4, 6] -> gap = 0.0 meters
            SceneEntity(
                id="R3",
                label="Rack 3",
                entity_type=EntityType.CRATE,
                bounds=Rect2D(x=6.0, y=4.0, width=2.0, height=2.0),
            ),
        ],
    )

    solver = GroundTruthSolver()

    # Numeric clearance with custom tolerance
    num_res = solver.solve_clearance(
        scene,
        ClearanceQuery(
            source_id="R1",
            target_id="R2",
            answer_type=AnswerType.NUMERIC,
            numeric_tolerance_meters=0.02,
        ),
    )
    assert num_res.distance_world_units == pytest.approx(3.0)
    assert num_res.distance_meters == pytest.approx(1.5)
    assert num_res.touching_or_overlapping is False
    assert num_res.reference_answer.answer_type == AnswerType.NUMERIC
    assert num_res.reference_answer.value == pytest.approx(1.5)
    assert num_res.reference_answer.unit == "m"
    assert num_res.reference_answer.numeric_tolerance == pytest.approx(0.02)

    # Boolean exact boundary equality (1.5 m >= 1.5 m -> True)
    bool_eq = solver.solve_clearance(
        scene,
        ClearanceQuery(
            source_id="R1",
            target_id="R2",
            answer_type=AnswerType.BOOLEAN,
            min_required_meters=1.5,
        ),
    )
    assert bool_eq.reference_answer.answer_type == AnswerType.BOOLEAN
    assert bool_eq.reference_answer.value is True

    # Boolean threshold failure (1.5 m >= 1.6 m -> False)
    bool_fail = solver.solve_clearance(
        scene,
        ClearanceQuery(
            source_id="R1",
            target_id="R2",
            answer_type=AnswerType.BOOLEAN,
            min_required_meters=1.6,
        ),
    )
    assert bool_fail.reference_answer.value is False

    # Touching entities -> distance 0.0 m and touching_or_overlapping=True
    touch_res = solver.solve_clearance(
        scene,
        ClearanceQuery(source_id="R1", target_id="R3", answer_type=AnswerType.NUMERIC),
    )
    assert touch_res.distance_meters == pytest.approx(0.0)
    assert touch_res.touching_or_overlapping is True


def test_determinism_immutability_and_error_handling() -> None:
    scene = _handcrafted_reachability_scene()
    before_json = scene.model_dump_json()

    query = ReachabilityQuery(start_zone_id="Z_START")
    ans_1 = solve_scene(scene, query)
    ans_2 = solve_scene(scene, query)

    assert ans_1 == ans_2
    assert scene.model_dump_json() == before_json

    solver = GroundTruthSolver()

    # Unknown zone / entity IDs raise UnsolvableTaskError
    with pytest.raises(UnsolvableTaskError, match="Unknown zone_id"):
        solver.solve(scene, RegionCountingQuery(zone_id="Z_MISSING"))

    with pytest.raises(UnsolvableTaskError, match="not found"):
        solver.solve(
            scene,
            ClearanceQuery(source_id="NO_SUCH_A", target_id="E1", answer_type=AnswerType.NUMERIC),
        )

    # Same source and target rejected by query validation
    with pytest.raises(ValidationError, match="distinct"):
        ClearanceQuery(source_id="E1", target_id="E1", answer_type=AnswerType.NUMERIC)

    # Unsupported query object rejected clearly
    with pytest.raises(UnsupportedTaskError, match="Unsupported solver query"):
        solver.solve(scene, "unsupported_query")  # type: ignore[arg-type]

    # Invalid scene type rejected clearly
    with pytest.raises(TypeError, match="Expected SceneSpec"):
        solver.solve({"not": "scene"}, query)  # type: ignore[arg-type]


def test_solver_integrates_with_generated_scene_and_benchmark_item() -> None:
    scene = generate_scene(
        VariationConfig(
            seed=101,
            topology_id="warehouse_split_aisle_v1",
            scale_meters_per_unit=0.5,
            min_obstacles=3,
            max_obstacles=8,
        )
    )
    solver = GroundTruthSolver()

    reach_ans = solver.solve(scene, ReachabilityQuery(start_zone_id="Z_STORAGE_1"))
    count_ans = solver.solve(
        scene,
        RegionCountingQuery(
            zone_id="Z_STORAGE_1",
            entity_types=(EntityType.PALLET_RACK,),
        ),
    )
    clear_ans = solver.solve(
        scene,
        ClearanceQuery(
            source_id=scene.entities[0].id,
            target_id=scene.entities[1].id,
            answer_type=AnswerType.NUMERIC,
        ),
    )

    q_reach = VisualQuestion(
        question_id="q_int_reach",
        scene_id=scene.scene_id,
        family=QuestionFamily.CONSTRAINED_REACHABILITY,
        template_id="reach_test_v1",
        prompt_text="Which exit is reachable via the shortest valid route from Storage Bay A?",
        expected_answer_type=AnswerType.EXIT_ID,
        reference_answer=reach_ans,
    )
    BenchmarkItem(scene=scene, question=q_reach)
    assert count_ans.answer_type == AnswerType.INTEGER
    assert clear_ans.answer_type == AnswerType.NUMERIC
    assert float(clear_ans.value) > 0.0
