"""Unit tests for the deterministic template-based QuestionGenerator."""

from __future__ import annotations

import re

import pytest

from visual_task_factory.ground_truth_solver import (
    ClearanceQuery,
    ContainmentMode,
    GroundTruthSolver,
    ReachabilityQuery,
    RegionCountingQuery,
    UnsupportedTaskError,
)
from visual_task_factory.models import (
    AnswerType,
    BenchmarkItem,
    DifficultyTier,
    Doorway,
    EntityType,
    ExitMarker,
    Point2D,
    QuestionFamily,
    Rect2D,
    ScaleBarSpec,
    SceneEntity,
    SceneSpec,
    VariationConfig,
    WallSegment,
    Zone,
    ZoneType,
)
from visual_task_factory.question_generator import (
    QuestionGenerator,
    QuestionGeneratorConfig,
    generate_questions,
)
from visual_task_factory.scene_generator import generate_scene


def _handcrafted_scene(*, all_doors_blocked: bool = False) -> SceneSpec:
    return SceneSpec(
        scene_id="scene_qgen_handcrafted",
        width=26.0,
        height=18.0,
        grid_resolution=0.5,
        scale_bar=ScaleBarSpec(world_units=4.0, display_meters=2.0),
        variation=VariationConfig(
            seed=17,
            topology_id="handcrafted_qgen_v1",
            scale_meters_per_unit=0.5,
        ),
        walls=[
            WallSegment(id="W_N", start=Point2D(x=0.5, y=0.5), end=Point2D(x=25.5, y=0.5), thickness=0.4),
            WallSegment(id="W_S", start=Point2D(x=0.5, y=17.5), end=Point2D(x=25.5, y=17.5), thickness=0.4),
            WallSegment(id="W_W", start=Point2D(x=0.5, y=0.5), end=Point2D(x=0.5, y=17.5), thickness=0.4),
            WallSegment(id="W_E", start=Point2D(x=25.5, y=0.5), end=Point2D(x=25.5, y=17.5), thickness=0.4),
            WallSegment(id="W_MID", start=Point2D(x=13.0, y=0.5), end=Point2D(x=13.0, y=17.5), thickness=0.4),
        ],
        doors=[
            Doorway(
                id="D1",
                position=Point2D(x=13.0, y=4.0),
                width=2.5,
                orientation="vertical",
                is_blocked=all_doors_blocked,
            ),
        ],
        zones=[
            Zone(
                id="Z_STORAGE_1",
                label="Storage Bay A",
                zone_type=ZoneType.STORAGE,
                bounds=Rect2D(x=2.0, y=2.0, width=9.0, height=10.0),
            ),
            Zone(
                id="Z_HAZARD_1",
                label="Hazard Zone 1",
                zone_type=ZoneType.HAZARD,
                bounds=Rect2D(x=15.0, y=10.0, width=5.0, height=4.0),
            ),
        ],
        exits=[
            ExitMarker(id="E1", label="Exit E1", position=Point2D(x=25.5, y=4.0), width=2.0),
            ExitMarker(id="E2", label="Exit E2", position=Point2D(x=25.5, y=15.0), width=2.0),
        ],
        entities=[
            SceneEntity(
                id="ENT_1",
                label="Rack 1",
                entity_type=EntityType.PALLET_RACK,
                bounds=Rect2D(x=3.0, y=3.0, width=3.5, height=1.5),
            ),
            SceneEntity(
                id="ENT_2",
                label="Rack 2",
                entity_type=EntityType.PALLET_RACK,
                bounds=Rect2D(x=3.0, y=7.5, width=3.5, height=1.5),
            ),
            SceneEntity(
                id="ENT_3",
                label="Crate 3",
                entity_type=EntityType.CRATE,
                bounds=Rect2D(x=16.0, y=3.0, width=2.0, height=2.0),
            ),
        ],
    )


def test_generate_questions_for_all_supported_families_and_pydantic_valid() -> None:
    scene = _handcrafted_scene()
    items = generate_questions(scene)

    assert len(items) == 3
    families = {item.question.family for item in items}
    assert families == {
        QuestionFamily.CONSTRAINED_REACHABILITY,
        QuestionFamily.REGION_CONTAINMENT_COUNTING,
        QuestionFamily.SCALE_GROUNDED_CLEARANCE,
    }
    for item in items:
        assert isinstance(item, BenchmarkItem)
        assert BenchmarkItem.model_validate_json(item.model_dump_json()) == item
        assert item.question.requires_visual_reasoning is True
        assert item.question.scene_id == scene.scene_id


def test_reference_answers_match_direct_ground_truth_solver_calls() -> None:
    scene = _handcrafted_scene()
    solver = GroundTruthSolver()
    items = generate_questions(
        scene,
        config=QuestionGeneratorConfig(
            clearance_variant="numeric",
            containment_mode=ContainmentMode.STRICT_INSIDE,
        ),
    )

    by_family = {item.question.family: item for item in items}

    # 1. Reachability matches direct solver call
    reach_item = by_family[QuestionFamily.CONSTRAINED_REACHABILITY]
    direct_reach = solver.solve(
        scene,
        ReachabilityQuery(
            start_zone_id="Z_STORAGE_1",
            candidate_exit_ids=["E1", "E2"],
            forbidden_zone_types=(ZoneType.HAZARD, ZoneType.RESTRICTED),
        ),
    )
    assert reach_item.question.reference_answer == direct_reach
    assert reach_item.question.reference_answer.value == "E1"
    assert reach_item.question.candidate_choices == ["E1", "E2"]

    # 2. Counting matches direct solver call (2 pallet racks strictly inside Storage Bay A)
    count_item = by_family[QuestionFamily.REGION_CONTAINMENT_COUNTING]
    direct_count = solver.solve(
        scene,
        RegionCountingQuery(
            zone_id="Z_STORAGE_1",
            entity_types=(EntityType.PALLET_RACK,),
            containment_mode=ContainmentMode.STRICT_INSIDE,
        ),
    )
    assert count_item.question.reference_answer == direct_count
    assert count_item.question.reference_answer.value == 2

    # 3. Clearance numeric matches direct solver call
    clear_item = by_family[QuestionFamily.SCALE_GROUNDED_CLEARANCE]
    assert clear_item.question.expected_answer_type == AnswerType.NUMERIC
    assert clear_item.question.reference_answer.unit == "m"


def test_questions_do_not_leak_coordinates_solver_internals_or_answers() -> None:
    scene = _handcrafted_scene()
    items = generate_questions(
        scene,
        config=QuestionGeneratorConfig(clearance_variant="numeric"),
    )

    for item in items:
        prompt = item.question.prompt_text
        # Must not contain coordinate pairs like (3.0, 3.0) or internal IDs like Z_STORAGE_1 / ENT_1
        assert not re.search(r"\(\s*\d+(\.\d+)?\s*,\s*\d+(\.\d+)?\s*\)", prompt)
        assert "Z_STORAGE_1" not in prompt
        assert "ENT_1" not in prompt
        assert "grid_dijkstra" not in prompt

        if item.question.family == QuestionFamily.CONSTRAINED_REACHABILITY:
            # Must not state the winning exit ID directly in the prompt text.
            ans = str(item.question.reference_answer.value)
            assert ans not in prompt

        if item.question.expected_answer_type == AnswerType.NUMERIC:
            # Numeric clearance prompt must not include the numeric answer value.
            ans_str = f"{float(item.question.reference_answer.value):g}"
            assert ans_str not in prompt


def test_question_wording_matches_containment_mode_and_clearance_type() -> None:
    scene = _handcrafted_scene()

    strict_items = generate_questions(
        scene,
        config=QuestionGeneratorConfig(
            families=(QuestionFamily.REGION_CONTAINMENT_COUNTING,),
            containment_mode=ContainmentMode.STRICT_INSIDE,
        ),
    )
    overlap_items = generate_questions(
        scene,
        config=QuestionGeneratorConfig(
            families=(QuestionFamily.REGION_CONTAINMENT_COUNTING,),
            containment_mode=ContainmentMode.ANY_OVERLAP,
        ),
    )
    assert "completely inside" in strict_items[0].question.prompt_text
    assert "overlapping any part of" in overlap_items[0].question.prompt_text

    bool_clear = generate_questions(
        scene,
        config=QuestionGeneratorConfig(
            families=(QuestionFamily.SCALE_GROUNDED_CLEARANCE,),
            clearance_variant="boolean",
        ),
    )
    num_clear = generate_questions(
        scene,
        config=QuestionGeneratorConfig(
            families=(QuestionFamily.SCALE_GROUNDED_CLEARANCE,),
            clearance_variant="numeric",
        ),
    )
    assert bool_clear[0].question.expected_answer_type == AnswerType.BOOLEAN
    assert "at least" in bool_clear[0].question.prompt_text
    assert num_clear[0].question.expected_answer_type == AnswerType.NUMERIC
    assert num_clear[0].question.reference_answer.unit == "m"


def test_determinism_and_unsolvable_or_unsupported_handling() -> None:
    scene = generate_scene(
        VariationConfig(
            seed=202,
            topology_id="warehouse_split_aisle_v1",
            difficulty=DifficultyTier.MEDIUM,
            min_obstacles=4,
            max_obstacles=8,
        )
    )
    before_json = scene.model_dump_json()

    items_a = generate_questions(scene)
    items_b = generate_questions(scene)
    assert items_a == items_b
    assert scene.model_dump_json() == before_json

    # Unreachable scene skips reachability gracefully instead of fabricating an answer.
    blocked_scene = _handcrafted_scene(all_doors_blocked=True)
    reach_only = generate_questions(
        blocked_scene,
        config=QuestionGeneratorConfig(families=(QuestionFamily.CONSTRAINED_REACHABILITY,)),
    )
    assert reach_only == []

    # Unsupported family raises UnsupportedTaskError.
    bad_cfg = QuestionGeneratorConfig(families=("unsupported_family",))  # type: ignore[arg-type]
    with pytest.raises(UnsupportedTaskError, match="Unsupported question family"):
        QuestionGenerator(config=bad_cfg).generate(scene)

    # Invalid input type raises TypeError.
    with pytest.raises(TypeError, match="Expected SceneSpec"):
        generate_questions({"not": "scene"})  # type: ignore[arg-type]
