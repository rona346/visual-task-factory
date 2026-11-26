"""Unit tests for Visual Task Factory Pydantic data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from visual_task_factory.models import (
    AnswerType,
    BenchmarkItem,
    ClutterLevel,
    DifficultyTier,
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
    VisualTheme,
    WallSegment,
    Zone,
    ZoneType,
)


def _sample_variation() -> VariationConfig:
    return VariationConfig(
        seed=42,
        topology_id="warehouse_split_aisle_v1",
        difficulty=DifficultyTier.MEDIUM,
        orientation_deg=90,
        scale_meters_per_unit=0.5,
        theme=VisualTheme.SAFETY_STANDARD,
        clutter_level=ClutterLevel.LOW,
        min_hazard_zones=1,
        min_obstacles=1,
        max_obstacles=6,
    )


def _sample_scene() -> SceneSpec:
    return SceneSpec(
        scene_id="scene_0001",
        width=40.0,
        height=30.0,
        grid_resolution=0.5,
        scale_bar=ScaleBarSpec(world_units=4.0, display_meters=2.0),
        variation=_sample_variation(),
        walls=[
            WallSegment(
                id="W1",
                start=Point2D(x=15.0, y=0.0),
                end=Point2D(x=15.0, y=20.0),
            )
        ],
        doors=[
            Doorway(
                id="D1",
                position=Point2D(x=15.0, y=10.0),
                width=2.0,
                orientation="vertical",
            )
        ],
        zones=[
            Zone(
                id="Z_HAZARD_1",
                label="Hazard A",
                zone_type=ZoneType.HAZARD,
                bounds=Rect2D(x=2.0, y=2.0, width=6.0, height=5.0),
            ),
            Zone(
                id="Z_STORAGE_1",
                label="Storage North",
                zone_type=ZoneType.STORAGE,
                bounds=Rect2D(x=20.0, y=2.0, width=14.0, height=12.0),
            ),
        ],
        exits=[
            ExitMarker(id="E1", label="Exit E1", position=Point2D(x=39.5, y=5.0)),
            ExitMarker(id="E2", label="Exit E2", position=Point2D(x=5.0, y=29.5)),
        ],
        entities=[
            SceneEntity(
                id="R1",
                label="Rack 1",
                entity_type=EntityType.PALLET_RACK,
                bounds=Rect2D(x=22.0, y=4.0, width=4.0, height=2.0),
                passable=False,
            ),
            SceneEntity(
                id="R2",
                label="Rack 2",
                entity_type=EntityType.PALLET_RACK,
                bounds=Rect2D(x=28.0, y=4.0, width=4.0, height=2.0),
                passable=False,
            ),
        ],
    )


def test_variation_config_validation() -> None:
    cfg = _sample_variation()
    assert cfg.seed == 42
    assert cfg.orientation_deg == 90

    with pytest.raises(ValidationError):
        VariationConfig(
            seed=1,
            topology_id="topo",
            min_obstacles=5,
            max_obstacles=2,
        )

    with pytest.raises(ValidationError):
        VariationConfig(
            seed=1,
            topology_id="topo",
            orientation_deg=45,  # type: ignore[arg-type]
        )


def test_scene_spec_valid_and_json_roundtrip() -> None:
    scene = _sample_scene()
    assert scene.scale_bar.meters_per_world_unit == pytest.approx(0.5)

    serialized = scene.model_dump_json()
    restored = SceneSpec.model_validate_json(serialized)
    assert restored == scene


def test_scene_spec_rejects_out_of_bounds_and_overlapping_entities() -> None:
    base = _sample_scene()

    with pytest.raises(ValidationError, match="exceed scene bounds"):
        SceneSpec(
            **{
                **base.model_dump(),
                "zones": [
                    {
                        "id": "Z_OOB",
                        "label": "Out of bounds",
                        "zone_type": "storage",
                        "bounds": {"x": 35.0, "y": 2.0, "width": 10.0, "height": 5.0},
                    }
                ],
            }
        )

    with pytest.raises(ValidationError, match="overlap"):
        SceneSpec(
            **{
                **base.model_dump(),
                "entities": [
                    {
                        "id": "R1",
                        "label": "Rack 1",
                        "entity_type": "pallet_rack",
                        "bounds": {"x": 20.0, "y": 4.0, "width": 4.0, "height": 2.0},
                        "passable": False,
                    },
                    {
                        "id": "R2",
                        "label": "Rack 2",
                        "entity_type": "crate",
                        "bounds": {"x": 23.0, "y": 5.0, "width": 3.0, "height": 2.0},
                        "passable": False,
                    },
                ],
            }
        )


def test_scene_spec_rejects_duplicate_ids_and_zero_length_walls() -> None:
    base = _sample_scene()

    with pytest.raises(ValidationError, match="Duplicate scene element id"):
        SceneSpec(
            **{
                **base.model_dump(),
                "exits": [
                    {"id": "W1", "label": "Collides with wall", "position": {"x": 1.0, "y": 1.0}},
                ],
            }
        )

    with pytest.raises(ValidationError, match="non-zero length"):
        WallSegment(id="W_ZERO", start=Point2D(x=5.0, y=5.0), end=Point2D(x=5.0, y=5.0))


def test_supported_normalized_question_and_answer_types() -> None:
    scene = _sample_scene()

    q_reach = VisualQuestion(
        question_id="q_001",
        scene_id=scene.scene_id,
        family=QuestionFamily.CONSTRAINED_REACHABILITY,
        template_id="reach_shortest_valid_exit_v1",
        prompt_text="Starting from Storage North, which exit is reachable via the shortest route without entering any hazard zone?",
        expected_answer_type=AnswerType.EXIT_ID,
        candidate_choices=["E1", "E2"],
        reference_answer=ReferenceAnswer(
            answer_type=AnswerType.EXIT_ID,
            value="E1",
            solver_name="grid_bfs_v1",
            reasoning_trace=["Start in Z_STORAGE_1", "Route clear to E1"],
        ),
    )

    q_count = VisualQuestion(
        question_id="q_002",
        scene_id=scene.scene_id,
        family=QuestionFamily.REGION_CONTAINMENT_COUNTING,
        template_id="count_racks_in_zone_v1",
        prompt_text="How many pallet racks lie completely inside the Storage North zone?",
        expected_answer_type=AnswerType.INTEGER,
        reference_answer=ReferenceAnswer(
            answer_type=AnswerType.INTEGER,
            value=2,
            solver_name="rect_containment_v1",
        ),
    )

    q_clearance_bool = VisualQuestion(
        question_id="q_003",
        scene_id=scene.scene_id,
        family=QuestionFamily.SCALE_GROUNDED_CLEARANCE,
        template_id="clearance_threshold_check_v1",
        prompt_text="Using the scale bar, is the clearance between Rack 1 and Rack 2 at least 1.5 meters?",
        expected_answer_type=AnswerType.BOOLEAN,
        reference_answer=ReferenceAnswer(
            answer_type=AnswerType.BOOLEAN,
            value=False,
            solver_name="euclidean_clearance_v1",
        ),
    )

    q_clearance_num = VisualQuestion(
        question_id="q_004",
        scene_id=scene.scene_id,
        family=QuestionFamily.SCALE_GROUNDED_CLEARANCE,
        template_id="clearance_distance_meters_v1",
        prompt_text="Using the scale bar, what is the minimum clearance in meters between Rack 1 and Rack 2?",
        expected_answer_type=AnswerType.NUMERIC,
        reference_answer=ReferenceAnswer(
            answer_type=AnswerType.NUMERIC,
            value=1.0,
            unit="m",
            numeric_tolerance=0.05,
            solver_name="euclidean_clearance_v1",
        ),
    )

    for question in (q_reach, q_count, q_clearance_bool, q_clearance_num):
        item = BenchmarkItem(scene=scene, question=question)
        assert item.scene.scene_id == item.question.scene_id


def test_answer_type_strictness_and_family_mismatch_rejected() -> None:
    scene = _sample_scene()

    # Bool must not pass as INTEGER
    with pytest.raises(ValidationError, match="non-negative integer"):
        ReferenceAnswer(
            answer_type=AnswerType.INTEGER,
            value=True,
            solver_name="solver",
        )

    # Int must not pass as BOOLEAN
    with pytest.raises(ValidationError, match="strict bool"):
        ReferenceAnswer(
            answer_type=AnswerType.BOOLEAN,
            value=1,
            solver_name="solver",
        )

    # NUMERIC requires explicit unit='m'
    with pytest.raises(ValidationError, match="unit='m'"):
        ReferenceAnswer(
            answer_type=AnswerType.NUMERIC,
            value=2.5,
            solver_name="solver",
        )

    # Family <-> answer_type mismatch rejected
    with pytest.raises(ValidationError, match="requires expected_answer_type"):
        VisualQuestion(
            question_id="q_bad_family",
            scene_id=scene.scene_id,
            family=QuestionFamily.REGION_CONTAINMENT_COUNTING,
            template_id="count_v1",
            prompt_text="How many pallet racks are in Storage North?",
            expected_answer_type=AnswerType.BOOLEAN,
            reference_answer=ReferenceAnswer(
                answer_type=AnswerType.BOOLEAN,
                value=True,
                solver_name="solver",
            ),
        )

    # BenchmarkItem rejects reference exit not in scene.exits
    bad_exit_question = VisualQuestion(
        question_id="q_bad_exit",
        scene_id=scene.scene_id,
        family=QuestionFamily.CONSTRAINED_REACHABILITY,
        template_id="reach_v1",
        prompt_text="Which exit is closest to Storage North without entering any hazard zone?",
        expected_answer_type=AnswerType.EXIT_ID,
        reference_answer=ReferenceAnswer(
            answer_type=AnswerType.EXIT_ID,
            value="E99",
            solver_name="solver",
        ),
    )
    with pytest.raises(ValidationError, match="does not exist in scene.exits"):
        BenchmarkItem(scene=scene, question=bad_exit_question)
