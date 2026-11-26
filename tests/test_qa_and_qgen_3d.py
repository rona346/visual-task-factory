"""Tests for isolated 3D question generation and ground-truth QA validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from visual_task_factory.models.scene import EntityType, Point2D, ScaleBarSpec
from visual_task_factory.models.scene3d import (
    Box3D,
    CameraPose3D,
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
    GroundTruthQAValidator3D,
    QuestionFamily3D,
    ReferenceAnswer3D,
    VisualQuestion3D,
)
from visual_task_factory.question_generator import QuestionGenerator as QuestionGeneratorV1
from visual_task_factory.question_generator_3d import (
    QuestionGenerator3D,
    QuestionGenerator3DConfig,
)
from visual_task_factory.scene_generator import generate_scene as generate_scene_v1
from visual_task_factory.validator import BenchmarkValidator as BenchmarkValidatorV1


def _rich_scene_3d() -> SceneSpec3D:
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
        bounds=Box3D(x=2.2, y=2.2, z=1.2, width=1.6, depth=1.6, height=0.9),
        supported_by="crate_floor",
    )
    side_rack = SceneEntity3D(
        id="rack_side",
        label="Side Rack",
        entity_type=EntityType.PALLET_RACK,
        bounds=Box3D(x=6.0, y=2.0, z=0.0, width=2.2, depth=2.0, height=3.0),
    )
    hidden_crate = SceneEntity3D(
        id="crate_hidden",
        label="Hidden Crate",
        entity_type=EntityType.CRATE,
        bounds=Box3D(x=10.0, y=2.0, z=0.0, width=1.5, depth=1.5, height=1.0),
    )
    front_rack = SceneEntity3D(
        id="rack_front",
        label="Front Rack",
        entity_type=EntityType.PALLET_RACK,
        bounds=Box3D(x=12.0, y=4.0, z=0.0, width=5.0, depth=5.0, height=4.5),
    )
    return SceneSpec3D(
        scene_id="wh3d_qgen",
        width=20.0,
        depth=16.0,
        ceiling_height=5.0,
        scale_bar=ScaleBarSpec(world_units=2.0, display_meters=2.0),
        variation=VariationConfig(seed=33, topology_id="qgen_3d"),
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
                height=4.5,
            )
        ],
        exits=[ExitMarker3D(id="exit_e", label="Exit E", position=Point3D(x=20.0, y=8.0, z=0.0))],
        entities=[floor_crate, stacked_crate, side_rack, hidden_crate, front_rack],
    )


def test_deterministic_generation_and_ground_truth_validation_for_all_families() -> None:
    scene = _rich_scene_3d()
    snapshot = scene.model_dump()

    cfg = QuestionGenerator3DConfig(max_questions_per_family=2, clearance_variant="both")
    gen = QuestionGenerator3D(config=cfg)
    validator = GroundTruthQAValidator3D()

    items_a = gen.generate(scene)
    items_b = gen.generate(scene)

    assert scene.model_dump() == snapshot
    assert [it.model_dump() for it in items_a] == [it.model_dump() for it in items_b]
    assert len(items_a) >= 4

    families_present = {it.question.family for it in items_a}
    assert families_present == {
        QuestionFamily3D.OBJECT_OVERLAP_3D,
        QuestionFamily3D.VERTICAL_RELATION_3D,
        QuestionFamily3D.VERTICAL_CLEARANCE_3D,
        QuestionFamily3D.CAMERA_VISIBILITY_3D,
    }

    batch_report = validator.validate_batch(items_a)
    assert batch_report.total_items == len(items_a)
    assert batch_report.invalid_items == 0
    assert batch_report.duplicate_question_ids == []
    assert batch_report.duplicate_prompts == []


def test_no_ambiguous_vertical_clearance_generated_and_ambiguous_item_rejected() -> None:
    # Scene with only two floor-level crates with disjoint XY footprints and equal Z ranges.
    scene = SceneSpec3D(
        scene_id="wh3d_ambig",
        width=14.0,
        depth=12.0,
        ceiling_height=4.0,
        scale_bar=ScaleBarSpec(world_units=2.0, display_meters=2.0),
        variation=VariationConfig(seed=5, topology_id="ambig"),
        camera=CameraPose3D(look_at=Point3D(x=7.0, y=6.0, z=1.0), azimuth_deg=45.0, elevation_deg=35.0),
        exits=[ExitMarker3D(id="ex1", label="Ex1", position=Point3D(x=14.0, y=6.0, z=0.0))],
        entities=[
            SceneEntity3D(
                id="crate_a",
                label="Crate A",
                entity_type=EntityType.CRATE,
                bounds=Box3D(x=1.0, y=1.0, z=0.0, width=2.0, depth=2.0, height=1.5),
            ),
            SceneEntity3D(
                id="crate_b",
                label="Crate B",
                entity_type=EntityType.CRATE,
                bounds=Box3D(x=8.0, y=7.0, z=0.0, width=2.0, depth=2.0, height=1.5),
            ),
        ],
    )

    gen = QuestionGenerator3D(
        QuestionGenerator3DConfig(
            families=(QuestionFamily3D.VERTICAL_CLEARANCE_3D,),
            include_ceiling_clearance=False,
        )
    )
    assert gen.generate(scene) == []

    # Craft an ambiguous entity-to-entity clearance question manually and ensure QA rejects it.
    bad_item = BenchmarkItem3D(
        scene=scene,
        question=VisualQuestion3D(
            question_id="q_ambig_clr",
            scene_id=scene.scene_id,
            family=QuestionFamily3D.VERTICAL_CLEARANCE_3D,
            template_id="manual_bad_clr",
            prompt_text="What is the vertical clearance in meters between Crate A and Crate B?",
            expected_answer_type=AnswerType3D.NUMERIC,
            reference_answer=ReferenceAnswer3D(answer_type=AnswerType3D.NUMERIC, value=0.0, unit="m"),
            solver_query={
                "family": "vertical_clearance_3d",
                "source_id": "crate_a",
                "target_id": "crate_b",
                "target_kind": "entity",
                "answer_type": "numeric",
            },
        ),
    )
    report = GroundTruthQAValidator3D().validate(bad_item)
    assert report.is_valid is False
    assert any(f.code == "AMBIGUOUS_OR_UNSUPPORTED_GROUND_TRUTH" for f in report.errors)


def test_validator_rejects_invalid_entity_refs_unsupported_types_wrong_answers_and_duplicates() -> None:
    scene = _rich_scene_3d()
    items = QuestionGenerator3D(QuestionGenerator3DConfig(max_questions_per_family=1)).generate(scene)
    validator = GroundTruthQAValidator3D()

    # 1. Unsupported question family rejected at schema level.
    with pytest.raises(ValidationError):
        VisualQuestion3D(
            question_id="q_bad_fam",
            scene_id=scene.scene_id,
            family="unknown_3d_family",  # type: ignore[arg-type]
            template_id="bad_fam",
            prompt_text="Unsupported 3D question family text.",
            expected_answer_type=AnswerType3D.BOOLEAN,
            reference_answer=ReferenceAnswer3D(answer_type=AnswerType3D.BOOLEAN, value=True),
        )

    # 2. Missing solver_query rejected.
    base_item = items[0]
    missing_query = BenchmarkItem3D(
        scene=scene,
        question=base_item.question.model_copy(update={"solver_query": None}),
    )
    rep_missing = validator.validate(missing_query)
    assert rep_missing.is_valid is False
    assert any(f.code == "MISSING_SOLVER_QUERY" for f in rep_missing.errors)

    # 3. Invalid entity reference rejected.
    bad_ref_query = dict(base_item.question.solver_query or {})
    if "source_id" in bad_ref_query:
        bad_ref_query["source_id"] = "nonexistent_entity"
    else:
        bad_ref_query["target_id"] = "nonexistent_entity"
    bad_ref_item = BenchmarkItem3D(
        scene=scene,
        question=base_item.question.model_copy(update={"solver_query": bad_ref_query}),
    )
    rep_ref = validator.validate(bad_ref_item)
    assert rep_ref.is_valid is False
    assert any(f.code == "UNKNOWN_ENTITY_ID" for f in rep_ref.errors)

    # 4. Corrupted reference answer rejected.
    bool_item = next(it for it in items if it.question.expected_answer_type == AnswerType3D.BOOLEAN)
    flipped = not bool(bool_item.question.reference_answer.value)
    wrong_ans_item = BenchmarkItem3D(
        scene=scene,
        question=bool_item.question.model_copy(
            update={
                "reference_answer": bool_item.question.reference_answer.model_copy(
                    update={"value": flipped}
                )
            }
        ),
    )
    rep_wrong = validator.validate(wrong_ans_item)
    assert rep_wrong.is_valid is False
    assert any(f.code == "REFERENCE_ANSWER_MISMATCH" for f in rep_wrong.errors)

    # 5. Duplicate batch detection.
    dup_batch = validator.validate_batch([bool_item, bool_item])
    assert dup_batch.valid_items == 1
    assert dup_batch.invalid_items == 1
    assert bool_item.question.question_id in dup_batch.duplicate_question_ids
    assert len(dup_batch.duplicate_prompts) == 1


def test_empty_and_minimal_scenes_plus_v1_compatibility() -> None:
    empty_scene = SceneSpec3D(
        scene_id="wh3d_empty",
        width=10.0,
        depth=10.0,
        ceiling_height=4.0,
        scale_bar=ScaleBarSpec(world_units=2.0, display_meters=2.0),
        variation=VariationConfig(seed=1, topology_id="empty"),
        camera=CameraPose3D(look_at=Point3D(x=5.0, y=5.0, z=1.0), azimuth_deg=45.0, elevation_deg=35.0),
        exits=[ExitMarker3D(id="e1", label="E1", position=Point3D(x=10.0, y=5.0, z=0.0))],
        entities=[],
    )
    gen = QuestionGenerator3D()
    assert gen.generate(empty_scene) == []

    # Minimal 1-entity scene: ceiling clearance + visibility succeed cleanly without crashing.
    single_scene = empty_scene.model_copy(
        update={
            "scene_id": "wh3d_single",
            "entities": [
                SceneEntity3D(
                    id="solo_crate",
                    label="Solo Crate",
                    entity_type=EntityType.CRATE,
                    bounds=Box3D(x=4.0, y=4.0, z=0.0, width=2.0, depth=2.0, height=1.5),
                )
            ],
        }
    )
    single_items = gen.generate(single_scene)
    assert len(single_items) >= 1
    assert GroundTruthQAValidator3D().validate_batch(single_items).invalid_items == 0

    # V1 end-to-end scene + question generation + validation remains unaffected.
    v1_scene = generate_scene_v1(VariationConfig(seed=42, topology_id="two_zone_corridor"))
    v1_items = QuestionGeneratorV1().generate(v1_scene)
    assert len(v1_items) >= 1
    v1_batch = BenchmarkValidatorV1().validate_batch(v1_items)
    assert v1_batch.invalid_items == 0
