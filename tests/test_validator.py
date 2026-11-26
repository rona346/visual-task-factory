"""Unit tests for the independent BenchmarkValidator gate."""

from __future__ import annotations

from visual_task_factory.ground_truth_solver import ContainmentMode
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
from visual_task_factory.question_generator import QuestionGeneratorConfig, generate_questions
from visual_task_factory.validator import (
    BatchValidationReport,
    BenchmarkValidator,
    ValidationReport,
    validate_benchmark_batch,
    validate_benchmark_item,
)


def _controlled_scene(*, include_pillar: bool = False) -> SceneSpec:
    entities = [
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
    ]
    if include_pillar:
        entities.append(
            SceneEntity(
                id="ENT_4",
                label="Pillar 4",
                entity_type=EntityType.PILLAR,
                bounds=Rect2D(x=18.0, y=7.0, width=1.0, height=1.0),
            )
        )

    return SceneSpec(
        scene_id="scene_val_test",
        width=26.0,
        height=18.0,
        grid_resolution=0.5,
        scale_bar=ScaleBarSpec(world_units=4.0, display_meters=2.0),
        variation=VariationConfig(
            seed=19,
            topology_id="val_fixture_v1",
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
                is_blocked=False,
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
        entities=entities,
    )


def _get_generated_by_family(scene: SceneSpec) -> dict[QuestionFamily, BenchmarkItem]:
    items = generate_questions(
        scene,
        config=QuestionGeneratorConfig(
            clearance_variant="numeric",
            containment_mode=ContainmentMode.STRICT_INSIDE,
        ),
    )
    return {item.question.family: item for item in items}


def test_valid_generated_benchmark_items_pass() -> None:
    scene = _controlled_scene()
    items = generate_questions(scene)

    assert len(items) == 3
    for item in items:
        report = validate_benchmark_item(item)
        assert isinstance(report, ValidationReport)
        assert report.is_valid is True
        assert report.errors == []
        assert any(f.code == "SOLVER_VERIFIED" for f in report.info)


def test_missing_or_malformed_solver_query_rejected() -> None:
    scene = _controlled_scene()
    base_item = _get_generated_by_family(scene)[QuestionFamily.REGION_CONTAINMENT_COUNTING]

    # Missing solver_query
    missing_q = base_item.question.model_copy(update={"solver_query": None})
    missing_report = validate_benchmark_item(BenchmarkItem(scene=scene, question=missing_q))
    assert missing_report.is_valid is False
    assert any(e.code == "MISSING_SOLVER_QUERY" for e in missing_report.errors)

    # Malformed solver_query (missing required zone_id for RegionCountingQuery)
    malformed_q = base_item.question.model_copy(
        update={"solver_query": {"family": "region_containment_counting"}}
    )
    malformed_report = validate_benchmark_item(BenchmarkItem(scene=scene, question=malformed_q))
    assert malformed_report.is_valid is False
    assert any(e.code == "MALFORMED_SOLVER_QUERY" for e in malformed_report.errors)


def test_query_family_mismatch_and_unsupported_family_rejected() -> None:
    scene = _controlled_scene()
    base_item = _get_generated_by_family(scene)[QuestionFamily.CONSTRAINED_REACHABILITY]

    # Query family mismatch constructed via model_copy (bypassing Pydantic validation)
    bad_q = base_item.question.model_copy(
        update={
            "solver_query": {"family": "region_containment_counting", "zone_id": "Z_STORAGE_1"},
        }
    )
    mismatch_item = BenchmarkItem.model_construct(scene=scene, question=bad_q)
    report = validate_benchmark_item(mismatch_item)
    assert report.is_valid is False
    assert any(e.code == "SCHEMA_VALIDATION_ERROR" for e in report.errors)

    # Unsupported family bypassed via model_copy
    unsupported_q = base_item.question.model_copy(
        update={"family": "unsupported_family"}  # type: ignore[dict-item]
    )
    unsupported_item = BenchmarkItem.model_construct(scene=scene, question=unsupported_q)
    unsupported_report = validate_benchmark_item(unsupported_item)
    assert unsupported_report.is_valid is False
    assert any(e.code == "SCHEMA_VALIDATION_ERROR" for e in unsupported_report.errors)


def test_corrupted_reference_answer_and_mismatched_type_unit_rejected() -> None:
    scene = _controlled_scene()
    by_family = _get_generated_by_family(scene)

    # 1. Wrong exit answer ("E2" instead of "E1")
    reach_item = by_family[QuestionFamily.CONSTRAINED_REACHABILITY]
    bad_exit_ref = reach_item.question.reference_answer.model_copy(update={"value": "E2"})
    bad_reach = BenchmarkItem(
        scene=scene,
        question=reach_item.question.model_copy(update={"reference_answer": bad_exit_ref}),
    )
    reach_report = validate_benchmark_item(bad_reach)
    assert reach_report.is_valid is False
    assert any(e.code == "REFERENCE_ANSWER_MISMATCH" for e in reach_report.errors)

    # 2. Mismatched answer type via model_copy (bypassing validators)
    bad_type_ref = reach_item.question.reference_answer.model_copy(
        update={"answer_type": AnswerType.INTEGER, "value": 1}
    )
    bad_type_q = reach_item.question.model_copy(update={"reference_answer": bad_type_ref})
    type_report = validate_benchmark_item(BenchmarkItem.model_construct(scene=scene, question=bad_type_q))
    assert type_report.is_valid is False
    assert any(e.code == "SCHEMA_VALIDATION_ERROR" for e in type_report.errors)

    # 3. Missing/wrong unit on NUMERIC clearance answer via model_copy
    clear_item = by_family[QuestionFamily.SCALE_GROUNDED_CLEARANCE]
    bad_unit_ref = clear_item.question.reference_answer.model_copy(update={"unit": None})
    bad_unit_q = clear_item.question.model_copy(update={"reference_answer": bad_unit_ref})
    unit_report = validate_benchmark_item(BenchmarkItem.model_construct(scene=scene, question=bad_unit_q))
    assert unit_report.is_valid is False
    assert any(e.code == "SCHEMA_VALIDATION_ERROR" for e in unit_report.errors)


def test_numeric_tolerance_respected_and_exceeding_tolerance_rejected() -> None:
    scene = _controlled_scene()
    clear_item = _get_generated_by_family(scene)[QuestionFamily.SCALE_GROUNDED_CLEARANCE]
    exact_val = float(clear_item.question.reference_answer.value)

    # Slight drift within tolerance (0.02 <= 0.05) passes with warning
    within_ref = clear_item.question.reference_answer.model_copy(
        update={"value": round(exact_val + 0.02, 4), "numeric_tolerance": 0.05}
    )
    within_item = BenchmarkItem(
        scene=scene,
        question=clear_item.question.model_copy(update={"reference_answer": within_ref}),
    )
    within_report = validate_benchmark_item(within_item)
    assert within_report.is_valid is True
    assert any(w.code == "NUMERIC_WITHIN_TOLERANCE_DRIFT" for w in within_report.warnings)

    # Drift exceeding tolerance (0.20 > 0.05) fails
    outside_ref = clear_item.question.reference_answer.model_copy(
        update={"value": round(exact_val + 0.20, 4), "numeric_tolerance": 0.05}
    )
    outside_item = BenchmarkItem(
        scene=scene,
        question=clear_item.question.model_copy(update={"reference_answer": outside_ref}),
    )
    outside_report = validate_benchmark_item(outside_item)
    assert outside_report.is_valid is False
    assert any(e.code == "NUMERIC_ANSWER_MISMATCH" for e in outside_report.errors)


def test_invalid_scene_reference_and_duplicate_candidate_choices_detected() -> None:
    scene = _controlled_scene()
    by_family = _get_generated_by_family(scene)

    # Invalid zone reference in solver_query
    count_item = by_family[QuestionFamily.REGION_CONTAINMENT_COUNTING]
    bad_zone_q = count_item.question.model_copy(
        update={
            "solver_query": {
                **count_item.question.solver_query,  # type: ignore[dict-item]
                "zone_id": "Z_MISSING",
            }
        }
    )
    zone_report = validate_benchmark_item(BenchmarkItem(scene=scene, question=bad_zone_q))
    assert zone_report.is_valid is False
    assert any(e.code == "UNKNOWN_TARGET_ZONE" for e in zone_report.errors)

    # Duplicate candidate choices bypassed via model_copy
    reach_item = by_family[QuestionFamily.CONSTRAINED_REACHABILITY]
    dup_choice_q = reach_item.question.model_copy(
        update={"candidate_choices": ["E1", "E1"]}
    )
    dup_choice_report = validate_benchmark_item(
        BenchmarkItem.model_construct(scene=scene, question=dup_choice_q)
    )
    assert dup_choice_report.is_valid is False
    assert any(e.code == "SCHEMA_VALIDATION_ERROR" for e in dup_choice_report.errors)


def test_leakage_and_non_blocking_warnings() -> None:
    scene = _controlled_scene(include_pillar=True)
    by_family = _get_generated_by_family(scene)

    # 1. Answer leakage in reachability prompt ("E1" exposed in text)
    reach_item = by_family[QuestionFamily.CONSTRAINED_REACHABILITY]
    leaked_q = reach_item.question.model_copy(
        update={
            "prompt_text": "Starting from the Storage Bay A zone, confirm why E1 is the shortest valid exit?"
        }
    )
    leak_report = validate_benchmark_item(BenchmarkItem(scene=scene, question=leaked_q))
    assert leak_report.is_valid is False
    assert any(e.code == "ANSWER_LEAKAGE" for e in leak_report.errors)

    # 2. Coordinate leakage
    coord_q = reach_item.question.model_copy(
        update={
            "prompt_text": "Starting from the Storage Bay A zone at (3.0, 4.0), which exit is closest?"
        }
    )
    coord_report = validate_benchmark_item(BenchmarkItem(scene=scene, question=coord_q))
    assert coord_report.is_valid is False
    assert any(e.code == "COORDINATE_LEAKAGE" for e in coord_report.errors)

    # 3. Valid item with unlabeled single pillar triggers WARNING without making item invalid
    pillar_item = BenchmarkItem(
        scene=scene,
        question=VisualQuestion(
            question_id="q_pillar_warn",
            scene_id=scene.scene_id,
            family=QuestionFamily.SCALE_GROUNDED_CLEARANCE,
            template_id="clearance_distance_numeric_v1",
            prompt_text="Using the scale bar, what is the minimum edge-to-edge clearance in meters between Rack 1 and Pillar 4?",
            expected_answer_type=AnswerType.NUMERIC,
            reference_answer=ReferenceAnswer(
                answer_type=AnswerType.NUMERIC,
                value=5.868347,
                unit="m",
                numeric_tolerance=0.05,
                solver_name="euclidean_clearance_v1",
            ),
            solver_query={
                "family": "scale_grounded_clearance",
                "source_id": "ENT_1",
                "target_id": "ENT_4",
                "answer_type": "numeric",
                "numeric_tolerance_meters": 0.05,
            },
        ),
    )
    warn_report = validate_benchmark_item(pillar_item)
    assert warn_report.is_valid is True
    assert any(w.code == "UNLABELED_ELEMENT_IN_RENDER" for w in warn_report.warnings)


def test_determinism_immutability_and_batch_validation() -> None:
    scene = _controlled_scene()
    items = generate_questions(scene)
    before_json = [item.model_dump_json() for item in items]

    validator = BenchmarkValidator()
    r1 = validator.validate(items[0])
    r2 = validator.validate(items[0])
    assert r1 == r2
    assert [item.model_dump_json() for item in items] == before_json

    # Batch validation: 3 valid items + 1 duplicate question_id & prompt + 1 invalid answer
    dup_item = items[0].model_copy(deep=True)
    bad_ref = items[1].question.reference_answer.model_copy(update={"value": 99})
    bad_item = BenchmarkItem(
        scene=scene,
        question=items[1].question.model_copy(
            update={"question_id": "q_unique_but_wrong", "reference_answer": bad_ref}
        ),
    )

    batch_report = validate_benchmark_batch([*items, dup_item, bad_item])
    assert isinstance(batch_report, BatchValidationReport)
    assert batch_report.total_items == 5
    assert batch_report.valid_items == 3
    assert batch_report.invalid_items == 2
    assert batch_report.warning_count >= 1
    assert items[0].question.question_id in batch_report.duplicate_question_ids
    assert items[0].question.prompt_text in batch_report.duplicate_prompts
