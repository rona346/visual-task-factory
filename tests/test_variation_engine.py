"""Unit tests for the Meaningful Variation Engine and structural scene signatures."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from visual_task_factory.ground_truth_solver import (
    GroundTruthSolver,
    RegionCountingQuery,
)
from visual_task_factory.models import (
    ClutterLevel,
    DifficultyTier,
    SceneSpec,
    VariationConfig,
    VisualTheme,
)
from visual_task_factory.question_generator import generate_questions
from visual_task_factory.renderer import render_scene
from visual_task_factory.scene_generator import SUPPORTED_TOPOLOGIES, generate_scene
from visual_task_factory.validator import validate_benchmark_batch
from visual_task_factory.variation_engine import (
    VariationEngine,
    VariationPlanConfig,
    compare_scenes,
    compute_scene_signature,
    generate_varied_scenes,
)


def test_same_seed_and_plan_produce_reproducible_results() -> None:
    plan = VariationPlanConfig(base_seed=300, count=6)
    res_a = generate_varied_scenes(plan)
    res_b = generate_varied_scenes(plan)

    assert res_a == res_b
    assert res_a.model_dump_json() == res_b.model_dump_json()


def test_structural_signature_ignores_metadata_only_changes() -> None:
    base_cfg = VariationConfig(
        seed=50,
        topology_id="warehouse_split_aisle_v1",
        difficulty=DifficultyTier.MEDIUM,
        theme=VisualTheme.SAFETY_STANDARD,
    )
    scene_a = generate_scene(base_cfg, scene_id="scene_alpha")
    # Same geometry, only ID/theme metadata changed.
    scene_meta_only = scene_a.model_copy(
        update={
            "scene_id": "scene_beta",
            "variation": base_cfg.model_copy(update={"theme": VisualTheme.BLUEPRINT}),
        }
    )

    diff_meta = compare_scenes(scene_a, scene_meta_only)
    assert diff_meta.is_meaningfully_different is False
    assert diff_meta.changed_dimensions == []
    assert diff_meta.signature_a.structural_hash == diff_meta.signature_b.structural_hash


def test_different_seeds_and_topologies_produce_meaningful_differences() -> None:
    scene_s1 = generate_scene(
        VariationConfig(seed=11, topology_id="warehouse_split_aisle_v1")
    )
    scene_s2 = generate_scene(
        VariationConfig(seed=77, topology_id="warehouse_split_aisle_v1")
    )
    diff_seeds = compare_scenes(scene_s1, scene_s2)
    assert diff_seeds.is_meaningfully_different is True
    assert len(diff_seeds.changed_dimensions) >= 2

    scene_corr = generate_scene(
        VariationConfig(seed=11, topology_id="warehouse_corridor_spine_v1")
    )
    scene_cross = generate_scene(
        VariationConfig(seed=11, topology_id="warehouse_cross_aisle_v1")
    )
    diff_topo = compare_scenes(scene_corr, scene_cross)
    assert diff_topo.is_meaningfully_different is True
    assert "wall_geometry" in diff_topo.changed_dimensions or "wall_topology" in diff_topo.changed_dimensions


def test_variation_respects_config_difficulty_and_invalid_configs_fail() -> None:
    plan = VariationPlanConfig(
        base_seed=10,
        count=3,
        topologies=SUPPORTED_TOPOLOGIES,
        difficulties=(DifficultyTier.EASY, DifficultyTier.MEDIUM, DifficultyTier.HARD),
        min_obstacles=2,
        max_obstacles=7,
        clutter_levels=(ClutterLevel.NONE, ClutterLevel.HIGH),
    )
    result = generate_varied_scenes(plan)

    assert len(result.items) == 3
    assert result.difficulty_coverage == {"easy": 1, "medium": 1, "hard": 1}
    assert set(result.topology_coverage.keys()) == set(SUPPORTED_TOPOLOGIES)
    for item in result.items:
        assert isinstance(item.scene, SceneSpec)
        assert 2 <= len(item.scene.entities) <= 7

    with pytest.raises(ValidationError, match="max_obstacles"):
        VariationPlanConfig(min_obstacles=8, max_obstacles=3)

    with pytest.raises(ValidationError, match="at least one entry"):
        VariationPlanConfig(topologies=())

    with pytest.raises(ValidationError, match="scales_meters_per_unit"):
        VariationPlanConfig(scales_meters_per_unit=(0.0,))


def test_varied_scenes_integrate_with_renderer_solver_qgen_and_validator(
    tmp_path: Path,
) -> None:
    plan = VariationPlanConfig(
        base_seed=42,
        count=6,
        min_obstacles=3,
        max_obstacles=9,
    )
    plan_snapshot = plan.model_dump_json()
    result = generate_varied_scenes(plan)

    # Plan input must remain immutable.
    assert plan.model_dump_json() == plan_snapshot

    solver = GroundTruthSolver()
    all_questions = []
    scene_snapshots = [item.scene.model_dump_json() for item in result.items]

    for item in result.items:
        # 1. Compatible with renderer
        png_path = tmp_path / f"{item.scene.scene_id}.png"
        render_res = render_scene(item.scene, png_path)
        assert render_res.output_path.is_file()

        # 2. Compatible with GroundTruthSolver
        count_res = solver.solve_region_counting(
            item.scene,
            RegionCountingQuery(zone_id="Z_STORAGE_1"),
        )
        assert count_res.count >= 0

        # 3. Compatible with QuestionGenerator
        questions = generate_questions(item.scene)
        assert len(questions) >= 2
        all_questions.extend(questions)

    # 4. Compatible with BenchmarkValidator
    batch_report = validate_benchmark_batch(all_questions)
    assert batch_report.total_items == len(all_questions)
    assert batch_report.invalid_items == 0

    # Generated scenes were not mutated by downstream modules.
    assert [item.scene.model_dump_json() for item in result.items] == scene_snapshots


def test_multi_seed_sweep_and_duplicate_rejection_behavior() -> None:
    # 1. Multi-seed sweep across 12 items covers all topologies, difficulties, orientations, themes.
    sweep_plan = VariationPlanConfig(
        base_seed=100,
        count=12,
        min_obstacles=3,
        max_obstacles=10,
    )
    sweep_res = generate_varied_scenes(sweep_plan)

    assert len(sweep_res.items) == 12
    assert sweep_res.unique_structural_signatures == 12
    assert sweep_res.unsolvable_rejections == 0
    assert set(sweep_res.topology_coverage.keys()) == set(SUPPORTED_TOPOLOGIES)
    assert set(sweep_res.difficulty_coverage.keys()) == {"easy", "medium", "hard"}
    assert set(sweep_res.orientation_coverage.keys()) == {0, 90, 180, 270}
    assert set(sweep_res.theme_coverage.keys()) == {t.value for t in VisualTheme}
    assert len(sweep_res.entity_types_observed) >= 3

    # 2. Verify deduplication retries if two slots start with the exact same seed/config.
    engine = VariationEngine()
    cfg = VariationConfig(seed=5, topology_id="warehouse_split_aisle_v1")
    sig1 = compute_scene_signature(generate_scene(cfg))
    sig2 = compute_scene_signature(generate_scene(cfg.model_copy(update={"seed": 5 + 1009})))
    assert sig1.structural_hash != sig2.structural_hash
