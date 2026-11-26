# Visual Task Factory — 2D Facility Spatial Layout & Safety Inspection

**Innodata AI Hackathon 2026 (Visual Task Factory Track)**

## Purpose

Visual Task Factory is a modular benchmark pipeline designed to evaluate multimodal models on practical visual-spatial reasoning tasks with **independently verifiable deterministic ground truth** and **meaningful structural/visual variation**.

Selected domain: **Procedural 2D Facility / Warehouse Spatial Layout & Safety Inspection**, covering three core question families:
1. **Constrained Reachability (`constrained_reachability`)** — normalized answer: `exit_id` (`str`, e.g., `"E1"`).
2. **Region Containment & Counting (`region_containment_counting`)** — normalized answer: `integer` (`int >= 0`).
3. **Scale-Grounded Clearance (`scale_grounded_clearance`)** — normalized answer: `boolean` (`bool`) or `numeric` (`float >= 0` with unit `"m"`).

## Current Implementation Status

- **Implemented:**
  - Foundational project scaffold and strictly validated Pydantic v2 data models (`VariationConfig`, `SceneSpec`, `ReferenceAnswer`, `VisualQuestion`, `BenchmarkItem`).
  - Procedural 2D warehouse `scene_generator` (`WarehouseSceneGenerator`, `generate_scene`) producing seeded, reproducible `SceneSpec` layouts across multiple structural topologies (`warehouse_split_aisle_v1`, `warehouse_corridor_spine_v1`, `warehouse_cross_aisle_v1`).
  - Deterministic 2D Pillow `renderer` (`WarehouseSceneRenderer`, `render_scene`) generating anti-aliased PNG floorplans with walls, open/blocked doorways, exits, hatched hazard/restricted zones, entity icons, scale bar, compass orientation, visual themes, and optional `audit_overlay`.
  - Independent deterministic `ground_truth_solver` (`GroundTruthSolver`, `solve_scene`) computing reference answers directly from `SceneSpec` for constrained reachability (discretized 8-connected grid Dijkstra), region containment counting, and scale-grounded Euclidean clearance.
  - Deterministic template-based `question_generator` (`QuestionGenerator`, `generate_questions`) producing `BenchmarkItem` records with serialized `solver_query` specs backed by `GroundTruthSolver`.
  - Independent `validator` gate (`BenchmarkValidator`, `validate_benchmark_item`, `validate_benchmark_batch`) verifying structural integrity, prompt consistency, leakage checks, batch duplicates, and independent solver-answer agreement.
  - Deterministic `variation_engine` (`VariationEngine`, `generate_varied_scenes`, `compute_scene_signature`, `compare_scenes`) orchestrating multi-axis variation across topologies, difficulties, orientations, scales, themes, and clutter levels with canonical layout deduplication.
  - Reproducible end-to-end `dataset_pipeline` (`DatasetPipeline`, `build_dataset`) producing portable `dataset.jsonl`, `rejected_items.jsonl`, `scenes.jsonl`, `images/*.png`, and `manifest.json` with validation gating and logical SHA-256 verification.
  - Provider-agnostic `model_evaluator` (`MultimodalModelAdapter`, `MockModelAdapter`, `ModelEvaluator`, `evaluate_dataset`) supporting conservative prediction normalization, error/missing-image isolation, and synthetic mock pipeline verification (`predictions.jsonl`, `evaluation_summary.json`).
  - Metrics and failure analyzer `metrics_analyzer` (`MetricsAnalyzer`, `analyze_evaluation`, `analyze_records`) producing `metrics_summary.json`, `failure_analysis.json`, and `failure_examples.jsonl` with family/type breakdowns and strict mock vs. real vs. unknown provenance labeling.
  - Lightweight local web `demo_app` & `demo_service` built on Python's standard-library `http.server` (zero extra UI dependencies) to generate small demo samples, inspect rendered warehouse PNGs, view isolated reference answers, and browse evaluation/failure reports.
- **Not Yet Implemented:** Live external VLM API adapters (pending verified credentials/quotas).

## Project Structure

```text
visual-task-factory\
├── .gitignore
├── pyproject.toml
├── README.md
├── src\
│   └── visual_task_factory\
│       ├── __init__.py
│       ├── dataset_pipeline.py
│       ├── demo_app.py
│       ├── demo_service.py
│       ├── ground_truth_solver.py
│       ├── metrics_analyzer.py
│       ├── model_evaluator.py
│       ├── question_generator.py
│       ├── renderer.py
│       ├── validator.py
│       ├── variation_engine.py
│       ├── models\
│       │   ├── __init__.py
│       │   ├── variation.py
│       │   ├── scene.py
│       │   └── qa.py
│       └── scene_generator\
│           ├── __init__.py
│           └── generator.py
└── tests\
    ├── test_models.py
    ├── test_scene_generator.py
    ├── test_renderer.py
    ├── test_ground_truth_solver.py
    ├── test_question_generator.py
    ├── test_validator.py
    ├── test_variation_engine.py
    ├── test_dataset_pipeline.py
    ├── test_model_evaluator.py
    ├── test_metrics_analyzer.py
    └── test_demo_interface.py
├── submission\
│   ├── DEMO_WALKTHROUGH.md
│   ├── SUBMISSION_OVERVIEW.md
│   └── submission_manifest.json
```

## Installation (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

*(If running directly from source without editable install, set `$env:PYTHONPATH="src"`.)*

## Running Tests (Windows PowerShell)

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Or if your virtual environment is activated:

```powershell
pytest
```

## Generating a Small Benchmark Dataset

Generated artifacts under `artifacts/` are gitignored and do not exist in a fresh clone until created locally:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -c "from pathlib import Path; from visual_task_factory.dataset_pipeline import DatasetPipelineConfig, build_dataset; from visual_task_factory.variation_engine import VariationPlanConfig; res = build_dataset(DatasetPipelineConfig(dataset_id='demo_v1', variation_plan=VariationPlanConfig(base_seed=101, count=3)), Path('artifacts/demo_v1')); print(res.manifest.valid_item_count, res.output_dir)"
```

## Running Phase 10 Metrics & Failure Analysis

Ensure an evaluation directory exists first (for example, by running a synthetic mock evaluation on `artifacts/demo_v1` with `evaluate_dataset` and `MockModelAdapter`, or using an existing local run under `artifacts/`), then run the analyzer:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -c "from visual_task_factory.model_evaluator import MockAdapterMode, MockModelAdapter, evaluate_dataset; from visual_task_factory.metrics_analyzer import analyze_evaluation; evaluate_dataset('artifacts/demo_v1', 'artifacts/demo_v1_eval', MockModelAdapter(seed=101, mode=MockAdapterMode.BLIND_HASH)); res = analyze_evaluation('artifacts/demo_v1_eval'); print(res.metrics_summary.model_dump_json(indent=2))"
```

Generated reports:
- **`metrics_summary.json`**: Overall accuracy, completed accuracy, unparseable/error/missing-image rates, and accuracy grouped by `family` and `expected_answer_type`.
- **`failure_analysis.json`**: Failure counts/rates by grounded category (`incorrect_prediction`, `unparseable_prediction`, `adapter_error`, `missing_image`) crossed with family and answer type.
- **`failure_examples.jsonl`**: Deterministically sorted failure records containing `question_id`, `family`, `expected_answer_type`, `reference_answer`, `raw_prediction`, `normalized_prediction`, `status`, and `failure_category`.

> **Mock vs. Real Model Evaluation:** Reports derived from `MockModelAdapter` are labeled `evaluation_nature: "mock"` and **`SYNTHETIC MOCK EVALUATION — NOT REAL MODEL CAPABILITY`**. Mock outputs verify pipeline execution only and must never be reported as real multimodal model accuracy or combined with real model evaluation runs.

## Running the Phase 11 Demo Interface

Uses Python's built-in `http.server` (no extra UI dependencies required):

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m visual_task_factory.demo_app --port 8501
```

Then open `http://127.0.0.1:8501/` in a browser to generate small demo samples (1–4 scenes), inspect warehouse PNGs, view isolated reference answers, and browse dataset/validation/evaluation/failure-analysis reports.

## Submission Documentation

- [SUBMISSION_OVERVIEW.md](submission/SUBMISSION_OVERVIEW.md) — full architecture, usage, test results, and limitations
- [DEMO_WALKTHROUGH.md](submission/DEMO_WALKTHROUGH.md) — step-by-step hackathon demo guide
- [submission_manifest.json](submission/submission_manifest.json) — lightweight package metadata
