# Visual Task Factory — Submission Overview

**Innodata AI Hackathon 2026 — Visual Task Factory Track**  
**Domain:** Procedural 2D Facility / Warehouse Spatial Layout & Safety Inspection

---

## 1. Project Overview

Visual Task Factory is a modular, CPU-friendly Python benchmark pipeline that procedurally generates synthetic 2D warehouse/facility floorplans, renders them as anti-aliased PNG images, formulates spatial reasoning questions with independently computed deterministic reference answers, validates every generated item, evaluates multimodal model adapters, and exports grounded metrics and failure-analysis reports alongside a local web demo interface.

---

## 2. Problem Statement & Intended Use

Evaluating multimodal models on spatial reasoning often suffers from data contamination, OCR shortcuts, or circular LLM-as-a-judge ground truth. This project addresses that gap by providing a reproducible benchmark factory where:
- Spatial geometry exists strictly inside rendered floorplan pixels (no leaked coordinates or hidden answer hints in text).
- Reference answers are computed deterministically from the underlying structured `SceneSpec` (`ground_truth_solver.py`)—never guessed by an LLM or inferred from rendered images.
- Every benchmark record passes an independent validation gate (`validator.py`) before export.

**Intended use:** Generating verifiable, multi-axis visual-spatial benchmark suites and running repeatable model evaluation & failure diagnostics on Windows CPU-only environments.

---

## 3. Architecture Overview

Strictly separated modular pipeline under `src/visual_task_factory/`:

1. **Models (`models/variation.py`, `models/scene.py`, `models/qa.py`):** Pydantic v2 schemas (`VariationConfig`, `SceneSpec`, `ReferenceAnswer`, `VisualQuestion`, `BenchmarkItem`) enforcing geometric bounds, unique IDs, non-overlapping solid obstacles, and strict answer types.
2. **Scene Generator (`scene_generator/generator.py`):** Seeded procedural layouts across 3 structural topologies (`warehouse_split_aisle_v1`, `warehouse_corridor_spine_v1`, `warehouse_cross_aisle_v1`) with walls, open/blocked doorways, exits, zones, racks/obstacles, scale bars, and `0°/90°/180°/270°` geometric rotation.
3. **2D Image Renderer (`renderer.py`):** Deterministic Pillow renderer (`2×` LANCZOS supersampling, 4 visual themes, optional `audit_overlay=False` by default, clamped exit/zone labels, exact footer scale bar, legend, and compass).
4. **Ground Truth Solver (`ground_truth_solver.py`):** Independent deterministic solver supporting:
   - `constrained_reachability` (`exit_id` via 8-connected grid Dijkstra at `scene.grid_resolution`)
   - `region_containment_counting` (`integer` via `strict_inside`, `center_inside`, `any_overlap`)
   - `scale_grounded_clearance` (`boolean` or `numeric` in meters via exact 2D Euclidean geometry + `ScaleBarSpec`)
5. **Question Generator (`question_generator.py`):** Deterministic template-based generator creating questions across all 3 families with serialized `solver_query` metadata and visible label filtering.
6. **Independent Validator (`validator.py`):** Re-validates schemas, reconstructs `solver_query` as untrusted input, checks scene consistency and leakage, independently recomputes answers via `GroundTruthSolver`, and flags batch duplicates.
7. **Meaningful Variation Engine (`variation_engine.py`):** Block-bijective multi-axis scheduling, canonical geometry hashing (`compute_scene_signature`) ignoring IDs/seeds/themes, reachability gating, and duplicate-layout retry.
8. **Dataset Pipeline (`dataset_pipeline.py`):** Exports portable `dataset.jsonl`, `rejected_items.jsonl`, `scenes.jsonl`, `images/*.png`, and `manifest.json` with `logical_dataset_sha256` and `image_sha256_by_scene`.
9. **Model Evaluator (`model_evaluator.py`):** Provider-agnostic `MultimodalModelAdapter`, deterministic `MockModelAdapter` (`BLIND_HASH` and `SCRIPTED` modes, zero access to `reference_answer`), conservative prediction normalization, and JSONL/JSON export.
10. **Metrics & Failure Analyzer (`metrics_analyzer.py`):** Computes overall/grouped accuracy (`by_family`, `by_answer_type`), grounded failure categories (`incorrect_prediction`, `unparseable_prediction`, `adapter_error`, `missing_image`), and separates `mock` vs. `real` vs. `unknown` provenance.
11. **Demo Interface (`demo_service.py`, `demo_app.py`):** Zero-dependency Python `http.server` web dashboard for small sample generation, PNG inspection, isolated reference-answer viewing, and report browsing.

---

## 4. Key Capabilities

- **Independently verifiable ground truth:** Deterministic pathfinding, containment, and Euclidean clearance directly on `SceneSpec`.
- **Meaningful structural & visual variation:** Orthogonal variation across topology, difficulty, orientation, scale, visual theme, and clutter with layout deduplication.
- **Anti-leakage & strict validation:** Coordinates, internal IDs, and answer strings are checked and rejected by `BenchmarkValidator`.
- **Repeatable CPU-only execution:** Runs cleanly on Windows 11 with Python 3.10.11 and 8 GB RAM using only `pydantic`, `pillow`, and `pytest` (dev).

---

## 5. How to Install & Run

### Environment Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

*(If running directly from source without editable install, set `$env:PYTHONPATH="src"`.)*

### Run the Full Test Suite

```powershell
.\.venv\Scripts\python.exe -m pytest
```

---

## 6. How to Generate a Small Dataset

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -c "from pathlib import Path; from visual_task_factory.dataset_pipeline import DatasetPipelineConfig, build_dataset; from visual_task_factory.variation_engine import VariationPlanConfig; res = build_dataset(DatasetPipelineConfig(dataset_id='demo_v1', variation_plan=VariationPlanConfig(base_seed=101, count=3)), Path('artifacts/demo_v1')); print(res.manifest.valid_item_count, res.output_dir)"
```

Outputs are written to `artifacts/demo_v1/` (`dataset.jsonl`, `manifest.json`, `rejected_items.jsonl`, `scenes.jsonl`, `images/*.png`).

---

## 7. How Evaluation & Metrics Work (and Mock Limitations)

1. `ModelEvaluator(adapter).evaluate(dataset_dir, eval_dir)` loads `dataset.jsonl`, resolves relative PNG paths safely, sends sanitized `ModelRequest`s (excluding `reference_answer`) to `adapter.predict(...)`, conservatively normalizes outputs, compares against `ReferenceAnswer`, and writes `predictions.jsonl` + `evaluation_summary.json`.
2. `analyze_evaluation(eval_dir)` generates `metrics_summary.json`, `failure_analysis.json`, and `failure_examples.jsonl`.

> **CRITICAL MOCK LIMITATION:** No live external VLM provider adapters are implemented or evaluated in this package (API credentials, model IDs, and quotas were not configured). All included evaluation runs use `MockModelAdapter` and are prominently labeled **`SYNTHETIC MOCK EVALUATION — NOT REAL MODEL CAPABILITY`**. Mock results verify pipeline mechanics only and must never be interpreted as real multimodal model accuracy.

---

## 8. How to Launch the Demo Interface

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m visual_task_factory.demo_app --port 8501
```

Open `http://127.0.0.1:8501/` in a browser.

---

## 9. Verified Test & Smoke Results

- **Full test suite:** `.\.venv\Scripts\python.exe -m pytest` → **74 passed, 1 warning in 123.12s**, exit code `0` *(the single `UserWarning` is Pydantic's expected enum serialization warning during intentional corrupted-enum testing in `test_validator.py`)*.
- **Phase 12 fresh end-to-end smoke run (`artifacts/smoke_phase12_e2e`):**
  - Generated `3/3` unique scenes, `9/9` validated benchmark items (`0` invalid), `9/9` PNGs verified via Pillow, and `9/9` independent `BenchmarkValidator` re-check passes.
  - Mock evaluation (`MockModelAdapter(seed=202612)`): `total=9`, `correct=4`, `incorrect=5`, `unparseable=0`, `errors=0`, `missing_img=0`, labeled **`SYNTHETIC MOCK EVALUATION — NOT REAL MODEL CAPABILITY`**.
- **Demo HTTP endpoint check:** Verified `GET /` (`200`), `GET /api/dataset` (`200`, `9` records), `GET /api/evaluation` (`200`, mock banner present), `GET /api/image` (`200`, `image/png`), `POST /api/generate` (`200`, `3` records), path traversal blocked (`403`), and missing-directory graceful error handling (`200` with `error_message`).

---

## 10. Known Limitations

1. **No live VLM API evaluation completed:** Live provider integrations await verified credentials, model IDs, and rate/cost quotas.
2. **Discretized grid pathfinding:** `GroundTruthSolver` uses an 8-connected grid at `scene.grid_resolution` (`0.5` world units) rather than continuous visibility-graph shortest paths.
3. **Axis-aligned geometry & small-entity labels:** Scenes use orthogonal walls and axis-aligned `Rect2D` boxes; tiny entities (`PILLAR`, narrow `FORKLIFT`) omit inline labels in benchmark PNGs and are excluded from clearance question selection.
4. **Font portability:** Byte-identical PNG rendering relies on `C:\Windows\Fonts\arial.ttf` (with fallback fonts on other systems); `logical_dataset_sha256` remains portable across environments.
5. **Browser visual UI check:** HTTP endpoints and rendered PNG files were verified programmatically and visually inspected via file reads; live interactive browser rendering was not manually opened.

---

## 11. Suggested Demo Walkthrough

See [DEMO_WALKTHROUGH.md](DEMO_WALKTHROUGH.md).
