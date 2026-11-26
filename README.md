# Visual Task Factory

**INNODATA AI HACKATHON 2026 — Procedural Spatial Reasoning Benchmark & Evaluation Pipeline**  
Repository: https://github.com/rona346/visual-task-factory

Visual Task Factory is a modular Python benchmark generation, validation, and evaluation framework for **2D facility spatial safety inspection** and **3D isometric warehouse spatial reasoning**. Rather than relying on static curated images prone to memorization or subjective annotations, it pairs procedural scene generation with deterministic visual rendering and independent CPU geometric solvers to produce verified multimodal tasks.

---

## Problem Statement & Motivation

Vision-Language Models (VLMs) are increasingly asked to reason about physical layouts, clearances, stacking relationships, and line-of-sight occlusion, yet static spatial benchmarks suffer from:
1. **Training-set contamination and shortcuts**, where models recall answers rather than inspect geometry.
2. **Coupled or noisy labels**, where question writers infer answers visually instead of computing exact geometric ground truth.
3. **Unreliable free-tier VLM evaluation harnesses**, where auto-routed models or unguarded reasoning models exhaust output tokens and return silent empty strings that skew accuracy analysis.

Visual Task Factory addresses this by decoupling **scene specification**, **deterministic PNG rendering**, **independent geometric ground-truth solving**, **anti-leakage QA gating**, and **failure-aware VLM evaluation**.

---

## What Visual Task Factory Does

1. **Generates deterministic warehouse scenes** in both 2D (`SceneSpec`) and 3D (`SceneSpec3D`) with configurable layouts, zones, obstacles, stacked objects, doorways, and camera poses.
2. **Renders reproducible PNG visuals** via Pillow-based 2D top-down floorplan and fixed-angle isometric 3D renderers (with supersampling, grid lines, object labels, scale bars, and optional wall cutaways).
3. **Solves exact ground truth from geometry alone** using independent CPU-only solvers (grid Dijkstra reachability, spatial containment/clearance in 2D; axis-aligned 3D bounding-box overlap, Z-axis vertical relations, vertical clearance, and camera ray-box line-of-sight occlusion in 3D).
4. **Enforces validation gates** (`BenchmarkValidator`, `GroundTruthQAValidator3D`) checking coordinate-leak patterns, prompt consistency, duplicate IDs, and solver re-verification before exporting datasets.
5. **Evaluates multimodal models with strict failure attribution**, isolating `correct`, `incorrect`, `parse_error`, `failed_request`, and `missing_image` outcomes, detecting reasoning-only exhaustion, retrying transient `429`/`5xx`/timeout errors conservatively, and redacting secrets from saved reports.

---

## Supported Benchmark & Task Categories

### 3D Warehouse Spatial Reasoning (`data/3d/`, `warehouse_spatial_3d_v1` — 2 scenes, 16 validated records)
- **`object_overlap_3d`** (`boolean`, 4 records in `data/3d`) — whether two 3D objects touch boundaries, overlap internally, or remain disjoint in 3D space.
- **`vertical_relation_3d`** (`boolean`, 4 records in `data/3d`) — vertical Z-axis ordering predicates (`above`, `below`, `touching_above`, `equal_height`).
- **`vertical_clearance_3d`** (`numeric` in `m` within `±0.05 m` tolerance or threshold `boolean`, 4 records in `data/3d`) — vertical gap between stacked/overlapping-XY objects or to the warehouse ceiling.
- **`camera_visibility_3d`** (`boolean`, 4 records in `data/3d`) — whether an object is clear, partially occluded, or fully occluded from `SceneSpec3D.camera.eye`.

### 2D Facility Spatial Layout & Safety Inspection (`dataset_pipeline`)
- **`constrained_reachability`** (`exit_id` string, e.g., `"E1"`) — shortest compliant path under blocked-doorway and hazard/restricted-zone constraints.
- **`region_containment_counting`** (`integer >= 0`) — counting entities inside named zones or regions.
- **`scale_grounded_clearance`** (`boolean` or `numeric` in `m`) — scale-calibrated Euclidean clearance checks between facility objects.

---

## Architecture & Workflow

```text
2D Pipeline:
  VariationEngine ─▶ WarehouseSceneGenerator (SceneSpec) ─▶ WarehouseSceneRenderer (PNG)
                                                          ├─▶ GroundTruthSolver ─▶ QuestionGenerator ─▶ BenchmarkValidator
                                                          └─▶ DatasetPipeline (dataset.jsonl + manifest.json)
                                                                └─▶ ModelEvaluator (MockModelAdapter) ─▶ MetricsAnalyzer

3D Pipeline:
  SceneSpec3D ─▶ Warehouse3DRenderer (Isometric PNG)
              ├─▶ GroundTruthSolver3D ─▶ QuestionGenerator3D ─▶ GroundTruthQAValidator3D
              └─▶ DatasetBuilder3D (data/3d/dataset.json + images/*.png)
                    ├─▶ demo_3d (offline solver/QA verification + visual report)
                    └─▶ Evaluation3DRunner (OpenAICompatibleVLM3DAdapter: OpenRouter / Groq)
                          └─▶ evaluation_3d_report.json + evaluation_3d_summary.md
```

---

## Current Implementation Features

- **Strict Pydantic v2 schemas** (`extra="forbid"`) across 2D (`variation.py`, `scene.py`, `qa.py`) and 3D (`scene3d.py`, `qa_3d.py`, `evaluator_3d.py`).
- **Deterministic 2D & 3D rendering** (`renderer.py`, `renderer_3d.py`) producing self-contained PNG assets without external GPU dependencies.
- **Independent geometric solvers** (`ground_truth_solver.py`, `solver_3d.py`) whose reasoning traces and tolerances are serialized into every benchmark record.
- **Explicit live VLM provider factory** (`create_live_vlm_adapter_from_env()` in `evaluator_3d.py`):
  - Supports **only** explicit `VLM_PROVIDER=openrouter` or `VLM_PROVIDER=groq` (never silently selects another provider or paid fallback).
  - Requires an explicit fixed `VLM_MODEL` for OpenRouter and **rejects `openrouter/free`** auto-routing to avoid non-deterministic routing and silent empty responses.
  - Defaults `groq` to `meta-llama/llama-4-scout-17b-16e-instruct` when `VLM_MODEL` is omitted.
  - Sends `temperature=0.0` and `max_tokens=2048` with base64-encoded PNG data URIs (`timeout_s=30.0`, `max_retries=2`, capped `Retry-After`/exponential backoff up to `5.0s`).
  - Explicitly detects reasoning-only responses (empty final `content` paired with `reasoning`/`reasoning_details` or `reasoning_tokens > 0`) as `failed_request` diagnostics and redacts API keys/Bearer tokens via `_redact_secrets()`.
- **Offline demos & failure analytics**: `demo_3d.py` CLI for 3D scene-to-QA verification, plus `demo_app.py` / `demo_service.py` (`http.server`) and `metrics_analyzer.py` for 2D inspection.
- **Automated test suite**: `106` unit/integration tests across 18 test modules (with live API providers mocked in unit tests).

---

## Installation & Setup (Windows PowerShell)

### 1. Create virtual environment and install package
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

For **live 3D VLM evaluation** (`evaluator_3d.py`), also ensure `httpx` is installed in `.venv` (if not already present, since `pyproject.toml` pins `pillow`, `pydantic`, and `pytest`):
```powershell
.\.venv\Scripts\python.exe -m pip install httpx
```

*(If running directly from source without editable install, set `$env:PYTHONPATH="src"`.)*

### 2. Configure environment variables via `.env.example`
Copy `.env.example` to a local gitignored `.env` file and populate your own credentials locally (never commit or print real keys):

```powershell
Copy-Item .env.example .env
```

`.env.example` defines:
- `VLM_PROVIDER`: `openrouter` or `groq` (must be set explicitly)
- `VLM_MODEL`: required fixed vision model ID for `openrouter` (e.g., `qwen/qwen3.8-27b`; `openrouter/free` is rejected); optional for `groq` (defaults to `meta-llama/llama-4-scout-17b-16e-instruct`)
- `OPENROUTER_API_KEY`: required when `VLM_PROVIDER=openrouter`
- `GROQ_API_KEY`: required when `VLM_PROVIDER=groq`

---

## Running the Project & Tests

### Run the full test suite (offline, no live API calls)
```powershell
.\.venv\Scripts\python.exe -m pytest
```
Verified test status on the current implementation: **`106 passed`** (with 1 pre-existing Pydantic serializer warning in `tests/test_validator.py`). Focused evaluator tests (`tests/test_evaluator_3d.py`): **`5 passed`**.

### Run the offline 3D end-to-end demo
```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m visual_task_factory.demo_3d --mode existing_dataset --dataset-dir data/3d --scene-index 0 --max-questions 6 --output-dir artifacts/demo_3d
```
Writes `demo_3d_report.json`, `demo_3d_summary.md`, and rendered scene PNGs under `artifacts/demo_3d/`.

### Run a controlled 3D live evaluation (requires API key & explicit env config)
Load your `.env` values into the shell process before calling `run_evaluation_3d` (note: `evaluator_3d.py` reads `os.environ` directly and does not auto-load `.env`):
```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -c "from visual_task_factory.evaluator_3d import Evaluation3DConfig, run_evaluation_3d; res = run_evaluation_3d('data/3d', output_dir='reports/custom_eval', config=Evaluation3DConfig(max_records=5)); print(res.json_report_path, res.report.overall_accuracy)"
```

### Run the 2D dataset generator, mock evaluation analyzer, and web UI
```powershell
# Generate a 2D dataset and synthetic mock evaluation report
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -c "from pathlib import Path; from visual_task_factory.dataset_pipeline import DatasetPipelineConfig, build_dataset; from visual_task_factory.variation_engine import VariationPlanConfig; from visual_task_factory.model_evaluator import MockAdapterMode, MockModelAdapter, evaluate_dataset; from visual_task_factory.metrics_analyzer import analyze_evaluation; build_dataset(DatasetPipelineConfig(dataset_id='demo_v1', variation_plan=VariationPlanConfig(base_seed=101, count=3)), Path('artifacts/demo_v1')); evaluate_dataset('artifacts/demo_v1', 'artifacts/demo_v1_eval', MockModelAdapter(seed=101, mode=MockAdapterMode.BLIND_HASH)); res = analyze_evaluation('artifacts/demo_v1_eval'); print(res.metrics_summary.overall_accuracy)"

# Launch local 2D inspection web UI (Python stdlib http.server)
.\.venv\Scripts\python.exe -m visual_task_factory.demo_app --port 8501
```
> **Mock vs. Live Provenance:** `MockModelAdapter` outputs are stamped `SYNTHETIC MOCK EVALUATION — NOT REAL MODEL CAPABILITY` and verify pipeline execution only.

---

## Evaluation Workflow & Results

Full verified evaluation figures are documented in **[`reports/benchmark_results.md`](reports/benchmark_results.md)** and **[`reports/phase9_final_summary.md`](reports/phase9_final_summary.md)**. Summary of saved Phase 9 OpenRouter runs on `warehouse_spatial_3d_v1`:

- **Earlier Auto-Router Baseline (`openrouter/free`, 16 records):** `11 / 16` correct (`68.75%` overall accuracy), `0` incorrect, `5` empty-output parse errors, `100.0%` parsed accuracy (`reports/openrouter_full16/evaluation_3d_report.json`). Replaced because dynamic model routing masked upstream model identity and emitted empty outputs.
- **Fixed-Model Smoke (`openrouter` / `qwen/qwen3.8-27b`, `max_tokens=1024`, 1 record):** `1 / 1` correct (`100.0%`, `reports/openrouter_qwen_smoke1/evaluation_3d_report.json`).
- **Fixed-Model Pilot (`openrouter` / `qwen/qwen3.8-27b`, `max_tokens=1024`, 5 records):** `3 / 5` correct (`60.0%` overall, `100.0%` parsed accuracy), with `2` `object_overlap_3d` failures caused by reasoning-token exhaustion (`finish_reason='length'`, `reasoning_tokens=1024` and `1029`; `reports/openrouter_qwen_pilot5/evaluation_3d_report.json`).
- **Targeted Overlap Retry (`openrouter` / `qwen/qwen3.8-27b`, `max_tokens=2048`, 2 records):** Retrying the two failed `object_overlap_3d` records after raising `max_tokens` to `2048` returned `2 / 2` correct (`100.0%`, `reports/openrouter_qwen_overlap_retry2/evaluation_3d_report.json`).

---

## Current Limitations

- **Partial fixed-model benchmark coverage:** Fixed-model `qwen/qwen3.8-27b` runs have evaluated **only the first 5 unique records** of the 16-record `warehouse_spatial_3d_v1` dataset across separate runs (`openrouter_qwen_pilot5` + `openrouter_qwen_overlap_retry2`). The remaining **11 records** (including `camera_visibility_3d`, numeric `vertical_clearance_3d`, and all 8 records in `scene_two_zone_corridor_0202`) have not yet been evaluated on `qwen/qwen3.8-27b`.
- **Selection granularity:** `Evaluation3DConfig` currently filters by `families` and `max_records` rather than arbitrary `record_id` / `question_id` lists.
- **No automated `.env` loader inside `evaluator_3d.py`:** `create_live_vlm_adapter_from_env()` reads `os.environ` directly; callers must export or load `.env` variables into the process environment before invoking evaluation.
- **Local-only execution:** Interactive inspection (`demo_app.py`) and 3D demo (`demo_3d.py`) run locally; no hosted cloud deployment is included.

---

## Repository Structure

```text
visual-task-factory/
├── .env.example                                  # Placeholder provider/model/key template
├── .gitignore
├── pyproject.toml                                # Package metadata and pytest settings
├── README.md
├── data/
│   └── 3d/
│       ├── dataset.json                          # Validated 16-record 3D warehouse dataset
│       └── images/                               # Rendered 3D isometric PNGs (2 scenes)
├── reports/
│   ├── benchmark_results.md                      # Submission benchmark report
│   ├── phase9_final_summary.md                   # Phase 9 closeout documentation
│   ├── openrouter_full16/                        # 16-record auto-router baseline report
│   ├── openrouter_qwen_smoke1/                   # 1-record fixed Qwen smoke report
│   ├── openrouter_qwen_pilot5/                   # 5-record fixed Qwen pilot report
│   └── openrouter_qwen_overlap_retry2/           # 2-record overlap retry report (max_tokens=2048)
├── src/
│   └── visual_task_factory/
│       ├── dataset_builder_3d.py                 # 3D dataset builder, validator, and loader
│       ├── dataset_pipeline.py                   # 2D end-to-end dataset builder
│       ├── demo_3d.py                            # Offline 3D demo runner and CLI
│       ├── demo_app.py                           # Local HTTP demo UI server (2D)
│       ├── demo_service.py                       # Demo inspection backend service (2D)
│       ├── evaluator_3d.py                       # Live 3D VLM evaluation framework (OpenRouter/Groq)
│       ├── ground_truth_solver.py                # 2D deterministic solver
│       ├── metrics_analyzer.py                   # 2D evaluation metrics & failure analyzer
│       ├── model_evaluator.py                    # 2D evaluation runner & MockModelAdapter
│       ├── qa_3d.py                              # 3D QA schemas & independent validator
│       ├── question_generator.py                 # 2D template question generator
│       ├── question_generator_3d.py              # 3D template question generator
│       ├── renderer.py                           # 2D top-down Pillow renderer
│       ├── renderer_3d.py                        # 3D isometric Pillow renderer
│       ├── solver_3d.py                          # 3D deterministic geometric & occlusion solver
│       ├── validator.py                          # 2D independent validation gate
│       ├── variation_engine.py                   # Multi-axis scene variation engine
│       ├── models/
│       │   ├── qa.py
│       │   ├── scene.py
│       │   ├── scene3d.py
│       │   └── variation.py
│       └── scene_generator/
│           └── generator.py
├── submission/
│   ├── DEMO_WALKTHROUGH.md
│   ├── SUBMISSION_OVERVIEW.md
│   └── submission_manifest.json
└── tests/                                        # 18 test modules (106 offline tests)
```
