# Visual Task Factory — 3D Benchmark Evaluation Results

**Hackathon:** INNODATA AI HACKATHON 2026  
**Dataset:** `warehouse_spatial_3d_v1` (`data/3d/dataset.json` — `2` scenes, `16` validated records across `4` question families)  
**Primary Sources:** `reports/phase9_final_summary.md` and saved JSON reports under `reports/`

---

## Executive Summary

Phase 9 evaluated live Vision-Language Models (VLMs) on the 3D warehouse spatial benchmark. Two distinct OpenRouter configurations are documented separately below:

1. **Earlier Auto-Router Baseline (`openrouter/free`, Full 16 Records):** Completed all `16` requests with `11` correct (`68.75%` overall accuracy), `0` incorrect parsed responses (`100.0%` parsed accuracy), and `5` empty-output parse errors. Because `openrouter/free` dynamically selects upstream models per request without attributing them, auto-routing was subsequently disabled in favor of explicit fixed models.
2. **Fixed-Model Qwen Evaluation (`openrouter` / `qwen/qwen3.8-27b`):** Tested across a `1`-record smoke run (`1 / 1` correct), a `5`-record pilot at `max_tokens=1024` (`3 / 5` correct, `2` failed due to reasoning-token exhaustion at `1024` and `1029` tokens), and a targeted `2`-record retry after raising `max_tokens` to `2048` (`2 / 2` correct).

> **Important Scope Notice:** Individual runs are reported separately and **must not** be merged into a full-benchmark `100%` accuracy figure. Under fixed `openrouter` / `qwen/qwen3.8-27b`, only the **first 5 unique records** of the 16-record benchmark (all from scene `wh3d_stacked_01`) have been evaluated so far (`3` passing in the 5-record pilot at `max_tokens=1024`, and the `2` failed `object_overlap_3d` records passing on targeted retry at `max_tokens=2048`). The remaining **11 records** have not yet been run on `qwen/qwen3.8-27b`.

---

## Summary Comparison Table (Separate Runs)

| Run | Report JSON | Provider / Model | `max_tokens` | Samples | Completed | Correct | Incorrect | Parse Errors | Failed Requests | Overall Acc. | Parsed Acc. |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Auto-Router Baseline (16 records) | `reports/openrouter_full16/evaluation_3d_report.json` | `openai_compatible` / `openrouter/free` | 1024 | 16 | 16 | 11 | 0 | 5 | 0 | 68.75% (`0.6875`) | 100.0% (`1.0`) |
| Fixed Qwen Smoke (1 record) | `reports/openrouter_qwen_smoke1/evaluation_3d_report.json` | `openrouter` / `qwen/qwen3.8-27b` | 1024 | 1 | 1 | 1 | 0 | 0 | 0 | 100.0% (`1.0`) | 100.0% (`1.0`) |
| Fixed Qwen Pilot (5 records) | `reports/openrouter_qwen_pilot5/evaluation_3d_report.json` | `openrouter` / `qwen/qwen3.8-27b` | 1024 | 5 | 3 | 3 | 0 | 0 | 2 | 60.0% (`0.6`) | 100.0% (`1.0`) |
| Targeted Overlap Retry (2 records) | `reports/openrouter_qwen_overlap_retry2/evaluation_3d_report.json` | `openrouter` / `qwen/qwen3.8-27b` | 2048 | 2 | 2 | 2 | 0 | 0 | 0 | 100.0% (`1.0`) | 100.0% (`1.0`) |

---

## 1. OpenRouter Free Auto-Router Baseline (16 Records)

- **Source Report:** `reports/openrouter_full16/evaluation_3d_report.json`
- **Timestamp (UTC):** `2026-11-26T13:29:45.194710+00:00`
- **Configuration:** `provider_name="openai_compatible"`, `model_name="openrouter/free"`, `max_tokens=1024`
- **Metrics:** `sample_count=16`, `completed_requests=16`, `correct_count=11`, `incorrect_count=0`, `parse_error_count=5`, `failed_request_count=0`, `missing_image_count=0`, `overall_accuracy=0.6875` (`68.75%`), `parsed_accuracy=1.0` (`100.0%`).

### Per-Family Breakdown (`openrouter/free`)

| Question Family | Total | Correct | Incorrect | Parse Errors | Failed Requests | Family Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `camera_visibility_3d` | 4 | 4 | 0 | 0 | 0 | 100.0% (`1.0`) |
| `object_overlap_3d` | 4 | 3 | 0 | 1 | 0 | 75.0% (`0.75`) |
| `vertical_relation_3d` | 4 | 3 | 0 | 1 | 0 | 75.0% (`0.75`) |
| `vertical_clearance_3d` | 4 | 1 | 0 | 3 | 0 | 25.0% (`0.25`) |

### Unparsed Failures in Baseline (`status="parse_error"`, `error_message="Empty model output."`)
1. `wh3d_stacked_01__ov3d__01` (`object_overlap_3d`, expected `True`)
2. `wh3d_stacked_01__vclr3d__02` (`vertical_clearance_3d`, expected `True`)
3. `scene_two_zone_corridor_0202__vrel3d__01` (`vertical_relation_3d`, expected `True`)
4. `scene_two_zone_corridor_0202__vclr3d__01` (`vertical_clearance_3d`, expected `3.4`)
5. `scene_two_zone_corridor_0202__vclr3d__02` (`vertical_clearance_3d`, expected `1.5`)

---

## 2. Fixed-Model Qwen Smoke & Pilot Results (`openrouter` / `qwen/qwen3.8-27b`, `max_tokens=1024`)

Following Phase 9 provider hardening, `openrouter/free` was disallowed in `create_live_vlm_adapter_from_env()` and OpenRouter requests were locked to the explicit fixed model `qwen/qwen3.8-27b`.

### 2a. Single-Record Smoke Test
- **Source Report:** `reports/openrouter_qwen_smoke1/evaluation_3d_report.json`
- **Timestamp (UTC):** `2026-11-26T15:22:01.487176+00:00`
- **Outcome:** `1 / 1` `correct` (`overall_accuracy=1.0`, `parsed_accuracy=1.0`)
  - `wh3d_stacked_01__ov3d__01` (`object_overlap_3d`): `raw_prediction="true"`, expected `True`, `latency_ms=2857.94`, `error_message=null`.

### 2b. 5-Record Pilot (`max_tokens=1024`)
- **Source Report:** `reports/openrouter_qwen_pilot5/evaluation_3d_report.json`
- **Timestamp (UTC):** `2026-11-26T15:24:47.651274+00:00`
- **Scope:** First `5` records of `warehouse_spatial_3d_v1` (all in scene `wh3d_stacked_01`).
- **Aggregate Metrics:** `sample_count=5`, `completed_requests=3`, `correct_count=3`, `incorrect_count=0`, `parse_error_count=0`, `failed_request_count=2`, `overall_accuracy=0.6` (`60.0%`), `parsed_accuracy=1.0` (`100.0%`).

| Question ID | Family | Status | Raw Prediction | Expected | Latency (ms) | Error / Diagnostic |
| --- | --- | --- | --- | --- | ---: | --- |
| `wh3d_stacked_01__ov3d__01` | `object_overlap_3d` | `failed_request` | `null` | `True` | — | `reasoning-only response with empty final content (model='qwen/qwen3.8-27b', finish_reason='length', reasoning_tokens=1024)` |
| `wh3d_stacked_01__ov3d__02` | `object_overlap_3d` | `failed_request` | `null` | `False` | — | `reasoning-only response with empty final content (model='qwen/qwen3.8-27b', finish_reason='length', reasoning_tokens=1029)` |
| `wh3d_stacked_01__vrel3d__01` | `vertical_relation_3d` | `correct` | `"true"` | `True` | `2529.4` | — |
| `wh3d_stacked_01__vrel3d__02` | `vertical_relation_3d` | `correct` | `"false"` | `False` | `4646.69` | — |
| `wh3d_stacked_01__vclr3d__01` | `vertical_clearance_3d` | `correct` | `"false"` | `False` | `2658.9` | — |

---

## 3. Targeted Overlap Retry Results (`openrouter` / `qwen/qwen3.8-27b`, `max_tokens=2048`)

To resolve reasoning-budget exhaustion without altering prompts, architecture, or unrelated records, `"max_tokens"` in `OpenAICompatibleVLM3DAdapter.predict()` (`src/visual_task_factory/evaluator_3d.py`) was increased from `1024` to `2048` (verified offline with `5 / 5` tests in `tests/test_evaluator_3d.py` and `106 / 106` tests across the full suite).

The two failed records were retried using `Evaluation3DConfig(families=(QuestionFamily3D.OBJECT_OVERLAP_3D,), max_records=2)`:

- **Source Report:** `reports/openrouter_qwen_overlap_retry2/evaluation_3d_report.json`
- **Timestamp (UTC):** `2026-11-26T15:32:02.487974+00:00`
- **Aggregate Metrics:** `sample_count=2`, `completed_requests=2`, `correct_count=2`, `incorrect_count=0`, `parse_error_count=0`, `failed_request_count=0`, `overall_accuracy=1.0` (`100.0%`), `parsed_accuracy=1.0` (`100.0%`).

| Question ID | Family | Status | Raw Prediction | Expected | Latency (ms) | Error |
| --- | --- | --- | --- | --- | ---: | --- |
| `wh3d_stacked_01__ov3d__01` | `object_overlap_3d` | `correct` | `"true"` | `True` | `2421.68` | `null` |
| `wh3d_stacked_01__ov3d__02` | `object_overlap_3d` | `correct` | `"false"` | `False` | `2628.23` | `null` |

---

## 4. Evaluation Scope & Limitations

1. **Only the first 5 unique records evaluated under fixed `qwen/qwen3.8-27b`:**
   - Across `openrouter_qwen_pilot5` (`3` passing records at `max_tokens=1024`) and `openrouter_qwen_overlap_retry2` (`2` retried overlap records passing at `max_tokens=2048`), `qwen/qwen3.8-27b` has answered the first `5` unique records of `warehouse_spatial_3d_v1` (all in scene `wh3d_stacked_01`) correctly across separate runs.
   - No single combined `max_tokens=2048` run covering the 5-record pilot or full 16-record dataset has been executed yet.
2. **11 of 16 records remain unevaluated on `qwen/qwen3.8-27b`:**
   - Untested records under fixed `qwen/qwen3.8-27b` include `wh3d_stacked_01__vclr3d__02`, `wh3d_stacked_01__vis3d__01`, `wh3d_stacked_01__vis3d__02`, and all `8` records in `scene_two_zone_corridor_0202` (which contain `camera_visibility_3d` queries and numeric `vertical_clearance_3d` ground truths `3.4 m` and `1.5 m`).
   - Full-benchmark accuracy for `qwen/qwen3.8-27b` is therefore not established.
3. **Auto-router baseline opacity:**
   - The earlier `openrouter/free` 16-record evaluation (`68.75%` overall accuracy) remains preserved for comparison, but cannot be attributed to a single fixed model because `openrouter/free` did not record per-request upstream routed model IDs.
4. **Operational constraints:**
   - `Evaluation3DConfig` currently filters by `families` and `max_records` rather than arbitrary `record_id` lists, and shared/free OpenRouter endpoints can still experience rate limits or reasoning-token overruns on more complex scenes.
