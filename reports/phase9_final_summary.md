# Phase 9 Final Summary — Live OpenRouter 3D VLM Evaluation

## Overview

Phase 9 validated live Vision-Language Model (VLM) evaluation for the 3D warehouse benchmark (`warehouse_spatial_3d_v1`, 16 records across 2 scenes) using **OpenRouter** (`VLM_PROVIDER=openrouter`).

After earlier diagnostics showed that automatic model routing (`openrouter/free`) intermittently returned empty responses and obscured which backend model handled each request, the evaluator was restricted to explicit fixed-model configuration (`openrouter/free` disallowed) with reasoning-only exhaustion detection (`Evaluator3DError`), and tested with the fixed model **`qwen/qwen3.8-27b`**. When two `object_overlap_3d` records in the 5-record fixed-model pilot exhausted `max_tokens=1024` during reasoning (`finish_reason='length'`), raising `OpenAICompatibleVLM3DAdapter.predict()` to `max_tokens=2048` resolved both failures cleanly on targeted retry.

---

## Saved Evaluation Reports Compared

| Report | Provider / Model Setting | `max_tokens` | Samples | Completed | Correct | Incorrect | Parse Errors | Failed Requests | Overall Acc. | Parsed Acc. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reports/openrouter_full16/evaluation_3d_report.json` | `openai_compatible` / `openrouter/free` (auto-router) | 1024 | 16 | 16 | 11 | 0 | 5 | 0 | 68.75% (`0.6875`) | 100.0% (`1.0`) |
| `reports/openrouter_qwen_smoke1/evaluation_3d_report.json` | `openrouter` / `qwen/qwen3.8-27b` (fixed) | 1024 | 1 | 1 | 1 | 0 | 0 | 0 | 100.0% (`1.0`) | 100.0% (`1.0`) |
| `reports/openrouter_qwen_pilot5/evaluation_3d_report.json` | `openrouter` / `qwen/qwen3.8-27b` (fixed) | 1024 | 5 | 3 | 3 | 0 | 0 | 2 | 60.0% (`0.6`) | 100.0% (`1.0`) |
| `reports/openrouter_qwen_overlap_retry2/evaluation_3d_report.json` | `openrouter` / `qwen/qwen3.8-27b` (fixed) | 2048 | 2 | 2 | 2 | 0 | 0 | 0 | 100.0% (`1.0`) | 100.0% (`1.0`) |

---

## 1. Earlier Auto-Router Baseline (`openrouter/free`, Full 16 Records)

- **Report:** `reports/openrouter_full16/evaluation_3d_report.json`
- **Evaluated At (UTC):** `2026-11-26T13:29:45.194710+00:00`
- **Provider / Model:** `openai_compatible` / `openrouter/free`
- **Aggregate Metrics:** `16` samples, `16` completed, `11` correct, `0` incorrect, `5` parse errors, `0` failed requests, `68.75%` overall accuracy, `100.0%` parsed accuracy.
- **Per-Family Breakdown:**
  - `camera_visibility_3d`: `4 / 4` correct (`100.0%`)
  - `object_overlap_3d`: `3 / 4` correct, `1` parse error (`75.0%`)
  - `vertical_relation_3d`: `3 / 4` correct, `1` parse error (`75.0%`)
  - `vertical_clearance_3d`: `1 / 4` correct, `3` parse errors (`25.0%`)
- **Remaining Failures in Report (all `status="parse_error"`, `raw_prediction=""`, `error_message="Empty model output."`):**
  1. `wh3d_stacked_01__ov3d__01` (`object_overlap_3d`, expected `True`)
  2. `wh3d_stacked_01__vclr3d__02` (`vertical_clearance_3d`, expected `True`)
  3. `scene_two_zone_corridor_0202__vrel3d__01` (`vertical_relation_3d`, expected `True`)
  4. `scene_two_zone_corridor_0202__vclr3d__01` (`vertical_clearance_3d`, expected `3.4`)
  5. `scene_two_zone_corridor_0202__vclr3d__02` (`vertical_clearance_3d`, expected `1.5`)
- **Key Observation:** `openrouter/free` routed requests dynamically across underlying free models without recording which upstream routed model produced each response, making behavior non-reproducible and mixing reasoning exhaustion into empty-output parse errors.

---

## 2. Fixed-Model Qwen Results (`openrouter` / `qwen/qwen3.8-27b`)

### 2a. Single-Record Smoke Test (`max_tokens=1024`)
- **Report:** `reports/openrouter_qwen_smoke1/evaluation_3d_report.json`
- **Evaluated At (UTC):** `2026-11-26T15:22:01.487176+00:00`
- **Result:** `1 / 1` `correct` (`overall_accuracy=1.0`, `parsed_accuracy=1.0`)
  - `wh3d_stacked_01__ov3d__01` (`object_overlap_3d`): `raw_prediction="true"`, expected `True`, `latency_ms=2857.94`, `error_message=null`.

### 2b. 5-Record Pilot (`max_tokens=1024`)
- **Report:** `reports/openrouter_qwen_pilot5/evaluation_3d_report.json`
- **Evaluated At (UTC):** `2026-11-26T15:24:47.651274+00:00`
- **Scope:** First 5 records of `warehouse_spatial_3d_v1` (scene `wh3d_stacked_01`).
- **Aggregate Metrics:** `5` samples, `3` completed, `3` correct, `0` incorrect, `0` parse errors, `2` failed requests, `60.0%` overall accuracy, `100.0%` parsed accuracy.
- **Per-Record Outcomes:**
  - `wh3d_stacked_01__ov3d__01` (`object_overlap_3d`): **`failed_request`** — `"Provider 'openrouter' returned a reasoning-only response with empty final content (model='qwen/qwen3.8-27b', finish_reason='length', reasoning_tokens=1024)."`
  - `wh3d_stacked_01__ov3d__02` (`object_overlap_3d`): **`failed_request`** — `"Provider 'openrouter' returned a reasoning-only response with empty final content (model='qwen/qwen3.8-27b', finish_reason='length', reasoning_tokens=1029)."`
  - `wh3d_stacked_01__vrel3d__01` (`vertical_relation_3d`): `correct` (`raw_prediction="true"`, expected `True`, `latency_ms=2529.4`)
  - `wh3d_stacked_01__vrel3d__02` (`vertical_relation_3d`): `correct` (`raw_prediction="false"`, expected `False`, `latency_ms=4646.69`)
  - `wh3d_stacked_01__vclr3d__01` (`vertical_clearance_3d`): `correct` (`raw_prediction="false"`, expected `False`, `latency_ms=2658.9`)

### 2c. Token-Budget Increase & Targeted Overlap Retry (`max_tokens=2048`)
- **Code Change:** Increased `"max_tokens"` in `OpenAICompatibleVLM3DAdapter.predict()` (`src/visual_task_factory/evaluator_3d.py`) from `1024` to `2048`, preserving provider, model, `temperature=0.0`, endpoint, retry policy, prompts, dataset, and Phase 1–8 architecture.
- **Offline Verification:** Focused tests in `tests/test_evaluator_3d.py` passed (`5 passed`), and the full test suite passed (`106 passed`, `1` pre-existing Pydantic serializer warning).
- **Targeted Live Retry Report:** `reports/openrouter_qwen_overlap_retry2/evaluation_3d_report.json`
- **Evaluated At (UTC):** `2026-11-26T15:32:02.487974+00:00`
- **Selection:** `Evaluation3DConfig(families=(QuestionFamily3D.OBJECT_OVERLAP_3D,), max_records=2)` (retrying only the two failed pilot records).
- **Result:** `2 / 2` `correct` (`overall_accuracy=1.0`, `parsed_accuracy=1.0`, `0` errors)
  - `wh3d_stacked_01__ov3d__01` (`object_overlap_3d`): `correct` (`raw_prediction="true"`, expected `True`, `latency_ms=2421.68`)
  - `wh3d_stacked_01__ov3d__02` (`object_overlap_3d`): `correct` (`raw_prediction="false"`, expected `False`, `latency_ms=2628.23`)

---

## Limitations & Remaining Coverage Gaps

1. **Full 16-Record Benchmark Not Yet Evaluated with Fixed `qwen/qwen3.8-27b` (`max_tokens=2048`):**
   - Across `openrouter_qwen_pilot5` (3 passing records at `max_tokens=1024`) and `openrouter_qwen_overlap_retry2` (2 retried records passing at `max_tokens=2048`), 5 distinct records in scene `wh3d_stacked_01` have received correct predictions under `qwen/qwen3.8-27b`, but in two separate runs rather than a single unified `max_tokens=2048` pass.
   - The remaining **11 of 16 records** in `warehouse_spatial_3d_v1` — including `camera_visibility_3d` questions, numeric `vertical_clearance_3d` questions (`3.4`, `1.5`), `wh3d_stacked_01__vclr3d__02`, `wh3d_stacked_01__vis3d__01`, `wh3d_stacked_01__vis3d__02`, and all 8 records in `scene_two_zone_corridor_0202` — have **not** been evaluated with fixed `openrouter` / `qwen/qwen3.8-27b`. Accordingly, full-benchmark 100% accuracy is **not** claimed.
2. **Historical Reports Preserved As-Is:**
   - `reports/openrouter_full16/evaluation_3d_report.json` remains at `68.75%` overall accuracy (`5` empty-output parse errors under `openrouter/free`).
   - `reports/openrouter_qwen_pilot5/evaluation_3d_report.json` remains at `60.0%` overall accuracy (`2` `failed_request` reasoning-only failures under `max_tokens=1024`).
3. **Evaluator Selection Granularity:**
   - `Evaluation3DConfig` currently filters by `families` and `max_records` rather than arbitrary `record_id` / `question_id` lists. Retrying only the two `object_overlap_3d` failures worked cleanly because they were the first two records in `QuestionFamily3D.OBJECT_OVERLAP_3D`, but selective retries of later records in a family would require either `record_ids` filtering or a full run.
4. **Upstream Reasoning Budget Behavior:**
   - Even with `max_tokens=2048`, reasoning-capable VLMs can still exhaust completion tokens on harder scenes or numeric queries if reasoning traces exceed `2048` tokens, and `openrouter` free/shared endpoints remain subject to upstream capacity and rate limits (`max_retries=2`, capped `Retry-After` backoff up to `5.0s`, no inter-request delay).
