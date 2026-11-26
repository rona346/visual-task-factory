# 3D Spatial Benchmark Evaluation Summary

- **Dataset ID:** `warehouse_spatial_3d_v1`
- **Provider / Model:** `openrouter` / `qwen/qwen3.8-27b`
- **Live Evaluation:** `True`
- **Evaluated At (UTC):** `2026-11-26T15:24:47.651274+00:00`
- **Sample Count:** 5
- **Completed Requests:** 3
- **Correct / Incorrect:** 3 / 0
- **Parse Errors / Failed Requests / Missing Images:** 0 / 2 / 0
- **Overall Accuracy:** 60.0%
- **Parsed-Only Accuracy:** 100.0%

## Accuracy by Question Family

| Family | Total | Correct | Incorrect | Parse Errors | Failed Requests | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `object_overlap_3d` | 2 | 0 | 0 | 0 | 2 | 0.0% |
| `vertical_clearance_3d` | 1 | 1 | 0 | 0 | 0 | 100.0% |
| `vertical_relation_3d` | 2 | 2 | 0 | 0 | 0 | 100.0% |

## Per-Question Results

| Question ID | Family | Status | Raw Prediction | Expected |
| --- | --- | --- | --- | --- |
| `wh3d_stacked_01__ov3d__01` | `object_overlap_3d` | `failed_request` | `-` | `True` |
| `wh3d_stacked_01__ov3d__02` | `object_overlap_3d` | `failed_request` | `-` | `False` |
| `wh3d_stacked_01__vrel3d__01` | `vertical_relation_3d` | `correct` | `true` | `True` |
| `wh3d_stacked_01__vrel3d__02` | `vertical_relation_3d` | `correct` | `false` | `False` |
| `wh3d_stacked_01__vclr3d__01` | `vertical_clearance_3d` | `correct` | `false` | `False` |
