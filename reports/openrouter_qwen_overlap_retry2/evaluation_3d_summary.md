# 3D Spatial Benchmark Evaluation Summary

- **Dataset ID:** `warehouse_spatial_3d_v1`
- **Provider / Model:** `openrouter` / `qwen/qwen3.8-27b`
- **Live Evaluation:** `True`
- **Evaluated At (UTC):** `2026-11-26T15:32:02.487974+00:00`
- **Sample Count:** 2
- **Completed Requests:** 2
- **Correct / Incorrect:** 2 / 0
- **Parse Errors / Failed Requests / Missing Images:** 0 / 0 / 0
- **Overall Accuracy:** 100.0%
- **Parsed-Only Accuracy:** 100.0%

## Accuracy by Question Family

| Family | Total | Correct | Incorrect | Parse Errors | Failed Requests | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `object_overlap_3d` | 2 | 2 | 0 | 0 | 0 | 100.0% |

## Per-Question Results

| Question ID | Family | Status | Raw Prediction | Expected |
| --- | --- | --- | --- | --- |
| `wh3d_stacked_01__ov3d__01` | `object_overlap_3d` | `correct` | `true` | `True` |
| `wh3d_stacked_01__ov3d__02` | `object_overlap_3d` | `correct` | `false` | `False` |
