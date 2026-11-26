# 3D Spatial Benchmark Evaluation Summary

- **Dataset ID:** `warehouse_spatial_3d_v1`
- **Provider / Model:** `openai_compatible` / `openrouter/free`
- **Live Evaluation:** `True`
- **Evaluated At (UTC):** `2026-11-26T13:29:45.194710+00:00`
- **Sample Count:** 16
- **Completed Requests:** 16
- **Correct / Incorrect:** 11 / 0
- **Parse Errors / Failed Requests / Missing Images:** 5 / 0 / 0
- **Overall Accuracy:** 68.8%
- **Parsed-Only Accuracy:** 100.0%

## Accuracy by Question Family

| Family | Total | Correct | Incorrect | Parse Errors | Failed Requests | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `camera_visibility_3d` | 4 | 4 | 0 | 0 | 0 | 100.0% |
| `object_overlap_3d` | 4 | 3 | 0 | 1 | 0 | 75.0% |
| `vertical_clearance_3d` | 4 | 1 | 0 | 3 | 0 | 25.0% |
| `vertical_relation_3d` | 4 | 3 | 0 | 1 | 0 | 75.0% |

## Per-Question Results

| Question ID | Family | Status | Raw Prediction | Expected |
| --- | --- | --- | --- | --- |
| `wh3d_stacked_01__ov3d__01` | `object_overlap_3d` | `parse_error` | `-` | `True` |
| `wh3d_stacked_01__ov3d__02` | `object_overlap_3d` | `correct` | `false` | `False` |
| `wh3d_stacked_01__vrel3d__01` | `vertical_relation_3d` | `correct` | `true` | `True` |
| `wh3d_stacked_01__vrel3d__02` | `vertical_relation_3d` | `correct` | `false` | `False` |
| `wh3d_stacked_01__vclr3d__01` | `vertical_clearance_3d` | `correct` | `false` | `False` |
| `wh3d_stacked_01__vclr3d__02` | `vertical_clearance_3d` | `parse_error` | `-` | `True` |
| `wh3d_stacked_01__vis3d__01` | `camera_visibility_3d` | `correct` | `false` | `False` |
| `wh3d_stacked_01__vis3d__02` | `camera_visibility_3d` | `correct` | `true` | `True` |
| `scene_two_zone_corridor_0202__ov3d__01` | `object_overlap_3d` | `correct` | `false` | `False` |
| `scene_two_zone_corridor_0202__ov3d__02` | `object_overlap_3d` | `correct` | `true` | `True` |
| `scene_two_zone_corridor_0202__vrel3d__01` | `vertical_relation_3d` | `parse_error` | `-` | `True` |
| `scene_two_zone_corridor_0202__vrel3d__02` | `vertical_relation_3d` | `correct` | `false` | `False` |
| `scene_two_zone_corridor_0202__vclr3d__01` | `vertical_clearance_3d` | `parse_error` | `-` | `3.4` |
| `scene_two_zone_corridor_0202__vclr3d__02` | `vertical_clearance_3d` | `parse_error` | `-` | `1.5` |
| `scene_two_zone_corridor_0202__vis3d__01` | `camera_visibility_3d` | `correct` | `true` | `True` |
| `scene_two_zone_corridor_0202__vis3d__02` | `camera_visibility_3d` | `correct` | `true` | `True` |
