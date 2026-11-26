# Visual Task Factory — Suggested Live Demo Walkthrough (3–5 Minutes)

1. **Start the local demo server:**
   ```powershell
   $env:PYTHONPATH="src"
   .\.venv\Scripts\python.exe -m visual_task_factory.demo_app --port 8501
   ```
   Open `http://127.0.0.1:8501/` in a browser.

2. **Point out the preloaded benchmark & synthetic mock notice (30s):**
   - Show the **Dataset & Validation Summary** (`generated_scene_count`, `unique_structural_signatures`, `valid_item_count`, `invalid_item_count`).
   - Highlight the prominent amber **`SYNTHETIC MOCK EVALUATION — NOT REAL MODEL CAPABILITY`** banner to demonstrate strict separation between synthetic pipeline checks and real VLM capability.

3. **Walk through all 3 spatial reasoning families (90s):**
   - In **Visual Task Explorer**, select one question from each family:
     - `constrained_reachability` (`exit_id` answer — show hazard zones, walls, open/blocked doors, and exit badges on the floorplan).
     - `region_containment_counting` (`integer` answer — show racks inside `Storage Bay A`).
     - `scale_grounded_clearance` (`boolean` / `numeric` in meters — point out the footer scale bar).
   - Keep the **Verified Reference Answer** collapsed while explaining the prompt (showing zero coordinate/answer leakage), then expand it to show the solver name (`grid_dijkstra_v1`, `rect_containment_v1`, `euclidean_clearance_v1`) and step-by-step reasoning trace.

4. **Generate a fresh sample live (60s):**
   - In **Generate Small Demo Sample**, pick a new seed (e.g., `2026`), set `Scenes = 2`, choose a topology (`warehouse_cross_aisle_v1` or `All Topologies`), leave `Run Synthetic Mock Evaluation` checked, and click **Generate Sample**.
   - Show the newly generated PNGs, updated validation counts, and refreshed metrics/failure tables.

5. **Wrap up with test & architecture guarantees (30s):**
   - Optionally run `.\.venv\Scripts\python.exe -m pytest` in a terminal to show all **74 automated tests** passing across schemas, generation, rendering, solver, validation, variation, dataset export, evaluator, metrics, and demo endpoints.
