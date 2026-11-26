"""Lightweight local HTTP demo interface using Python's built-in http.server."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from visual_task_factory.demo_service import (
    BENCHMARK_3D_NOTICE,
    DemoSampleRequest,
    DemoService,
)
from visual_task_factory.scene_generator import SUPPORTED_TOPOLOGIES

_DEFAULT_DATASET_DIR = "artifacts/smoke_phase8_dataset"
_DEFAULT_EVAL_DIR = "artifacts/smoke_phase9_eval"
_DEFAULT_DEMO_OUTPUT_DIR = "artifacts/demo_sample"
_DEFAULT_DATASET_3D_DIR = "data/3d"


def _build_html_page(
    default_dataset_dir: str,
    default_eval_dir: str,
    default_dataset_3d_dir: str = _DEFAULT_DATASET_3D_DIR,
) -> str:
    topologies_options = "".join(
        f'<option value="{t}">{t}</option>' for t in SUPPORTED_TOPOLOGIES
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Visual Task Factory — Demo Interface</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #1e293b;
      --panel-alt: #0b1324;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --ok: #22c55e;
      --warn: #f59e0b;
      --err: #ef4444;
      --border: #334155;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    header {{
      background: #020617;
      border-bottom: 1px solid var(--border);
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    header h1 {{ margin: 0; font-size: 1.25rem; }}
    header .sub {{ color: var(--muted); font-size: 0.9rem; }}
    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 20px;
    }}
    main > section, .card, .stat, .img-wrap {{
      min-width: 0;
    }}
    @media (max-width: 960px) {{
      main {{ grid-template-columns: 1fr; }}
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .card h2 {{
      margin: 0 0 12px 0;
      font-size: 1.05rem;
      color: var(--accent);
    }}
    label {{
      display: block;
      font-size: 0.85rem;
      color: var(--muted);
      margin: 8px 0 4px 0;
    }}
    input, select, button {{
      width: 100%;
      padding: 8px 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--panel-alt);
      color: var(--text);
      font-size: 0.9rem;
    }}
    button {{
      background: #0284c7;
      border-color: #0369a1;
      font-weight: 600;
      cursor: pointer;
      margin-top: 12px;
    }}
    button:hover {{ background: #0369a1; }}
    button.secondary {{
      background: #334155;
      border-color: #475569;
    }}
    .row-inline {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 10px;
    }}
    .row-inline input[type="checkbox"] {{ width: auto; }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .stat {{
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
    }}
    .stat-wide {{ grid-column: span 2; }}
    @media (max-width: 420px) {{
      .stat-wide {{ grid-column: auto; }}
    }}
    .stat .k {{ font-size: 0.78rem; color: var(--muted); }}
    .stat .v {{
      font-size: 1.15rem;
      font-weight: 700;
      margin-top: 2px;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .banner-mock {{
      background: rgba(245, 158, 11, 0.16);
      border: 2px solid var(--warn);
      color: #fde68a;
      padding: 10px 14px;
      border-radius: 8px;
      font-weight: 700;
      margin-bottom: 12px;
    }}
    .banner-err {{
      background: rgba(239, 68, 68, 0.16);
      border: 1px solid var(--err);
      color: #fecaca;
      padding: 10px 12px;
      border-radius: 8px;
      margin-top: 8px;
      font-size: 0.9rem;
    }}
    .pill {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 600;
      background: #0f172a;
      border: 1px solid var(--border);
      margin: 0 6px 4px 0;
    }}
    .img-wrap {{
      background: #020617;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
      text-align: center;
      margin: 12px 0;
      max-width: 100%;
    }}
    .img-wrap img {{
      display: block;
      margin: 0 auto;
      max-width: 100%;
      width: auto;
      height: auto;
      border-radius: 4px;
    }}
    details {{
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      margin-top: 10px;
    }}
    summary {{ cursor: pointer; font-weight: 600; color: var(--ok); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      margin-top: 8px;
    }}
    th, td {{
      text-align: left;
      padding: 6px 8px;
      border-bottom: 1px solid var(--border);
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    code {{ color: #7dd3fc; }}
    .tabs {{
      max-width: 1320px;
      margin: 14px auto 0 auto;
      padding: 0 20px;
      display: flex;
      gap: 10px;
    }}
    .tab-btn {{
      width: auto;
      margin: 0;
      padding: 8px 16px;
      background: var(--panel);
      border: 1px solid var(--border);
      color: var(--muted);
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
    }}
    .tab-btn.active {{
      background: #0284c7;
      border-color: #0369a1;
      color: #ffffff;
    }}
    .nav-row {{
      display: flex;
      gap: 8px;
      align-items: center;
      margin-top: 10px;
    }}
    .nav-row button {{
      width: auto;
      min-width: 90px;
      margin-top: 0;
    }}
    .controls-3d {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: end;
      margin-bottom: 12px;
    }}
    @media (max-width: 720px) {{
      .controls-3d {{ grid-template-columns: 1fr; }}
    }}
    #refAnswerBox, #ref3dAnswerBox {{
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .banner-info {{
      background: rgba(56, 189, 248, 0.14);
      border: 1px solid var(--accent);
      color: #bae6fd;
      padding: 10px 14px;
      border-radius: 8px;
      font-weight: 600;
      margin-bottom: 12px;
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Visual Task Factory — Warehouse Spatial Reasoning</h1>
      <div class="sub">Innodata AI Hackathon 2026 | Deterministic Benchmark Pipeline Demo</div>
    </div>
    <div class="sub">Offline CPU Pipeline • Reference Answers Isolated From Model Prompts</div>
  </header>

  <nav class="tabs" aria-label="Benchmark Mode">
    <button id="tabBtn2d" class="tab-btn active" type="button" onclick="switchTab('2d')">2D Demo</button>
    <button id="tabBtn3d" class="tab-btn" type="button" onclick="switchTab('3d')">3D Benchmark</button>
  </nav>

  <div id="tab2d">
  <main>
    <section>
      <div class="card">
        <h2>1. Generate Small Demo Sample</h2>
        <label for="seed">Base Seed</label>
        <input id="seed" type="number" min="0" value="101" />

        <label for="sceneCount">Scenes (1–4)</label>
        <input id="sceneCount" type="number" min="1" max="4" value="2" />

        <label for="topology">Topology</label>
        <select id="topology">
          <option value="">All Topologies (Varied)</option>
          {topologies_options}
        </select>

        <label for="difficulty">Difficulty</label>
        <select id="difficulty">
          <option value="">All Difficulties (Varied)</option>
          <option value="easy">easy</option>
          <option value="medium">medium</option>
          <option value="hard">hard</option>
        </select>

        <div class="row-inline">
          <input id="runMockEval" type="checkbox" checked />
          <label for="runMockEval" style="margin:0">Run Synthetic Mock Evaluation</label>
        </div>

        <button id="btnGenerate" onclick="generateSample()">Generate Sample</button>
        <div id="genStatus" class="sub" style="margin-top:8px;"></div>
      </div>

      <div class="card">
        <h2>2. Load Existing Artifacts</h2>
        <label for="datasetDir">Dataset Directory</label>
        <input id="datasetDir" type="text" value="{default_dataset_dir}" />
        <button class="secondary" onclick="loadDataset()">Load Dataset</button>

        <label for="evalDir" style="margin-top:14px;">Evaluation Directory</label>
        <input id="evalDir" type="text" value="{default_eval_dir}" />
        <button class="secondary" onclick="loadEvaluation()">Load Evaluation &amp; Metrics</button>
      </div>
    </section>

    <section>
      <div class="card">
        <h2>Dataset &amp; Validation Summary</h2>
        <div id="datasetError"></div>
        <div id="datasetStats" class="metrics-grid"></div>
      </div>

      <div class="card">
        <h2>Visual Task Explorer</h2>
        <label for="recordSelect">Select Validated Question</label>
        <select id="recordSelect" onchange="renderSelectedRecord()"></select>
        <div id="questionCard" style="margin-top:10px;"></div>
        <div class="img-wrap" id="imageBox">Select or generate a dataset to view floorplan PNG.</div>
        <details id="refDetails">
          <summary>Show Verified Reference Answer (Isolated From Model Prompts)</summary>
          <div id="refAnswerBox" style="margin-top:8px;"></div>
        </details>
      </div>

      <div class="card">
        <h2>Evaluation, Metrics &amp; Failure Analysis</h2>
        <div id="evalBanner"></div>
        <div id="evalError"></div>
        <div id="evalStats" class="metrics-grid"></div>
        <div id="evalTables"></div>
      </div>
    </section>
  </main>
  </div>

  <div id="tab3d" style="display:none;">
    <main>
      <section>
        <div class="card">
          <h2>1. Load 3D Benchmark Dataset</h2>
          <label for="dataset3dDir">3D Dataset Directory</label>
          <input id="dataset3dDir" type="text" value="{default_dataset_3d_dir}" />
          <button id="btnLoad3d" type="button" onclick="loadDataset3d()">Load 3D Benchmark</button>
          <div class="sub" style="margin-top:8px;">
            Offline dataset inspection with recomputed CPU solver verification (no live VLM API calls).
          </div>
        </div>
      </section>

      <section>
        <div class="card">
          <h2>3D Benchmark Overview</h2>
          <div id="banner3dNotice" class="banner-info">{BENCHMARK_3D_NOTICE}</div>
          <div id="dataset3dError"></div>
          <div id="dataset3dStats" class="metrics-grid"></div>
        </div>

        <div class="card">
          <h2>3D Visual Task &amp; Solver Explorer</h2>
          <div class="controls-3d">
            <div>
              <label for="record3dSelect" style="margin-top:0;">Select 3D Question Record</label>
              <select id="record3dSelect" onchange="renderSelected3dRecord()"></select>
            </div>
            <div>
              <div class="nav-row" style="margin-top:0;">
                <button id="btnPrev3d" class="secondary" type="button" onclick="step3dRecord(-1)">Previous</button>
                <button id="btnNext3d" class="secondary" type="button" onclick="step3dRecord(1)">Next</button>
              </div>
              <div id="counter3d" class="sub" style="margin-top:6px;text-align:right;">Record 0 / 0</div>
            </div>
          </div>
          <div id="question3dCard"></div>
          <div class="img-wrap" id="image3dBox">Load 3D benchmark dataset to view isometric scene PNG.</div>
          <details id="ref3dDetails" open>
            <summary>Ground Truth &amp; Offline CPU Solver Verification</summary>
            <div id="ref3dAnswerBox" style="margin-top:8px;"></div>
          </details>
        </div>
      </section>
    </main>
  </div>

  <script>
    let currentDataset = null;
    let currentDataset3d = null;

    function switchTab(mode) {{
      const is3d = mode === "3d";
      document.getElementById("tab2d").style.display = is3d ? "none" : "block";
      document.getElementById("tab3d").style.display = is3d ? "block" : "none";
      document.getElementById("tabBtn2d").classList.toggle("active", !is3d);
      document.getElementById("tabBtn3d").classList.toggle("active", is3d);
      if (is3d && !currentDataset3d) {{
        loadDataset3d();
      }}
    }}

    function esc(str) {{
      return String(str ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }}

    async function loadDataset(dirOverride) {{
      const dir = dirOverride ?? document.getElementById("datasetDir").value;
      const res = await fetch("/api/dataset?dir=" + encodeURIComponent(dir));
      const data = await res.json();
      renderDataset(data);
    }}

    async function loadEvaluation(dirOverride) {{
      const dir = dirOverride ?? document.getElementById("evalDir").value;
      const res = await fetch("/api/evaluation?dir=" + encodeURIComponent(dir));
      const data = await res.json();
      renderEvaluation(data);
    }}

    async function generateSample() {{
      const btn = document.getElementById("btnGenerate");
      const status = document.getElementById("genStatus");
      btn.disabled = true;
      status.textContent = "Generating sample scenes, PNGs, questions, and validation...";
      try {{
        const payload = {{
          base_seed: Number(document.getElementById("seed").value || 0),
          scene_count: Number(document.getElementById("sceneCount").value || 2),
          topology_id: document.getElementById("topology").value || null,
          difficulty: document.getElementById("difficulty").value || null,
          run_mock_eval: document.getElementById("runMockEval").checked
        }};
        const res = await fetch("/api/generate", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload)
        }});
        const data = await res.json();
        if (!res.ok) {{
          status.textContent = "Error: " + (data.error || res.statusText);
          return;
        }}
        document.getElementById("datasetDir").value = data.dataset.dataset_dir;
        renderDataset(data.dataset);
        if (data.evaluation) {{
          document.getElementById("evalDir").value = data.evaluation.eval_dir;
          renderEvaluation(data.evaluation);
        }}
        status.textContent = "Sample generated cleanly.";
      }} finally {{
        btn.disabled = false;
      }}
    }}

    function renderDataset(ds) {{
      currentDataset = ds;
      const errBox = document.getElementById("datasetError");
      const statsBox = document.getElementById("datasetStats");
      if (ds.error_message) {{
        errBox.innerHTML = `<div class="banner-err">${{esc(ds.error_message)}}</div>`;
      }} else if (ds.missing_files && ds.missing_files.length) {{
        errBox.innerHTML = `<div class="banner-err">Missing artifacts: ${{esc(ds.missing_files.join(", "))}}</div>`;
      }} else {{
        errBox.innerHTML = "";
      }}

      const m = ds.manifest;
      if (m) {{
        statsBox.innerHTML = `
          <div class="stat"><div class="k">Dataset ID</div><div class="v">${{esc(m.dataset_id)}}</div></div>
          <div class="stat"><div class="k">Generated Scenes</div><div class="v">${{m.generated_scene_count}} / ${{m.requested_scene_count}}</div></div>
          <div class="stat"><div class="k">Unique Layouts</div><div class="v">${{m.unique_structural_signatures}}</div></div>
          <div class="stat"><div class="k">Valid Questions</div><div class="v" style="color:var(--ok)">${{m.valid_item_count}}</div></div>
          <div class="stat"><div class="k">Invalid Excluded</div><div class="v">${{m.invalid_item_count}}</div></div>
          <div class="stat"><div class="k">Validation Warnings</div><div class="v">${{m.warning_count}}</div></div>
        `;
      }} else {{
        statsBox.innerHTML = `<div class="stat"><div class="k">Loaded Records</div><div class="v">${{ds.records.length}}</div></div>`;
      }}

      const sel = document.getElementById("recordSelect");
      sel.innerHTML = ds.records.map((r, i) =>
        `<option value="${{i}}">[${{esc(r.family)}}] ${{esc(r.question_id)}}</option>`
      ).join("");
      renderSelectedRecord();
    }}

    function renderSelectedRecord() {{
      const idx = Number(document.getElementById("recordSelect").value || 0);
      const qCard = document.getElementById("questionCard");
      const imgBox = document.getElementById("imageBox");
      const refBox = document.getElementById("refAnswerBox");
      document.getElementById("refDetails").open = false;

      if (!currentDataset || !currentDataset.records || !currentDataset.records.length) {{
        qCard.innerHTML = "<div class='sub'>No validated records loaded.</div>";
        imgBox.textContent = "No image available.";
        refBox.textContent = "";
        return;
      }}
      const rec = currentDataset.records[idx] || currentDataset.records[0];
      const choicesHtml = rec.candidate_choices
        ? `<div class="sub" style="margin-top:6px;">Candidate Choices: <code>${{esc(rec.candidate_choices.join(", "))}}</code></div>`
        : "";
      qCard.innerHTML = `
        <div>
          <span class="pill">Family: ${{esc(rec.family)}}</span>
          <span class="pill">Answer Type: ${{esc(rec.expected_answer_type)}}</span>
          <span class="pill">Topology: ${{esc(rec.variation.topology_id)}}</span>
          <span class="pill">Difficulty: ${{esc(rec.variation.difficulty)}}</span>
          <span class="pill">Orient: ${{rec.variation.orientation_deg}}°</span>
        </div>
        <p style="font-size:1.02rem;margin:10px 0 4px 0;"><strong>Question:</strong> ${{esc(rec.prompt_text)}}</p>
        ${{choicesHtml}}
      `;
      const imgUrl = `/api/image?dataset_dir=${{encodeURIComponent(currentDataset.dataset_dir)}}&image_rel_path=${{encodeURIComponent(rec.image_rel_path)}}`;
      imgBox.innerHTML = `<img src="${{imgUrl}}" alt="Warehouse floorplan ${{esc(rec.scene_id)}}" />`;

      const ref = rec.reference_answer;
      const unitStr = ref.unit ? ` ${{esc(ref.unit)}}` : "";
      const tolStr = ref.numeric_tolerance !== null && ref.numeric_tolerance !== undefined
        ? ` (±${{ref.numeric_tolerance}} m)` : "";
      refBox.innerHTML = `
        <div><strong>Answer:</strong> <code>${{esc(ref.value)}}${{unitStr}}</code>${{tolStr}}</div>
        <div class="sub" style="margin-top:4px;">Solver: <code>${{esc(ref.solver_name)}}</code> | Validated: ${{rec.validation.is_valid}}</div>
        <div class="sub" style="margin-top:4px;">Trace: ${{esc((ref.reasoning_trace || []).join(" → "))}}</div>
      `;
    }}

    function renderEvaluation(ev) {{
      const bannerBox = document.getElementById("evalBanner");
      const errBox = document.getElementById("evalError");
      const statsBox = document.getElementById("evalStats");
      const tablesBox = document.getElementById("evalTables");

      if (ev.mock_warning_banner) {{
        bannerBox.innerHTML = `<div class="banner-mock">⚠ ${{esc(ev.mock_warning_banner)}}</div>`;
      }} else {{
        bannerBox.innerHTML = "";
      }}

      if (ev.error_message) {{
        errBox.innerHTML = `<div class="banner-err">${{esc(ev.error_message)}}</div>`;
      }} else if (ev.missing_files && ev.missing_files.length) {{
        errBox.innerHTML = `<div class="banner-err">Missing evaluation/metrics files: ${{esc(ev.missing_files.join(", "))}}</div>`;
      }} else {{
        errBox.innerHTML = "";
      }}

      const ms = ev.metrics_summary;
      const es = ev.evaluation_summary;
      if (ms) {{
        statsBox.innerHTML = `
          <div class="stat"><div class="k">Model / Provider</div><div class="v" style="font-size:0.95rem">${{esc(ms.model_name)}} / ${{esc(ms.provider_name)}}</div></div>
          <div class="stat"><div class="k">Evaluation Nature</div><div class="v">${{esc(ms.evaluation_nature)}}</div></div>
          <div class="stat"><div class="k">Total Items</div><div class="v">${{ms.total_items}}</div></div>
          <div class="stat"><div class="k">Accuracy</div><div class="v">${{(ms.accuracy * 100).toFixed(1)}}%</div></div>
          <div class="stat"><div class="k">Unparseable / Errors</div><div class="v">${{ms.unparseable_count}} / ${{ms.adapter_error_count}}</div></div>
          <div class="stat"><div class="k">Failures</div><div class="v">${{ms.failure_count}}</div></div>
        `;
      }} else if (es) {{
        statsBox.innerHTML = `
          <div class="stat"><div class="k">Model</div><div class="v">${{esc(es.model_name)}}</div></div>
          <div class="stat"><div class="k">Total Records</div><div class="v">${{es.total_records}}</div></div>
          <div class="stat"><div class="k">Raw Accuracy</div><div class="v">${{(es.raw_accuracy * 100).toFixed(1)}}%</div></div>
        `;
      }} else {{
        statsBox.innerHTML = "";
      }}

      let html = "";
      if (ms && ms.by_family) {{
        html += "<h3>Accuracy by Question Family</h3><table><tr><th>Family</th><th>Total</th><th>Correct</th><th>Accuracy</th></tr>";
        for (const [k, v] of Object.entries(ms.by_family)) {{
          html += `<tr><td>${{esc(k)}}</td><td>${{v.total}}</td><td>${{v.correct}}</td><td>${{(v.accuracy * 100).toFixed(1)}}%</td></tr>`;
        }}
        html += "</table>";
      }}
      if (ev.failure_analysis && ev.failure_analysis.category_counts) {{
        html += "<h3 style='margin-top:14px;'>Failure Analysis</h3><table><tr><th>Category</th><th>Count</th><th>Rate</th></tr>";
        for (const [k, cnt] of Object.entries(ev.failure_analysis.category_counts)) {{
          const rate = (ev.failure_analysis.category_rates[k] || 0) * 100;
          html += `<tr><td>${{esc(k)}}</td><td>${{cnt}}</td><td>${{rate.toFixed(1)}}%</td></tr>`;
        }}
        html += "</table>";
      }}
      if (ev.failure_examples && ev.failure_examples.length) {{
        html += "<h3 style='margin-top:14px;'>Sample Failure Records</h3><table><tr><th>Question ID</th><th>Family</th><th>Ref</th><th>Pred</th><th>Category</th></tr>";
        for (const fx of ev.failure_examples.slice(0, 10)) {{
          html += `<tr><td><code>${{esc(fx.question_id)}}</code></td><td>${{esc(fx.family)}}</td><td>${{esc(fx.reference_answer.value)}}</td><td>${{esc(fx.raw_prediction)}}</td><td>${{esc(fx.failure_category)}}</td></tr>`;
        }}
        html += "</table>";
      }}
      tablesBox.innerHTML = html;
    }}

    async function loadDataset3d(dirOverride) {{
      const btn = document.getElementById("btnLoad3d");
      const errBox = document.getElementById("dataset3dError");
      const dir = dirOverride ?? document.getElementById("dataset3dDir").value;
      btn.disabled = true;
      try {{
        const res = await fetch("/api/dataset_3d?dir=" + encodeURIComponent(dir));
        if (!res.ok) {{
          throw new Error("HTTP " + res.status + " " + res.statusText);
        }}
        const data = await res.json();
        renderDataset3d(data);
      }} catch (err) {{
        errBox.innerHTML = `<div class="banner-err">Failed to load 3D dataset: ${{esc(err && err.message ? err.message : err)}}</div>`;
      }} finally {{
        btn.disabled = false;
      }}
    }}

    function renderDataset3d(ds3d) {{
      currentDataset3d = ds3d;
      const noticeBox = document.getElementById("banner3dNotice");
      const errBox = document.getElementById("dataset3dError");
      const statsBox = document.getElementById("dataset3dStats");
      noticeBox.textContent = ds3d.benchmark_notice || "{BENCHMARK_3D_NOTICE}";

      if (ds3d.error_message) {{
        errBox.innerHTML = `<div class="banner-err">${{esc(ds3d.error_message)}}</div>`;
      }} else {{
        errBox.innerHTML = "";
      }}

      statsBox.innerHTML = `
        <div class="stat stat-wide"><div class="k">Dataset ID</div><div class="v" style="font-size:0.95rem">${{esc(ds3d.dataset_id || "-")}}</div></div>
        <div class="stat"><div class="k">Scenes</div><div class="v">${{ds3d.scene_count}}</div></div>
        <div class="stat"><div class="k">Total 3D Records</div><div class="v">${{ds3d.record_count}}</div></div>
        <div class="stat"><div class="k">Solver Matched</div><div class="v" style="color:var(--ok)">${{ds3d.matched_count}} / ${{ds3d.record_count}}</div></div>
        <div class="stat"><div class="k">QA Validated</div><div class="v">${{ds3d.valid_count}} / ${{ds3d.record_count}}</div></div>
      `;

      const sel = document.getElementById("record3dSelect");
      sel.innerHTML = (ds3d.records || []).map((r, i) =>
        `<option value="${{i}}" title="${{esc(r.record_id)}}">${{i + 1}}. ${{esc(r.question_id)}} (${{esc(r.family)}})</option>`
      ).join("");
      renderSelected3dRecord();
    }}

    function step3dRecord(delta) {{
      if (!currentDataset3d || !currentDataset3d.records || !currentDataset3d.records.length) {{
        return;
      }}
      const sel = document.getElementById("record3dSelect");
      const total = currentDataset3d.records.length;
      const nextIdx = (Number(sel.value || 0) + delta + total) % total;
      sel.value = String(nextIdx);
      renderSelected3dRecord();
    }}

    function renderSelected3dRecord() {{
      const sel = document.getElementById("record3dSelect");
      const idx = Number(sel.value || 0);
      const qCard = document.getElementById("question3dCard");
      const imgBox = document.getElementById("image3dBox");
      const refBox = document.getElementById("ref3dAnswerBox");
      const counter = document.getElementById("counter3d");

      const recs = (currentDataset3d && currentDataset3d.records) || [];
      if (!recs.length) {{
        counter.textContent = "Record 0 / 0";
        qCard.innerHTML = "<div class='sub'>No 3D benchmark records loaded.</div>";
        imgBox.textContent = "No 3D scene image available.";
        refBox.textContent = "";
        return;
      }}

      const safeIdx = Math.max(0, Math.min(idx, recs.length - 1));
      const rec = recs[safeIdx];
      sel.title = rec.record_id;
      counter.textContent = `Record ${{safeIdx + 1}} / ${{recs.length}}`;

      qCard.innerHTML = `
        <div>
          <span class="pill">Scene: ${{esc(rec.scene_id)}}</span>
          <span class="pill">Family: ${{esc(rec.family)}}</span>
          <span class="pill">Answer Type: ${{esc(rec.expected_answer_type)}}</span>
          <span class="pill">Resolution: ${{rec.image_width_px}}x${{rec.image_height_px}}</span>
        </div>
        <p style="font-size:1.02rem;margin:10px 0 4px 0;"><strong>Question:</strong> ${{esc(rec.prompt_text)}}</p>
      `;

      const imgUrl = `/api/image?dataset_dir=${{encodeURIComponent(currentDataset3d.dataset_dir)}}&image_rel_path=${{encodeURIComponent(rec.image_rel_path)}}`;
      imgBox.innerHTML = `<img src="${{imgUrl}}" alt="3D warehouse scene ${{esc(rec.scene_id)}}" />`;

      const gt = rec.reference_answer;
      const gtUnit = gt.unit ? ` ${{esc(gt.unit)}}` : "";
      const gtTol = gt.numeric_tolerance !== null && gt.numeric_tolerance !== undefined
        ? ` (±${{gt.numeric_tolerance}} m)` : "";

      const sp = rec.solver_prediction;
      const spUnit = sp.unit ? ` ${{esc(sp.unit)}}` : "";
      const matchBadge = rec.solver_matches_ground_truth
        ? `<span style="color:var(--ok);font-weight:700;">MATCH</span>`
        : `<span style="color:var(--err);font-weight:700;">MISMATCH</span>`;

      refBox.innerHTML = `
        <div><strong>Ground-Truth Answer:</strong> <code>${{esc(gt.value)}}${{gtUnit}}</code>${{gtTol}}</div>
        <div style="margin-top:4px;"><strong>Recomputed CPU Solver Answer:</strong> <code>${{esc(sp.value)}}${{spUnit}}</code> • ${{matchBadge}}</div>
        <div class="sub" style="margin-top:6px;">Solver: <code>${{esc(sp.solver_name || gt.solver_name)}}</code> | QA Validated: ${{rec.validation_is_valid}}</div>
        <div class="sub" style="margin-top:6px;"><strong>Reasoning Trace:</strong> ${{esc((sp.reasoning_trace && sp.reasoning_trace.length ? sp.reasoning_trace : gt.reasoning_trace || []).join(" → "))}}</div>
      `;
    }}

    window.addEventListener("DOMContentLoaded", () => {{
      loadDataset();
      loadEvaluation();
      loadDataset3d();
    }});
  </script>
</body>
</html>"""


def create_demo_handler(
    service: DemoService | None = None,
    *,
    default_dataset_dir: str = _DEFAULT_DATASET_DIR,
    default_eval_dir: str = _DEFAULT_EVAL_DIR,
    demo_output_dir: Path | str = _DEFAULT_DEMO_OUTPUT_DIR,
    default_dataset_3d_dir: str = _DEFAULT_DATASET_3D_DIR,
) -> type[BaseHTTPRequestHandler]:
    """Create a configured HTTP request handler bound to DemoService."""
    svc = service or DemoService()
    demo_out = Path(demo_output_dir)

    class DemoRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # Keep server output clean and quiet.

        def _send_json(self, status: HTTPStatus, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)

            if parsed.path == "/":
                html_bytes = _build_html_page(
                    default_dataset_dir,
                    default_eval_dir,
                    default_dataset_3d_dir,
                ).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)
                return

            if parsed.path == "/api/dataset":
                ds_dir = qs.get("dir", [default_dataset_dir])[0]
                view = svc.load_dataset_view(ds_dir)
                self._send_json(HTTPStatus.OK, view.model_dump(mode="json"))
                return

            if parsed.path == "/api/dataset_3d":
                ds3d_dir = qs.get("dir", [default_dataset_3d_dir])[0]
                view3d = svc.load_dataset_3d_view(ds3d_dir)
                self._send_json(HTTPStatus.OK, view3d.model_dump(mode="json"))
                return

            if parsed.path == "/api/evaluation":
                ev_dir = qs.get("dir", [default_eval_dir])[0]
                view = svc.load_evaluation_view(ev_dir)
                self._send_json(HTTPStatus.OK, view.model_dump(mode="json"))
                return

            if parsed.path == "/api/image":
                ds_dir = qs.get("dataset_dir", [""])[0]
                rel_path = qs.get("image_rel_path", [""])[0]
                try:
                    img_path = svc.resolve_safe_image_path(ds_dir, rel_path)
                except PermissionError as exc:
                    self._send_json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
                    return
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                except FileNotFoundError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                    return

                img_bytes = img_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(img_bytes)))
                self.end_headers()
                self.wfile.write(img_bytes)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/generate":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
                return

            try:
                content_len = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length."})
                return

            raw_body = self.rfile.read(max(0, content_len))
            try:
                body_dict = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                req = DemoSampleRequest.model_validate(body_dict)
            except (json.JSONDecodeError, ValidationError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            target_dir = demo_out / f"sample_s{req.base_seed}_n{req.scene_count}"
            res = svc.generate_demo_sample(req, target_dir)
            self._send_json(HTTPStatus.OK, res.model_dump(mode="json"))

    return DemoRequestHandler


def create_demo_server(
    host: str = "127.0.0.1",
    port: int = 8501,
    *,
    default_dataset_dir: str = _DEFAULT_DATASET_DIR,
    default_eval_dir: str = _DEFAULT_EVAL_DIR,
    demo_output_dir: Path | str = _DEFAULT_DEMO_OUTPUT_DIR,
    default_dataset_3d_dir: str = _DEFAULT_DATASET_3D_DIR,
) -> ThreadingHTTPServer:
    """Instantiate a ThreadingHTTPServer serving the Visual Task Factory demo UI."""
    handler = create_demo_handler(
        DemoService(),
        default_dataset_dir=default_dataset_dir,
        default_eval_dir=default_eval_dir,
        demo_output_dir=demo_output_dir,
        default_dataset_3d_dir=default_dataset_3d_dir,
    )
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the lightweight Visual Task Factory local demo server."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--dataset-dir", default=_DEFAULT_DATASET_DIR)
    parser.add_argument("--eval-dir", default=_DEFAULT_EVAL_DIR)
    parser.add_argument("--demo-output-dir", default=_DEFAULT_DEMO_OUTPUT_DIR)
    parser.add_argument("--dataset-3d-dir", default=_DEFAULT_DATASET_3D_DIR)
    args = parser.parse_args()

    server = create_demo_server(
        host=args.host,
        port=args.port,
        default_dataset_dir=args.dataset_dir,
        default_eval_dir=args.eval_dir,
        demo_output_dir=args.demo_output_dir,
        default_dataset_3d_dir=args.dataset_3d_dir,
    )
    print(f"Visual Task Factory Demo UI running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
