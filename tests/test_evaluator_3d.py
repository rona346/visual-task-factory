"""Unit tests for isolated 3D VLM evaluation framework (Phase 6 & Phase 9 remediation)."""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
import pytest

from visual_task_factory.dataset_builder_3d import (
    Dataset3DConfig,
    DatasetBuilder3D,
    build_default_3d_scenes,
)
from visual_task_factory.evaluator_3d import (
    DEFAULT_GROQ_VISION_MODEL,
    GROQ_BASE_URL,
    OPENROUTER_BASE_URL,
    Eval3DStatus,
    Evaluation3DConfig,
    Evaluation3DReport,
    Evaluation3DRunner,
    Evaluator3DError,
    MissingCredentials3DError,
    MissingModelConfig3DError,
    OpenAICompatibleVLM3DAdapter,
    UnsupportedProvider3DError,
    VLM3DAdapter,
    VLM3DRequest,
    VLM3DResponse,
    create_live_vlm_adapter_from_env,
    normalize_prediction_3d,
    score_prediction_3d,
)
from visual_task_factory.qa_3d import AnswerType3D, QuestionFamily3D, ReferenceAnswer3D


class _StubProviderAdapter(VLM3DAdapter):
    """Test-only adapter mimicking provider outputs without solver access."""

    def __init__(
        self,
        responses_by_qid: dict[str, str],
        *,
        fail_on_qids: set[str] | None = None,
        timeout_on_qids: set[str] | None = None,
    ) -> None:
        self._responses = dict(responses_by_qid)
        self._fail_on_qids = set(fail_on_qids or set())
        self._timeout_on_qids = set(timeout_on_qids or set())
        self.seen_requests: list[VLM3DRequest] = []

    @property
    def model_name(self) -> str:
        return "test_stub_vlm"

    @property
    def provider_name(self) -> str:
        return "test_provider"

    @property
    def is_live(self) -> bool:
        return False

    def predict(self, request: VLM3DRequest) -> VLM3DResponse:
        assert not hasattr(request, "reference_answer")
        assert not hasattr(request, "solver_query")
        assert request.image_path.is_file()
        self.seen_requests.append(request)

        if request.question_id in self._timeout_on_qids:
            raise TimeoutError("Simulated request timeout for unit test.")
        if request.question_id in self._fail_on_qids:
            raise RuntimeError(
                "Upstream provider error with fake keys sk-testsecret123456 and gsk_testsecret987654."
            )

        return VLM3DResponse(
            raw_output=self._responses.get(request.question_id, "maybe"),
            model_name=self.model_name,
            provider_name=self.provider_name,
            is_live=False,
            latency_ms=12.5,
        )


def _clear_all_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "VLM_PROVIDER",
        "VLM_MODEL",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_fake_png_request(tmp_path: Path) -> tuple[VLM3DRequest, bytes]:
    png_path = tmp_path / "scene.png"
    raw_bytes = b"\x89PNG\r\n\x1a\nfake-3d-scene-png"
    png_path.write_bytes(raw_bytes)
    req = VLM3DRequest(
        record_id="rec_1",
        scene_id="wh3d_test",
        question_id="q_1",
        family=QuestionFamily3D.VERTICAL_CLEARANCE_3D,
        image_path=png_path,
        prompt_text="What is the vertical clearance in meters between Rack 1 and ceiling?",
        expected_answer_type=AnswerType3D.NUMERIC,
        unit="m",
    )
    return req, raw_bytes


def test_conservative_prediction_normalization_and_scoring() -> None:
    assert normalize_prediction_3d("true", AnswerType3D.BOOLEAN).parsed_value is True
    assert normalize_prediction_3d('{"answer": "No"}', AnswerType3D.BOOLEAN).parsed_value is False
    assert normalize_prediction_3d("maybe true", AnswerType3D.BOOLEAN).is_valid is False

    num_res = normalize_prediction_3d("1.75 m", AnswerType3D.NUMERIC, expected_unit="m")
    assert num_res.is_valid is True
    assert num_res.parsed_value == pytest.approx(1.75)
    assert num_res.parsed_unit == "m"

    assert normalize_prediction_3d("50 cm", AnswerType3D.NUMERIC, expected_unit="m").is_valid is False
    assert normalize_prediction_3d("about two meters", AnswerType3D.NUMERIC).is_valid is False

    bool_ref = ReferenceAnswer3D(answer_type=AnswerType3D.BOOLEAN, value=True)
    assert score_prediction_3d(True, bool_ref) is True
    assert score_prediction_3d(False, bool_ref) is False

    num_ref = ReferenceAnswer3D(
        answer_type=AnswerType3D.NUMERIC, value=1.5, unit="m", numeric_tolerance=0.05
    )
    assert score_prediction_3d(1.54, num_ref) is True
    assert score_prediction_3d(1.60, num_ref) is False


def test_evaluation_3d_scoring_parse_errors_failures_family_metrics_and_reports(
    tmp_path: Path,
) -> None:
    ds_dir = tmp_path / "data" / "3d"
    report_dir = tmp_path / "reports" / "3d_evaluation"
    scenes = build_default_3d_scenes()[:1]
    build_res = DatasetBuilder3D().build_from_scenes(
        scenes,
        output_dir=ds_dir,
        config=Dataset3DConfig(
            dataset_id="eval3d_test_ds",
            image_width_px=480,
            image_height_px=360,
            supersample=1,
            max_questions_per_family=2,
        ),
    )

    records = build_res.bundle.records
    assert len(records) >= 5

    scripted: dict[str, str] = {}
    fail_qids: set[str] = set()
    timeout_qids: set[str] = set()

    for idx, rec in enumerate(records):
        val = rec.reference_answer.value
        if idx == 1:
            if rec.expected_answer_type == AnswerType3D.BOOLEAN:
                scripted[rec.question_id] = "false" if val is True else "true"
            else:
                scripted[rec.question_id] = f"{float(val) + 2.0:.2f} m"
        elif idx == 2:
            scripted[rec.question_id] = "cannot tell from angle"
        elif idx == 3:
            fail_qids.add(rec.question_id)
        elif idx == 4:
            timeout_qids.add(rec.question_id)
        else:
            if rec.expected_answer_type == AnswerType3D.BOOLEAN:
                scripted[rec.question_id] = "true" if val is True else "false"
            else:
                scripted[rec.question_id] = f"{float(val):.2f} m"

    adapter = _StubProviderAdapter(
        scripted,
        fail_on_qids=fail_qids,
        timeout_on_qids=timeout_qids,
    )
    run_res = Evaluation3DRunner(adapter).evaluate(
        ds_dir,
        output_dir=report_dir,
        config=Evaluation3DConfig(max_records=len(records)),
    )

    rep = run_res.report
    assert rep.dataset_id == "eval3d_test_ds"
    assert rep.model_name == "test_stub_vlm"
    assert rep.provider_name == "test_provider"
    assert rep.sample_count == len(records)
    assert rep.completed_requests == len(records) - 2
    assert rep.incorrect_count == 1
    assert rep.parse_error_count == 1
    assert rep.failed_request_count == 2
    assert rep.correct_count == len(records) - 4
    assert rep.overall_accuracy == pytest.approx(rep.correct_count / rep.sample_count, abs=1e-6)
    assert sum(m.total for m in rep.by_family.values()) == rep.sample_count
    assert QuestionFamily3D.OBJECT_OVERLAP_3D.value in rep.by_family

    json_text = run_res.json_report_path.read_text(encoding="utf-8")
    md_text = run_res.markdown_summary_path.read_text(encoding="utf-8")
    assert "sk-testsecret" not in json_text
    assert "gsk_testsecret" not in json_text
    assert "gsk_testsecret" not in md_text
    assert "[REDACTED]" in json_text

    reloaded = Evaluation3DReport.model_validate_json(json_text)
    assert reloaded == rep
    assert reloaded.predictions[2].status == Eval3DStatus.PARSE_ERROR
    assert reloaded.predictions[3].status == Eval3DStatus.FAILED_REQUEST
    assert reloaded.predictions[4].status == Eval3DStatus.FAILED_REQUEST


def test_explicit_provider_selection_groq_and_openrouter_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_all_provider_env(monkeypatch)

    # Missing VLM_PROVIDER must fail even if Gemini/OpenAI/provider keys exist.
    monkeypatch.setenv("GEMINI_API_KEY", "placeholder-gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder-openai")
    with pytest.raises(UnsupportedProvider3DError):
        create_live_vlm_adapter_from_env()

    # Unsupported providers (gemini, openai, custom) must be rejected.
    for bad_provider in ("gemini", "openai", "anthropic"):
        monkeypatch.setenv("VLM_PROVIDER", bad_provider)
        with pytest.raises(UnsupportedProvider3DError):
            create_live_vlm_adapter_from_env()

    # Groq selection uses verified Llama-4 Scout vision default and requires GROQ_API_KEY.
    monkeypatch.setenv("VLM_PROVIDER", "groq")
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder-openrouter")
    with pytest.raises(MissingCredentials3DError):
        create_live_vlm_adapter_from_env()  # Must not fall back to OpenRouter.

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_placeholder_0001")
    groq_adapter = create_live_vlm_adapter_from_env()
    assert groq_adapter.provider_name == "groq"
    assert groq_adapter.model_name == DEFAULT_GROQ_VISION_MODEL
    assert groq_adapter.is_live is True
    assert isinstance(groq_adapter, OpenAICompatibleVLM3DAdapter)
    assert groq_adapter._endpoint == f"{GROQ_BASE_URL}/chat/completions"

    # OpenRouter selection requires OPENROUTER_API_KEY and explicit fixed VLM_MODEL.
    monkeypatch.setenv("VLM_PROVIDER", "openrouter")
    monkeypatch.delenv("VLM_MODEL", raising=False)
    with pytest.raises(MissingModelConfig3DError):
        create_live_vlm_adapter_from_env()

    monkeypatch.setenv("VLM_MODEL", "openrouter/free")
    with pytest.raises(MissingModelConfig3DError):
        create_live_vlm_adapter_from_env()

    monkeypatch.setenv("VLM_MODEL", "qwen/qwen2.5-vl-72b-instruct:free")
    or_adapter = create_live_vlm_adapter_from_env()
    assert or_adapter.provider_name == "openrouter"
    assert or_adapter.model_name == "qwen/qwen2.5-vl-72b-instruct:free"
    assert or_adapter._endpoint == f"{OPENROUTER_BASE_URL}/chat/completions"

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingCredentials3DError):
        create_live_vlm_adapter_from_env()  # Must not fall back to Groq.


def test_groq_image_payload_and_response_plus_report_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[dict[str, object]] = []

    class _FakeHttpResponse:
        status_code = 200
        headers: dict[str, str] = {}
        text = '{"ok": true}'

        def json(self) -> dict[str, object]:
            return {
                "model": DEFAULT_GROQ_VISION_MODEL,
                "choices": [{"message": {"content": " 1.5 m "}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 4, "total_tokens": 204},
            }

    def _fake_post(url: str, *, json: dict[str, object], headers: dict[str, str], timeout: float):
        captured.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return _FakeHttpResponse()

    _clear_all_provider_env(monkeypatch)
    monkeypatch.setenv("VLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_placeholder_0002")
    monkeypatch.setattr(httpx, "post", _fake_post)

    adapter = create_live_vlm_adapter_from_env(timeout_s=22.0)
    req, raw_png = _make_fake_png_request(tmp_path)
    resp = adapter.predict(req)

    assert len(captured) == 1
    assert captured[0]["url"] == f"{GROQ_BASE_URL}/chat/completions"
    assert captured[0]["timeout"] == 22.0
    body = captured[0]["json"]
    assert isinstance(body, dict)
    assert body["model"] == DEFAULT_GROQ_VISION_MODEL
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 2048
    content_parts = body["messages"][0]["content"]  # type: ignore[index]
    assert content_parts[0]["type"] == "text"
    assert content_parts[1]["type"] == "image_url"
    expected_b64 = base64.b64encode(raw_png).decode("ascii")
    assert content_parts[1]["image_url"]["url"] == f"data:image/png;base64,{expected_b64}"

    assert resp.raw_output == "1.5 m"
    assert resp.provider_name == "groq"
    assert resp.model_name == DEFAULT_GROQ_VISION_MODEL

    # Verify provider/model names propagate into saved Evaluation3DReport JSON.
    ds_dir = tmp_path / "ds3d"
    rep_dir = tmp_path / "rep3d"
    DatasetBuilder3D().build_from_scenes(
        build_default_3d_scenes()[:1],
        output_dir=ds_dir,
        config=Dataset3DConfig(image_width_px=400, image_height_px=300, supersample=1),
    )
    eval_res = Evaluation3DRunner(adapter).evaluate(
        ds_dir, output_dir=rep_dir, config=Evaluation3DConfig(max_records=1)
    )
    assert eval_res.report.provider_name == "groq"
    assert eval_res.report.model_name == DEFAULT_GROQ_VISION_MODEL


def test_openrouter_empty_reasoning_429_5xx_and_timeout_handling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_all_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-placeholder-key")
    req, _ = _make_fake_png_request(tmp_path)

    class _MockResp:
        def __init__(self, status_code: int, payload: dict[str, object] | None = None, text: str = "") -> None:
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text
            self.headers: dict[str, str] = {}

        def json(self) -> dict[str, object]:
            return self._payload

    # 1. Plain empty content returns empty string (Runner classifies as PARSE_ERROR).
    adapter = OpenAICompatibleVLM3DAdapter(
        model_id="qwen/qwen2.5-vl-72b-instruct:free",
        base_url=OPENROUTER_BASE_URL,
        api_key_env_var="OPENROUTER_API_KEY",
        provider_name="openrouter",
        max_retries=2,
        retry_backoff_s=0.0,
        sleep_fn=lambda _: None,
    )
    seen_payloads: list[dict[str, object]] = []

    def _capture_empty(*args, **kwargs):
        seen_payloads.append(kwargs["json"])
        return _MockResp(
            200,
            {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]},
        )

    monkeypatch.setattr(httpx, "post", _capture_empty)
    empty_resp = adapter.predict(req)
    assert seen_payloads[0]["max_tokens"] == 2048
    assert seen_payloads[0]["temperature"] == 0.0
    assert empty_resp.raw_output == ""

    # 2. Reasoning-only response raises explicit diagnostic error.
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: _MockResp(
            200,
            {
                "choices": [
                    {
                        "message": {"content": None, "reasoning": "thinking step..."},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"completion_tokens_details": {"reasoning_tokens": 1024}},
            },
        ),
    )
    with pytest.raises(Evaluator3DError, match="reasoning-only response"):
        adapter.predict(req)

    # 3. 429 transient recovers within bounded retries, and persistent 429 fails clearly.
    call_count = 0

    def _429_then_ok(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _MockResp(429, text="rate limited")
        return _MockResp(200, {"choices": [{"message": {"content": "true"}}]})

    monkeypatch.setattr(httpx, "post", _429_then_ok)
    assert adapter.predict(req).raw_output == "true"
    assert call_count == 2

    always_429_calls = 0

    def _always_429(*args, **kwargs):
        nonlocal always_429_calls
        always_429_calls += 1
        return _MockResp(429, text="still limited")

    monkeypatch.setattr(httpx, "post", _always_429)
    with pytest.raises(Evaluator3DError, match="HTTP 429"):
        adapter.predict(req)
    assert always_429_calls == 3  # 1 initial + 2 bounded retries

    # 4. 5xx retries bounded and fails clearly.
    always_503_calls = 0

    def _always_503(*args, **kwargs):
        nonlocal always_503_calls
        always_503_calls += 1
        return _MockResp(503, text="upstream unavailable")

    monkeypatch.setattr(httpx, "post", _always_503)
    with pytest.raises(Evaluator3DError, match="HTTP 503"):
        adapter.predict(req)
    assert always_503_calls == 3

    # 5. Timeout retries bounded and raises explicit timeout error.
    timeout_calls = 0

    def _always_timeout(*args, **kwargs):
        nonlocal timeout_calls
        timeout_calls += 1
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(httpx, "post", _always_timeout)
    with pytest.raises(Evaluator3DError, match="timed out"):
        adapter.predict(req)
    assert timeout_calls == 3
