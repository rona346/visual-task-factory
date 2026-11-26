"""Isolated live VLM evaluation framework for the 3D warehouse benchmark (Phase 6)."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from visual_task_factory.dataset_builder_3d import (
    Dataset3DRecord,
    load_and_validate_dataset_3d,
)
from visual_task_factory.qa_3d import (
    AnswerType3D,
    QuestionFamily3D,
    ReferenceAnswer3D,
)

DEFAULT_3D_EVAL_REPORT_DIR = Path("reports/3d_evaluation")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SUPPORTED_PROVIDERS_3D: tuple[str, ...] = ("groq", "openrouter")

_FLOAT_WITH_UNIT_PATTERN = re.compile(
    r"^\s*(?:(?:answer|clearance|distance)\s*[:=]\s*)?([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*\.?\s*$",
    re.IGNORECASE,
)
_SECRET_LEAK_PATTERN = re.compile(
    r"(gsk_[A-Za-z0-9_-]{8,}|sk-[A-Za-z0-9_-]{8,}|AIza[A-Za-z0-9_-]{10,}|Bearer\s+[A-Za-z0-9._-]+|key=[A-Za-z0-9._-]+)",
    re.IGNORECASE,
)


class Evaluator3DError(RuntimeError):
    """Base error for 3D VLM evaluation."""


class MissingCredentials3DError(Evaluator3DError):
    """Raised when credentials for the selected VLM provider are missing."""


class UnsupportedProvider3DError(Evaluator3DError):
    """Raised when VLM_PROVIDER is missing or not in ('groq', 'openrouter')."""


class MissingModelConfig3DError(Evaluator3DError):
    """Raised when OpenRouter lacks an explicit non-auto model configuration."""


class Eval3DStatus(str, Enum):
    """Execution and parsing outcome for one 3D benchmark record."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARSE_ERROR = "parse_error"
    FAILED_REQUEST = "failed_request"
    MISSING_IMAGE = "missing_image"


class VLM3DRequest(BaseModel):
    """Sanitized multimodal request sent to a 3D VLM adapter (no reference answers)."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    scene_id: str
    question_id: str
    family: QuestionFamily3D
    image_path: Path
    prompt_text: str
    expected_answer_type: AnswerType3D
    unit: str | None = None


class VLM3DResponse(BaseModel):
    """Raw output returned by a 3D VLM provider adapter."""

    model_config = ConfigDict(extra="forbid")

    raw_output: str
    model_name: str
    provider_name: str
    is_live: bool = True
    latency_ms: float | None = Field(default=None, ge=0.0)
    token_usage: dict[str, int] | None = None


class Normalized3DPrediction(BaseModel):
    """Result of conservative parsing of a raw model response."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    parsed_value: bool | float | None = None
    parsed_unit: str | None = None
    error_reason: str | None = None


class PerQuestion3DEvalRecord(BaseModel):
    """Machine-readable per-question prediction and scoring record."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    scene_id: str
    question_id: str
    family: QuestionFamily3D
    expected_answer_type: AnswerType3D
    image_rel_path: str
    prompt_text: str
    status: Eval3DStatus
    is_correct: bool
    raw_prediction: str | None = None
    normalized_prediction: bool | float | None = None
    reference_answer: ReferenceAnswer3D
    latency_ms: float | None = Field(default=None, ge=0.0)
    error_message: str | None = None


class Family3DMetrics(BaseModel):
    """Aggregated evaluation metrics for a single QuestionFamily3D."""

    model_config = ConfigDict(extra="forbid")

    family: QuestionFamily3D
    total: int = Field(..., ge=0)
    correct: int = Field(..., ge=0)
    incorrect: int = Field(..., ge=0)
    parse_error: int = Field(..., ge=0)
    failed_request: int = Field(..., ge=0)
    missing_image: int = Field(..., ge=0)
    accuracy: float = Field(..., ge=0.0, le=1.0)


class Evaluation3DConfig(BaseModel):
    """Controls 3D benchmark record selection and request behavior."""

    model_config = ConfigDict(extra="forbid")

    max_records: int | None = Field(default=12, ge=1)
    families: tuple[QuestionFamily3D, ...] | None = None
    fail_fast: bool = False


class Evaluation3DReport(BaseModel):
    """Complete JSON-serializable 3D benchmark evaluation report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str
    model_name: str
    provider_name: str
    is_live_evaluation: bool
    evaluated_at_utc: str
    sample_count: int = Field(..., ge=0)
    completed_requests: int = Field(..., ge=0)
    correct_count: int = Field(..., ge=0)
    incorrect_count: int = Field(..., ge=0)
    parse_error_count: int = Field(..., ge=0)
    failed_request_count: int = Field(..., ge=0)
    missing_image_count: int = Field(..., ge=0)
    overall_accuracy: float = Field(..., ge=0.0, le=1.0)
    parsed_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    by_family: dict[str, Family3DMetrics]
    predictions: list[PerQuestion3DEvalRecord]


class Evaluation3DRunResult(BaseModel):
    """Result returned by Evaluation3DRunner with output artifact paths."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    json_report_path: Path
    markdown_summary_path: Path
    report: Evaluation3DReport


def _redact_secrets(text: str) -> str:
    return _SECRET_LEAK_PATTERN.sub("[REDACTED]", text)


def _format_vlm_prompt(request: VLM3DRequest) -> str:
    if request.expected_answer_type == AnswerType3D.BOOLEAN:
        format_instruction = "Respond with strictly one word: true or false."
    else:
        unit_note = f" in {request.unit}" if request.unit else ""
        format_instruction = (
            f"Respond with strictly a single numeric value{unit_note} (for example: 1.5 m)."
        )
    return (
        "You are evaluating a 3D isometric warehouse image.\n"
        f"Question: {request.prompt_text}\n"
        f"{format_instruction}"
    )


def _unwrap_json_scalar(raw_output: str) -> str:
    text = raw_output.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                for key in ("answer", "value", "prediction"):
                    if key in loaded and len(loaded) <= 2:
                        inner = loaded[key]
                        if isinstance(inner, bool):
                            return "true" if inner else "false"
                        if isinstance(inner, (int, float, str)):
                            return str(inner)
        except json.JSONDecodeError:
            pass
    return text


def normalize_prediction_3d(
    raw_output: str,
    expected_type: AnswerType3D,
    *,
    expected_unit: str | None = None,
) -> Normalized3DPrediction:
    """Conservatively parse raw VLM text into BOOLEAN or NUMERIC without guessing."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        return Normalized3DPrediction(is_valid=False, error_reason="Empty model output.")

    text = _unwrap_json_scalar(raw_output).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()

    if expected_type == AnswerType3D.BOOLEAN:
        cleaned = text.rstrip(".").strip().lower()
        if cleaned in ("true", "yes"):
            return Normalized3DPrediction(is_valid=True, parsed_value=True)
        if cleaned in ("false", "no"):
            return Normalized3DPrediction(is_valid=True, parsed_value=False)
        return Normalized3DPrediction(
            is_valid=False,
            error_reason=f"Could not unambiguously parse boolean from {raw_output!r}.",
        )

    if expected_type == AnswerType3D.NUMERIC:
        match = _FLOAT_WITH_UNIT_PATTERN.match(text)
        if match is None:
            return Normalized3DPrediction(
                is_valid=False,
                error_reason=f"Could not unambiguously parse numeric clearance from {raw_output!r}.",
            )
        val = float(match.group(1))
        if not math.isfinite(val) or val < 0.0:
            return Normalized3DPrediction(
                is_valid=False,
                error_reason=f"Numeric prediction must be finite and >= 0; got {val}.",
            )
        unit_tok = match.group(2).lower()
        parsed_unit: str | None = None
        if unit_tok:
            if unit_tok in ("m", "meter", "meters"):
                parsed_unit = "m"
            else:
                return Normalized3DPrediction(
                    is_valid=False,
                    error_reason=(
                        f"Unsupported unit {unit_tok!r} (expected {expected_unit!r})."
                    ),
                )
        if expected_unit is not None and parsed_unit is not None and parsed_unit != expected_unit:
            return Normalized3DPrediction(
                is_valid=False,
                error_reason=f"Parsed unit {parsed_unit!r} != expected {expected_unit!r}.",
            )
        return Normalized3DPrediction(is_valid=True, parsed_value=val, parsed_unit=parsed_unit)

    return Normalized3DPrediction(
        is_valid=False,
        error_reason=f"Unsupported expected_type={expected_type!r}.",
    )


def score_prediction_3d(
    parsed_value: bool | float,
    reference: ReferenceAnswer3D,
) -> bool:
    """Compare a normalized 3D prediction against ReferenceAnswer3D."""
    if reference.answer_type == AnswerType3D.BOOLEAN:
        return type(parsed_value) is bool and parsed_value == reference.value

    if reference.answer_type == AnswerType3D.NUMERIC:
        if type(parsed_value) is bool or not isinstance(parsed_value, (int, float)):
            return False
        tol = reference.numeric_tolerance if reference.numeric_tolerance is not None else 0.05
        return abs(float(parsed_value) - float(reference.value)) <= tol + 1e-9

    return False


class VLM3DAdapter(ABC):
    """Abstract multimodal adapter interface for live or scripted 3D evaluation."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the evaluated model."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier of the inference provider."""

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """True only for genuine live model calls."""

    @abstractmethod
    def predict(self, request: VLM3DRequest) -> VLM3DResponse:
        """Execute a multimodal inference request for ``request``."""


class GeminiVLM3DAdapter(VLM3DAdapter):
    """Live Google Gemini multimodal adapter using installed ``google.genai`` SDK."""

    def __init__(
        self,
        *,
        model_id: str = "gemini-2.5-flash",
        api_key_env_var: str = "GEMINI_API_KEY",
        timeout_s: float = 30.0,
    ) -> None:
        api_key = os.environ.get(api_key_env_var) or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise MissingCredentials3DError(
                "Gemini live evaluation requires GEMINI_API_KEY or GOOGLE_API_KEY in environment."
            )
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be > 0.")

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise Evaluator3DError("google-genai package is not available.") from exc

        self._genai_types = types
        self._client = genai.Client(api_key=api_key)
        self._model_id = model_id
        self._timeout_s = timeout_s

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def provider_name(self) -> str:
        return "google_gemini"

    @property
    def is_live(self) -> bool:
        return True

    def predict(self, request: VLM3DRequest) -> VLM3DResponse:
        img_bytes = request.image_path.read_bytes()
        prompt = _format_vlm_prompt(request)
        t0 = time.perf_counter()
        response = self._client.models.generate_content(
            model=self._model_id,
            contents=[
                self._genai_types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                prompt,
            ],
            config=self._genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1024,
                http_options=self._genai_types.HttpOptions(timeout=int(self._timeout_s * 1000)),
            ),
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        text = (getattr(response, "text", None) or "").strip()
        usage = None
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta is not None:
            usage = {
                k: int(v)
                for k, v in (
                    ("prompt_tokens", getattr(usage_meta, "prompt_token_count", None)),
                    ("output_tokens", getattr(usage_meta, "candidates_token_count", None)),
                )
                if isinstance(v, int)
            } or None

        return VLM3DResponse(
            raw_output=text,
            model_name=self.model_name,
            provider_name=self.provider_name,
            is_live=True,
            latency_ms=elapsed_ms,
            token_usage=usage,
        )


class OpenAICompatibleVLM3DAdapter(VLM3DAdapter):
    """Live OpenAI-compatible vision chat-completions adapter (Groq & OpenRouter)."""

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str,
        api_key_env_var: str,
        provider_name: str = "openai_compatible",
        timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        api_key = os.environ.get(api_key_env_var, "").strip()
        if not api_key:
            raise MissingCredentials3DError(
                f"Live evaluation for provider {provider_name!r} requires {api_key_env_var} in environment."
            )
        if not model_id.strip():
            raise MissingModelConfig3DError(
                f"Provider {provider_name!r} requires a non-empty model_id."
            )
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be > 0.")
        if max_retries < 0 or max_retries > 5:
            raise ValueError("max_retries must be between 0 and 5.")
        if retry_backoff_s < 0.0:
            raise ValueError("retry_backoff_s must be >= 0.")

        try:
            import httpx
        except ImportError as exc:
            raise Evaluator3DError("httpx package is not available.") from exc

        self._httpx = httpx
        self._api_key = api_key
        self._model_id = model_id.strip()
        self._provider_name = provider_name.strip()
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._sleep = sleep_fn or time.sleep

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def is_live(self) -> bool:
        return True

    def _backoff_delay(self, attempt: int, retry_after_header: str | None = None) -> float:
        if retry_after_header:
            try:
                parsed = float(retry_after_header.strip())
                if math.isfinite(parsed) and parsed > 0.0:
                    return min(5.0, max(0.2, parsed))
            except ValueError:
                pass
        return min(5.0, self._retry_backoff_s * (2**attempt))

    def predict(self, request: VLM3DRequest) -> VLM3DResponse:
        b64_png = base64.b64encode(request.image_path.read_bytes()).decode("ascii")
        payload = {
            "model": self._model_id,
            "temperature": 0.0,
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _format_vlm_prompt(request)},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_png}"},
                        },
                    ],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        total_attempts = self._max_retries + 1
        t0 = time.perf_counter()
        resp = None

        for attempt in range(total_attempts):
            try:
                resp = self._httpx.post(
                    self._endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self._timeout_s,
                )
            except self._httpx.TimeoutException as exc:
                if attempt + 1 < total_attempts:
                    self._sleep(self._backoff_delay(attempt))
                    continue
                raise Evaluator3DError(
                    f"Provider {self._provider_name!r} timed out after {total_attempts} "
                    f"attempt(s) (timeout={self._timeout_s}s): {_redact_secrets(str(exc))}"
                ) from exc
            except self._httpx.HTTPError as exc:
                if attempt + 1 < total_attempts:
                    self._sleep(self._backoff_delay(attempt))
                    continue
                raise Evaluator3DError(
                    f"Transport error calling provider {self._provider_name!r} after "
                    f"{total_attempts} attempt(s): {_redact_secrets(str(exc))}"
                ) from exc

            if resp.status_code == 429:
                if attempt + 1 < total_attempts:
                    retry_after = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
                    self._sleep(self._backoff_delay(attempt, retry_after))
                    continue
                raise Evaluator3DError(
                    f"Provider {self._provider_name!r} rate-limited (HTTP 429) after "
                    f"{total_attempts} attempt(s): {_redact_secrets(resp.text[:300])}"
                )

            if resp.status_code >= 500:
                if attempt + 1 < total_attempts:
                    self._sleep(self._backoff_delay(attempt))
                    continue
                raise Evaluator3DError(
                    f"Provider {self._provider_name!r} server error (HTTP {resp.status_code}) after "
                    f"{total_attempts} attempt(s): {_redact_secrets(resp.text[:300])}"
                )

            if resp.status_code >= 400:
                raise Evaluator3DError(
                    f"Provider {self._provider_name!r} returned HTTP {resp.status_code}: "
                    f"{_redact_secrets(resp.text[:300])}"
                )
            break

        assert resp is not None
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        data = resp.json()
        choices = data.get("choices") or []
        first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        msg = first_choice.get("message") or {}
        raw_content = msg.get("content")
        content = "" if raw_content is None else str(raw_content).strip()
        finish_reason = first_choice.get("finish_reason")

        usage_raw = data.get("usage")
        usage = None
        reasoning_tokens: int | None = None
        if isinstance(usage_raw, dict):
            usage = {
                k: int(usage_raw[k])
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                if isinstance(usage_raw.get(k), int)
            } or None
            details = usage_raw.get("completion_tokens_details")
            if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), int):
                reasoning_tokens = details["reasoning_tokens"]

        has_reasoning_trace = bool(msg.get("reasoning") or msg.get("reasoning_details"))
        if not content and (has_reasoning_trace or (reasoning_tokens is not None and reasoning_tokens > 0)):
            raise Evaluator3DError(
                f"Provider {self._provider_name!r} returned a reasoning-only response with empty "
                f"final content (model={self._model_id!r}, finish_reason={finish_reason!r}, "
                f"reasoning_tokens={reasoning_tokens})."
            )

        return VLM3DResponse(
            raw_output=content,
            model_name=self.model_name,
            provider_name=self.provider_name,
            is_live=True,
            latency_ms=elapsed_ms,
            token_usage=usage,
        )


def create_live_vlm_adapter_from_env(
    *,
    timeout_s: float = 30.0,
    max_retries: int = 2,
) -> VLM3DAdapter:
    """Create a live VLM adapter strictly from explicit VLM_PROVIDER ('groq' or 'openrouter')."""
    provider = os.environ.get("VLM_PROVIDER", "").strip().lower()
    if not provider:
        raise UnsupportedProvider3DError(
            "VLM_PROVIDER must be explicitly set to 'groq' or 'openrouter'."
        )
    if provider not in SUPPORTED_PROVIDERS_3D:
        raise UnsupportedProvider3DError(
            f"Unsupported VLM_PROVIDER={provider!r}. Supported providers: 'groq', 'openrouter'."
        )

    if provider == "groq":
        model = os.environ.get("VLM_MODEL", "").strip() or DEFAULT_GROQ_VISION_MODEL
        return OpenAICompatibleVLM3DAdapter(
            model_id=model,
            base_url=GROQ_BASE_URL,
            api_key_env_var="GROQ_API_KEY",
            provider_name="groq",
            timeout_s=timeout_s,
            max_retries=max_retries,
        )

    # provider == "openrouter"
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise MissingCredentials3DError(
            "Live evaluation for provider 'openrouter' requires OPENROUTER_API_KEY in environment."
        )
    model = os.environ.get("VLM_MODEL", "").strip()
    if not model:
        raise MissingModelConfig3DError(
            "VLM_PROVIDER='openrouter' requires VLM_MODEL to be explicitly set to a fixed vision model ID."
        )
    if model.lower() == "openrouter/free":
        raise MissingModelConfig3DError(
            "VLM_MODEL='openrouter/free' is disallowed due to automatic routing to reasoning models; "
            "configure an explicit fixed OpenRouter model ID."
        )
    return OpenAICompatibleVLM3DAdapter(
        model_id=model,
        base_url=OPENROUTER_BASE_URL,
        api_key_env_var="OPENROUTER_API_KEY",
        provider_name="openrouter",
        timeout_s=timeout_s,
        max_retries=max_retries,
    )


def _select_records(
    records: list[Dataset3DRecord],
    config: Evaluation3DConfig,
) -> list[Dataset3DRecord]:
    selected: list[Dataset3DRecord] = []
    for rec in records:
        if config.families is not None and rec.family not in config.families:
            continue
        selected.append(rec)
        if config.max_records is not None and len(selected) >= config.max_records:
            break
    return selected


def _build_markdown_summary(report: Evaluation3DReport) -> str:
    parsed_acc_str = (
        f"{report.parsed_accuracy:.1%}" if report.parsed_accuracy is not None else "N/A"
    )
    lines = [
        "# 3D Spatial Benchmark Evaluation Summary",
        "",
        f"- **Dataset ID:** `{report.dataset_id}`",
        f"- **Provider / Model:** `{report.provider_name}` / `{report.model_name}`",
        f"- **Live Evaluation:** `{report.is_live_evaluation}`",
        f"- **Evaluated At (UTC):** `{report.evaluated_at_utc}`",
        f"- **Sample Count:** {report.sample_count}",
        f"- **Completed Requests:** {report.completed_requests}",
        f"- **Correct / Incorrect:** {report.correct_count} / {report.incorrect_count}",
        f"- **Parse Errors / Failed Requests / Missing Images:** "
        f"{report.parse_error_count} / {report.failed_request_count} / {report.missing_image_count}",
        f"- **Overall Accuracy:** {report.overall_accuracy:.1%}",
        f"- **Parsed-Only Accuracy:** {parsed_acc_str}",
        "",
        "## Accuracy by Question Family",
        "",
        "| Family | Total | Correct | Incorrect | Parse Errors | Failed Requests | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fam_key, metrics in report.by_family.items():
        lines.append(
            f"| `{fam_key}` | {metrics.total} | {metrics.correct} | {metrics.incorrect} "
            f"| {metrics.parse_error} | {metrics.failed_request} | {metrics.accuracy:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Per-Question Results",
            "",
            "| Question ID | Family | Status | Raw Prediction | Expected |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for item in report.predictions:
        raw_disp = _redact_secrets((item.raw_prediction or "-").replace("\n", " "))
        exp_disp = str(item.reference_answer.value)
        lines.append(
            f"| `{item.question_id}` | `{item.family.value}` | `{item.status.value}` "
            f"| `{raw_disp}` | `{exp_disp}` |"
        )
    return "\n".join(lines) + "\n"


class Evaluation3DRunner:
    """Loads a validated 3D dataset, runs a VLM adapter, scores, and saves reports."""

    def __init__(self, adapter: VLM3DAdapter) -> None:
        if not isinstance(adapter, VLM3DAdapter):
            raise TypeError(f"Expected VLM3DAdapter instance, got {type(adapter).__name__}.")
        self.adapter = adapter

    def evaluate(
        self,
        dataset_path_or_dir: Path | str,
        output_dir: Path | str = DEFAULT_3D_EVAL_REPORT_DIR,
        *,
        config: Evaluation3DConfig | None = None,
    ) -> Evaluation3DRunResult:
        """Evaluate selected 3D records and export JSON + Markdown reports."""
        cfg = config or Evaluation3DConfig()
        src_path = Path(dataset_path_or_dir)
        dataset_root = src_path.resolve() if src_path.is_dir() else src_path.parent.resolve()

        bundle = load_and_validate_dataset_3d(src_path, verify_images=True, revalidate_qa=True)
        selected = _select_records(bundle.records, cfg)

        out_dir = Path(output_dir)
        if out_dir.exists() and not out_dir.is_dir():
            raise FileExistsError(f"output_dir points to a non-directory file: '{out_dir}'.")
        out_dir.mkdir(parents=True, exist_ok=True)

        predictions: list[PerQuestion3DEvalRecord] = []
        completed_requests = 0

        for rec in selected:
            resolved_img = (dataset_root / rec.image_rel_path).resolve()
            if not resolved_img.is_relative_to(dataset_root) or not resolved_img.is_file():
                predictions.append(
                    PerQuestion3DEvalRecord(
                        record_id=rec.record_id,
                        scene_id=rec.scene_id,
                        question_id=rec.question_id,
                        family=rec.family,
                        expected_answer_type=rec.expected_answer_type,
                        image_rel_path=rec.image_rel_path,
                        prompt_text=rec.prompt_text,
                        status=Eval3DStatus.MISSING_IMAGE,
                        is_correct=False,
                        reference_answer=rec.reference_answer,
                        error_message=f"Image file missing or invalid: {rec.image_rel_path!r}.",
                    )
                )
                continue

            request = VLM3DRequest(
                record_id=rec.record_id,
                scene_id=rec.scene_id,
                question_id=rec.question_id,
                family=rec.family,
                image_path=resolved_img,
                prompt_text=rec.prompt_text,
                expected_answer_type=rec.expected_answer_type,
                unit=rec.reference_answer.unit,
            )

            try:
                response = self.adapter.predict(request)
                completed_requests += 1
            except Exception as exc:
                if cfg.fail_fast:
                    raise
                predictions.append(
                    PerQuestion3DEvalRecord(
                        record_id=rec.record_id,
                        scene_id=rec.scene_id,
                        question_id=rec.question_id,
                        family=rec.family,
                        expected_answer_type=rec.expected_answer_type,
                        image_rel_path=rec.image_rel_path,
                        prompt_text=rec.prompt_text,
                        status=Eval3DStatus.FAILED_REQUEST,
                        is_correct=False,
                        reference_answer=rec.reference_answer,
                        error_message=_redact_secrets(str(exc)),
                    )
                )
                continue

            norm = normalize_prediction_3d(
                response.raw_output,
                rec.expected_answer_type,
                expected_unit=rec.reference_answer.unit,
            )
            if not norm.is_valid or norm.parsed_value is None:
                predictions.append(
                    PerQuestion3DEvalRecord(
                        record_id=rec.record_id,
                        scene_id=rec.scene_id,
                        question_id=rec.question_id,
                        family=rec.family,
                        expected_answer_type=rec.expected_answer_type,
                        image_rel_path=rec.image_rel_path,
                        prompt_text=rec.prompt_text,
                        status=Eval3DStatus.PARSE_ERROR,
                        is_correct=False,
                        raw_prediction=response.raw_output,
                        normalized_prediction=None,
                        reference_answer=rec.reference_answer,
                        latency_ms=response.latency_ms,
                        error_message=norm.error_reason,
                    )
                )
                continue

            is_correct = score_prediction_3d(norm.parsed_value, rec.reference_answer)
            predictions.append(
                PerQuestion3DEvalRecord(
                    record_id=rec.record_id,
                    scene_id=rec.scene_id,
                    question_id=rec.question_id,
                    family=rec.family,
                    expected_answer_type=rec.expected_answer_type,
                    image_rel_path=rec.image_rel_path,
                    prompt_text=rec.prompt_text,
                    status=Eval3DStatus.CORRECT if is_correct else Eval3DStatus.INCORRECT,
                    is_correct=is_correct,
                    raw_prediction=response.raw_output,
                    normalized_prediction=norm.parsed_value,
                    reference_answer=rec.reference_answer,
                    latency_ms=response.latency_ms,
                )
            )

        sample_count = len(predictions)
        correct_count = sum(1 for p in predictions if p.status == Eval3DStatus.CORRECT)
        incorrect_count = sum(1 for p in predictions if p.status == Eval3DStatus.INCORRECT)
        parse_error_count = sum(1 for p in predictions if p.status == Eval3DStatus.PARSE_ERROR)
        failed_count = sum(1 for p in predictions if p.status == Eval3DStatus.FAILED_REQUEST)
        missing_count = sum(1 for p in predictions if p.status == Eval3DStatus.MISSING_IMAGE)

        overall_acc = (correct_count / sample_count) if sample_count > 0 else 0.0
        parsed_total = correct_count + incorrect_count
        parsed_acc = (correct_count / parsed_total) if parsed_total > 0 else None

        by_family: dict[str, Family3DMetrics] = {}
        grouped: dict[QuestionFamily3D, list[PerQuestion3DEvalRecord]] = {}
        for pred in predictions:
            grouped.setdefault(pred.family, []).append(pred)

        for fam in sorted(grouped, key=lambda f: f.value):
            grp = grouped[fam]
            tot = len(grp)
            cor = sum(1 for p in grp if p.status == Eval3DStatus.CORRECT)
            inc = sum(1 for p in grp if p.status == Eval3DStatus.INCORRECT)
            prs = sum(1 for p in grp if p.status == Eval3DStatus.PARSE_ERROR)
            fld = sum(1 for p in grp if p.status == Eval3DStatus.FAILED_REQUEST)
            mis = sum(1 for p in grp if p.status == Eval3DStatus.MISSING_IMAGE)
            by_family[fam.value] = Family3DMetrics(
                family=fam,
                total=tot,
                correct=cor,
                incorrect=inc,
                parse_error=prs,
                failed_request=fld,
                missing_image=mis,
                accuracy=round(cor / tot, 6) if tot > 0 else 0.0,
            )

        report = Evaluation3DReport(
            dataset_id=bundle.dataset_id,
            model_name=self.adapter.model_name,
            provider_name=self.adapter.provider_name,
            is_live_evaluation=self.adapter.is_live,
            evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
            sample_count=sample_count,
            completed_requests=completed_requests,
            correct_count=correct_count,
            incorrect_count=incorrect_count,
            parse_error_count=parse_error_count,
            failed_request_count=failed_count,
            missing_image_count=missing_count,
            overall_accuracy=round(overall_acc, 6),
            parsed_accuracy=round(parsed_acc, 6) if parsed_acc is not None else None,
            by_family=by_family,
            predictions=predictions,
        )

        json_path = out_dir / "evaluation_3d_report.json"
        md_path = out_dir / "evaluation_3d_summary.md"
        json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_build_markdown_summary(report), encoding="utf-8")

        return Evaluation3DRunResult(
            output_dir=out_dir,
            json_report_path=json_path,
            markdown_summary_path=md_path,
            report=report,
        )


def run_evaluation_3d(
    dataset_path_or_dir: Path | str,
    *,
    output_dir: Path | str = DEFAULT_3D_EVAL_REPORT_DIR,
    adapter: VLM3DAdapter | None = None,
    config: Evaluation3DConfig | None = None,
) -> Evaluation3DRunResult:
    """Convenience entry point to evaluate a 3D dataset with a live (or supplied) adapter."""
    resolved_adapter = adapter if adapter is not None else create_live_vlm_adapter_from_env()
    return Evaluation3DRunner(resolved_adapter).evaluate(
        dataset_path_or_dir=dataset_path_or_dir,
        output_dir=output_dir,
        config=config,
    )
