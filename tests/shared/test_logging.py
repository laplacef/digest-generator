"""Tests for ``digest_generator/shared/logging.py``: StageSpan and log_stage primitives.

The loguru sinks themselves are configured at module import time and not
re-exercised here; these tests focus on the in-process primitives (``StageSpan``,
``current_span``, ``current_stage``, ``log_stage``) that downstream
helpers (``chat_with_logging``, telemetry counters) rely on.

``TestSecretSafety`` is the exception: it pins the secret-leak regression
guard.
"""

from io import StringIO

import pytest
from loguru import logger

from digest_generator.shared.logging import (
    _SECRET_SAFE_SINK_KWARGS,
    StageSpan,
    _redact_secrets,
    collect_stage_telemetry,
    current_span,
    current_stage,
    log_stage,
)

# Token-shaped strings used by TestSecretSafety and TestRedactingFilter.
# Hoisted to module level so the secret-detection pragma sits next to the
# literal once instead of repeating at every reference site. ruff-format
# splits long-line tuples in @pytest.mark.parametrize and decouples inline
# pragmas from the literal; module-level constants are immune to that.
_FAKE_HF_TOKEN = (
    "hf_LEAKED_TOKEN_DO_NOT_PRINT_abcdef0123456789"  # pragma: allowlist secret  # gitleaks:allow
)
_FAKE_OLLAMA_KEY = (
    "sk-LEAKED_OLLAMA_KEY_zyxwvu0987654321"  # pragma: allowlist secret  # gitleaks:allow
)
# Vendor-pattern fixtures for TestRedactingFilter. Each is a synthetic token
# that matches its vendor regex but has no real-world counterpart.
_FAKE_HF_PARAM = "hf_abcdefghij0123456789LMNOP"  # pragma: allowlist secret  # gitleaks:allow
_FAKE_SK_PARAM = "sk-abcdefghij0123456789LMNOP"  # pragma: allowlist secret  # gitleaks:allow
_FAKE_AKIA = "AKIAIOSFODNN7EXAMPLE"  # pragma: allowlist secret  # gitleaks:allow
_FAKE_ASIA = "ASIAIOSFODNN7EXAMPLE"  # pragma: allowlist secret  # gitleaks:allow
_FAKE_GHP = "ghp_" + "x" * 36  # pragma: allowlist secret  # gitleaks:allow
_FAKE_GLPAT = "glpat-" + "x" * 20  # pragma: allowlist secret  # gitleaks:allow
_FAKE_SLACK = "xoxb-1234567890-abcdef"  # pragma: allowlist secret  # gitleaks:allow
_FAKE_HF_IN_MESSAGE = (
    "hf_LeAkEdInMessage0123456789LMNOP"  # pragma: allowlist secret  # gitleaks:allow
)
_FAKE_SK_IN_EXTRAS = (
    "sk-LeAkEdInExtras0123456789LMNOP"  # pragma: allowlist secret  # gitleaks:allow
)


class TestStageSpanSet:
    def test_set_overwrites(self):
        span = StageSpan()
        span.set(articles=10)
        span.set(articles=20)
        assert span.fields == {"articles": 20}

    def test_set_multiple_fields(self):
        span = StageSpan()
        span.set(articles=10, sections=3, model="x")
        assert span.fields == {"articles": 10, "sections": 3, "model": "x"}


class TestStageSpanAdd:
    def test_add_sums_when_existing_numeric(self):
        span = StageSpan()
        span.add(prompt_tokens=100)
        span.add(prompt_tokens=250)
        assert span.fields == {"prompt_tokens": 350}

    def test_add_initializes_when_unset(self):
        span = StageSpan()
        span.add(llm_calls=1)
        assert span.fields == {"llm_calls": 1}

    def test_add_handles_floats(self):
        span = StageSpan()
        span.add(score=1.5)
        span.add(score=2.25)
        assert span.fields == {"score": 3.75}

    def test_add_overwrites_non_numeric(self):
        span = StageSpan()
        span.set(model="a")
        span.add(model="b")
        assert span.fields == {"model": "b"}

    def test_add_does_not_treat_bool_as_number(self):
        """``True + True`` would be 2; guard against that surprise."""
        span = StageSpan()
        span.add(ok=True)
        span.add(ok=True)
        assert span.fields == {"ok": True}

    def test_add_then_set_overwrites(self):
        span = StageSpan()
        span.add(prompt_tokens=100)
        span.set(prompt_tokens=0)
        assert span.fields == {"prompt_tokens": 0}

    def test_set_then_add_accumulates(self):
        span = StageSpan()
        span.set(prompt_tokens=50)
        span.add(prompt_tokens=25)
        assert span.fields == {"prompt_tokens": 75}


class TestLogStageContext:
    def test_current_span_outside_block_is_none(self):
        assert current_span() is None
        assert current_stage() is None

    def test_current_span_inside_block_is_the_span(self):
        with log_stage("writer") as span:
            assert current_span() is span
            assert current_stage() == "writer"

    def test_nested_blocks_restore_outer_context(self):
        with log_stage("outer") as outer_span:
            assert current_span() is outer_span
            assert current_stage() == "outer"
            with log_stage("inner") as inner_span:
                assert current_span() is inner_span
                assert current_stage() == "inner"
            assert current_span() is outer_span
            assert current_stage() == "outer"

    def test_context_cleared_after_block(self):
        with log_stage("writer"):
            pass
        assert current_span() is None
        assert current_stage() is None

    def test_context_cleared_on_exception(self):
        boom = RuntimeError("boom")
        try:
            with log_stage("writer"):
                raise boom
        except RuntimeError:
            pass
        assert current_span() is None
        assert current_stage() is None

    def test_span_fields_accessible_after_block(self):
        """The span object outlives its log_stage block; users can inspect ``fields``."""
        with log_stage("writer") as span:
            span.set(articles=10)
            span.add(prompt_tokens=500)
        assert span.fields == {"articles": 10, "prompt_tokens": 500}

    def test_start_fields_seeded_into_span(self):
        """Identifying kwargs on log_stage land in span.fields and persist on done."""
        with log_stage("fetcher", feed="techcrunch") as span:
            assert span.fields == {"feed": "techcrunch"}
            span.set(entries=12)
        assert span.fields == {"feed": "techcrunch", "entries": 12}

    def test_start_fields_dont_collide_with_set(self):
        """``span.set`` overrides start-seeded values when the caller wants to refine."""
        with log_stage("fetcher", feed="provisional") as span:
            span.set(feed="resolved-name")
        assert span.fields == {"feed": "resolved-name"}


class TestCollectStageTelemetry:
    """collect_stage_telemetry: orchestrator-bound sink for stage span fields."""

    def test_records_each_stage_with_duration_and_fields(self):
        with collect_stage_telemetry() as sink:
            with log_stage("writer") as span:
                span.set(sections=5, articles=42)
            with log_stage("editor") as span:
                span.set(rewritten=4, fell_back=1)
        assert "writer" in sink
        assert "editor" in sink
        assert sink["writer"]["sections"] == 5
        assert sink["writer"]["articles"] == 42
        assert sink["editor"]["rewritten"] == 4
        # duration_ms is recorded but value depends on wall time; just check presence.
        assert "duration_ms" in sink["writer"]
        assert "duration_ms" in sink["editor"]

    def test_records_empty_fields_dict_for_silent_stages(self):
        with collect_stage_telemetry() as sink, log_stage("composer"):
            pass
        assert sink["composer"] == {"duration_ms": sink["composer"]["duration_ms"]}

    def test_failed_stages_are_not_recorded(self):
        boom = RuntimeError("boom")
        with collect_stage_telemetry() as sink:
            with log_stage("writer") as span:
                span.set(sections=5)
            try:
                with log_stage("editor"):
                    raise boom
            except RuntimeError:
                pass
        assert "writer" in sink
        assert "editor" not in sink

    def test_no_sink_outside_context_manager(self):
        """log_stage outside a sink block must not raise; the sink is opt-in."""
        with log_stage("writer") as span:
            span.set(sections=5)
        # No assertion needed; just must not raise.

    def test_sink_isolated_per_context_block(self):
        with collect_stage_telemetry() as outer:
            with log_stage("a"):
                pass
            assert "a" in outer
        # New block starts with an empty sink.
        with collect_stage_telemetry() as inner:
            with log_stage("b"):
                pass
            assert "a" not in inner
            assert "b" in inner

    def test_repeated_stage_name_merges_numeric_fields(self):
        """Multiple log_stage calls with the same name accumulate counters."""
        with collect_stage_telemetry() as sink:
            with log_stage("summarizer") as span:
                span.set(entries=10, llm_calls=5, model="m")
            with log_stage("summarizer") as span:
                span.set(entries=20, llm_calls=8, model="m")
        # Numeric fields summed; non-numeric (model) overwrites.
        assert sink["summarizer"]["entries"] == 30
        assert sink["summarizer"]["llm_calls"] == 13
        assert sink["summarizer"]["model"] == "m"

    def test_repeated_stage_name_overwrites_non_numeric(self):
        """Non-numeric fields take the last value when stage repeats."""
        with collect_stage_telemetry() as sink:
            with log_stage("framer") as span:
                span.set(title_retried=False)
            with log_stage("framer") as span:
                span.set(title_retried=True)
        assert sink["framer"]["title_retried"] is True


class TestSecretSafety:
    """Regression guard: tracebacks must not dump frame-local values.

    An ``httpx.ConnectTimeout`` traceback can render ``settings.hf_token``
    inline under the loguru ``diagnose=True`` default. ``_SECRET_SAFE_SINK_KWARGS``
    pins ``diagnose=False`` on every project sink. These tests verify both
    halves: the constant itself, and the behavioral contract under a fresh sink
    configured with the same kwargs.
    """

    def test_secret_safe_kwargs_disable_diagnose(self):
        assert _SECRET_SAFE_SINK_KWARGS["diagnose"] is False

    def test_exception_traceback_omits_local_secret(self):
        """With ``diagnose=False``, a secret-shaped local must not appear in output."""
        buffer = StringIO()
        sink_id = logger.add(
            buffer,
            level="DEBUG",
            format="{message}",
            **_SECRET_SAFE_SINK_KWARGS,
        )
        try:
            secret_value = _FAKE_HF_TOKEN
            try:
                # Local in scope when the exception is raised; with diagnose=True
                # loguru would print the literal value as a frame-local. With
                # diagnose=False it must not appear.
                _ = secret_value
                msg = "boom"
                raise RuntimeError(msg)
            except RuntimeError:
                logger.exception("captured")
        finally:
            logger.remove(sink_id)
        assert _FAKE_HF_TOKEN not in buffer.getvalue()

    def test_stage_error_traceback_omits_local_secret(self):
        """``log_stage`` uses ``logger.exception`` on the failure path; same guard."""
        buffer = StringIO()
        sink_id = logger.add(
            buffer,
            level="DEBUG",
            format="{message}",
            **_SECRET_SAFE_SINK_KWARGS,
        )
        try:
            secret_value = _FAKE_OLLAMA_KEY
            try:
                with log_stage("writer"):
                    _ = secret_value
                    msg = "stage blew up"
                    raise RuntimeError(msg)
            except RuntimeError:
                pass
        finally:
            logger.remove(sink_id)
        assert _FAKE_OLLAMA_KEY not in buffer.getvalue()


class TestRedactingFilter:
    """Regression guard: loguru patcher redacts vendor-prefixed
    secrets in log messages and extras before any sink emits.

    Complements TestSecretSafety, which covers the structural traceback-leak
    guard (`diagnose=False`). These tests cover the cases that guard
    doesn't reach: secrets interpolated into log messages, secrets bound into
    `extra`, and the false-positive guard against legitimate hex content
    (commit SHAs, model revision pins).
    """

    @pytest.mark.parametrize(
        ("vendor_label", "secret", "expected_prefix"),
        [
            ("HuggingFace", _FAKE_HF_PARAM, "hf_"),
            ("OpenAI-style", _FAKE_SK_PARAM, "sk-"),
            ("AWS access key", _FAKE_AKIA, "AKIA"),
            ("AWS STS key", _FAKE_ASIA, "ASIA"),
            ("GitHub classic PAT", _FAKE_GHP, "ghp_"),
            ("GitLab PAT", _FAKE_GLPAT, "glpat-"),
            ("Slack bot token", _FAKE_SLACK, "xoxb-"),
        ],
    )
    def test_redact_secrets_masks_vendor_prefix_then_redacted(
        self, vendor_label, secret, expected_prefix
    ):
        """Each vendor pattern is masked to `<prefix>[REDACTED]`."""
        del vendor_label  # parametrize id only
        out = _redact_secrets(f"calling api with token={secret} done")
        assert secret not in out
        assert f"{expected_prefix}[REDACTED]" in out
        # Surrounding text must survive verbatim; only the secret is masked.
        assert out.startswith("calling api with token=")
        assert out.endswith(" done")

    def test_redact_secrets_handles_pem_block_header(self):
        """PEM private-key block headers are redacted (any key type)."""
        for key_type in ("RSA", "OPENSSH", "EC", "DSA"):
            header = f"-----BEGIN {key_type} PRIVATE KEY-----"
            out = _redact_secrets(f"loaded {header}\n<key body>")
            assert header not in out
            assert "-----BEGIN [REDACTED]" in out

    def test_redact_secrets_passes_through_clean_text(self):
        """Text with no secret patterns is returned unchanged."""
        clean = "fetched 42 entries from techcrunch in 1.3s"
        assert _redact_secrets(clean) == clean

    @pytest.mark.parametrize(
        "legitimate_content",
        [
            # HuggingFace model commit SHAs, same shape as a topic_revision pin.
            "topic_revision=d7645e127eaf1aefc7862fd59a17a5aa8558b8ce",
            # Piper voice-repo commit SHA from settings.
            "voice_revision=375a0fe641dea077c2a47b4e9a056d6da521eed3",
            # Generic 40-char hex (git commit hash).
            "commit abc123def456789012345678901234567890abcd in main",
            # Base64-shaped non-secret (e.g. sha256 hash output).
            "fingerprint=Qm5sZWtKdHFRWnpKVnp6ekxKenc9PQ==",  # pragma: allowlist secret  # gitleaks:allow
            # File path containing hex.
            "/home/user/.cache/digest_generator/abcdef1234567890.onnx",
            # Hex stage-id from the run-dir suffix.
            "run_dir=output/2026-05-14-153000-a4b3",
        ],
    )
    def test_redact_secrets_does_not_match_commit_shas_or_hashes(self, legitimate_content):
        """Conservative pattern set must not false-positive on legitimate hex content.

        This pins the design choice to use vendor-prefix matching only,
        rejecting the generic-high-entropy regex that would catch unknown-
        shape secrets at the cost of redacting commit SHAs, model revision
        pins, and file hashes throughout digest_generator's logs.
        """
        assert _redact_secrets(legitimate_content) == legitimate_content

    def test_patcher_redacts_message_at_sink(self):
        """End-to-end: a `logger.info` call with a secret in the message must
        emit a redacted line through a freshly-configured sink."""
        buffer = StringIO()
        sink_id = logger.add(
            buffer,
            level="DEBUG",
            format="{message}",
            **_SECRET_SAFE_SINK_KWARGS,
        )
        try:
            logger.info("calling whoami with {}", _FAKE_HF_IN_MESSAGE)
        finally:
            logger.remove(sink_id)
        out = buffer.getvalue()
        assert _FAKE_HF_IN_MESSAGE not in out
        assert "hf_[REDACTED]" in out
        assert "calling whoami with" in out

    def test_patcher_redacts_extras_at_sink(self):
        """A secret bound into `extra` via `logger.bind` must be redacted when
        the format string references it."""
        buffer = StringIO()
        sink_id = logger.add(
            buffer,
            level="DEBUG",
            format="{extra[api_key]} | {message}",
            **_SECRET_SAFE_SINK_KWARGS,
        )
        try:
            logger.bind(api_key=_FAKE_SK_IN_EXTRAS).info("calling ollama")
        finally:
            logger.remove(sink_id)
        out = buffer.getvalue()
        assert _FAKE_SK_IN_EXTRAS not in out
        assert "sk-[REDACTED]" in out
        assert "calling ollama" in out
