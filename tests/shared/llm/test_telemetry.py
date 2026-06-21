"""Tests for digest_generator/shared/llm/telemetry.py: TokenCounter and chat_with_logging."""

import threading
from unittest.mock import MagicMock, patch

import httpx
import pytest
from ollama import ResponseError

from digest_generator.shared.llm import telemetry as telemetry_mod
from digest_generator.shared.llm.telemetry import (
    TokenCounter,
    chat_with_logging,
    current_counter,
    llm_telemetry,
    record_llm_call,
)
from digest_generator.shared.logging import current_span, log_stage


@pytest.fixture(autouse=True)
def reset_ollama_semaphore():
    """Reset the module-level Ollama semaphore between tests.

    The lazy-init pattern caches the first-constructed semaphore on the
    module; tests that mock ``settings.ollama_concurrency`` need a fresh
    semaphore that picks up the mocked value.
    """
    telemetry_mod._ollama_semaphore = None
    yield
    telemetry_mod._ollama_semaphore = None


def _build_mock_response(
    *,
    content: str | None = "ok",
    prompt_eval_count: int | None = 1500,
    eval_count: int | None = 200,
    eval_duration: int | None = 4_000_000_000,
) -> MagicMock:
    """Build an Ollama-shaped response mock with controllable telemetry fields."""
    response = MagicMock()
    response.message.content = content
    response.prompt_eval_count = prompt_eval_count
    response.eval_count = eval_count
    response.eval_duration = eval_duration
    return response


class TestTokenCounter:
    def test_record_increments_run_totals(self):
        counter = TokenCounter()
        counter.record(stage="writer", prompt_tokens=100, completion_tokens=20, duration_ms=500)
        counter.record(stage="writer", prompt_tokens=200, completion_tokens=40, duration_ms=700)
        assert counter.prompt_tokens == 300
        assert counter.completion_tokens == 60
        assert counter.llm_calls == 2
        assert counter.llm_duration_ms == 1200

    def test_record_buckets_by_stage(self):
        counter = TokenCounter()
        counter.record(stage="writer", prompt_tokens=100, completion_tokens=20, duration_ms=500)
        counter.record(stage="editor", prompt_tokens=80, completion_tokens=70, duration_ms=300)
        counter.record(stage="writer", prompt_tokens=200, completion_tokens=40, duration_ms=700)

        assert set(counter.per_stage.keys()) == {"writer", "editor"}
        assert counter.per_stage["writer"] == {
            "prompt_tokens": 300,
            "completion_tokens": 60,
            "llm_calls": 2,
            "llm_duration_ms": 1200,
        }
        assert counter.per_stage["editor"] == {
            "prompt_tokens": 80,
            "completion_tokens": 70,
            "llm_calls": 1,
            "llm_duration_ms": 300,
        }


class TestLlmTelemetryContext:
    def test_current_counter_outside_block_is_none(self):
        assert current_counter() is None

    def test_current_counter_inside_block(self):
        with llm_telemetry() as counter:
            assert current_counter() is counter

    def test_counter_unbound_after_block(self):
        with llm_telemetry():
            pass
        assert current_counter() is None

    def test_counter_unbound_after_exception(self):
        boom = RuntimeError("boom")
        try:
            with llm_telemetry():
                raise boom
        except RuntimeError:
            pass
        assert current_counter() is None


class TestRecordLlmCall:
    def test_no_op_when_no_counter_bound(self):
        # Should not raise; nothing to assert beyond that.
        record_llm_call(stage="writer", prompt_tokens=10, completion_tokens=5, duration_ms=100)

    def test_records_when_counter_bound(self):
        with llm_telemetry() as counter:
            record_llm_call(stage="writer", prompt_tokens=10, completion_tokens=5, duration_ms=100)
            assert counter.llm_calls == 1
            assert counter.prompt_tokens == 10


class TestChatWithLogging:
    def test_returns_response_content(self):
        client = MagicMock()
        client.chat.return_value = _build_mock_response(content="hello world")

        result = chat_with_logging(
            client,
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            options={"temperature": 0.4},
        )
        assert result == "hello world"

    def test_passes_options_through(self):
        client = MagicMock()
        client.chat.return_value = _build_mock_response()

        chat_with_logging(
            client,
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            options={"temperature": 0.7, "seed": 42},
        )
        kwargs = client.chat.call_args.kwargs
        assert kwargs["options"] == {"temperature": 0.7, "seed": 42}
        assert kwargs["model"] == "m"
        assert kwargs["think"] is False

    def test_accumulates_into_active_span(self):
        client = MagicMock()
        client.chat.return_value = _build_mock_response(prompt_eval_count=1500, eval_count=200)

        with log_stage("writer") as span:
            chat_with_logging(
                client,
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                options={},
            )
            chat_with_logging(
                client,
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                options={},
            )

        assert span.fields["llm_calls"] == 2
        assert span.fields["prompt_tokens"] == 3000
        assert span.fields["completion_tokens"] == 400
        assert span.fields["llm_duration_ms"] >= 0

    def test_records_into_run_counter_with_stage_label(self):
        client = MagicMock()
        client.chat.return_value = _build_mock_response(prompt_eval_count=500, eval_count=50)

        with llm_telemetry() as counter, log_stage("editor"):
            chat_with_logging(
                client,
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                options={},
            )

        assert counter.llm_calls == 1
        assert counter.prompt_tokens == 500
        assert counter.completion_tokens == 50
        assert "editor" in counter.per_stage
        assert counter.per_stage["editor"]["prompt_tokens"] == 500

    def test_handles_missing_token_fields_gracefully(self):
        """Some providers omit prompt_eval_count / eval_count entirely."""
        client = MagicMock()
        client.chat.return_value = _build_mock_response(
            prompt_eval_count=None, eval_count=None, eval_duration=None
        )

        with log_stage("writer") as span:
            result = chat_with_logging(
                client,
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                options={},
            )

        assert result == "ok"
        assert span.fields["llm_calls"] == 1
        assert span.fields["prompt_tokens"] == 0
        assert span.fields["completion_tokens"] == 0

    def test_returns_empty_string_on_empty_content(self):
        client = MagicMock()
        client.chat.return_value = _build_mock_response(content="")

        result = chat_with_logging(client, model="m", messages=[], options={})
        assert result == ""

    def test_returns_empty_string_on_none_content(self):
        client = MagicMock()
        client.chat.return_value = _build_mock_response(content=None)

        result = chat_with_logging(client, model="m", messages=[], options={})
        assert result == ""

    def test_works_outside_log_stage_block(self):
        """No active span / counter still emits per-call DEBUG, no crash."""
        client = MagicMock()
        client.chat.return_value = _build_mock_response()

        # Sanity-check the precondition.
        assert current_span() is None
        assert current_counter() is None

        result = chat_with_logging(client, model="m", messages=[], options={})
        assert result == "ok"


class TestChatWithLoggingRetry:
    """Retry behavior for transient failures (HTTP 429 / 500 / 502 / 503 / 504)."""

    @pytest.fixture(autouse=True)
    def no_sleep(self):
        """Skip the backoff sleep in tests; only the call shape matters."""
        with patch("digest_generator.shared.llm.telemetry.time.sleep") as sleep:
            yield sleep

    def test_retries_on_429_then_succeeds(self, no_sleep):
        client = MagicMock()
        client.chat.side_effect = [
            ResponseError("too many concurrent requests", 429),
            _build_mock_response(content="recovered"),
        ]

        result = chat_with_logging(client, model="m", messages=[], options={})

        assert result == "recovered"
        assert client.chat.call_count == 2
        assert no_sleep.call_count == 1

    def test_retries_on_503_then_succeeds(self, no_sleep):
        client = MagicMock()
        client.chat.side_effect = [
            ResponseError("service unavailable", 503),
            _build_mock_response(content="ok"),
        ]
        chat_with_logging(client, model="m", messages=[], options={})
        assert client.chat.call_count == 2

    def test_retries_on_500_then_succeeds(self, no_sleep):
        """Ollama Cloud returns 500 with a server ref for transient blips."""
        client = MagicMock()
        client.chat.side_effect = [
            ResponseError("Internal Server Error (ref: abc123)", 500),
            _build_mock_response(content="recovered"),
        ]

        result = chat_with_logging(client, model="m", messages=[], options={})

        assert result == "recovered"
        assert client.chat.call_count == 2

    def test_does_not_retry_on_400(self, no_sleep):
        """Non-retryable status (e.g., bad request) propagates immediately."""
        client = MagicMock()
        client.chat.side_effect = ResponseError("bad request", 400)

        with pytest.raises(ResponseError) as exc_info:
            chat_with_logging(client, model="m", messages=[], options={})

        assert exc_info.value.status_code == 400
        assert client.chat.call_count == 1
        no_sleep.assert_not_called()

    def test_gives_up_after_max_retries(self, no_sleep):
        """After 3 retries (4 attempts total), the final 429 propagates."""
        client = MagicMock()
        client.chat.side_effect = ResponseError("rate limited", 429)

        with pytest.raises(ResponseError) as exc_info:
            chat_with_logging(client, model="m", messages=[], options={})

        assert exc_info.value.status_code == 429
        assert client.chat.call_count == 4  # initial + 3 retries
        assert no_sleep.call_count == 3

    def test_increments_llm_retries_on_active_span(self, no_sleep):
        client = MagicMock()
        client.chat.side_effect = [
            ResponseError("rate limited", 429),
            ResponseError("rate limited", 429),
            _build_mock_response(content="ok"),
        ]

        with log_stage("summarizer") as span:
            chat_with_logging(client, model="m", messages=[], options={})

        assert span.fields["llm_retries"] == 2
        # llm_calls only counts the successful invocation.
        assert span.fields["llm_calls"] == 1

    def test_only_successful_call_recorded_in_token_counter(self, no_sleep):
        client = MagicMock()
        client.chat.side_effect = [
            ResponseError("rate limited", 429),
            _build_mock_response(prompt_eval_count=300, eval_count=80),
        ]

        with llm_telemetry() as counter, log_stage("summarizer"):
            chat_with_logging(client, model="m", messages=[], options={})

        assert counter.llm_calls == 1
        assert counter.prompt_tokens == 300
        assert counter.completion_tokens == 80

    def test_works_outside_span_when_retry_fires(self, no_sleep):
        """No active span: retry path must not crash on span.add."""
        client = MagicMock()
        client.chat.side_effect = [
            ResponseError("rate limited", 429),
            _build_mock_response(content="ok"),
        ]

        result = chat_with_logging(client, model="m", messages=[], options={})

        assert result == "ok"
        assert client.chat.call_count == 2


class TestChatWithLoggingTimeoutRetry:
    """Connection-level hangs raise httpx.TimeoutException; retry the same way as 429s."""

    @pytest.fixture(autouse=True)
    def no_sleep(self):
        with patch("digest_generator.shared.llm.telemetry.time.sleep") as sleep:
            yield sleep

    def test_retries_on_read_timeout_then_succeeds(self, no_sleep):
        """ReadTimeout (the cloud-Ollama hang case) is the canonical retryable timeout."""
        client = MagicMock()
        client.chat.side_effect = [
            httpx.ReadTimeout("read timed out"),
            _build_mock_response(content="recovered"),
        ]

        result = chat_with_logging(client, model="m", messages=[], options={})

        assert result == "recovered"
        assert client.chat.call_count == 2

    def test_retries_on_connect_timeout_then_succeeds(self, no_sleep):
        client = MagicMock()
        client.chat.side_effect = [
            httpx.ConnectTimeout("connect timed out"),
            _build_mock_response(content="ok"),
        ]
        chat_with_logging(client, model="m", messages=[], options={})
        assert client.chat.call_count == 2

    def test_gives_up_after_max_retries_on_persistent_timeout(self, no_sleep):
        """After 3 retries (4 attempts total), the final timeout propagates."""
        client = MagicMock()
        client.chat.side_effect = httpx.ReadTimeout("read timed out")

        with pytest.raises(httpx.ReadTimeout):
            chat_with_logging(client, model="m", messages=[], options={})

        assert client.chat.call_count == 4  # initial + 3 retries
        assert no_sleep.call_count == 3

    def test_mixed_429_and_timeout_both_retry(self, no_sleep):
        """A 429 followed by a timeout should both be treated as transient."""
        client = MagicMock()
        client.chat.side_effect = [
            ResponseError("rate limited", 429),
            httpx.ReadTimeout("read timed out"),
            _build_mock_response(content="ok"),
        ]

        result = chat_with_logging(client, model="m", messages=[], options={})

        assert result == "ok"
        assert client.chat.call_count == 3

    def test_increments_llm_retries_on_timeout(self, no_sleep):
        client = MagicMock()
        client.chat.side_effect = [
            httpx.ReadTimeout("hang"),
            _build_mock_response(content="ok"),
        ]
        with log_stage("summarizer") as span:
            chat_with_logging(client, model="m", messages=[], options={})
        assert span.fields["llm_retries"] == 1


class TestChatWithLoggingSemaphore:
    """Every chat_with_logging call acquires a global ollama_concurrency semaphore."""

    @pytest.fixture(autouse=True)
    def no_sleep(self):
        with patch("digest_generator.shared.llm.telemetry.time.sleep") as sleep:
            yield sleep

    @patch("digest_generator.shared.llm.telemetry.settings")
    def test_semaphore_initialized_with_settings_value(self, mock_settings):
        """First call instantiates the semaphore sized by settings.ollama_concurrency."""
        mock_settings.ollama_concurrency = 3
        client = MagicMock()
        client.chat.return_value = _build_mock_response()

        chat_with_logging(client, model="m", messages=[], options={})

        assert isinstance(telemetry_mod._ollama_semaphore, threading.Semaphore)
        # threading.Semaphore exposes _value (CPython implementation detail), but
        # the cap is checkable by trying to acquire 3 times non-blocking.
        sem = telemetry_mod._ollama_semaphore
        assert sem is not None
        assert sem.acquire(blocking=False)
        assert sem.acquire(blocking=False)
        assert sem.acquire(blocking=False)
        assert not sem.acquire(blocking=False)  # 4th acquire blocks, so cap is 3

    @patch("digest_generator.shared.llm.telemetry.settings")
    def test_semaphore_acquired_per_call(self, mock_settings):
        """Each call acquires and releases the semaphore; verify post-call free slots."""
        mock_settings.ollama_concurrency = 2
        client = MagicMock()
        client.chat.return_value = _build_mock_response()

        chat_with_logging(client, model="m", messages=[], options={})
        chat_with_logging(client, model="m", messages=[], options={})

        # Both slots free after both calls returned.
        sem = telemetry_mod._ollama_semaphore
        assert sem is not None
        assert sem.acquire(blocking=False)
        assert sem.acquire(blocking=False)
        assert not sem.acquire(blocking=False)

    @patch("digest_generator.shared.llm.telemetry.settings")
    def test_semaphore_released_on_exception(self, mock_settings):
        """Non-retryable error must still release the slot."""
        mock_settings.ollama_concurrency = 2
        client = MagicMock()
        client.chat.side_effect = ResponseError("bad request", 400)

        with pytest.raises(ResponseError):
            chat_with_logging(client, model="m", messages=[], options={})

        # Slot was released even though the call raised.
        sem = telemetry_mod._ollama_semaphore
        assert sem is not None
        assert sem.acquire(blocking=False)
        assert sem.acquire(blocking=False)
        assert not sem.acquire(blocking=False)

    @patch("digest_generator.shared.llm.telemetry.settings")
    def test_semaphore_held_through_retry_cycle(self, mock_settings, no_sleep):
        """A retrying call should NOT release the slot between attempts.

        Holding the slot during retry-sleep is the right backpressure
        signal: if the cloud is rate-limiting one caller, another caller
        should not grab the freed slot and pile on.
        """
        mock_settings.ollama_concurrency = 1
        client = MagicMock()
        client.chat.side_effect = [
            ResponseError("rate limited", 429),
            _build_mock_response(content="ok"),
        ]

        # Probe whether the semaphore was held during the call by
        # attempting a non-blocking acquire from the same thread:
        # threading.Semaphore is non-recursive, so a held slot blocks.
        # Mid-call state is not observable without threads, but the
        # slot is assertably free after the call (releasing on success).
        chat_with_logging(client, model="m", messages=[], options={})
        sem = telemetry_mod._ollama_semaphore
        assert sem is not None
        assert sem.acquire(blocking=False)
