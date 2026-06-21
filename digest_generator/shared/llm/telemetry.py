"""Telemetry helpers shared by every LLM call site.

Two related concerns live here:

1. **Run-level token accounting**: a ``TokenCounter`` bound via
   ``llm_telemetry()`` sums LLM token usage across every stage in a run.
   The digest orchestrator (and any other LLM-using pipeline) wraps its
   body in this context manager and reads the totals back at the end.

2. **Per-call LLM logging**: ``chat_with_logging()`` wraps
   ``ollama.Client.chat`` to:
   - retry transient failures (HTTP 429 / 5xx) with exponential backoff +
     jitter so cloud rate-limits degrade gracefully instead of killing
     the whole stage
   - log a per-call ``llm.call`` DEBUG line with prompt/completion token
     counts and durations
   - accumulate per-stage totals into the active ``StageSpan`` so
     ``stage.done`` carries them
   - record into the active ``TokenCounter`` so the surrounding pipeline's
     terminal ``*.tokens`` summary line aggregates the run

Both layers degrade gracefully: ``record_llm_call`` is a no-op outside an
``llm_telemetry()`` block, and ``chat_with_logging`` works fine outside a
``log_stage`` block (per-call DEBUG line still emitted, span accumulation
skipped).
"""

from __future__ import annotations

import random
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any

import httpx
from ollama import ResponseError

from digest_generator.shared.logging import current_span, current_stage, logger
from digest_generator.shared.settings import settings

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ollama import Client


# Status codes worth retrying: 429 (rate limit), 500/502/503/504 (server-side
# failure). 500 is included because Ollama Cloud uses it (with a server
# reference ID in the body) for the same transient upstream blips that produce
# 502/503/504 elsewhere, so an un-retried 500 would otherwise kill a long run
# after most feeds have already completed.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_LLM_MAX_RETRIES = 3
_LLM_INITIAL_BACKOFF_S = 1.0
_LLM_BACKOFF_FACTOR = 2.0
_LLM_BACKOFF_MAX_S = 30.0
_LLM_BACKOFF_JITTER = 0.5


# Module-level cap on in-flight Ollama calls across every stage.
# Lazy-initialized so settings overrides (env vars, test mocks) take effect.
# Reset to None in tests that want a fresh semaphore for a different cap.
_ollama_semaphore: threading.Semaphore | None = None
_ollama_semaphore_lock = threading.Lock()


def _get_ollama_semaphore() -> threading.Semaphore:
    """Return the module-level Ollama in-flight cap, lazy-initializing on first call.

    Threading (not asyncio) semaphore because ``chat_with_logging`` is a
    sync function called from both async-via-``asyncio.to_thread`` (the
    summarizer) and plain sync (every digest stage). A threading
    primitive works in both contexts: the asyncio path blocks its
    worker thread when the cap is hit, which queues correctly without
    starving the event loop.
    """
    global _ollama_semaphore  # noqa: PLW0603 (intentional thread-safe lazy init)
    if _ollama_semaphore is None:
        with _ollama_semaphore_lock:
            if _ollama_semaphore is None:
                _ollama_semaphore = threading.Semaphore(settings.ollama_concurrency)
    return _ollama_semaphore


@dataclass
class TokenCounter:
    """Mutable accumulator for run-level LLM telemetry.

    Bound by ``llm_telemetry()`` for the duration of a pipeline run.
    Stages indirectly write to it via ``record_llm_call`` (called from
    ``chat_with_logging``); the surrounding pipeline reads the totals at
    the end and emits its own ``*.tokens`` summary line.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    llm_duration_ms: int = 0
    per_stage: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(
        self,
        *,
        stage: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: int,
    ) -> None:
        """Accumulate one LLM call's metrics into the run totals."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.llm_calls += 1
        self.llm_duration_ms += duration_ms

        bucket = self.per_stage.setdefault(
            stage,
            {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0, "llm_duration_ms": 0},
        )
        bucket["prompt_tokens"] += prompt_tokens
        bucket["completion_tokens"] += completion_tokens
        bucket["llm_calls"] += 1
        bucket["llm_duration_ms"] += duration_ms


_counter: ContextVar[TokenCounter | None] = ContextVar("_llm_token_counter", default=None)


@contextmanager
def llm_telemetry() -> Iterator[TokenCounter]:
    """Bind a fresh ``TokenCounter`` for the duration of a pipeline run.

    Use as the outer context for any pipeline that issues LLM calls::

        with llm_telemetry() as counter:
            run_stages(...)
            logger.info("digest.tokens prompt={} ...", counter.prompt_tokens)
    """
    counter = TokenCounter()
    token = _counter.set(counter)
    try:
        yield counter
    finally:
        _counter.reset(token)


def current_counter() -> TokenCounter | None:
    """Return the active ``TokenCounter``, or ``None`` if no pipeline run is in progress."""
    return _counter.get()


def record_llm_call(
    *, stage: str, prompt_tokens: int, completion_tokens: int, duration_ms: int
) -> None:
    """Record one LLM call into the active ``TokenCounter`` (no-op if unbound)."""
    counter = _counter.get()
    if counter is None:
        return
    counter.record(
        stage=stage,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_ms=duration_ms,
    )


def _coerce_int(value: Any) -> int:
    """Convert an Ollama-reported field to ``int``, defaulting to ``0`` on missing/None."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _retry_sleep_seconds(attempt: int) -> float:
    """Exponential backoff with jitter for retryable LLM failures.

    ``attempt`` is the zero-based count of failures so far (i.e., 0 means
    the first retry). Capped at ``_LLM_BACKOFF_MAX_S`` before jitter.
    """
    base = min(
        _LLM_INITIAL_BACKOFF_S * (_LLM_BACKOFF_FACTOR**attempt),
        _LLM_BACKOFF_MAX_S,
    )
    # `random.random()` is fine for jitter; not security-sensitive.
    return base * (1.0 + _LLM_BACKOFF_JITTER * random.random())  # nosec: B311


def chat_with_logging(
    client: Client,
    *,
    model: str,
    messages: list[dict[str, str]],
    options: dict[str, Any],
    think: bool = False,
) -> str:
    """Call ``client.chat`` and emit telemetry; return the response text.

    Acquires the module-level Ollama in-flight semaphore (sized by
    ``settings.ollama_concurrency``) for the entire call+retry cycle so
    every Ollama call site shares one global cap regardless of which
    stage initiated the call.

    Retries transient failures with exponential backoff + jitter, up to
    ``_LLM_MAX_RETRIES`` attempts:

    - ``ResponseError`` with status in ``{429, 500, 502, 503, 504}``: cloud
      rate-limit or upstream server failure. 500 is included because Ollama
      Cloud uses it for transient incidents (with a server reference ID),
      not just for genuine model errors.
    - ``httpx.TimeoutException``: connect, read, write, or pool timeout.
      Read timeouts are the most common in practice, signalling a hung
      cloud connection that the SDK's ``timeout=None`` default would
      have left blocked forever.

    Other ``ResponseError``\\s and httpx exceptions propagate immediately.
    Each retry emits a ``llm.retry`` WARNING and increments ``llm_retries``
    on the active span; only the successful call's tokens / duration are
    recorded.

    Logs a per-call ``llm.call`` line at DEBUG with prompt/completion token
    counts and total duration. Accumulates the same fields into the active
    ``StageSpan`` (via ``current_span``) and the active ``TokenCounter``
    (via ``record_llm_call``). Both lookups are no-ops when nothing is bound.

    Returns an empty string if the model produces no content, emitting a
    warning in that case.

    Args:
        client: Ollama client to invoke.
        model: Model identifier (used for the chat call and for telemetry).
        messages: ``[{"role": ..., "content": ...}, ...]`` chat messages.
        options: Ollama ``options`` dict (temperature, top_p, etc.).
        think: Forwarded to ``client.chat``; keep ``False`` for consistent
            ``message.content`` across models.
    """
    semaphore = _get_ollama_semaphore()
    with semaphore:
        attempt = 0
        while True:
            start = perf_counter()
            try:
                response = client.chat(
                    model=model,
                    messages=messages,
                    think=think,
                    options=options,
                )
                break
            except (ResponseError, httpx.TimeoutException) as exc:
                retryable = isinstance(exc, httpx.TimeoutException) or (
                    isinstance(exc, ResponseError) and exc.status_code in _RETRYABLE_STATUS_CODES
                )
                if not retryable or attempt >= _LLM_MAX_RETRIES:
                    raise
                if isinstance(exc, httpx.TimeoutException):
                    cause = f"timeout({type(exc).__name__})"
                else:
                    cause = f"status={exc.status_code}"
                sleep_s = _retry_sleep_seconds(attempt)
                logger.warning(
                    "llm.retry {} attempt={}/{} sleep_s={:.2f} model={}",
                    cause,
                    attempt + 1,
                    _LLM_MAX_RETRIES,
                    sleep_s,
                    model,
                )
                span = current_span()
                if span is not None:
                    span.add(llm_retries=1)
                time.sleep(sleep_s)
                attempt += 1

        elapsed_ms = int((perf_counter() - start) * 1000)

    prompt_tokens = _coerce_int(getattr(response, "prompt_eval_count", None))
    completion_tokens = _coerce_int(getattr(response, "eval_count", None))
    # Ollama reports duration fields in nanoseconds.
    eval_duration_ns = _coerce_int(getattr(response, "eval_duration", None))
    eval_duration_ms = eval_duration_ns // 1_000_000

    span = current_span()
    if span is not None:
        span.add(
            llm_calls=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            llm_duration_ms=elapsed_ms,
        )

    record_llm_call(
        stage=current_stage() or "-",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_ms=elapsed_ms,
    )

    logger.debug(
        "llm.call model={} prompt_tokens={} completion_tokens={} "
        "total_duration_ms={} eval_duration_ms={}",
        model,
        prompt_tokens,
        completion_tokens,
        elapsed_ms,
        eval_duration_ms,
    )

    content = response.message.content or ""
    if not content:
        logger.warning("LLM returned empty response")
    return content
