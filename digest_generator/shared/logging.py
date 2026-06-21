"""Centralized loguru logging configuration.

Three sinks:

- **Console** (stderr): configurable level, concise timestamp, colorized. Shows
  the current stage when bound via ``logger.contextualize(stage=...)``.
- **Global file** (``logs/digest_generator.log``): size-based rotation with file-count
  retention. Catches ambient and pre-run lines that fall outside any specific
  pipeline run (startup, CLI errors, feed listing).
- **Per-run file** (``{run_dir}/run.log``): added dynamically via ``run_context``
  when a pipeline run starts. Self-contained log for one run, sitting next to
  ``meta.json`` and ``summaries/``.

All modules import the pre-configured logger::

    from digest_generator.shared.logging import logger

Orchestration entry points wrap pipeline runs with ``run_context`` to bind the
``run_id`` and attach the per-run sink::

    from digest_generator.shared.logging import logger, run_context

    with run_context(run_id, run_dir):
        logger.info("pipeline starting")

Stages emit structured start/done markers via ``log_stage``::

    from digest_generator.shared.logging import log_stage

    with log_stage("writer") as span:
        ...
        span.set(sections=3, articles=42)
"""

from __future__ import annotations

import re
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from loguru import logger

from digest_generator.shared.settings import settings

if TYPE_CHECKING:
    from collections.abc import Iterator

    from loguru import Record

_CONSOLE_FORMAT = "<level>{time:HH:mm:ss} | {level:<8} | {extra[stage]:<9} | {message}</level>"
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
    "run={extra[run_id]} | stage={extra[stage]:<9} | "
    "{module}:{function}:{line} | {message}"
)

# Secret-safety: loguru defaults to ``diagnose=True``, which prints per-frame
# local-variable values inline in tracebacks. Any exception caught while a
# secret (``HF_TOKEN``, ``OLLAMA_API_KEY``, etc.) is in scope would leak the
# value to stdout, the global log, and every per-run log. Diagnose is disabled
# on every sink; extended backtraces stay enabled so multi-frame traces still
# render.
_SECRET_SAFE_SINK_KWARGS: dict[str, Any] = {"backtrace": True, "diagnose": False}

# Defense-in-depth secret redaction. ``diagnose=False`` removes the
# structural traceback-leak surface, but a secret can still reach a sink if a
# contributor interpolates it into a log message (``f"token={hf_token}"``) or
# binds it into ``extra`` (``logger.bind(api_key=...)``). The patcher below
# scans ``record["message"]`` and ``record["extra"]`` once per record and
# rewrites known vendor-prefixed secrets to ``<prefix>[REDACTED]`` before any
# sink formats the line.
#
# Pattern set is intentionally conservative: vendor prefixes only, no generic
# high-entropy regex. Generic-entropy matching would risk redacting legitimate
# log content (commit SHAs, model revision pins, file hashes), the
# false-positive class that vendor-prefix matching avoids.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),  # HuggingFace API tokens
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),  # OpenAI / Anthropic / Ollama
    re.compile(r"AKIA[A-Z0-9]{16}"),  # AWS access key ID
    re.compile(r"ASIA[A-Z0-9]{16}"),  # AWS STS temporary access key
    re.compile(r"ghp_[A-Za-z0-9]{36}"),  # GitHub personal access token (classic)
    re.compile(r"github_pat_[A-Za-z0-9_]{82}"),  # GitHub fine-grained PAT
    re.compile(r"glpat-[A-Za-z0-9_-]{20}"),  # GitLab PAT
    re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}"),  # Slack tokens
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),  # PEM private-key block headers
)


def _redact_secrets(text: str) -> str:
    """Replace known vendor-prefixed secret patterns with ``<prefix>[REDACTED]``.

    Preserves the vendor prefix (``hf_``, ``sk-``, ``AKIA``, etc.) so a
    redacted log line still identifies *which* secret type was leaked,
    without exposing any characters of the secret body. Returns ``text``
    unchanged if no pattern matches (cheap when no secret is present,
    the common case for every log line in digest_generator).
    """
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: f"{_prefix_for(m.group(0))}[REDACTED]", text)
    return text


def _prefix_for(match: str) -> str:
    """Return the vendor prefix portion of a secret match (everything up to
    and including the first digit-free identifier segment).

    Examples (placeholder shapes used to avoid matching real-secret regexes):
        ``hf_abcd1234`` becomes ``hf_``
        ``sk-proj-abc`` becomes ``sk-``
        ``AKIA<placeholder>`` becomes ``AKIA``
        ``ghp_xxxxxxxx…`` becomes ``ghp_``
        ``-----BEGIN <type> PRIVATE KEY-----`` becomes ``-----BEGIN ``
    """
    # Vendor prefixes are short and well-defined; static lookup beats a regex
    # match-group capture that would entangle the redaction layer with the
    # detection layer's pattern shape.
    for prefix in (
        "hf_",
        "sk-",
        "AKIA",
        "ASIA",
        "ghp_",
        "github_pat_",
        "glpat-",
        "xoxa-",
        "xoxb-",
        "xoxp-",
        "xoxr-",
        "xoxs-",
        "-----BEGIN ",
    ):
        if match.startswith(prefix):
            return prefix
    return ""  # unreachable given _SECRET_PATTERNS coverage


def _redact_record(record: Record) -> None:
    """Loguru patcher: redact secrets in ``record["message"]`` and ``extra``.

    Runs once per log record before any sink formats it, so every sink
    (stderr, ``logs/digest_generator.log``, per-run ``run.log``) sees redacted
    content uniformly. The record dict is mutated in place per loguru's
    patcher contract.

    The traceback rendering for exceptions is already secret-safe via
    ``diagnose=False``; this patcher does not touch
    ``record["exception"]``.
    """
    message = record.get("message")
    if isinstance(message, str):
        record["message"] = _redact_secrets(message)

    extra = record.get("extra")
    if isinstance(extra, dict):
        for key, value in extra.items():
            if isinstance(value, str):
                extra[key] = _redact_secrets(value)


# Default extras so format strings always resolve, even when no context is bound.
# ``patcher`` runs once per record before any sink emits, applying secret
# redaction as defense in depth on top of the per-sink ``diagnose=False`` guard.
logger.configure(extra={"run_id": "-", "stage": "-"}, patcher=_redact_record)

logger.remove()

logger.add(
    sys.stderr,
    level=settings.log_level_console,
    format=_CONSOLE_FORMAT,
    colorize=True,
    **_SECRET_SAFE_SINK_KWARGS,
)

_log_dir = Path(settings.log_dir)
_log_dir.mkdir(exist_ok=True)

logger.add(
    _log_dir / "digest_generator.log",
    level=settings.log_level_file,
    format=_FILE_FORMAT,
    rotation=settings.log_rotation,
    retention=settings.log_retention,
    compression="zip",
    enqueue=True,
    **_SECRET_SAFE_SINK_KWARGS,
)


@contextmanager
def run_context(run_id: str, run_dir: Path | None = None) -> Iterator[None]:
    """Bind ``run_id`` to every log line and attach a per-run file sink.

    When ``run_dir`` is provided, a ``run.log`` sink is added for the duration
    of the context, capturing every log line emitted during the run at the
    configured file level.

    Args:
        run_id: Unique identifier for this pipeline run (e.g. the run-dir
            timestamp ``YYYY-MM-DD_HHmmss``).
        run_dir: Directory to write ``run.log`` into. Pass ``None`` to skip the
            per-run sink (still binds ``run_id``).
    """
    sink_id: int | None = None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        sink_id = logger.add(
            run_dir / "run.log",
            level=settings.log_level_file,
            format=_FILE_FORMAT,
            enqueue=False,
            **_SECRET_SAFE_SINK_KWARGS,
        )
    try:
        with logger.contextualize(run_id=run_id):
            yield
    finally:
        if sink_id is not None:
            logger.remove(sink_id)


class StageSpan:
    """Accumulator for fields emitted on a stage's ``stage.done`` line.

    Two semantics:

    - ``set(**fields)`` overwrites; use for terminal counts, model names, etc.
    - ``add(**fields)`` sums numeric values, overwrites non-numeric; use when
      a field accumulates across multiple internal calls (e.g., a stage's
      total ``prompt_tokens`` summed over per-batch LLM requests).
    """

    __slots__ = ("fields",)

    def __init__(self) -> None:
        self.fields: dict[str, Any] = {}

    def set(self, **fields: Any) -> None:
        """Record fields to include in the ``stage.done`` log line (overwrite)."""
        self.fields.update(fields)

    def add(self, **fields: Any) -> None:
        """Accumulate numeric fields; overwrite non-numeric.

        Useful when a stage emits multiple LLM calls and wants the
        ``stage.done`` line to carry the sum (e.g., ``prompt_tokens``,
        ``llm_calls``). Booleans are treated as non-numeric to avoid
        ``True + True == 2`` surprises.
        """
        for key, value in fields.items():
            current = self.fields.get(key)
            if (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and isinstance(current, int | float)
                and not isinstance(current, bool)
            ):
                self.fields[key] = current + value
            elif isinstance(value, int | float) and not isinstance(value, bool) and current is None:
                self.fields[key] = value
            else:
                self.fields[key] = value


_active_span: ContextVar[StageSpan | None] = ContextVar("_active_span", default=None)
_active_stage: ContextVar[str | None] = ContextVar("_active_stage", default=None)
_stage_telemetry_sink: ContextVar[dict[str, dict[str, Any]] | None] = ContextVar(
    "_stage_telemetry_sink", default=None
)


def current_span() -> StageSpan | None:
    """Return the innermost ``StageSpan`` bound by ``log_stage``, or ``None``.

    Helpers (e.g., ``chat_with_logging``) use this to record per-call metrics
    without threading the span through every internal method.
    """
    return _active_span.get()


def current_stage() -> str | None:
    """Return the innermost stage name bound by ``log_stage``, or ``None``."""
    return _active_stage.get()


def _merge_stage_fields(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Merge stage telemetry fields with sum-on-numeric semantics.

    Used when ``log_stage`` is opened more than once with the same name in
    a single ``collect_stage_telemetry`` block (the summarizer fans out per
    feed, the fetcher per feed; both want their per-call counts aggregated
    into a single sink entry rather than overwritten). Non-numeric fields
    (model name, booleans) overwrite; repeating a stage with a different
    model is a configuration error, not a sum-able event.
    """
    out = dict(existing)
    for key, value in new.items():
        cur = out.get(key)
        if (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and isinstance(cur, int | float)
            and not isinstance(cur, bool)
        ):
            out[key] = cur + value
        else:
            out[key] = value
    return out


@contextmanager
def collect_stage_telemetry() -> Iterator[dict[str, dict[str, Any]]]:
    """Collect each stage's terminal span fields, keyed by stage name.

    Bound by an orchestrator (the digest pipeline) around its run.
    Each ``log_stage`` block, on successful exit, appends
    ``{name: {duration_ms, **span.fields}}`` to the sink dict so the caller
    can build per-stage ``StageMeta`` records for ``meta.json`` without
    parsing ``run.log``. Stages that raise are not recorded; failures
    surface via the orchestrator's exception handling, not the telemetry
    sink.

    Nested invocation is not supported (a single sink per orchestrator).
    Stages outside an active sink continue to log normally; the sink is
    a side channel, not a behavior change.
    """
    sink: dict[str, dict[str, Any]] = {}
    token = _stage_telemetry_sink.set(sink)
    try:
        yield sink
    finally:
        _stage_telemetry_sink.reset(token)


@contextmanager
def log_stage(name: str, **start_fields: Any) -> Iterator[StageSpan]:
    """Bind ``stage=name`` and emit ``stage.start`` / ``stage.done`` markers.

    The ``stage.done`` line includes ``duration_ms`` plus any fields recorded
    via ``span.set(...)`` or ``span.add(...)``. Exceptions propagate, so the
    caller still sees them, but ``stage.done`` is replaced by a
    ``stage.error`` entry with the duration.

    Also binds the span as the active span (via ``ContextVar``) so helpers
    inside the block can find it without explicit threading.

    ``start_fields`` are *identifying* fields (typically ``feed`` /
    ``section``) that distinguish concurrent invocations of the same stage.
    They render into the ``stage.start`` line and seed the span so they
    also appear on ``stage.done`` without a redundant ``span.set`` call.

    Example::

        with log_stage("writer") as span:
            drafts = build_drafts()
            span.set(sections=len(drafts), articles=total)

        with log_stage("fetcher", feed="techcrunch") as span:
            ...
        # produces stage.start feed=techcrunch
        # produces stage.done duration_ms=1234 feed=techcrunch entries=12 ...
    """
    span = StageSpan()
    if start_fields:
        span.fields.update(start_fields)
    span_token = _active_span.set(span)
    stage_token = _active_stage.set(name)
    with logger.contextualize(stage=name):
        start = perf_counter()
        if start_fields:
            extras = " ".join(f"{k}={v}" for k, v in start_fields.items())
            logger.info("stage.start {}", extras)
        else:
            logger.info("stage.start")
        try:
            yield span
        except Exception:
            elapsed_ms = int((perf_counter() - start) * 1000)
            logger.exception("stage.error duration_ms={}", elapsed_ms)
            raise
        else:
            elapsed_ms = int((perf_counter() - start) * 1000)
            sink = _stage_telemetry_sink.get()
            if sink is not None:
                new_fields: dict[str, Any] = {"duration_ms": elapsed_ms, **span.fields}
                existing = sink.get(name)
                if existing is None:
                    sink[name] = new_fields
                else:
                    sink[name] = _merge_stage_fields(existing, new_fields)
            if span.fields:
                extras = " ".join(f"{k}={v}" for k, v in span.fields.items())
                logger.info("stage.done duration_ms={} {}", elapsed_ms, extras)
            else:
                logger.info("stage.done duration_ms={}", elapsed_ms)
        finally:
            _active_span.reset(span_token)
            _active_stage.reset(stage_token)
