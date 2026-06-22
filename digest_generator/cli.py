"""CLI interface for the digest-generator pipeline.

Thin adapter over ``digest_generator.api``: parses arguments, delegates to the public API,
and handles display/exit codes. No business logic lives here.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from digest_generator.core.types import Filter
    from digest_generator.shared.llm.telemetry import TokenCounter

app = typer.Typer(
    name="digest_generator",
    help="Fetch, summarize, label, and digest news articles from RSS feeds.",
    no_args_is_help=True,
)


def _setup_run(
    *,
    output_dir: str | None,
    content_types: list[str] | None,
    feed_names: list[str] | None,
    days: int | None,
    parsed_since: datetime | None,
    parsed_until: datetime | None,
    limit: int | None,
    feeds_file: str | None = None,
    config_dir: str | None = None,
) -> tuple[Path, list, "Filter", tuple[str, str] | None]:
    """Shared setup for ``cli.run`` and ``cli.fetch``.

    Resolves feeds, builds a Filter, creates run_dir, computes the user-
    facing date_range. Caller is responsible for opening run_context /
    log_stage / llm_telemetry around the actual work.

    Returns (run_dir, feeds, resolved_filter, date_range).
    """
    from datetime import timedelta

    from digest_generator.api import resolve_feeds
    from digest_generator.core.types import Filter
    from digest_generator.shared.runtime.dirs import create_run_dir
    from digest_generator.shared.settings import settings
    from digest_generator.sources.rss.config import load_configured_feeds

    run_dir = create_run_dir(Path(output_dir) if output_dir else None)
    if content_types or feed_names:
        feeds = resolve_feeds(
            content_types, feed_names, feeds_file=feeds_file, config_dir=config_dir
        )
    else:
        feeds = load_configured_feeds(feeds_file=feeds_file, config_dir=config_dir)
    days_back = days if days is not None else settings.days_back
    resolved_filter = Filter.resolve(
        days_back=days_back,
        since=parsed_since,
        until=parsed_until,
        limit=limit,
    )

    # User-facing display dates: CLI adds 1 day to --until for inclusive
    # filtering; reverse that for display.
    display_since: str | None = (
        parsed_since.strftime("%Y-%m-%d")
        if parsed_since
        else resolved_filter.since.strftime("%Y-%m-%d")
        if resolved_filter.since
        else None
    )
    if parsed_until is not None:
        display_until: str | None = (parsed_until - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        display_until = (
            resolved_filter.until.strftime("%Y-%m-%d") if resolved_filter.until else None
        )
    date_range = (display_since, display_until) if display_since and display_until else None

    return run_dir, feeds, resolved_filter, date_range


def _write_initial_meta(
    *,
    run_dir: Path,
    feeds_count: int,
    article_count: int,
    content_types_used: list[str],
    duration_seconds: float,
    date_range: tuple[str, str] | None,
    start_iso: str,
) -> None:
    """Write meta.json after the corpus build completes."""
    from digest_generator.shared.runtime.meta import RunMeta, write_run_meta
    from digest_generator.shared.settings import settings

    display_since, display_until = date_range if date_range else (None, None)
    models: dict[str, str | None] = {
        "summarizer": settings.summarizer_model,
        "topic": settings.topic_model,
        "writer": settings.writer_model,
        "editor": settings.editorial_model,
        "framer": settings.framer_model,
        "watcher": settings.watcher_model,
    }
    revisions: dict[str, str] = {}
    if settings.topic_revision:
        revisions["topic"] = settings.topic_revision
    meta = RunMeta(
        timestamp=start_iso,
        duration_seconds=round(duration_seconds, 2),
        since=display_since,
        until=display_until,
        feed_count=feeds_count,
        article_count=article_count,
        content_types=content_types_used,
        models=models,
        revisions=revisions,
    )
    write_run_meta(meta, run_dir)


def _maybe_run_digest(
    run_dir: Path,
    *,
    enabled: bool,
    date_range: tuple[str, str] | None,
) -> bool:
    """Run the digest stage inside ``cli.run`` if enabled. Returns success.

    ``cli.run`` is the pipeline-level entry point and treats digest
    failure as a warning (the corpus is still on disk and can be
    re-digested via ``cli.digest``). Returning ``bool`` lets the caller
    gate downstream stages like audio on success.
    """
    if not enabled:
        return False
    from digest_generator import api
    from digest_generator.core.digest.io import build_digest_filename, build_digest_markdown
    from digest_generator.shared.logging import logger
    from digest_generator.shared.runtime.meta import update_run_meta_digest

    try:
        digest_result = api.digest(run_dir=run_dir, date_range=date_range)
    except Exception as e:
        logger.exception("Digest generation failed: {}", e)
        return False
    if digest_result is None:
        logger.warning("No summarized articles found in {}", run_dir)
        return False
    if not digest_result.content:
        logger.warning("Digest generation returned empty content")
        return False
    digest_filename = build_digest_filename(digest_result)
    digest_path = run_dir / digest_filename
    digest_path.write_text(build_digest_markdown(digest_result), encoding="utf-8")
    update_run_meta_digest(run_dir, digest_result.title, digest_filename)
    logger.info("Digest written to {}", digest_path)
    return True


def _maybe_render_audio(run_dir: Path, *, enabled: bool) -> None:
    """Render audio after a successful digest write, swallowing failures.

    Pipeline-level wrapper used by ``cli.run``: a missing piper binary or
    voice download glitch shouldn't fail the entire run after the digest
    already wrote successfully. The standalone ``cli.audio`` command and
    ``cli.digest --audio`` exit non-zero on failure because audio is the
    point of those invocations.
    """
    if not enabled:
        return
    from digest_generator import api
    from digest_generator.shared.logging import logger

    try:
        opus_path = api.render_audio(run_dir=run_dir)
        logger.info("Audio written to {}", opus_path)
    except Exception as e:
        logger.exception("Audio rendering failed: {}", e)


def _emit_pipeline_tokens(counter: "TokenCounter") -> None:
    """Emit the pipeline.tokens summary line bound to stage=pipeline."""
    from digest_generator.shared.logging import logger

    logger.bind(stage="pipeline").info(
        "pipeline.tokens prompt={} completion={} llm_calls={} llm_duration_ms={} per_stage={}",
        counter.prompt_tokens,
        counter.completion_tokens,
        counter.llm_calls,
        counter.llm_duration_ms,
        counter.per_stage,
    )


def _apply_env_overrides(
    device: str | None,
    output_dir: str | None,
) -> None:
    """Set environment variables before settings are imported.

    Must be called before any import that triggers ``Settings()`` instantiation.
    """
    if device:
        os.environ["DEVICE"] = device
    if output_dir:
        os.environ["OUTPUT_DIR"] = output_dir


def _parse_date(value: str) -> datetime:
    """Parse a YYYY-MM-DD date string to a UTC datetime at midnight.

    Args:
        value: Date string in ISO format (YYYY-MM-DD).

    Returns:
        Timezone-aware datetime at midnight UTC.

    Raises:
        typer.BadParameter: If the format is invalid.
    """
    from datetime import UTC

    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        msg = f"Invalid date format '{value}'. Expected YYYY-MM-DD."
        raise typer.BadParameter(msg) from None


def _validate_date_options(
    days: int | None,
    since: str | None,
    until: str | None,
) -> tuple[datetime | None, datetime | None]:
    """Validate mutual exclusivity and parse date options.

    Args:
        days: ``--days`` value (mutually exclusive with since/until).
        since: ``--since`` value as YYYY-MM-DD string.
        until: ``--until`` value as YYYY-MM-DD string.

    Returns:
        Tuple of (parsed_since, parsed_until) datetimes.

    Raises:
        typer.Exit: If ``--days`` is combined with ``--since``/``--until``.
    """
    from datetime import timedelta

    if days is not None and (since is not None or until is not None):
        typer.echo("Error: --days and --since/--until are mutually exclusive.")
        raise typer.Exit(1)

    parsed_since = _parse_date(since) if since else None
    parsed_until = None
    if until:
        # Add 1 day so --until 2026-03-15 includes all of March 15
        parsed_until = _parse_date(until) + timedelta(days=1)

    return parsed_since, parsed_until


@app.command()
def run(
    days: Annotated[
        int | None,
        typer.Option("--days", "-d", help="Number of days back to fetch articles."),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option("--since", help="Start date (YYYY-MM-DD). Mutually exclusive with --days."),
    ] = None,
    until: Annotated[
        str | None,
        typer.Option(
            "--until", help="End date inclusive (YYYY-MM-DD). Mutually exclusive with --days."
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-l", help="Max entries per feed."),
    ] = None,
    content_type: Annotated[
        list[str] | None,
        typer.Option("--content-type", "-c", help="Filter feeds by content type (repeatable)."),
    ] = None,
    feed: Annotated[
        list[str] | None,
        typer.Option("--feed", "-f", help="Filter feeds by name (repeatable)."),
    ] = None,
    feeds_file: Annotated[
        str | None,
        typer.Option("--feeds", help="Path to a feeds.yaml file (overrides discovery)."),
    ] = None,
    config_dir: Annotated[
        str | None,
        typer.Option("--config", help="Config directory holding feeds.yaml."),
    ] = None,
    output_dir: Annotated[
        str | None,
        typer.Option("--output-dir", "-o", help="Override output directory."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(help="Compute device: cpu, cuda, or mps."),
    ] = None,
    no_digest: Annotated[
        bool,
        typer.Option("--no-digest", help="Skip digest generation."),
    ] = False,
    audio: Annotated[
        bool,
        typer.Option(
            "--audio",
            help="Render the digest to an Opus audio file after generation. "
            "Requires the digest stage (mutually exclusive with --no-digest).",
        ),
    ] = False,
) -> None:
    """Run the full pipeline: fetch, summarize, label, and generate digest.

    Digest model and sampling overrides are not exposed here; use
    ``digest_generator digest <run_dir>`` for tuned digest output, which can re-run
    against the cached corpus without paying summarizer cost again.
    """
    if audio and no_digest:
        typer.echo("Error: --audio requires the digest stage; remove --no-digest.")
        raise typer.Exit(1)
    _apply_env_overrides(device, output_dir)

    parsed_since, parsed_until = _validate_date_options(days, since, until)

    from datetime import UTC
    from datetime import datetime as dt

    from digest_generator import api
    from digest_generator.shared.hf_hub import auto_login
    from digest_generator.shared.llm.telemetry import llm_telemetry
    from digest_generator.shared.logging import log_stage, logger, run_context
    from digest_generator.sources.rss.io import iter_fetched

    try:
        run_dir, feeds, resolved_filter, date_range = _setup_run(
            output_dir=output_dir,
            content_types=content_type or None,
            feed_names=feed or None,
            days=days,
            parsed_since=parsed_since,
            parsed_until=parsed_until,
            limit=limit,
            feeds_file=feeds_file,
            config_dir=config_dir,
        )
    except ValueError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from None

    with (
        run_context(run_dir.name, run_dir),
        log_stage("pipeline") as span,
        llm_telemetry() as counter,
    ):
        auto_login()
        start = dt.now(tz=UTC)
        logger.info(
            "run_dir={} feeds={} since={} until={} limit={}",
            run_dir,
            len(feeds),
            resolved_filter.since.isoformat() if resolved_filter.since else None,
            resolved_filter.until.isoformat() if resolved_filter.until else None,
            limit,
        )

        api.run(feeds, resolved_filter, run_dir=run_dir, with_digest=False)
        duration = (dt.now(tz=UTC) - start).total_seconds()

        # Tally counts from source-fetched/, the canonical source of
        # feed/article counts and the stage that carries content_type
        # in its entry data.
        feed_count = 0
        article_count = 0
        content_types_used: set[str] = set()
        for content_type_value, _feed_name, entries in iter_fetched(run_dir):
            feed_count += 1
            article_count += len(entries)
            content_types_used.add(str(content_type_value))
        logger.info("Pipeline complete: {} articles from {} feeds", article_count, feed_count)

        _write_initial_meta(
            run_dir=run_dir,
            feeds_count=feed_count,
            article_count=article_count,
            content_types_used=sorted(content_types_used),
            duration_seconds=duration,
            date_range=date_range,
            start_iso=start.isoformat(),
        )
        span.set(feeds=feed_count, articles=article_count)

        digest_written = _maybe_run_digest(
            run_dir,
            enabled=not no_digest and feed_count > 0,
            date_range=date_range,
        )
        _maybe_render_audio(run_dir, enabled=audio and digest_written)

        _emit_pipeline_tokens(counter)


def _require_fetched(run_dir: Path) -> None:
    """Validate that ``run_dir`` exists and has a populated ``source-fetched/`` subdir."""
    if not run_dir.is_dir():
        typer.echo(f"Error: '{run_dir}' is not a directory.")
        raise typer.Exit(1)
    if not (run_dir / "source-fetched").is_dir():
        typer.echo(f"Error: no source-fetched/ subdirectory in '{run_dir}'.")
        raise typer.Exit(1)


@app.command()
def fetch(
    days: Annotated[
        int | None,
        typer.Option("--days", "-d", help="Number of days back to fetch articles."),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option("--since", help="Start date (YYYY-MM-DD). Mutually exclusive with --days."),
    ] = None,
    until: Annotated[
        str | None,
        typer.Option(
            "--until", help="End date inclusive (YYYY-MM-DD). Mutually exclusive with --days."
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-l", help="Max entries per feed."),
    ] = None,
    content_type: Annotated[
        list[str] | None,
        typer.Option("--content-type", "-c", help="Filter feeds by content type (repeatable)."),
    ] = None,
    feed: Annotated[
        list[str] | None,
        typer.Option("--feed", "-f", help="Filter feeds by name (repeatable)."),
    ] = None,
    feeds_file: Annotated[
        str | None,
        typer.Option("--feeds", help="Path to a feeds.yaml file (overrides discovery)."),
    ] = None,
    config_dir: Annotated[
        str | None,
        typer.Option("--config", help="Config directory holding feeds.yaml."),
    ] = None,
    output_dir: Annotated[
        str | None,
        typer.Option("--output-dir", "-o", help="Override output directory."),
    ] = None,
) -> None:
    """Fetch articles only; print the new run directory path on stdout."""
    _apply_env_overrides(None, output_dir)

    parsed_since, parsed_until = _validate_date_options(days, since, until)

    import asyncio
    from datetime import UTC
    from datetime import datetime as dt

    from digest_generator import api
    from digest_generator.shared.logging import log_stage, logger, run_context
    from digest_generator.sources.rss.io import iter_fetched

    try:
        run_dir, feeds, resolved_filter, date_range = _setup_run(
            output_dir=output_dir,
            content_types=content_type or None,
            feed_names=feed or None,
            days=days,
            parsed_since=parsed_since,
            parsed_until=parsed_until,
            limit=limit,
            feeds_file=feeds_file,
            config_dir=config_dir,
        )
    except ValueError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from None

    with run_context(run_dir.name, run_dir), log_stage("pipeline") as span:
        start = dt.now(tz=UTC)
        logger.info(
            "run_dir={} feeds={} since={} until={} limit={}",
            run_dir,
            len(feeds),
            resolved_filter.since.isoformat() if resolved_filter.since else None,
            resolved_filter.until.isoformat() if resolved_filter.until else None,
            limit,
        )
        asyncio.run(api.fetch(feeds, resolved_filter, run_dir=run_dir))
        duration = (dt.now(tz=UTC) - start).total_seconds()

        feed_count = 0
        article_count = 0
        content_types_used: set[str] = set()
        for content_type_value, _feed_name, entries in iter_fetched(run_dir):
            feed_count += 1
            article_count += len(entries)
            content_types_used.add(str(content_type_value))
        logger.info("Fetch complete: {} articles from {} feeds", article_count, feed_count)

        _write_initial_meta(
            run_dir=run_dir,
            feeds_count=feed_count,
            article_count=article_count,
            content_types_used=sorted(content_types_used),
            duration_seconds=duration,
            date_range=date_range,
            start_iso=start.isoformat(),
        )
        span.set(feeds=feed_count, articles=article_count)

    typer.echo(str(run_dir))


@app.command()
def summarize(
    run_dir: Annotated[
        Path,
        typer.Argument(help="Path to a run directory with source-fetched/ populated."),
    ],
) -> None:
    """Summarize fetched articles in an existing run directory."""
    _require_fetched(run_dir)

    import asyncio

    from digest_generator import api
    from digest_generator.shared.llm.telemetry import llm_telemetry
    from digest_generator.shared.logging import log_stage, logger, run_context

    with (
        run_context(run_dir.name, run_dir),
        log_stage("pipeline"),
        llm_telemetry() as counter,
    ):
        logger.info("run_dir={}", run_dir)
        asyncio.run(api.summarize(run_dir=run_dir))
        _emit_pipeline_tokens(counter)


@app.command()
def label(
    run_dir: Annotated[
        Path,
        typer.Argument(help="Path to a run directory with source-fetched/ populated."),
    ],
    device: Annotated[
        str | None,
        typer.Option(help="Compute device: cpu, cuda, or mps."),
    ] = None,
) -> None:
    """Topic-classify fetched articles in an existing run directory."""
    _apply_env_overrides(device, None)
    _require_fetched(run_dir)

    import asyncio

    from digest_generator import api
    from digest_generator.shared.hf_hub import auto_login
    from digest_generator.shared.logging import log_stage, logger, run_context

    with run_context(run_dir.name, run_dir), log_stage("pipeline"):
        auto_login()
        logger.info("run_dir={}", run_dir)
        asyncio.run(api.label(run_dir=run_dir))


def _read_date_range(run_dir: Path) -> tuple[str, str] | None:
    """Read the since/until date range from a run directory's meta.json."""
    import json

    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return None
    with meta_path.open(encoding="utf-8") as f:
        data = json.load(f)
    since_val = data.get("since")
    until_val = data.get("until")
    if since_val and until_val:
        return (since_val, until_val)
    return None


@app.command()
def digest(
    run_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to a pipeline run directory containing source-summarized/ and source-labeled/."
        ),
    ],
    writer_model: Annotated[
        str | None,
        typer.Option("--writer-model", help="Override section writer Ollama model."),
    ] = None,
    writer_top_p: Annotated[
        float | None,
        typer.Option("--writer-top-p", help="Writer top_p (nucleus sampling cutoff)."),
    ] = None,
    writer_repetition_penalty: Annotated[
        float | None,
        typer.Option(
            "--writer-repetition-penalty",
            help="Writer repetition penalty (Ollama repeat_penalty).",
        ),
    ] = None,
    writer_seed: Annotated[
        int | None,
        typer.Option("--writer-seed", help="Writer RNG seed for reproducible output."),
    ] = None,
    editorial_model: Annotated[
        str | None,
        typer.Option("--editorial-model", help="Override editorial-pass Ollama model."),
    ] = None,
    editorial_top_p: Annotated[
        float | None,
        typer.Option("--editorial-top-p", help="Editorial top_p (nucleus sampling cutoff)."),
    ] = None,
    editorial_repetition_penalty: Annotated[
        float | None,
        typer.Option(
            "--editorial-repetition-penalty",
            help="Editorial repetition penalty (Ollama repeat_penalty).",
        ),
    ] = None,
    editorial_seed: Annotated[
        int | None,
        typer.Option("--editorial-seed", help="Editorial RNG seed for reproducible output."),
    ] = None,
    framer_model: Annotated[
        str | None,
        typer.Option("--framer-model", help="Override framer Ollama model."),
    ] = None,
    framer_top_p: Annotated[
        float | None,
        typer.Option("--framer-top-p", help="Framer top_p (nucleus sampling cutoff)."),
    ] = None,
    framer_repetition_penalty: Annotated[
        float | None,
        typer.Option(
            "--framer-repetition-penalty",
            help="Framer repetition penalty (Ollama repeat_penalty).",
        ),
    ] = None,
    framer_seed: Annotated[
        int | None,
        typer.Option("--framer-seed", help="Framer RNG seed for reproducible output."),
    ] = None,
    watcher_model: Annotated[
        str | None,
        typer.Option("--watcher-model", help="Override watcher Ollama model."),
    ] = None,
    watcher_top_p: Annotated[
        float | None,
        typer.Option("--watcher-top-p", help="Watcher top_p (nucleus sampling cutoff)."),
    ] = None,
    watcher_repetition_penalty: Annotated[
        float | None,
        typer.Option(
            "--watcher-repetition-penalty",
            help="Watcher repetition penalty (Ollama repeat_penalty).",
        ),
    ] = None,
    watcher_seed: Annotated[
        int | None,
        typer.Option("--watcher-seed", help="Watcher RNG seed for reproducible output."),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help="Global RNG seed; applied to any stage without a per-stage --*-seed override.",
        ),
    ] = None,
    audio: Annotated[
        bool,
        typer.Option(
            "--audio",
            help="Render the digest to an Opus audio file after generation.",
        ),
    ] = False,
) -> None:
    """Generate a digest from an existing pipeline run directory."""
    if not run_dir.is_dir():
        typer.echo(f"Error: '{run_dir}' is not a directory.")
        raise typer.Exit(1)

    summarized_dir = run_dir / "source-summarized"
    if not summarized_dir.is_dir():
        typer.echo(f"Error: no source-summarized/ subdirectory in '{run_dir}'.")
        raise typer.Exit(1)

    from digest_generator import api
    from digest_generator.core.digest.io import (
        build_digest_filename,
        build_digest_markdown,
    )
    from digest_generator.shared.llm.sampling import SamplingConfig
    from digest_generator.shared.logging import logger
    from digest_generator.shared.runtime.meta import update_run_meta_digest

    date_range = _read_date_range(run_dir)
    logger.info("Generating digest from {}", run_dir)

    writer_sampling = SamplingConfig(
        top_p=writer_top_p,
        repetition_penalty=writer_repetition_penalty,
        seed=writer_seed,
    ).with_seed_default(seed)
    editorial_sampling = SamplingConfig(
        top_p=editorial_top_p,
        repetition_penalty=editorial_repetition_penalty,
        seed=editorial_seed,
    ).with_seed_default(seed)
    framer_sampling = SamplingConfig(
        top_p=framer_top_p,
        repetition_penalty=framer_repetition_penalty,
        seed=framer_seed,
    ).with_seed_default(seed)
    watcher_sampling = SamplingConfig(
        top_p=watcher_top_p,
        repetition_penalty=watcher_repetition_penalty,
        seed=watcher_seed,
    ).with_seed_default(seed)

    digest_result = api.digest(
        run_dir=run_dir,
        writer_model=writer_model,
        editorial_model=editorial_model,
        framer_model=framer_model,
        watcher_model=watcher_model,
        writer_sampling=writer_sampling,
        editorial_sampling=editorial_sampling,
        framer_sampling=framer_sampling,
        watcher_sampling=watcher_sampling,
        date_range=date_range,
    )
    if not digest_result or not digest_result.content:
        logger.warning("No digest generated (empty result)")
        raise typer.Exit(1)

    digest_filename = build_digest_filename(digest_result)
    digest_path = run_dir / digest_filename
    markdown = build_digest_markdown(digest_result)
    digest_path.write_text(markdown, encoding="utf-8")

    # Update meta.json if it exists (created by `run` command, not by standalone `digest`)
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        update_run_meta_digest(run_dir, digest_result.title, digest_filename)

    logger.info("Digest written to {}", digest_path)

    if audio:
        from digest_generator.shared.logging import run_context

        # No auto_login(): Piper voices live in `rhasspy/piper-voices`,
        # a public HuggingFace repo; hf_hub_download works without auth.
        with run_context(run_dir.name, run_dir):
            try:
                opus_path = api.render_audio(run_dir=run_dir)
                logger.info("Audio written to {}", opus_path)
            except Exception as e:
                logger.exception("Audio rendering failed: {}", e)
                raise typer.Exit(1) from None


@app.command()
def audio(
    run_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to a pipeline run directory containing the composed {date}.md.",
        ),
    ],
    bitrate_kbps: Annotated[
        int | None,
        typer.Option(
            "--bitrate-kbps",
            help="Override the configured Opus bitrate (default from settings.audio_bitrate_kbps).",
        ),
    ] = None,
) -> None:
    """Render the digest at ``run_dir`` to an Opus audio file.

    Workhorse for iterating on the narration pre-pass, with no LLM cost:
    markdown -> Piper -> ffmpeg. Output lands at
    ``{run_dir}/audio/{date}.opus``. Cache-aware: re-running
    against the same markdown + voice + bitrate is a no-op.
    """
    if not run_dir.is_dir():
        typer.echo(f"Error: '{run_dir}' is not a directory.")
        raise typer.Exit(1)

    from digest_generator import api
    from digest_generator.core.audio.io import find_digest_md
    from digest_generator.shared.logging import logger, run_context

    try:
        find_digest_md(run_dir)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from None

    # No auto_login() here: Piper voices live in `rhasspy/piper-voices`,
    # a public HuggingFace repo. hf_hub_download works for public repos
    # without auth, so a defensive login would just turn transient HF API
    # network blips into hard failures on a path that doesn't need auth.
    with run_context(run_dir.name, run_dir):
        try:
            opus_path = api.render_audio(run_dir=run_dir, bitrate_kbps=bitrate_kbps)
        except Exception as e:
            logger.exception("Audio rendering failed: {}", e)
            raise typer.Exit(1) from None
        logger.info("Audio written to {}", opus_path)
        typer.echo(str(opus_path))


@app.command()
def feeds(
    content_type: Annotated[
        list[str] | None,
        typer.Option("--content-type", "-c", help="Filter by content type (repeatable)."),
    ] = None,
    feeds_file: Annotated[
        str | None,
        typer.Option("--feeds", help="Path to a feeds.yaml file (overrides discovery)."),
    ] = None,
    config_dir: Annotated[
        str | None,
        typer.Option("--config", help="Config directory holding feeds.yaml."),
    ] = None,
) -> None:
    """List the configured RSS feeds."""
    from digest_generator.sources.rss.config import load_configured_categories

    try:
        from digest_generator.api import list_feeds

        filtered = list_feeds(
            content_types=content_type or None, feeds_file=feeds_file, config_dir=config_dir
        )
        categories = load_configured_categories(feeds_file=feeds_file, config_dir=config_dir)
    except ValueError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from None

    for category in categories:
        cat_feeds = [f for f in filtered if f.content_type == category.id]
        if not cat_feeds:
            continue
        typer.echo(f"\n{category.title} ({len(cat_feeds)})")
        typer.echo("-" * 40)
        for f in cat_feeds:
            typer.echo(f"  {f.name:<35} {f.url}")

    typer.echo(f"\nTotal: {len(filtered)} feeds")


@app.command()
def init(
    config_dir: Annotated[
        str | None,
        typer.Option("--config", help="Config directory to write feeds.yaml into."),
    ] = None,
    feeds_file: Annotated[
        str | None,
        typer.Option("--feeds", help="Exact path to write the feeds.yaml to."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing feeds.yaml."),
    ] = False,
) -> None:
    """Write a starter feeds.yaml to get started.

    Defaults to ~/.config/digest-generator/feeds.yaml. Edit the file to add
    your own categories and feeds, then run `digest-generator feeds` to check it.
    """
    from digest_generator.sources.rss.config import write_starter_feeds

    try:
        written = write_starter_feeds(feeds_file=feeds_file, config_dir=config_dir, force=force)
    except (ValueError, OSError) as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from None

    typer.echo(f"Wrote starter feeds config to {written}")
    typer.echo("Edit it to add your feeds, then run: digest-generator feeds")
