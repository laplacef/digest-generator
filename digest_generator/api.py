"""Public programmatic API for the digest-generator pipeline.

Six entry points: three async per-stage primitives (``fetch``,
``summarize``, ``label``), the sync digest pipeline (``digest``), the
``run`` composition, and the audio renderer (``render_audio``). Each
per-stage primitive is independently invocable and cache-aware: re-running on
the same ``run_dir`` skips work whose batch file already exists.

Run setup (``run_dir`` creation, ``run_context``, ``log_stage``,
``llm_telemetry``, ``meta.json``) is the caller's responsibility.
``digest_generator.cli`` handles all of that for command-line use; programmatic
callers stitch together what they need.

Usage::

    import asyncio
    from pathlib import Path

    from digest_generator.api import resolve_feeds, run, digest
    from digest_generator.core.types import Filter

    feeds = resolve_feeds(content_types=["ai"])
    filter = Filter.resolve(days_back=7)
    run_dir = Path("output/myrun")
    run_dir.mkdir(parents=True, exist_ok=True)

    run(feeds, filter, run_dir=run_dir, with_digest=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from digest_generator.core.audio.renderer import AudioRenderer
    from digest_generator.core.digest.types import DigestResult
    from digest_generator.core.label import TopicClassifier
    from digest_generator.core.summary import ContentSummarizer
    from digest_generator.core.types import Filter
    from digest_generator.shared.llm.sampling import SamplingConfig
    from digest_generator.sources.rss.fetcher import FeedFetcher
    from digest_generator.sources.rss.types import Feed


def resolve_feeds(
    content_types: list[str] | None = None,
    feed_names: list[str] | None = None,
    *,
    feeds_file: str | None = None,
    config_dir: str | None = None,
) -> list[Feed]:
    """Load the configured feeds and filter by content type and/or name.

    Args:
        content_types: Content type values to include (e.g., ``["ai", "security"]``).
        feed_names: Feed names to include (e.g., ``["openai-news"]``).
        feeds_file: Explicit ``feeds.yaml`` path; overrides discovery.
        config_dir: Config directory holding ``feeds.yaml``; overrides discovery.

    Returns:
        Filtered list of ``Feed`` objects.

    Raises:
        FeedsConfigError: If no feeds file is found or it is invalid.
        ValueError: If a content type or feed name is invalid, or no feeds match.
    """
    from digest_generator.sources.rss.config import (
        load_configured_categories,
        load_configured_feeds,
    )

    all_feeds = load_configured_feeds(feeds_file=feeds_file, config_dir=config_dir)
    result = all_feeds

    if content_types:
        valid = load_configured_categories(feeds_file=feeds_file, config_dir=config_dir).id_set()
        for ct in content_types:
            if ct not in valid:
                msg = f"Unknown content type '{ct}'. Valid: {', '.join(sorted(valid))}"
                raise ValueError(msg)
        result = [f for f in result if f.content_type in content_types]

    if feed_names:
        known = {f.name for f in all_feeds}
        for name in feed_names:
            if name not in known:
                msg = f"Unknown feed '{name}'."
                raise ValueError(msg)
        result = [f for f in result if f.name in set(feed_names)]

    if not result:
        msg = "No feeds matched the given filters."
        raise ValueError(msg)

    return result


async def fetch(
    feeds: list[Feed],
    filter: Filter,
    *,
    run_dir: Path,
    fetcher: FeedFetcher | None = None,
) -> None:
    """Fetch entries from each feed and persist them under ``source-fetched/``.

    Per-feed task fans out via ``asyncio.TaskGroup``, capped by
    ``settings.fetch_concurrency``. Skips any feed whose batch file
    already exists, so re-running on the same ``run_dir`` is idempotent
    and only fetches feeds that haven't landed yet. Each per-feed task
    runs the fetcher's own ``log_stage("fetcher")`` span.

    Args:
        feeds: Feeds to fetch.
        filter: Resolved date / limit filter.
        run_dir: Run root for cache files.
        fetcher: Injected fetcher (mostly for tests). Defaults to a fresh
            ``FeedFetcher()``.
    """
    import asyncio

    from digest_generator.shared.settings import settings
    from digest_generator.sources.rss.fetcher import FeedFetcher
    from digest_generator.sources.rss.io import load_entries, save_entries

    fetcher_obj = fetcher or FeedFetcher()
    semaphore = asyncio.Semaphore(settings.fetch_concurrency)

    async def _one(feed: Feed) -> None:
        if load_entries(run_dir, feed.name) is not None:
            return  # cache hit, skip
        try:
            async with semaphore:
                entries = await fetcher_obj.fetch_entries(feed, filter)
        except Exception as e:
            # One bad feed must not kill the run. ``asyncio.TaskGroup`` cancels
            # every sibling task on an unhandled exception, so per-feed failures
            # (truncated responses, SSL handshake errors, malformed XML) get
            # swallowed here and surface as a logged warning instead. The lack
            # of a saved batch leaves the feed cache-miss-able on the next
            # run, so a transient failure is automatically retryable.
            from digest_generator.shared.logging import logger

            logger.warning(
                "Fetch failed for feed '{}' ({}): {} — skipping; cache miss preserved for retry",
                feed.name,
                type(e).__name__,
                e,
            )
            return
        save_entries(run_dir, feed.name, entries)

    async with asyncio.TaskGroup() as tg:
        for feed in feeds:
            tg.create_task(_one(feed))


async def summarize(
    *,
    run_dir: Path,
    summarizer: ContentSummarizer | None = None,
) -> None:
    """Read fetched batches, summarize each via LLM, persist under ``source-summarized/``.

    Iterates ``iter_fetched(run_dir)``. Skips any feed whose summarized
    batch already exists. Per-feed processing is sequential at this
    layer; in-flight LLM concurrency is capped by the summarizer's own
    instance-level semaphore (``summarizer_concurrency``).

    Topic labels are NOT populated here; the summarized JSON has empty
    topic lists. The label branch writes ``source-labeled/`` separately;
    ``api.digest`` joins both at read time.

    Stage telemetry + materialized sampling are persisted to ``meta.json``
    after the per-feed loop completes (once per call rather than once per
    feed), so concurrent ``api.summarize`` and ``api.label`` from ``api.run``
    each contribute exactly one meta-file write.

    Args:
        run_dir: Run root with ``source-fetched/`` populated.
        summarizer: Injected summarizer (mostly for tests). Defaults to
            a fresh ``ContentSummarizer()``.
    """
    from digest_generator.core.digest.orchestrator import _build_sampling_state
    from digest_generator.core.summary import ContentSummarizer
    from digest_generator.core.summary.io import load_summarized, save_summarized
    from digest_generator.shared.logging import collect_stage_telemetry
    from digest_generator.shared.runtime.meta import StageMeta, update_run_meta_telemetry
    from digest_generator.shared.settings import settings
    from digest_generator.sources.rss.io import iter_fetched

    summarizer_obj = summarizer or ContentSummarizer()

    # Materialize sampling once for the run so repeated ``summarize_entries``
    # calls (one per feed) all share the same seed, and that seed lands in
    # ``meta.json`` even if the user didn't pass one. Materialization here
    # rather than inside ``ContentSummarizer.__init__`` keeps the stage
    # class agnostic to meta-file plumbing.
    user_sampling = getattr(summarizer_obj, "_sampling", None)
    materialized_sampling, sampling_meta = _build_sampling_state(
        user=user_sampling,
        model=summarizer_obj.model,
        default_temperature=settings.summarizer_temperature,
        default_top_p=settings.summarizer_top_p,
        default_repetition_penalty=settings.summarizer_repetition_penalty,
        default_seed=settings.summarizer_seed,
    )
    summarizer_obj._sampling = materialized_sampling

    with collect_stage_telemetry() as sink:
        for _content_type, feed_name, entries in iter_fetched(run_dir):
            if load_summarized(run_dir, feed_name) is not None:
                continue
            summaries = await summarizer_obj.summarize_entries(entries, feed=feed_name)
            save_summarized(run_dir, feed_name, summaries)

    if "summarizer" in sink:
        fields = sink["summarizer"]
        common_keys = {
            "duration_ms",
            "llm_calls",
            "prompt_tokens",
            "completion_tokens",
            "llm_duration_ms",
            "model",
        }
        stage_meta = StageMeta(
            duration_ms=int(fields.get("duration_ms", 0)),
            llm_calls=int(fields.get("llm_calls", 0)),
            prompt_tokens=int(fields.get("prompt_tokens", 0)),
            completion_tokens=int(fields.get("completion_tokens", 0)),
            llm_duration_ms=int(fields.get("llm_duration_ms", 0)),
            model=fields.get("model"),
            extras={k: v for k, v in fields.items() if k not in common_keys},
        )
        if (run_dir / "meta.json").exists():
            update_run_meta_telemetry(
                run_dir,
                stages={"summarizer": stage_meta},
                sampling={"summarizer": sampling_meta},
            )


async def label(
    *,
    run_dir: Path,
    classifier: TopicClassifier | None = None,
) -> None:
    """Read fetched batches, classify raw text, persist under ``source-labeled/``.

    Iterates ``iter_fetched(run_dir)``. Skips any feed whose labeled
    batch already exists. Inference runs in ``asyncio.to_thread`` per
    batch (BART-MNLI is blocking).

    Independent of the summarizer: uses ``entry.content_head`` (with
    title + description fallback) so the two stages can run
    concurrently against the same fetched corpus. The output is
    URL-keyed so ``api.digest`` can join it against ``source-summarized/``
    at read time without a persistent merged artifact.

    Stage telemetry is harvested into ``meta.json``'s ``stages.topic``
    block (no sampling, since the topic classifier is HF rather than Ollama).

    Args:
        run_dir: Run root with ``source-fetched/`` populated.
        classifier: Injected classifier (mostly for tests). Defaults to
            a fresh ``TopicClassifier`` from ``model_registry``.
    """
    import asyncio

    from digest_generator.core.label import TopicClassifier
    from digest_generator.core.label.io import load_labeled, save_labeled
    from digest_generator.shared.logging import collect_stage_telemetry
    from digest_generator.shared.runtime.meta import StageMeta, update_run_meta_telemetry
    from digest_generator.shared.transformers.registry import model_registry
    from digest_generator.sources.rss.io import iter_fetched

    classifier_obj = classifier or TopicClassifier(model_config=model_registry.topic)

    with collect_stage_telemetry() as sink:
        for _content_type, feed_name, entries in iter_fetched(run_dir):
            if load_labeled(run_dir, feed_name) is not None:
                continue
            labels_per_entry = await asyncio.to_thread(
                classifier_obj.classify_entries, entries, feed=feed_name
            )
            urls = [e.url for e in entries]
            save_labeled(
                run_dir,
                feed_name,
                urls=urls,
                labels_per_entry=labels_per_entry,
            )

    if "topic" in sink:
        fields = sink["topic"]
        common_keys = {
            "duration_ms",
            "llm_calls",
            "prompt_tokens",
            "completion_tokens",
            "llm_duration_ms",
            "model",
        }
        stage_meta = StageMeta(
            duration_ms=int(fields.get("duration_ms", 0)),
            llm_calls=int(fields.get("llm_calls", 0)),
            prompt_tokens=int(fields.get("prompt_tokens", 0)),
            completion_tokens=int(fields.get("completion_tokens", 0)),
            llm_duration_ms=int(fields.get("llm_duration_ms", 0)),
            model=fields.get("model"),
            extras={k: v for k, v in fields.items() if k not in common_keys},
        )
        if (run_dir / "meta.json").exists():
            update_run_meta_telemetry(run_dir, stages={"topic": stage_meta})


def run(
    feeds: list[Feed],
    filter: Filter,
    *,
    run_dir: Path,
    fetcher: FeedFetcher | None = None,
    summarizer: ContentSummarizer | None = None,
    classifier: TopicClassifier | None = None,
    with_digest: bool = True,
    with_audio: bool = False,
    writer_model: str | None = None,
    editorial_model: str | None = None,
    framer_model: str | None = None,
    watcher_model: str | None = None,
    writer_sampling: SamplingConfig | None = None,
    editorial_sampling: SamplingConfig | None = None,
    framer_sampling: SamplingConfig | None = None,
    watcher_sampling: SamplingConfig | None = None,
    date_range: tuple[str, str] | None = None,
) -> Path:
    """Build the corpus and (optionally) the digest.

    Composition: ``fetch``, then ``summarize`` and ``label`` in parallel, then digest
    (if ``with_digest``). The summarize and label branches run
    concurrently against the same fetched corpus and produce
    independent on-disk artifacts (``source-summarized/`` and ``source-labeled/``).
    ``api.digest`` joins them in memory at read time, with no persistent
    merged artifact.

    Run setup (``run_dir`` creation, ``meta.json``, ``run_context``,
    ``llm_telemetry()``, feed resolution) is the caller's responsibility.

    Args:
        feeds: Feeds to fetch / summarize / label.
        filter: Resolved date / limit filter.
        run_dir: Run root for cache files.
        fetcher: Injected fetcher (mostly for tests).
        summarizer: Injected summarizer (mostly for tests).
        classifier: Injected classifier (mostly for tests).
        with_digest: When ``True`` (default), generate the markdown digest
            after the corpus build. Maps to today's ``--no-digest`` CLI flag.
        with_audio: When ``True``, render the digest to audio after the
            digest stage. Default ``False`` so dev iteration stays cheap;
            pass ``--audio`` explicitly to enable it.
            Implies ``with_digest=True``.
        writer_model / editorial_model / framer_model / watcher_model:
            Override Ollama models for the digest stages.
        writer_sampling / editorial_sampling / framer_sampling / watcher_sampling:
            Sampling overrides for digest stages.
        date_range: Optional ``(since, until)`` date strings for digest period.

    Returns:
        ``run_dir`` (data lives on disk; ``api.digest`` is the canonical
        consumer that joins ``source-summarized/`` + ``source-labeled/``).
    """
    import asyncio

    async def _build_corpus() -> None:
        await fetch(feeds, filter, run_dir=run_dir, fetcher=fetcher)
        await asyncio.gather(
            summarize(run_dir=run_dir, summarizer=summarizer),
            label(run_dir=run_dir, classifier=classifier),
        )

    asyncio.run(_build_corpus())

    if with_digest:
        digest(
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

    if with_audio:
        if not with_digest:
            msg = "with_audio=True requires with_digest=True (no digest, nothing to narrate)"
            raise ValueError(msg)
        render_audio(run_dir=run_dir)

    return run_dir


def _load_digest_input(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Build the merged article view ``api.digest`` consumes.

    Reads ``source-summarized/<feed>.json`` (Summary fields), looks up
    ``source-labeled/<feed>.json`` for the URL-keyed topics, and
    ``source-fetched/<feed>.json`` for the content_type tier
    (carried in entry data). Returns ``{feed_name: [merged_dict, ...]}``.

    A feed without a labeled or fetched batch is still included: the
    article gets ``topics: {}`` or ``content_type: None`` respectively.
    The digest stages tolerate missing fields, and per-stage failures
    are surfaced via run.log rather than by dropping articles silently.
    """
    from digest_generator.core.label.io import load_labeled
    from digest_generator.core.summary.io import iter_summarized
    from digest_generator.sources.rss.io import iter_fetched

    feed_content_type: dict[str, str] = {
        feed_name: str(content_type) for content_type, feed_name, _ in iter_fetched(run_dir)
    }

    results: dict[str, list[dict[str, Any]]] = {}
    for _src, feed_name, summarized_articles in iter_summarized(run_dir):
        labels_by_url = load_labeled(run_dir, feed_name) or {}
        content_type = feed_content_type.get(feed_name)

        merged: list[dict[str, Any]] = []
        for article in summarized_articles:
            url = article.get("url", "")
            topics = labels_by_url.get(url, [])
            merged.append(
                {
                    **article,
                    "content_type": content_type,
                    "topics": {label.value: label.confidence for label in topics},
                }
            )
        results[feed_name] = merged
    return results


def digest(
    *,
    run_dir: Path,
    writer_model: str | None = None,
    editorial_model: str | None = None,
    framer_model: str | None = None,
    watcher_model: str | None = None,
    writer_sampling: SamplingConfig | None = None,
    editorial_sampling: SamplingConfig | None = None,
    framer_sampling: SamplingConfig | None = None,
    watcher_sampling: SamplingConfig | None = None,
    date_range: tuple[str, str] | None = None,
) -> DigestResult | None:
    """Generate the markdown digest from a run dir.

    Reads ``source-summarized/`` and ``source-labeled/`` and joins them
    in memory by URL into the merged article shape downstream digest
    stages expect. Returns ``None`` when ``source-summarized/`` is empty
    (nothing to digest).

    Args:
        run_dir: Run directory containing ``source-summarized/<feed>.json``
            and ``source-labeled/<feed>.json``.
        writer_model / editorial_model / framer_model / watcher_model:
            Override Ollama models for digest stages.
        writer_sampling / editorial_sampling / framer_sampling / watcher_sampling:
            Sampling overrides.
        date_range: Optional ``(since, until)`` strings for the digest period.
    """
    from digest_generator.core.digest.orchestrator import run_digest_from_json
    from digest_generator.shared.logging import logger, run_context

    with run_context(run_dir.name, run_dir):
        results = _load_digest_input(run_dir)
        if not results:
            logger.warning("No summarized files found in {}", run_dir / "source-summarized")
            return None

        total = sum(len(v) for v in results.values())
        logger.info("Loaded {} articles from {} feeds", total, len(results))

        return run_digest_from_json(
            results,
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


def render_audio(
    *,
    run_dir: Path,
    bitrate_kbps: int | None = None,
    renderer: AudioRenderer | None = None,
) -> Path:
    """Render the digest at ``run_dir`` to an Opus audio file.

    Locates the deliverable ``{date}-{slug}.md`` at the run root,
    narrates it via ``core.audio.narration``, synthesizes via Piper,
    encodes via ffmpeg, and writes ``audio/{date}-{slug}.opus``.
    Cache-aware: re-running against the same markdown + voice + bitrate
    is a no-op (cache hit short-circuits at the renderer level).

    Telemetry is harvested into ``meta.json``'s ``stages.audio`` block
    (``duration_ms``, ``voice``, ``narration_chars``, ``audio_bytes``,
    ``audio_duration_s``, ``real_time_factor``, ``cached``) plus
    ``models.audio = "piper:<voice_id>"`` when the run has a
    ``meta.json``. Telemetry is skipped silently if ``meta.json`` is
    absent (programmatic callers without the CLI's meta lifecycle).

    Voice and bitrate come from ``settings.audio_voice_model`` /
    ``settings.audio_bitrate_kbps`` unless ``bitrate_kbps`` overrides
    or a fully-configured ``renderer`` is injected. Multi-voice
    support is a deferred follow-up.

    Args:
        run_dir: Run directory holding the digest ``.md`` deliverable.
        bitrate_kbps: Override the configured Opus bitrate (24 by default).
        renderer: Injected renderer (mostly for tests).

    Returns:
        Path to the ``.opus`` artifact under ``audio/``.

    Raises:
        FileNotFoundError: No ``.md`` deliverable at the run root.
        ValueError: Multiple ``.md`` files at the run root (ambiguous).
    """
    from digest_generator.core.audio.io import find_digest_md
    from digest_generator.core.audio.renderer import AudioRenderer as _AudioRenderer
    from digest_generator.shared.logging import collect_stage_telemetry, log_stage
    from digest_generator.shared.runtime.meta import StageMeta, update_run_meta_telemetry
    from digest_generator.shared.settings import settings
    from digest_generator.shared.tts.registry import voice_registry

    digest_md = find_digest_md(run_dir)
    bitrate = bitrate_kbps if bitrate_kbps is not None else settings.audio_bitrate_kbps

    if renderer is None:
        voice = voice_registry.default
        renderer = _AudioRenderer(
            voice=voice,
            bitrate_kbps=bitrate,
            sentence_silence_s=settings.audio_sentence_silence_s,
            ffmpeg_path=settings.audio_ffmpeg_path,
        )

    with collect_stage_telemetry() as sink, log_stage("audio") as span:
        artifact = renderer.render(run_dir, digest_md)
        span.set(
            voice=artifact.voice_id,
            bitrate_kbps=artifact.bitrate_kbps,
            narration_chars=artifact.narration_chars,
            audio_bytes=artifact.audio_bytes,
            audio_duration_s=round(artifact.audio_duration_s, 1),
            cached=artifact.cached,
        )

    if "audio" in sink and (run_dir / "meta.json").exists():
        fields = dict(sink["audio"])
        duration_ms = int(fields.get("duration_ms", 0))
        # real_time_factor = seconds of audio per second of wall time. >1 means
        # synthesis runs faster than real time. Skip on cache hit since the
        # synthesis pipeline didn't fire and ``duration_ms`` only reflects the
        # cache-lookup overhead.
        if not artifact.cached and duration_ms > 0:
            fields["real_time_factor"] = round(
                artifact.audio_duration_s / (duration_ms / 1000.0), 1
            )
        common_keys = {
            "duration_ms",
            "llm_calls",
            "prompt_tokens",
            "completion_tokens",
            "llm_duration_ms",
            "model",
        }
        stage_meta = StageMeta(
            duration_ms=duration_ms,
            extras={k: v for k, v in fields.items() if k not in common_keys},
        )
        update_run_meta_telemetry(
            run_dir,
            models={"audio": f"piper:{artifact.voice_id}"},
            stages={"audio": stage_meta},
        )

    return artifact.opus_path


def list_feeds(
    content_types: list[str] | None = None,
    *,
    feeds_file: str | None = None,
    config_dir: str | None = None,
) -> list[Feed]:
    """Return the configured feeds, optionally filtered by content type.

    Args:
        content_types: Content type values to filter by.
        feeds_file: Explicit ``feeds.yaml`` path; overrides discovery.
        config_dir: Config directory holding ``feeds.yaml``; overrides discovery.

    Returns:
        List of ``Feed`` objects.

    Raises:
        FeedsConfigError: If no feeds file is found or it is invalid.
    """
    if content_types:
        return resolve_feeds(
            content_types=content_types, feeds_file=feeds_file, config_dir=config_dir
        )

    from digest_generator.sources.rss.config import load_configured_feeds

    return load_configured_feeds(feeds_file=feeds_file, config_dir=config_dir)
