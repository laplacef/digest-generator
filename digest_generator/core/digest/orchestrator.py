"""Six-stage digest pipeline orchestration.

Wires the stage classes from ``digest_generator.core.digest.stages`` into the linear
the clusterer, writer, editor, framer, watcher, and composer stages in sequence, with
per-stage caching under ``run_dir/``. Each stage's output file acts as its
cache key: re-running on the same ``run_dir`` skips completed stages.

The framer's intro paragraph is threaded into ``_run_watcher`` as a dedup
signal so the watcher can pick complementary tensions instead of restating
the lede's thesis.

Public entry point:

- ``run_digest_from_json``: runs from pre-serialized ``dict[str, list[dict]]``
  (caller builds the merged article view; ``digest_generator.api._load_digest_input``
  is the canonical builder, joining ``source-summarized/`` + ``source-labeled/`` in memory)

``digest_generator.api.digest`` wraps this for external callers; the CLI uses
``digest_generator.api``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from digest_generator.core.digest.stages.editorial import SectionEditor
    from digest_generator.core.digest.types import (
        Cluster,
        DigestFraming,
        DigestResult,
        SectionDraft,
        WatchItem,
    )
    from digest_generator.shared.llm.sampling import SamplingConfig
    from digest_generator.shared.runtime.meta import SamplingMeta, StageMeta


def run_digest_from_json(
    results: dict[str, list[dict[str, Any]]],
    *,
    run_dir: Path | None = None,
    writer_model: str | None = None,
    editorial_model: str | None = None,
    framer_model: str | None = None,
    watcher_model: str | None = None,
    clusterer_model: str | None = None,
    writer_sampling: SamplingConfig | None = None,
    editorial_sampling: SamplingConfig | None = None,
    framer_sampling: SamplingConfig | None = None,
    watcher_sampling: SamplingConfig | None = None,
    clusterer_sampling: SamplingConfig | None = None,
    date_range: tuple[str, str] | None = None,
) -> DigestResult:
    """Run the digest pipeline against pre-serialized summary dicts.

    Materializes each stage's sampling (resolving the user override, then
    settings, then random layers into a concrete config with a pinned seed) before the
    pipeline runs so the same inputs and the same recorded ``meta.json``
    can reproduce the run bit-for-bit. The materialized state is
    persisted to ``meta.json`` v2 alongside per-stage telemetry, per-
    section editor outcomes, and the digest's terminal metrics. The
    caller does not need to thread any of this through.
    """
    from digest_generator.core.digest.stages.clusterer import ArticleClusterer
    from digest_generator.core.digest.stages.writer import SectionWriter
    from digest_generator.shared.settings import settings

    resolved_models: dict[str, str | None] = {
        "writer": writer_model or settings.writer_model,
        "editor": editorial_model or settings.editorial_model,
        "framer": framer_model or settings.framer_model or settings.writer_model,
        "watcher": watcher_model or settings.watcher_model or settings.writer_model,
        "clusterer": clusterer_model or settings.clusterer_model or settings.writer_model,
    }
    sampling_state = {
        "writer": _build_sampling_state(
            user=writer_sampling,
            model=resolved_models["writer"],
            default_temperature=settings.writer_temperature,
            default_top_p=settings.writer_top_p,
            default_repetition_penalty=settings.writer_repetition_penalty,
            default_seed=settings.writer_seed,
        ),
        "editor": _build_sampling_state(
            user=editorial_sampling,
            model=resolved_models["editor"],
            default_temperature=settings.editorial_temperature,
            default_top_p=settings.editorial_top_p,
            default_repetition_penalty=settings.editorial_repetition_penalty,
            default_seed=settings.editorial_seed,
        ),
        "framer": _build_sampling_state(
            user=framer_sampling,
            model=resolved_models["framer"],
            default_temperature=settings.framer_temperature,
            default_top_p=settings.framer_top_p,
            default_repetition_penalty=settings.framer_repetition_penalty,
            default_seed=settings.framer_seed,
        ),
        "watcher": _build_sampling_state(
            user=watcher_sampling,
            model=resolved_models["watcher"],
            default_temperature=settings.watcher_temperature,
            default_top_p=settings.watcher_top_p,
            default_repetition_penalty=settings.watcher_repetition_penalty,
            default_seed=settings.watcher_seed,
        ),
        "clusterer": _build_sampling_state(
            user=clusterer_sampling,
            model=resolved_models["clusterer"],
            default_temperature=settings.clusterer_temperature,
            default_top_p=settings.clusterer_top_p,
            default_repetition_penalty=settings.clusterer_repetition_penalty,
            default_seed=settings.clusterer_seed,
        ),
    }
    sampling_metas = {k: v[1] for k, v in sampling_state.items()}
    materialized = {k: v[0] for k, v in sampling_state.items()}

    def build_clusters() -> list[Cluster]:
        return ArticleClusterer(model=clusterer_model, sampling=materialized["clusterer"]).cluster(
            results
        )

    def build_sections(clusters: list[Cluster]) -> list[SectionDraft]:
        return SectionWriter(
            model=writer_model, sampling=materialized["writer"]
        ).write_all_from_json(results, date_range=date_range, clusters=clusters)

    return _orchestrate(
        build_sections,
        build_clusters=build_clusters,
        run_dir=run_dir,
        editorial_model=editorial_model,
        framer_model=framer_model,
        watcher_model=watcher_model,
        editorial_sampling=materialized["editor"],
        framer_sampling=materialized["framer"],
        watcher_sampling=materialized["watcher"],
        date_range=date_range,
        sampling_metas=sampling_metas,
        resolved_models=resolved_models,
    )


def _build_sampling_state(
    *,
    user: SamplingConfig | None,
    model: str | None,
    default_temperature: float | None,
    default_top_p: float | None,
    default_repetition_penalty: float | None,
    default_seed: int | None,
) -> tuple[SamplingConfig, SamplingMeta]:
    """Materialize sampling for one stage and build its ``SamplingMeta`` record.

    Returns ``(materialized_config, sampling_meta)``. The materialized config
    is what the stage will actually pass to Ollama; the ``SamplingMeta``
    captures all three layers (user / model_defaults / effective) for
    ``meta.json``. Querying ``ollama.show`` is cached per-model in
    ``fetch_model_defaults`` so concurrent stages on the same model pay the
    network call once.
    """
    from digest_generator.shared.llm.clients import client_registry
    from digest_generator.shared.llm.sampling import (
        fetch_model_defaults,
        materialize_sampling,
    )
    from digest_generator.shared.runtime.meta import SamplingLayer, SamplingMeta

    if model is None:
        # Defensive: every stage's model resolves to a real string at the
        # call site (writer/editor have non-None settings defaults; framer/
        # watcher fall back to writer_model). If model is None here it means
        # the stage has no resolvable model and would crash in Ollama anyway,
        # so return an empty meta and let the orchestrator surface that cleanly.
        return SamplingConfig(), SamplingMeta()
    model_defaults = fetch_model_defaults(client_registry.ollama, model)
    resolved, seed_source = materialize_sampling(
        user,
        default_temperature=default_temperature,
        default_top_p=default_top_p,
        default_repetition_penalty=default_repetition_penalty,
        default_seed=default_seed,
    )
    user_layer = SamplingLayer(
        temperature=user.temperature if user is not None else None,
        top_p=user.top_p if user is not None else None,
        repetition_penalty=user.repetition_penalty if user is not None else None,
        seed=user.seed if user is not None else None,
    )
    md_seed = model_defaults.get("seed")
    md_layer = SamplingLayer(
        temperature=model_defaults.get("temperature"),
        top_p=model_defaults.get("top_p"),
        # Ollama wire format uses ``repeat_penalty``; the user-facing name is
        # ``repetition_penalty``. Translate at the boundary so meta.json carries
        # one consistent name across all three layers.
        repetition_penalty=model_defaults.get("repeat_penalty"),
        seed=int(md_seed) if md_seed is not None else None,
    )
    eff_layer = SamplingLayer(
        temperature=resolved.temperature
        if resolved.temperature is not None
        else md_layer.temperature,
        top_p=resolved.top_p if resolved.top_p is not None else md_layer.top_p,
        repetition_penalty=resolved.repetition_penalty
        if resolved.repetition_penalty is not None
        else md_layer.repetition_penalty,
        seed=resolved.seed,
    )
    meta = SamplingMeta(
        user=user_layer,
        model_defaults=md_layer,
        effective=eff_layer,
        seed_source=seed_source,
    )
    return resolved, meta


def _orchestrate(
    build_sections: Callable[[list[Cluster]], list[SectionDraft]],
    *,
    build_clusters: Callable[[], list[Cluster]] | None = None,
    run_dir: Path | None,
    editorial_model: str | None,
    framer_model: str | None,
    watcher_model: str | None,
    editorial_sampling: SamplingConfig | None,
    framer_sampling: SamplingConfig | None,
    watcher_sampling: SamplingConfig | None,
    date_range: tuple[str, str] | None,
    sampling_metas: dict[str, SamplingMeta] | None = None,
    resolved_models: dict[str, str | None] | None = None,
) -> DigestResult:
    """Run the full six-stage pipeline with per-stage caching.

    When ``run_dir`` is provided, stage telemetry / sampling state /
    section outcomes / digest output metrics are harvested into the
    ``meta.json`` v2 schema via ``update_run_meta_telemetry`` after the
    pipeline completes. The caller does not need to thread telemetry
    through: this function owns the harvest.
    """
    from digest_generator.core.digest.stages.composer import DigestComposer
    from digest_generator.core.digest.stages.editorial import SectionEditor
    from digest_generator.core.digest.types import DigestResult
    from digest_generator.shared.llm.telemetry import llm_telemetry
    from digest_generator.shared.logging import collect_stage_telemetry, logger

    # Editor instance is constructed up-front so we can read its per-section
    # outcomes after the run (they're populated by clean_all and reset on
    # each call). Other stages stay inline since they don't expose post-run
    # state beyond their span fields.
    editor = SectionEditor(model=editorial_model, sampling=editorial_sampling)

    with llm_telemetry() as counter, collect_stage_telemetry() as stage_sink:
        clusters: list[Cluster] = []
        if build_clusters is not None:
            clusters = _cached_or_fresh_clusters(run_dir, build_clusters)

        drafts = _cached_or_fresh_sections(run_dir, "sections", lambda: build_sections(clusters))

        if not drafts:
            logger.warning("No section drafts generated — skipping remaining stages")
            return DigestResult(
                title="",
                content="",
                date=date_range[1] if date_range else "",
                word_count=0,
                reading_time_minutes=0,
                article_count=0,
                section_counts={},
            )

        cleaned = _cached_or_fresh_sections(
            run_dir,
            "sections_edited",
            lambda: editor.clean_all(drafts),
        )
        framing = _cached_or_fresh_framing(
            run_dir,
            lambda: _run_framer(cleaned, framer_model, framer_sampling, date_range),
        )
        watch = _cached_or_fresh_watch(
            run_dir,
            lambda: _run_watcher(cleaned, framing.intro, watcher_model, watcher_sampling, clusters),
        )
        result = DigestComposer().compose(cleaned, framing, watch, date_range=date_range)

        logger.bind(stage="digest").info(
            "digest.tokens prompt={} completion={} llm_calls={} llm_duration_ms={} per_stage={}",
            counter.prompt_tokens,
            counter.completion_tokens,
            counter.llm_calls,
            counter.llm_duration_ms,
            counter.per_stage,
        )

    if run_dir is not None:
        _persist_telemetry(
            run_dir=run_dir,
            stage_sink=stage_sink,
            editor=editor,
            counter=counter,
            result=result,
            watch_count=len(watch),
            sampling_metas=sampling_metas,
            resolved_models=resolved_models,
        )

    return result


def _persist_telemetry(
    *,
    run_dir: Path,
    stage_sink: dict[str, dict[str, Any]],
    editor: SectionEditor,
    counter: Any,
    result: DigestResult,
    watch_count: int,
    sampling_metas: dict[str, SamplingMeta] | None,
    resolved_models: dict[str, str | None] | None,
) -> None:
    """Roll the orchestrator's harvested state into ``meta.json``.

    Each block lands as a single ``update_run_meta_telemetry`` call so a
    crash mid-pipeline leaves a consistent file on disk (write happens
    once after every harvest source has reported).
    """
    from digest_generator.shared.runtime.meta import (
        DigestMeta,
        StageMeta,
        update_run_meta_telemetry,
    )

    common_keys = {
        "duration_ms",
        "llm_calls",
        "prompt_tokens",
        "completion_tokens",
        "llm_duration_ms",
        "model",
    }
    stages_meta: dict[str, StageMeta] = {}
    for name, fields in stage_sink.items():
        extras = {k: v for k, v in fields.items() if k not in common_keys}
        stages_meta[name] = StageMeta(
            duration_ms=int(fields.get("duration_ms", 0)),
            llm_calls=int(fields.get("llm_calls", 0)),
            prompt_tokens=int(fields.get("prompt_tokens", 0)),
            completion_tokens=int(fields.get("completion_tokens", 0)),
            llm_duration_ms=int(fields.get("llm_duration_ms", 0)),
            model=fields.get("model"),
            extras=extras,
        )
    digest_meta = DigestMeta(
        title=result.title,
        filename=None,  # CLI patches this via update_run_meta_digest after build_digest_filename
        word_count=result.word_count,
        reading_time_minutes=result.reading_time_minutes,
        article_count=result.article_count,
        section_counts=dict(result.section_counts),
        watch_item_count=watch_count,
    )
    update_run_meta_telemetry(
        run_dir,
        models=resolved_models,
        stages=stages_meta,
        sampling=sampling_metas,
        sections=editor.section_outcomes,
        totals={
            "llm_calls": counter.llm_calls,
            "prompt_tokens": counter.prompt_tokens,
            "completion_tokens": counter.completion_tokens,
            "llm_duration_ms": counter.llm_duration_ms,
        },
        digest=digest_meta,
    )


def _run_framer(
    sections: list[SectionDraft],
    model: str | None,
    sampling: SamplingConfig | None,
    date_range: tuple[str, str] | None,
) -> DigestFraming:
    from digest_generator.core.digest.stages.framer import DigestFramer

    return DigestFramer(model=model, sampling=sampling).frame(sections, date_range=date_range)


def _run_watcher(
    sections: list[SectionDraft],
    lede_intro: str,
    model: str | None,
    sampling: SamplingConfig | None,
    clusters: list[Cluster] | None = None,
) -> list[WatchItem]:
    from digest_generator.core.digest.stages.watcher import WhatToWatch

    return WhatToWatch(model=model, sampling=sampling).generate(
        sections, lede_intro=lede_intro, clusters=clusters
    )


# Stage-cache directory names. The orchestrator passes the stage_name
# ("sections" / "sections_edited") to ``_cached_or_fresh_sections``; the
# helper maps it to the cache directory name via this mapping. Stage
# identities in meta.json / log_stage stay decoupled from directory names
# by design.
_SECTION_DIRS = {
    "sections": "section-drafts",
    "sections_edited": "section-edits",
}
_ASSEMBLY_DIR = "assembly"


def _cached_or_fresh_sections(
    run_dir: Path | None,
    stage_name: str,
    builder: Callable[[], list[SectionDraft]],
) -> list[SectionDraft]:
    """Return cached section drafts if present, else build and save them."""
    from digest_generator.core.digest.io import load_section_drafts, save_section_drafts
    from digest_generator.shared.logging import logger

    target_name = _SECTION_DIRS.get(stage_name, stage_name)

    if run_dir is not None:
        cache_dir = run_dir / target_name
        cached = load_section_drafts(cache_dir)
        if cached:
            logger.bind(stage="cache").info(
                "cache-hit stage={} sections={}", stage_name, len(cached)
            )
            return cached

    drafts = builder()
    if run_dir is not None:
        save_section_drafts(drafts, run_dir / target_name)
    return drafts


def _cached_or_fresh_framing(
    run_dir: Path | None,
    builder: Callable[[], DigestFraming],
) -> DigestFraming:
    """Return cached framing if present, else build and save."""
    from digest_generator.core.digest.io import load_json_stage, save_json_stage
    from digest_generator.core.digest.types import DigestFraming
    from digest_generator.shared.logging import logger

    if run_dir is not None:
        path = run_dir / _ASSEMBLY_DIR / "framing.json"
        loaded: Any = load_json_stage(path, DigestFraming)
        if isinstance(loaded, DigestFraming):
            logger.bind(stage="cache").info("cache-hit stage=framing")
            return loaded

    framing = builder()
    if run_dir is not None:
        save_json_stage(framing, run_dir / _ASSEMBLY_DIR / "framing.json")
    return framing


def _cached_or_fresh_watch(
    run_dir: Path | None,
    builder: Callable[[], list[WatchItem]],
) -> list[WatchItem]:
    """Return cached watch items if present, else build and save."""
    from digest_generator.core.digest.io import load_json_stage, save_json_stage
    from digest_generator.core.digest.types import WatchItem
    from digest_generator.shared.logging import logger

    if run_dir is not None:
        path = run_dir / _ASSEMBLY_DIR / "watch.json"
        if path.exists():
            loaded: Any = load_json_stage(path, WatchItem)
            if isinstance(loaded, list):
                logger.bind(stage="cache").info("cache-hit stage=watch items={}", len(loaded))
                return loaded

    watch = builder()
    if run_dir is not None:
        save_json_stage(watch, run_dir / _ASSEMBLY_DIR / "watch.json")
    return watch


def _cached_or_fresh_clusters(
    run_dir: Path | None,
    builder: Callable[[], list[Cluster]],
) -> list[Cluster]:
    """Return cached clusters if present, else build and save.

    The cache key is file presence (matches every other digest stage).
    Empty cluster lists are persisted as ``[]`` and treated as cache hits.
    No clusters is a legitimate output for empty-corpus runs.
    """
    from digest_generator.core.digest.io import load_json_stage, save_json_stage
    from digest_generator.core.digest.types import Cluster
    from digest_generator.shared.logging import logger

    if run_dir is not None:
        path = run_dir / _ASSEMBLY_DIR / "clusters.json"
        if path.exists():
            loaded: Any = load_json_stage(path, Cluster)
            if isinstance(loaded, list):
                logger.bind(stage="cache").info("cache-hit stage=clusters clusters={}", len(loaded))
                return loaded

    clusters = builder()
    if run_dir is not None:
        save_json_stage(clusters, run_dir / _ASSEMBLY_DIR / "clusters.json")
    return clusters
