"""Story-cluster stage for the digest pipeline.

``ArticleClusterer`` groups articles describing the same story and picks
each cluster's primary category (the section that owns the full
writeup) plus optional secondary sections that should cross-reference
rather than expand the story. The downstream writer consumes the result
so a story M&A in three sections gets one full paragraph plus two
``<covered-elsewhere>`` references rather than three full paragraphs (e.g.
a single Cloudflare/Stripe story that would otherwise be expanded in both
the Infrastructure and Business sections).

Output is cached as ``clusters.json`` under ``run_dir/`` and surfaces in
``meta.json``'s ``stages.clusterer`` block via the standard
``log_stage`` / ``collect_stage_telemetry`` harvest.

Two clustering strategies share a single output contract:

- **LLM** (default when ``settings.clusterer_model`` resolves and the
  input is non-empty): the model reads every article block, groups
  same-story articles, picks ``primary_section`` editorially, and emits
  optional ``secondary_sections``.
- **Trivial fallback** (one size-1 cluster per article, no secondaries,
  primary = the article's ``content_type``): fires when the LLM
  returns empty / non-JSON / non-array output, when no parsed cluster
  survives validation, or when a network/Ollama error raises during
  the call. Mirrors ``WhatToWatch``'s malformed-JSON-to-empty-list
  pattern: clustering must never block the pipeline.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from digest_generator.core.categories import CategorySet, category_registry
from digest_generator.core.digest.prompts import load_prompt
from digest_generator.core.digest.types import Cluster
from digest_generator.shared.llm.clients import client_registry
from digest_generator.shared.llm.sampling import SamplingConfig, resolve_ollama_options
from digest_generator.shared.llm.telemetry import chat_with_logging
from digest_generator.shared.logging import log_stage, logger
from digest_generator.shared.settings import settings

if TYPE_CHECKING:
    from ollama import Client


_CLUSTER_SYSTEM_PROMPT = load_prompt("cluster_system")

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)

# Articles fed to the prompt carry only the fields needed for clustering.
# Full content is omitted; the LLM-generated summary is the dense substitute.
_MAX_SUMMARY_CHARS = 600  # truncate runaway summaries to keep prompt budget predictable
_MAX_TOPICS = 3


class ArticleClusterer:
    """Group articles into story clusters with primary/secondary section routing.

    Args:
        client: Ollama client instance. Defaults to auto-detected client from
                ``ClientRegistry`` (cloud if ``OLLAMA_API_KEY`` is set, else local).
        model: Ollama model name. Defaults to ``CLUSTERER_MODEL`` env var, or
               ``WRITER_MODEL`` if ``CLUSTERER_MODEL`` is unset.
        sampling: Per-call sampling overrides (``temperature``, ``top_p``,
                ``repetition_penalty``, ``seed``). Unset fields fall back to
                the matching ``clusterer_*`` settings.
        categories: Section set used to validate cluster routing. Defaults to
                the configured category set from ``category_registry``.
    """

    def __init__(
        self,
        client: Client | None = None,
        model: str | None = None,
        sampling: SamplingConfig | None = None,
        categories: CategorySet | None = None,
    ) -> None:
        self._client = client or client_registry.ollama
        self.model = model or settings.clusterer_model or settings.writer_model
        self._sampling = sampling
        self._categories = categories or category_registry.active

    def cluster(self, results: dict[str, list[dict[str, Any]]]) -> list[Cluster]:
        """Cluster pre-joined article dicts (one entry per article) into stories.

        Input shape matches what the writer receives: the merged
        ``source-summarized/`` + ``source-labeled/`` view built by
        ``digest_generator.api._load_digest_input``, keyed by feed name. Output is
        a flat ``list[Cluster]`` (cluster id is the cross-cluster
        discriminator, not the feed).

        Returns ``[]`` for empty input. Otherwise returns at least one
        cluster per article: the LLM path may merge articles into
        multi-article clusters; the fallback path emits size-1 clusters.
        """
        with log_stage("clusterer") as span:
            articles = _flatten(results)
            if not articles:
                span.set(clusters=0, articles_covered=0, model=self.model)
                return []

            # Build an id-to-article index up-front so every code path (LLM
            # success, LLM failure, partial coverage) can backfill cleanly.
            indexed = _assign_article_ids(articles)
            valid_sections = self._categories.id_set()
            clusters, strategy = self._cluster_via_llm(indexed, valid_sections)
            if not clusters:
                clusters = _trivial_fallback(indexed, valid_sections)
                strategy = "trivial_one_per_article"

            primary_counts: dict[str, int] = {}
            for c in clusters:
                if c.primary_section:
                    primary_counts[c.primary_section] = primary_counts.get(c.primary_section, 0) + 1
            span.set(
                clusters=len(clusters),
                articles_covered=sum(len(c.article_urls) for c in clusters),
                primary_section_counts=primary_counts,
                secondary_refs=sum(len(c.secondary_sections) for c in clusters),
                model=self.model,
                strategy=strategy,
            )
            return clusters

    def _cluster_via_llm(
        self, indexed: list[tuple[str, dict[str, Any]]], valid_sections: frozenset[str]
    ) -> tuple[list[Cluster], str]:
        """Run the LLM clustering call and validate the response.

        Returns ``([], strategy)`` on any failure mode so the caller can
        fall through to the trivial fallback. ``strategy`` reports what
        actually happened (``"llm"`` on success; ``"llm_error"`` /
        ``"llm_empty"`` / ``"llm_malformed"`` on the soft failures) and
        is recorded on the stage span for eval-harness inspection.
        """
        if not self.model:
            return [], "no_model_configured"

        user_prompt = _build_user_prompt(indexed, self._categories)
        try:
            raw = self._call_llm(_CLUSTER_SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            logger.warning("Clusterer LLM call failed ({}); falling back to one-per-article", e)
            return [], "llm_error"

        if not raw or not raw.strip():
            logger.warning("Clusterer returned empty output; falling back to one-per-article")
            return [], "llm_empty"

        parsed = _parse_clusters_json(raw)
        if parsed is None:
            return [], "llm_malformed"

        clusters = _validate_and_normalize(parsed, indexed, valid_sections)
        if not clusters:
            logger.warning("Clusterer returned no valid clusters; falling back to one-per-article")
            return [], "llm_no_valid_clusters"

        return clusters, "llm"

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        options = resolve_ollama_options(
            self._sampling,
            temperature=settings.clusterer_temperature,
            top_p=settings.clusterer_top_p,
            repetition_penalty=settings.clusterer_repetition_penalty,
            seed=settings.clusterer_seed,
        )
        return chat_with_logging(
            self._client,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options=options,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (module-level so they're trivially unit-testable without an instance)
# ─────────────────────────────────────────────────────────────────────────────


def _flatten(results: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten the feed-to-articles dict into a single ordered list of articles."""
    out: list[dict[str, Any]] = []
    for feed_articles in results.values():
        out.extend(feed_articles)
    return out


def _assign_article_ids(
    articles: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """Assign a stable LLM-facing id (``a0001``, …) to each article.

    Returned order matches input order; the canonical cluster id is
    assigned later by the validator/fallback.
    """
    return [(f"a{i:04d}", article) for i, article in enumerate(articles, start=1)]


def _build_user_prompt(indexed: list[tuple[str, dict[str, Any]]], categories: CategorySet) -> str:
    """Render the article blocks for the LLM. One ``<article>`` per input row.

    Opens with a ``<sections>`` block listing the configured category ids and
    titles so the model routes clusters to real section ids. Tags per article:
    ``title``, ``origin``, ``published`` (date only), ``feed_section`` (the
    article's content_type), ``topics`` (top-N by confidence), ``summary``,
    ``url``. Description and full content are deliberately omitted; the
    summary is the dense substitute.
    """
    lines: list[str] = [
        "<task>",
        f"Cluster the {len(indexed)} articles below into story groups and",
        "route each cluster to a primary section (plus optional secondary",
        "sections). Output a JSON array per the system prompt's output-format.",
        "</task>",
        "",
        "<sections>",
    ]
    for category in categories:
        lines.append(f'  <section id="{category.id}">{category.title}</section>')
    lines.append("</sections>")
    lines.append("")
    lines.append("<articles>")
    for aid, article in indexed:
        lines.append(f'<article id="{aid}">')
        lines.append(f"  <title>{_clean(article.get('title', ''))}</title>")
        if article.get("origin"):
            lines.append(f"  <origin>{_clean(article['origin'])}</origin>")
        published = _date_only(article.get("published", ""))
        if published:
            lines.append(f"  <published>{published}</published>")
        if article.get("content_type"):
            lines.append(f"  <feed_section>{article['content_type']}</feed_section>")
        topics = _format_topics(article.get("topics") or [])
        if topics:
            lines.append(f"  <topics>{topics}</topics>")
        summary = _clean(article.get("summary", ""))
        if summary:
            if len(summary) > _MAX_SUMMARY_CHARS:
                summary = summary[:_MAX_SUMMARY_CHARS].rsplit(" ", 1)[0] + "…"
            lines.append(f"  <summary>{summary}</summary>")
        if article.get("url"):
            lines.append(f"  <url>{article['url']}</url>")
        lines.append("</article>")
    lines.append("</articles>")
    return "\n".join(lines)


def _clean(text: str) -> str:
    """Strip whitespace and collapse internal newlines so XML stays well-formed."""
    return " ".join(str(text).split())


def _date_only(published: str) -> str:
    """Return the ``YYYY-MM-DD`` prefix of a published timestamp, or ``""``."""
    if not published:
        return ""
    return str(published)[:10]


def _format_topics(topics: Any) -> str:
    """Render the top-N topic labels by confidence as a comma-separated string.

    ``_load_digest_input`` emits ``topics`` as a ``dict[label, confidence]``
    pre-sorted by confidence desc (the upstream BART-MNLI classifier's
    output shape). Falls back to handling ``list[Label]`` / ``list[dict]``
    for direct programmatic callers who pass the raw label objects.
    """
    if isinstance(topics, dict):
        ranked = sorted(topics.items(), key=lambda kv: kv[1], reverse=True)
        return ", ".join(str(label) for label, _ in ranked[:_MAX_TOPICS])
    if isinstance(topics, list):
        values: list[str] = []
        for t in topics[:_MAX_TOPICS]:
            if isinstance(t, dict) and "value" in t:
                values.append(str(t["value"]))
            elif hasattr(t, "value"):
                values.append(str(t.value))
            else:
                values.append(str(t))
        return ", ".join(v for v in values if v)
    return ""


def _parse_clusters_json(raw: str) -> list[dict[str, Any]] | None:
    """Strip optional code fence, parse JSON, return raw list-of-dicts or ``None``.

    ``None`` signals a soft failure the caller should treat as "use the
    trivial fallback." Non-list payloads and non-dict elements are filtered
    here so the validator stage only sees plausible inputs.
    """
    text = raw.strip()
    fence_match = _JSON_FENCE_RE.match(text)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Clusterer returned non-JSON output: {}", e)
        return None
    if not isinstance(data, list):
        logger.warning("Clusterer returned non-array JSON: {}", type(data).__name__)
        return None
    return [item for item in data if isinstance(item, dict)]


def _validate_and_normalize(
    raw_clusters: list[dict[str, Any]],
    indexed: list[tuple[str, dict[str, Any]]],
    valid_sections: frozenset[str],
) -> list[Cluster]:
    """Validate LLM-emitted clusters, backfill missing articles, re-id canonically.

    Per-article rule: every input article must appear in exactly one
    output cluster. Articles the LLM omitted or duplicated are reconciled:

    - **First wins** for duplicates (an article assigned to multiple
      clusters is kept in the first one it appears in; later occurrences
      drop it).
    - **Backfill as size-1** for omissions (each unreferenced article
      becomes its own size-1 cluster appended at the end, primary
      section = its ``content_type`` when valid, no secondaries).

    Per-cluster rule: clusters with no surviving articles are dropped.
    Invalid ``primary_section`` values fall back to the cluster's
    majority article ``content_type``; if even that fails, the cluster
    is dropped (its articles flow into backfill).

    Cluster IDs are re-assigned canonically (``c0001`` …) on output so
    the LLM-chosen handles are scoped to the parse step only.
    """
    by_id = dict(indexed)
    assigned: set[str] = set()
    accepted: list[Cluster] = []

    for raw in raw_clusters:
        ids_raw = raw.get("articles", [])
        if not isinstance(ids_raw, list):
            continue
        article_ids = [aid for aid in ids_raw if isinstance(aid, str) and aid in by_id]
        # Drop already-assigned articles (first-wins on cross-cluster duplicates).
        article_ids = [aid for aid in article_ids if aid not in assigned]
        if not article_ids:
            continue

        primary = _coerce_section(raw.get("primary_section", ""), valid_sections)
        if not primary:
            primary = _majority_content_type([by_id[aid] for aid in article_ids], valid_sections)
        if not primary:
            continue  # cluster has no defensible section; backfill will pick up its articles

        secondary = _coerce_secondary_sections(
            raw.get("secondary_sections", []), primary, valid_sections
        )
        lede = _clean(str(raw.get("lede", ""))) or _clean(by_id[article_ids[0]].get("title", ""))
        entities = _coerce_entities(raw.get("entities", []))

        urls = [str(by_id[aid].get("url", "")) for aid in article_ids]
        urls = [u for u in urls if u]
        if not urls:
            continue

        accepted.append(
            Cluster(
                id="",  # re-numbered below
                lede=lede,
                article_urls=urls,
                primary_section=primary,
                secondary_sections=secondary,
                entities=entities,
            )
        )
        assigned.update(article_ids)

    # Backfill any article the LLM dropped as its own size-1 cluster.
    for aid, article in indexed:
        if aid in assigned:
            continue
        primary = _coerce_section(article.get("content_type", ""), valid_sections)
        url = str(article.get("url", ""))
        accepted.append(
            Cluster(
                id="",
                lede=_clean(article.get("title", "")),
                article_urls=[url] if url else [],
                primary_section=primary,
                secondary_sections=[],
                entities=[],
            )
        )

    # Canonical re-numbering: c0001, c0002, ...
    return [
        Cluster(
            id=f"c{idx:04d}",
            lede=c.lede,
            article_urls=c.article_urls,
            primary_section=c.primary_section,
            secondary_sections=c.secondary_sections,
            entities=c.entities,
        )
        for idx, c in enumerate(accepted, start=1)
    ]


def _coerce_section(value: Any, valid_sections: frozenset[str]) -> str:
    """Return the section string if it's a known category id, else ``""``."""
    if not isinstance(value, str):
        return ""
    return value if value in valid_sections else ""


_MAX_ENTITIES = 4
_MAX_ENTITY_CHARS = 60


def _coerce_entities(value: Any) -> list[str]:
    """Coerce LLM-emitted entities to a clean ``list[str]`` capped at length.

    Non-list payloads become ``[]``. Each entity is whitespace-collapsed,
    stripped, truncated to ``_MAX_ENTITY_CHARS``, deduped (case-insensitive),
    and the list is capped at ``_MAX_ENTITIES``.
    """
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = _clean(item)
        if not cleaned:
            continue
        if len(cleaned) > _MAX_ENTITY_CHARS:
            cleaned = cleaned[:_MAX_ENTITY_CHARS].rstrip()
        key = cleaned.lower()
        if key in seen:
            continue
        out.append(cleaned)
        seen.add(key)
        if len(out) >= _MAX_ENTITIES:
            break
    return out


def _coerce_secondary_sections(
    value: Any, primary: str, valid_sections: frozenset[str]
) -> list[str]:
    """Filter to known category ids, drop ``primary``, dedupe, cap at 2."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        section = _coerce_section(item, valid_sections)
        if not section or section == primary or section in seen:
            continue
        out.append(section)
        seen.add(section)
        if len(out) >= 2:
            break
    return out


def _majority_content_type(articles: list[dict[str, Any]], valid_sections: frozenset[str]) -> str:
    """Pick the most common valid ``content_type`` across the cluster.

    Ties broken by input order (first-seen wins). Returns ``""`` when
    no article has a valid content_type.
    """
    counts: dict[str, int] = {}
    for article in articles:
        ct = _coerce_section(article.get("content_type", ""), valid_sections)
        if ct:
            counts[ct] = counts.get(ct, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _trivial_fallback(
    indexed: list[tuple[str, dict[str, Any]]], valid_sections: frozenset[str]
) -> list[Cluster]:
    """Emit one size-1 cluster per article with no cross-section refs.

    Primary section is the article's fetched ``content_type`` when
    valid; articles with missing or unrecognized ``content_type`` get
    an empty ``primary_section`` so downstream consumers can flag them
    (matches the writer's drop-on-invalid behavior for section
    assignment without losing the article from the cluster record).
    """
    clusters: list[Cluster] = []
    for index, (_aid, article) in enumerate(indexed, start=1):
        url = str(article.get("url", ""))
        title = _clean(article.get("title", ""))
        primary = _coerce_section(article.get("content_type", ""), valid_sections)
        clusters.append(
            Cluster(
                id=f"c{index:04d}",
                lede=title,
                article_urls=[url] if url else [],
                primary_section=primary,
                secondary_sections=[],
            )
        )
    return clusters
