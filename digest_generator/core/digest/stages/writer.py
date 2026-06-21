"""Map-phase section drafting for the weekly digest pipeline.

``SectionWriter`` generates one ``SectionDraft`` per category bucket
via one LLM call (or per-batch + merge for large buckets). It produces no
digest assembly, title, or framing; those are separate downstream stages.

Uses the native ``ollama`` client via DI from ``ClientRegistry``.
Configured via ``WRITER_MODEL`` environment variable.
"""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from ollama import Client

from digest_generator.core.categories import CategorySet, category_registry
from digest_generator.core.digest.prompts import load_prompt
from digest_generator.core.digest.types import Cluster, SectionDraft
from digest_generator.shared.llm.clients import client_registry
from digest_generator.shared.llm.sampling import SamplingConfig, resolve_ollama_options
from digest_generator.shared.llm.telemetry import chat_with_logging
from digest_generator.shared.logging import log_stage, logger
from digest_generator.shared.settings import settings

_SECTION_SYSTEM_PROMPT = load_prompt("section_system")
_SECTION_MERGE_SYSTEM_PROMPT = load_prompt("section_merge_system")


class SectionWriter:
    """Generates per-section prose drafts from labeled article summaries.

    Connects to an Ollama instance (local or cloud) via the native ``ollama``
    client. For each category bucket, makes one LLM call to produce a
    ``SectionDraft``. Large buckets are split into batches that are drafted
    per-batch then merged in one additional LLM call.

    Args:
        client: Ollama client instance. Defaults to auto-detected client from
                ``ClientRegistry`` (cloud if ``OLLAMA_API_KEY`` is set, else local).
        model: Ollama model name. Defaults to ``WRITER_MODEL`` env var.
        sampling: Per-call sampling overrides (``temperature``, ``top_p``,
                ``repetition_penalty``, ``seed``). Unset fields fall back to
                the matching ``writer_*`` settings.
        categories: Section set (order + titles). Defaults to the configured
                category set from ``category_registry``.
    """

    def __init__(
        self,
        client: Client | None = None,
        model: str | None = None,
        sampling: SamplingConfig | None = None,
        categories: CategorySet | None = None,
    ):
        self._client = client or client_registry.ollama
        self.model = model or settings.writer_model
        self._sampling = sampling
        self._categories = categories or category_registry.active

    def write_all_from_json(
        self,
        results: dict[str, list[dict[str, Any]]],
        *,
        date_range: tuple[str, str] | None = None,
        clusters: list[Cluster] | None = None,
    ) -> list[SectionDraft]:
        """Generate section drafts from dict-based JSON input.

        Accepts pre-loaded article dicts (e.g. read from ``summaries/*.json``)
        and produces one ``SectionDraft`` per non-empty category bucket.

        When ``clusters`` is supplied (produced by ``ArticleClusterer``),
        article-to-section routing follows ``cluster.primary_section``
        rather than each article's ``content_type``. Sections also receive
        a ``<covered-elsewhere>`` block per cluster that lists them in its
        ``secondary_sections``: the writer references but does not expand
        those stories. When ``clusters`` is ``None`` the writer falls back
        to grouping by ``content_type`` for callers that bypass the clusterer.
        """
        with log_stage("writer") as span:
            grouped = self._group_by_clusters(results, clusters)
            cross_refs = self._build_cross_refs(clusters or [])
            drafts = self._draft_sections(grouped, cross_refs, date_range=date_range)
            span.set(
                sections=len(drafts),
                articles=sum(d.article_count for d in drafts),
                cross_refs=sum(len(v) for v in cross_refs.values()),
                model=self.model,
            )
            return drafts

    def _draft_sections(
        self,
        grouped: dict[str, list[dict[str, Any]]],
        cross_refs: dict[str, list[Cluster]],
        *,
        date_range: tuple[str, str] | None = None,
    ) -> list[SectionDraft]:
        """Map phase: one ``SectionDraft`` per category in section order."""
        drafts: list[SectionDraft] = []
        for category in self._categories:
            articles = grouped.get(category.id, [])
            section_cross_refs = cross_refs.get(category.id, [])
            if not articles and not section_cross_refs:
                continue
            if not articles:
                # Section has only cross-references and no primary articles,
                # so skip rather than emit an empty section. The cross-references
                # for this section can only appear inline with primary content;
                # without that anchor the secondary references would float.
                logger.debug(
                    "Skipping {} section — {} cross-refs but no primary articles",
                    category.title,
                    len(section_cross_refs),
                )
                continue
            display_name = category.title
            logger.info(
                "Generating {} section ({} articles, {} cross-refs) via {}",
                display_name,
                len(articles),
                len(section_cross_refs),
                self.model,
            )
            content = self._write_section(
                display_name, articles, section_cross_refs, date_range=date_range
            )
            if not content:
                continue
            logger.debug(
                "{} section draft: {} words, {} chars",
                display_name,
                len(content.split()),
                len(content),
            )
            drafts.append(
                SectionDraft(name=display_name, content=content, article_count=len(articles))
            )
        return drafts

    def _call_llm(
        self, system_prompt: str, user_prompt: str, *, temperature: float | None = None
    ) -> str:
        """Send a chat request to Ollama and return the response text.

        Returns an empty string if the model returns no content.
        """
        sampling = self._sampling
        if temperature is not None:
            sampling = replace(sampling or SamplingConfig(), temperature=temperature)
        options = resolve_ollama_options(
            sampling,
            temperature=settings.writer_temperature,
            top_p=settings.writer_top_p,
            repetition_penalty=settings.writer_repetition_penalty,
            seed=settings.writer_seed,
        )
        content = chat_with_logging(
            self._client,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options=options,
        )
        if not content:
            return ""
        logger.debug("LLM response: {} chars, {} words", len(content), len(content.split()))
        return content

    @staticmethod
    def _rank_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort articles by their highest topic confidence score, descending."""

        def max_confidence(article: dict[str, Any]) -> float:
            topics = article.get("topics", {})
            if not topics:
                return 0.0
            return float(max(topics.values()))

        return sorted(articles, key=max_confidence, reverse=True)

    def _write_section(
        self,
        section_name: str,
        articles: list[dict[str, Any]],
        cross_refs: list[Cluster],
        *,
        date_range: tuple[str, str] | None = None,
    ) -> str:
        """Generate a section draft, splitting into batches if needed.

        When articles exceed ``settings.writer_section_batch_size``, splits into
        sub-batches, generates a draft per batch, then merges the sub-drafts.
        ``cross_refs`` are threaded into every batch: each batch sees the same
        ``<covered-elsewhere>`` block so the writer respects the no-expand rule
        across all sub-drafts.
        """
        articles = self._rank_articles(articles)
        if len(articles) <= settings.writer_section_batch_size:
            return self._write_section_draft(
                section_name, articles, cross_refs, date_range=date_range
            )

        batches = [
            articles[i : i + settings.writer_section_batch_size]
            for i in range(0, len(articles), settings.writer_section_batch_size)
        ]
        logger.info(
            "Splitting {} into {} batches of ~{} articles",
            section_name,
            len(batches),
            settings.writer_section_batch_size,
        )

        sub_drafts: list[str] = []
        for i, batch in enumerate(batches, 1):
            logger.info("  Batch {}/{} ({} articles)", i, len(batches), len(batch))
            draft = self._write_section_draft(
                section_name, batch, cross_refs, date_range=date_range
            )
            if draft:
                sub_drafts.append(draft)

        if not sub_drafts:
            return ""
        if len(sub_drafts) == 1:
            return sub_drafts[0]

        logger.info("Merging {} sub-drafts for {} section", len(sub_drafts), section_name)
        return self._merge_section_drafts(section_name, sub_drafts)

    def _write_section_draft(
        self,
        section_name: str,
        articles: list[dict[str, Any]],
        cross_refs: list[Cluster],
        *,
        date_range: tuple[str, str] | None = None,
    ) -> str:
        """Generate a single section draft from a batch of articles via one LLM call."""
        if date_range:
            period = f"covering {date_range[0]} to {date_range[1]}"
        else:
            period = f"week of {datetime.now(tz=UTC).strftime('%Y-%m-%d')}"
        lines = [
            "<task>",
            f'Write the "{section_name}" section for the weekly digest ({period}).',
            f"You have {len(articles)} articles to synthesize.",
            "</task>",
            "",
        ]
        if cross_refs:
            lines.extend(self._format_cross_refs(cross_refs))
            lines.append("")
        lines.extend(self._format_articles(articles))
        return self._call_llm(_SECTION_SYSTEM_PROMPT, "\n".join(lines))

    def _merge_section_drafts(self, section_name: str, drafts: list[str]) -> str:
        """Merge multiple sub-drafts of the same section into one cohesive narrative."""
        lines = [
            "<task>",
            f'Merge these {len(drafts)} partial drafts into a single "{section_name}" section.',
            "</task>",
            "",
        ]
        for i, draft in enumerate(drafts, 1):
            lines.append(f'<partial-draft number="{i}">')
            lines.append(draft)
            lines.append("</partial-draft>")
            lines.append("")

        return self._call_llm(_SECTION_MERGE_SYSTEM_PROMPT, "\n".join(lines))

    @staticmethod
    def _format_articles(articles: list[dict[str, Any]]) -> list[str]:
        """Format articles as XML elements for structured LLM prompts.

        Each article is wrapped in ``<article>`` tags with sub-elements for
        title, url, source, published date, description (publisher-authored
        RSS blurb, when present), summary (LLM-generated by ContentSummarizer), and top-3 topics.
        The ``description`` tag is omitted for feeds that don't carry one.
        """
        lines = ["<articles>"]
        for article in articles:
            lines.append("<article>")
            lines.append(f"  <title>{article['title']}</title>")
            lines.append(f"  <url>{article['url']}</url>")
            lines.append(f"  <origin>{article.get('origin', '')}</origin>")
            lines.append(f"  <published>{article.get('published', '')}</published>")
            description = str(article.get("description", "")).strip()
            if description:
                lines.append(f"  <description>{description}</description>")
            lines.append(f"  <summary>{article['summary']}</summary>")
            content_head = str(article.get("content_head", "")).strip()
            if content_head:
                lines.append(f"  <content_head>{content_head}</content_head>")
            topics = article.get("topics", {})
            if topics:
                top_topics = sorted(topics.items(), key=lambda t: t[1], reverse=True)[:3]
                topic_str = ", ".join(name for name, _ in top_topics)
                lines.append(f"  <topics>{topic_str}</topics>")
            lines.append("</article>")
        lines.append("</articles>")
        return lines

    def _group_by_clusters(
        self,
        results: dict[str, list[dict[str, Any]]],
        clusters: list[Cluster] | None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Group articles by ``cluster.primary_section`` (fallback to ``content_type``).

        Builds a map from ``url`` to ``primary_section`` from ``clusters`` and routes
        each article by URL lookup. Articles whose URL isn't in the map
        (cluster-less callers, or clusters with empty ``primary_section``)
        fall back to their ``content_type``. Articles whose section isn't a
        known category by either path are dropped.

        ``cluster.primary_section`` always overrides ``content_type``: the
        LLM clusterer has full routing authority, including for
        single-article clusters where it re-categorizes the article away
        from its feed default.
        """
        url_to_primary: dict[str, str] = {}
        if clusters:
            for cluster in clusters:
                if not cluster.primary_section:
                    continue
                for url in cluster.article_urls:
                    url_to_primary[url] = cluster.primary_section

        grouped: dict[str, list[dict[str, Any]]] = {}
        for articles in results.values():
            for article in articles:
                url = article.get("url", "")
                section = url_to_primary.get(url) or article.get("content_type", "")
                if section not in self._categories:
                    continue
                grouped.setdefault(section, []).append(article)
        return grouped

    def _build_cross_refs(self, clusters: list[Cluster]) -> dict[str, list[Cluster]]:
        """Map each section to the clusters that touch it as a secondary."""
        refs: dict[str, list[Cluster]] = {}
        for cluster in clusters:
            for section in cluster.secondary_sections:
                if section not in self._categories:
                    continue
                refs.setdefault(section, []).append(cluster)
        return refs

    def _format_cross_refs(self, cross_refs: list[Cluster]) -> list[str]:
        """Render the ``<covered-elsewhere>`` block for the section's user prompt.

        One ``<cluster>`` element per cross-reference with ``lede``,
        ``entities``, and ``urls`` sub-elements. The ``primary`` attribute
        is the section's title (e.g. ``"AI & Machine Learning"``, not
        ``"ai"``) so the writer can reference the section by its rendered
        name when citing the cross-reference.
        """
        lines = ["<covered-elsewhere>"]
        for cluster in cross_refs:
            primary_display = self._categories.title(cluster.primary_section)
            lines.append(f'  <cluster primary="{primary_display}">')
            lines.append(f"    <lede>{cluster.lede}</lede>")
            if cluster.entities:
                lines.append(f"    <entities>{', '.join(cluster.entities)}</entities>")
            if cluster.article_urls:
                lines.append(f"    <urls>{', '.join(cluster.article_urls)}</urls>")
            lines.append("  </cluster>")
        lines.append("</covered-elsewhere>")
        return lines
