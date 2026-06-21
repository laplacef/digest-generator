"""Cross-section "what to watch" stage for the weekly digest pipeline.

``WhatToWatch`` reads every cleaned section draft and surfaces 2-3
cross-cutting watch items (tensions, emerging trends, or collisions
between sections) that deserve attention over the next few weeks. Output
is a list of ``WatchItem`` parsed from the model's JSON response.

Configured via ``WATCHER_MODEL`` environment variable. Falls back to
``WRITER_MODEL`` if unset.
"""

import json
import re
from typing import Any

from ollama import Client

from digest_generator.core.categories import CategorySet, category_registry
from digest_generator.core.digest.prompts import load_prompt
from digest_generator.core.digest.types import Cluster, SectionDraft, WatchItem
from digest_generator.shared.llm.clients import client_registry
from digest_generator.shared.llm.sampling import SamplingConfig, resolve_ollama_options
from digest_generator.shared.llm.telemetry import chat_with_logging
from digest_generator.shared.llm.typography import normalize_typography
from digest_generator.shared.logging import log_stage, logger
from digest_generator.shared.settings import settings

_WATCH_SYSTEM_PROMPT = load_prompt("watch_system")

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


class WhatToWatch:
    """Generates 2-3 cross-cutting watch items from the cleaned section drafts.

    Args:
        client: Ollama client instance. Defaults to auto-detected client from
                ``ClientRegistry`` (cloud if ``OLLAMA_API_KEY`` is set, else local).
        model: Ollama model name. Defaults to ``WATCHER_MODEL`` env var, or
               ``WRITER_MODEL`` if ``WATCHER_MODEL`` is unset.
        sampling: Per-call sampling overrides (``temperature``, ``top_p``,
                ``repetition_penalty``, ``seed``). Unset fields fall back to
                the matching ``watcher_*`` settings.
    """

    def __init__(
        self,
        client: Client | None = None,
        model: str | None = None,
        sampling: SamplingConfig | None = None,
        categories: CategorySet | None = None,
    ):
        self._client = client or client_registry.ollama
        self.model = model or settings.watcher_model or settings.writer_model
        self._sampling = sampling
        self._categories = categories or category_registry.active

    def generate(
        self,
        sections: list[SectionDraft],
        *,
        lede_intro: str | None = None,
        clusters: list[Cluster] | None = None,
    ) -> list[WatchItem]:
        """Generate watch items from cleaned section drafts.

        Args:
            sections: Cleaned section drafts produced by the editor.
            lede_intro: Optional framer intro paragraph. When supplied, the
                user prompt includes a ``<lede-already-framed>`` block so the
                watcher knows which thesis the digest's Overview already
                covers and can pick complementary tensions instead.
            clusters: Optional list of story clusters from ``ArticleClusterer``.
                When supplied, the user prompt includes a ``<clusters>``
                index (id + lede + primary/secondary sections) so the
                watcher picks tensions that span ≥2 *clusters* (the unit
                of story) rather than ≥2 sections (the unit of topic).
                Cluster ledes are the same ones cited in the writer's
                cross-references; the watcher should not restate them as
                watch-item headings.

        Returns an empty list if the model returns empty or malformed output.
        The outer pipeline treats an empty watch list as "skip the section".
        """
        with log_stage("watcher") as span:
            if not sections:
                span.set(items=0, reason="no-sections")
                return []
            logger.info("Generating What to Watch items via {}", self.model)
            raw = self._call_llm(
                _WATCH_SYSTEM_PROMPT,
                self._build_user_prompt(sections, lede_intro=lede_intro, clusters=clusters),
            )
            if not raw:
                span.set(items=0, reason="empty-response", model=self.model)
                return []
            items = _parse_watch_items(raw)
            span.set(
                items=len(items),
                model=self.model,
                lede_seeded=bool(lede_intro),
                clusters_seeded=bool(clusters),
                clusters_supplied=len(clusters) if clusters else 0,
            )
            return items

    def _build_user_prompt(
        self,
        sections: list[SectionDraft],
        *,
        lede_intro: str | None = None,
        clusters: list[Cluster] | None = None,
    ) -> str:
        lines = [
            "<task>",
            f"Read the {len(sections)} section drafts below and surface 2-3 cross-cutting",
            "watch items. Each item must draw from at least two clusters (story units; see",
            "the <clusters> index when present) or combine multiple concrete facts from",
            "within one section that the section itself does not connect.",
            "</task>",
            "",
        ]
        if lede_intro and lede_intro.strip():
            lines.extend(
                [
                    "<lede-already-framed>",
                    "The digest's Overview paragraph (already written, do NOT restate) frames",
                    "the week as follows. Pick watch items that surface different threads from",
                    "the sections — not a rephrase of the same thesis.",
                    "",
                    lede_intro.strip(),
                    "</lede-already-framed>",
                    "",
                ]
            )
        if clusters:
            lines.extend(_format_cluster_index(clusters, self._categories))
            lines.append("")
        lines.append("<sections>")
        for section in sections:
            lines.append("<section>")
            lines.append(f"  <name>{section.name}</name>")
            lines.append(f"  <article-count>{section.article_count}</article-count>")
            lines.append("  <content>")
            lines.append(section.content)
            lines.append("  </content>")
            lines.append("</section>")
        lines.append("</sections>")
        return "\n".join(lines)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        options = resolve_ollama_options(
            self._sampling,
            temperature=settings.watcher_temperature,
            top_p=settings.watcher_top_p,
            repetition_penalty=settings.watcher_repetition_penalty,
            seed=settings.watcher_seed,
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


def _format_cluster_index(clusters: list[Cluster], categories: CategorySet) -> list[str]:
    """Render the ``<clusters>`` block listing every story cluster for the watcher.

    Each cluster appears as one element: ``id`` and ``primary`` (display
    name) as attributes, ``secondaries`` attribute when non-empty, lede as
    element text. Sorted so the highest-signal clusters appear first:
    multi-article clusters, then clusters with cross-section references,
    then size-1 clusters. The watcher uses this index to pick tensions
    that span ≥2 clusters rather than ≥2 topic sections; size-1 clusters
    remain valid material when their facts connect across other clusters.

    Cluster ledes are the same strings cited in the writer's
    ``<covered-elsewhere>`` blocks. The watcher must NOT reuse a single
    lede as a watch-item heading, since the cluster already gets its writeup
    in the primary section. The signal value here is *combinations*.
    """

    def signal_key(c: Cluster) -> tuple[int, int]:
        # Sort key: (multi-article? 0 first, else 1, then secondary count desc).
        size_bucket = 0 if len(c.article_urls) > 1 else 1
        return (size_bucket, -len(c.secondary_sections))

    ordered = sorted(clusters, key=signal_key)
    lines = ["<clusters>"]
    for c in ordered:
        if not c.lede or not c.primary_section:
            continue
        primary_display = categories.title(c.primary_section)
        secondaries = ",".join(categories.title(s) for s in c.secondary_sections)
        attr = f'id="{c.id}" primary="{primary_display}"'
        if secondaries:
            attr += f' secondaries="{secondaries}"'
        lines.append(f"  <cluster {attr}>{c.lede}</cluster>")
    lines.append("</clusters>")
    return lines


def _parse_watch_items(raw: str) -> list[WatchItem]:
    """Parse watch items from the model's JSON array response.

    Strips an optional ```json code fence, then loads the JSON. Skips items
    missing either field. Returns an empty list on malformed JSON.
    """
    text = raw.strip()
    fence_match = _JSON_FENCE_RE.match(text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Watcher returned non-JSON output: {}", e)
        return []

    if not isinstance(data, list):
        logger.warning("Watcher returned non-array JSON: {}", type(data).__name__)
        return []

    items: list[WatchItem] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        heading = normalize_typography(str(entry.get("heading", "")).strip())
        body = normalize_typography(str(entry.get("body", "")).strip())
        if not heading or not body:
            continue
        items.append(WatchItem(heading=heading, body=body))
    return items
