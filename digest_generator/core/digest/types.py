"""Digest-pipeline dataclasses produced and consumed by the digest stages.

These types flow between the writer, editor, framer, watcher, and composer
stages, and out to the API layer for persistence. Cross-source types
(``Label``, ``TopicType``, ``Filter``) live in ``digest_generator.core.types``.
Shared LLM infrastructure (``SamplingConfig``, ``resolve_ollama_options``)
lives in ``digest_generator.shared.llm.sampling``. Pipeline run metadata
(``RunMeta``) lives in ``digest_generator.shared.runtime.meta``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DigestResult:
    """Structured output from digest generation.

    Separates content from metadata so callers can build frontmatter,
    choose filenames, and populate ``meta.json`` without parsing markdown.
    """

    title: str
    content: str
    date: str
    word_count: int
    reading_time_minutes: int
    article_count: int
    section_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class SectionDraft:
    """A single per-section prose draft produced by ``SectionWriter``.

    Passed between pipeline stages (writer, then editor, then composer). ``name`` is
    the section display title (a category's ``title``).
    """

    name: str
    content: str
    article_count: int


@dataclass
class DigestFraming:
    """Title and intro lede produced by ``DigestFramer`` for a digest."""

    title: str
    intro: str


@dataclass
class WatchItem:
    """A single cross-cutting watch item produced by ``WhatToWatch``."""

    heading: str
    body: str


@dataclass
class Cluster:
    """One story cluster produced by ``ArticleClusterer``.

    A cluster groups articles describing the same story (often across
    multiple publishers) and records the editorial routing: which
    category section owns the full writeup (``primary_section``)
    and which sections should reference rather than expand it
    (``secondary_sections``). Members are URLs because URL is the
    cross-stage identifier already used to join ``source-summarized/``
    with ``source-labeled/`` at digest time.

    ``id`` is a stable in-run handle (e.g. ``c0042``) used by the LLM
    prompt to reference clusters in its JSON output without echoing long
    URLs. ``lede`` is a one-line description the writer can cite
    verbatim in ``<covered-elsewhere>`` cross-references so it doesn't
    re-summarize on the secondary side. ``entities`` are 2-4 short
    identifying names or numbers (company / product / CVE / dollar
    amount) the writer can drop into a one-sentence cross-reference
    without re-extracting them from article content.

    ``primary_section`` and ``secondary_sections`` carry category ids as
    strings, validated against the configured category set. Size-1 clusters
    with no secondaries are the no-op shape: equivalent to "no clustering
    applied to this article."
    """

    id: str
    lede: str
    article_urls: list[str]
    primary_section: str
    secondary_sections: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
