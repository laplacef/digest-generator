"""Digest-specific filesystem I/O: filenames, frontmatter, and stage caches.

These helpers sit next to the digest stages because they only exist to
persist and re-read digest artifacts. ``slugify_title`` lives here too;
its only consumer is digest filename construction, so it sits with the
digest io it serves. Run-directory creation (``create_run_dir``)
lives in ``digest_generator.shared.runtime.dirs``; pipeline run metadata (``RunMeta``,
``write_run_meta``, ``update_run_meta_digest``) lives in
``digest_generator.shared.runtime.meta``.
"""

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from digest_generator.core.digest.types import DigestResult, SectionDraft
from digest_generator.shared.logging import logger

_SUMMARY_SOFT_CAP = 200
_SUMMARY_HARD_CAP = 155
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")

# Invisible Unicode whitespace LLMs sometimes emit as unit separators
# (e.g. "10,000<U+202F>RPS") or thousand separators. Normalized to ASCII
# space at write time so downstream consumers (markdown renderers, the
# audio narrator, search-engine snippets) don't trip over non-ASCII
# whitespace that looks identical to a regular space.
# - U+00A0 NO-BREAK SPACE
# - U+2009 THIN SPACE
# - U+202F NARROW NO-BREAK SPACE
_INVISIBLE_WS_RE = re.compile(r"[\u00A0\u2009\u202F]")


def _summary_for_frontmatter(text: str) -> str:
    """Pick a meta-description summary that ends on a sentence boundary.

    The frontmatter ``summary`` field renders as the search-engine snippet and
    social-card description. Cutting mid-clause produces half-sentences on link
    previews ("…contributing to a drop in the average exploit lifecycle from…"),
    so the summary ends on a real sentence terminator.

    Walks sentence boundaries within the lede and packs as many full sentences
    as fit under ``_SUMMARY_SOFT_CAP``. If the first sentence alone exceeds the
    soft cap (or the lede has no terminator at all), falls back to
    word-boundary truncation at ``_SUMMARY_HARD_CAP`` with an ellipsis.
    """
    text = text.strip()
    if not text:
        return ""
    boundaries = [m.end() for m in _SENTENCE_END_RE.finditer(text)]
    if not boundaries:
        return text[:_SUMMARY_HARD_CAP].rsplit(" ", 1)[0] + "..."
    chosen_end = boundaries[0]
    for end in boundaries[1:]:
        if end > _SUMMARY_SOFT_CAP:
            break
        chosen_end = end
    candidate = text[:chosen_end]
    if len(candidate) <= _SUMMARY_SOFT_CAP:
        return candidate
    return text[:_SUMMARY_HARD_CAP].rsplit(" ", 1)[0] + "..."


def slugify_title(title: str, max_length: int = 60) -> str:
    """Convert a title string into a URL/filename-safe slug.

    Args:
        title: Human-readable title text.
        max_length: Maximum slug length (truncates at word boundary).

    Returns:
        Lowercase hyphen-separated slug (e.g., ``"ai-agents-reshape-workflows"``).
    """
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")

    if len(slug) <= max_length:
        return slug

    # Truncate at word boundary
    truncated = slug[:max_length]
    last_hyphen = truncated.rfind("-")
    if last_hyphen > 0:
        truncated = truncated[:last_hyphen]
    return truncated


def build_digest_slug(result: DigestResult) -> str:
    """Build the stable, generic URL slug for a digest.

    Returns the issue date (``YYYY-MM-DD``), emitted into the frontmatter
    ``slug`` field. The slug is deliberately **title-independent**: it is
    the digest's permanent identity as a dated issue, so editing the
    ``title`` later changes the displayed name without moving the published
    URL. Downstream consumers key the public URL off this field, never off
    the title. It matches the deliverable's filename stem (see
    ``build_digest_filename``).

    One digest per date is assumed (the publish cadence is weekly); the
    run *directory* name carries the collision-proof uniqueness for the
    working artifact. A title cannot influence the slug, so adversarial
    titles can't reach any URL or path built from it.
    """
    return result.date


def build_digest_filename(result: DigestResult) -> str:
    """Build the stable on-disk filename for the digest deliverable.

    Format: ``{date}.md`` (e.g. ``2026-03-17.md``) — the issue date, which
    is stable, decoupled from the title, and matches the ``slug`` / public
    URL / opus stem so every published artifact keys on the same date
    string. Regenerating a digest in the same ``run_dir`` produces the same
    date and overwrites this one file rather than spawning a second ``.md``
    (which would make ``audio.io.find_digest_md`` ambiguous).

    The run *directory* (``{date}-{HHmmss}-{hex}``) supplies collision-proof
    uniqueness for the working artifact, so the deliverable inside it needs
    only the date. The opus rendition shares this stem via
    ``audio.io.opus_path_for_digest``, keeping both artifacts on one name.
    """
    return f"{result.date}.md"


def build_digest_markdown(result: DigestResult) -> str:
    """Prepend YAML frontmatter to digest content for static site consumption.

    Frontmatter fields, in order: title, slug, date, reading_time,
    article_count, summary, sections. ``title`` is the human-facing article
    name and may be edited post-generation; ``slug`` is the stable, generic
    URL slug (the issue date — see ``build_digest_slug``) the static site
    keys the published URL off, decoupled from both the title and the
    on-disk filename so a title edit never moves the URL. ``summary`` is
    extracted from the first non-heading paragraph of the composed content via
    ``_summary_for_frontmatter``, which cuts on a real sentence boundary when
    one fits under the soft cap.

    The final string is normalized to ASCII whitespace via ``_INVISIBLE_WS_RE``
    so invisible Unicode separators emitted by upstream LLMs (U+00A0 nbsp,
    U+2009 thin space, U+202F narrow no-break space, which appear in digest
    output as ``10,000<U+202F>RPS``) don't reach downstream consumers.
    """
    escaped_title = result.title.replace('"', '\\"')

    # ``slug`` MUST be quoted. Unquoted ``slug: 2026-03-17`` is parsed by YAML
    # as a Date, and Jekyll's UrlDrop#title calls ``.gsub`` on it while building
    # the permalink, which raises on a non-string. ``date`` below is left
    # unquoted on purpose: Jekyll expects a real Date there.
    lines = [
        "---",
        f'title: "{escaped_title}"',
        f'slug: "{build_digest_slug(result)}"',
        f"date: {result.date}",
        f"reading_time: {result.reading_time_minutes} min",
        f"article_count: {result.article_count}",
    ]

    summary = ""
    for block in result.content.strip().split("\n\n"):
        text = block.strip()
        if text and not text.startswith("#"):
            summary = _summary_for_frontmatter(text).replace('"', '\\"')
            break
    if summary:
        lines.append(f'summary: "{summary}"')

    if result.section_counts:
        lines.append("sections:")
        for section, count in result.section_counts.items():
            escaped_section = section.replace('"', '\\"')
            lines.append(f'  "{escaped_section}": {count}')

    lines.append("---")
    lines.append("")

    return _INVISIBLE_WS_RE.sub(" ", "\n".join(lines) + result.content)


def save_section_drafts(drafts: list[SectionDraft], target_dir: Path) -> None:
    """Persist section drafts as ``<slug>.md`` files for per-stage caching.

    Each file prepends an HTML comment with the article count so the draft
    round-trips with enough metadata to rebuild the ``SectionDraft``. The H2
    heading in the prose carries the section name.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    for draft in drafts:
        slug = slugify_title(draft.name, max_length=40)
        path = target_dir / f"{slug}.md"
        header = f"<!-- articles: {draft.article_count} -->\n"
        path.write_text(header + draft.content, encoding="utf-8")


def load_section_drafts(source_dir: Path) -> list[SectionDraft]:
    """Load section drafts from ``<slug>.md`` files written by ``save_section_drafts``.

    Returns an empty list if the directory does not exist. Files missing an
    article-count header or an H2 heading are skipped with a warning.
    """
    if not source_dir.exists():
        return []

    drafts: list[SectionDraft] = []
    for path in sorted(source_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        match = re.match(r"<!-- articles: (\d+) -->\n(.*)", raw, re.DOTALL)
        if not match:
            logger.warning("Cached section {} missing article-count header; skipping", path.name)
            continue
        article_count = int(match.group(1))
        content = match.group(2)
        name = _extract_section_name(content)
        if not name:
            logger.warning("Cached section {} missing H2 heading; skipping", path.name)
            continue
        drafts.append(SectionDraft(name=name, content=content, article_count=article_count))
    return drafts


def _extract_section_name(content: str) -> str:
    """Return the first H2 heading text from markdown content, or empty string."""
    for line in content.lstrip().splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            return stripped[3:].strip()
        if stripped:
            return ""
    return ""


def save_json_stage(data: Any, path: Path) -> None:
    """Persist a dataclass (or list of dataclasses) to ``path`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in data] if isinstance(data, list) else asdict(data)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json_stage[T](path: Path, cls: type[T]) -> T | list[T] | None:
    """Load JSON from ``path`` and instantiate ``cls`` (or a list of ``cls``).

    Returns ``None`` if the file does not exist.
    """
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [cls(**item) for item in data]
    return cls(**data)
