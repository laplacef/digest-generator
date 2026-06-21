"""Mechanical digest assembly for the weekly digest pipeline.

``DigestComposer`` concatenates cleaned section drafts, the framing (title
and intro), and the watch items into the final Markdown digest. No LLM
call: pure string assembly at H2 boundaries.
"""

from datetime import UTC, datetime

from digest_generator.core.digest.types import DigestFraming, DigestResult, SectionDraft, WatchItem
from digest_generator.shared.logging import log_stage

# Calibrated against Piper amy-medium narration: a 2177-word digest renders to
# ~18.6 min of audio (~117 md-wpm). Rounded to 120 so the stated time tracks
# the audio rendition rather than silent-reading speed.
_WORDS_PER_MINUTE = 120
_OVERVIEW_HEADING = "## Overview"
_WATCH_HEADING = "## What to Watch"


class DigestComposer:
    """Assembles the final digest Markdown from cleaned pipeline outputs."""

    def compose(
        self,
        sections: list[SectionDraft],
        framing: DigestFraming,
        watch: list[WatchItem],
        *,
        date_range: tuple[str, str] | None = None,
    ) -> DigestResult:
        """Concatenate sections + framing + watch into the final digest.

        The returned ``DigestResult.content`` starts with a ``## Overview``
        heading and the lede paragraph, then each section draft in the
        provided order (the outer pipeline supplies them in ``ContentType``
        enum order), then the ``## What to Watch`` H2 section. The title is
        carried in ``DigestResult.title`` and rendered into the YAML
        frontmatter by ``io.build_digest_markdown``. The title is never
        emitted as an H1 in the body, since every consumer renders it from
        frontmatter.

        The ``## Overview`` heading gives the lede its own anchor so the
        newsletter's section-pill navigation can jump to it; the heading is
        skipped when no intro paragraph was produced.

        If ``sections`` is empty, returns a ``DigestResult`` with empty
        content so callers can short-circuit without raising.
        """
        with log_stage("composer") as span:
            digest_date = _resolve_date(date_range)
            if not sections:
                span.set(sections=0, words=0, reason="empty-sections")
                return DigestResult(
                    title=framing.title,
                    content="",
                    date=digest_date,
                    word_count=0,
                    reading_time_minutes=0,
                    article_count=0,
                    section_counts={},
                )

            parts: list[str] = []
            if framing.intro:
                parts.append(_OVERVIEW_HEADING)
                parts.append(framing.intro)
            for section in sections:
                parts.append(section.content.strip())
            watch_block = _render_watch(watch)
            if watch_block:
                parts.append(watch_block)

            content = "\n\n".join(parts).strip() + "\n"
            section_counts = {s.name: s.article_count for s in sections}
            total_articles = sum(section_counts.values())
            word_count = len(content.split())
            span.set(
                sections=len(sections),
                articles=total_articles,
                watch_items=len(watch),
                words=word_count,
            )
            return DigestResult(
                title=framing.title,
                content=content,
                date=digest_date,
                word_count=word_count,
                reading_time_minutes=max(1, word_count // _WORDS_PER_MINUTE),
                article_count=total_articles,
                section_counts=section_counts,
            )


def _resolve_date(date_range: tuple[str, str] | None) -> str:
    if date_range:
        return date_range[1]
    return datetime.now(tz=UTC).strftime("%Y-%m-%d")


def _render_watch(items: list[WatchItem]) -> str:
    """Render watch items as a ``## What to Watch`` section with H3 headings."""
    if not items:
        return ""
    lines = [_WATCH_HEADING]
    for item in items:
        lines.append("")
        lines.append(f"### {item.heading}")
        lines.append("")
        lines.append(item.body)
    return "\n".join(lines)
