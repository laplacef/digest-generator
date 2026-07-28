"""Tests for the importable golden-output lint."""

from __future__ import annotations

from digest_generator.core.digest.lint import (
    ERROR,
    WARNING,
    canonical_url,
    format_findings,
    has_errors,
    lint_digest,
)

KNOWN_URLS = frozenset(
    canonical_url(u) for u in ("https://x.io/r", "https://x.io/p", "https://x.io/f")
)

CLEAN_DIGEST = """---
title: "Agents Collide with Identity as Budgets Tighten"
slug: "2026-07-22"
date: 2026-07-22
summary: "Enterprises moved agents into production while identity layers lagged."
---

## Overview

Adoption accelerated, and [one report](https://x.io/r) framed the shift.

## Infrastructure

Providers shipped updates, per [a post](https://x.io/p).

Costs rose to $9.36 billion, up 37% quarter over quarter, per [filing](https://x.io/f).

## What to Watch

### Identity

Whether identity layers keep pace is the open question.
"""


def _categories(markdown: str) -> set[str]:
    return {f.category for f in lint_digest(markdown)}


class TestCleanDigest:
    def test_no_findings(self) -> None:
        findings = lint_digest(CLEAN_DIGEST)
        assert findings == [], format_findings(findings)

    def test_no_errors_helper(self) -> None:
        assert has_errors(lint_digest(CLEAN_DIGEST)) is False


class TestLinkTargets:
    """Both defects here shipped in a real digest and survived every LLM stage."""

    def test_clean_digest_has_no_link_findings(self) -> None:
        findings = lint_digest(CLEAN_DIGEST, known_urls=KNOWN_URLS)
        assert findings == [], format_findings(findings)

    def test_unquoted_whitespace_in_target_is_an_error(self) -> None:
        md = CLEAN_DIGEST.replace("https://x.io/p", "https://x.io/p Newly")
        findings = [f for f in lint_digest(md) if f.category == "link-target-malformed"]
        assert len(findings) == 1
        assert findings[0].severity == ERROR

    def test_malformed_target_needs_no_corpus(self) -> None:
        """The whitespace case is wrong on its own terms, so it gates without one."""
        md = CLEAN_DIGEST.replace("https://x.io/p", "https://x.io/p Newly")
        assert has_errors(lint_digest(md, known_urls=None)) is True

    def test_commonmark_quoted_title_is_well_formed(self) -> None:
        md = CLEAN_DIGEST.replace("(https://x.io/p)", '(https://x.io/p "A Title")')
        findings = [f for f in lint_digest(md) if f.category == "link-target-malformed"]
        assert findings == []

    def test_mutated_slug_is_caught_against_the_corpus(self) -> None:
        md = CLEAN_DIGEST.replace("https://x.io/p", "https://x.io/p-for-some-coders")
        findings = [
            f for f in lint_digest(md, known_urls=KNOWN_URLS) if f.category == "link-target-unknown"
        ]
        assert len(findings) == 1
        assert findings[0].severity == ERROR

    def test_no_corpus_means_no_membership_check(self) -> None:
        md = CLEAN_DIGEST.replace("https://x.io/p", "https://x.io/totally-invented")
        assert [f for f in lint_digest(md) if f.category == "link-target-unknown"] == []

    def test_trailing_slash_and_fragment_do_not_false_positive(self) -> None:
        md = CLEAN_DIGEST.replace("https://x.io/p", "https://X.io/p/#section")
        findings = [
            f for f in lint_digest(md, known_urls=KNOWN_URLS) if f.category == "link-target-unknown"
        ]
        assert findings == []

    def test_relative_targets_are_not_membership_checked(self) -> None:
        md = CLEAN_DIGEST.replace("https://x.io/p", "/local/page")
        findings = [
            f for f in lint_digest(md, known_urls=KNOWN_URLS) if f.category == "link-target-unknown"
        ]
        assert findings == []


class TestCanonicalUrl:
    def test_folds_host_case_trailing_slash_and_fragment(self) -> None:
        assert canonical_url("HTTPS://Example.COM/Path/") == "https://example.com/Path"
        assert canonical_url("https://example.com/p#frag") == "https://example.com/p"

    def test_folds_www_prefix(self) -> None:
        assert canonical_url("https://www.x.io/a") == canonical_url("https://x.io/a")

    def test_drops_feed_tracking_parameters(self) -> None:
        """RSS tags the canonical link, so the corpus and a clean citation must match."""
        tagged = "https://x.io/a/?utm_source=rss&utm_medium=rss&utm_campaign=a"
        assert canonical_url(tagged) == canonical_url("https://x.io/a/")

    def test_keeps_content_selecting_parameters(self) -> None:
        assert canonical_url("https://x.io/a?id=7&utm_source=rss") == "https://x.io/a?id=7"

    def test_path_case_is_significant(self) -> None:
        """A changed path is the defect being detected, so it must not fold."""
        assert canonical_url("https://x.io/Slug") != canonical_url("https://x.io/slug")


class TestStructuralErrors:
    def test_title_shape_flagged(self) -> None:
        md = CLEAN_DIGEST.replace(
            "Agents Collide with Identity as Budgets Tighten",
            "AI Agents Reshape Everything",
        )
        assert "title-shape" in _categories(md)

    def test_h1_in_body_flagged(self) -> None:
        md = CLEAN_DIGEST.replace("## Overview", "# A Title\n\n## Overview")
        assert "h1-in-body" in _categories(md)

    def test_citation_key_leak_flagged(self) -> None:
        md = CLEAN_DIGEST.replace("the shift.", "the shift (c0108).")
        cats = _categories(md)
        assert "citation-key-leak" in cats

    def test_per_section_watch_flagged(self) -> None:
        md = CLEAN_DIGEST.replace(
            "## Infrastructure\n",
            "## Infrastructure\n\n### What to Watch\n\nA forecast here.\n",
        )
        assert "per-section-watch" in _categories(md)

    def test_terminal_watch_not_flagged(self) -> None:
        assert "per-section-watch" not in _categories(CLEAN_DIGEST)


class TestMechanicalResidue:
    def test_percent_space_flagged(self) -> None:
        md = CLEAN_DIGEST.replace("up 37% quarter", "up 37 % quarter")
        assert "residue-percent-with-space" in _categories(md)

    def test_scale_abbreviation_flagged(self) -> None:
        md = CLEAN_DIGEST.replace("$9.36 billion", "$9.36bn")
        assert "residue-scale-abbreviation" in _categories(md)

    def test_residue_inside_link_target_not_flagged(self) -> None:
        # A "37%"-shaped fragment inside a URL must not trip the residue scan.
        md = CLEAN_DIGEST.replace("[a post](https://x.io/p)", "[a post](https://x.io/p?q=37%20now)")
        assert not any(c.startswith("residue-") for c in _categories(md))


class TestCitationFloor:
    def test_multi_paragraph_section_without_links_warns(self) -> None:
        md = CLEAN_DIGEST.replace(
            "Providers shipped updates, per [a post](https://x.io/p).",
            "Providers shipped updates this quarter.",
        ).replace(
            "Costs rose to $9.36 billion, up 37% quarter over quarter, per [filing](https://x.io/f).",
            "Costs rose sharply over the quarter.",
        )
        findings = [f for f in lint_digest(md) if f.category == "section-no-citations"]
        assert findings
        assert findings[0].severity == WARNING

    def test_overview_exempt_from_floor(self) -> None:
        # Overview is framing, not an article section; no citation floor.
        assert "section-no-citations" not in _categories(CLEAN_DIGEST)


class TestPhraseChecks:
    def test_hollow_opener_flagged(self) -> None:
        md = CLEAN_DIGEST.replace("Adoption accelerated,", "This week saw adoption accelerate,")
        cats = _categories(md)
        assert "phrase-hollow_week_openers" in cats

    def test_phrases_are_warnings(self) -> None:
        md = CLEAN_DIGEST.replace(
            "Whether identity layers keep pace is the open question.",
            "The race is on for orchestration.",
        )
        findings = [f for f in lint_digest(md) if f.category.startswith("phrase-")]
        assert findings
        assert all(f.severity == WARNING for f in findings)


class TestSeverityGating:
    def test_has_errors_true_on_error(self) -> None:
        md = CLEAN_DIGEST.replace("the shift.", "the shift (c0108).")
        assert has_errors(lint_digest(md)) is True

    def test_findings_sorted_by_line(self) -> None:
        md = CLEAN_DIGEST.replace("the shift.", "the shift (c0108).").replace(
            "$9.36 billion", "$9.36bn"
        )
        lines = [f.line for f in lint_digest(md)]
        assert lines == sorted(lines)

    def test_error_and_warning_constants(self) -> None:
        assert ERROR == "error"
        assert WARNING == "warning"
