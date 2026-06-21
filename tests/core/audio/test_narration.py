"""Tests for digest_generator/core/audio/narration.py: markdown to speech-friendly script.

The narration pre-pass uses Piper-native pause cues (newlines plus sentence
punctuation) since Piper does not parse SSML.
"""

from textwrap import dedent

import pytest

from digest_generator.core.audio.narration import (
    NARRATION_VERSION,
    load_overrides,
    markdown_to_narration,
)


class TestVersionConstant:
    """NARRATION_VERSION participates in the audio cache key."""

    def test_version_is_v3(self):
        # The narration shape uses newlines plus punctuation, pre-walk
        # normalizers, and em-dash paragraph padding. Bumping the narration
        # shape requires bumping this constant so cached renders invalidate.
        assert NARRATION_VERSION == "v3"


class TestLoadOverrides:
    """The bundled overrides file ships with the package."""

    def test_default_path_returns_dict(self):
        overrides = load_overrides()
        # The bundled file ships with at least the seed acronyms.
        assert isinstance(overrides, dict)
        assert "AGI" in overrides
        assert overrides["AGI"] == "A G I"

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_overrides(tmp_path / "nope.yaml") == {}

    def test_custom_path(self, tmp_path):
        path = tmp_path / "custom.yaml"
        path.write_text("FOO: F O O\nBAR: B A R\n")
        result = load_overrides(path)
        assert result == {"FOO": "F O O", "BAR": "B A R"}

    def test_rejects_non_mapping(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- not\n- a\n- mapping\n")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_overrides(path)


class TestFrontmatter:
    """YAML frontmatter is stripped before the AST walk."""

    def test_strips_top_of_file_frontmatter(self):
        md = dedent("""\
            ---
            title: foo
            date: 2026-05-11
            ---

            Body sentence.
            """)
        out = markdown_to_narration(md, overrides={})
        assert "title:" not in out
        assert "Body sentence." in out

    def test_no_frontmatter_passes_through(self):
        out = markdown_to_narration("Just a paragraph.", overrides={})
        assert "Just a paragraph." in out


class TestNoSsmlInOutput:
    """The output never contains SSML, which Piper would speak literally."""

    def test_no_break_tags(self):
        md = dedent("""\
            # Title

            Para.

            ## Section

            - item one
            - item two

            ```python
            code
            ```

            More.
            """)
        out = markdown_to_narration(md, overrides={})
        # The SSML <break time="..."/> pattern must never appear in output.
        assert "<break" not in out
        assert 'ms"' not in out
        assert "ms/>" not in out


class TestHeadings:
    """Headings end with sentence punctuation and get paragraph breaks for pacing."""

    def test_h1_emits_sentence_terminator_and_break(self):
        out = markdown_to_narration("# Title\n", overrides={})
        assert "Title." in out
        # Paragraph break after the title (Piper interprets \n\n as a strong pause).
        # When the title is the only content, the trailing break gets stripped, so
        # this test pins the in-string emission via the round-trip integration test.

    def test_h2_emits_sentence_terminator(self):
        out = markdown_to_narration("## Subtitle\n", overrides={})
        assert "Subtitle." in out

    def test_h3_emits_sentence_terminator(self):
        out = markdown_to_narration("### Tertiary\n", overrides={})
        assert "Tertiary." in out

    def test_heading_keeps_existing_punctuation(self):
        out = markdown_to_narration("## What now?\n", overrides={})
        assert "What now?" in out
        # No double-punctuation.
        assert "What now?." not in out

    def test_heading_followed_by_paragraph_separated_by_blank_line(self):
        """Heading + body should be on separate paragraphs (blank line between)."""
        md = "## Section\n\nBody.\n"
        out = markdown_to_narration(md, overrides={})
        assert "Section." in out
        assert "Body." in out
        # Empty line between heading and body in the output.
        assert "Section.\n\nBody." in out


class TestLinks:
    """Links keep their text and drop the URL."""

    def test_link_text_preserved(self):
        out = markdown_to_narration("See [the docs](https://example.com) here.", overrides={})
        assert "the docs" in out
        assert "https://" not in out
        assert "example.com" not in out
        assert "(" not in out
        assert "[" not in out

    def test_bare_url_in_prose_dropped(self):
        out = markdown_to_narration("Visit https://example.com today.", overrides={})
        assert "https://" not in out
        assert "Visit" in out
        assert "today" in out


class TestEmphasis:
    """Bold / italic / strikethrough markers stripped, content kept."""

    def test_bold_stripped(self):
        out = markdown_to_narration("This is **important** stuff.", overrides={})
        assert "important" in out
        assert "**" not in out

    def test_italic_stripped(self):
        out = markdown_to_narration("This is *subtle* stuff.", overrides={})
        assert "subtle" in out
        assert "*subtle*" not in out

    def test_inline_code_kept_no_backticks(self):
        out = markdown_to_narration("Run `pytest` to test.", overrides={})
        assert "pytest" in out
        assert "`" not in out


class TestCodeBlocks:
    """Fenced and indented code blocks are replaced with a paragraph break."""

    def test_fenced_code_dropped(self):
        md = dedent("""\
            Before code.

            ```python
            print("hi")
            secret = "xyz"
            ```

            After code.
            """)
        out = markdown_to_narration(md, overrides={})
        assert "print" not in out
        assert "secret" not in out
        assert "Before code." in out
        assert "After code." in out
        # The fenced-code elision becomes a paragraph break between surrounding text;
        # the preceding paragraph's em-dash padding lands in that gap.
        assert "Before code.\n\n—.\n\nAfter code." in out

    def test_indented_code_dropped(self):
        md = "Para.\n\n    indented code line\n\nAfter.\n"
        out = markdown_to_narration(md, overrides={})
        assert "indented code line" not in out
        assert "Para." in out
        assert "After." in out


class TestLists:
    """Bullet and ordered lists narrate as individual sentences (newline-separated)."""

    def test_bullet_list_items_become_sentences(self):
        md = dedent("""\
            - first
            - second
            - third
            """)
        out = markdown_to_narration(md, overrides={})
        assert "first." in out
        assert "second." in out
        assert "third." in out
        # Each item terminated as its own sentence so Piper inflects on each.
        # Items are separated by a single newline (a short beat), not a paragraph
        # break (which would create a long pause between every item).
        assert "first.\nsecond.\nthird." in out

    def test_ordered_list_same_shape(self):
        md = dedent("""\
            1. one
            2. two
            """)
        out = markdown_to_narration(md, overrides={})
        assert "one." in out
        assert "two." in out
        assert "one.\ntwo." in out


class TestParagraphs:
    """Consecutive paragraphs are separated by a paragraph break plus an em-dash empty sentence."""

    def test_paragraph_break_between_paragraphs(self):
        # Paragraph boundaries get an em-dash empty sentence between
        # them so Piper applies --sentence-silence twice (a longer pause
        # than between sentences within the same paragraph).
        md = "First para.\n\nSecond para.\n"
        out = markdown_to_narration(md, overrides={})
        assert "First para.\n\n—.\n\nSecond para." in out

    def test_em_dash_padding_not_appended_at_document_end(self):
        # The trailing padding has no successor paragraph; trim it so the
        # audio doesn't end on a stray em-dash sentence.
        out = markdown_to_narration("Only paragraph.\n", overrides={})
        assert out.rstrip().endswith("Only paragraph.")
        assert not out.rstrip().endswith("—.")

    def test_em_dash_padding_not_between_list_items(self):
        # List items use a single-newline beat; em-dash padding is only
        # for paragraph-to-paragraph boundaries.
        md = "- first\n- second\n- third\n"
        out = markdown_to_narration(md, overrides={})
        assert "first.\nsecond.\nthird." in out
        assert "—" not in out


class TestPreWalkNormalizers:
    """Pre-walk text normalizers run on raw markdown before parsing."""

    def test_currency_dollars_reorder_with_magnitude(self):
        out = markdown_to_narration("Raised $700 million today.", overrides={})
        assert "700 million dollars" in out
        assert "$" not in out

    def test_currency_dollars_bare_amount(self):
        out = markdown_to_narration("Cost $50.", overrides={})
        assert "50 dollars" in out

    def test_currency_dollars_with_decimal_and_magnitude(self):
        out = markdown_to_narration("A $1.5 billion deal.", overrides={})
        assert "1.5 billion dollars" in out

    def test_nx_multiplier_expanded(self):
        out = markdown_to_narration("9.2x greater throughput.", overrides={})
        assert "9.2 times greater" in out
        assert "9.2x" not in out

    def test_nx_does_not_match_xlarge(self):
        # The word-boundary right side blocks matches like "xlarge".
        out = markdown_to_narration("Use db.r6g.4xlarge nodes.", overrides={})
        assert "xlarge" in out
        assert "4 times" not in out

    def test_unit_compact_form_expanded(self):
        out = markdown_to_narration("10GW capacity online.", overrides={})
        assert "10 gigawatts" in out
        # Bare "GW" no longer present after expansion.
        assert " GW " not in out
        assert "10GW" not in out

    def test_unit_spaced_form_expanded(self):
        out = markdown_to_narration("800 MW plant came online.", overrides={})
        assert "800 megawatts" in out

    def test_unit_tbps_expanded(self):
        out = markdown_to_narration("A 3.5 Tbps attack.", overrides={})
        assert "3.5 terabits per second" in out
        assert "Tbps" not in out

    def test_unit_decimal_gigabytes(self):
        out = markdown_to_narration("Allocate 1.5GB of RAM.", overrides={})
        assert "1.5 gigabytes" in out

    def test_dotted_us_normalized(self):
        out = markdown_to_narration("The U.S. Department of Defense.", overrides={})
        assert "U.S." not in out
        assert "US Department of Defense" in out

    def test_dotted_uk_normalized(self):
        out = markdown_to_narration("The U.K. responded today.", overrides={})
        assert "U.K." not in out
        assert "UK responded" in out

    def test_dotted_dc_normalized(self):
        out = markdown_to_narration("Based in D.C. for years.", overrides={})
        assert "D.C." not in out
        assert "DC for years" in out

    def test_tilde_as_approximate(self):
        out = markdown_to_narration("Roughly ~35,000 lines per second.", overrides={})
        assert "approximately 35,000" in out
        assert "~35" not in out

    def test_tilde_without_digit_untouched(self):
        # Lookahead requires a digit after `~`, so tildes followed by a
        # non-digit (e.g. a home-dir path) don't trigger replacement.
        out = markdown_to_narration("Path ~/foo/bar.txt exists.", overrides={})
        assert "approximately" not in out

    def test_trailing_acronym_definition_dropped(self):
        out = markdown_to_narration(
            "Fine Grained Authorization (FGA) is the new model.",
            overrides={"FGA": "fine-grained authorization"},
        )
        # The "(FGA)" definition is stripped pre-walk so the override
        # never fires; the listener hears the expansion once, not twice.
        assert "(FGA)" not in out
        assert "(fine-grained authorization)" not in out
        assert "Fine Grained Authorization is the new model." in out

    def test_trailing_acronym_plural_dropped(self):
        out = markdown_to_narration(
            "Large Language Models (LLMs) are popular.",
            overrides={},
        )
        assert "(LLMs)" not in out
        assert "Large Language Models are popular." in out

    def test_trailing_acronym_single_letter_kept(self):
        # The pattern requires 2+ uppercase letters, so "(A)" survives.
        out = markdown_to_narration("See footnote (A) below.", overrides={})
        assert "(A)" in out


class TestHtml:
    """Inline and block HTML are stripped."""

    def test_inline_html_dropped(self):
        out = markdown_to_narration("Wrap <span>this</span> text.", overrides={})
        assert "<span>" not in out
        assert "</span>" not in out

    def test_block_html_dropped(self):
        md = dedent("""\
            Para before.

            <div class="note">Some block.</div>

            Para after.
            """)
        out = markdown_to_narration(md, overrides={})
        assert "<div" not in out
        assert "Para before." in out
        assert "Para after." in out


class TestOverrides:
    """Pronunciation overrides are whole-word, case-sensitive."""

    def test_default_overrides_applied(self):
        out = markdown_to_narration("Discuss LLMs and AGI now.")
        assert "L L Ms" in out
        assert "A G I" in out

    def test_custom_overrides_override_default(self):
        out = markdown_to_narration("Discuss LLMs and AGI now.", overrides={"AGI": "ay-gee-eye"})
        assert "ay-gee-eye" in out
        # Custom dict replaces defaults, so LLMs gets no override here.
        assert "LLMs" in out

    def test_overrides_disabled_with_empty_dict(self):
        out = markdown_to_narration("Discuss LLMs and AGI now.", overrides={})
        assert "LLMs" in out
        assert "AGI" in out

    def test_word_boundary_respected(self):
        # "API" should match, "RAPID" must not.
        out = markdown_to_narration("API but not RAPID.", overrides={"API": "ay-pee-eye"})
        assert "ay-pee-eye" in out
        assert "RAPID" in out

    def test_case_sensitive(self):
        out = markdown_to_narration("agi vs AGI.", overrides={"AGI": "X"})
        assert "agi" in out  # lowercase not overridden
        assert "X" in out


class TestIntegration:
    """End-to-end smoke against a digest-shaped input."""

    def test_full_digest_shape(self):
        md = dedent("""\
            ---
            title: Weekly Digest
            ---

            # Weekly AI Digest

            Opening lede with a [link](https://example.com).

            ## Models

            New GPU released. *Italic* and **bold** noted.

            - first item
            - second item

            ```bash
            echo "internal"
            ```

            Closing paragraph.
            """)
        out = markdown_to_narration(md)
        # Frontmatter gone, no SSML, structural shape correct.
        assert "title:" not in out
        assert "<break" not in out
        assert "Weekly AI Digest." in out
        assert "Models." in out
        assert "G P U" in out  # default override applied
        assert "first item." in out
        assert "second item." in out
        assert "first item.\nsecond item." in out  # list-item newlines
        assert "Closing paragraph." in out
        # No internal code echoed, no link URL leaked.
        assert "echo" not in out
        assert "https://" not in out
        assert "[link]" not in out


class TestWhitespace:
    """Newlines are preserved; horizontal whitespace and excess newlines collapse."""

    def test_horizontal_whitespace_collapsed(self):
        md = "Body with  multiple  internal  spaces.\n"
        out = markdown_to_narration(md, overrides={})
        assert "  " not in out

    def test_consecutive_newlines_capped_at_two(self):
        md = dedent("""\
            First.

            Second.



            Third.
            """)
        out = markdown_to_narration(md, overrides={})
        # Three-plus consecutive newlines collapse to a paragraph break (2).
        assert "\n\n\n" not in out
        # Em-dash empty-sentence padding sits between every paragraph pair.
        assert "First.\n\n—.\n\nSecond." in out
        assert "Second.\n\n—.\n\nThird." in out

    def test_no_leading_or_trailing_whitespace(self):
        out = markdown_to_narration("\n\n# Title\n\nBody.\n\n", overrides={})
        assert out == out.strip()
