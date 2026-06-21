"""Tests for digest_generator/core/style.py: the forbidden-phrase catalogue.

Covers:

- ``render_bullets`` Markdown formatting
- ``expand_style_placeholders`` substitution and unknown-category errors
- End-to-end resolution via the digest ``load_prompt`` for the three
  subscribing digest templates
- Sanity checks that representative tics from the 2026-04-26 run
  (``This week saw``, ``took center stage``, ``the landscape of``,
  ``critical``) are present in the catalogue
"""

import pytest

from digest_generator.core.digest.prompts import load_prompt
from digest_generator.core.style import (
    ABSTRACT_INFORMATION_VERBS,
    ABSTRACT_LANDSCAPE_OPENERS,
    ENUMERATION_OPENERS,
    FILLER_ADJECTIVES,
    GENERIC_TRANSITION_OPENERS,
    HOLLOW_WEEK_OPENERS,
    STAGE_DIRECTION_CLICHES,
    WATCH_WEAK_FORECASTS,
    ForbiddenPhrase,
    expand_style_placeholders,
    render_bullets,
)

# =============================================================================
# render_bullets
# =============================================================================


class TestRenderBullets:
    def test_formats_each_item_as_markdown_bullet(self):
        items = [ForbiddenPhrase("alpha"), ForbiddenPhrase("beta")]
        assert render_bullets(items) == "- alpha\n- beta"

    def test_empty_list_returns_empty_string(self):
        assert render_bullets([]) == ""

    def test_preserves_display_text_verbatim(self):
        items = [ForbiddenPhrase('"quoted, with comma"')]
        assert render_bullets(items) == '- "quoted, with comma"'


# =============================================================================
# expand_style_placeholders
# =============================================================================


class TestExpandStylePlaceholders:
    def test_substitutes_known_category(self):
        text = "Before\n{{style:generic_transition_openers}}\nAfter"
        result = expand_style_placeholders(text)
        assert '- "Meanwhile,"' in result
        assert '- "Concurrently,"' in result
        assert "{{style:" not in result

    def test_no_op_when_no_placeholders(self):
        text = "Plain text with no placeholders.\n"
        assert expand_style_placeholders(text) == text

    def test_unknown_category_raises_keyerror(self):
        with pytest.raises(KeyError, match="typo_category"):
            expand_style_placeholders("{{style:typo_category}}")

    def test_multiple_placeholders_resolve_independently(self):
        text = "{{style:hollow_week_openers}}\n---\n{{style:filler_adjectives}}"
        result = expand_style_placeholders(text)
        assert '"This week saw..."' in result
        assert '"critical,"' in result
        assert "---" in result

    def test_repeated_same_placeholder_resolves_each(self):
        text = "{{style:stage_direction_cliches}}\n{{style:stage_direction_cliches}}"
        result = expand_style_placeholders(text)
        assert result.count('- "took center stage"') == 2


# =============================================================================
# Catalogue contents: sanity checks for known regression patterns
# =============================================================================


class TestCatalogueContents:
    """Lock in that representative tics from real runs stay catalogued."""

    def test_this_week_saw_in_hollow_openers(self):
        displays = [p.display for p in HOLLOW_WEEK_OPENERS]
        assert any("This week saw" in d for d in displays)

    def test_took_center_stage_in_stage_cliches(self):
        displays = [p.display for p in STAGE_DIRECTION_CLICHES]
        assert any("took center stage" in d for d in displays)

    def test_landscape_in_abstract_landscape_openers(self):
        displays = [p.display for p in ABSTRACT_LANDSCAPE_OPENERS]
        assert any("landscape of" in d for d in displays)

    def test_critical_in_filler_adjectives(self):
        displays = [p.display for p in FILLER_ADJECTIVES]
        assert any('"critical,"' in d for d in displays)

    def test_underscore_in_abstract_verbs(self):
        displays = [p.display for p in ABSTRACT_INFORMATION_VERBS]
        assert any("underscore" in d for d in displays)

    def test_meanwhile_in_generic_transitions(self):
        displays = [p.display for p in GENERIC_TRANSITION_OPENERS]
        assert any("Meanwhile" in d for d in displays)

    def test_race_is_on_in_watch_forecasts(self):
        displays = [p.display for p in WATCH_WEAK_FORECASTS]
        assert any("race is on" in d for d in displays)

    def test_drastically_filler_intensifier_catalogued(self):
        displays = [p.display for p in FILLER_ADJECTIVES]
        assert any("drastically" in d for d in displays)

    def test_growing_tension_in_stage_cliches(self):
        displays = [p.display for p in STAGE_DIRECTION_CLICHES]
        assert any("growing tension" in d for d in displays)

    def test_inflection_point_in_stage_cliches(self):
        displays = [p.display for p in STAGE_DIRECTION_CLICHES]
        assert any("inflection point" in d for d in displays)

    def test_shifting_toward_in_stage_cliches(self):
        displays = [p.display for p in STAGE_DIRECTION_CLICHES]
        assert any("shifting toward" in d for d in displays)

    def test_vendor_also_in_enumeration_openers(self):
        displays = [p.display for p in ENUMERATION_OPENERS]
        assert any("Vendor X also released" in d for d in displays)

    def test_on_the_topic_front_in_enumeration_openers(self):
        displays = [p.display for p in ENUMERATION_OPENERS]
        assert any("On the [topic] front" in d for d in displays)


# =============================================================================
# End-to-end: load_prompt resolves placeholders for subscribing templates
# =============================================================================


# The bundled prompt templates ship as placeholders, so the tests below that
# assert on specific prompt wording are expected to fail until the baseline
# prompt content is written.
_PLACEHOLDER_PROMPT_XFAIL = pytest.mark.xfail(
    reason="Bundled prompt templates are placeholders pending baseline content.",
    strict=False,
)


class TestLoadPromptResolution:
    @pytest.mark.parametrize(
        "template",
        [
            "editorial_pass_system",
            "intro_system",
            "watch_system",
            "section_system",
            "section_merge_system",
        ],
    )
    def test_subscribing_templates_have_no_unresolved_placeholders(self, template):
        text = load_prompt(template)
        assert "{{style:" not in text

    @_PLACEHOLDER_PROMPT_XFAIL
    def test_editorial_pass_includes_canonical_catalogue(self):
        text = load_prompt("editorial_pass_system")
        assert '"This week saw..."' in text
        assert '"took center stage"' in text
        assert '"underscore," "underscores," "underscoring"' in text
        assert "Vendor X also released" in text

    @_PLACEHOLDER_PROMPT_XFAIL
    def test_intro_includes_shared_catalogue_categories(self):
        text = load_prompt("intro_system")
        assert '"This week saw..."' in text
        assert '"The landscape of..."' in text
        assert '"crystallized"' in text  # was NOT in intro's pre-extraction list

    @_PLACEHOLDER_PROMPT_XFAIL
    def test_watch_includes_weak_forecasts_and_shared_categories(self):
        text = load_prompt("watch_system")
        assert "race is on" in text
        assert '"took center stage"' in text
        assert '"underscore," "underscores," "underscoring"' in text

    @_PLACEHOLDER_PROMPT_XFAIL
    def test_section_includes_enumeration_openers(self):
        text = load_prompt("section_system")
        assert "Vendor X also released" in text
        assert "On the [topic] front" in text

    @_PLACEHOLDER_PROMPT_XFAIL
    def test_section_includes_shared_catalogue_categories(self):
        text = load_prompt("section_system")
        assert '"critical,"' in text
        assert '"underscore," "underscores," "underscoring"' in text
        assert '"took center stage"' in text

    @_PLACEHOLDER_PROMPT_XFAIL
    def test_section_merge_includes_enumeration_openers(self):
        text = load_prompt("section_merge_system")
        assert "Vendor X also released" in text

    @_PLACEHOLDER_PROMPT_XFAIL
    def test_section_merge_includes_shared_catalogue_categories(self):
        text = load_prompt("section_merge_system")
        assert '"critical,"' in text
        assert '"underscore," "underscores," "underscoring"' in text
        assert '"took center stage"' in text

    def test_non_subscribing_templates_unchanged_by_resolver(self):
        # Only title_system has no placeholders.
        for template in ("title_system",):
            text = load_prompt(template)
            assert "{{style:" not in text
