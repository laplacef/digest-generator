"""Single source of truth for the forbidden-phrase catalogue.

Multiple prompts (editorial pass, intro, watcher, plus the LLM-driven
article summarizer) forbid the same LLM tics: hollow openers, filler
adjectives, abstract-information verbs, and stage-direction cliches.
This module is the canonical list; each prompt template subscribes to
the categories it cares about via ``{{style:CATEGORY}}`` placeholders
that the per-package ``load_prompt`` resolves on read.

It lives at ``digest_generator/core/style.py`` (not under ``digest/``)
because both the digest sub-package's prompts and the article
summarizer's prompt consume it.

Consumers
---------
- Digest prompt templates via ``digest_generator.core.digest.prompts.load_prompt``.
- The article-summarizer prompt template via the summarizer's local
  ``load_prompt`` (mirrors the digest pattern).
- A digest-style regression suite can import the category lists
  directly to lint generated output.
- A prompt eval harness can import the catalogue to track which
  patterns the model still emits between baseline and tuned runs.

Adding a new tic
----------------
Add a ``ForbiddenPhrase`` to the matching category list. Every prompt
that subscribes to the category picks it up automatically on the next
``load_prompt`` call. If the tic doesn't fit an existing category, add
a new constant and register it in ``_CATEGORIES``, then update each
consuming prompt to reference the new placeholder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PLACEHOLDER_RE = re.compile(r"\{\{style:([a-z_]+)\}\}")


@dataclass(frozen=True)
class ForbiddenPhrase:
    """One forbidden phrase rendered into a prompt as a Markdown bullet.

    ``display`` is the bullet content as it should appear in the prompt
    (without the leading ``- ``). Multi-conjugation entries group their
    forms together (e.g. ``"underscore," "underscores," "underscoring"``)
    so each related variant is forbidden in one bullet.

    ``pattern`` is an optional regex (as a raw string) that the golden-output
    lint compiles and searches the generated digest against. It is set only on
    entries precise enough to be a *hard* signal — an unambiguous multi-word
    cliche that is always wrong, never a bare adjective like "critical" that
    has legitimate uses (a "critical" CVE severity) the editorial pass judges
    in context. Leaving ``pattern`` ``None`` keeps a phrase prompt-enforced
    only. See ``digest_generator.core.digest.lint``.
    """

    display: str
    pattern: str | None = None


GENERIC_TRANSITION_OPENERS: list[ForbiddenPhrase] = [
    # Anchored to a line start: these words are only a tic as paragraph
    # openers, so a mid-sentence "meanwhile" stays legitimate. Paragraphs in
    # composed output are one line each, so ``^`` (MULTILINE) is the opener.
    ForbiddenPhrase('"Meanwhile,"', pattern=r"(?im)^\s*meanwhile,"),
    ForbiddenPhrase('"Concurrently,"', pattern=r"(?im)^\s*concurrently,"),
    ForbiddenPhrase('"Similarly,"', pattern=r"(?im)^\s*similarly,"),
    ForbiddenPhrase('"Alongside,"', pattern=r"(?im)^\s*alongside,"),
    ForbiddenPhrase('"Additionally,"', pattern=r"(?im)^\s*additionally,"),
    ForbiddenPhrase('"Furthermore,"', pattern=r"(?im)^\s*furthermore,"),
    ForbiddenPhrase('"Moreover,"', pattern=r"(?im)^\s*moreover,"),
    ForbiddenPhrase('"In parallel,"', pattern=r"(?im)^\s*in parallel,"),
    ForbiddenPhrase('"At the same time,"', pattern=r"(?im)^\s*at the same time,"),
]

HOLLOW_WEEK_OPENERS: list[ForbiddenPhrase] = [
    ForbiddenPhrase(
        '"This week\'s developments highlight..."',
        pattern=r"(?i)\bthis week'?s developments\b",
    ),
    ForbiddenPhrase('"This week saw..."', pattern=r"(?i)\bthis week saw\b"),
    ForbiddenPhrase('"This week was dominated by..."', pattern=r"(?i)\bthis week was dominated\b"),
    ForbiddenPhrase(
        '"This week underscored..." / "This week underscores..."',
        pattern=r"(?i)\bthis week underscore[sd]\b",
    ),
    ForbiddenPhrase(
        '"This week was defined by..." / "This week was marked by..."',
        pattern=r"(?i)\bthis week was (?:defined|marked)\b",
    ),
    ForbiddenPhrase('"This week offered..."', pattern=r"(?i)\bthis week offered\b"),
    ForbiddenPhrase('"The week also highlighted..."', pattern=r"(?i)\bthe week also\b"),
]

ABSTRACT_LANDSCAPE_OPENERS: list[ForbiddenPhrase] = [
    ForbiddenPhrase('"The landscape of..."', pattern=r"(?i)\bthe landscape of\b"),
    ForbiddenPhrase('"The intersection of..."', pattern=r"(?i)\bthe intersection of\b"),
    ForbiddenPhrase('"In the realm of..."', pattern=r"(?i)\bin the realm of\b"),
    ForbiddenPhrase('"Beneath the X,"'),
]

FILLER_ADJECTIVES: list[ForbiddenPhrase] = [
    ForbiddenPhrase(
        '"critical," "significant," "comprehensive," "key," "robust," "unprecedented," "sophisticated"'
    ),
    ForbiddenPhrase(
        '"paramount," "blistering," "staggering," "sweeping," "remarkable," "striking"'
    ),
    ForbiddenPhrase('"seamless," "cutting-edge," "pivotal," "crucial"'),
    ForbiddenPhrase(
        '"drastically," "dramatically," "rapidly" (intensifier filler — name the magnitude instead)'
    ),
]

ABSTRACT_INFORMATION_VERBS: list[ForbiddenPhrase] = [
    ForbiddenPhrase('"underscore," "underscores," "underscoring"'),
    ForbiddenPhrase('"highlight," "highlights," "highlighting," "highlighted by"'),
    ForbiddenPhrase('"illustrate," "illustrated by"'),
    ForbiddenPhrase('"demonstrate," "demonstrated," "showcased," "showcasing"'),
    ForbiddenPhrase('"emphasize," "emphasized"'),
    ForbiddenPhrase('"dominate," "dominated"'),
    ForbiddenPhrase('"manifest," "manifested"'),
    ForbiddenPhrase('"reveal" (as in "X reveals the trend of Y")'),
]

STAGE_DIRECTION_CLICHES: list[ForbiddenPhrase] = [
    ForbiddenPhrase('"took center stage"', pattern=r"(?i)\btook center stage\b"),
    ForbiddenPhrase('"emerged as a central theme"', pattern=r"(?i)\bemerged as a central theme\b"),
    ForbiddenPhrase('"crystallized"'),
    ForbiddenPhrase(
        '"remained active and sophisticated"',
        pattern=r"(?i)\bremained active and sophisticated\b",
    ),
    ForbiddenPhrase(
        '"demanded immediate attention"', pattern=r"(?i)\bdemanded immediate attention\b"
    ),
    ForbiddenPhrase('"growing tension," "growing tensions," "rising tension"'),
    ForbiddenPhrase('"critical inflection point," "reached an inflection point"'),
    ForbiddenPhrase(
        '"shifting toward," "shift toward," "shifts toward," "is moving toward" (the rule fires'
        ' when used as vague hand-wave: "the industry is shifting toward agents." If both the'
        " starting point AND the destination are named with concrete entities, the phrase"
        ' passes — "Enterprise AI is shifting from chatbots to agents that orchestrate'
        ' multi-step workflows" is fine; "the landscape is shifting toward agents" is not)'
    ),
    ForbiddenPhrase(
        '"moving from experimentation to production," "moving from X to production-grade"'
    ),
]

WATCH_WEAK_FORECASTS: list[ForbiddenPhrase] = [
    ForbiddenPhrase(
        '"The race is on..." (overused cliché; name the specific race if it\'s real)',
        pattern=r"(?i)\bthe race is on\b",
    ),
    ForbiddenPhrase(
        '"Watch for continued..." / "Expect continued..." (weak forecast; name what specifically changes)'
    ),
]

ENUMERATION_OPENERS: list[ForbiddenPhrase] = [
    ForbiddenPhrase(
        '"Vendor X also released..." / "X also launched..." (the "also" pattern flattens'
        " a paragraph into a vendor catalogue; lead with the thesis the vendor's release evidences instead)"
    ),
    ForbiddenPhrase(
        '"On the [topic] front,..." / "On the security front,..." / "On the AI front,...'
        '" (topic-bucket transitions; name the specific link between the items)'
    ),
    ForbiddenPhrase(
        '"In [area], several updates surfaced..." / "In the X space,..." (groups items by'
        " topic without a thesis binding them)"
    ),
    ForbiddenPhrase(
        '"This week, [vendor] announced..." / "[Vendor] announced..." as a paragraph opener'
        " when the vendor's announcement is one of several items in the paragraph (lead with"
        " the cross-cutting claim, not the announcer)"
    ),
]


_CATEGORIES: dict[str, list[ForbiddenPhrase]] = {
    "generic_transition_openers": GENERIC_TRANSITION_OPENERS,
    "hollow_week_openers": HOLLOW_WEEK_OPENERS,
    "abstract_landscape_openers": ABSTRACT_LANDSCAPE_OPENERS,
    "filler_adjectives": FILLER_ADJECTIVES,
    "abstract_information_verbs": ABSTRACT_INFORMATION_VERBS,
    "stage_direction_cliches": STAGE_DIRECTION_CLICHES,
    "watch_weak_forecasts": WATCH_WEAK_FORECASTS,
    "enumeration_openers": ENUMERATION_OPENERS,
}


# Brand/product names that read as vendor marketing when they lead a title
# with a generic announcement verb ("OpenAI Releases…"). Shared by the framer's
# title-retry guard and the golden-output lint's title check.
BRAND_LED_VERBS: tuple[str, ...] = (
    "Finds",
    "Releases",
    "Launches",
    "Unveils",
    "Announces",
    "Reveals",
    "Drops",
    "Ships",
    "Patches",
    "Debuts",
    "Introduces",
    "Adds",
    "Brings",
    "Delivers",
    "Rolls Out",
)

# Forbidden title shapes as ``(regex, reason)`` pairs. Single source of truth
# for both the framer's retry-with-feedback guard (``framer.py``) and the
# golden-output lint (``digest.lint``), so a rule promoted from prompt-only to
# enforced lives in exactly one place. Reasons are phrased for the framer's
# ``<retry-feedback>`` block ("It <reason>").
TITLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^AI\s", "starts with 'AI ' — AI is the digest's topic, not news"),
    (r"^\w+:\s", "uses the 'X: Y' colon-split format"),
    (
        r"^(?:[A-Z][a-zA-Z0-9]*\s+){1,3}(?:"
        + "|".join(v.replace(" ", r"\s+") for v in BRAND_LED_VERBS)
        + r")\s",
        (
            "leads with a brand/product name plus a generic announcement verb "
            "(e.g. 'OpenAI Releases', 'Claude Mythos Finds') — reads like vendor "
            "marketing, not editorial framing"
        ),
    ),
)


def iter_lintable_phrases() -> list[tuple[str, ForbiddenPhrase]]:
    """Return ``(category, phrase)`` pairs for every phrase carrying a regex.

    The golden-output lint iterates this to match generated prose against the
    precise subset of the catalogue. Phrases with ``pattern=None`` (bare
    adjectives, prose-shaped descriptions) are prompt-enforced only and skipped.
    """
    return [
        (category, phrase)
        for category, phrases in _CATEGORIES.items()
        for phrase in phrases
        if phrase.pattern is not None
    ]


def render_bullets(items: list[ForbiddenPhrase]) -> str:
    """Render a category as Markdown bullets for inclusion in a prompt."""
    return "\n".join(f"- {item.display}" for item in items)


def expand_style_placeholders(text: str) -> str:
    """Replace ``{{style:CATEGORY}}`` placeholders with rendered bullets.

    Raises ``KeyError`` for an unknown category, surfacing typos at
    ``load_prompt`` time (stage-module import time) rather than
    silently shipping a prompt with a literal ``{{style:typo}}`` in it.
    """

    def replace(match: re.Match[str]) -> str:
        category = match.group(1)
        if category not in _CATEGORIES:
            msg = f"Unknown style category {category!r}. Known categories: {sorted(_CATEGORIES)}"
            raise KeyError(msg)
        return render_bullets(_CATEGORIES[category])

    return _PLACEHOLDER_RE.sub(replace, text)
