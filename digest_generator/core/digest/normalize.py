"""Deterministic post-composer normalization of the deliverable markdown.

A terminal pass over the assembled digest body (and title), run after
``DigestComposer`` and before the frontmatter is prepended. It fixes the
classes of mechanical defect that recur in published digests, survive every
LLM stage, and don't belong in a prompt because they are pure string
transforms: locale drift (``37 %`` -> ``37%``, British spelling), scale and
currency abbreviations (``$9.36bn`` -> ``$9.36 billion``, ``21 k`` ->
``21,000``), leaked notation (a multiplication sign between digits -> ``x``,
``$N=200$`` -> ``N=200``), and canonical entity casing.

Several of these are not merely visual: ``bn`` and ``21 k`` narrate badly, so
the defect is audible in the TTS rendition, not just cosmetic on the page.

Span safety
-----------
Transforms run only over prose. Fenced/indented code, inline code, markdown
link targets *and anchor text*, and bare/auto URLs are protected before any
transform and restored after, because:

- a URL slug or a ``step.do()`` token must survive verbatim, and
- **link anchor text and quoted source titles are citation-grade**. The
  2026-07-12 digest cited a real article titled "How Meta's First Canadian
  Date Centre Catalyses AI Progress"; a naive locale pass rewriting
  ``Centre``/``Catalyses`` inside that anchor would silently corrupt a
  source's own title, a worse defect than the drift it fixes.

Bare quoted titles that are *not* markdown links remain a residual: they are
indistinguishable from ordinary quoted prose without a citation model, so the
locale and entity transforms may touch them. Prefer linking cited titles.

Pairs with ``digest_generator.core.digest.lint``: once this pass runs, the
lint can assert zero residual ``\\d\\s%`` / ``bn`` / ``$…$`` as a regression
guard, sharing the intent (not the code) of the transforms here.
"""

from __future__ import annotations

import re

# Private-use sentinels wrapping a protected span's index. The open/close pair
# is deliberately outside [A-Za-z0-9_] so it reads as a word boundary, and the
# index digits between them can never satisfy any transform below (each needs a
# trailing unit / space-group / slash / times-sign / $ / %, none of which a
# sentinel supplies). See module docstring "Span safety".
_PROTECT_OPEN = "\ue000"
_PROTECT_CLOSE = "\ue001"

# Order matters: fenced code first (swallows anything inside), then whole
# markdown links (anchor + target), then autolinks / bare URLs, then inline
# code. A link whose anchor contains an inline-code span is matched whole by
# the link alternative before the inline-code alternative is tried.
_PROTECT_RE = re.compile(
    r"```[\s\S]*?```"  # fenced code (backtick)
    r"|~~~[\s\S]*?~~~"  # fenced code (tilde)
    r"|\[[^\]]*\]\([^)]*\)"  # markdown link: [anchor](target)
    r"|<https?://[^>]+>"  # autolink
    r"|https?://\S+"  # bare URL
    r"|`[^`]+`"  # inline code
)

# Invisible Unicode whitespace some models emit as unit / thousands separators
# (e.g. "10,000<U+202F>RPS"). Collapsed to ASCII space so the space-grouped
# thousands rule below can then comma-group it, and so downstream string
# comparisons don't trip on a space that looks identical to a regular one.
_INVISIBLE_WS_RE = re.compile("[\u00a0\u2009\u202f]")

# ``37 %`` -> ``37%`` (US house style; the editor leaves the space untouched
# because it isn't a catalogue phrase).
_PERCENT_RE = re.compile(r"(\d)\s+%")

# Space-grouped thousands (``430 000`` -> ``430,000``, ``1 234 567`` ->
# ``1,234,567``). The 1-3 leading-digit anchor keeps a 4-digit year ("2024
# 300") from matching, since a year is not a valid thousands lead.
_SPACE_THOUSANDS_RE = re.compile(r"\b(\d{1,3}(?: \d{3})+)\b")

# ``21 k`` and currency-prefixed ``$430k`` -> expanded, comma-grouped integers.
# Bare integer+k ("4k", "8k") is deliberately NOT matched: it collides with
# display resolutions ("4K", "8K") and product names. The space or the currency
# prefix is the disambiguating "thousand" signal.
_K_SPACED_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s+k\b")
_K_CURRENCY_RE = re.compile(r"([$£€¥])(\d+(?:\.\d+)?)\s?[kK]\b")

# Scale suffixes that spell out rather than expand digits (``$9.36bn`` ->
# ``$9.36 billion``). ``bn``/``tn``/``trn``/``mn`` are unit-unambiguous; bare
# ``m`` is left alone ("5m" could be meters or million).
_SCALE_WORDS = {"bn": "billion", "tn": "trillion", "trn": "trillion", "mn": "million"}
_SCALE_RE = re.compile(r"(\d)\s?(bn|trn|tn|mn)\b")

# Capital scale letters on a money amount (``$100 M`` -> ``$100 million``,
# ``$1.2B`` -> ``$1.2 billion``). The currency prefix is the whole
# disambiguator: a bare capital after a number is far too common to touch
# ("Plan B", "class B shares", "Category 5"), so only an amount qualifies.
# ``K`` is absent here on purpose — ``_K_CURRENCY_RE`` already expands it to
# comma-grouped digits, and two rules spelling the same suffix differently
# would be worse than either. Matters beyond house style: the audio narrator
# reads these aloud, and "one hundred M" is not a number.
_SCALE_LETTERS = {"M": "million", "B": "billion", "T": "trillion"}
_SCALE_CURRENCY_RE = re.compile(r"([$£€¥]\d+(?:\.\d+)?)\s?([MBT])\b")

# Inline LaTeX math leaking into prose (``$N=200$``, ``$k=5$``, ``$\alpha$``).
# Conservative: the span must start with a letter or backslash (so currency
# "$9.36" / "$700 million", which start with a digit, never match) and contain
# ``=`` or a backslash. Delimiters are stripped, inner text kept.
_LATEX_RE = re.compile(r"\$([A-Za-z\\][^$]{0,60}?)\$")

# A multiplication sign (U+00D7) adjacent to a digit -> ``x`` (e.g. "3x",
# "100x", "2x3"). A bare times-sign between words (a real dimension "A x B") is
# left alone. The literal in the pattern is the U+00D7 the re module matches.
_TIMES_RE = re.compile(r"(\d)\s*×")  # noqa: RUF001 (literal U+00D7 we match on)

# ``GPT 5`` / ``GPT5`` -> ``GPT-5`` (canonical product hyphenation). Already
# hyphenated forms are unaffected.
_GPT_HYPHEN_RE = re.compile(r"\bGPT[- ]?(\d)")

# Small fractions spelled out in prose. Guarded against dates ("1/3/2026") and
# ratios by refusing an adjacent digit or slash on either side.
_FRACTIONS = {
    "1/2": "one-half",
    "1/3": "one-third",
    "2/3": "two-thirds",
    "1/4": "one-quarter",
    "3/4": "three-quarters",
}
_FRACTION_RE = re.compile(r"(?<![\d/])(1/2|1/3|2/3|1/4|3/4)(?![\d/])")

# British -> US spelling, whole-word, case-preserving. Curated and conservative:
# only forms the editorial pass reliably leaves (they aren't catalogue phrases)
# and that have no common US homograph. Noun/verb pairs the two locales spell
# differently in only one direction (licence/license, practise/practice) are
# omitted to avoid miscorrection.
_BRITISH_TO_US = {
    "colour": "color",
    "colours": "colors",
    "behaviour": "behavior",
    "behaviours": "behaviors",
    "favour": "favor",
    "favours": "favors",
    "labour": "labor",
    "centre": "center",
    "centres": "centers",
    "metre": "meter",
    "metres": "meters",
    "fibre": "fiber",
    "fibres": "fibers",
    "defence": "defense",
    "offence": "offense",
    "organise": "organize",
    "organised": "organized",
    "organisation": "organization",
    "organisations": "organizations",
    "realise": "realize",
    "realised": "realized",
    "recognise": "recognize",
    "recognised": "recognized",
    "analyse": "analyze",
    "analysed": "analyzed",
    "catalyse": "catalyze",
    "catalyses": "catalyzes",
    "optimise": "optimize",
    "optimised": "optimized",
    "prioritise": "prioritize",
    "prioritised": "prioritized",
    "normalisation": "normalization",
    "generalise": "generalize",
    "generalised": "generalized",
    "labelled": "labeled",
    "labelling": "labeling",
    "modelled": "modeled",
    "modelling": "modeling",
    "signalled": "signaled",
    "cancelled": "canceled",
    "travelled": "traveled",
    "travelling": "traveling",
    "catalogue": "catalog",
    "programme": "program",
    "artefact": "artifact",
    "artefacts": "artifacts",
    "grey": "gray",
    "whilst": "while",
    "amongst": "among",
    "learnt": "learned",
}
_BRITISH_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _BRITISH_TO_US) + r")\b",
    re.IGNORECASE,
)

# Canonical entity / product casing. Kept intentionally small and high
# confidence: each entry is a form the house style commits to and that a model
# reliably renders inconsistently within one digest. This table is the
# maintainer's editorial call — grow it from observed drift, not speculatively.
_ENTITY_CASING = {
    "Open AI": "OpenAI",
    "OpenAi": "OpenAI",
    "Github": "GitHub",
    "GitHub": "GitHub",
    "Pytorch": "PyTorch",
    "Tensorflow": "TensorFlow",
    "Deepmind": "DeepMind",
    "Huggingface": "Hugging Face",
}
_ENTITY_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _ENTITY_CASING) + r")\b",
)


def _protect(text: str) -> tuple[str, list[str]]:
    """Replace protected spans with sentinels; return (masked_text, spans)."""
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return f"{_PROTECT_OPEN}{len(spans) - 1}{_PROTECT_CLOSE}"

    return _PROTECT_RE.sub(stash, text), spans


def _restore(text: str, spans: list[str]) -> str:
    """Reinsert protected spans in place of their sentinels."""
    for i, span in enumerate(spans):
        text = text.replace(f"{_PROTECT_OPEN}{i}{_PROTECT_CLOSE}", span)
    return text


def _match_case(source: str, replacement: str) -> str:
    """Return ``replacement`` cased to mirror ``source`` (all-caps/Title/lower)."""
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _expand_k(number: str) -> str:
    """Turn a ``k``-suffixed number string into a comma-grouped integer."""
    value = round(float(number) * 1000)
    return f"{value:,}"


def _apply_transforms(text: str) -> str:
    """Run every prose transform over already-masked text."""
    text = _INVISIBLE_WS_RE.sub(" ", text)
    text = _PERCENT_RE.sub(r"\1%", text)
    text = _SPACE_THOUSANDS_RE.sub(lambda m: m.group(1).replace(" ", ","), text)
    text = _K_CURRENCY_RE.sub(lambda m: f"{m.group(1)}{_expand_k(m.group(2))}", text)
    text = _K_SPACED_RE.sub(lambda m: _expand_k(m.group(1)), text)
    text = _SCALE_RE.sub(lambda m: f"{m.group(1)} {_SCALE_WORDS[m.group(2)]}", text)
    text = _SCALE_CURRENCY_RE.sub(lambda m: f"{m.group(1)} {_SCALE_LETTERS[m.group(2)]}", text)
    text = _LATEX_RE.sub(
        lambda m: m.group(1) if ("=" in m.group(1) or "\\" in m.group(1)) else m.group(0),
        text,
    )
    text = _TIMES_RE.sub(r"\1x", text)
    text = _GPT_HYPHEN_RE.sub(r"GPT-\1", text)
    text = _FRACTION_RE.sub(lambda m: _FRACTIONS[m.group(1)], text)
    text = _BRITISH_RE.sub(
        lambda m: _match_case(m.group(1), _BRITISH_TO_US[m.group(1).lower()]),
        text,
    )
    text = _ENTITY_RE.sub(lambda m: _ENTITY_CASING[m.group(1)], text)
    return text


def mask_protected_spans(text: str) -> str:
    """Blank out protected spans, preserving length, offsets, and newlines.

    Replaces every non-newline character inside a code / link / URL span with a
    space so a scanner (the golden-output lint) can search prose for residual
    defects without matching inside a code token or a link target, while line
    and column numbers of everything else stay exact.
    """

    def blank(match: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    return _PROTECT_RE.sub(blank, text)


def normalize_output(text: str) -> str:
    """Normalize a block of digest prose (body or title) in place.

    Protects code, links (target and anchor), and URLs, applies the mechanical
    transforms to the remaining prose, then restores the protected spans. Safe
    to run on the composed body or a bare title string; idempotent for text
    that is already normalized.
    """
    if not text:
        return text
    masked, spans = _protect(text)
    masked = _apply_transforms(masked)
    return _restore(masked, spans)
