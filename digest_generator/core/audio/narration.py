"""Markdown to Piper-friendly narration script.

Walks a ``markdown-it-py`` token stream and emits plain text. Pause
cues are sentence-terminator punctuation plus newlines, not SSML.
Piper reads SSML tags as literal text ("break time equals four
hundred milliseconds"), so the narration emits plain pause cues rather
than ``<break time="..."/>`` tags.

Pause control (Piper-native):

- Headings (H1/H2/H3) become standalone sentences with paragraph
  breaks on both sides. Piper drops pitch at the period and pauses
  for ``--sentence-silence`` seconds at each boundary.
- Paragraphs become standalone sentences with a paragraph break
  after.
- List items become individual sentences with a single newline
  after, giving each item a distinct beat without the longer
  paragraph-break pause.
- Fenced code blocks, indented code blocks, and horizontal rules
  collapse to a paragraph break. The surrounding context handles
  the boundary; there is no narrated content.

Pause duration is tuned globally via ``settings.audio_sentence_silence_s``,
threaded through to Piper's ``--sentence-silence`` flag. Raising that
value lengthens every sentence boundary; the narration script itself
stays unchanged. Paragraph boundaries additionally emit a non-vocalized
em-dash empty sentence so paragraph-to-paragraph pauses are twice
the sentence-within-paragraph pause without affecting list-item beats.

Strips formatting markers (bold/italic/inline code), drops link URLs
(keeping link text), skips fenced code and YAML frontmatter, applies
pre-walk text normalizers (currency reorder, Nx becomes "times", compact
units, dotted national acronyms, "~" becomes "approximately", trailing
acronym-definition strip), and applies pronunciation overrides from
``narration_overrides.yaml``.

Public API:

- ``markdown_to_narration(md_text, *, overrides=None)``: the main entry
  point; returns a single string ready to feed to ``piper`` on stdin.
- ``load_overrides(path=None)``: read the bundled overrides file (or a
  caller-supplied path) and return the override dict.
- ``NARRATION_VERSION``: bump when the narration output shape changes
  (included in the audio cache key so version bumps invalidate
  existing renders).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from markdown_it import MarkdownIt
from markdown_it.token import Token

__all__ = [
    "NARRATION_VERSION",
    "load_overrides",
    "markdown_to_narration",
]

# Bump when the narration output shape changes. Included in the audio
# cache key (`digest_generator.core.audio.io.compute_cache_key`) so a version
# bump invalidates existing renders.
#
# The current shape (v3) uses newlines plus sentence-terminator
# punctuation for pacing, with the --sentence-silence flag controlling
# pause duration at sentence boundaries. It applies pre-walk text
# normalizers (currency reorder, "Nx" to "times", compact units, dotted
# national acronyms, "~" to "approximately", trailing acronym-definition
# strip) and pads paragraph boundaries with an em-dash empty sentence so
# they get a second --sentence-silence interval without affecting
# sentence-within-paragraph or list-item pacing.
NARRATION_VERSION = "v3"

# Strip YAML frontmatter: the first --- delimited block at the top
# of the file. Eval scripts and the digest composer both emit this.
_FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)

# Bare URLs in prose: keep narration smooth by dropping them outright.
_URL_RE = re.compile(r"https?://\S+")

# Collapse horizontal whitespace runs (spaces/tabs) but preserve newlines,
# which are load-bearing pause cues for Piper.
_HSPACE_RE = re.compile(r"[ \t]+")
# Cap consecutive newlines at 2 (paragraph break). Three or more would
# read identically to two, so collapse for cleaner output.
_NEWLINE_COLLAPSE_RE = re.compile(r"\n{3,}")

# Pre-walk text normalizers run on the raw markdown string before
# MarkdownIt.parse so the parser sees already-normalized prose. Each
# targets a specific pattern Piper voices poorly.

# Currency: "$700 million" becomes "700 million dollars". Captures optional
# magnitude word so the unit moves with the amount.
_CURRENCY_RE = re.compile(r"\$([\d,]+(?:\.\d+)?)((?:\s+(?:thousand|million|billion|trillion))?)")

# Numeric multipliers: "9.2x" becomes "9.2 times". Word-boundary on the right
# blocks matches like "xlarge".
_NX_TIMES_RE = re.compile(r"\b(\d+(?:\.\d+)?)x\b")

# Compact + spaced units: "10GW" / "10 GW" becomes "10 gigawatts". This regex
# supersedes the spaced-form unit overrides in narration_overrides.yaml.
_UNIT_EXPANSIONS = {
    "GW": "gigawatts",
    "MW": "megawatts",
    "GB": "gigabytes",
    "MB": "megabytes",
    "KB": "kilobytes",
    "TB": "terabytes",
    "Tbps": "terabits per second",
    "Gbps": "gigabits per second",
    "Mbps": "megabits per second",
    "Kbps": "kilobits per second",
}
_UNIT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(" + "|".join(_UNIT_EXPANSIONS) + r")\b")

# Dotted national acronyms: "U.S." becomes "US". Without this the period reads
# as a sentence terminator mid-sentence ("U.S. Department" becomes two
# sentences in Piper's view).
_DOTTED_ACRONYM_RE = re.compile(r"\b(U\.S|U\.K|U\.N|D\.C)\.")

# Tilde-as-approximate: "~35,000" becomes "approximately 35,000". Lookahead so
# bare tildes in prose (rare) aren't touched.
_TILDE_NUM_RE = re.compile(r"~(?=\d)")

# Trailing acronym definitions: " (FGA)" after a phrase, dropped. The prose
# already names the concept; the override would otherwise pronounce the
# expansion twice (e.g. "Fine Grained Authorization (fine-grained
# authorization)"). Restricted to space-leading (not newline-leading) so
# paragraph-opening parentheticals aren't accidentally merged into the
# preceding paragraph.
_TRAILING_ACRONYM_RE = re.compile(r" +\([A-Z]{2,}s?\)")

# Paragraph pause padding. A standalone sentence containing only an
# em-dash + period emits a second --sentence-silence interval between
# paragraphs without affecting sentence-within-paragraph or list-item
# pacing. Em-dash is non-vocalized by Piper's phonemizer (eSpeak treats
# U+2014 as silence); the period is the sentence-boundary trigger.
_PARAGRAPH_PAUSE_MARKER = "—."


def _normalize_pre_walk(text: str) -> str:
    """Apply text-normalization regexes to raw markdown before parsing.

    Runs on the markdown source string (after frontmatter strip, before
    MarkdownIt.parse) so the parser sees the already-normalized prose.
    Patterns targeted are listed in the module-level regex constants;
    each fixes a specific class of Piper mispronunciation.
    """
    text = _CURRENCY_RE.sub(lambda m: f"{m.group(1)}{m.group(2)} dollars", text)
    text = _NX_TIMES_RE.sub(r"\1 times", text)
    text = _UNIT_RE.sub(lambda m: f"{m.group(1)} {_UNIT_EXPANSIONS[m.group(2)]}", text)
    text = _DOTTED_ACRONYM_RE.sub(lambda m: m.group(1).replace(".", ""), text)
    text = _TILDE_NUM_RE.sub("approximately ", text)
    text = _TRAILING_ACRONYM_RE.sub("", text)
    return text


def _bundled_overrides_path() -> Path:
    """Path to the YAML overrides file shipped alongside this module."""
    return Path(__file__).with_name("narration_overrides.yaml")


def load_overrides(path: Path | None = None) -> dict[str, str]:
    """Load pronunciation overrides from a YAML file.

    Args:
        path: Override file location. Defaults to the bundled
            ``narration_overrides.yaml`` next to this module.

    Returns:
        Mapping from literal token to speech form. Empty dict if the file
        is missing or empty (fail-soft: missing overrides are not fatal).
    """
    resolved = path or _bundled_overrides_path()
    if not resolved.exists():
        return {}
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        msg = f"overrides file {resolved} must be a YAML mapping"
        raise ValueError(msg)
    return {str(k): str(v) for k, v in raw.items()}


def _render_inline(token: Token) -> str:
    """Render an ``inline`` token's children to plain text, dropping markers."""
    if not token.children:
        return _URL_RE.sub("", token.content)
    parts: list[str] = []
    for child in token.children:
        kind = child.type
        if kind in ("text", "code_inline"):
            parts.append(child.content)
        elif kind in ("softbreak", "hardbreak"):
            parts.append(" ")
        elif kind in (
            "em_open",
            "em_close",
            "strong_open",
            "strong_close",
            "s_open",
            "s_close",
            "link_open",
            "link_close",
            "image",
        ):
            # Drop the marker; the surrounding text/children carry the content.
            # Link URLs and image references are dropped entirely; the renderer
            # only narrates spoken content.
            continue
        elif kind == "html_inline":
            # Strip inline HTML.
            continue
        elif child.content:
            # Unknown inline type: best-effort fall back to content.
            parts.append(child.content)
    # Bare URLs that didn't get caught as markdown links (commonmark doesn't
    # autolink raw URLs) are stripped here, after marker-aware rendering, so
    # that "[text](url)" structures are not accidentally truncated.
    return _URL_RE.sub("", "".join(parts))


def _apply_overrides(text: str, overrides: dict[str, str]) -> str:
    """Whole-word, case-sensitive replacement of override keys."""
    if not overrides:
        return text
    for key, replacement in overrides.items():
        pattern = rf"(?<!\w){re.escape(key)}(?!\w)"
        text = re.sub(pattern, replacement, text)
    return text


def _ensure_terminator(text: str) -> str:
    """Make sure text ends with sentence-terminator punctuation.

    Piper inflects pitch on a sentence-ending punctuation mark and inserts
    a ``--sentence-silence`` pause after it. Without a terminator, the
    next clause runs together with this one in the audio output.
    """
    text = text.rstrip()
    if text and text[-1] not in ".!?:;,":
        text += "."
    return text


def _walk(tokens: list[Token], overrides: dict[str, str]) -> str:
    """Walk the top-level token stream and build the narration string.

    Uses Piper-native pause cues only: sentence-terminator punctuation
    plus newlines. No SSML, since Piper doesn't parse it.
    """
    parts: list[str] = []
    consumed: set[int] = set()
    list_depth = 0

    for i, token in enumerate(tokens):
        if i in consumed:
            continue
        kind = token.type

        if kind == "heading_open":
            # Heading text becomes its own sentence with paragraph breaks
            # on both sides. (Leading break collapses with the previous
            # paragraph's trailing break; the collapse pass handles it.)
            consumed.add(i + 1)
            text = _apply_overrides(_render_inline(tokens[i + 1]), overrides)
            parts.append("\n\n")
            parts.append(_ensure_terminator(text))
            parts.append("\n\n")
        elif kind in ("bullet_list_open", "ordered_list_open"):
            list_depth += 1
        elif kind in ("bullet_list_close", "ordered_list_close"):
            list_depth -= 1
            # Paragraph break after the list ends so it's distinct from
            # whatever follows (will collapse with the last item's newline).
            parts.append("\n")
        elif kind in ("fence", "code_block", "hr"):
            # No narrated content, just a paragraph break to mark the
            # structural boundary.
            parts.append("\n\n")
        elif kind == "inline":
            text = _apply_overrides(_render_inline(token), overrides)
            parts.append(_ensure_terminator(text))
            # List items get a single newline (short beat between items).
            # Paragraphs get a paragraph break plus an em-dash empty
            # sentence: Piper applies --sentence-silence at every sentence
            # boundary uniformly, so emitting a non-vocalized sentence
            # between paragraphs doubles the boundary pause without
            # touching sentence-within-paragraph or list-item pacing.
            if list_depth > 0:
                parts.append("\n")
            else:
                parts.append(f"\n\n{_PARAGRAPH_PAUSE_MARKER}\n\n")

    return "".join(parts)


def _collapse_whitespace(text: str) -> str:
    """Tighten whitespace while preserving newlines as Piper pause cues."""
    # Collapse horizontal whitespace (spaces, tabs) to single space.
    text = _HSPACE_RE.sub(" ", text)
    # Cap consecutive newlines at 2 (paragraph break).
    text = _NEWLINE_COLLAPSE_RE.sub("\n\n", text)
    # Strip per-line whitespace to avoid whitespace-only lines or
    # trailing spaces before newlines.
    text = "\n".join(line.strip() for line in text.split("\n"))
    # Final outer strip removes any leading/trailing blank lines.
    text = text.strip()
    # A trailing paragraph-pause marker has no successor paragraph to
    # buffer against; trim it so the audio doesn't end on a stray
    # em-dash sentence.
    if text.endswith(_PARAGRAPH_PAUSE_MARKER):
        text = text[: -len(_PARAGRAPH_PAUSE_MARKER)].rstrip()
    return text


def markdown_to_narration(
    md_text: str,
    *,
    overrides: dict[str, str] | None = None,
) -> str:
    """Convert digest markdown into a Piper-friendly narration string.

    Args:
        md_text: Source markdown (typically the composer's
            ``{date}.md`` deliverable).
        overrides: Pronunciation overrides; falls back to the bundled
            ``narration_overrides.yaml`` when ``None``. Pass ``{}`` to
            disable overrides entirely.

    Returns:
        Plain text with newlines and sentence-terminator punctuation
        as pause cues, ready to feed to ``piper`` on stdin. Tune the
        global pause duration via ``settings.audio_sentence_silence_s``
        (Piper's ``--sentence-silence`` flag).
    """
    if overrides is None:
        overrides = load_overrides()

    stripped = _FRONTMATTER_RE.sub("", md_text, count=1)
    normalized = _normalize_pre_walk(stripped)
    md = MarkdownIt("commonmark")
    tokens = md.parse(normalized)
    raw = _walk(tokens, overrides)
    return _collapse_whitespace(raw)
