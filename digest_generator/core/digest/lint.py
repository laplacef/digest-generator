"""Importable golden-output lint over a generated digest.

Regex-based structural and phrase checks on assembled digest markdown. The
per-section editor polishes prose within a section and has no whole-document
view, so a class of defect survives every LLM stage: leaked internal citation
keys, whole sections with no citations, H1s in the body, per-section "what to
watch" blocks, fabricated or malformed link targets, and mechanical residue the
normalizer should have removed. Each maps to a deterministic pattern, so a lint
is a more reliable safety net than re-reading by eye every publish cycle.

``lint_digest`` is a plain function, not test-internal logic, precisely so
three callers share one implementation: the pytest regression suite (against a
committed fixture), the eventual prompt eval harness (which needs a
programmatic "is this output OK?" signal), and a human running it against
``output/<run>/<date>.md`` before publishing.

Findings carry a severity. ``error`` marks something always wrong (a leaked
citation key, an H1 in the body, mechanical residue); ``warning`` marks a
strong tell that occasionally has a legitimate reading (a prose cliche, a
citation-light section). ``has_errors`` gates CI on the former only.

Pattern sources are shared, never duplicated: forbidden phrases and title
shapes come from ``digest_generator.core.style``; the code/link masking that
keeps the mechanical scans off code tokens and link targets comes from
``digest_generator.core.digest.normalize``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from digest_generator.core.digest.normalize import mask_protected_spans
from digest_generator.core.style import TITLE_PATTERNS, iter_lintable_phrases

ERROR = "error"
WARNING = "warning"

_CITATION_KEY_RE = re.compile(r"\(c\d{4}\)")
_H1_RE = re.compile(r"(?m)^# \S")
_H2_RE = re.compile(r"(?m)^## (.+?)\s*$")
_WATCH_HEADING_RE = re.compile(r"(?im)^(#{2,4})\s+what to\s+watch\b")
_MD_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")
# Capture the destination of every inline link so link targets can be checked
# independently of the surrounding prose. Deliberately permissive about what
# sits inside the parens: a malformed destination is exactly what this finds.
_MD_LINK_TARGET_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
# A well-formed inline destination: one whitespace-free target, optionally
# followed by a CommonMark quoted title. Anything else containing whitespace
# is malformed, and the space silently terminates the destination at render
# time, so the whole citation publishes as literal bracket text.
_WELL_FORMED_TARGET_RE = re.compile(r'^\S+(?:\s+"[^"]*")?$')
_HTTP_URL_RE = re.compile(r"(?i)^https?://")
# Query parameters that carry campaign/referrer attribution rather than
# selecting content, so dropping them cannot change the page addressed.
_TRACKING_PARAM_RE = re.compile(r"(?i)^(utm_[a-z_]+|fbclid|gclid|mc_cid|mc_eid|ref|source)$")

# Mechanical residue the normalizer removes; a hit means the normalizer did not
# run or a new case slipped past it. ``(label, regex)`` — scanned over masked
# prose so a URL or code token never trips them.
_RESIDUE_CHECKS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("percent-with-space", re.compile(r"\d\s+%")),
    ("scale-abbreviation", re.compile(r"\d\s?(?:bn|trn|tn|mn)\b")),
    ("thousands-suffix", re.compile(r"(?:\b\d+(?:\.\d+)?\s+k\b|[$£€¥]\d+(?:\.\d+)?k\b)")),
    ("space-grouped-thousands", re.compile(r"\b\d{1,3}(?: \d{3})+\b")),
    ("latex-span", re.compile(r"\$[A-Za-z\\][^$]{0,60}?[=\\][^$]*\$")),
    ("multiplication-sign", re.compile(r"\d\s*×")),  # noqa: RUF001 (literal U+00D7 we match on)
)

# Sections that are structural framing, not article write-ups, so the
# per-section citation floor does not apply to them.
_NON_ARTICLE_SECTIONS = {"overview", "what to watch"}


@dataclass(frozen=True)
class LintFinding:
    """One lint hit. ``line`` is 1-indexed into the full markdown (0 = N/A)."""

    category: str
    severity: str
    line: int
    message: str
    snippet: str


def _line_of(text: str, index: int) -> int:
    """1-indexed line number of a character offset into ``text``."""
    return text.count("\n", 0, index) + 1


def _split_frontmatter(markdown: str) -> tuple[dict[str, str], str, int]:
    """Return (frontmatter fields, body, body_line_offset).

    Only the flat scalar fields the lint needs (``title``, ``summary``) are
    parsed — this is not a general YAML reader. ``body_line_offset`` is the
    number of lines the frontmatter occupied so body findings can report a
    line number into the original document.
    """
    if not markdown.startswith("---"):
        return {}, markdown, 0
    match = re.match(r"---\n(.*?)\n---\n?", markdown, re.DOTALL)
    if not match:
        return {}, markdown, 0
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        field = re.match(r"(\w+):\s*(.*)$", line)
        if field:
            fields[field.group(1)] = field.group(2).strip().strip('"')
    body = markdown[match.end() :]
    offset = markdown.count("\n", 0, match.end())
    return fields, body, offset


def _check_title(fields: dict[str, str]) -> list[LintFinding]:
    title = fields.get("title", "")
    findings: list[LintFinding] = []
    for pattern, reason in TITLE_PATTERNS:
        if re.search(pattern, title):
            findings.append(LintFinding("title-shape", ERROR, 1, f"Title {reason}.", title))
    return findings


def _check_summary(fields: dict[str, str]) -> list[LintFinding]:
    summary = fields.get("summary", "")
    if summary.endswith(("...", "…")):
        return [
            LintFinding(
                "summary-truncated",
                WARNING,
                1,
                "Frontmatter summary ends mid-sentence; it is the meta description "
                "and social-card blurb and should be a self-contained sentence.",
                summary,
            )
        ]
    return []


def _check_citation_keys(body: str, offset: int) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for m in _CITATION_KEY_RE.finditer(body):
        findings.append(
            LintFinding(
                "citation-key-leak",
                ERROR,
                offset + _line_of(body, m.start()),
                f"Internal citation key {m.group(0)!r} leaked into published prose.",
                m.group(0),
            )
        )
    return findings


def _check_structure(body: str, offset: int) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for m in _H1_RE.finditer(body):
        findings.append(
            LintFinding(
                "h1-in-body",
                ERROR,
                offset + _line_of(body, m.start()),
                "H1 heading in body; the title belongs in frontmatter only, so an "
                "H1 renders the title twice.",
                body[m.start() : m.start() + 40].splitlines()[0],
            )
        )
    watch_headings = list(_WATCH_HEADING_RE.finditer(body))
    for m in watch_headings:
        # The one legitimate "What to Watch" is a terminal H2. Any H3+ or any
        # second H2 is a per-section block the watcher stage should own instead.
        if len(m.group(1)) != 2 or m is not watch_headings[-1]:
            findings.append(
                LintFinding(
                    "per-section-watch",
                    ERROR,
                    offset + _line_of(body, m.start()),
                    "Per-section 'what to watch' block; forward-looking items belong "
                    "only in the single terminal '## What to Watch' section.",
                    body[m.start() : m.start() + 40].splitlines()[0],
                )
            )
    return findings


def _iter_sections(body: str) -> list[tuple[str, str, int]]:
    """Yield (heading_text, section_body, heading_char_offset) per H2 block."""
    headings = list(_H2_RE.finditer(body))
    sections: list[tuple[str, str, int]] = []
    for i, m in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        sections.append((m.group(1).strip(), body[m.end() : end], m.start()))
    return sections


def _check_section_citations(body: str, offset: int) -> list[LintFinding]:
    """Flag a multi-paragraph article section carrying zero markdown links.

    Link detection runs on the raw section body, not the masked one: masking
    blanks links (they are protected spans), which would defeat the very thing
    being detected. A link literally inside inline code counting as a citation
    is a near-zero-risk edge accepted for that simplicity.
    """
    findings: list[LintFinding] = []
    for heading, section_body, start in _iter_sections(body):
        if heading.lower() in _NON_ARTICLE_SECTIONS:
            continue
        paragraphs = [p for p in section_body.split("\n\n") if p.strip()]
        if len(paragraphs) < 2:
            continue
        if not _MD_LINK_RE.search(section_body):
            findings.append(
                LintFinding(
                    "section-no-citations",
                    WARNING,
                    offset + _line_of(body, start),
                    f"Section {heading!r} has {len(paragraphs)} paragraphs and no "
                    "citations; every sourced claim should carry a markdown link.",
                    heading,
                )
            )
    return findings


def canonical_url(url: str) -> str:
    """Fold a URL to a comparison key for corpus membership.

    Folds only what cannot change *which article* a link addresses, since the
    check exists to catch a writer citing a different (usually nonexistent)
    page. Specifically: scheme and host are lowercased, a leading ``www.`` is
    dropped, the fragment goes, feed-tracking query parameters are removed,
    and a trailing slash is trimmed.

    The tracking-parameter fold is load-bearing rather than cosmetic. RSS feeds
    routinely append ``utm_*`` to the canonical link, so the corpus holds the
    tagged form while a writer sensibly cites the clean one. Comparing raw
    strings would flag every such citation as fabricated, and a check that
    cries wolf on correct output is worse than no check.

    The path is left cased and otherwise untouched: a changed path is exactly
    the defect being detected.
    """
    url = url.strip().split("#", 1)[0]
    match = re.match(r"(?i)^(https?://)([^/]+)(.*)$", url)
    if match:
        scheme, host, rest = match.groups()
        host = host.lower().removeprefix("www.")
        url = scheme.lower() + host + rest
    if "?" in url:
        path, _, query = url.partition("?")
        kept = [
            param
            for param in query.split("&")
            if param and not _TRACKING_PARAM_RE.match(param.split("=", 1)[0])
        ]
        url = f"{path}?{'&'.join(kept)}" if kept else path
    return url.rstrip("/")


def _check_link_targets(
    body: str, offset: int, known_urls: frozenset[str] | None
) -> list[LintFinding]:
    """Flag malformed link destinations, and unknown ones when a corpus is given.

    Two distinct defects, both of which survive every LLM stage because the
    editorial pass diffs against the *writer's* output and so preserves a
    fabricated target faithfully:

    - A destination containing unquoted whitespace. Always an error: the space
      terminates the destination, so the citation renders as literal text
      rather than a link, and a link checker that tokenizes on whitespace
      cannot see it either.
    - A destination absent from the run's fetched corpus, which means the
      writer invented or mutated a slug. Only checked when ``known_urls`` is
      supplied, since a caller whose prompts permit citing sources outside the
      corpus should simply not pass one.
    """
    findings: list[LintFinding] = []
    for m in _MD_LINK_TARGET_RE.finditer(body):
        target = m.group(1)
        line = offset + _line_of(body, m.start())
        if not _WELL_FORMED_TARGET_RE.match(target.strip()):
            findings.append(
                LintFinding(
                    "link-target-malformed",
                    ERROR,
                    line,
                    "Link destination contains unquoted whitespace; it will render "
                    "as literal text instead of a link.",
                    target,
                )
            )
            continue
        if known_urls is None:
            continue
        url = target.strip().split()[0]
        if not _HTTP_URL_RE.match(url):
            continue
        if canonical_url(url) not in known_urls:
            findings.append(
                LintFinding(
                    "link-target-unknown",
                    ERROR,
                    line,
                    "Link destination is not in the run's fetched corpus, so the "
                    "slug was invented or mutated rather than cited verbatim.",
                    url,
                )
            )
    return findings


def _check_residue(masked_body: str, body: str, offset: int) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for label, pattern in _RESIDUE_CHECKS:
        for m in pattern.finditer(masked_body):
            findings.append(
                LintFinding(
                    f"residue-{label}",
                    ERROR,
                    offset + _line_of(body, m.start()),
                    f"Un-normalized {label} in prose: {body[m.start() : m.end()]!r}.",
                    body[m.start() : m.end()],
                )
            )
    return findings


def _check_phrases(masked_body: str, body: str, offset: int) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for category, phrase in iter_lintable_phrases():
        if phrase.pattern is None:  # guaranteed non-None, but narrows for the type checker
            continue
        for m in re.finditer(phrase.pattern, masked_body):
            findings.append(
                LintFinding(
                    f"phrase-{category}",
                    WARNING,
                    offset + _line_of(body, m.start()),
                    f"Forbidden phrase ({category}): {body[m.start() : m.end()]!r}.",
                    body[m.start() : m.end()].strip(),
                )
            )
    return findings


def lint_digest(markdown: str, *, known_urls: frozenset[str] | None = None) -> list[LintFinding]:
    """Lint assembled digest markdown; return findings sorted by line.

    Runs structural, mechanical, and phrase checks. Mechanical and phrase scans
    operate on a length-preserving mask of the body (code / links / URLs
    blanked) so a defect inside a code token or a link target never trips them,
    while reported line numbers stay exact.

    Args:
        markdown: Assembled digest, frontmatter included.
        known_urls: Canonicalized URLs of the run's fetched articles, as
            produced by ``canonical_url``. When given, every http(s) link
            target is checked for membership so an invented slug is caught;
            when omitted, only malformed targets are flagged. Keeping this
            optional is what lets the function stay a pure string check with
            no dependency on the source layer — the caller loads the corpus.
    """
    fields, body, offset = _split_frontmatter(markdown)
    masked_body = mask_protected_spans(body)

    findings: list[LintFinding] = []
    findings += _check_title(fields)
    findings += _check_summary(fields)
    findings += _check_citation_keys(body, offset)
    findings += _check_structure(body, offset)
    findings += _check_section_citations(body, offset)
    findings += _check_link_targets(body, offset, known_urls)
    findings += _check_residue(masked_body, body, offset)
    findings += _check_phrases(masked_body, body, offset)
    return sorted(findings, key=lambda f: (f.line, f.category))


def has_errors(findings: list[LintFinding]) -> bool:
    """True if any finding is an ``error`` (the CI-gating severity)."""
    return any(f.severity == ERROR for f in findings)


def format_findings(findings: list[LintFinding]) -> str:
    """Render findings as one ``line:severity:category  message`` block."""
    if not findings:
        return "No lint findings."
    return "\n".join(f"L{f.line}:{f.severity}:{f.category}  {f.message}" for f in findings)
