"""Prompt template loader for the article summarizer.

Mirrors the digest sub-package's ``digest_generator.core.digest.prompts``:
templates live as ``.md`` files under ``templates/``; the summarizer
imports its prompt via ``load_prompt("name")`` at module import time so
file I/O and ``{{style:CATEGORY}}`` placeholder resolution happen once
per process.

The style catalogue lives at ``digest_generator.core.style`` (shared with the
digest prompts); adding a forbidden phrase there propagates to both
this package's templates and the digest templates on the next process load.
"""

from pathlib import Path

from digest_generator.core.style import expand_style_placeholders

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def load_prompt(name: str) -> str:
    """Load a prompt template by name (without the ``.md`` extension)."""
    raw = (_TEMPLATES_DIR / f"{name}.md").read_text(encoding="utf-8")
    return expand_style_placeholders(raw)
