"""Prompt template loader for the article summarizer.

Mirrors the digest sub-package's ``digest_generator.core.digest.prompts``:
templates live as ``.md`` files under ``templates/``; the summarizer
imports its prompt via ``load_prompt("name")`` at module import time so
file I/O and ``{{style:CATEGORY}}`` placeholder resolution happen once
per process.

The style catalogue lives at ``digest_generator.core.style`` (shared with the
digest prompts); adding a forbidden phrase there propagates to both
this package's templates and the digest templates on the next process load.
A user can override the bundled template by dropping a ``<name>.md`` into a
prompts directory; see ``digest_generator.core.prompt_loader`` for the order.
"""

from pathlib import Path

from digest_generator.core.prompt_loader import resolve_prompt

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def load_prompt(name: str) -> str:
    """Load a prompt template by name (override dir first, else bundled)."""
    return resolve_prompt(name, bundled_dir=_TEMPLATES_DIR)
