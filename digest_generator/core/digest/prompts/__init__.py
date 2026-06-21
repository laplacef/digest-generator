"""Prompt template loader for digest-pipeline LLM stages.

Templates live as ``.md`` files under ``templates/``. Each stage imports its
prompt via ``load_prompt("name")`` at module import time, so file I/O and
placeholder resolution happen once per process.

Templates may reference shared style fragments via ``{{style:CATEGORY}}``
placeholders; see ``digest_generator.core.style`` for the catalogue (shared with the
LLM-driven summarizer).
"""

from pathlib import Path

from digest_generator.core.style import expand_style_placeholders

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def load_prompt(name: str) -> str:
    """Load a prompt template by name (without the ``.md`` extension)."""
    raw = (_TEMPLATES_DIR / f"{name}.md").read_text(encoding="utf-8")
    return expand_style_placeholders(raw)
