"""Prompt template loader for digest-pipeline LLM stages.

Templates live as ``.md`` files under ``templates/``. Each stage imports its
prompt via ``load_prompt("name")`` at module import time, so file I/O and
placeholder resolution happen once per process.

Templates may reference shared style fragments via ``{{style:CATEGORY}}``
placeholders; see ``digest_generator.core.style`` for the catalogue (shared with the
LLM-driven summarizer). A user can override any template by dropping a
``<name>.md`` into a prompts directory; see
``digest_generator.core.prompt_loader`` for the override search order.
"""

from pathlib import Path

from digest_generator.core.prompt_loader import resolve_prompt

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def load_prompt(name: str) -> str:
    """Load a prompt template by name (override dir first, else bundled)."""
    return resolve_prompt(name, bundled_dir=_TEMPLATES_DIR)
