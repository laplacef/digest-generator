"""Shared prompt resolution: a user override directory, else the bundled template.

Both prompt packages (``core/digest/prompts`` and ``core/summary/prompts``)
ship bundled templates under their own ``templates/`` dir and delegate here.
A user can override any subset by dropping ``<name>.md`` files into one
override directory: ``PROMPTS_DIR`` when set, else ``<config dir>/prompts/``
(``DIGEST_CONFIG``, else ``~/.config/digest-generator``). A template missing
there uses the bundled baseline. Directories are never layered and the working
directory is never consulted, so every template comes from one known source.

Style placeholders are expanded on whichever template is used, so overrides
can subscribe to the shared catalog too.
"""

from __future__ import annotations

from pathlib import Path

from digest_generator.core.style import expand_style_placeholders
from digest_generator.shared.settings import settings

_PROMPTS_SUBDIR = "prompts"


def override_dir() -> Path:
    """Return the prompt-override directory: ``PROMPTS_DIR``, else ``<config dir>/prompts``."""
    if settings.prompts_dir:
        return Path(settings.prompts_dir).expanduser()
    return settings.config_dir / _PROMPTS_SUBDIR


def resolve_prompt(name: str, *, bundled_dir: Path) -> str:
    """Load a prompt template: a user override if present, else the bundled one.

    Args:
        name: Template name without the ``.md`` extension.
        bundled_dir: The package's bundled ``templates/`` directory.

    Returns:
        The resolved template text with ``{{style:...}}`` placeholders expanded.
    """
    override = override_dir() / f"{name}.md"
    if override.is_file():
        return expand_style_placeholders(override.read_text(encoding="utf-8"))
    raw = (bundled_dir / f"{name}.md").read_text(encoding="utf-8")
    return expand_style_placeholders(raw)
