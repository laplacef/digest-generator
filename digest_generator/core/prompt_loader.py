"""Shared prompt resolution: a user override directory, else the bundled template.

Both prompt packages (``core/digest/prompts`` and ``core/summary/prompts``)
ship bundled templates under their own ``templates/`` dir and delegate here.
A user can override any subset by dropping ``<name>.md`` files into a prompts
directory; templates not overridden fall back to the bundled baseline.

Override search order (first directory containing ``<name>.md`` wins):

1. ``PROMPTS_DIR`` (``settings.prompts_dir``).
2. ``<DIGEST_CONFIG>/prompts/`` when ``DIGEST_CONFIG`` is set.
3. ``./digest-generator/prompts/`` (project-local).
4. ``~/.config/digest-generator/prompts/`` (user-level).

Style placeholders are expanded on whichever template is used, so overrides
can subscribe to the shared catalogue too.
"""

from __future__ import annotations

from pathlib import Path

from digest_generator.core.style import expand_style_placeholders
from digest_generator.shared.settings import settings

_PROMPTS_SUBDIR = "prompts"
_PROJECT_DIR = "digest-generator"
_USER_CONFIG_DIR = Path.home() / ".config" / "digest-generator"


def override_dirs() -> list[Path]:
    """Return the ordered prompt-override search directories."""
    dirs: list[Path] = []
    if settings.prompts_dir:
        dirs.append(Path(settings.prompts_dir).expanduser())
    if settings.digest_config:
        dirs.append(Path(settings.digest_config).expanduser() / _PROMPTS_SUBDIR)
    dirs.append(Path.cwd() / _PROJECT_DIR / _PROMPTS_SUBDIR)
    dirs.append(_USER_CONFIG_DIR / _PROMPTS_SUBDIR)
    return dirs


def resolve_prompt(name: str, *, bundled_dir: Path) -> str:
    """Load a prompt template: a user override if present, else the bundled one.

    Args:
        name: Template name without the ``.md`` extension.
        bundled_dir: The package's bundled ``templates/`` directory.

    Returns:
        The resolved template text with ``{{style:...}}`` placeholders expanded.
    """
    for directory in override_dirs():
        candidate = directory / f"{name}.md"
        if candidate.is_file():
            return expand_style_placeholders(candidate.read_text(encoding="utf-8"))
    raw = (bundled_dir / f"{name}.md").read_text(encoding="utf-8")
    return expand_style_placeholders(raw)
