"""RSS feed configuration: discovery and loading from ``feeds.yaml``.

Feeds are not hardcoded in the package. The user supplies a ``feeds.yaml``
(see ``feeds.example.yaml`` for the schema), which this module discovers
across a fixed search path and validates into a ``list[Feed]``.

Search order (first existing file wins):

1. An explicit feeds file (``--feeds`` / ``FEEDS_FILE``).
2. ``<config-dir>/feeds.yaml`` where the config dir comes from
   ``--config`` / ``DIGEST_CONFIG``.
3. ``./digest-generator/feeds.yaml`` (project-local).
4. ``~/.config/digest-generator/feeds.yaml`` (user-level).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from digest_generator.core.types import ContentType
from digest_generator.shared.settings import settings
from digest_generator.sources.rss.types import Feed

_CONFIG_FILENAME = "feeds.yaml"
_PROJECT_DIR = "digest-generator"
_USER_CONFIG_DIR = Path.home() / ".config" / "digest-generator"


class FeedsConfigError(ValueError):
    """Raised when the feeds configuration is missing or invalid."""


class _FeedEntry(BaseModel):
    """One feed row in ``feeds.yaml``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    category: ContentType


class _FeedsDoc(BaseModel):
    """Top-level ``feeds.yaml`` document.

    Unknown top-level keys are ignored so a config carrying forward-looking
    sections (e.g. a future ``categories:`` block) still loads.
    """

    model_config = ConfigDict(extra="ignore")

    feeds: list[_FeedEntry] = Field(min_length=1)


def candidate_paths(
    feeds_file: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> list[Path]:
    """Return the ordered feeds-file search path for the given overrides.

    Includes every candidate regardless of existence; callers pick the
    first one that exists.
    """
    paths: list[Path] = []
    if feeds_file:
        paths.append(Path(feeds_file).expanduser())
    if config_dir:
        paths.append(Path(config_dir).expanduser() / _CONFIG_FILENAME)
    paths.append(Path.cwd() / _PROJECT_DIR / _CONFIG_FILENAME)
    paths.append(_USER_CONFIG_DIR / _CONFIG_FILENAME)
    return paths


def discover_feeds_file(
    feeds_file: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> Path | None:
    """Return the first existing feeds file in the search path, or ``None``."""
    for path in candidate_paths(feeds_file, config_dir):
        if path.is_file():
            return path
    return None


def load_feeds(path: Path) -> list[Feed]:
    """Parse and validate a ``feeds.yaml`` file into a list of ``Feed``.

    Args:
        path: Path to a ``feeds.yaml`` file.

    Returns:
        The feeds declared in the file, in document order.

    Raises:
        FeedsConfigError: If the file is unreadable, empty, malformed,
            declares an unknown category, or repeats a feed name.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"Cannot read feeds config at {path}: {exc}"
        raise FeedsConfigError(msg) from exc
    except yaml.YAMLError as exc:
        msg = f"Invalid YAML in feeds config at {path}: {exc}"
        raise FeedsConfigError(msg) from exc

    if raw is None:
        msg = f"Feeds config at {path} is empty. See feeds.example.yaml."
        raise FeedsConfigError(msg)

    try:
        doc = _FeedsDoc.model_validate(raw)
    except ValidationError as exc:
        valid = ", ".join(ct.value for ct in ContentType)
        msg = (
            f"Invalid feeds config at {path}:\n{exc}\n"
            f"Each feed needs name, url, and a category (one of: {valid})."
        )
        raise FeedsConfigError(msg) from exc

    names = [entry.name for entry in doc.feeds]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        msg = f"Duplicate feed name(s) in {path}: {', '.join(duplicates)}"
        raise FeedsConfigError(msg)

    return [
        Feed(name=entry.name, url=entry.url, content_type=entry.category) for entry in doc.feeds
    ]


def load_configured_feeds(
    feeds_file: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> list[Feed]:
    """Discover and load the active feeds configuration.

    Falls back to ``settings.feeds_file`` / ``settings.digest_config`` when
    the corresponding argument is ``None``, then walks the search path.

    Raises:
        FeedsConfigError: If no feeds file is found, or the resolved file
            is invalid.
    """
    feeds_file = feeds_file or settings.feeds_file or None
    config_dir = config_dir or settings.digest_config or None

    path = discover_feeds_file(feeds_file, config_dir)
    if path is None:
        searched = "\n  ".join(str(p) for p in candidate_paths(feeds_file, config_dir))
        msg = (
            "No feeds.yaml found. Searched:\n  "
            f"{searched}\n"
            "Copy feeds.example.yaml to one of these locations (or pass "
            "--feeds / --config) and add your feeds."
        )
        raise FeedsConfigError(msg)
    return load_feeds(path)
