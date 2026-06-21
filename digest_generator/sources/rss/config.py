"""RSS feed configuration: discovery and loading from ``feeds.yaml``.

Feeds and categories are not hardcoded in the package. The user supplies a
``feeds.yaml`` (see ``feeds.example.yaml`` for the schema) with a
``categories:`` block (the digest sections, in order) and a ``feeds:`` block.
This module discovers that file across a fixed search path and validates it
into a ``CategorySet`` plus a ``list[Feed]``.

Search order (first existing file wins):

1. An explicit feeds file (``--feeds`` / ``FEEDS_FILE``).
2. ``<config-dir>/feeds.yaml`` where the config dir comes from
   ``--config`` / ``DIGEST_CONFIG``.
3. ``./digest-generator/feeds.yaml`` (project-local).
4. ``~/.config/digest-generator/feeds.yaml`` (user-level).
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from digest_generator.core.categories import Category, CategorySet
from digest_generator.shared.settings import settings
from digest_generator.sources.rss.types import Feed

_CONFIG_FILENAME = "feeds.yaml"
_EXAMPLE_FILENAME = "feeds.example.yaml"
_PROJECT_DIR = "digest-generator"
_USER_CONFIG_DIR = Path.home() / ".config" / "digest-generator"


class FeedsConfigError(ValueError):
    """Raised when the feeds configuration is missing or invalid."""


class _CategoryEntry(BaseModel):
    """One category row in the ``categories:`` block of ``feeds.yaml``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)


class _FeedEntry(BaseModel):
    """One feed row in ``feeds.yaml``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    category: str = Field(min_length=1)


class _FeedsDoc(BaseModel):
    """Top-level ``feeds.yaml`` document: categories plus feeds.

    Unknown top-level keys are ignored so a config carrying forward-looking
    sections still loads.
    """

    model_config = ConfigDict(extra="ignore")

    categories: list[_CategoryEntry] = Field(min_length=1)
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


def load_config(path: Path) -> tuple[CategorySet, list[Feed]]:
    """Parse and validate a ``feeds.yaml`` file into categories and feeds.

    Categories and feeds are interdependent (every feed's ``category`` must
    name a defined category), so they load together.

    Args:
        path: Path to a ``feeds.yaml`` file.

    Returns:
        The ``CategorySet`` (in section order) and the feeds (in document
        order).

    Raises:
        FeedsConfigError: If the file is unreadable, empty, malformed,
            repeats a category id or feed name, or a feed names an unknown
            category.
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
        msg = (
            f"Invalid feeds config at {path}:\n{exc}\n"
            "Needs a categories list (id + title) and a feeds list "
            "(name, url, category)."
        )
        raise FeedsConfigError(msg) from exc

    category_ids = [entry.id for entry in doc.categories]
    duplicate_categories = sorted({cid for cid in category_ids if category_ids.count(cid) > 1})
    if duplicate_categories:
        msg = f"Duplicate category id(s) in {path}: {', '.join(duplicate_categories)}"
        raise FeedsConfigError(msg)

    names = [entry.name for entry in doc.feeds]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        msg = f"Duplicate feed name(s) in {path}: {', '.join(duplicate_names)}"
        raise FeedsConfigError(msg)

    known = set(category_ids)
    unknown = sorted({entry.category for entry in doc.feeds if entry.category not in known})
    if unknown:
        valid = ", ".join(category_ids)
        msg = (
            f"Feed(s) in {path} name unknown categor(ies): {', '.join(unknown)}. "
            f"Defined categories: {valid}."
        )
        raise FeedsConfigError(msg)

    categories = CategorySet([Category(id=entry.id, title=entry.title) for entry in doc.categories])
    feeds = [
        Feed(name=entry.name, url=entry.url, content_type=entry.category) for entry in doc.feeds
    ]
    return categories, feeds


def load_feeds(path: Path) -> list[Feed]:
    """Load just the feeds from a ``feeds.yaml`` file. See ``load_config``."""
    return load_config(path)[1]


def load_categories(path: Path) -> CategorySet:
    """Load just the categories from a ``feeds.yaml`` file. See ``load_config``."""
    return load_config(path)[0]


def _resolve_config_path(
    feeds_file: str | Path | None,
    config_dir: str | Path | None,
) -> Path:
    """Discover the active feeds.yaml or raise an actionable not-found error."""
    feeds_file = feeds_file or settings.feeds_file or None
    config_dir = config_dir or settings.digest_config or None

    path = discover_feeds_file(feeds_file, config_dir)
    if path is None:
        searched = "\n  ".join(str(p) for p in candidate_paths(feeds_file, config_dir))
        msg = (
            "No feeds.yaml found. Searched:\n  "
            f"{searched}\n"
            "Run 'digest-generator init' to create a starter feeds.yaml, "
            "or pass --feeds / --config to point at an existing one."
        )
        raise FeedsConfigError(msg)
    return path


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
    return load_feeds(_resolve_config_path(feeds_file, config_dir))


def load_configured_categories(
    feeds_file: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> CategorySet:
    """Discover and load the active category set. See ``load_configured_feeds``."""
    return load_categories(_resolve_config_path(feeds_file, config_dir))


def example_feeds_text() -> str:
    """Return the bundled starter ``feeds.yaml`` content.

    Reads the copy packaged into the wheel, or, in a source checkout where it
    is not packaged, the repo-root ``feeds.example.yaml``.
    """
    packaged = files("digest_generator") / _EXAMPLE_FILENAME
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    root = Path(__file__).resolve().parents[3] / _EXAMPLE_FILENAME
    return root.read_text(encoding="utf-8")


def write_starter_feeds(
    *,
    feeds_file: str | Path | None = None,
    config_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Write a starter ``feeds.yaml`` and return the path written.

    The target is the explicit ``feeds_file``, else ``<config_dir>/feeds.yaml``,
    else ``~/.config/digest-generator/feeds.yaml``.

    Raises:
        FeedsConfigError: If the target already exists and ``force`` is False.
    """
    if feeds_file:
        target = Path(feeds_file).expanduser()
    elif config_dir:
        target = Path(config_dir).expanduser() / _CONFIG_FILENAME
    else:
        target = _USER_CONFIG_DIR / _CONFIG_FILENAME

    if target.exists() and not force:
        msg = f"{target} already exists. Pass --force to overwrite it."
        raise FeedsConfigError(msg)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(example_feeds_text(), encoding="utf-8")
    return target
