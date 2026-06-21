"""Per-feed persistence for the RSS fetch stage.

Writes / reads ``Entry`` batches under
``<run_dir>/source-fetched/<feed_name>.json``. The flat layout works
uniformly across content sources: source_type and content_type live
in each entry's data rather than the directory path.
"""

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from digest_generator.core.types import Entry
from digest_generator.shared.logging import logger

_FETCHED_DIR = "source-fetched"


def fetched_dir(run_dir: Path) -> Path:
    """Return the flat fetched directory under ``run_dir``."""
    return run_dir / _FETCHED_DIR


def fetched_path(run_dir: Path, feed_name: str) -> Path:
    """Return the canonical path for one feed's fetched batch."""
    return fetched_dir(run_dir) / f"{feed_name}.json"


def save_entries(run_dir: Path, feed_name: str, entries: list[Entry]) -> Path:
    """Serialize one feed's entries to JSON; return the written path.

    Schema (per entry): ``title``, ``url``, ``origin``, ``source_type``,
    ``published`` (ISO-8601), ``description``, ``content``, plus
    ``content_head`` / ``content_type`` / ``fetched_at`` only when set.
    Atomic via tempfile + rename.
    """
    target = fetched_path(run_dir, feed_name)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = [_serialize(e) for e in entries]

    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    tmp.replace(target)
    return target


def load_entries(run_dir: Path, feed_name: str) -> list[Entry] | None:
    """Read one feed's fetched JSON. Returns ``None`` when absent.

    Reconstructs ``Entry`` dataclasses (round-trip) so the summarizer /
    classifier can consume them directly without a dict-to-Entry adapter.
    Raises ``ValueError`` on malformed payloads.
    """
    target = fetched_path(run_dir, feed_name)
    if not target.exists():
        return None
    with target.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        msg = f"Fetched file {target} is not a JSON list"
        raise ValueError(msg)
    return [_deserialize(item) for item in data]


def iter_fetched(
    run_dir: Path,
) -> Iterator[tuple[str, str, list[Entry]]]:
    """Yield ``(content_type, feed_name, entries)`` for every fetched batch.

    Walks ``<run_dir>/source-fetched/*.json``. ``content_type`` is
    sourced from each entry's ``content_type`` field. Empty files and
    files whose entries lack content_type are skipped with a warning.
    Skips files that fail to parse, emitting WARNINGs so the caller can
    surface partial-corpus runs.
    """
    root = fetched_dir(run_dir)
    if not root.exists():
        return
    for json_path in sorted(root.glob("*.json")):
        try:
            with json_path.open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "iter_fetched: skipping {} ({}: {})",
                json_path,
                type(exc).__name__,
                exc,
            )
            continue
        if not isinstance(data, list):
            logger.warning("iter_fetched: {} is not a JSON list, skipping", json_path)
            continue
        try:
            entries = [_deserialize(item) for item in data]
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "iter_fetched: malformed entry in {} ({}: {})",
                json_path,
                type(exc).__name__,
                exc,
            )
            continue
        content_type = entries[0].content_type if entries else None
        if content_type is None:
            logger.warning(
                "iter_fetched: {} has no content_type in entries — skipping",
                json_path,
            )
            continue
        yield content_type, json_path.stem, entries


def _serialize(entry: Entry) -> dict[str, Any]:
    """Convert one ``Entry`` to JSON-safe shape."""
    payload: dict[str, Any] = {
        "title": entry.title,
        "url": entry.url,
        "origin": entry.origin,
        "source_type": entry.source_type,
        "published": entry.published.isoformat(),
        "description": entry.description,
        "content": entry.content,
    }
    if entry.content_head:
        payload["content_head"] = entry.content_head
    if entry.content_type is not None:
        payload["content_type"] = entry.content_type
    if entry.fetched_at is not None:
        payload["fetched_at"] = entry.fetched_at.isoformat()
    return payload


def _deserialize(item: dict[str, Any]) -> Entry:
    """Reconstruct an ``Entry`` from the JSON shape produced by ``_serialize``."""
    content_type = item.get("content_type")
    fetched_at_raw = item.get("fetched_at")
    fetched_at = datetime.fromisoformat(fetched_at_raw) if fetched_at_raw is not None else None
    return Entry(
        title=item["title"],
        url=item["url"],
        origin=item["origin"],
        published=datetime.fromisoformat(item["published"]),
        description=item["description"],
        content=item["content"],
        content_head=item.get("content_head", ""),
        source_type=item.get("source_type", "rss"),
        content_type=content_type,
        fetched_at=fetched_at,
    )
