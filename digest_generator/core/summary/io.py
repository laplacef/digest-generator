"""Per-feed persistence for the summary stage.

Writes ``Summary`` batches (without tags) to
``<run_dir>/source-summarized/<feed>.json``. Topics live in
``source-labeled/<feed>.json`` and join back in via
``api._load_digest_input`` at digest read time.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from digest_generator.core.types import Summary
from digest_generator.shared.logging import logger

_SUMMARIZED_DIR = "source-summarized"


def summarized_dir(run_dir: Path) -> Path:
    """Return the flat summarized directory under ``run_dir``."""
    return run_dir / _SUMMARIZED_DIR


def summarized_path(run_dir: Path, feed_name: str) -> Path:
    """Return the canonical path for one feed's summarized batch."""
    return summarized_dir(run_dir) / f"{feed_name}.json"


def save_summarized(run_dir: Path, feed_name: str, summaries: list[Summary]) -> Path:
    """Serialize one feed's summaries (no tags) to JSON; return the written path.

    Schema (per article): ``title``, ``url``, ``origin``, ``source_type``,
    ``published`` (ISO-8601), ``description``, ``summary``,
    ``summary_length``, plus ``content_head`` / ``content_type`` /
    ``fetched_at`` only when set. Atomic via tempfile + rename.

    Topics from ``Summary.topics`` are NOT serialized; they live in
    the ``source-labeled/`` artifact and join back in via
    ``api._load_digest_input`` at digest read time.
    """
    target = summarized_path(run_dir, feed_name)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = [_serialize(s) for s in summaries]

    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    tmp.replace(target)
    return target


def load_summarized(run_dir: Path, feed_name: str) -> list[dict[str, Any]] | None:
    """Read one feed's summarized JSON. Returns ``None`` when absent."""
    target = summarized_path(run_dir, feed_name)
    if not target.exists():
        return None
    with target.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        msg = f"Summarized file {target} is not a JSON list"
        raise ValueError(msg)
    return data


def iter_summarized(
    run_dir: Path,
) -> Iterator[tuple[str, str, list[dict[str, Any]]]]:
    """Yield ``(source_type, feed_name, articles)`` for every summarized batch.

    Walks ``<run_dir>/source-summarized/*.json``. ``source_type``
    comes from each record's ``source_type`` field; defaults to
    ``"rss"`` for records that lack it. Skips files that fail to
    parse, emitting a WARNING so the caller can surface partial-corpus
    runs.
    """
    root = summarized_dir(run_dir)
    if not root.exists():
        return
    for json_path in sorted(root.glob("*.json")):
        try:
            with json_path.open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "iter_summarized: skipping {} ({}: {})",
                json_path,
                type(exc).__name__,
                exc,
            )
            continue
        if not isinstance(data, list):
            logger.warning("iter_summarized: {} is not a JSON list, skipping", json_path)
            continue
        source_type = data[0].get("source_type", "rss") if data else "rss"
        yield source_type, json_path.stem, data


def _serialize(summary: Summary) -> dict[str, Any]:
    """Convert one ``Summary`` to the summarized-stage schema.

    Identity fields (``source_type``, ``content_type``, ``fetched_at``,
    ``origin``) carry forward from the underlying ``Entry`` so the
    summarized record preserves its provenance.
    """
    payload: dict[str, Any] = {
        "title": summary.entry.title,
        "url": summary.entry.url,
        "origin": summary.entry.origin,
        "source_type": summary.entry.source_type,
        "published": summary.entry.published.isoformat(),
        "description": summary.entry.description,
        "summary": summary.summary,
        "summary_length": summary.length,
    }
    if summary.entry.content_head:
        payload["content_head"] = summary.entry.content_head
    if summary.entry.content_type is not None:
        payload["content_type"] = summary.entry.content_type.value
    if summary.entry.fetched_at is not None:
        payload["fetched_at"] = summary.entry.fetched_at.isoformat()
    return payload
