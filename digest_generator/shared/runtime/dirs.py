"""Run-output directory creation."""

import secrets
from datetime import UTC, datetime
from pathlib import Path

from digest_generator.shared.settings import settings


def create_run_dir(output_dir: Path | None = None) -> Path:
    """Create a unique, timestamped run directory for pipeline output.

    The directory name combines a UTC timestamp with a short random suffix so
    concurrent runs (e.g. a cron trigger racing with a manual CLI invocation)
    never share a directory. The timestamp keeps names sortable and
    human-meaningful, the suffix guarantees uniqueness.

    Structure:
        output/{YYYY-MM-DD-HHmmss-xxxx}/
            source-fetched/<feed>.json
            source-summarized/<feed>.json
            source-labeled/<feed>.json
            section-drafts/<section>.json
            section-edits/<section>.json
            assembly/{clusters,framing,watch}.json
            audio/{date}.opus, audio/cache_key.txt
            {date}.md
            meta.json
            run.log

    Sub-directories are created lazily by each stage's ``io.py``;
    ``create_run_dir`` only mints the empty run root.

    Args:
        output_dir: Root output directory (default: output/).

    Returns:
        Path to the created run directory.
    """
    output_dir = output_dir or Path(settings.output_dir)
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d-%H%M%S")
    suffix = secrets.token_hex(2)
    run_dir = output_dir / f"{timestamp}-{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
