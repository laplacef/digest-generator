"""Filesystem I/O for the audio renderer: cache key, output paths.

Slots into the run layout as one more peer subdirectory:
``output/{run_dir}/audio/`` holds the cache key sentinel and the
deliverable ``.opus``. The slug-matching naming policy means
``audio/{date}-{slug}.opus`` shares its stem with the deliverable
``{date}-{slug}.md`` in the run root, so a downstream publish
step can ``cp`` both without a rename step.

Cache key contract: SHA-256 over the digest markdown bytes, the voice
id, the bitrate, the sentence-silence value, and the narration-version
constant. Any change in any of those invalidates the cache and forces
a re-render.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from digest_generator.core.audio.narration import NARRATION_VERSION

__all__ = [
    "audio_dir",
    "cache_key_path",
    "compute_cache_key",
    "find_digest_md",
    "opus_path_for_digest",
    "read_cache_key",
    "write_cache_key",
]


def audio_dir(run_dir: Path) -> Path:
    """The ``audio/`` subdirectory of a run, peer to ``assembly/`` etc."""
    return run_dir / "audio"


def cache_key_path(run_dir: Path) -> Path:
    """Path to the ``cache_key.txt`` sentinel inside the audio dir."""
    return audio_dir(run_dir) / "cache_key.txt"


def opus_path_for_digest(run_dir: Path, digest_md_path: Path) -> Path:
    """Audio output path matching the digest deliverable's stem.

    For example, ``{run_dir}/{date}-weekly-ai-digest.md`` maps to
    ``{run_dir}/audio/{date}-weekly-ai-digest.opus``.
    """
    return audio_dir(run_dir) / f"{digest_md_path.stem}.opus"


def compute_cache_key(
    md_bytes: bytes,
    voice_id: str,
    bitrate_kbps: int,
    *,
    sentence_silence_s: float | None = None,
) -> str:
    """SHA-256 over every input that affects the rendered ``.opus`` file.

    Inputs are NUL-separated so a change in any field (markdown content,
    voice swap, bitrate change, pause-duration tweak, narration shape
    bump) flips the hash; concatenation without separators could in
    principle alias different inputs to the same digest.

    Components:

    - ``md_bytes``: the digest markdown verbatim
    - ``voice_id``: Piper voice
    - ``bitrate_kbps``: Opus encode rate
    - ``sentence_silence_s``: Piper ``--sentence-silence`` value
      (or empty string when unset, meaning Piper's compiled default)
    - ``NARRATION_VERSION``: bumped when the narration-pre-pass output
      shape changes, so cache entries from prior versions auto-invalidate
    """
    h = hashlib.sha256()
    h.update(md_bytes)
    h.update(b"\0")
    h.update(voice_id.encode("utf-8"))
    h.update(b"\0")
    h.update(str(bitrate_kbps).encode("ascii"))
    h.update(b"\0")
    silence_repr = "" if sentence_silence_s is None else str(sentence_silence_s)
    h.update(silence_repr.encode("ascii"))
    h.update(b"\0")
    h.update(NARRATION_VERSION.encode("ascii"))
    return h.hexdigest()


def read_cache_key(run_dir: Path) -> str | None:
    """Return the stored cache key, or ``None`` if no prior render exists."""
    path = cache_key_path(run_dir)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def write_cache_key(run_dir: Path, key: str) -> None:
    """Persist the cache key after a successful render."""
    path = cache_key_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key, encoding="utf-8")


def find_digest_md(run_dir: Path) -> Path:
    """Locate the deliverable ``{date}-{slug}.md`` in the run root.

    The composer writes exactly one ``.md`` file at the run root.
    Caller-facing helpers (``api.render_audio``, ``cli.audio``) use this
    so they don't have to thread the path through every call.

    Raises:
        FileNotFoundError: No ``.md`` deliverable in ``run_dir``.
        ValueError: More than one ``.md`` is present, which would make
            the choice ambiguous.
    """
    candidates = sorted(p for p in run_dir.iterdir() if p.is_file() and p.suffix == ".md")
    if not candidates:
        msg = f"no digest markdown found at run root: {run_dir}"
        raise FileNotFoundError(msg)
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        msg = f"multiple .md deliverables in {run_dir} (ambiguous): {names}"
        raise ValueError(msg)
    return candidates[0]
