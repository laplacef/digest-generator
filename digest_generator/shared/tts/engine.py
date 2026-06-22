"""Subprocess wrapper around Piper TTS + ffmpeg encoding.

The engine boundary is a process line, not a Python import. Swapping
Piper for Kokoro / XTTS / a different self-hosted engine later changes
this module's internals; ``core/audio/`` keeps calling ``synthesize``
and stays unaware of the engine identity.

Two operations, both subprocess-shaped:

- ``synthesize_to_wav(text, voice)`` pipes text into ``piper`` and
  captures the WAV bytes from stdout. Piper's CLI reads one line of
  text per invocation when ``--output-raw`` is unset and writes a WAV
  to stdout when ``--output_file -`` is passed.
- ``encode_opus(wav_bytes, out_path, *, bitrate_kbps)`` pipes the WAV
  into ``ffmpeg`` and writes an Opus file at the requested bitrate.

``synthesize`` is the public composition: text, then WAV, then Opus on disk.
Callers in ``core/audio/`` use this single entry point; the
two-step shape is exposed for tests that want to inspect intermediate
output without round-tripping through ffmpeg.

External dependencies (``piper``, ``ffmpeg``) are resolved by name via
``$PATH`` unless an absolute path is passed. Both are system-level
binaries, mirroring the Ollama dependency posture.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 (binaries resolved via shutil.which, no shell, no user-controlled argv[0])
from pathlib import Path

from digest_generator.shared.logging import logger
from digest_generator.shared.tts.types import VoiceConfig

__all__ = [
    "EngineError",
    "MissingBinaryError",
    "encode_opus",
    "estimate_audio_duration_s",
    "synthesize",
    "synthesize_to_wav",
]


class EngineError(RuntimeError):
    """Raised when a TTS subprocess exits non-zero or produces no output."""


class MissingBinaryError(EngineError):
    """Raised when ``piper`` or ``ffmpeg`` isn't on ``$PATH``."""


def _require_binary(name: str, override: str | None = None) -> str:
    """Resolve a binary to an absolute path. Raises ``MissingBinaryError``."""
    candidate = override or name
    resolved = shutil.which(candidate)
    if resolved is None:
        msg = f"{name!r} binary not found on PATH" + (
            f" (looked for {override!r})" if override and override != name else ""
        )
        raise MissingBinaryError(msg)
    return resolved


def synthesize_to_wav(
    text: str,
    voice: VoiceConfig,
    *,
    piper_path: str = "piper",
    sentence_silence_s: float | None = None,
) -> bytes:
    """Run ``piper`` against ``text`` and return WAV bytes.

    Args:
        text: Narration script. Piper interprets sentence-terminator
            punctuation (``.!?``) and newlines as pause cues (see
            ``digest_generator.core.audio.narration`` for the formatter that
            produces these scripts). Piper does *not* process SSML.
        voice: Loaded voice configuration. ``voice.model_path`` must exist on
            disk along with its sidecar ``.onnx.json``.
        piper_path: Override for the Piper binary location. Defaults to
            looking up ``piper`` on ``$PATH``.
        sentence_silence_s: Pause duration (seconds) inserted at every
            sentence terminator. ``None`` lets Piper use its compiled
            default (~0.2s). Passed through as ``--sentence-silence``.

    Returns:
        Raw WAV bytes ready to pipe into ffmpeg.

    Raises:
        MissingBinaryError: Piper is not installed.
        EngineError: Piper exited non-zero or produced no output.
    """
    binary = _require_binary("piper", piper_path)
    if not voice.model_path.exists():
        msg = f"voice model not found: {voice.model_path}"
        raise EngineError(msg)

    cmd = [
        binary,
        "--model",
        str(voice.model_path),
        "--output_file",
        "-",
    ]
    if sentence_silence_s is not None:
        cmd.extend(["--sentence-silence", str(sentence_silence_s)])
    logger.debug("piper invocation: {} ({} chars)", voice.voice_id, len(text))
    try:
        # shell=False (default), argv is a list; binary path is shutil.which-resolved.
        result = subprocess.run(  # nosec B603
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        msg = f"piper failed (exit={exc.returncode}): {stderr}"
        raise EngineError(msg) from exc

    if not result.stdout:
        msg = "piper produced empty output"
        raise EngineError(msg)
    return result.stdout


def encode_opus(
    wav_bytes: bytes,
    out_path: Path,
    *,
    bitrate_kbps: int,
    ffmpeg_path: str = "ffmpeg",
) -> Path:
    """Encode WAV bytes to Opus at ``out_path``.

    Args:
        wav_bytes: Raw WAV produced by ``synthesize_to_wav``.
        out_path: Destination ``.opus`` path. Parent directory must exist.
        bitrate_kbps: Target Opus bitrate (e.g. 24 for spoken-word).
        ffmpeg_path: Override for the ffmpeg binary location.

    Returns:
        ``out_path`` after a successful encode.

    Raises:
        MissingBinaryError: ffmpeg is not installed.
        EngineError: ffmpeg exited non-zero.
    """
    binary = _require_binary("ffmpeg", ffmpeg_path)

    # `--` separates options from the positional output path. This defends
    # against flag injection via a `-`-leading filename, independent of the
    # date-based naming invariant in `core/digest/io.py:build_digest_filename`
    # (opus stems derive from the digit-leading issue date, never from a
    # user-influenced title) that already prevents such filenames (see
    # TestSubprocessSafety in tests/core/digest/test_io.py).
    cmd = [
        binary,
        "-loglevel",
        "error",
        "-y",
        "-f",
        "wav",
        "-i",
        "-",
        "-c:a",
        "libopus",
        "-b:a",
        f"{bitrate_kbps}k",
        "-ac",
        "1",
        "--",
        str(out_path),
    ]
    logger.debug("ffmpeg encode: {} ({} kbps)", out_path.name, bitrate_kbps)
    try:
        # shell=False (default), argv is a list; binary path is shutil.which-resolved.
        subprocess.run(  # nosec B603
            cmd,
            input=wav_bytes,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        msg = f"ffmpeg failed (exit={exc.returncode}): {stderr}"
        raise EngineError(msg) from exc

    return out_path


def estimate_audio_duration_s(byte_size: int, bitrate_kbps: int) -> float:
    """Approximate Opus duration from file size and target bitrate.

    Opus headers add a few KB of overhead but for typical spoken-word
    digests the estimate is within ~1% of the true duration, accurate
    enough for the ``real_time_factor`` telemetry signal in
    ``meta.json`` without requiring a separate ``ffprobe`` call.
    Returns 0.0 for an empty or absent file.
    """
    if byte_size <= 0 or bitrate_kbps <= 0:
        return 0.0
    return byte_size * 8.0 / (bitrate_kbps * 1000.0)


def synthesize(
    text: str,
    voice: VoiceConfig,
    out_path: Path,
    *,
    bitrate_kbps: int,
    piper_path: str = "piper",
    ffmpeg_path: str = "ffmpeg",
    sentence_silence_s: float | None = None,
) -> Path:
    """Synthesize ``text`` to an Opus file at ``out_path``.

    Composition of ``synthesize_to_wav`` and ``encode_opus``. The WAV
    stays in memory; for typical digest lengths it's a few MB, well
    below the threshold where streaming through a pipe would matter.

    ``sentence_silence_s`` is forwarded to Piper as ``--sentence-silence``
    for global pause-duration tuning.
    """
    wav = synthesize_to_wav(
        text,
        voice,
        piper_path=piper_path,
        sentence_silence_s=sentence_silence_s,
    )
    return encode_opus(wav, out_path, bitrate_kbps=bitrate_kbps, ffmpeg_path=ffmpeg_path)
