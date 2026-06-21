"""Audio domain types.

Dataclasses that flow through the audio renderer. Vendor / engine types
(``VoiceConfig``, etc.) live in ``digest_generator.shared.tts.types``; this
module owns only the renderer's own inputs and outputs.
"""

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "AudioArtifact",
    "NarrationScript",
]


@dataclass(frozen=True)
class NarrationScript:
    """Speech-ready text produced by the narration pre-pass.

    Returned by the renderer's internal narration step. The text uses
    sentence-terminator punctuation and newlines as pause cues, and Piper
    is fed this string directly on stdin.
    """

    text: str

    @property
    def char_count(self) -> int:
        """Character count, used for telemetry (``narration_chars``)."""
        return len(self.text)


@dataclass(frozen=True)
class AudioArtifact:
    """Result of a single render call.

    Surfaces everything the telemetry harvest in ``api.render_audio``
    needs plus the consumer-facing ``opus_path``. ``cached`` is ``True``
    when the render was a no-op cache hit; ``narration_chars`` is ``0``
    in that case because the narration step is skipped.
    """

    opus_path: Path
    voice_id: str
    bitrate_kbps: int
    narration_chars: int
    audio_bytes: int
    audio_duration_s: float
    cached: bool
