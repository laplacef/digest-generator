"""TTS infrastructure types.

Holds voice configuration. The ``core/audio/`` domain types (narration
script, audio artifact) live in ``digest_generator.core.audio.types``, the same
split as ``shared/transformers/types.py`` vs. ``core/types.py``.
"""

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "VoiceConfig",
]


@dataclass(frozen=True)
class VoiceConfig:
    """Piper voice configuration.

    A Piper voice is a pair of files: an ONNX model and a sidecar JSON
    config (``{voice}.onnx`` + ``{voice}.onnx.json``). Both must live in
    the same directory; Piper reads the JSON automatically when given
    the ONNX path.

    Attributes:
        voice_id: Piper voice identifier, e.g. ``en_US-amy-medium``.
        model_path: Absolute path to the ``.onnx`` file on disk.
        sample_rate: Native sample rate the voice synthesizes at
            (Piper voices ship at 16000 or 22050 Hz depending on model).
    """

    voice_id: str
    model_path: Path
    sample_rate: int

    @property
    def config_path(self) -> Path:
        """Sidecar JSON config path that Piper expects next to the ONNX file."""
        return self.model_path.with_suffix(self.model_path.suffix + ".json")
