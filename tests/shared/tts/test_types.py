"""Tests for digest_generator/shared/tts/types.py: VoiceConfig."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from digest_generator.shared.tts.types import VoiceConfig


class TestVoiceConfig:
    """VoiceConfig holds an ONNX model path + sidecar JSON convention."""

    def test_construction(self):
        v = VoiceConfig(
            voice_id="en_US-amy-medium",
            model_path=Path("/tmp/amy.onnx"),
            sample_rate=22050,
        )
        assert v.voice_id == "en_US-amy-medium"
        assert v.model_path == Path("/tmp/amy.onnx")
        assert v.sample_rate == 22050

    def test_config_path_appends_dot_json(self):
        """Piper sidecar lives at ``<model>.onnx.json``."""
        v = VoiceConfig(
            voice_id="en_US-amy-medium",
            model_path=Path("/cache/en_US-amy-medium.onnx"),
            sample_rate=22050,
        )
        assert v.config_path == Path("/cache/en_US-amy-medium.onnx.json")

    def test_frozen(self):
        """VoiceConfig is immutable so the registry cache can't be mutated."""
        v = VoiceConfig(
            voice_id="en_US-amy-medium",
            model_path=Path("/tmp/v.onnx"),
            sample_rate=22050,
        )
        with pytest.raises(FrozenInstanceError):
            v.voice_id = "other"  # type: ignore[misc]
