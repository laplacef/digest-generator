"""Tests for digest_generator/shared/tts/registry.py: VoiceRegistry + path helpers."""

from pathlib import Path
from unittest.mock import patch

import pytest

from digest_generator.shared.tts.registry import VoiceRegistry, _piper_hf_paths
from digest_generator.shared.tts.types import VoiceConfig


class TestPiperHfPaths:
    """Voice id parsing into HuggingFace repo paths."""

    def test_en_us_amy_medium(self):
        onnx, cfg = _piper_hf_paths("en_US-amy-medium")
        assert onnx == "en/en_US/amy/medium/en_US-amy-medium.onnx"
        assert cfg == "en/en_US/amy/medium/en_US-amy-medium.onnx.json"

    def test_en_gb_alan_medium(self):
        onnx, cfg = _piper_hf_paths("en_GB-alan-medium")
        assert onnx == "en/en_GB/alan/medium/en_GB-alan-medium.onnx"
        assert cfg == "en/en_GB/alan/medium/en_GB-alan-medium.onnx.json"

    def test_de_de_thorsten_high(self):
        onnx, cfg = _piper_hf_paths("de_DE-thorsten-high")
        assert onnx == "de/de_DE/thorsten/high/de_DE-thorsten-high.onnx"
        assert cfg == "de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json"

    def test_rejects_non_three_part_id(self):
        with pytest.raises(ValueError, match="doesn't match Piper convention"):
            _piper_hf_paths("invalid")

    def test_rejects_two_part_id(self):
        with pytest.raises(ValueError, match="doesn't match Piper convention"):
            _piper_hf_paths("en_US-amy")


class TestVoiceRegistry:
    """Registry caches downloaded voice configs and surfaces sample-rate from settings."""

    @patch("digest_generator.shared.tts.registry.hf_hub_download")
    def test_default_downloads_both_files(self, mock_download, tmp_path):
        mock_download.side_effect = lambda **kw: str(tmp_path / Path(kw["filename"]).name)

        with patch("digest_generator.shared.tts.registry.settings") as mock_settings:
            mock_settings.audio_voice_model = "en_US-amy-medium"
            mock_settings.audio_voice_cache = tmp_path
            mock_settings.audio_voice_revision = "abc123"
            mock_settings.audio_sample_rate = 22050

            registry = VoiceRegistry()
            voice = registry.default

        assert isinstance(voice, VoiceConfig)
        assert voice.voice_id == "en_US-amy-medium"
        assert voice.sample_rate == 22050
        assert voice.model_path == tmp_path / "en_US-amy-medium.onnx"

        # Both ONNX and sidecar JSON downloaded, both pinned to the revision.
        assert mock_download.call_count == 2
        downloaded = {call.kwargs["filename"] for call in mock_download.call_args_list}
        assert downloaded == {
            "en/en_US/amy/medium/en_US-amy-medium.onnx",
            "en/en_US/amy/medium/en_US-amy-medium.onnx.json",
        }
        for call in mock_download.call_args_list:
            assert call.kwargs["revision"] == "abc123"

    @patch("digest_generator.shared.tts.registry.hf_hub_download")
    def test_default_is_cached(self, mock_download, tmp_path):
        """``@cached_property`` short-circuits the second access."""
        mock_download.side_effect = lambda **kw: str(tmp_path / Path(kw["filename"]).name)

        with patch("digest_generator.shared.tts.registry.settings") as mock_settings:
            mock_settings.audio_voice_model = "en_US-amy-medium"
            mock_settings.audio_voice_cache = tmp_path
            mock_settings.audio_voice_revision = "abc123"
            mock_settings.audio_sample_rate = 22050

            registry = VoiceRegistry()
            first = registry.default
            second = registry.default

        assert first is second
        # Two downloads on first access, none on second.
        assert mock_download.call_count == 2

    @patch("digest_generator.shared.tts.registry.hf_hub_download")
    def test_default_creates_cache_dir(self, mock_download, tmp_path):
        """Cache directory is auto-created if missing."""
        cache = tmp_path / "nested" / "piper-voices"
        mock_download.side_effect = lambda **kw: str(cache / Path(kw["filename"]).name)

        with patch("digest_generator.shared.tts.registry.settings") as mock_settings:
            mock_settings.audio_voice_model = "en_US-amy-medium"
            mock_settings.audio_voice_cache = cache
            mock_settings.audio_sample_rate = 22050

            _ = VoiceRegistry().default

        assert cache.exists()

    @patch("digest_generator.shared.tts.registry.hf_hub_download")
    def test_invalid_voice_id_raises_before_download(self, mock_download, tmp_path):
        """Bad voice ids fail fast; no HF round-trip."""
        with patch("digest_generator.shared.tts.registry.settings") as mock_settings:
            mock_settings.audio_voice_model = "bogus"
            mock_settings.audio_voice_cache = tmp_path
            mock_settings.audio_sample_rate = 22050

            with pytest.raises(ValueError, match="doesn't match Piper convention"):
                _ = VoiceRegistry().default

        mock_download.assert_not_called()
