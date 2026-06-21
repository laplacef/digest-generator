"""Tests for digest_generator/shared/settings.py: Settings defaults and env var overrides."""

from unittest.mock import patch

from digest_generator.shared.settings import Settings


def _clean_settings(**overrides):
    """Create Settings isolated from .env file and real env vars.

    Constructs a Settings instance that ignores .env and uses only
    the provided overrides (if any) on top of class defaults.
    """
    return Settings(_env_file=None, **overrides)


class TestSettingsDefaults:
    """Verify all default values match expectations."""

    def test_device_default(self):
        s = _clean_settings()
        assert s.device == "cpu"

    def test_summarizer_defaults(self):
        s = _clean_settings()
        assert s.summarizer_model == "gemma4:31b-cloud"
        assert s.summarizer_temperature == 0.2
        assert s.summarizer_top_p is None
        assert s.summarizer_repetition_penalty is None
        assert s.summarizer_seed is None
        assert s.summarizer_concurrency == 8

    def test_classifier_defaults(self):
        s = _clean_settings()
        assert s.topic_model == "facebook/bart-large-mnli"
        assert s.topic_threshold == 0.5
        assert s.topic_max_length == 512

    def test_ollama_defaults(self):
        s = _clean_settings()
        assert s.ollama_host == "http://localhost:11434"
        assert s.ollama_api_key == ""
        assert s.ollama_concurrency == 3
        assert s.ollama_read_timeout_s == 300

    def test_writer_model_defaults(self):
        s = _clean_settings()
        assert s.writer_model == "gemma4:31b-cloud"
        assert s.writer_temperature == 0.4

    def test_editorial_defaults(self):
        s = _clean_settings()
        assert s.editorial_model == "gpt-oss:120b-cloud"
        assert s.editorial_temperature == 0.4

    def test_framer_defaults(self):
        s = _clean_settings()
        assert s.framer_model == "gpt-oss:120b-cloud"
        assert s.framer_temperature == 0.4

    def test_watcher_defaults(self):
        s = _clean_settings()
        assert s.watcher_model == "gpt-oss:120b-cloud"
        assert s.watcher_temperature == 0.4

    def test_fetcher_defaults(self):
        s = _clean_settings()
        assert s.fetch_timeout == 10
        assert s.fetch_rate_limit == 0.4
        assert s.min_content_length == 200
        assert s.max_boilerplate_ratio == 0.05

    def test_pipeline_defaults(self):
        s = _clean_settings()
        assert s.days_back == 7
        assert s.output_dir == "output"

    def test_logging_defaults(self):
        s = _clean_settings()
        assert s.log_level_console == "INFO"
        assert s.log_level_file == "DEBUG"
        assert s.log_dir == "logs"

    def test_writer_defaults(self):
        s = _clean_settings()
        assert s.writer_section_batch_size == 30

    def test_audio_defaults(self):
        s = _clean_settings()
        assert s.audio_enabled is False
        assert s.audio_voice_model == "en_US-amy-medium"
        # Pin matches rhasspy/piper-voices v1.0.0 tag; see audio_voice_revision docstring.
        expected = (
            "375a0fe641dea077c2a47b4e9a056d6da521eed3"  # pragma: allowlist secret  # gitleaks:allow
        )
        assert s.audio_voice_revision == expected
        assert s.audio_bitrate_kbps == 24
        assert s.audio_sample_rate == 22050
        assert s.audio_ffmpeg_path == "ffmpeg"
        assert s.audio_sentence_silence_s == 0.4
        assert s.audio_voice_cache.name == "piper-voices"
        assert s.audio_voice_cache.parent.name == "digest_generator"


class TestSettingsEnvOverride:
    """Verify environment variables override defaults."""

    @patch.dict("os.environ", {"DEVICE": "cuda"})
    def test_device_override(self):
        s = _clean_settings()
        assert s.device == "cuda"

    @patch.dict("os.environ", {"WRITER_MODEL": "llama3:8b"})
    def test_writer_model_override(self):
        s = _clean_settings()
        assert s.writer_model == "llama3:8b"

    @patch.dict("os.environ", {"EDITORIAL_MODEL": "gpt-oss:20b-cloud"})
    def test_editorial_model_override(self):
        s = _clean_settings()
        assert s.editorial_model == "gpt-oss:20b-cloud"

    @patch.dict("os.environ", {"DAYS_BACK": "14"})
    def test_days_back_override(self):
        s = _clean_settings()
        assert s.days_back == 14

    @patch.dict("os.environ", {"TOPIC_THRESHOLD": "0.8"})
    def test_topic_threshold_override(self):
        s = _clean_settings()
        assert s.topic_threshold == 0.8

    @patch.dict("os.environ", {"AUDIO_ENABLED": "true", "AUDIO_VOICE_MODEL": "en_GB-alan-medium"})
    def test_audio_overrides(self):
        s = _clean_settings()
        assert s.audio_enabled is True
        assert s.audio_voice_model == "en_GB-alan-medium"
