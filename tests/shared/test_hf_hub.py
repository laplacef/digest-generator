"""Tests for digest_generator/shared/hf_hub.py: auto_login() against HuggingFace Hub."""

from unittest.mock import patch

import pytest

from digest_generator.shared.hf_hub import auto_login


class TestAutoLogin:
    @patch("digest_generator.shared.hf_hub.login")
    @patch("digest_generator.shared.hf_hub.settings")
    def test_calls_login_with_token(self, mock_settings, mock_login):
        mock_settings.hf_token = "test-token"
        auto_login()
        mock_login.assert_called_once_with(token="test-token")

    @patch("digest_generator.shared.hf_hub.login")
    @patch("digest_generator.shared.hf_hub.settings")
    def test_raises_on_login_failure(self, mock_settings, mock_login):
        mock_settings.hf_token = "bad-token"
        mock_login.side_effect = ValueError("Invalid token")
        with pytest.raises(ValueError, match="Invalid token"):
            auto_login()
