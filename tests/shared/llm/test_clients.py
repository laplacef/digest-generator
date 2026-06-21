"""Tests for digest_generator/shared/llm/clients.py: ClientRegistry auto-detection.

Two important testing concepts here:

1. **Mocking** (unittest.mock.patch):
   Instead of connecting to real Ollama, the Client class is replaced
   ("patched") with a fake. This tests the registry logic without a
   running Ollama instance.

2. **Fixtures** (pytest.fixture):
   Reusable setup/teardown code. Since ClientRegistry uses @cached_property,
   each test needs a fresh instance; the fixture handles that.
"""

from unittest.mock import patch

import httpx
import pytest

from digest_generator.shared.llm.clients import ClientRegistry

# =============================================================================
# Fixtures
# =============================================================================
# A fixture is a function that provides test data or setup. When a test
# function has a parameter matching a fixture name, pytest injects it
# automatically.
#
# Why we need this: ClientRegistry uses @cached_property, which caches
# the result on the instance forever. If test A calls .ollama_local,
# test B would get the same cached client. A fresh instance per test
# avoids this cross-contamination.


@pytest.fixture
def registry():
    """Create a fresh ClientRegistry for each test."""
    return ClientRegistry()


def _client_kwargs(mock_client):
    """Return the kwargs of the most recent Client(...) call."""
    assert mock_client.call_args is not None, "Client() was never called"
    return mock_client.call_args.kwargs


# =============================================================================
# Auto-detection tests
# =============================================================================


class TestAutoDetection:
    """The .ollama property returns cloud if OLLAMA_API_KEY is set, else local."""

    @patch("digest_generator.shared.llm.clients.settings")
    @patch("digest_generator.shared.llm.clients.Client")
    def test_auto_detects_cloud_when_api_key_set(self, mock_client, mock_settings, registry):
        mock_settings.ollama_api_key = "test-key-123"  # pragma: allowlist secret  # gitleaks:allow
        mock_settings.ollama_host = "http://localhost:11434"
        mock_settings.ollama_read_timeout_s = 300
        _ = registry.ollama

        assert _client_kwargs(mock_client)["host"] == "https://ollama.com"

    @patch("digest_generator.shared.llm.clients.settings")
    @patch("digest_generator.shared.llm.clients.Client")
    def test_auto_detects_local_when_no_api_key(self, mock_client, mock_settings, registry):
        mock_settings.ollama_api_key = ""
        mock_settings.ollama_host = "http://localhost:11434"
        mock_settings.ollama_read_timeout_s = 300
        _ = registry.ollama

        assert _client_kwargs(mock_client)["host"] == "http://localhost:11434"


# =============================================================================
# Explicit client tests
# =============================================================================


class TestOllamaLocal:
    @patch("digest_generator.shared.llm.clients.settings")
    @patch("digest_generator.shared.llm.clients.Client")
    def test_creates_local_client(self, mock_client, mock_settings, registry):
        mock_settings.ollama_host = "http://localhost:11434"
        mock_settings.ollama_read_timeout_s = 300
        _ = registry.ollama_local
        kwargs = _client_kwargs(mock_client)
        assert kwargs["host"] == "http://localhost:11434"
        assert isinstance(kwargs["timeout"], httpx.Timeout)

    @patch("digest_generator.shared.llm.clients.settings")
    @patch("digest_generator.shared.llm.clients.Client")
    def test_caches_client(self, mock_client, mock_settings, registry):
        """@cached_property: second access reuses the same instance."""
        mock_settings.ollama_host = "http://localhost:11434"
        mock_settings.ollama_read_timeout_s = 300
        client1 = registry.ollama_local
        client2 = registry.ollama_local

        mock_client.assert_called_once()
        assert client1 is client2


class TestOllamaCloud:
    @patch("digest_generator.shared.llm.clients.settings")
    @patch("digest_generator.shared.llm.clients.Client")
    def test_creates_cloud_client(self, mock_client, mock_settings, registry):
        mock_settings.ollama_api_key = "test-key-123"  # pragma: allowlist secret  # gitleaks:allow
        mock_settings.ollama_read_timeout_s = 300
        _ = registry.ollama_cloud
        assert _client_kwargs(mock_client)["host"] == "https://ollama.com"

    @patch("digest_generator.shared.llm.clients.settings")
    def test_raises_without_api_key(self, mock_settings, registry):
        """ollama_cloud should raise ValueError when OLLAMA_API_KEY is missing."""
        mock_settings.ollama_api_key = ""

        with pytest.raises(ValueError, match="OLLAMA_API_KEY is required"):
            _ = registry.ollama_cloud


# =============================================================================
# Timeout configuration
# =============================================================================


class TestTimeoutConfiguration:
    """Each Client is constructed with bounded httpx timeouts.

    The Ollama SDK defaults to ``timeout=None`` (no timeout), meaning
    a hung socket blocks forever. Passing
    ``httpx.Timeout(connect=10, read=N, write=60, pool=5)`` closes that hole,
    where ``N`` is ``settings.ollama_read_timeout_s`` (default 300).
    """

    @patch("digest_generator.shared.llm.clients.settings")
    @patch("digest_generator.shared.llm.clients.Client")
    def test_local_client_has_bounded_timeout(self, mock_client, mock_settings, registry):
        mock_settings.ollama_host = "http://localhost:11434"
        mock_settings.ollama_read_timeout_s = 300
        _ = registry.ollama_local

        timeout = _client_kwargs(mock_client)["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        # httpx.Timeout exposes per-component values via attributes.
        assert timeout.connect == 10.0
        assert timeout.read == 300.0
        assert timeout.write == 60.0
        assert timeout.pool == 5.0

    @patch("digest_generator.shared.llm.clients.settings")
    @patch("digest_generator.shared.llm.clients.Client")
    def test_cloud_client_has_bounded_timeout(self, mock_client, mock_settings, registry):
        mock_settings.ollama_api_key = "test-key-123"  # pragma: allowlist secret  # gitleaks:allow
        mock_settings.ollama_read_timeout_s = 300
        _ = registry.ollama_cloud

        timeout = _client_kwargs(mock_client)["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read == 300.0

    @patch("digest_generator.shared.llm.clients.settings")
    @patch("digest_generator.shared.llm.clients.Client")
    def test_read_timeout_honors_setting_override(self, mock_client, mock_settings, registry):
        """Setting `ollama_read_timeout_s=60` should produce a 60s read timeout."""
        mock_settings.ollama_host = "http://localhost:11434"
        mock_settings.ollama_read_timeout_s = 60
        _ = registry.ollama_local

        timeout = _client_kwargs(mock_client)["timeout"]
        assert timeout.read == 60.0
