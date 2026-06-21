"""Lazy-loaded client registry for Ollama, shared across LLM call sites.

Provides a singleton ``client_registry`` whose properties instantiate clients
on first access via ``@cached_property``. Auto-detection selects cloud vs local
Ollama based on whether ``OLLAMA_API_KEY`` is set.

Host, API key, and the read timeout are configured via ``Settings``.

Timeouts are critical: the Ollama SDK constructs ``httpx.Client`` with
``timeout=None`` by default, meaning a connection that the server accepts
but never responds to blocks the worker thread forever. This module sets
bounded timeouts to prevent that.
"""

from functools import cached_property

import httpx
from ollama import Client

from digest_generator.shared.logging import logger
from digest_generator.shared.settings import settings

_OLLAMA_CLOUD_HOST = "https://ollama.com"

# Connect / write / pool are short because failure on those phases is
# almost always permanent at human-noticeable timescales (DNS, TCP,
# pool exhaustion). Read is the long one because legitimate LLM
# generations can take minutes (see `settings.ollama_read_timeout_s`).
_CONNECT_TIMEOUT_S = 10.0
_WRITE_TIMEOUT_S = 60.0
_POOL_TIMEOUT_S = 5.0


def _build_timeout() -> httpx.Timeout:
    """Build the httpx Timeout used for every Ollama call.

    Read at client-construction time so settings overrides are honored.
    """
    return httpx.Timeout(
        connect=_CONNECT_TIMEOUT_S,
        read=float(settings.ollama_read_timeout_s),
        write=_WRITE_TIMEOUT_S,
        pool=_POOL_TIMEOUT_S,
    )


class ClientRegistry:
    """Registry of pre-configured API clients, loaded lazily on first access."""

    @cached_property
    def ollama_local(self) -> Client:
        """Local Ollama instance (no auth required)."""
        logger.info("Creating local Ollama client: {}", settings.ollama_host)
        return Client(host=settings.ollama_host, timeout=_build_timeout())

    @cached_property
    def ollama_cloud(self) -> Client:
        """Ollama cloud instance (requires ``OLLAMA_API_KEY``)."""
        if not settings.ollama_api_key:
            msg = "OLLAMA_API_KEY is required for cloud Ollama"
            raise ValueError(msg)
        logger.info("Creating cloud Ollama client: {}", _OLLAMA_CLOUD_HOST)
        # Pass the bearer header explicitly. The Ollama SDK reads ``OLLAMA_API_KEY``
        # from ``os.environ`` at construction; ``pydantic-settings`` only populates
        # ``settings.ollama_api_key`` and never exports to the process env, so a
        # caller relying purely on ``.env`` (no shell ``export``) gets a 401.
        return Client(
            host=_OLLAMA_CLOUD_HOST,
            timeout=_build_timeout(),
            headers={"Authorization": f"Bearer {settings.ollama_api_key}"},
        )

    @cached_property
    def ollama(self) -> Client:
        """Auto-detected Ollama client: cloud if ``OLLAMA_API_KEY`` is set, otherwise local."""
        if settings.ollama_api_key:
            logger.info("OLLAMA_API_KEY detected — using cloud Ollama")
            return self.ollama_cloud
        logger.info("No OLLAMA_API_KEY — using local Ollama")
        return self.ollama_local


client_registry = ClientRegistry()
