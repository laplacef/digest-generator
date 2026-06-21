"""HuggingFace Hub authentication, shared by every HF Hub consumer.

A single ``auto_login()`` against ``settings.hf_token`` covers both
``shared/transformers/`` (BART-MNLI classifier downloads via
``from_pretrained``) and ``shared/tts/`` (Piper voice files via
``hf_hub_download``). Centralized here because Hub auth is orthogonal
to the serving engine: the Hub is the artifact source, and the package
that consumes those artifacts is the serving engine.

The CLI per-stage entrypoints that touch a Hub-downloaded artifact
call ``auto_login()`` once before the relevant stage runs.
"""

from huggingface_hub import login

from digest_generator.shared.logging import logger
from digest_generator.shared.settings import settings


def auto_login() -> None:
    """Authenticate with HuggingFace Hub using the ``HF_TOKEN`` setting.

    Raises:
        Exception: If authentication fails (e.g., missing or invalid token).
    """
    try:
        login(token=settings.hf_token)
        logger.info("Logged in to Hugging Face")
    except Exception as e:
        logger.exception("Failed to log in to Hugging Face: {}", e)
        raise
