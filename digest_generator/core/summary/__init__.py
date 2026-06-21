"""Article summary domain: produces ``Summary`` objects from ``Entry`` input.

Mirrors ``digest_generator.core.digest`` in shape: per-stage classes live under
``stages/``, prompts under ``prompts/``, per-feed persistence in ``io.py``.
The stage is ``ContentSummarizer`` (LLM-driven fact extraction); additional
passes slot in as siblings under ``stages/``.
"""

from digest_generator.core.summary.stages.summarizer import ContentSummarizer

__all__ = ["ContentSummarizer"]
