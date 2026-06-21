"""Summary pipeline stages.

Mirrors the shape of ``digest_generator.core.digest.stages``: one focused class
per file, wired together by the parent package's orchestrator (when
multiple stages exist). The stage is ``summarizer.py``
(``ContentSummarizer``); additional passes (such as extraction or
critique) slot in here as siblings.
"""
