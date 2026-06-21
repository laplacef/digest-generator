"""Label domain: produces ``Label`` objects from ``Entry`` input.

Mirrors ``digest_generator.core.digest`` and ``digest_generator.core.summary`` in shape:
per-stage classes live under ``stages/``, per-feed persistence in
``io.py``. The stage is ``TopicClassifier`` (zero-shot topic
classification via BART-MNLI); additional passes (sentiment, claim
entailment) slot in as siblings under ``stages/`` with their own
enums (``SentimentType``, ``EntailmentType``).

Per-stage IO (``save_labeled`` / ``load_labeled`` / ``iter_labeled``
operating on URL-keyed Label lists) is in ``digest_generator.core.label.io``.
"""

from digest_generator.core.label.stages.topic import TopicClassifier

__all__ = ["TopicClassifier"]
