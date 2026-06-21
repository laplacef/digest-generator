"""Label pipeline stages.

Mirrors the shape of ``digest_generator.core.digest.stages``: one focused class
per file. The stage is ``topic.py`` (``TopicClassifier``,
zero-shot topic classification via BART-MNLI); additional passes
(sentiment, claim entailment) slot in here as siblings, each with
its own ``Label`` vocabulary.
"""
