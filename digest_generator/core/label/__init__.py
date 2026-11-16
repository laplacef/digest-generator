"""Label domain: produces ``Label`` objects from ``Entry`` input.

The stage is ``TopicClassifier`` in ``stages/topic.py``, and ``io.py`` holds
the URL-keyed per-feed persistence (``save_labeled``, ``load_labeled``,
``iter_labeled``).

This package does not re-export the stage: importing ``stages.topic`` loads
torch and gliclass, and the digest path imports ``io`` without needing
either.
"""
