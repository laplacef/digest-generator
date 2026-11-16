"""Zero-shot topic classification using GLiClass.

``TopicClassifier.classify_entries`` labels one feed's entries for
``api.label``. It reads the article text rather than the LLM summary, so
labeling runs alongside the summarizer.
"""

from collections.abc import Callable
from time import perf_counter
from typing import Any

import torch
from gliclass import ZeroShotClassificationPipeline

from digest_generator.core.types import Entry, Label, TopicType
from digest_generator.shared.logging import log_stage
from digest_generator.shared.settings import settings
from digest_generator.shared.transformers.types import ModelConfig


class TopicClassifier:
    """Labels entries with ``TopicType`` topics using a GLiClass zero-shot model.

    GLiClass encodes every label together with the text, so one forward pass
    scores the whole vocabulary for an article. Each label gets an independent
    sigmoid score (multi-label).

    Args:
        model_config: A loaded ``ModelConfig`` with tokenizer and model hydrated.
        pipeline: Callable with the ``gliclass.ZeroShotClassificationPipeline``
            interface. Defaults to one built over ``model_config``.
    """

    def __init__(self, model_config: ModelConfig, pipeline: Callable[..., Any] | None = None):
        self.labels = [t.value for t in TopicType]
        # The text the model reads for each label, mapped back to the label:
        # "large language models" stands for "large-language-models".
        self._label_by_text = {lab.replace("-", " "): lab for lab in self.labels}
        if pipeline is None:
            pipeline = ZeroShotClassificationPipeline(
                model_config.model,
                model_config.tokenizer,
                max_length=settings.topic_max_length,
                classification_type="multi-label",
                # The pipeline uses a torch.device as given but maps any string
                # without "cuda" to CPU, which would ignore "mps".
                device=torch.device(model_config.device_str),
                progress_bar=False,
            )
        self.pipeline: Callable[..., Any] = pipeline

    def classify_entries(
        self, entries: list[Entry], *, feed: str | None = None
    ) -> list[list[Label]]:
        """Label each entry, returning one ``list[Label]`` per entry in input order.

        Scores ``entry.content_head``, or the title and description when the
        fetcher found no article prose. ``feed`` tags the stage span so
        concurrent per-feed calls are distinguishable in ``run.log``.
        """
        min_score = settings.topic_min_score
        start_fields = {"feed": feed} if feed else {}
        with log_stage("topic", **start_fields) as span:
            start = perf_counter()
            texts = [e.content_head or f"{e.title}\n{e.description}" for e in entries]
            labels_per_entry = self._infer_batch(texts, min_score)
            elapsed = perf_counter() - start

            n = len(entries)
            total_labels = sum(len(labels) for labels in labels_per_entry)
            span.set(
                entries=n,
                labels=total_labels,
                avg_labels_per_entry=round(total_labels / n, 2) if n else 0.0,
                min_score=min_score,
                vocabulary=len(self.labels),
                articles_per_batch=settings.topic_articles_per_batch,
                articles_per_sec=round(n / elapsed, 3) if elapsed > 0 else 0.0,
            )
            return labels_per_entry

    def _infer_batch(self, texts: list[str], min_score: float) -> list[list[Label]]:
        """Score each text against every label and keep those at or above ``min_score``.

        The pipeline returns every label's score (it gets threshold 0), so the
        cut happens here, and a text with no label that reaches it keeps its
        top label. Raises ``RuntimeError`` unless the pipeline returns one row per
        text, since a short result would misalign labels with entries.
        """
        if not texts:
            return []
        rows = self.pipeline(
            texts,
            list(self._label_by_text),
            threshold=0.0,
            batch_size=settings.topic_articles_per_batch,
        )
        if len(rows) != len(texts):
            msg = f"GLiClass returned {len(rows)} result rows for {len(texts)} texts"
            raise RuntimeError(msg)

        results: list[list[Label]] = []
        for row in rows:
            scored = sorted(
                ((self._label_by_text[r["label"]], float(r["score"])) for r in row),
                key=lambda x: x[1],
                reverse=True,
            )
            chosen = [(lab, sc) for lab, sc in scored if sc >= min_score] or scored[:1]
            results.append([Label(value=TopicType(lab), confidence=sc) for lab, sc in chosen])
        return results
