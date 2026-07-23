"""Zero-shot topic classification using BART-large-MNLI.

Three public entry points:

- ``classify_text(text)`` takes a single text and returns ``list[Label]``.
  This is the canonical shape; ``api.label`` feeds raw entry text rather
  than the post-summarizer string.
- ``classify_entries(entries)`` takes a batch ``list[Entry]`` and returns
  ``list[list[Label]]``, using ``entry.content_head`` (with
  ``title + description`` fallback when the head is empty). Wrapped in
  ``log_stage("topic")``.
- ``classify_summaries(summaries)`` is a backward-compatible wrapper for
  direct programmatic callers. It builds the input string from
  ``title + description + summary`` and attaches topics to copies of the
  input summaries. The production path does not use it.
"""

from dataclasses import replace
from time import perf_counter

import torch

from digest_generator.core.types import Entry, Label, Summary, TopicType
from digest_generator.shared.logging import log_stage
from digest_generator.shared.settings import settings
from digest_generator.shared.transformers.types import ModelConfig


class TopicClassifier:
    """Classifies text into ``TopicType`` categories using zero-shot NLI.

    Builds hypotheses of the form "This article is about {label}" for each
    ``TopicType`` value and scores them via entailment probability.

    Args:
        model_config: A pre-loaded ``ModelConfig`` with tokenizer and model hydrated.
    """

    def __init__(self, model_config: ModelConfig):
        self.tokenizer = model_config.tokenizer
        self.model = model_config.model
        self.device = model_config.device
        self.model.eval()
        self.labels = [t.value for t in TopicType]

    def classify_text(self, text: str, threshold: float | None = None) -> list[Label]:
        """Run NLI over ``text``, return labels above threshold (top-1 fallback).

        No log_stage wrapper; meant to be called from a batch entry point or
        from the api layer's ``label`` orchestrator that wraps its own span.
        """
        threshold = threshold if threshold is not None else settings.topic_threshold
        return self._infer(text, threshold)

    def classify_entries(
        self,
        entries: list[Entry],
        threshold: float | None = None,
        *,
        feed: str | None = None,
    ) -> list[list[Label]]:
        """Classify a batch of entries via raw text.

        Uses ``entry.content_head`` when present; falls back to
        ``title + '\\n' + description`` when the head is empty (fetcher-side
        fallback case for feeds without extractable prose). Returns
        ``list[list[Label]]`` aligned with the input order.

        ``feed`` (optional) tags the ``stage.start`` / ``stage.done`` lines
        so concurrent per-feed invocations are distinguishable in run.log.
        """
        threshold = threshold if threshold is not None else settings.topic_threshold

        start_fields = {"feed": feed} if feed else {}
        with log_stage("topic", **start_fields) as span:
            start = perf_counter()
            texts = [e.content_head or f"{e.title}\n{e.description}" for e in entries]
            labels_per_entry = self._infer_batch(texts, threshold)
            total_labels = sum(len(labels) for labels in labels_per_entry)

            elapsed = perf_counter() - start
            n = len(entries)
            articles_per_sec = round(n / elapsed, 3) if elapsed > 0 else 0.0
            avg_labels = round(total_labels / n, 2) if n > 0 else 0.0
            span.set(
                entries=n,
                labels=total_labels,
                avg_labels_per_entry=avg_labels,
                threshold=threshold,
                vocabulary=len(self.labels),
                batch_size=settings.topic_batch_size,
                articles_per_sec=articles_per_sec,
            )
            return labels_per_entry

    def classify_summaries(
        self,
        summaries: list[Summary],
        threshold: float | None = None,
        *,
        feed: str | None = None,
    ) -> list[Summary]:
        """Classify based on title + description + summary (backward-compatible).

        Returns new ``Summary`` objects with ``topics`` populated; the
        originals are not modified. Provided for direct programmatic callers;
        the production path uses ``classify_entries`` via ``api.label``.

        Args:
            summaries: Summaries to classify.
            threshold: Minimum entailment probability to accept a label.
            feed: Optional tag rendered on ``stage.start`` / ``stage.done``
                so concurrent per-feed invocations are distinguishable.

        Returns:
            New list of summaries with ``topics`` populated.
        """
        threshold = threshold if threshold is not None else settings.topic_threshold

        start_fields = {"feed": feed} if feed else {}
        with log_stage("topic", **start_fields) as span:
            start = perf_counter()
            texts = [f"{s.entry.title}\n{s.entry.description}\n{s.summary}" for s in summaries]
            labels_per = self._infer_batch(texts, threshold)
            labeled = [
                replace(s, topics=labels) for s, labels in zip(summaries, labels_per, strict=True)
            ]
            total_labels = sum(len(labels) for labels in labels_per)

            elapsed = perf_counter() - start
            n = len(labeled)
            articles_per_sec = round(n / elapsed, 3) if elapsed > 0 else 0.0
            avg_labels = round(total_labels / n, 2) if n > 0 else 0.0
            span.set(
                summaries=n,
                labels=total_labels,
                avg_labels_per_summary=avg_labels,
                threshold=threshold,
                vocabulary=len(self.labels),
                batch_size=settings.topic_batch_size,
                articles_per_sec=articles_per_sec,
            )
            return labeled

    def _infer(self, text: str, threshold: float) -> list[Label]:
        """Run NLI on a single text; thin wrapper over the batched path."""
        return self._infer_batch([text], threshold)[0]

    def _infer_batch(self, texts: list[str], threshold: float) -> list[list[Label]]:
        """Score many texts against every ``TopicType`` hypothesis, batched.

        Each text is paired with all label hypotheses; texts are grouped so that
        at most ``settings.topic_batch_size`` (premise, hypothesis) pairs go
        through the model per forward pass, cutting the pass count by roughly the
        texts-per-chunk factor versus one text at a time. A text's full set of
        hypotheses always stays within one chunk, so the logits reshape back to
        one row per text is exact. Returns one ``list[Label]`` per input text, in
        order.
        """
        if not texts:
            return []
        hypotheses = [f"This article is about {lab.replace('-', ' ')}." for lab in self.labels]
        n_labels = len(hypotheses)
        texts_per_chunk = max(1, settings.topic_batch_size // n_labels)

        results: list[list[Label]] = []
        for start in range(0, len(texts), texts_per_chunk):
            chunk = texts[start : start + texts_per_chunk]
            premises = [t for t in chunk for _ in range(n_labels)]
            hyps = hypotheses * len(chunk)

            inputs = self.tokenizer(
                premises,
                hyps,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=settings.topic_max_length,
            )

            with torch.no_grad():
                logits = self.model(**inputs).logits

            # ``entail[i, j]`` is text i's entailment probability for label j.
            entail = torch.softmax(logits, dim=1)[:, 2].view(len(chunk), n_labels)
            for row in entail:
                scored = sorted(
                    ((lab, float(score)) for lab, score in zip(self.labels, row, strict=True)),
                    key=lambda x: x[1],
                    reverse=True,
                )
                chosen = [(lab, sc) for lab, sc in scored if sc >= threshold] or [scored[0]]
                results.append(
                    [Label(value=TopicType(lab), confidence=float(sc)) for lab, sc in chosen]
                )
        return results
