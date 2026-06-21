"""LLM-driven fact-extraction summarizer for article entries.

A constrained LLM call (Ollama) produces a fact-dense 2-4 sentence
summary per article. The summary string is consumed by
``TopicClassifier`` (zero-shot NLI tagging) and surfaces in the digest
writer's ``<summary>`` source-signal slot. The output schema
(``Summary.summary`` as a single string) keeps the per-feed JSON files
simple and stable across runs.

Concurrency: ``summarize_entries`` fans out across articles via
``asyncio.gather``, capped by an instance-level ``asyncio.Semaphore`` sized
to ``summarizer_concurrency`` (default 8). The semaphore is **shared
across every** ``summarize_entries`` **call on the same instance**. The
pipeline calls this once per feed, so a per-call semaphore would multiply
the in-flight ceiling by the number of feeds running concurrently and
trip cloud Ollama's per-account rate limit (HTTP 429). Each per-article
LLM call is sync (Ollama's native client is blocking) and is shipped to
a thread via ``asyncio.to_thread`` so the event loop is never blocked.
``digest_generator.api.summarize`` awaits this directly without a further
``asyncio.to_thread`` wrap.

Telemetry: the run wraps in ``log_stage("summarizer")``; per-call LLM
metrics (``prompt_tokens``, ``completion_tokens``, ``llm_calls``,
``llm_duration_ms``) accumulate into the active ``StageSpan`` via
``chat_with_logging``. Fallback events emit a WARNING with the entry title.
"""

from __future__ import annotations

import asyncio

from ollama import Client

from digest_generator.core.summary.prompts import load_prompt
from digest_generator.core.types import Entry, Summary
from digest_generator.shared.llm.clients import client_registry
from digest_generator.shared.llm.sampling import SamplingConfig, resolve_ollama_options
from digest_generator.shared.llm.telemetry import chat_with_logging
from digest_generator.shared.logging import log_stage, logger
from digest_generator.shared.settings import settings

_SYSTEM_PROMPT = load_prompt("article_summary_system")

_FALLBACK_CONTENT_HEAD_CHARS = 1000


class ContentSummarizer:
    """Generate fact-dense per-article summaries via an LLM call.

    Args:
        client: Ollama client instance. Defaults to ``client_registry.ollama``
                (cloud if ``OLLAMA_API_KEY`` is set, else local).
        model: Ollama model name. Defaults to ``settings.summarizer_model``.
        sampling: Per-call sampling overrides (``temperature``, ``top_p``,
                  ``repetition_penalty``, ``seed``). Unset fields fall back
                  to the matching ``summarizer_*`` settings.
    """

    def __init__(
        self,
        client: Client | None = None,
        model: str | None = None,
        sampling: SamplingConfig | None = None,
    ):
        self._client = client or client_registry.ollama
        self.model = model or settings.summarizer_model
        self._sampling = sampling
        self._concurrency = settings.summarizer_concurrency
        # Lazy-init on first call. `asyncio.Semaphore()` is safe to construct
        # outside an event loop in 3.10+, but lazy init keeps the binding to
        # the running loop unambiguous when this instance survives across
        # multiple `asyncio.run()` invocations (mostly tests).
        self._semaphore: asyncio.Semaphore | None = None

    async def summarize_entries(
        self, entries: list[Entry], *, feed: str | None = None
    ) -> list[Summary]:
        """Summarize every entry concurrently, capped by ``summarizer_concurrency``.

        Order is preserved across the input. Empty input short-circuits.
        The concurrency cap is shared across every call on this instance
        (see the module docstring for why).

        ``feed`` (optional) tags the ``stage.start`` / ``stage.done`` lines
        so concurrent per-feed invocations are distinguishable in run.log.
        """
        start_fields = {"feed": feed} if feed else {}
        with log_stage("summarizer", **start_fields) as span:
            if not entries:
                span.set(entries=0, model=self.model)
                return []

            if self._semaphore is None:
                self._semaphore = asyncio.Semaphore(self._concurrency)
            semaphore = self._semaphore

            async def _bounded(entry: Entry) -> Summary:
                async with semaphore:
                    return await self.summarize_entry(entry)

            summaries = await asyncio.gather(*(_bounded(entry) for entry in entries))

            output_chars = sum(len(s.summary) for s in summaries)
            span.set(
                entries=len(entries),
                output_chars=output_chars,
                model=self.model,
                concurrency=self._concurrency,
            )
            return summaries

    async def summarize_entry(self, entry: Entry) -> Summary:
        """Summarize one entry; on empty LLM response, fall back to source text.

        Fallback order: ``content_head[:1000]`` if present, else
        ``description``. Always returns a non-None ``Summary``.
        """
        user_prompt = self._build_user_prompt(entry)
        text = (await asyncio.to_thread(self._call_llm, user_prompt)).strip()
        if not text:
            text = self._fallback_summary(entry)
            logger.warning("summarizer.fallback entry={!r} chars={}", entry.title, len(text))
        return Summary(entry=entry, summary=text, length=len(text), topics=[])

    def _call_llm(self, user_prompt: str) -> str:
        options = resolve_ollama_options(
            self._sampling,
            temperature=settings.summarizer_temperature,
            top_p=settings.summarizer_top_p,
            repetition_penalty=settings.summarizer_repetition_penalty,
            seed=settings.summarizer_seed,
        )
        return chat_with_logging(
            self._client,
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options=options,
        )

    @staticmethod
    def _build_user_prompt(entry: Entry) -> str:
        """Format one Entry as an XML-tag block, mirroring writer._format_articles.

        ``<content_head>`` is omitted because it is a derived truncation of
        ``content``; sending both is redundant when the LLM already sees
        the full content. Empty optional fields are omitted entirely so
        the prompt does not signal "this article has no description" when
        in fact none was provided.
        """
        lines = [
            "<article>",
            f"  <title>{entry.title}</title>",
            f"  <url>{entry.url}</url>",
            f"  <origin>{entry.origin}</origin>",
            f"  <published>{entry.published.isoformat()}</published>",
        ]
        description = entry.description.strip()
        if description:
            lines.append(f"  <description>{description}</description>")
        content = entry.content.strip()
        if content:
            lines.append(f"  <content>{content}</content>")
        lines.append("</article>")
        return "\n".join(lines)

    @staticmethod
    def _fallback_summary(entry: Entry) -> str:
        """Empty-LLM fallback: ``content_head[:1000]`` if present, else description."""
        content_head = entry.content_head.strip()
        if content_head:
            return content_head[:_FALLBACK_CONTENT_HEAD_CHARS]
        return entry.description.strip()
