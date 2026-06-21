"""Per-section editorial cleanup pass for the weekly digest pipeline.

``SectionEditor`` takes raw ``SectionDraft`` output from ``SectionWriter`` and
runs a constrained LLM cleanup pass that removes LLM tics while preserving
every Markdown link, heading, and factual claim. A structural validator falls
back to the original draft when the edited output drops or rewrites links,
mangles the H2 heading, or diverges too far in length.

Configured via ``EDITORIAL_MODEL`` environment variable.

Telemetry: every validator outcome emits a structured ``editor.validator.*``
DEBUG line (``rejected check=<reason> ...`` or ``passed delta_words=...``)
into ``run.log``, and bumps a per-reason counter on the active stage span so
the ``editor`` ``stage.done`` line carries ``rejected_link_set``,
``rejected_h2_heading``, and ``rejected_length_delta`` totals alongside the
existing ``rewritten`` / ``fell_back`` counts. This lets a digest-quality
audit grep ``run.log`` to answer "did the editorial LLM try to fix this
forbidden phrase, and was the fix rejected?" without reading the cached
``sections/`` and ``sections_edited/`` directories side-by-side.
"""

from __future__ import annotations

import re
from typing import Any

from ollama import Client

from digest_generator.core.digest.prompts import load_prompt
from digest_generator.core.digest.types import SectionDraft
from digest_generator.shared.llm.clients import client_registry
from digest_generator.shared.llm.sampling import SamplingConfig, resolve_ollama_options
from digest_generator.shared.llm.telemetry import chat_with_logging
from digest_generator.shared.llm.typography import normalize_typography
from digest_generator.shared.logging import current_span, log_stage, logger
from digest_generator.shared.runtime.meta import RejectedReason, SectionMeta
from digest_generator.shared.settings import settings

_EDITORIAL_SYSTEM_PROMPT = load_prompt("editorial_pass_system")

_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)

_LENGTH_DELTA_MAX = 0.30


class SectionEditor:
    """Cleans ``SectionDraft`` prose via a constrained LLM rewrite pass.

    Connects to an Ollama instance (local or cloud) via the native ``ollama``
    client. For each draft, sends the section prose to the configured model
    with the editorial system prompt, then runs a structural validator on the
    response. If the validator rejects the output, the original draft is
    returned unchanged.

    Args:
        client: Ollama client instance. Defaults to auto-detected client from
                ``ClientRegistry`` (cloud if ``OLLAMA_API_KEY`` is set, else local).
        model: Ollama model name. Defaults to ``EDITORIAL_MODEL`` env var.
        sampling: Per-call sampling overrides (``temperature``, ``top_p``,
                ``repetition_penalty``, ``seed``). Unset fields fall back to
                the matching ``editorial_*`` settings.
    """

    def __init__(
        self,
        client: Client | None = None,
        model: str | None = None,
        sampling: SamplingConfig | None = None,
    ):
        self._client = client or client_registry.ollama
        self.model = model or settings.editorial_model
        self._sampling = sampling
        self._section_outcomes: list[SectionMeta] = []

    @property
    def section_outcomes(self) -> list[SectionMeta]:
        """Per-section editor outcomes from the most recent ``clean_all`` call.

        Populated by ``clean`` as each section is processed and reset at the
        start of each ``clean_all``. The orchestrator harvests this list to
        populate ``meta.json``'s ``sections`` block.
        """
        return list(self._section_outcomes)

    def clean_all(self, drafts: list[SectionDraft]) -> list[SectionDraft]:
        """Run the editorial pass on every draft, preserving order."""
        self._section_outcomes = []
        with log_stage("editor") as span:
            cleaned = [self.clean(draft) for draft in drafts]
            kept_original = sum(
                1 for before, after in zip(drafts, cleaned, strict=True) if before is after
            )
            span.set(
                sections=len(cleaned),
                rewritten=len(cleaned) - kept_original,
                fell_back=kept_original,
                model=self.model,
            )
            return cleaned

    def clean(self, draft: SectionDraft) -> SectionDraft:
        """Run a single editorial pass, falling back to the original on failure."""
        logger.info(
            "Editorial pass for {} section ({} words) via {}",
            draft.name,
            len(draft.content.split()),
            self.model,
        )
        edited = self._call_llm(_EDITORIAL_SYSTEM_PROMPT, self._build_user_prompt(draft))
        if not edited:
            logger.warning("Editorial pass returned empty for {}; keeping original", draft.name)
            self._record_outcome(draft, "fell_back", "empty_response")
            return draft
        edited = normalize_typography(edited)
        rejection = self._validate(draft, edited)
        if rejection is not None:
            self._record_outcome(draft, "fell_back", rejection)
            return draft
        self._record_outcome(draft, "rewritten", None)
        return SectionDraft(name=draft.name, content=edited, article_count=draft.article_count)

    def _record_outcome(
        self,
        draft: SectionDraft,
        outcome: str,
        reason: RejectedReason | None,
    ) -> None:
        """Append the section's outcome for the orchestrator to harvest."""
        self._section_outcomes.append(
            SectionMeta(
                name=draft.name,
                articles=draft.article_count,
                edit_outcome=outcome,  # type: ignore[arg-type]
                rejected_reason=reason,
            )
        )

    @staticmethod
    def _build_user_prompt(draft: SectionDraft) -> str:
        return "\n".join(
            [
                "<task>",
                f'Run the editorial cleanup pass on the "{draft.name}" section below.',
                "Remove LLM tics; preserve every link, heading, number, and factual claim.",
                "</task>",
                "",
                "<section-draft>",
                draft.content,
                "</section-draft>",
            ]
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        options = resolve_ollama_options(
            self._sampling,
            temperature=settings.editorial_temperature,
            top_p=settings.editorial_top_p,
            repetition_penalty=settings.editorial_repetition_penalty,
            seed=settings.editorial_seed,
        )
        return chat_with_logging(
            self._client,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options=options,
        )

    @staticmethod
    def _validate(original: SectionDraft, edited: str) -> RejectedReason | None:
        """Return ``None`` when the edited output passes; the rejection reason otherwise.

        Every outcome (one rejection or pass per call) emits a structured
        ``editor.validator.*`` DEBUG line and, when called inside a
        ``log_stage("editor")`` block, bumps a per-reason counter on the
        active span. Static-method invocations outside a span (typical in
        unit tests) silently skip the span update. Returning the reason
        rather than a bool lets ``clean()`` thread it into the per-section
        outcome record without re-running the validator logic.
        """
        original_links = set(_MARKDOWN_LINK_RE.findall(original.content))
        edited_links = set(_MARKDOWN_LINK_RE.findall(edited))
        if original_links != edited_links:
            dropped = original_links - edited_links
            added = edited_links - original_links
            _log_rejection("link_set", original.name, dropped=len(dropped), added=len(added))
            return "link_set"

        original_h2s = set(_H2_RE.findall(original.content))
        edited_h2s = set(_H2_RE.findall(edited))
        if original_h2s != edited_h2s:
            _log_rejection(
                "h2_heading",
                original.name,
                original=sorted(original_h2s),
                edited=sorted(edited_h2s),
            )
            return "h2_heading"

        original_words = len(original.content.split())
        edited_words = len(edited.split())
        if original_words == 0:
            _log_pass(original.name, delta_words=edited_words)
            return None
        delta = abs(edited_words - original_words) / original_words
        if delta > _LENGTH_DELTA_MAX:
            _log_rejection(
                "length_delta",
                original.name,
                delta_pct=int(delta * 100),
                max_pct=int(_LENGTH_DELTA_MAX * 100),
            )
            return "length_delta"

        _log_pass(original.name, delta_words=edited_words - original_words)
        return None


def _log_rejection(check: str, section: str, **fields: Any) -> None:
    """Emit a structured DEBUG rejection line and bump the per-reason span counter."""
    extras = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.debug("editor.validator.rejected check={} section={!r} {}", check, section, extras)
    span = current_span()
    if span is not None:
        span.add(**{f"rejected_{check}": 1})


def _log_pass(section: str, *, delta_words: int) -> None:
    """Emit a structured DEBUG pass line; no span counter (acceptance is the default)."""
    logger.debug("editor.validator.passed section={!r} delta_words={}", section, delta_words)
