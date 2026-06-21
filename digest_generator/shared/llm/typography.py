"""Post-LLM typography normalization for digest-pipeline output.

Some Ollama-served models (notably ``gpt-oss:120b``) reliably substitute
typographic Unicode characters for their plain ASCII equivalents inside
compound terms, e.g. U+2011 (NON-BREAKING HYPHEN) for U+002D (HYPHEN-MINUS)
in "AI-ready", "Policy-as-Code", or "non-human". The output renders identically
in browsers but breaks grep, copy-paste, and downstream string comparisons.

Each LLM-calling stage that surfaces text into the published digest applies
``normalize_typography`` immediately after the model response (before
validation, parsing, or persistence). The strip table is intentionally small
and curated: it only swaps replacements that have a single unambiguous ASCII
equivalent and that the model substitutes systematically.
"""

# (replacement, replacement) pairs applied left-to-right.
# Left side is the typographic Unicode character; right side is plain ASCII.
_TYPOGRAPHIC_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("‑", "-"),  # noqa: RUF001 (left side is the literal U+2011 we are stripping)
)


def normalize_typography(text: str) -> str:
    """Replace systematic LLM typography artifacts with ASCII equivalents."""
    for old, new in _TYPOGRAPHIC_REPLACEMENTS:
        text = text.replace(old, new)
    return text
