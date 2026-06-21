"""Per-call LLM sampling knobs and the Ollama options-dict resolver.

Used by every LLM call site (summarizer + digest stages). The ``SamplingConfig``
dataclass and ``resolve_ollama_options`` helper are pure data / pure functions.
``fetch_model_defaults`` makes one network call per unique model (cached) to
capture the Modelfile-pinned parameters Ollama would otherwise silently apply.
``materialize_sampling`` resolves the user / settings / random layers into a
single ``SamplingConfig`` whose seed is always concrete, so a run with a
``None``-resolved seed can still be reproduced by reading the materialized
seed back out of ``meta.json``.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from digest_generator.shared.logging import logger

if TYPE_CHECKING:
    from ollama import Client


@dataclass(frozen=True)
class SamplingConfig:
    """Per-stage LLM sampling knobs passed to Ollama.

    All fields are ``None`` by default; callers layer overrides on top of
    ``Settings`` defaults, and ``None`` means "fall back to the settings
    value (or Ollama default) for that field." The user-facing name
    ``repetition_penalty`` maps to Ollama's ``repeat_penalty`` option key.
    """

    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None

    def to_options(self, *, default_temperature: float) -> dict[str, Any]:
        """Build the Ollama ``options`` dict, omitting unset fields.

        ``temperature`` always appears (using ``default_temperature`` when
        unset) so every stage carries an explicit temperature. Other fields
        are included only when explicitly set, so Ollama falls back to its
        own defaults otherwise.
        """
        opts: dict[str, Any] = {
            "temperature": self.temperature if self.temperature is not None else default_temperature
        }
        if self.top_p is not None:
            opts["top_p"] = self.top_p
        if self.repetition_penalty is not None:
            opts["repeat_penalty"] = self.repetition_penalty
        if self.seed is not None:
            opts["seed"] = self.seed
        return opts

    def with_defaults(self, defaults: SamplingConfig) -> SamplingConfig:
        """Return a copy where unset (``None``) fields are filled from ``defaults``."""
        return SamplingConfig(
            temperature=self.temperature if self.temperature is not None else defaults.temperature,
            top_p=self.top_p if self.top_p is not None else defaults.top_p,
            repetition_penalty=(
                self.repetition_penalty
                if self.repetition_penalty is not None
                else defaults.repetition_penalty
            ),
            seed=self.seed if self.seed is not None else defaults.seed,
        )

    def with_seed_default(self, seed: int | None) -> SamplingConfig:
        """Return a copy with ``seed`` populated if it was unset.

        Used by the CLI's ``--seed`` shortcut: applies a global seed to any
        stage that doesn't have its own per-stage seed override.
        """
        if seed is None or self.seed is not None:
            return self
        return SamplingConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            seed=seed,
        )


def resolve_ollama_options(
    sampling: SamplingConfig | None,
    *,
    temperature: float,
    top_p: float | None,
    repetition_penalty: float | None,
    seed: int | None,
) -> dict[str, Any]:
    """Build the Ollama ``options`` dict from a stage's sampling override + settings defaults.

    ``sampling`` carries CLI / API overrides (any field may be ``None``); the
    keyword arguments carry the stage's settings-level defaults. Fields unset
    in both layers are omitted so Ollama uses its own defaults.
    """
    return (
        (sampling or SamplingConfig())
        .with_defaults(
            SamplingConfig(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )
        )
        .to_options(default_temperature=temperature)
    )


# Ollama's documented sampling defaults. Used as a per-key floor when
# ``ollama.show()`` doesn't return a Modelfile parameter for the model
# (cloud-hosted models routinely omit some). These are the values Ollama
# applies when a request includes nothing for the field. The ``repeat_penalty``
# key matches Ollama's wire format; ``materialize_sampling`` translates to/from
# the user-facing ``repetition_penalty`` name.
OLLAMA_DEFAULTS: dict[str, float | int] = {
    "temperature": 0.8,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "seed": 0,
}

# Per-process cache for ``fetch_model_defaults``. Keyed by model name; in
# practice one client per process, so model-only keying is enough. Reset
# helpers are exposed for tests (``_reset_model_defaults_cache``).
_MODEL_DEFAULTS_CACHE: dict[str, dict[str, float | int]] = {}


SeedSource = Literal["user", "settings", "cli", "random"]


def _parse_modelfile_parameters(params_str: str) -> dict[str, float | int]:
    """Parse Ollama's Modelfile ``parameters`` block.

    The ``ollama show`` API returns parameters as newline-separated
    ``param value`` lines. Only sampling-relevant keys are kept; unparseable
    or non-numeric entries are skipped silently (the Modelfile may carry
    ``stop "..."`` lines and other non-sampling directives).
    """
    keep = {"temperature", "top_p", "repeat_penalty", "seed"}
    out: dict[str, float | int] = {}
    for raw in params_str.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            key, val = line.split(maxsplit=1)
        except ValueError:
            continue
        if key not in keep:
            continue
        val = val.strip().strip('"')
        try:
            parsed = float(val) if "." in val else int(val)
        except ValueError:
            continue
        out[key] = parsed
    return out


def fetch_model_defaults(client: Client, model: str) -> dict[str, float | int]:
    """Return the model's effective sampling defaults (Modelfile + Ollama floor).

    Calls ``client.show(model)`` once per model per process and caches the
    result. The Ollama Modelfile may pin ``temperature`` / ``top_p`` /
    ``repeat_penalty`` / ``seed``; values not pinned fall back to
    ``OLLAMA_DEFAULTS``. On any error (network failure, cloud model that
    doesn't expose parameters), returns ``OLLAMA_DEFAULTS`` and logs a
    warning, so the caller still gets a complete dict.
    """
    if model in _MODEL_DEFAULTS_CACHE:
        return _MODEL_DEFAULTS_CACHE[model]
    parsed: dict[str, float | int] = {}
    try:
        resp = client.show(model)
        # ``ollama.show`` returns a ShowResponse; access via attribute or dict-get.
        params_str = getattr(resp, "parameters", None) or (
            resp.get("parameters") if isinstance(resp, dict) else None
        )
        if params_str:
            parsed = _parse_modelfile_parameters(params_str)
    except Exception as exc:
        logger.warning(
            "ollama.show failed for {}: {} — falling back to OLLAMA_DEFAULTS",
            model,
            exc,
        )
    result: dict[str, float | int] = {**OLLAMA_DEFAULTS, **parsed}
    _MODEL_DEFAULTS_CACHE[model] = result
    return result


def _reset_model_defaults_cache() -> None:
    """Clear the per-process model-defaults cache. For tests."""
    _MODEL_DEFAULTS_CACHE.clear()


def materialize_sampling(
    user: SamplingConfig | None,
    *,
    default_temperature: float | None,
    default_top_p: float | None,
    default_repetition_penalty: float | None,
    default_seed: int | None,
    cli_seed: int | None = None,
) -> tuple[SamplingConfig, SeedSource]:
    """Resolve the user/settings/cli/random layers into a concrete ``SamplingConfig``.

    Resolution order per field: user override (``user.<field>``), then settings
    default (``default_<field>``). Seed gets one extra layer: ``cli_seed`` is
    applied when the user didn't set a per-stage seed, and a random
    31-bit int is materialized if every layer leaves the seed ``None``.
    The returned ``SeedSource`` records which layer supplied the seed so
    ``meta.json`` can describe how a run's reproducibility was established.

    The 31-bit width fits Ollama's signed integer seed without wrap-around
    surprises across SDK versions; ``secrets.randbits`` is preferred over
    ``random.randint`` so test seeding can't accidentally pin run reproducibility.
    """
    layered = (user or SamplingConfig()).with_defaults(
        SamplingConfig(
            temperature=default_temperature,
            top_p=default_top_p,
            repetition_penalty=default_repetition_penalty,
            seed=default_seed,
        )
    )

    if user is not None and user.seed is not None:
        seed_source: SeedSource = "user"
        seed_value: int = user.seed
    elif default_seed is not None:
        seed_source = "settings"
        seed_value = default_seed
    elif cli_seed is not None:
        seed_source = "cli"
        seed_value = cli_seed
    else:
        seed_source = "random"
        seed_value = secrets.randbits(31)

    return (
        SamplingConfig(
            temperature=layered.temperature,
            top_p=layered.top_p,
            repetition_penalty=layered.repetition_penalty,
            seed=seed_value,
        ),
        seed_source,
    )
