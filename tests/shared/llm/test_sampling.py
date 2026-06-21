"""Tests for digest_generator/shared/llm/sampling.py: SamplingConfig, resolve_ollama_options, model defaults, and seed materialization."""

from unittest.mock import MagicMock

import pytest

from digest_generator.shared.llm.sampling import (
    OLLAMA_DEFAULTS,
    SamplingConfig,
    _parse_modelfile_parameters,
    _reset_model_defaults_cache,
    fetch_model_defaults,
    materialize_sampling,
    resolve_ollama_options,
)


@pytest.fixture(autouse=True)
def _clear_model_defaults_cache():
    """Each test starts with a clean per-process cache."""
    _reset_model_defaults_cache()
    yield
    _reset_model_defaults_cache()


class TestSamplingConfigToOptions:
    """SamplingConfig.to_options(): the Ollama options-dict builder."""

    def test_temperature_uses_default_when_unset(self):
        opts = SamplingConfig().to_options(default_temperature=0.4)
        assert opts == {"temperature": 0.4}

    def test_explicit_temperature_overrides_default(self):
        opts = SamplingConfig(temperature=0.9).to_options(default_temperature=0.4)
        assert opts == {"temperature": 0.9}

    def test_top_p_included_only_when_set(self):
        opts = SamplingConfig(top_p=0.85).to_options(default_temperature=0.4)
        assert opts == {"temperature": 0.4, "top_p": 0.85}

    def test_repetition_penalty_maps_to_repeat_penalty(self):
        opts = SamplingConfig(repetition_penalty=1.15).to_options(default_temperature=0.4)
        assert opts == {"temperature": 0.4, "repeat_penalty": 1.15}
        assert "repetition_penalty" not in opts

    def test_seed_included_only_when_set(self):
        opts = SamplingConfig(seed=42).to_options(default_temperature=0.4)
        assert opts == {"temperature": 0.4, "seed": 42}

    def test_all_fields_set(self):
        opts = SamplingConfig(
            temperature=0.7, top_p=0.9, repetition_penalty=1.2, seed=7
        ).to_options(default_temperature=0.4)
        assert opts == {
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.2,
            "seed": 7,
        }


class TestSamplingConfigWithDefaults:
    """SamplingConfig.with_defaults(): layered fallback semantics."""

    def test_unset_fields_filled_from_defaults(self):
        result = SamplingConfig().with_defaults(
            SamplingConfig(temperature=0.4, top_p=0.8, repetition_penalty=1.1, seed=5)
        )
        assert result == SamplingConfig(temperature=0.4, top_p=0.8, repetition_penalty=1.1, seed=5)

    def test_set_fields_preserved(self):
        result = SamplingConfig(temperature=0.9, seed=99).with_defaults(
            SamplingConfig(temperature=0.4, top_p=0.8, seed=5)
        )
        assert result == SamplingConfig(temperature=0.9, top_p=0.8, seed=99)


class TestSamplingConfigWithSeedDefault:
    """SamplingConfig.with_seed_default(): CLI --seed shortcut."""

    def test_no_seed_returns_self_unchanged(self):
        cfg = SamplingConfig(temperature=0.4)
        assert cfg.with_seed_default(None) is cfg

    def test_per_stage_seed_wins_over_global(self):
        cfg = SamplingConfig(seed=10)
        assert cfg.with_seed_default(99).seed == 10

    def test_global_seed_applied_when_per_stage_unset(self):
        cfg = SamplingConfig(top_p=0.8)
        out = cfg.with_seed_default(99)
        assert out.seed == 99
        assert out.top_p == 0.8


class TestResolveOllamaOptions:
    """resolve_ollama_options(): the helper stages call from _call_llm."""

    def test_settings_defaults_used_when_no_override(self):
        opts = resolve_ollama_options(
            None,
            temperature=0.4,
            top_p=None,
            repetition_penalty=None,
            seed=None,
        )
        assert opts == {"temperature": 0.4}

    def test_settings_provide_top_p_default(self):
        opts = resolve_ollama_options(
            None,
            temperature=0.4,
            top_p=0.85,
            repetition_penalty=None,
            seed=None,
        )
        assert opts == {"temperature": 0.4, "top_p": 0.85}

    def test_override_wins_over_settings(self):
        opts = resolve_ollama_options(
            SamplingConfig(top_p=0.95, seed=7),
            temperature=0.4,
            top_p=0.85,
            repetition_penalty=1.1,
            seed=None,
        )
        # top_p overridden, seed overridden, repetition_penalty falls through to settings
        assert opts == {
            "temperature": 0.4,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "seed": 7,
        }

    def test_repetition_penalty_keyed_as_repeat_penalty(self):
        opts = resolve_ollama_options(
            SamplingConfig(repetition_penalty=1.3),
            temperature=0.4,
            top_p=None,
            repetition_penalty=None,
            seed=None,
        )
        assert opts == {"temperature": 0.4, "repeat_penalty": 1.3}


class TestParseModelfileParameters:
    """_parse_modelfile_parameters(): Ollama Modelfile parameter-block parser."""

    def test_parses_known_keys(self):
        block = "temperature 0.7\ntop_p 0.92\nrepeat_penalty 1.05\nseed 12345"
        assert _parse_modelfile_parameters(block) == {
            "temperature": 0.7,
            "top_p": 0.92,
            "repeat_penalty": 1.05,
            "seed": 12345,
        }

    def test_skips_unknown_keys(self):
        block = 'stop "<|eot_id|>"\nstop "<|user|>"\ntemperature 0.5\nnum_ctx 8192'
        # num_ctx and stop entries are not sampling-relevant
        assert _parse_modelfile_parameters(block) == {"temperature": 0.5}

    def test_skips_empty_and_malformed_lines(self):
        block = "\n  \ntemperature\ntop_p 0.9\n"
        assert _parse_modelfile_parameters(block) == {"top_p": 0.9}

    def test_strips_quoted_values(self):
        block = 'temperature "0.7"\nseed "42"'
        assert _parse_modelfile_parameters(block) == {"temperature": 0.7, "seed": 42}

    def test_empty_input_yields_empty_dict(self):
        assert _parse_modelfile_parameters("") == {}


class TestFetchModelDefaults:
    """fetch_model_defaults(): caches Modelfile params per model, falls back on error."""

    def test_returns_modelfile_params_layered_over_ollama_defaults(self):
        client = MagicMock()
        # Modelfile pins temperature only; top_p / repeat_penalty / seed inherit from OLLAMA_DEFAULTS.
        client.show.return_value = {"parameters": "temperature 0.7"}
        result = fetch_model_defaults(client, "writer-model")
        assert result == {**OLLAMA_DEFAULTS, "temperature": 0.7}

    def test_caches_per_model(self):
        client = MagicMock()
        client.show.return_value = {"parameters": "temperature 0.5"}
        fetch_model_defaults(client, "model-a")
        fetch_model_defaults(client, "model-a")
        # Cache hit on the second call; no extra network call.
        assert client.show.call_count == 1

    def test_distinct_models_cache_separately(self):
        client = MagicMock()
        client.show.side_effect = [
            {"parameters": "temperature 0.5"},
            {"parameters": "temperature 0.9"},
        ]
        a = fetch_model_defaults(client, "model-a")
        b = fetch_model_defaults(client, "model-b")
        assert a["temperature"] == 0.5
        assert b["temperature"] == 0.9
        assert client.show.call_count == 2

    def test_falls_back_to_ollama_defaults_on_show_error(self):
        client = MagicMock()
        client.show.side_effect = RuntimeError("connection refused")
        result = fetch_model_defaults(client, "broken-model")
        assert result == OLLAMA_DEFAULTS

    def test_falls_back_when_parameters_field_empty(self):
        client = MagicMock()
        client.show.return_value = {"parameters": ""}
        result = fetch_model_defaults(client, "no-params-model")
        assert result == OLLAMA_DEFAULTS

    def test_handles_attribute_style_response(self):
        # Ollama SDK returns ShowResponse (object with .parameters), not a plain dict.
        resp = MagicMock(spec=["parameters"])
        resp.parameters = "temperature 0.6\ntop_p 0.85"
        client = MagicMock()
        client.show.return_value = resp
        result = fetch_model_defaults(client, "obj-model")
        assert result["temperature"] == 0.6
        assert result["top_p"] == 0.85


class TestMaterializeSampling:
    """materialize_sampling(): three-layer resolution + concrete seed."""

    def test_user_seed_wins_recorded_as_user(self):
        cfg, src = materialize_sampling(
            SamplingConfig(seed=7),
            default_temperature=0.5,
            default_top_p=None,
            default_repetition_penalty=None,
            default_seed=42,
            cli_seed=99,
        )
        assert cfg.seed == 7
        assert src == "user"

    def test_settings_seed_when_user_unset(self):
        cfg, src = materialize_sampling(
            SamplingConfig(),
            default_temperature=0.5,
            default_top_p=None,
            default_repetition_penalty=None,
            default_seed=42,
            cli_seed=99,
        )
        assert cfg.seed == 42
        assert src == "settings"

    def test_cli_seed_when_user_and_settings_unset(self):
        cfg, src = materialize_sampling(
            None,
            default_temperature=0.5,
            default_top_p=None,
            default_repetition_penalty=None,
            default_seed=None,
            cli_seed=99,
        )
        assert cfg.seed == 99
        assert src == "cli"

    def test_random_seed_materialized_when_all_layers_unset(self):
        cfg, src = materialize_sampling(
            None,
            default_temperature=0.5,
            default_top_p=None,
            default_repetition_penalty=None,
            default_seed=None,
            cli_seed=None,
        )
        assert cfg.seed is not None
        assert 0 <= cfg.seed < 2**31
        assert src == "random"

    def test_random_seed_is_distinct_across_calls(self):
        # Two unseeded calls produce different seeds (with overwhelming probability).
        seeds = {
            materialize_sampling(
                None,
                default_temperature=0.5,
                default_top_p=None,
                default_repetition_penalty=None,
                default_seed=None,
                cli_seed=None,
            )[0].seed
            for _ in range(5)
        }
        assert len(seeds) > 1

    def test_non_seed_fields_layer_user_over_settings(self):
        cfg, _ = materialize_sampling(
            SamplingConfig(temperature=0.9, top_p=0.85),
            default_temperature=0.5,
            default_top_p=0.92,
            default_repetition_penalty=1.1,
            default_seed=42,
        )
        assert cfg.temperature == 0.9
        assert cfg.top_p == 0.85
        assert cfg.repetition_penalty == 1.1


class TestOllamaDefaults:
    """OLLAMA_DEFAULTS: the documented Ollama floor used as fallback."""

    def test_carries_all_four_sampling_keys(self):
        assert set(OLLAMA_DEFAULTS) == {"temperature", "top_p", "repeat_penalty", "seed"}

    def test_documented_values(self):
        assert OLLAMA_DEFAULTS["temperature"] == 0.8
        assert OLLAMA_DEFAULTS["top_p"] == 0.9
        assert OLLAMA_DEFAULTS["repeat_penalty"] == 1.1
        assert OLLAMA_DEFAULTS["seed"] == 0
