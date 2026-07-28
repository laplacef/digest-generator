"""Tests for the deterministic post-composer normalization pass."""

from __future__ import annotations

import pytest

from digest_generator.core.digest.normalize import mask_protected_spans, normalize_output


class TestLocaleNormalization:
    @pytest.mark.parametrize(
        ("src", "want"),
        [
            ("37 % of runs", "37% of runs"),
            ("up 12  % year over year", "up 12% year over year"),
            ("the centre of gravity", "the center of gravity"),
            ("Centre of Excellence", "Center of Excellence"),  # case preserved
            ("CENTRE", "CENTER"),  # all-caps preserved
            ("catalyses AI progress", "catalyzes AI progress"),
            ("whilst travelling", "while traveling"),
            ("roughly 1/3 of teams", "roughly one-third of teams"),
            ("split 3/4 to 1/4", "split three-quarters to one-quarter"),
        ],
    )
    def test_transforms(self, src: str, want: str) -> None:
        assert normalize_output(src) == want

    def test_fraction_in_date_untouched(self) -> None:
        assert normalize_output("on 1/3/2026 they met") == "on 1/3/2026 they met"


class TestScaleAndCurrency:
    @pytest.mark.parametrize(
        ("src", "want"),
        [
            ("raised $9.36bn", "raised $9.36 billion"),
            ("spent $9.36 bn", "spent $9.36 billion"),
            ("¥380bn", "¥380 billion"),
            ("a 5trn market", "a 5 trillion market"),
            ("24bn overall", "24 billion overall"),  # preceding digits preserved
            ("grew to 21 k users", "grew to 21,000 users"),
            ("cost $430k total", "cost $430,000 total"),
            ("$1.5k each", "$1,500 each"),
            ("handles 430 000 requests", "handles 430,000 requests"),
            ("over 1 234 567 rows", "over 1,234,567 rows"),
            ("courting $100 M", "courting $100 million"),
            ("a $1.2 B valuation", "a $1.2 billion valuation"),
            ("hit $10.3B", "hit $10.3 billion"),
            ("€5 T market", "€5 trillion market"),
            ("a $430 K seed", "a $430,000 seed"),  # capital K expands, never "thousand"
        ],
    )
    def test_transforms(self, src: str, want: str) -> None:
        assert normalize_output(src) == want

    @pytest.mark.parametrize(
        "src",
        [
            "priced at $700 million",  # currency, no abbreviation
            "the 4K display",  # resolution, not 4,000
            "an 8K monitor",
            "5m of cable",  # bare m left alone (could be meters)
            "Plan B shipped",  # capital scale letters need a currency prefix
            "class B shares rose",
            "reached 100 M users",  # no currency, so not a money amount
        ],
    )
    def test_untouched(self, src: str) -> None:
        assert normalize_output(src) == src


class TestNotation:
    @pytest.mark.parametrize(
        ("src", "want"),
        [
            ("a 3× speedup", "a 3x speedup"),
            ("100× faster", "100x faster"),
            ("a 2×3 grid", "a 2x3 grid"),
            ("set $N=200$ inputs", "set N=200 inputs"),
            ("with $k=5$ folds", "with k=5 folds"),
            ("GPT 5.6 and GPT4", "GPT-5.6 and GPT-4"),
            ("GPT-4o unchanged", "GPT-4o unchanged"),
        ],
    )
    def test_transforms(self, src: str, want: str) -> None:
        assert normalize_output(src) == want

    def test_bare_times_between_words_untouched(self) -> None:
        assert normalize_output("A × B dimensions") == "A × B dimensions"

    def test_gptq_not_hyphenated(self) -> None:
        assert normalize_output("the GPTQ method") == "the GPTQ method"


class TestEntityCasing:
    @pytest.mark.parametrize(
        ("src", "want"),
        [
            ("Open AI shipped", "OpenAI shipped"),
            ("OpenAi released", "OpenAI released"),
            ("built on Pytorch", "built on PyTorch"),
            ("via Github", "via GitHub"),
        ],
    )
    def test_transforms(self, src: str, want: str) -> None:
        assert normalize_output(src) == want


class TestSpanSafety:
    def test_link_anchor_and_target_survive(self) -> None:
        src = "See [How Centre Catalyses AI](https://x.io/a?b=1) for 37 %."
        got = normalize_output(src)
        assert "[How Centre Catalyses AI](https://x.io/a?b=1)" in got
        assert got.endswith("37%.")

    def test_inline_code_survives(self) -> None:
        src = "the `centre_id=430 000` field is centre-aligned at 37 %"
        got = normalize_output(src)
        assert "`centre_id=430 000`" in got
        assert "center-aligned" in got
        assert "37%" in got

    def test_fenced_code_survives(self) -> None:
        src = "```\nx = 430 000  # centre\n```\nprose centre at 37 %"
        got = normalize_output(src)
        assert "x = 430 000  # centre" in got
        assert "prose center at 37%" in got

    def test_bare_url_survives(self) -> None:
        src = "at https://x.io/centre?n=430 000 today"
        got = normalize_output(src)
        assert "https://x.io/centre?n=430" in got


class TestIdempotenceAndEdges:
    def test_idempotent(self) -> None:
        src = "raised $9.36bn, up 37 %, a 3× gain at the centre"
        once = normalize_output(src)
        assert normalize_output(once) == once

    def test_empty(self) -> None:
        assert normalize_output("") == ""


class TestMaskProtectedSpans:
    def test_preserves_length_and_newlines(self) -> None:
        src = "a\n[x](http://y) `z`\nb"
        masked = mask_protected_spans(src)
        assert len(masked) == len(src)
        assert masked.count("\n") == src.count("\n")

    def test_blanks_protected_content(self) -> None:
        masked = mask_protected_spans("see [anchor](http://u) now")
        assert "anchor" not in masked
        assert "http" not in masked
        assert masked.startswith("see ")
        assert masked.endswith(" now")
