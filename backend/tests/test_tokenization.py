"""Tests for tokenization utilities."""

import pytest

from backend.app.services.preprocessor.tokenization import tokenize_text


@pytest.mark.parametrize(
    "text", [None, "", "   ", 42, float("nan"), "123 !!!"]
)
def test_tokenize_empty_input(text):
    assert tokenize_text(text) == []


def test_tokenize_text():
    raw = "<p>The cafés don't close!</p> Visit https://example.com or info@test.com."
    assert tokenize_text(raw) == [
        "the",
        "cafes",
        "do",
        "not",
        "close",
        "visit",
        "or",
    ]
    assert tokenize_text("Hello WORLD!", lowercase=False) == ["Hello", "WORLD"]


def test_tokenize_strips_wire_by_default():
    tokens = tokenize_text("WASHINGTON (Reuters) - The bill passed today.")
    assert "reuters" not in tokens
    assert "washington" not in tokens
    assert "bill" in tokens
    assert "passed" in tokens