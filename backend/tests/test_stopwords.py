"""Tests for stopword filtering utilities."""

from backend.app.services.preprocessor.stopwords import filter_stopwords


def test_filter_stopwords_defaults():
    assert filter_stopwords(["The", "news", "is", "not", "true"]) == ["news", "true"]
    assert filter_stopwords(["The", "news", "is", "not", "true"], keep_words=["NOT"]) == [
        "news",
        "not",
        "true",
    ]


def test_filter_stopwords_custom_list():
    tokens = (token for token in ["News", "the", "NEWS", "story"])
    assert filter_stopwords(tokens, stopwords=["news"]) == ["the", "story"]
    assert filter_stopwords(["The", "news"], stopwords=[]) == ["The", "news"]


def test_filter_stopwords_preserves_order_and_case():
    tokens = ["The", "Quick", "Brown", "Fox", "jumps"]
    assert filter_stopwords(tokens) == ["Quick", "Brown", "Fox", "jumps"]