import pytest
from backend.app.services.preprocessor import (
    clean_text_regex,
    expand_contractions,
    remove_wire_artifacts,
    TextPreprocessor,
)


def test_remove_wire_artifacts():
    reuters_text = "WASHINGTON (Reuters) - The U.S. Congress voted on the bill today."
    cleaned = remove_wire_artifacts(reuters_text)
    assert not cleaned.startswith("WASHINGTON")
    assert "(Reuters)" not in cleaned
    assert "The U.S. Congress voted" in cleaned


def test_expand_contractions():
    raw = "They don't know that we couldn't go, but it's fine."
    expanded = expand_contractions(raw)
    assert "do not" in expanded
    assert "could not" in expanded
    assert "it is" in expanded


def test_clean_text_regex_html_and_urls():
    raw = '<p>Check out <a href="https://example.com/news">this link</a> or email info@test.com!</p>'
    cleaned = clean_text_regex(raw)
    assert "https" not in cleaned
    assert "example.com" not in cleaned
    assert "info@test.com" not in cleaned
    assert "<p>" not in cleaned
    assert "this link" in cleaned


def test_clean_text_regex_wire_and_media():
    raw = "SEATTLE (Reuters) - [VIDEO] Officials reported new findings today."
    cleaned = clean_text_regex(raw)
    assert "reuters" not in cleaned
    assert "video" not in cleaned
    assert "officials reported new findings today" in cleaned


def test_clean_text_regex_empty_or_none():
    assert clean_text_regex("") == ""
    assert clean_text_regex(None) == ""
    assert clean_text_regex("   ") == ""


def test_text_preprocessor_regex_mode():
    preprocessor = TextPreprocessor(use_spacy=False)
    res = preprocessor.clean("Don't worry, the White House announced new rules!")
    assert "do not" in res
    assert "white house announced new rules" in res


def test_text_preprocessor_spacy_mode():
    preprocessor = TextPreprocessor(use_spacy=True, remove_stopwords=True)
    res = preprocessor.clean("The presidents were organizing major conferences.")
    assert "president" in res  # lemmatized
    assert "organize" in res    # lemmatized
    assert "the" not in res.split()  # stopword removed
