"""Tests for regex-based cleaning utilities."""

from backend.app.services.preprocessor.cleaning import (
    clean_text_regex,
    expand_contractions,
    remove_wire_artifacts,
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


def test_email_removed_in_full():
    assert clean_text_regex("Contact info@test.com for news") == "contact for news"


def test_clean_text_regex_preserves_punctuation():
    cleaned = clean_text_regex("Hello, world! Don't stop.", preserve_punctuation=True, expand_contr=False)
    assert "don't" in cleaned.lower()
    assert "hello" in cleaned
    assert "world" in cleaned
    assert "," in cleaned or "!" in cleaned