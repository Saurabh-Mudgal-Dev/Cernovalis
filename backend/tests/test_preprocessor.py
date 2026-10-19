import pytest
from backend.app.services.preprocessor import (
    clean_text_regex,
    expand_contractions,
    remove_wire_artifacts,
    TextPreprocessor,
    tokenize_text,
    filter_stopwords,
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
    preprocessor = TextPreprocessor(use_spacy=False, remove_stopwords=False)
    res = preprocessor.clean("Don't worry, the White House announced new rules!")
    assert "do not" in res
    assert "white house announced new rules" in res


@pytest.fixture(scope="module")
def spacy_preprocessor():
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("en_core_web_sm"):
        pytest.skip("Install en_core_web_sm to run spaCy integration tests")
    return TextPreprocessor(use_spacy=True, remove_stopwords=True)


def test_text_preprocessor_spacy_mode(spacy_preprocessor):
    preprocessor = spacy_preprocessor
    res = preprocessor.clean("The presidents were organizing major conferences.")
    assert "president" in res  # lemmatized
    assert "organize" in res    # lemmatized
    assert "the" not in res.split()  # stopword removed


@pytest.mark.parametrize("text", [None, "", "   ", 42, float("nan"), "123 !!!"])
def test_tokenize_empty_input(text):
    assert tokenize_text(text) == []
    assert TextPreprocessor(remove_stopwords=False).preprocess_tokens(text) == []


def test_tokenize_text():
    raw = "<p>The cafés don’t close!</p> Visit https://example.com or info@test.com."
    assert tokenize_text(raw) == [
        "the", "cafes", "do", "not", "close", "visit", "or"
    ]
    assert tokenize_text("Hello WORLD!", lowercase=False) == ["Hello", "WORLD"]


def test_email_removed_in_full():
    assert clean_text_regex("Contact info@test.com for news") == "contact for news"


def test_filter_stopwords_defaults():
    assert filter_stopwords(["The", "news", "is", "not", "true"]) == ["news", "true"]
    assert filter_stopwords(["The", "news", "is", "not", "true"], keep_words=["NOT"]) == [
        "news", "not", "true"
    ]


def test_filter_stopwords_custom_list():
    tokens = (token for token in ["News", "the", "NEWS", "story"])
    assert filter_stopwords(tokens, stopwords=["news"]) == ["the", "story"]
    assert filter_stopwords(["The", "news"], stopwords=[]) == ["The", "news"]


def test_regex_stopword_configuration():
    preprocessor = TextPreprocessor(keep_words=["not"])
    assert preprocessor.clean("The claims are not true") == "claims not true"
    custom = TextPreprocessor(custom_stopwords=["claims"], lowercase=False)
    assert custom.clean("The CLAIMS are true") == "The are true"
    assert TextPreprocessor(custom_stopwords=[]).clean("The claims") == "the claims"


def test_token_length_and_surface_tokens():
    preprocessor = TextPreprocessor(min_token_length=2, custom_stopwords=["the"])
    assert preprocessor.tokenize("A cat and the dogs") == ["a", "cat", "and", "the", "dogs"]
    assert preprocessor.preprocess_tokens("A cat and the dogs") == ["cat", "and", "dogs"]


@pytest.mark.parametrize("use_spacy", [False, True])
def test_batch_matches_single(use_spacy, request):
    preprocessor = (
        request.getfixturevalue("spacy_preprocessor")
        if use_spacy else TextPreprocessor()
    )
    texts = ["The presidents were organizing major conferences.", None, "", "123!", "Cats ran."]
    assert preprocessor.pipe(iter(texts), batch_size=2) == [
        preprocessor.clean(text) for text in texts
    ]
    assert preprocessor.pipe([]) == []


def test_spacy_options(spacy_preprocessor, monkeypatch):
    import spacy

    # Reuse the loaded model without changing the shared fixture's configuration.
    monkeypatch.setattr(spacy, "load", lambda *args, **kwargs: spacy_preprocessor._nlp)
    preprocessor = TextPreprocessor(
        use_spacy=True, lemmatize=False, remove_stopwords=False, lowercase=False
    )
    assert preprocessor.clean("The presidents were organizing conferences.") == (
        "The presidents were organizing conferences"
    )
    preprocessor.lemmatize = True
    assert preprocessor.clean("The presidents were organizing conferences.") == (
        "the president be organize conference"
    )


def test_invalid_configuration():
    with pytest.raises(ValueError, match="min_token_length"):
        TextPreprocessor(min_token_length=0)
    with pytest.raises(ValueError, match="batch_size"):
        TextPreprocessor(remove_stopwords=False).pipe([], batch_size=0)


def test_missing_model_has_actionable_error(monkeypatch):
    import spacy

    def missing_model(*args, **kwargs):
        raise OSError("missing model")

    monkeypatch.setattr(spacy, "load", missing_model)
    with pytest.raises(RuntimeError, match="python -m spacy download en_core_web_sm"):
        TextPreprocessor(use_spacy=True)
