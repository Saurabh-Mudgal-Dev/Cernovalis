"""Tests for the TextPreprocessor class."""

# pyrefly: ignore[missing-import]
import pytest

from backend.app.services.preprocessor import TextPreprocessor


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
    assert "organize" in res  # lemmatized
    assert "the" not in res.split()  # stopword removed


@pytest.mark.parametrize("text", [None, "", "   ", 42, float("nan"), "123 !!!"])
def test_preprocess_tokens_empty_input(text):
    assert TextPreprocessor(remove_stopwords=False).preprocess_tokens(text) == []


def test_regex_stopword_configuration():
    preprocessor = TextPreprocessor(keep_words=["not"])
    assert preprocessor.clean("The claims are not true") == "claims not true"
    custom = TextPreprocessor(custom_stopwords=["claims"], lowercase=False)
    assert custom.clean("The CLAIMS are true") == "The are true"
    assert TextPreprocessor(custom_stopwords=[]).clean("The claims") == "the claims"


def test_token_length_and_surface_tokens():
    preprocessor = TextPreprocessor(min_token_length=2, custom_stopwords=["the"])
    assert preprocessor.tokenize("A cat and the dogs") == [
        "a",
        "cat",
        "and",
        "the",
        "dogs",
    ]
    assert preprocessor.preprocess_tokens("A cat and the dogs") == ["cat", "and", "dogs"]


@pytest.mark.parametrize("use_spacy", [False, True])
def test_batch_matches_single(use_spacy, request):
    preprocessor = (
        request.getfixturevalue("spacy_preprocessor")
        if use_spacy
        else TextPreprocessor()
    )
    texts = [
        "The presidents were organizing major conferences.",
        None,
        "",
        "123!",
        "Cats ran.",
    ]
    assert preprocessor.pipe(iter(texts), batch_size=2) == [
        preprocessor.clean(text) for text in texts
    ]
    assert preprocessor.pipe([]) == []


def test_spacy_options(spacy_preprocessor, monkeypatch):
    
    # pyrefly: ignore[missing-import]
    import spacy

    # Reuse the loaded model without changing the shared fixture's configuration.
    monkeypatch.setattr(spacy, "load", lambda *args, **kwargs: spacy_preprocessor._nlp)
    preprocessor = TextPreprocessor(
        use_spacy=True,
        lemmatize=False,
        remove_stopwords=False,
        lowercase=False,
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

    # pyrefly: ignore[missing-import]
    import spacy

    def missing_model(*args, **kwargs):
        raise OSError("missing model")

    monkeypatch.setattr(spacy, "load", missing_model)
    with pytest.raises(RuntimeError, match="python -m spacy download en_core_web_sm"):
        TextPreprocessor(use_spacy=True)