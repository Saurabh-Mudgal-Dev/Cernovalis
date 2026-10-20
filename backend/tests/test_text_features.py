"""Tests for stylometric text feature extraction."""

from backend.app.utils.text_features import (
    extract_all_stylometric_features,
    extract_capitalization_features,
    extract_lexical_diversity_features,
    extract_punctuation_features,
    extract_readability_features,
    extract_sentiment_features,
    extract_subjectivity_features,
    extract_text_statistics,
    stylometric_feature_names,
)


def test_extract_text_statistics_empty():
    assert extract_text_statistics("") == {
        "char_count": 0.0,
        "alpha_char_count": 0.0,
        "word_count": 0.0,
        "sentence_count": 0.0,
        "avg_word_length": 0.0,
    }
    assert extract_text_statistics(None)["word_count"] == 0.0


def test_extract_text_statistics_basic():
    text = "Hello world. This is a test!"
    stats = extract_text_statistics(text)
    assert stats["word_count"] == 6.0
    assert stats["sentence_count"] == 2.0
    assert stats["char_count"] == float(len(text))
    assert stats["avg_word_length"] > 0


def test_extract_capitalization_features_shouting():
    text = "THIS IS AN URGENT ALERT!!!"
    feats = extract_capitalization_features(text)
    assert feats["uppercase_char_ratio"] == 1.0
    assert feats["uppercase_word_ratio"] == 1.0
    assert feats["all_caps_word_count"] == 5.0


def test_extract_capitalization_features_normal():
    text = "Hello world"
    feats = extract_capitalization_features(text)
    assert feats["uppercase_char_ratio"] < 1.0
    assert feats["all_caps_word_count"] == 0.0


def test_extract_punctuation_features_exclamations():
    text = "Wow!!! Really??"
    feats = extract_punctuation_features(text)
    assert feats["exclamation_density"] > 0
    assert feats["question_density"] > 0
    assert feats["punctuation_density"] > 0


def test_extract_punctuation_features_none():
    feats = extract_punctuation_features("Hello world")
    assert feats["exclamation_density"] == 0.0
    assert feats["question_density"] == 0.0


def test_type_token_ratio_repeated():
    feats = extract_all_stylometric_features("cat cat cat cat")
    assert feats["type_token_ratio"] == 0.25


def test_invalid_input_returns_zeros():
    for bad in ["", None, 123]:
        feats = extract_all_stylometric_features(bad)
        assert all(v == 0.0 for v in feats.values())


def test_whitespace_only_has_no_words():
    feats = extract_all_stylometric_features("   ")
    assert feats["char_count"] == 3.0
    assert feats["word_count"] == 0.0
    assert feats["sentence_count"] == 0.0
    assert feats["avg_word_length"] == 0.0


def test_extract_sentiment_features_positive():
    feats = extract_sentiment_features("I love this wonderful amazing day!")
    assert feats["sentiment_polarity"] > 0
    assert feats["sentiment_pos"] > 0


def test_extract_sentiment_features_negative():
    feats = extract_sentiment_features("This is terrible and awful.")
    assert feats["sentiment_polarity"] < 0
    assert feats["sentiment_neg"] > 0


def test_extract_sentiment_features_neutral():
    feats = extract_sentiment_features("The meeting is at 3 PM.")
    assert abs(feats["sentiment_polarity"]) < 0.5


def test_extract_sentiment_features_empty():
    feats = extract_sentiment_features("")
    assert feats["sentiment_polarity"] == 0.0
    assert feats["sentiment_pos"] == 0.0
    assert feats["sentiment_neu"] == 0.0
    assert feats["sentiment_neg"] == 0.0


def test_extract_subjectivity_features():
    feats = extract_subjectivity_features("I love this amazing wonderful day!")
    assert 0.0 <= feats["subjectivity"] <= 1.0
    assert feats["subjectivity"] > 0


def test_extract_subjectivity_features_neutral():
    feats = extract_subjectivity_features("The meeting is at 3 PM.")
    assert feats["subjectivity"] >= 0.0


def test_extract_readability_features_basic():
    text = "The cat sat on the mat. The dog ran fast."
    feats = extract_readability_features(text)
    assert "flesch_kincaid_grade" in feats
    assert "flesch_reading_ease" in feats
    assert "syllables_per_word" in feats
    assert feats["syllables_per_word"] > 0


def test_extract_readability_features_empty():
    feats = extract_readability_features("")
    assert feats["flesch_kincaid_grade"] == 0.0
    assert feats["flesch_reading_ease"] == 0.0
    assert feats["syllables_per_word"] == 0.0


def test_extract_lexical_diversity_features():
    feats = extract_lexical_diversity_features("cat cat cat cat")
    assert feats["type_token_ratio"] == 0.25


def test_extract_lexical_diversity_features_unique():
    feats = extract_lexical_diversity_features("cat dog bird fish")
    assert feats["type_token_ratio"] == 1.0


def test_extract_all_stylometric_features_merged():
    text = "BREAKING: The President says it's true! Do you believe it?"
    feats = extract_all_stylometric_features(text)
    # Contains statistics
    assert "word_count" in feats
    assert "avg_word_length" in feats
    # Contains capitalization
    assert "uppercase_char_ratio" in feats
    assert "all_caps_word_count" in feats
    # Contains punctuation
    assert "exclamation_density" in feats
    assert "quotation_density" in feats
    # Contains sentiment
    assert "sentiment_polarity" in feats
    assert "sentiment_pos" in feats
    assert "sentiment_neu" in feats
    assert "sentiment_neg" in feats
    # Contains subjectivity
    assert "subjectivity" in feats
    # Contains readability
    assert "flesch_kincaid_grade" in feats
    assert "flesch_reading_ease" in feats
    assert "syllables_per_word" in feats
    # Contains lexical diversity
    assert "type_token_ratio" in feats
    assert 0.0 <= feats["type_token_ratio"] <= 1.0


def test_stylometric_feature_names_consistent():
    names = stylometric_feature_names()
    sample = extract_all_stylometric_features("Hello world.")
    assert set(names) == set(sample.keys())
    assert len(names) == len(sample)