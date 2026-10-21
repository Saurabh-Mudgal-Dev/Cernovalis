"""Stylometric and text feature extraction utilities."""

from __future__ import annotations

import re
import string
from typing import Dict, List

# A "word" is a run of alphanumeric characters and apostrophes.
_RE_WORD = re.compile(r"[A-Za-z0-9']+")
# Only alphabetic tokens count as real words for lexical metrics.
_RE_ALPHA_WORD = re.compile(r"[A-Za-z']+")
# Sentence boundaries are runs of terminal punctuation.
_RE_SENTENCE_TERMINAL = re.compile(r"[.!?]+")
# Any ASCII punctuation character.
_RE_PUNCTUATION = re.compile(f"[{re.escape(string.punctuation)}]")
_RE_EXCLAMATION = re.compile(r"!")
_RE_QUESTION = re.compile(r"\?")
_RE_QUOTATION = re.compile(r'["\']')
_RE_UPPERCASE_WORD = re.compile(r"\b[A-Z][A-Z0-9']{1,}\b")
_RE_TITLECASE_WORD = re.compile(r"\b[A-Z][a-z0-9']+\b")
# Vowel groups for heuristic syllable counting.
_RE_VOWEL_GROUPS = re.compile(r"[aeiouy]+", re.IGNORECASE)

# Lazily loaded VADER sentiment analyzer (NLTK).
_vader: object = None


def _get_vader():
    """Return a shared VADER SentimentIntensityAnalyzer, loading NLTK data on demand."""
    global _vader
    if _vader is None:

        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        _vader = SentimentIntensityAnalyzer()
    return _vader


def _count_syllables(word: str) -> int:
    """Heuristic syllable count for a single word (no external dictionary)."""
    word = word.lower()
    groups = _RE_VOWEL_GROUPS.findall(word)
    count = len(groups)
    # Silent trailing 'e' (unless part of a vowel team or the only syllable).
    if (
        word.endswith("e")
        and not word.endswith(("le", "ee", "ye", "ie"))
        and count > 1
    ):
        count -= 1
    return max(1, count)


def _syllable_count(text: str) -> int:
    return sum(_count_syllables(w) for w in _RE_ALPHA_WORD.findall(text or ""))


def _words(text: str) -> List[str]:
    return _RE_WORD.findall(text or "")


def _alpha_words(text: str) -> List[str]:
    return [w for w in _RE_ALPHA_WORD.findall(text or "") if any(c.isalpha() for c in w)]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def extract_text_statistics(text: str) -> Dict[str, float]:
    """Basic document statistics.
    Returns:
        char_count, alpha_char_count,
        word_count, sentence_count, avg_word_length
    """
    text = text if isinstance(text, str) else ""
    words = _words(text)
    alpha_words = _alpha_words(text)
    char_count = len(text)
    alpha_char_count = sum(1 for c in text if c.isalpha())
    sentence_count = len(_RE_SENTENCE_TERMINAL.findall(text))
    avg_word_length = (
        sum(len(w) for w in words) / len(words) if words else 0.0
    )
    return {
        "char_count": float(char_count),
        "alpha_char_count": float(alpha_char_count),
        "word_count": float(len(words)),
        "sentence_count": float(sentence_count),
        "avg_word_length": float(avg_word_length),
    }


def extract_capitalization_features(text: str) -> Dict[str, float]:
    """Capitalization / shouting metrics.
    Returns:
        uppercase_char_ratio, uppercase_word_ratio,
        title_case_word_ratio, all_caps_word_count
    """
    text = text if isinstance(text, str) else ""
    words = _words(text)
    alpha_chars = [c for c in text if c.isalpha()]
    uppercase_char_count = sum(1 for c in alpha_chars if c.isupper())
    uppercase_char_ratio = _safe_ratio(uppercase_char_count, len(alpha_chars))

    all_caps_words = _RE_UPPERCASE_WORD.findall(text)
    title_case_words = _RE_TITLECASE_WORD.findall(text)
    word_count = len(words)
    uppercase_word_ratio = _safe_ratio(len(all_caps_words), word_count)
    title_case_word_ratio = _safe_ratio(len(title_case_words), word_count)

    return {
        "uppercase_char_ratio": float(uppercase_char_ratio),
        "uppercase_word_ratio": float(uppercase_word_ratio),
        "title_case_word_ratio": float(title_case_word_ratio),
        "all_caps_word_count": float(len(all_caps_words)),
    }


def extract_punctuation_features(text: str) -> Dict[str, float]:
    """Punctuation density metrics.
    Returns:
    punctuation_density, exclamation_density,
    question_density, quotation_density, sentence_end_density
    """
    text = text if isinstance(text, str) else ""
    char_count = len(text)
    punctuation_count = len(_RE_PUNCTUATION.findall(text))
    exclamation_count = len(_RE_EXCLAMATION.findall(text))
    question_count = len(_RE_QUESTION.findall(text))
    quotation_count = len(_RE_QUOTATION.findall(text))
    sentence_end_count = len(_RE_SENTENCE_TERMINAL.findall(text))

    return {
        "punctuation_density": _safe_ratio(punctuation_count, char_count),
        "exclamation_density": _safe_ratio(exclamation_count, char_count),
        "question_density": _safe_ratio(question_count, char_count),
        "quotation_density": _safe_ratio(quotation_count, char_count),
        "sentence_end_density": _safe_ratio(sentence_end_count, char_count),
    }


def extract_sentiment_features(text: str) -> Dict[str, float]:
    """VADER sentiment polarity.
    Returns:
        sentiment_polarity, sentiment_pos,
        sentiment_neu, sentiment_neg
    """
    text = text if isinstance(text, str) else ""
    scores = _get_vader().polarity_scores(text)
    return {
        "sentiment_polarity": float(scores["compound"]),
        "sentiment_pos": float(scores["pos"]),
        "sentiment_neu": float(scores["neu"]),
        "sentiment_neg": float(scores["neg"]),
    }


def extract_subjectivity_features(text: str) -> Dict[str, float]:
    """Subjectivity derived from VADER sentiment.
    Returns:
        subjectivity
    """
    text = text if isinstance(text, str) else ""
    scores = _get_vader().polarity_scores(text)
    return {"subjectivity": float(scores["pos"] + scores["neg"])}


def extract_readability_features(text: str) -> Dict[str, float]:
    """Flesch-Kincaid readability metrics.
    Returns:
        flesch_kincaid_grade, flesch_reading_ease,
        syllables_per_word
    """
    text = text if isinstance(text, str) else ""
    words = _words(text)
    word_count = len(words)
    sentence_count = len(_RE_SENTENCE_TERMINAL.findall(text))
    syllables = _syllable_count(text)

    if word_count == 0 or sentence_count == 0:
        return {
            "flesch_kincaid_grade": 0.0,
            "flesch_reading_ease": 0.0,
            "syllables_per_word": 0.0,
        }

    words_per_sentence = word_count / sentence_count
    syllables_per_word = syllables / word_count
    fk_grade = (
        0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    )
    reading_ease = (
        206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    )
    return {
        "flesch_kincaid_grade": float(fk_grade),
        "flesch_reading_ease": float(reading_ease),
        "syllables_per_word": float(syllables_per_word),
    }


def extract_lexical_diversity_features(text: str) -> Dict[str, float]:
    """Lexical diversity (vocabulary richness) metrics.
    Returns:
        type_token_ratio
    """
    text = text if isinstance(text, str) else ""
    words = [w.lower() for w in _words(text)]
    return {
        "type_token_ratio": _safe_ratio(len(set(words)), len(words)),
    }


def extract_all_stylometric_features(text: str) -> Dict[str, float]:
    """Merge all feature groups into one dict."""
    text = text if isinstance(text, str) else ""
    features: Dict[str, float] = {}
    features.update(extract_text_statistics(text))
    features.update(extract_capitalization_features(text))
    features.update(extract_punctuation_features(text))
    features.update(extract_sentiment_features(text))
    features.update(extract_subjectivity_features(text))
    features.update(extract_readability_features(text))
    features.update(extract_lexical_diversity_features(text))
    return features


def stylometric_feature_names() -> List[str]:
    """Ordered list of all feature keys returned by extract_all_stylometric_features."""
    return list(
        extract_all_stylometric_features("The quick brown fox jumps over the lazy dog.").keys()
    )