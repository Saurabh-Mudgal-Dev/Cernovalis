# Stopword filtering utilities for text preprocessing.

from __future__ import annotations

from typing import Iterable, List, Optional


def filter_stopwords(
    tokens: Iterable[str],
    stopwords: Optional[Iterable[str]] = None,
    keep_words: Iterable[str] = (),
) -> List[str]:
    """Filter tokens case-insensitively, retaining their order and original case.

    By default uses spaCy's English stopwords (no trained model required).
    ``stopwords`` replaces that list, while ``keep_words`` overrides removals,
    for example to preserve negations such as 'not', 'no', and 'never'.
    """
    if stopwords is None:
        
        from spacy.lang.en.stop_words import STOP_WORDS

        stopwords = STOP_WORDS
    excluded = {word.lower() for word in stopwords}
    excluded.difference_update(word.lower() for word in keep_words)
    return [token for token in tokens if token.lower() not in excluded]