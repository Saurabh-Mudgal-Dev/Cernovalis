# Tokenization utilities for text preprocessing.


from __future__ import annotations

from typing import List

from backend.app.services.preprocessor.cleaning import clean_text_regex


def tokenize_text(
    text: str,
    strip_wire: bool = True,
    expand_contr: bool = True,
    lowercase: bool = True,
) -> List[str]:
    """Clean text and split it into alphabetic tokens without loading a model.
    Punctuation and digits are always removed for tokenization.
    """
    return clean_text_regex(
        text,
        strip_wire=strip_wire,
        expand_contr=expand_contr,
        lowercase=lowercase,
        preserve_punctuation=False,
    ).split()