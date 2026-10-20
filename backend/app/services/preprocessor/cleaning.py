# Regex-based text cleaning functions.


from __future__ import annotations

import html
import re
import unicodedata

# Pre-compiled regex patterns for performance
RE_WIRE_PREFIX = re.compile(
    r"^(\s*\(In[^)]+\)\s*)?([A-Z\s,./'-]+)?\s*\(Reuters\)\s*[-—:\s]*",
    re.IGNORECASE,
)
RE_HTML_TAGS = re.compile(r"<[^>]+>")
RE_URL = re.compile(
    r"(?:https?://|www\.)[^\s<>\"']+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s<>\"']*)?",
    re.IGNORECASE,
)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
RE_IMAGE_OR_VIDEO_TAGS = re.compile(
    r"\[(?:video|image|photo|audio|graphic|tweet|watch|breaking)[^\]]*]|\((?:video|image|photo|watch)[^)]*\)",
    re.IGNORECASE,
)
RE_SPECIAL_CHARS = re.compile(r"[^a-zA-Z0-9\s.,!?'\"-]")
RE_NON_ALPHA = re.compile(r"[^a-zA-Z\s]")
RE_MULTIPLE_SPACES = re.compile(r"\s+")

# Common English contraction mappings
CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

# Regex for word boundary contraction replacement
RE_CONTRACTIONS = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTION_MAP.keys()) + r")\b",
    re.IGNORECASE,
)


def expand_contractions(text: str) -> str:
    """Expand common English contractions (e.g. don't -> do not)."""

    def _replace(match: re.Match) -> str:
        matched_str = match.group(0).lower()
        expanded = CONTRACTION_MAP.get(matched_str, matched_str)
        if match.group(0)[0].isupper():
            return expanded.capitalize()
        return expanded

    # Standardize apostrophes
    normalized_text = text.replace("’", "'").replace("`", "'").replace("‘", "'")
    return RE_CONTRACTIONS.sub(_replace, normalized_text)


def remove_wire_artifacts(text: str) -> str:
    """Remove wire service attribution prefixes (e.g. 'WASHINGTON (Reuters) - ')."""
    return RE_WIRE_PREFIX.sub("", text)


def clean_text_regex(
    text: str,
    strip_wire: bool = True,
    expand_contr: bool = True,
    lowercase: bool = True,
    preserve_punctuation: bool = False,
) -> str:
    """Apply pure-regex normalization and cleaning pipeline."""
    
    if not isinstance(text, str) or not text:
        return ""

    # Unescape HTML entities
    text = html.unescape(text)

    # Strip wire datelines (e.g. Reuters headers)
    if strip_wire:
        text = remove_wire_artifacts(text)

    # Remove media markers common in headlines/body
    text = RE_IMAGE_OR_VIDEO_TAGS.sub(" ", text)

    # Remove URLs and emails
    text = RE_EMAIL.sub(" ", text)
    text = RE_URL.sub(" ", text)
    text = RE_HTML_TAGS.sub(" ", text)

    # Expand contractions
    if expand_contr:
        text = expand_contractions(text)

    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

    # Lowercase if requested
    if lowercase:
        text = text.lower()

    # Punctuation handling
    if preserve_punctuation:
        text = RE_SPECIAL_CHARS.sub(" ", text)
    else:
        text = RE_NON_ALPHA.sub(" ", text)

    # Collapse extra whitespace
    text = RE_MULTIPLE_SPACES.sub(" ", text).strip()
    return text