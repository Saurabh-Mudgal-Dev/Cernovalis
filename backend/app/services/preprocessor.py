"""
Text preprocessing and cleaning pipeline.
Provides regex-based normalization, source artifact/dateline removal, HTML/URL stripping,
contraction expansion, and lemmatization/stopword filtering.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Iterable, List, Optional, Set

# Pre-compiled regex patterns for performance
RE_WIRE_PREFIX = re.compile(
    r"^(\s*\(In[^\)]+\)\s*)?([A-Z\s,./\'-]+)?\s*\(Reuters\)\s*[-—:\s]*",
    re.IGNORECASE,
)
RE_HTML_TAGS = re.compile(r"<[^>]+>")
RE_URL = re.compile(
    r"(?:https?://|www\.)[^\s<>\"]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s<>\"]*)?",
    re.IGNORECASE,
)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
RE_IMAGE_OR_VIDEO_TAGS = re.compile(
    r"\[(?:video|image|photo|audio|graphic|tweet|watch|breaking)[^\]]*\]|\((?:video|image|photo|watch)[^\)]*\)",
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
    """
    Remove wire service attribution prefixes (e.g. 'WASHINGTON (Reuters) - ').
    In the raw dataset, 99.2% of real articles contain Reuters datelines, which
    causes model leakage if not stripped.
    """
    return RE_WIRE_PREFIX.sub("", text)


def clean_text_regex(
    text: str,
    strip_wire: bool = True,
    expand_contr: bool = True,
    lowercase: bool = True,
    preserve_punctuation: bool = False,
) -> str:
    """
    Apply pure-regex normalization and cleaning pipeline:
      1. Handle nulls / non-strings
      2. Decode HTML entities and unescape markup
      3. Strip wire service dateline artifacts if requested
      4. Remove media markers ([VIDEO], [WATCH], etc.)
      5. Remove URLs and email addresses
      6. Remove HTML tags
      7. Expand contractions
      8. Unicode NFKD normalization
      9. Clean special characters or non-alphabetics
      10. Normalize whitespace
    """
    if not isinstance(text, str) or not text:
        return ""

    # Unescape HTML entities
    text = html.unescape(text)

    # Strip wire datelines (e.g. Reuters headers)
    if strip_wire:
        text = remove_wire_artifacts(text)

    # Remove media markers common in fake headlines/body
    text = RE_IMAGE_OR_VIDEO_TAGS.sub(" ", text)

    # Remove URLs and emails
    text = RE_EMAIL.sub(" ", text)
    text = RE_URL.sub(" ", text)

    # Remove remaining HTML tags
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
        text, strip_wire=strip_wire, expand_contr=expand_contr,
        lowercase=lowercase, preserve_punctuation=False,
    ).split()


def filter_stopwords(
    tokens: Iterable[str],
    stopwords: Optional[Iterable[str]] = None,
    keep_words: Iterable[str] = (),
) -> List[str]:
    """Filter tokens case-insensitively, retaining their order and original case.

    By default use spaCy's English stopwords (no trained model required).
    ``stopwords`` replaces that list, while ``keep_words`` overrides removals,
    for example to preserve negations such as 'not', 'no', and 'never'.
    """
    if stopwords is None:
        from spacy.lang.en.stop_words import STOP_WORDS

        stopwords = STOP_WORDS
    excluded = {word.lower() for word in stopwords}
    excluded.difference_update(word.lower() for word in keep_words)
    return [token for token in tokens if token.lower() not in excluded]


class TextPreprocessor:
    """
    Configurable text preprocessor with optional spaCy lemmatization and stopword removal.
    Suitable for production inference (single texts) and batch dataset preparation.

    Regex mode (the default) does not lemmatize. Stopword filtering works in
    either mode. ``custom_stopwords`` replaces the English defaults and
    ``keep_words`` exempts words from filtering. No models are auto-downloaded.
    """

    def __init__(
        self,
        use_spacy: bool = False,
        spacy_model: str = "en_core_web_sm",
        remove_stopwords: bool = True,
        lowercase: bool = True,
        strip_wire: bool = True,
        lemmatize: bool = True,
        custom_stopwords: Optional[Iterable[str]] = None,
        keep_words: Iterable[str] = (),
        min_token_length: int = 1,
    ) -> None:
        if min_token_length < 1:
            raise ValueError("min_token_length must be at least 1")
        self.use_spacy = use_spacy
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.strip_wire = strip_wire
        self.lemmatize = lemmatize
        self.min_token_length = min_token_length
        self.stopwords: Set[str] = set()
        if remove_stopwords:
            if custom_stopwords is None:
                from spacy.lang.en.stop_words import STOP_WORDS

                custom_stopwords = STOP_WORDS
            self.stopwords = {word.lower() for word in custom_stopwords}
            self.stopwords.difference_update(word.lower() for word in keep_words)
        self._nlp = None

        if self.use_spacy:
            self._load_spacy(spacy_model)

    def _load_spacy(self, model_name: str) -> None:
        import spacy

        try:
            self._nlp = spacy.load(model_name, disable=["parser", "ner"])
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. Install it with "
                f"'python -m spacy download {model_name}', or use use_spacy=False "
                "for preprocessing without lemmatization."
            ) from exc

    def tokenize(self, text: str) -> List[str]:
        """Return normalized surface tokens before stopword removal/lemmatization."""
        return tokenize_text(
            text, strip_wire=self.strip_wire, lowercase=self.lowercase
        )

    def _process_doc(self, doc) -> List[str]:
        tokens = []
        for token in doc:
            if not token.is_alpha or len(token.text) < self.min_token_length:
                continue
            if self.remove_stopwords and token.text.lower() in self.stopwords:
                continue
            value = token.lemma_ if self.lemmatize else token.text
            if value:
                tokens.append(value.lower() if self.lowercase else value)
        return tokens

    def preprocess_tokens(self, text: str) -> List[str]:
        """Return model-ready tokens; lemmatization requires ``use_spacy=True``.

        Stopwords are checked against surface forms so inflections such as
        'was' do not survive filtering after being lemmatized to 'be'.
        """
        tokens = self.tokenize(text)
        if self._nlp is not None and tokens:
            return self._process_doc(self._nlp(" ".join(tokens)))
        return [
            token for token in tokens
            if len(token) >= self.min_token_length
            and (not self.remove_stopwords or token.lower() not in self.stopwords)
        ]

    def clean(self, text: str) -> str:
        """Return normalized, optionally lemmatized and stopword-filtered text."""
        return " ".join(self.preprocess_tokens(text))

    def pipe(self, texts: Iterable[str], batch_size: int = 256) -> List[str]:
        """Process an iterable of texts efficiently."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not self.use_spacy or self._nlp is None:
            return [self.clean(t) for t in texts]

        # Stage 1: Fast regex cleaning
        regex_cleaned = (" ".join(self.tokenize(text)) for text in texts)

        # Stage 2: Batch spaCy lemmatization
        results: List[str] = []
        for doc in self._nlp.pipe(regex_cleaned, batch_size=batch_size):
            results.append(" ".join(self._process_doc(doc)))
        return results
