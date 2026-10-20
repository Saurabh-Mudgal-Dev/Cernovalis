"""Configurable text preprocessor orchestrating cleaning, tokenization,
stopword removal, and optional spaCy lemmatization.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Set

from backend.app.services.preprocessor.tokenization import tokenize_text


class TextPreprocessor:
    """Configurable text preprocessor with optional spaCy lemmatization and stopword removal."""

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

                # pyrefly: ignore[missing-import]
                from spacy.lang.en.stop_words import STOP_WORDS

                custom_stopwords = STOP_WORDS
            self.stopwords = {word.lower() for word in custom_stopwords}
            self.stopwords.difference_update(word.lower() for word in keep_words)
        self._nlp = None

        if self.use_spacy:
            self._load_spacy(spacy_model)

    def _load_spacy(self, model_name: str) -> None:
        
        # pyrefly: ignore[missing-import]
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
        """Return model-ready tokens; lemmatization requires ``use_spacy=True``."""
        tokens = self.tokenize(text)
        if self._nlp is not None and tokens:
            return self._process_doc(self._nlp(" ".join(tokens)))
        return [
            token
            for token in tokens
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