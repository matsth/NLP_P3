"""Feature representation factories for traditional classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sklearn.feature_extraction.text import TfidfVectorizer


RepresentationName = Literal["word_tfidf", "char_ngrams"]


@dataclass
class TextRepresentation:
    """Thin wrapper that enforces train-only fitting for vectorizers."""

    representation_type: str
    max_features: int | None = 100_000

    def __post_init__(self) -> None:
        self.vectorizer = self._build_vectorizer(self.representation_type)

    def fit_transform(self, texts: list[str]):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: list[str]):
        return self.vectorizer.transform(texts)

    def _build_vectorizer(self, representation_type: str) -> TfidfVectorizer:
        normalized = normalize_representation_name(representation_type)
        if normalized == "word_tfidf":
            return TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_features=self.max_features,
                sublinear_tf=True,
            )
        if normalized == "char_ngrams":
            return TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                max_features=self.max_features,
                sublinear_tf=True,
            )
        raise ValueError(f"Unsupported representation_type: {representation_type}")


def normalize_representation_name(representation_type: str) -> str:
    cleaned = representation_type.lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "word": "word_tfidf",
        "tfidf": "word_tfidf",
        "word_tfidf": "word_tfidf",
        "word_level_tfidf": "word_tfidf",
        "char": "char_ngrams",
        "char_ngram": "char_ngrams",
        "char_ngrams": "char_ngrams",
        "character_ngrams": "char_ngrams",
    }
    if cleaned not in aliases:
        raise ValueError("representation_type must be `word_tfidf` or `char_ngrams`.")
    return aliases[cleaned]
