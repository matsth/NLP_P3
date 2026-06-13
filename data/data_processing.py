"""Core data cleaning engine for the Google GoEmotions dataset."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Callable

import polars as pl
from tqdm.auto import tqdm


DATASET_URI = "hf://datasets/mrm8488/goemotions/goemotions.csv"
RANDOM_SEED = 42

EMOTION_COLUMNS: list[str] = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

NEGATION_AND_CONTRAST_WORDS = {
    "not",
    "no",
    "nor",
    "never",
    "neither",
    "but",
    "against",
}


@dataclass(frozen=True)
class DatasetSplits:
    """Container for train, test, and pristine validation Polars frames."""

    train: pl.DataFrame
    test: pl.DataFrame
    validation: pl.DataFrame


class GoEmotionsProcessor:
    """Download, split, and clean GoEmotions with strict label preservation."""

    dataset_uri: str = DATASET_URI
    label_columns: list[str] = EMOTION_COLUMNS

    def download_raw(self) -> pl.DataFrame:
        """Load GoEmotions using the HuggingFace datasets library and convert to Polars."""
        with tqdm(total=1, desc="Downloading GoEmotions", unit="file") as progress:
            # Securely fetch from HuggingFace using the official datasets loader backend
            from datasets import load_dataset
            
            # This streams/downloads the dataset as an Arrow-backed structure
            hf_dataset = load_dataset("mrm8488/goemotions", split="train")
            
            # Instantly cast the underlying Arrow table directly into a Polars DataFrame
            frame = pl.from_arrow(hf_dataset.data.table)
            progress.update(1)

        required_columns = ["text", *self.label_columns]
        missing = sorted(set(required_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        return frame.select(required_columns).with_columns(
            pl.col("text").cast(pl.Utf8).fill_null("")
        )

    def split(self, frame: pl.DataFrame) -> DatasetSplits:
        """Create an 80/10/10 split after metadata filtering and before cleaning."""
        if frame.is_empty():
            raise ValueError("Cannot split an empty dataset.")

        shuffled = frame.sample(fraction=1.0, shuffle=True, seed=RANDOM_SEED)
        row_count = shuffled.height
        train_end = int(row_count * 0.8)
        test_end = train_end + int(row_count * 0.1)

        return DatasetSplits(
            train=shuffled.slice(0, train_end),
            test=shuffled.slice(train_end, test_end - train_end),
            validation=shuffled.slice(test_end),
        )

    def build_variant(self, splits: DatasetSplits, variant: str) -> DatasetSplits:
        """Clean train/test only; validation remains pristine raw text."""
        cleaner = self._cleaner_for_variant(variant)
        normalized_variant = normalize_variant_name(variant)
        return DatasetSplits(
            train=self._map_text(splits.train, cleaner, f"train {normalized_variant}"),
            test=self._map_text(splits.test, cleaner, f"test {normalized_variant}"),
            validation=splits.validation,
        )

    def build_all_variants(self, splits: DatasetSplits) -> dict[str, DatasetSplits]:
        """Build all requested text processing variants."""
        return {
            "raw": splits,
            "variant_a": self.build_variant(splits, "variant_a"),
            "variant_b": self.build_variant(splits, "variant_b"),
            "variant_c": self.build_variant(splits, "variant_c"),
        }

    def _map_text(
        self, frame: pl.DataFrame, cleaner: Callable[[str], str], description: str
    ) -> pl.DataFrame:
        texts = frame.get_column("text").to_list()
        cleaned = [cleaner(text) for text in tqdm(texts, desc=f"Cleaning {description}")]
        return frame.with_columns(pl.Series("text", cleaned))

    def _cleaner_for_variant(self, variant: str) -> Callable[[str], str]:
        normalized = normalize_variant_name(variant)
        if normalized == "variant_a":
            return basic_clean
        if normalized == "variant_b":
            return standard_nltk_clean
        if normalized == "variant_c":
            return nltk_clean_keep_negation
        raise ValueError(f"Unsupported processing variant: {variant}")


def normalize_variant_name(variant: str) -> str:
    """Normalize short and descriptive variant names to cache-safe identifiers."""
    cleaned = variant.lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "raw": "raw",
        "pristine": "raw",
        "a": "variant_a",
        "basic": "variant_a",
        "basic_clean": "variant_a",
        "variant_a": "variant_a",
        "b": "variant_b",
        "standard": "variant_b",
        "standard_nltk": "variant_b",
        "standard_nltk_clean": "variant_b",
        "variant_b": "variant_b",
        "c": "variant_c",
        "nltk_negation": "variant_c",
        "negation": "variant_c",
        "nltk_clean_with_negation_exclusion": "variant_c",
        "variant_c": "variant_c",
    }
    if cleaned not in aliases:
        # Update your error message to reflect the new option
        raise ValueError(
            "processing_variant must be one of: raw, a, b, c, basic_clean, "
            "standard_nltk_clean, nltk_clean_with_negation_exclusion"
        )
    return aliases[cleaned]


def basic_clean(text: str) -> str:
    """Lowercase text and normalize whitespace."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def standard_nltk_clean(text: str) -> str:
    """Lowercase, remove URLs/punctuation, and strip English stopwords."""
    return _nltk_clean(text, keep_words=set())


def nltk_clean_keep_negation(text: str) -> str:
    """NLTK cleaning that preserves negation and contrast words."""
    return _nltk_clean(text, keep_words=NEGATION_AND_CONTRAST_WORDS)


def _nltk_clean(text: str, keep_words: set[str]) -> str:
    stop_words = _english_stopwords() - keep_words
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in text.split() if token not in stop_words]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def _english_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except LookupError as exc:
        raise RuntimeError(
            "NLTK stopwords are not installed. Run: python -m nltk.downloader stopwords"
        ) from exc