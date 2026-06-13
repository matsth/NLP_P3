"""Caching data loader for GoEmotions variants."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

from data.data_processing import DatasetSplits, GoEmotionsProcessor, normalize_variant_name


class EmotionDataLoader:
    """Serve cached train/test/validation splits for a processing variant."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.processor = GoEmotionsProcessor()

    def load(self, processing_variant: str) -> DatasetSplits:
        """Load a variant from CSV cache or build all variants if cache is missing."""
        variant = normalize_variant_name(processing_variant)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # If they ask for raw, ensure raw cache exists and return it
        if variant == "raw":
            if not self._variant_cache_exists("raw"):
                self._build_raw_cache()
            return DatasetSplits(
                train=self._read_split("train", "raw"),
                test=self._read_split("test", "raw"),
                validation=self._read_split("validation", "raw"),
            )

        # Otherwise, handle the standard cleaning variants
        if not self._variant_cache_exists(variant):
            self._build_cache()

        return DatasetSplits(
            train=self._read_split("train", variant),
            test=self._read_split("test", variant),
            validation=self._read_split("validation", variant),
        )

    def load_raw(self) -> DatasetSplits:
        """Load raw unprocessed train/test/validation splits for language models."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if not self._variant_cache_exists("raw"):
            self._build_raw_cache()

        return DatasetSplits(
            train=self._read_split("train", "raw"),
            test=self._read_split("test", "raw"),
            validation=self._read_split("validation", "raw"),
        )

    def _build_cache(self) -> None:
        if not self._variant_cache_exists("raw"):
            self._build_raw_cache()

        splits = DatasetSplits(
            train=self._read_split("train", "raw"),
            test=self._read_split("test", "raw"),
            validation=self._read_split("validation", "raw"),
        )
        variants = self.processor.build_all_variants(splits)

        for variant, variant_splits in tqdm(
            variants.items(), desc="Writing variant CSV caches"
        ):
            self._write_split(variant_splits.train, "train", variant)
            self._write_split(variant_splits.test, "test", variant)
            self._write_split(variant_splits.validation, "validation", variant)

    def _build_raw_cache(self) -> None:
        raw = self.processor.download_raw()
        splits = self.processor.split(raw)

        for split_name, frame in tqdm(
            {
                "train": splits.train,
                "test": splits.test,
                "validation": splits.validation,
            }.items(),
            desc="Writing raw CSV caches",
        ):
            self._write_split(frame, split_name, "raw")

    def _variant_cache_exists(self, variant: str) -> bool:
        paths = [self._path(split, variant) for split in ("train", "test", "validation")]
        return all(path.exists() for path in paths)

    def _path(self, split: str, variant: str) -> Path:
        return self.data_dir / f"{split}_{variant}.csv"

    def _read_split(self, split: str, variant: str) -> pl.DataFrame:
        path = self._path(split, variant)
        with tqdm(total=1, desc=f"Loading {path.name}", unit="file") as progress:
            frame = pl.read_csv(path)
            progress.update(1)
        return frame

    def _write_split(self, frame: pl.DataFrame, split: str, variant: str) -> None:
        frame.write_csv(self._path(split, variant))
