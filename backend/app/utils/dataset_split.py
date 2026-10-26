"""Stratified train/validation/test split pipeline for the ISOT dataset.

Implements (70% train, 15% validation,
15% test, stratified on the label) as index-based splits
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
DEFAULT_RANDOM_STATE = 42

RATIO_TOLERANCE = 1e-9
DEFAULT_SPLIT_PATH = Path("backend/ml/data/processed/split_indices.json")


@dataclass
class DatasetSplit:
    """Index-based three-way split into train, validation, and test sets."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    random_state: int = DEFAULT_RANDOM_STATE
    train_ratio: float = TRAIN_RATIO
    val_ratio: float = VAL_RATIO
    test_ratio: float = TEST_RATIO

    def __len__(self) -> int:
        return len(self.train_idx) + len(self.val_idx) + len(self.test_idx)

    def subset(self, items: Sequence) -> Tuple[Sequence, Sequence, Sequence]:
        """Slice any sequence (titles, texts, labels, ...) into the three splits."""
        train = [items[i] for i in self.train_idx]
        val = [items[i] for i in self.val_idx]
        test = [items[i] for i in self.test_idx]
        return train, val, test

    @property
    def counts(self) -> Dict[str, int]:
        return {
            "train": int(len(self.train_idx)),
            "val": int(len(self.val_idx)),
            "test": int(len(self.test_idx)),
        }

    def label_distribution(self, labels: Sequence[int]) -> Dict[str, Dict[str, int]]:
        """Counts of each class label per split."""
        labels_arr = np.asarray(labels)
        distribution: Dict[str, Dict[str, int]] = {}
        for name, idx in (
            ("train", self.train_idx),
            ("val", self.val_idx),
            ("test", self.test_idx),
        ):
            split_labels = labels_arr[idx]
            values, counts = np.unique(split_labels, return_counts=True)
            distribution[name] = {str(int(v)): int(c) for v, c in zip(values, counts)}
        return distribution


def stratified_split(
    labels: Sequence[int],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DatasetSplit:
    """Split indices into stratified train / validation / test partitions. """
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("split ratios must all be positive")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > RATIO_TOLERANCE:
        raise ValueError("split ratios must sum to 1.0")

    labels_arr = np.asarray(labels)
    if len(labels_arr) == 0:
        raise ValueError("Cannot split an empty label array")
    if len(np.unique(labels_arr)) < 2:
        raise ValueError("Stratified split requires at least two classes")

    indices = np.arange(len(labels_arr))
    rest_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=random_state,
        stratify=labels_arr,
    )
    rest_labels = labels_arr[rest_idx]
    relative_val = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        rest_idx,
        test_size=relative_val,
        random_state=random_state,
        stratify=rest_labels,
    )

    return DatasetSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        random_state=random_state,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


def save_splits(
    split: DatasetSplit,
    path: Path | str = DEFAULT_SPLIT_PATH,
    labels: Sequence[int] | None = None,
) -> Path:
    """Persist the split indices and split metadata to a JSON manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, object] = {
        "random_state": split.random_state,
        "ratios": {
            "train": split.train_ratio,
            "val": split.val_ratio,
            "test": split.test_ratio,
        },
        "counts": split.counts,
        "indices": {
            "train": [int(i) for i in split.train_idx],
            "val": [int(i) for i in split.val_idx],
            "test": [int(i) for i in split.test_idx],
        },
    }
    if labels is not None:
        manifest["label_distribution"] = split.label_distribution(labels)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    return path


def load_splits(path: Path | str = DEFAULT_SPLIT_PATH) -> DatasetSplit:
    """Load a previously persisted split manifest into a DatasetSplit."""
    with open(Path(path), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    ratios = manifest["ratios"]
    return DatasetSplit(
        train_idx=np.asarray(manifest["indices"]["train"], dtype=np.int64),
        val_idx=np.asarray(manifest["indices"]["val"], dtype=np.int64),
        test_idx=np.asarray(manifest["indices"]["test"], dtype=np.int64),
        random_state=manifest["random_state"],
        train_ratio=ratios["train"],
        val_ratio=ratios["val"],
        test_ratio=ratios["test"],
    )
