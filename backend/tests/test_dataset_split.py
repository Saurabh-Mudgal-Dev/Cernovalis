"""Tests for the stratified train/validation/test split pipeline."""

import json
from pathlib import Path

import pytest

from backend.app.utils.dataset_split import (
    DEFAULT_SPLIT_PATH,
    DatasetSplit,
    load_splits,
    save_splits,
    stratified_split,
)


def _make_labels(n_per_class: int) -> list[int]:
    return [1] * n_per_class + [0] * n_per_class


def test_three_way_split_exact_ratios():
    labels = _make_labels(5000)
    split = stratified_split(labels)
    assert len(split) == 10000
    assert split.counts == {"train": 7000, "val": 1500, "test": 1500}


def test_class_distribution_preserved():
    labels = _make_labels(5000)
    split = stratified_split(labels)
    distribution = split.label_distribution(labels)
    for counts in distribution.values():
        total = sum(counts.values())
        for value in counts.values():
            assert abs(value / total - 0.5) < 0.01


def test_splits_are_disjoint_and_complete():
    labels = _make_labels(500)
    split = stratified_split(labels)
    all_indices = sorted(
        list(split.train_idx) + list(split.val_idx) + list(split.test_idx)
    )
    assert all_indices == list(range(len(labels)))


def test_deterministic_given_seed():
    labels = _make_labels(500)
    first = stratified_split(labels, random_state=42)
    second = stratified_split(labels, random_state=42)
    assert list(first.train_idx) == list(second.train_idx)
    assert list(first.val_idx) == list(second.val_idx)
    assert list(first.test_idx) == list(second.test_idx)


def test_subset_slices_sequences():
    labels = _make_labels(500)
    split = stratified_split(labels)
    train_labels, val_labels, test_labels = split.subset(labels)
    assert len(train_labels) == split.counts["train"]
    assert len(val_labels) == split.counts["val"]
    assert len(test_labels) == split.counts["test"]
    assert len(split.train_idx) > 0 and len(split.val_idx) > 0 and len(split.test_idx) > 0
    for subset in (train_labels, val_labels, test_labels):
        assert set(subset) == {0, 1}


def test_small_dataset_produces_nonempty_splits():
    labels = _make_labels(10)
    split = stratified_split(labels)
    for name in ("train", "val", "test"):
        assert split.counts[name] > 0


def test_invalid_ratio_sum_raises():
    labels = _make_labels(100)
    with pytest.raises(ValueError, match="sum to 1.0"):
        stratified_split(labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.15)


def test_zero_ratio_raises():
    labels = _make_labels(100)
    with pytest.raises(ValueError, match="positive"):
        stratified_split(labels, val_ratio=0.0)


def test_empty_labels_raise():
    with pytest.raises(ValueError, match="empty"):
        stratified_split([])


def test_single_class_labels_raise():
    with pytest.raises(ValueError, match="two classes"):
        stratified_split([1] * 100)


def test_default_config_matches_plan():
    """Confirm the PLAN.md Section 5 defaults (70/15/15, stratified)."""
    labels = _make_labels(1000)
    split = stratified_split(labels)
    total = len(split)
    assert split.counts["train"] / total == pytest.approx(0.70)
    assert split.counts["val"] / total == pytest.approx(0.15)
    assert split.counts["test"] / total == pytest.approx(0.15)
    assert split.random_state == 42


def test_save_and_load_roundtrip(tmp_path):
    labels = _make_labels(500)
    split = stratified_split(labels)
    path = tmp_path / "split_indices.json"
    saved_path = save_splits(split, path=path, labels=labels)
    assert saved_path.exists()

    loaded = load_splits(saved_path)
    assert isinstance(loaded, DatasetSplit)
    assert list(loaded.train_idx) == list(split.train_idx)
    assert list(loaded.val_idx) == list(split.val_idx)
    assert list(loaded.test_idx) == list(split.test_idx)
    assert loaded.random_state == 42

    with open(saved_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["counts"] == split.counts
    assert manifest["ratios"] == {"train": 0.7, "val": 0.15, "test": 0.15}
    assert "0" in manifest["label_distribution"]["train"]


def test_default_path_constant():
    assert DEFAULT_SPLIT_PATH == Path("backend/ml/data/processed/split_indices.json")
