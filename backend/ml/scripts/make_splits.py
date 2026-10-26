"""Persist stratified train/validation/test split indices for the ISOT dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.utils.dataset_split import (  # noqa: E402
    DEFAULT_SPLIT_PATH,
    DatasetSplit,
    save_splits,
    stratified_split,
)
from backend.ml.scripts.train_features import load_data  # noqa: E402

DATA_DIR = Path("backend/ml/data/raw")


def make_splits(
    data_dir: Path = DATA_DIR,
    out_path: Path = DEFAULT_SPLIT_PATH,
) -> DatasetSplit:
    """Load the merged ISOT dataset and persist stratified 70/15/15 indices."""
    df = load_data(data_dir)
    labels = df["label"].tolist()
    split = stratified_split(labels)
    save_splits(split, path=out_path, labels=labels)
    print(
        f"Split produced {split.counts['train']} train / "
        f"{split.counts['val']} validation / {split.counts['test']} test rows. "
        f"Saved to {out_path}"
    )
    return split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persist stratified train/val/test split indices."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing True.csv and Fake.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help="Output path for the split indices JSON manifest",
    )
    args = parser.parse_args()
    make_splits(data_dir=args.data_dir, out_path=args.out)


if __name__ == "__main__":
    main()
