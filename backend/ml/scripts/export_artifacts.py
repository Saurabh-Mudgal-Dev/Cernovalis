"""Export trained model, tokenizer, vectorizer, scaler and ensemble artifacts into a manifest-checked bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.utils.exporter import (  # noqa: E402
    default_timestamp_dir,
    export_bundle,
    source_map,
    verify_bundle,
)
from backend.app.utils.bilstm import (  # noqa: E402
    DEFAULT_MAX_VOCAB,
)
from backend.ml.scripts.train_baselines import load_dataset, split_docs_labels  # noqa: E402

DATA_DIR = Path("backend/ml/data/raw")
TOKENIZER_PATH = Path("backend/ml/models/saved/bilstm_tokenizer.json")
BILSTM_REPORT_PATH = Path("backend/ml/models/saved/bilstm_report.json")

ARTIFACT_HINTS: dict[str, str] = {
    "logistic_regression": "run backend/ml/scripts/train_baselines.py",
    "naive_bayes": "run backend/ml/scripts/train_baselines.py",
    "tfidf_vectorizer": "run backend/ml/scripts/train_baselines.py (it fits and saves the train-split vectorizer)",
    "random_forest": "run backend/ml/scripts/train_forest.py",
    "random_forest_report": "run backend/ml/scripts/train_forest.py",
    "xgboost": "run backend/ml/scripts/train_xgboost.py",
    "xgboost_report": "run backend/ml/scripts/train_xgboost.py",
    "bilstm_model": "run backend/ml/scripts/train_bilstm.py",
    "bilstm_report": "run backend/ml/scripts/train_bilstm.py",
    "bilstm_tokenizer": "run backend/ml/scripts/train_bilstm.py, or re-run this export with --with-tokenizer",
    "meta_classifier": "run backend/ml/scripts/train_ensemble.py",
    "ensemble_report": "run backend/ml/scripts/train_ensemble.py",
    "feature_scaler": "run backend/ml/scripts/train_features.py",
    "feature_pipeline": "run backend/ml/scripts/train_features.py",
    "baseline_report": "run backend/ml/scripts/train_baselines.py",
    "split_manifest": "run backend/ml/scripts/make_splits.py",
}


def build_tokenizer_word_index() -> dict[str, int]:
    """Rebuild the BiLSTM word index deterministically from the training split."""
    from backend.app.utils.bilstm import build_word_index, tokenize

    max_vocab = DEFAULT_MAX_VOCAB
    if BILSTM_REPORT_PATH.exists():
        with open(BILSTM_REPORT_PATH, "r", encoding="utf-8") as handle:
            max_vocab = int(json.load(handle).get("tokenization", {}).get("max_vocab", max_vocab))

    df, _ = load_dataset(DATA_DIR)
    docs, _ = split_docs_labels(df)
    token_seqs = {name: tokenize(docs[name]) for name in ("train", "val", "test")}
    return build_word_index(token_seqs["train"], max_vocab=max_vocab)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export trained artifacts into a manifest-checked bundle."
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="Export directory (default: timestamped dir under backend/ml/export/)")
    parser.add_argument("--no-reports", action="store_true",
                        help="Skip the JSON training reports and split manifest")
    parser.add_argument("--with-tokenizer", action="store_true",
                        help="Rebuild and save the BiLSTM word index if its JSON is missing "
                             "(re-runs dataset cleaning; takes a couple of minutes)")
    parser.add_argument("--strict", action="store_true",
                        help="Fail instead of recording artifacts that do not exist")
    args = parser.parse_args()

    if args.with_tokenizer and not TOKENIZER_PATH.exists():
        TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        word_index = build_tokenizer_word_index()
        with open(TOKENIZER_PATH, "w", encoding="utf-8") as handle:
            json.dump(word_index, handle)
        print(f"tokenizer rebuilt and saved to {TOKENIZER_PATH} ({len(word_index)} words)")

    sources = source_map(report=not args.no_reports)
    out_dir = args.out if args.out else default_timestamp_dir()

    existing = {name: path for name, path in sources.items() if Path(path).exists()}
    missing = [name for name, path in sources.items() if not Path(path).exists()]
    out_dir, manifest = export_bundle(out_dir, existing, strict=args.strict)

    print(f"exported {manifest['artifact_count']} artifacts to {out_dir}")
    for name, entry in manifest["artifacts"].items():
        print(f"  {name}: {entry['filename']} ({entry['size_bytes']} bytes)")
    if missing:
        joined = ", ".join(missing)
        print(f"MISSING ({len(missing)}): {joined}")
        for name in missing:
            hint = ARTIFACT_HINTS.get(name, "no hint registered — check backend/ml/scripts/")
            print(f"  hint [{name}]: {hint}")

    check = verify_bundle(out_dir)
    status = "OK" if check["ok"] else "FAILED"
    print(f"integrity check: {status} ({check['checked']} artifacts verified)")

    with open(out_dir / "manifest.json", "r", encoding="utf-8") as handle:
        saved_versions = json.load(handle)["versions"]
    print(f"bundle versions: {saved_versions}")


if __name__ == "__main__":
    main()
