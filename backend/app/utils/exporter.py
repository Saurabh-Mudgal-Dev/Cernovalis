"""Export of trained model artifacts into a deploy-ready bundle with an integrity manifest."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping

DEFAULT_EXPORT_ROOT = "backend/ml/export"

MODEL_ARTIFACTS: Dict[str, str] = {
    "logistic_regression": "backend/ml/models/saved/logistic_regression.joblib",
    "naive_bayes": "backend/ml/models/saved/naive_bayes.joblib",
    "random_forest": "backend/ml/models/saved/random_forest.joblib",
    "xgboost": "backend/ml/models/saved/xgboost_model.joblib",
    "bilstm_model": "backend/ml/models/saved/bilstm_model.keras",
    "bilstm_tokenizer": "backend/ml/models/saved/bilstm_tokenizer.json",
    "meta_classifier": "backend/ml/models/saved/meta_classifier.joblib",
    "tfidf_vectorizer": "backend/ml/models/vectorizers/tfidf_vectorizer.joblib",
    "feature_scaler": "backend/ml/models/vectorizers/feature_scaler.joblib",
    "feature_pipeline": "backend/ml/models/saved/feature_pipeline.joblib",
}
REPORT_ARTIFACTS: Dict[str, str] = {
    "baseline_report": "backend/ml/models/saved/baseline_report.json",
    "random_forest_report": "backend/ml/models/saved/random_forest_report.json",
    "xgboost_report": "backend/ml/models/saved/xgboost_report.json",
    "bilstm_report": "backend/ml/models/saved/bilstm_report.json",
    "ensemble_report": "backend/ml/models/saved/ensemble_report.json",
    "split_manifest": "backend/ml/data/processed/split_indices.json",
}

MANIFEST_NAME = "manifest.json"


def sha256_file(path: Path | str) -> str:
    """Stream a file through SHA-256 and return its hex digest."""
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version_info() -> Dict[str, object]:
    """Record the library versions the bundle was produced with, when importable."""
    versions: Dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for module_name in ("sklearn", "xgboost", "joblib", "numpy"):
        try:
            import importlib

            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", None)
        except Exception:
            versions[module_name] = None
    try:
        import keras

        versions["keras"] = keras.__version__
    except Exception:
        versions["keras"] = None
    return versions


def verify_bundle(export_dir: Path | str) -> Dict[str, object]:
    """Re-hash every artifact in a bundle and compare against its manifest."""
    export_dir = Path(export_dir)
    manifest_path = export_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest found at {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    failures = []
    for name, entry in manifest["artifacts"].items():
        artifact_path = export_dir / entry["filename"]
        if not artifact_path.exists():
            failures.append({"artifact": name, "problem": "missing"})
        elif sha256_file(artifact_path) != entry["sha256"]:
            failures.append({"artifact": name, "problem": "checksum_mismatch"})
    return {
        "ok": not failures,
        "checked": len(manifest["artifacts"]),
        "failures": failures,
    }


def export_bundle(
    out_dir: Path | str,
    sources: Mapping[str, Path | str],
    strict: bool = False,
) -> tuple[Path, Dict[str, object]]:
    """Copy artifacts into `out_dir`, write manifest.json, and return both.

    `sources` maps artifact name -> path on disk. `strict=True` raises
    FileNotFoundError instead of recording the gaps as missing.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, object] = {}
    missing: list[str] = []
    for name, source in sources.items():
        source_path = Path(source)
        if not source_path.exists():
            missing.append(name)
            continue
        filename = source_path.name
        shutil.copy2(source_path, out_path / filename)
        artifacts[name] = {
            "filename": filename,
            "sha256": sha256_file(out_path / filename),
            "size_bytes": (out_path / filename).stat().st_size,
            "source": str(source_path),
        }

    if strict and missing:
        raise FileNotFoundError("missing artifacts: " + ", ".join(missing))

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "versions": version_info(),
        "artifacts": artifacts,
        "missing": missing,
        "complete": not missing,
        "artifact_count": len(artifacts),
    }
    manifest_path = out_path / MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return out_path, manifest


def default_timestamp_dir() -> Path:
    """A fresh export directory named by UTC timestamp under the default root."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(DEFAULT_EXPORT_ROOT) / f"model_bundle_{stamp}"


def source_map(report: bool = True) -> Dict[str, str]:
    """The standard artifact map, optionally extended with the metrics reports."""
    sources = dict(MODEL_ARTIFACTS)
    if report:
        sources.update(REPORT_ARTIFACTS)
    return sources
