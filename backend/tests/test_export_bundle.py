"""Tests for the artifact export bundle and integrity manifest helpers."""

import json

import numpy as np
import pytest

from backend.app.utils.exporter import (
    DEFAULT_EXPORT_ROOT,
    MANIFEST_NAME,
    MODEL_ARTIFACTS,
    REPORT_ARTIFACTS,
    export_bundle,
    sha256_file,
    source_map,
    verify_bundle,
    version_info,
)


@pytest.fixture
def toy_sources(tmp_path):
    """A fake artifact set: two existing files on disk and one gap."""
    root = tmp_path / "artifacts"
    root.mkdir()
    model = root / "model.joblib"
    model.write_bytes(b"\x00\x01\x02fakejoblib")
    config = root / "config.json"
    config.write_text(json.dumps({"max_len": 300}))
    return {
        "model": model,
        "config": config,
        "gap": root / "does_not_exist.joblib",
    }


def test_sha256_file_matches_expected_digest(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"hello world")
    import hashlib

    assert sha256_file(path) == hashlib.sha256(b"hello world").hexdigest()


def test_sha256_file_missing_raises(tmp_path):
    with pytest.raises(OSError):
        sha256_file(tmp_path / "absent.bin")


def test_sha256_file_streams_large_file(tmp_path):
    path = tmp_path / "big.bin"
    chunk = np.random.default_rng(0).bytes(1 << 20)
    with open(path, "wb") as handle:
        for _ in range(3):
            handle.write(chunk)
    import hashlib

    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_bundle_copies_and_manifests(tmp_path, toy_sources):
    out_dir = tmp_path / "bundle"
    written, manifest = export_bundle(out_dir, toy_sources)
    assert written == out_dir
    assert manifest["artifact_count"] == 2
    assert manifest["missing"] == ["gap"]
    assert manifest["complete"] is False
    assert manifest["artifacts"]["model"]["size_bytes"] == len(b"\x00\x01\x02fakejoblib")
    assert manifest["artifacts"]["model"]["sha256"] == sha256_file(toy_sources["model"])
    assert (out_dir / "model.joblib").read_bytes() == b"\x00\x01\x02fakejoblib"
    assert (out_dir / "config.json").exists()
    assert (out_dir / MANIFEST_NAME).exists()
    assert manifest["exported_at"]  # timestamp present


def test_export_bundle_strict_raises_on_missing(tmp_path, toy_sources):
    with pytest.raises(FileNotFoundError, match="gap"):
        export_bundle(tmp_path / "bundle2", toy_sources, strict=True)


def test_export_bundle_overwrites_and_checksums_differ(tmp_path, toy_sources):
    out_dir = tmp_path / "bundle"
    export_bundle(out_dir, toy_sources)
    toy_sources["model"].write_bytes(b"new content")
    written, manifest = export_bundle(out_dir, toy_sources)
    assert written == out_dir
    assert (out_dir / "model.joblib").read_bytes() == b"new content"
    assert manifest["artifacts"]["model"]["sha256"] == sha256_file(toy_sources["model"])


def test_verify_bundle_detects_tampering(tmp_path, toy_sources):
    out_dir = tmp_path / "bundle"
    export_bundle(out_dir, toy_sources)
    first = verify_bundle(out_dir)
    assert first == {"ok": True, "checked": 2, "failures": []}

    (out_dir / "model.joblib").write_bytes(b"tampered")
    second = verify_bundle(out_dir)
    assert second["ok"] is False
    assert second["failures"] == [{"artifact": "model", "problem": "checksum_mismatch"}]


def test_verify_bundle_detects_deleted_artifact(tmp_path, toy_sources):
    out_dir = tmp_path / "bundle"
    export_bundle(out_dir, toy_sources)
    (out_dir / "config.json").unlink()
    check = verify_bundle(out_dir)
    assert check["ok"] is False
    assert check["failures"] == [{"artifact": "config", "problem": "missing"}]


def test_verify_bundle_requires_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        verify_bundle(tmp_path)


def test_version_info_reports_installs():
    versions = version_info()
    import sklearn

    assert versions["sklearn"] == sklearn.__version__
    assert versions["numpy"] == np.__version__
    assert versions["python"].count(".") == 2
    assert "platform" in versions


def test_article_names_default_sources():
    sources = source_map(report=True)
    assert sources["logistic_regression"] == MODEL_ARTIFACTS["logistic_regression"]
    assert "split_manifest" in sources
    assert "ensemble_report" in sources
    without = source_map(report=False)
    assert "ensemble_report" not in without
    assert without["bilstm_model"].endswith("bilstm_model.keras")
    assert REPORT_ARTIFACTS["baseline_report"].endswith("baseline_report.json")
    assert DEFAULT_EXPORT_ROOT == "backend/ml/export"
    assert MANIFEST_NAME == "manifest.json"
