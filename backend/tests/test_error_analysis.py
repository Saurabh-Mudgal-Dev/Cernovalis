"""Tests for held-out model error analysis helpers."""

import pytest

from backend.app.utils.error_analysis import (
    analyze_misclassifications,
    shared_error_samples,
    summarize_errors,
)


def test_analyze_misclassifications_filters_correct_predictions_and_labels():
    errors = analyze_misclassifications(
        sample_indices=[10, 11, 12, 13],
        y_true=[0, 1, 0, 1],
        y_pred=[0, 0, 1, 1],
        probability_real=[0.1, 0.3, 0.8, 0.9],
        titles=["correct fake", "false negative", "false positive", "correct real"],
        texts=["a", "b", "c", "d"],
    )
    assert [error["dataset_index"] for error in errors] == [12, 11]
    fp, fn = errors
    assert (fp["true_class"], fp["predicted_class"], fp["error_type"]) == (
        "fake", "real", "false_positive"
    )
    assert fp["predicted_confidence"] == pytest.approx(0.8)
    assert (fn["true_class"], fn["predicted_class"], fn["error_type"]) == (
        "real", "fake", "false_negative"
    )
    assert fn["predicted_confidence"] == pytest.approx(0.7)


def test_analyze_misclassifications_truncates_body_excerpt():
    errors = analyze_misclassifications(
        [3], [0], [1], [0.8], ["headline"], ["0123456789"], text_chars=4
    )
    assert errors[0]["text_excerpt"] == "0123"


def test_analyze_misclassifications_validates_lengths_and_scores():
    with pytest.raises(ValueError, match="equal length"):
        analyze_misclassifications([1], [0], [1], [0.9], ["title"], [])
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        analyze_misclassifications([1], [0], [1], [1.1], ["title"], ["text"])
    with pytest.raises(ValueError, match="non-negative"):
        analyze_misclassifications([1], [0], [1], [0.9], ["title"], ["text"], -1)


def test_analyze_misclassifications_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="0 \\(fake\\) or 1 \\(real\\)"):
        analyze_misclassifications([1], [2], [1], [0.9], ["title"], ["text"])


def test_summarize_errors_counts_fp_fn_and_rates():
    errors = analyze_misclassifications(
        [0, 1, 2, 3], [0, 1, 0, 1], [1, 0, 0, 1],
        [0.9, 0.2, 0.1, 0.95], ["a", "b", "c", "d"], ["", "", "", ""],
    )
    summary = summarize_errors(errors, n_samples=4)
    assert summary["n_errors"] == 2
    assert summary["error_rate"] == pytest.approx(0.5)
    assert summary["false_positives"] == 1
    assert summary["false_negatives"] == 1
    assert summary["mean_error_confidence"] == pytest.approx(0.85)


def test_summarize_errors_empty():
    summary = summarize_errors([], 0)
    assert summary["n_errors"] == 0
    assert summary["error_rate"] == 0.0
    assert summary["mean_error_confidence"] is None


def test_shared_error_samples_intersects_models_and_sorts():
    first = analyze_misclassifications(
        [0, 1, 2], [0, 1, 0], [1, 0, 1], [0.9, 0.1, 0.99],
        ["a", "b", "c"], ["text a", "text b", "text c"],
    )
    second = analyze_misclassifications(
        [0, 1, 2], [0, 1, 0], [0, 0, 1], [0.1, 0.8, 0.7],
        ["a", "b", "c"], ["text a", "text b", "text c"],
    )
    shared = shared_error_samples({"one": first, "two": second})
    assert [item["dataset_index"] for item in shared] == [2, 1]
    assert shared[0]["models"] == ["one", "two"]
    assert shared[1]["models"] == ["one", "two"]


def test_shared_error_samples_requires_at_least_two_models():
    with pytest.raises(ValueError, match="at least 2"):
        shared_error_samples({}, min_models=1)
