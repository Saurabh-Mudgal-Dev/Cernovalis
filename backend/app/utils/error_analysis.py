"""Helpers for inspecting held-out classification errors."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

LABEL_NAMES = {0: "fake", 1: "real"}


def analyze_misclassifications(
    sample_indices: Sequence[int],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probability_real: Sequence[float],
    titles: Sequence[str],
    texts: Sequence[str],
    text_chars: int = 500,
) -> List[Dict[str, object]]:
    """Return test errors with labels, score confidence and bounded article excerpts."""
    arrays = [
        np.asarray(sample_indices).ravel(),
        np.asarray(y_true).ravel(),
        np.asarray(y_pred).ravel(),
        np.asarray(probability_real, dtype=float).ravel(),
        np.asarray(titles, dtype=object).ravel(),
        np.asarray(texts, dtype=object).ravel(),
    ]
    lengths = {array.size for array in arrays}
    if len(lengths) != 1:
        raise ValueError("sample indices, labels, scores, titles and texts must have equal length")
    if text_chars < 0:
        raise ValueError("text_chars must be non-negative")

    indices, truth, predicted, scores, title_values, text_values = arrays
    if np.any(~np.isin(truth, [0, 1])) or np.any(~np.isin(predicted, [0, 1])):
        raise ValueError("labels must be 0 (fake) or 1 (real)")
    if np.any(~np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("probability_real values must be finite and within [0, 1]")

    errors: List[Dict[str, object]] = []
    for index, actual, guess, score, title, text in zip(
        indices, truth, predicted, scores, title_values, text_values
    ):
        actual, guess = int(actual), int(guess)
        if actual == guess:
            continue
        score = float(score)
        errors.append(
            {
                "dataset_index": int(index),
                "true_label": actual,
                "true_class": LABEL_NAMES[actual],
                "predicted_label": guess,
                "predicted_class": LABEL_NAMES[guess],
                "error_type": "false_positive" if actual == 0 else "false_negative",
                "probability_real": score,
                "predicted_confidence": score if guess == 1 else 1.0 - score,
                "title": str(title or ""),
                "text_excerpt": str(text or "")[:text_chars],
            }
        )

    errors.sort(key=lambda item: (-item["predicted_confidence"], item["dataset_index"]))
    return errors


def summarize_errors(errors: Sequence[Dict[str, object]], n_samples: int) -> Dict[str, object]:
    """Summarize a complete misclassification list, including class-specific rates."""
    false_positives = sum(error["error_type"] == "false_positive" for error in errors)
    false_negatives = sum(error["error_type"] == "false_negative" for error in errors)
    mean_confidence = (
        float(np.mean([error["predicted_confidence"] for error in errors]))
        if errors
        else None
    )
    return {
        "n_samples": int(n_samples),
        "n_errors": len(errors),
        "error_rate": float(len(errors) / n_samples) if n_samples else 0.0,
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "mean_error_confidence": mean_confidence,
        "errors_included": len(errors),
    }


def shared_error_samples(
    errors_by_model: Dict[str, Sequence[Dict[str, object]]], min_models: int = 2
) -> List[Dict[str, object]]:
    """Find test articles misclassified by multiple models, ordered by model count."""
    if min_models < 2:
        raise ValueError("min_models must be at least 2")

    by_index: Dict[int, Dict[str, object]] = {}
    for model_name, errors in errors_by_model.items():
        for error in errors:
            index = int(error["dataset_index"])
            shared = by_index.setdefault(
                index,
                {
                    "dataset_index": index,
                    "true_label": int(error["true_label"]),
                    "true_class": error["true_class"],
                    "title": error["title"],
                    "text_excerpt": error["text_excerpt"],
                    "models": [],
                    "highest_error_confidence": 0.0,
                },
            )
            shared["models"].append(model_name)
            shared["highest_error_confidence"] = max(
                float(shared["highest_error_confidence"]),
                float(error["predicted_confidence"]),
            )

    matches = [item for item in by_index.values() if len(item["models"]) >= min_models]
    return sorted(
        matches,
        key=lambda item: (-len(item["models"]), -item["highest_error_confidence"], item["dataset_index"]),
    )
