"""Reproduce every number in SchoolWhool's first public ML project video.

Run ``python project.py`` from this folder. The notebook and video use the
same calculations; no score is hand-entered into the animation.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "penguins.csv"
# Two dimensions let a beginner see the actual data and decision regions.
# They are chosen for the story before the final holdout split is evaluated.
FEATURES = (
    "bill_length_mm", "bill_depth_mm"
)
CLASSES = ("Adelie", "Chinstrap", "Gentoo")
SEED = 20260925


def load_rows() -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, list[int]]:
    """Keep the four measured columns; identify incomplete rows explicitly."""
    with DATA.open(
        newline="", encoding="utf-8"
    ) as handle:
        source = list(csv.DictReader(handle))
    clean: list[list[float]] = []
    labels: list[str] = []
    removed: list[int] = []
    for source_index, row in enumerate(source):
        raw = [row[name] for name in FEATURES]
        if any(
            value in ("", "NA") for value in raw
        ):
            removed.append(source_index)
            continue
        measurements = [float(v) for v in raw]
        clean.append(measurements)
        labels.append(row["species"])
    return source, np.asarray(clean, dtype=float), np.asarray(labels), removed


def fit_project() -> dict[str, object]:
    """Return the real fitted pipeline and split, for code and visualizations."""
    source, x, y, removed = load_rows()
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.20, random_state=SEED, stratify=y
    )
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_counts = Counter(y_train)
    majority = max(CLASSES, key=lambda label: train_counts[label])
    baseline_predictions = np.repeat(
        majority, len(test_idx)
    )

    # The scaler is fit on training rows only because it lives inside the
    # pipeline. The test rows never influence its means or standard deviations.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000, random_state=SEED
        ),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(
        x_test
    )
    baseline_correct = int(np.sum(baseline_predictions == y_test))
    return locals()


def analyze() -> dict[str, object]:
    artifacts = fit_project()
    source = artifacts["source"]
    x = artifacts["x"]
    y = artifacts["y"]
    removed = artifacts["removed"]
    train_idx = artifacts["train_idx"]
    test_idx = artifacts["test_idx"]
    x_train = artifacts["x_train"]
    x_test = artifacts["x_test"]
    y_train = artifacts["y_train"]
    y_test = artifacts["y_test"]
    train_counts = artifacts["train_counts"]
    majority = artifacts["majority"]
    baseline_predictions = artifacts["baseline_predictions"]
    baseline_correct = artifacts["baseline_correct"]
    model = artifacts["model"]
    predictions = artifacts["predictions"]
    probabilities = artifacts["probabilities"]
    correct = int(np.sum(
        predictions == y_test
    ))
    errors = np.flatnonzero(
        predictions != y_test
    )
    matrix = confusion_matrix(y_test, predictions, labels=CLASSES)
    scaler = model.named_steps["standardscaler"]
    estimator = model.named_steps["logisticregression"]

    # For a visible one-row walkthrough, choose a correct prediction with
    # nontrivial probabilities rather than a hand-picked perfect-looking row.
    correct_positions = np.flatnonzero(predictions == y_test)
    walkthrough_pos = int(
        min(correct_positions, key=lambda pos: probabilities[pos].max())
    )
    exemplar = {
        "test_position": walkthrough_pos,
        "source_clean_index": int(test_idx[walkthrough_pos]),
        "measurements": dict(zip(FEATURES, x_test[walkthrough_pos].tolist())),
        "actual_species": str(y_test[walkthrough_pos]),
        "predicted_species": str(predictions[walkthrough_pos]),
        "probabilities": dict(zip(estimator.classes_.tolist(), probabilities[walkthrough_pos].tolist())),
        "standardized_measurements": dict(
            zip(FEATURES, scaler.transform(x_test[[walkthrough_pos]])[0].tolist())
        ),
    }

    return {
        "dataset": "Palmer Penguins",
        "seed": SEED,
        "features": list(FEATURES),
        "classes": list(CLASSES),
        "raw_rows": len(source),
        "incomplete_rows_removed": len(removed),
        "removed_source_indices_zero_based": removed,
        "clean_rows": len(y),
        "class_counts": {name: int(np.sum(y == name)) for name in CLASSES},
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_counts": {name: int(train_counts[name]) for name in CLASSES},
        "test_counts": {name: int(np.sum(y_test == name)) for name in CLASSES},
        "baseline_label": majority,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": float(accuracy_score(y_test, baseline_predictions)),
        "model_correct": correct,
        "model_errors": int(len(errors)),
        "model_accuracy": float(accuracy_score(y_test, predictions)),
        "confusion_matrix": matrix.tolist(),
        "train_means": dict(zip(FEATURES, scaler.mean_.tolist())),
        "train_scales": dict(zip(FEATURES, scaler.scale_.tolist())),
        "exemplar": exemplar,
        "error_examples": [
            {
                "test_position": int(pos),
                "actual": str(y_test[pos]),
                "predicted": str(predictions[pos]),
                "measurements": dict(zip(FEATURES, x_test[pos].tolist())),
            }
            for pos in errors
        ],
        "test_actual": y_test.tolist(),
        "test_predicted": predictions.tolist(),
        "test_probability_by_class": probabilities.tolist(),
        "test_indices_into_clean_rows": test_idx.tolist(),
    }


if __name__ == "__main__":
    result = analyze()
    destination = HERE / "results.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"{result['raw_rows']} raw; {result['clean_rows']} usable; "
        f"{result['train_rows']} train / {result['test_rows']} test; "
        f"baseline {result['baseline_correct']}/{result['test_rows']} "
        f"({result['baseline_accuracy']:.1%}); model "
        f"{result['model_correct']}/{result['test_rows']} "
        f"({result['model_accuracy']:.1%}); "
        f"errors {result['model_errors']}"
    )
    for item in result["error_examples"]:
        print(item["actual"],
              item["predicted"])
        print(item["measurements"])
