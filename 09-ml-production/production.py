"""Local examples of fitted-pipeline inference; no server or deployment."""
from __future__ import annotations
import csv
import hashlib
from numbers import Real
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
FEATURES = ("bill_length_mm", "bill_depth_mm")
SEED = 20260925
DATA_SHA256 = "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"

def fit_training_pipeline():
    """Reproduce the first project's training split; leave test rows unused."""
    data = HERE / "data/penguins.csv"
    if hashlib.sha256(data.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("Dataset differs from the version used in the lesson")
    with data.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    usable = [r for r in rows if all(r[f] not in ("", "NA") for f in FEATURES)]
    x = np.array([[float(r[f]) for f in FEATURES] for r in usable])
    y = np.array([r["species"] for r in usable])
    train, test = train_test_split(np.arange(len(y)), test_size=.20,
                                  random_state=SEED, stratify=y)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    pipeline.fit(x[train], y[train])
    return pipeline, {"raw_rows": len(rows), "usable_rows": len(usable),
                      "training_rows": len(train), "sealed_test_rows": len(test)}

def prepare_input(request):
    """Check named millimeter measurements and return one ordered feature row."""
    if not isinstance(request, dict):
        raise ValueError("Input must be a dictionary of named measurements")
    missing = set(FEATURES) - set(request)
    if missing:
        raise ValueError("Missing fields: " + ", ".join(sorted(missing)))
    if set(request) - set(FEATURES):
        raise ValueError("Unexpected fields in request")
    values = []
    for name in FEATURES:
        value = request[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be numeric; booleans and strings are not measurements")
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive millimeter measurement")
        values.append(value)
    return np.array([values], dtype=float)

def predict_request(pipeline, request, model_version="penguins-v1"):
    """Use the existing fitted pipeline. This function never calls fit."""
    row = prepare_input(request)
    return {"predicted_species": str(pipeline.predict(row)[0]), "model_version": model_version}

def predict_batch(pipeline, requests):
    """Validate each request before predicting a collection together."""
    if not requests:
        return []
    rows = np.concatenate([prepare_input(request) for request in requests], axis=0)
    return pipeline.predict(rows).tolist()
