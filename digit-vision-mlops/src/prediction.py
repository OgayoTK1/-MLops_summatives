"""
prediction.py
-------------
Loads the persisted model once and exposes a single predict_image()
function used by the API. Keeping this separate from api/main.py means
the same prediction logic can be unit-tested or reused by a CLI/notebook.
"""

import numpy as np
from src.preprocessing import preprocess_image_bytes, CLASS_NAMES
from src.model import load_trained_model

_model = None  # lazy singleton, loaded once per process


def get_model():
    global _model
    if _model is None:
        _model = load_trained_model()
    return _model


def predict_image(image_bytes: bytes) -> dict:
    """Run the full pipeline: raw bytes -> preprocessing -> model ->
    human-readable prediction with confidence + full class probabilities."""
    model = get_model()
    x = preprocess_image_bytes(image_bytes)
    x = np.expand_dims(x, axis=0)  # batch dim
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "class_probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
    }


def reload_model():
    """Force the API to pick up a freshly retrained model file without
    restarting the process."""
    global _model
    _model = load_trained_model()
    return _model
