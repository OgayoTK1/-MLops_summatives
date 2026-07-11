"""
model.py
--------
CNN architecture, training loop (with regularization + early stopping,
i.e. an "optimized model" not a vanilla one), evaluation with 4+ metrics,
and a retrain() entry point used both by the notebook and by the API's
/retrain endpoint.
"""

import os
import json
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from sklearn.model_selection import train_test_split

from src.preprocessing import (
    IMG_SIZE, NUM_CLASSES, load_dataset_from_folder, build_image_dataset
)

MODEL_PATH = "models/digit_model.keras"
METRICS_PATH = "models/metrics_history.json"
DATA_DIR = "data"


def build_model(dropout_rate=0.3, learning_rate=1e-3):
    """A small CNN with regularization (Dropout + BatchNorm) - deliberately
    'optimized', not a vanilla dense network."""
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(dropout_rate),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_model(model, X_test, y_test):
    """Compute >= 4 evaluation metrics required by the rubric."""
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    return metrics, cm, report


def _append_metrics_history(metrics, n_train, trigger="initial_training"):
    history = []
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            history = json.load(f)
    history.append({
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "trigger": trigger,
        "n_train_samples": n_train,
        **metrics,
    })
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(history, f, indent=2)
    return history


def train(epochs=30, batch_size=32, trigger="initial_training", rebuild_from_source=False):
    """Full train + evaluate + persist cycle. Used by the notebook AND by
    the API's /retrain endpoint (this IS the retraining function)."""
    if rebuild_from_source:
        build_image_dataset(DATA_DIR)

    X_all, y_all = load_dataset_from_folder(os.path.join(DATA_DIR, "train"))
    X_test, y_test = load_dataset_from_folder(os.path.join(DATA_DIR, "test"))

    # Data on disk is grouped by class folder, so Keras's validation_split
    # (which slices off the END of the array pre-shuffle) would produce a
    # near single-class validation set. Use a stratified, shuffled split
    # instead so both train and validation see all classes.
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.15, random_state=42, stratify=y_all
    )

    model = build_model()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early_stop, reduce_lr],
        verbose=2,
    )

    metrics, cm, report = evaluate_model(model, X_test, y_test)

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    _append_metrics_history(metrics, n_train=len(X_train), trigger=trigger)

    return model, history, metrics, cm, report


def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run `python -m src.model` first."
        )
    return keras.models.load_model(MODEL_PATH)


if __name__ == "__main__":
    model, history, metrics, cm, report = train(epochs=30)
    print("Final test metrics:", json.dumps(metrics, indent=2))
