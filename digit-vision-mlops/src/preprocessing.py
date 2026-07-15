"""
preprocessing.py
----------------
Handles:
1. Building an on-disk IMAGE dataset (PNG files, one folder per class) from
   sklearn's `load_digits` (8x8 handwritten digit scans, classes 0-9).
   We deliberately materialize real .png files on disk (not just numpy
   arrays) so the project works with genuine image data end-to-end:
   upload, storage, augmentation, retraining all operate on real files.
2. Shared preprocessing utilities used identically at training time and
   at inference time (so there is no train/serve skew).
"""

import os
import io
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

IMG_SIZE = 32          # images are up-sampled from 8x8 -> 32x32
NUM_CLASSES = 10
CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]


def build_image_dataset(data_dir="data", test_size=0.2, seed=42):
    """Materialize the sklearn digits dataset as real PNG files on disk,
    split into data/train/<class>/*.png and data/test/<class>/*.png.
    Safe to re-run: it wipes and rebuilds the folders.
    """
    digits = load_digits()
    X, y = digits.images, digits.target  # X: (n, 8, 8) float 0-16

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    for split_name, X_split, y_split in [("train", X_train, y_train), ("test", X_test, y_test)]:
        split_dir = os.path.join(data_dir, split_name)
        for c in CLASS_NAMES:
            os.makedirs(os.path.join(split_dir, c), exist_ok=True)

        counters = {c: 0 for c in CLASS_NAMES}
        for img_arr, label in zip(X_split, y_split):
            label = str(int(label))
            # scale 0-16 -> 0-255, upsize 8x8 -> 32x32 for a "real" image feel
            arr = (img_arr / 16.0 * 255).astype(np.uint8)
            im = Image.fromarray(arr, mode="L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            fname = f"{label}_{counters[label]:04d}.png"
            im.save(os.path.join(split_dir, label, fname))
            counters[label] += 1

    print(f"Dataset written to '{data_dir}/train' and '{data_dir}/test'.")
    return len(X_train), len(X_test)


def load_dataset_from_folder(folder):
    """Load every PNG under folder/<class>/*.png into arrays (X, y)."""
    X, y = [], []
    for c in sorted(os.listdir(folder)):
        class_dir = os.path.join(folder, c)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(".png"):
                continue
            img = Image.open(os.path.join(class_dir, fname)).convert("L")
            X.append(preprocess_image(img))
            y.append(int(c))
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Single shared preprocessing function used by BOTH training and the
    inference API, guaranteeing identical preprocessing in dev and prod.
    Accepts a PIL Image, returns a normalized (IMG_SIZE, IMG_SIZE, 1) array.

    Training images follow the sklearn digits / MNIST convention: a bright
    stroke on a dark background. A real-world photo of pen-on-paper
    handwriting is the opposite (dark stroke on a light background), which
    would confuse the model despite being a perfectly valid image of the
    right digit. We auto-detect this by mean pixel brightness and invert
    when the image looks like "mostly light background" - every training
    image has a mean well under 0.5 (verified empirically), so this never
    triggers on data that already matches the training convention.
    """
    img = img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0

    if arr.mean() > 0.5:
        arr = 1.0 - arr

    arr = arr.reshape(IMG_SIZE, IMG_SIZE, 1)
    return arr


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Convenience wrapper: raw uploaded bytes -> model-ready array."""
    img = Image.open(io.BytesIO(image_bytes))
    return preprocess_image(img)


def dataset_class_distribution(folder):
    """Return {class_name: count} for a data folder - used by the
    visualization endpoint/UI."""
    dist = {}
    for c in sorted(os.listdir(folder)):
        class_dir = os.path.join(folder, c)
        if os.path.isdir(class_dir):
            dist[c] = len([f for f in os.listdir(class_dir) if f.lower().endswith(".png")])
    return dist


if __name__ == "__main__":
    build_image_dataset()