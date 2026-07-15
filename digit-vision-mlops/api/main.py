"""
api/main.py
-----------
FastAPI service exposing:
  GET  /health              -> liveness probe
  GET  /uptime              -> model/service uptime for the UI
  POST /predict             -> single-image prediction
  POST /upload              -> bulk image upload for retraining (saved to data/retrain_incoming)
  POST /retrain              -> trigger retraining using base + newly uploaded data
  GET  /retrain/status       -> whether an automatic retrain trigger threshold has been hit
  GET  /visualizations       -> data insights consumed by the UI's charts
  GET  /metrics/history      -> history of every training/retraining run's metrics
"""

import os
import io
import json
import shutil
import time
import threading
import datetime
from collections import Counter

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.preprocessing import dataset_class_distribution, CLASS_NAMES, IMG_SIZE
from src.prediction import predict_image, reload_model
from src.model import train as train_model, METRICS_PATH

APP_START_TIME = time.time()
DATA_DIR = "data"
INCOMING_DIR = os.path.join(DATA_DIR, "retrain_incoming")
RETRAIN_TRIGGER_THRESHOLD = 25  # auto-flag "ready to retrain" once this many new images arrive

os.makedirs(INCOMING_DIR, exist_ok=True)
for c in CLASS_NAMES:
    os.makedirs(os.path.join(INCOMING_DIR, c), exist_ok=True)

app = FastAPI(title="Digit Vision MLOps API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# simple in-memory state (a real production system would use a DB / Redis)
STATE = {
    "request_count": 0,
    "last_retrain_time": None,
    "retrain_in_progress": False,
    "last_retrain_metrics": None,
}
_lock = threading.Lock()


@app.middleware("http")
async def count_requests(request, call_next):
    with _lock:
        STATE["request_count"] += 1
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/uptime")
def uptime():
    seconds = time.time() - APP_START_TIME
    return {
        "uptime_seconds": round(seconds, 1),
        "uptime_human": str(datetime.timedelta(seconds=int(seconds))),
        "requests_served": STATE["request_count"],
        "retrain_in_progress": STATE["retrain_in_progress"],
        "last_retrain_time": STATE["last_retrain_time"],
        "started_at": datetime.datetime.fromtimestamp(APP_START_TIME).isoformat(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file (png/jpg).")
    image_bytes = await file.read()
    try:
        result = predict_image(image_bytes)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")
    return result


@app.post("/upload")
async def upload_bulk(files: list[UploadFile] = File(...), label: str = None):
    """Bulk upload of images for retraining. If `label` is provided, all
    files are assumed to belong to that class (e.g. UI sends one label
    per batch); otherwise the server tries to infer the label from each
    filename's leading digit (e.g. '7_scan.png' -> class 7)."""
    saved = 0
    skipped = []
    for f in files:
        content = await f.read()
        inferred_label = label
        if inferred_label is None:
            leading = f.filename.split("_")[0].split(".")[0]
            inferred_label = leading if leading in CLASS_NAMES else None
        if inferred_label not in CLASS_NAMES:
            skipped.append(f.filename)
            continue
        try:
            img = Image.open(io.BytesIO(content))
            img.verify()
        except Exception:
            skipped.append(f.filename)
            continue
        dest = os.path.join(INCOMING_DIR, inferred_label, f"upload_{int(time.time()*1000)}_{f.filename}")
        with open(dest, "wb") as out:
            out.write(content)
        saved += 1

    total_incoming = sum(dataset_class_distribution(INCOMING_DIR).values())
    return {
        "saved": saved,
        "skipped": skipped,
        "total_pending_incoming": total_incoming,
        "retrain_recommended": total_incoming >= RETRAIN_TRIGGER_THRESHOLD,
    }


@app.get("/retrain/status")
def retrain_status():
    total_incoming = sum(dataset_class_distribution(INCOMING_DIR).values())
    return {
        "pending_new_images": total_incoming,
        "threshold": RETRAIN_TRIGGER_THRESHOLD,
        "retrain_recommended": total_incoming >= RETRAIN_TRIGGER_THRESHOLD,
        "retrain_in_progress": STATE["retrain_in_progress"],
    }


def _do_retrain():
    STATE["retrain_in_progress"] = True
    try:
        # merge newly uploaded images into the training set
        for c in CLASS_NAMES:
            src_dir = os.path.join(INCOMING_DIR, c)
            dst_dir = os.path.join(DATA_DIR, "train", c)
            os.makedirs(dst_dir, exist_ok=True)
            for fname in os.listdir(src_dir):
                shutil.move(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

    model, history, metrics, cm, report = train_model(
            epochs=30, trigger="manual_api_retrain"
        )
        if metrics.get("model_saved", True):
            reload_model()
        STATE["last_retrain_metrics"] = metrics
        STATE["last_retrain_time"] = datetime.datetime.now(datetime.timezone.utc).isoformat()


@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    if STATE["retrain_in_progress"]:
        raise HTTPException(409, "A retrain is already in progress.")
    background_tasks.add_task(_do_retrain)
    return {"message": "Retraining started in the background. Poll /uptime or /retrain/status."}


@app.get("/visualizations")
def visualizations():
    train_dist = dataset_class_distribution(os.path.join(DATA_DIR, "train"))
    test_dist = dataset_class_distribution(os.path.join(DATA_DIR, "test"))
    incoming_dist = dataset_class_distribution(INCOMING_DIR)

    total_train = sum(train_dist.values()) or 1
    avg_pixel_intensity = {}  # a 3rd "feature" insight: brightness per class
    for c in CLASS_NAMES:
        class_dir = os.path.join(DATA_DIR, "train", c)
        if not os.path.isdir(class_dir):
            continue
        vals = []
        for fname in os.listdir(class_dir)[:50]:  # sample for speed
            try:
                img = Image.open(os.path.join(class_dir, fname)).convert("L")
                vals.append(sum(img.getdata()) / (IMG_SIZE * IMG_SIZE))
            except Exception:
                continue
        if vals:
            avg_pixel_intensity[c] = round(sum(vals) / len(vals), 2)

    return {
        "train_class_distribution": train_dist,
        "test_class_distribution": test_dist,
        "incoming_pending_distribution": incoming_dist,
        "avg_pixel_intensity_by_class": avg_pixel_intensity,
        "total_train_images": total_train,
    }


@app.get("/metrics/history")
def metrics_history():
    if not os.path.exists(METRICS_PATH):
        return []
    with open(METRICS_PATH) as f:
        return json.load(f)
