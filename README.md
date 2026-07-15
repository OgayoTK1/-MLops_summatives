# Digit Vision — End-to-End ML Pipeline (MLOps Summative)

**African Leadership University — Machine Learning Pipeline Summative Assignment**

An end-to-end MLOps pipeline for classifying handwritten digit **images** (0-9):
data acquisition → preprocessing → CNN training/evaluation → a retrainable,
containerized FastAPI service → a Streamlit UI → Docker Compose scaling →
Locust load testing.

- 🎥 **Video Demo:** `<PASTE YOUR YOUTUBE LINK HERE>`
- 🌐 **Live API URL:** `<PASTE YOUR DEPLOYED API URL HERE>`
- 🌐 **Live UI URL:** `<PASTE YOUR DEPLOYED UI URL HERE>`
- 📦 **GitHub Repo:** `<PASTE YOUR GITHUB REPO URL HERE>`

> These four placeholders are the only things left for you to fill in after
> you (1) record a screen+camera walkthrough and upload it to YouTube, and
> (2) push this repo and deploy it — see **Deployment** below for exact
> steps. Everything else in this repository is complete, tested, and runs
> end-to-end.

---

## 1. Project Description

This project classifies **images** of handwritten digits (0-9) using a
Convolutional Neural Network. It is built on `sklearn.datasets.load_digits`
(1,797 real 8×8 grayscale digit scans), which we materialize to disk as
genuine **PNG image files** — `data/train/<class>/*.png` and
`data/test/<class>/*.png` — so the whole system (upload, storage,
retraining, inference) operates on real images end-to-end, not on
pre-packaged feature vectors.

The system implements every stage of the ML lifecycle:

| Stage | Where |
|---|---|
| Data acquisition | `src/preprocessing.py::build_image_dataset` |
| Data preprocessing | `src/preprocessing.py::preprocess_image` (shared by train + inference) |
| Model creation | `src/model.py::build_model` (CNN, BatchNorm, Dropout, Adam) |
| Model testing / evaluation | `src/model.py::evaluate_model` + `notebook/digit_vision_pipeline.ipynb` |
| Retraining + trigger | `src/model.py::train()`, exposed via `POST /retrain` and the UI's "Trigger Retraining" button |
| API | `api/main.py` (FastAPI) |
| UI | `ui/app.py` (Streamlit) |
| Containerization / scaling | `Dockerfile.api`, `Dockerfile.ui`, `docker-compose.yml`, `nginx.conf` |
| Load testing | `locustfile.py` |

**Model performance (test set, 360 held-out images, never seen in training):**

| Metric | Score |
|---|---|
| Accuracy | 0.989 |
| Precision (macro) | 0.989 |
| Recall (macro) | 0.989 |
| F1 (macro) | 0.989 |
| Loss | 0.024 |

Full classification report, confusion matrix, and training curves are in
`notebook/digit_vision_pipeline.ipynb` (already executed, with outputs saved).

---

## 2. Repository Structure
