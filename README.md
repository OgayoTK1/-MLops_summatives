# Digit Vision - End-to-End ML Pipeline (MLOps Summative)

 **Machine Learning Pipeline Summative Assignment**

An end-to-end MLOps pipeline for classifying handwritten digit **images** (0-9):
data acquisition → preprocessing → CNN training/evaluation → a retrainable,
containerized FastAPI service → a Streamlit UI → Docker Compose scaling →
Locust load testing.

-  **Video Demo:** `<>`
-  **Live API URL:** `<https://digit-vision-api.onrender.com/>`
-  **Live UI URL:** `<https://digit-vision-ui.onrender.com/>`
-  **GitHub Repo:** `<https://github.com/OgayoTK1/-MLops_summatives>`

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


digit-vision-mlops/

│
├── README.md
├── requirements.txt         # full local/dev env (notebook, Locust, EDA)
├── requirements-api.txt     # slim deps for the API Docker image (no streamlit)
├── requirements-ui.txt      # slim deps for the UI Docker image (no tensorflow)
├── Dockerfile.api          # FastAPI service image
├── Dockerfile.ui           # Streamlit UI image
├── docker-compose.yml      # orchestrates api (scalable) + nginx LB + ui
├── nginx.conf              # load balancer config for scaled API replicas
├── locustfile.py           # load test
│
├── notebook/
│   └── digit_vision_pipeline.ipynb   # data acquisition -> preprocessing -> training -> evaluation -> retrain demo
│
├── src/
│   ├── preprocessing.py    # dataset build + shared preprocessing (train == serve)
│   ├── model.py            # CNN architecture, train(), evaluate_model(), load_trained_model()
│   └── prediction.py       # predict_image() used by the API
│
├── api/
│   └── main.py             # FastAPI: /predict /upload /retrain /uptime /visualizations
│
├── ui/
│   └── app.py              # Streamlit dashboard: Predict / Visualizations / Upload & Retrain / Uptime
│
├── data/
│   ├── train/<0-9>/.png
│   ├── test/<0-9>/.png
│   └── retrain_incoming/<0-9>/   # staging area for newly uploaded images awaiting retrain
│
├── models/
│   ├── digit_model.keras           # trained model artifact
│   ├── metrics_history.json        # audit trail of every train/retrain run
│   └── *.png                       # saved evaluation/EDA figures
│
└── results/                # Locust CSV outputs go here (see Section 6)

---

## 3. Setup — Run Locally (no Docker)

```bash
git clone 
cd digit-vision-mlops
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Build the image dataset (real PNGs to data/train, data/test)
python3 src/preprocessing.py

# 2. Train the model (saves models/digit_model.keras + models/metrics_history.json)
python3 -m src.model

# 3. Start the API (terminal 1)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Start the UI (terminal 2)
API_URL=http://localhost:8000 streamlit run ui/app.py
```

Open the UI at `http://localhost:8501`, and interactive API docs (Swagger)
at `http://localhost:8000/docs`.

Alternatively, open and run `notebook/digit_vision_pipeline.ipynb` for the
full guided walkthrough with all preprocessing, training, and evaluation
steps explained and already executed.

---

## 4. Setup - Run with Docker (scalable)

```bash
# Build + start with a single API replica
docker compose up --build

# Scale the API to 3 replicas behind the nginx load balancer
docker compose up --build --scale api=3
```

- API (via load balancer): `http://localhost:8000`
- UI: `http://localhost:8501`

`docker-compose.yml` deliberately does **not** publish a host port on the
`api` service directly — it's designed to be scaled (`--scale api=N`), with
`nginx` (the `loadbalancer` service) as the single stable entry point that
round-robins across however many API replicas are running (via Docker's
embedded DNS + `resolver` directive in `nginx.conf`).

---

## 5. Using the System

### Predict a single image
- **UI:** " Predict" tab → upload one PNG/JPG digit image → "Run Prediction".
- **API directly:**
```bash
  curl -X POST http://localhost:8000/predict -F "file=@data/test/7/7_0000.png"
```

### Upload bulk data + trigger retraining
- **UI:** " Upload & Retrain" tab → pick the digit class the batch belongs
  to → upload multiple images → "Upload batch" → once enough new images
  have accumulated (or any time), click " Trigger Retraining Now".
- **API directly:**
```bash
  curl -X POST "http://localhost:8000/upload?label=5" \
       -F "files=@img1.png" -F "files=@img2.png"

  curl -X POST http://localhost:8000/retrain
```
  Retraining runs in a background task; poll `GET /retrain/status` or the
  UI's Uptime tab to see when it finishes. The API reloads the newly
  retrained model automatically — no restart needed.

### Model uptime
- **UI:** "Model Uptime" tab, or `GET /uptime` directly — shows service
  uptime, total requests served, and last retrain time.

### Data visualizations
- **UI:** " Data Visualizations" tab, or `GET /visualizations` — shows
  class balance, train/test split composition, and average pixel
  intensity per class, each with an interpretation of what it reveals
  about the dataset.

---

## 6. Load Testing with Locust (flood simulation)

`locustfile.py` simulates realistic traffic: `/predict` calls dominate
(the expensive model-inference path), mixed with lighter `/health`,
`/uptime`, and `/visualizations` calls.

**Interactive mode** (watch live in the Locust web UI):
```bash
locust -f locustfile.py --host http://localhost:8000
# open http://localhost:8089, set users + spawn rate, click Start
```

**Headless mode** (for the report - repeat once per container count):
```bash
# 1 API container
docker compose up --build -d --scale api=1
locust -f locustfile.py --host http://localhost:8000 \
       --headless -u 100 -r 10 -t 60s --csv=results/run_1container

# 2 API containers
docker compose up --build -d --scale api=2
locust -f locustfile.py --host http://localhost:8000 \
       --headless -u 100 -r 10 -t 60s --csv=results/run_2containers

# 4 API containers
docker compose up --build -d --scale api=4
locust -f locustfile.py --host http://localhost:8000 \
       --headless -u 100 -r 10 -t 60s --csv=results/run_4containers
```

Each run writes `results/run_<N>containers_stats.csv` with per-endpoint
average/median/p95/p99 latency and requests/sec. **Paste your three
resulting tables below** to show how latency drops and throughput rises
as container count increases — this is the "Results from Flood Request
Simulation" the rubric requires here in the README.

### Sample baseline (1 container, single-process dev run, 20 users, 20s)

*(Re-run the three-container-count comparison above against your actual
Docker Compose deployment for your submission - the row below is a
sanity-check baseline captured during development, not the final report.)*

| Endpoint | Requests | Avg (ms) | p95 (ms) | p99 (ms) | Req/s | Failures |
|---|---|---|---|---|---|---|
| POST /predict | 222 | 378 | 760 | 860 | 12.1 | 0 |
| GET /uptime | 76 | 321 | 620 | 950 | 4.4 | 0 |
| GET /health | 65 | 334 | 700 | 870 | 3.6 | 0 |
| GET /visualizations | 46 | 431 | 900 | 970 | 1.6 | 0 |
| **Aggregated** | **409** | **367** | **750** | **910** | **21.7** | **0** |

Expected trend once you repeat this at 1 / 2 / 4 containers behind nginx:
average and p95 latency should **decrease** and aggregate req/s should
**increase** as replica count grows, up to the point where the host
machine's own CPU becomes the bottleneck.

---

## 7. Deployment (cloud platform)

Any platform that runs Docker containers works (Render, Railway, AWS
ECS/EC2, GCP Cloud Run, Azure Container Apps, DigitalOcean App Platform).
General steps, using **Render** as a concrete example:

1.  Repo is pushed to GitHub.
2. On Render: **New → Web Service** → connect the repo → set **Language**
   to **Docker** → **Root Directory** to the folder containing this
   README (e.g. `digit-vision-mlops` if your repo nests it in a
   subfolder) → **Dockerfile Path** to `Dockerfile.api` → deploy. Note the
   resulting URL (this is your API URL).
3. **New → Web Service** again → same settings but **Dockerfile Path**
   `Dockerfile.ui` → add environment variable `API_URL=<the API URL from
   step 2>` → deploy.

---


