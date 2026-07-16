"""
ui/app.py
---------
Streamlit dashboard covering every required UI capability:
  1. Model up-time             -> Uptime tab
  2. Data visualizations       -> Visualizations tab
  3. Predict a single image    -> Predict tab
  4. Bulk upload + trigger retraining -> Upload & Retrain tab

Run with:  streamlit run ui/app.py
Configure the backend URL via the API_URL env var (defaults to localhost,
override to point at the deployed/dockerized API, e.g. http://api:8000
inside docker-compose, or your cloud API URL in production).
"""

import os
import time
import requests
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Digit Vision MLOps", page_icon="🔢", layout="wide")
st.title("🔢 Digit Vision - MLOps Dashboard")
st.caption(f"Backend API: `{API_URL}`")

tab_predict, tab_viz, tab_upload, tab_uptime = st.tabs(
    [" Predict", " Data Visualizations", "Upload & Retrain", "⏱ Model Uptime"]
)

# - PREDICT
with tab_predict:
    st.subheader("Predict a single handwritten digit")
    st.write("Upload one PNG/JPG image of a handwritten digit (0-9).")
    img_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="predict_uploader")
    if img_file is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img_file, caption="Uploaded image", width=200)
        if st.button("Run Prediction", type="primary"):
            with st.spinner("Calling /predict ..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (img_file.name, img_file.getvalue(), img_file.type)},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    with col2:
                        st.success(f"Predicted class: **{result['predicted_class']}**  "
                                   f"(confidence: {result['confidence']*100:.2f}%)")
                        probs = pd.Series(result["class_probabilities"]).sort_index()
                        st.bar_chart(probs)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

#  VISUALIZATIONS
with tab_viz:
    st.subheader("Dataset insights")
    if st.button("Refresh visualizations"):
        st.rerun()
    try:
        viz = requests.get(f"{API_URL}/visualizations", timeout=15).json()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1. Class balance (train set)**")
            st.bar_chart(pd.Series(viz["train_class_distribution"]).sort_index())
            st.caption(
                "Classes are near-perfectly balanced (~139-146 images each). This matters "
                "because a skewed class distribution would bias the model toward majority "
                "digits and inflate accuracy while hurting minority-class recall."
            )
        with c2:
            st.markdown("**2. Train vs. test split size per class**")
            df = pd.DataFrame({
                "train": viz["train_class_distribution"],
                "test": viz["test_class_distribution"],
            }).fillna(0)
            st.bar_chart(df)
            st.caption(
                "An 80/20 stratified split per class ensures every digit is represented "
                "proportionally in both sets, so test accuracy is a fair estimate of "
                "real-world performance rather than an artifact of a lucky/unlucky split."
            )

        st.markdown("**3. Average pixel intensity (brightness) by class**")
        st.bar_chart(pd.Series(viz["avg_pixel_intensity_by_class"]).sort_index())
        st.caption(
            "Digits like '1' are drawn with thin strokes (lower average brightness) while "
            "digits like '0' and '8' use more of the canvas (higher brightness). This "
            "confirms the images carry genuine, learnable visual structure per class - "
            "it's not noise the CNN is fitting to."
        )

        if sum(viz["incoming_pending_distribution"].values()) > 0:
            st.markdown("**Pending images awaiting retraining**")
            st.bar_chart(pd.Series(viz["incoming_pending_distribution"]).sort_index())
    except Exception as e:
        st.error(f"Could not load visualizations: {e}")

# UPLOAD & RETRAIN
with tab_upload:
    st.subheader("Bulk upload new labeled images")
    st.write(
        "Upload multiple images that all belong to the **same digit class** to expand "
        "the training set, then trigger retraining."
    )
    label = st.selectbox("Digit class for this batch", [str(i) for i in range(10)])
    bulk_files = st.file_uploader(
        "Choose multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="bulk_uploader"
    )
    if bulk_files and st.button("Upload batch"):
        with st.spinner("Uploading..."):
            files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in bulk_files]
            try:
                resp = requests.post(f"{API_URL}/upload", params={"label": label}, files=files_payload, timeout=60)
                resp.raise_for_status()
                res = resp.json()
                st.success(f"Saved {res['saved']} image(s). Pending for retraining: {res['total_pending_incoming']}.")
                if res["retrain_recommended"]:
                    st.warning("Enough new data has accumulated — retraining is recommended!")
                if res["skipped"]:
                    st.info(f"Skipped (unreadable/invalid): {res['skipped']}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.divider()
    st.subheader("Retraining")
    try:
        status = requests.get(f"{API_URL}/retrain/status", timeout=10).json()
        st.write(f"Pending new images: **{status['pending_new_images']}** / threshold {status['threshold']}")
        if status["retrain_recommended"]:
            st.warning("Retrain trigger condition met — enough new data has been uploaded.")
        if status["retrain_in_progress"]:
            st.info("A retraining run is currently in progress...")
    except Exception as e:
        st.error(f"Could not fetch retrain status: {e}")

    if st.button("Trigger Retraining Now", type="primary"):
        try:
            resp = requests.post(f"{API_URL}/retrain", timeout=15)
            resp.raise_for_status()
            st.success(resp.json()["message"])
            with st.spinner("Retraining in background — polling status..."):
                for _ in range(60):
                    time.sleep(5)
                    s = requests.get(f"{API_URL}/retrain/status", timeout=10).json()
                    if not s["retrain_in_progress"]:
                        st.success("Retraining complete! Model reloaded with the new data.")
                        break
        except Exception as e:
            st.error(f"Could not trigger retraining: {e}")

    st.divider()
    st.subheader("Training / retraining history")
    try:
        hist = requests.get(f"{API_URL}/metrics/history", timeout=10).json()
        if hist:
            st.dataframe(pd.DataFrame(hist))
        else:
            st.info("No training history yet.")
    except Exception as e:
        st.error(f"Could not load metrics history: {e}")

# ---------------------------------------------------------------- UPTIME
with tab_uptime:
    st.subheader("Service / model uptime")
    if st.button("Refresh uptime"):
        st.rerun()
    try:
        up = requests.get(f"{API_URL}/uptime", timeout=10).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Uptime", up["uptime_human"])
        c2.metric("Requests served", up["requests_served"])
        c3.metric("Retrain in progress", "Yes" if up["retrain_in_progress"] else "No")
        st.write(f"Service started at: `{up['started_at']}`")
        st.write(f"Last retrain: `{up['last_retrain_time'] or 'never'}`")
    except Exception as e:
        st.error(f"Could not reach API at {API_URL}: {e}")
