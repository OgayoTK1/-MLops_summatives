"""
locustfile.py
-------------
Simulates a flood of requests against the deployed API. Mixes real
/predict calls (the expensive, model-inference path) with lightweight
/health and /uptime checks, matching realistic traffic patterns.

Usage:
  locust -f locustfile.py --host http://localhost:8000

Then open http://localhost:8089, set number of users + spawn rate, and run.
For headless CI-style runs (used to produce the report tables in the README):
  locust -f locustfile.py --host http://localhost:8000 \
         --headless -u 100 -r 10 -t 60s --csv=results/run_1container
"""

import os
import glob
import random
from locust import HttpUser, task, between

SAMPLE_IMAGES = glob.glob("data/test/*/*.png")


class DigitVisionUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task(5)
    def predict(self):
        if not SAMPLE_IMAGES:
            return
        img_path = random.choice(SAMPLE_IMAGES)
        with open(img_path, "rb") as f:
            self.client.post(
                "/predict",
                files={"file": (os.path.basename(img_path), f, "image/png")},
                name="/predict",
            )

    @task(2)
    def uptime(self):
        self.client.get("/uptime", name="/uptime")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")

    @task(1)
    def visualizations(self):
        self.client.get("/visualizations", name="/visualizations")
