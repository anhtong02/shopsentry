"""Locust load test for ShopSentry API.

Run from project root:
    locust -f load_tests/locustfile.py --host http://localhost:8000

Then open http://localhost:8089 to start the test via web UI.
Or headless:
    locust -f load_tests/locustfile.py --host http://localhost:8000 \
        --users 100 --spawn-rate 10 --run-time 60s --headless
"""
import random
from pathlib import Path

import pandas as pd
from locust import HttpUser, between, task


# Load real session_ids from offline parquet so we hit the cache, not defaults
def _load_session_ids() -> list[str]:
    parquet_dir = Path("feature_repo/feature_repo/data/offline_features")
    if not parquet_dir.exists():
        print("WARNING: parquet dir missing, using fake session IDs")
        return [f"fake_session_{i}" for i in range(1000)]
    df = pd.read_parquet(parquet_dir)
    ids = df["session_id"].dropna().unique().tolist()
    print(f"Loaded {len(ids)} session IDs for load test")
    return ids


SESSION_IDS = _load_session_ids()


# Fixed normal-looking features for explicit-feature requests
NORMAL_FEATURES = {
    "events_per_minute": 5.0,
    "unique_pages_visited": 3.0,
    "avg_time_between_events": 15.0,
    "cart_to_purchase_ratio": 0.5,
    "session_duration_seconds": 120.0,
    "event_type_diversity": 4,
    "has_payment": 0,
    "signup_to_purchase_speed": 0.0,
    "page_revisit_ratio": 0.1,
}

BOT_FEATURES = {
    "events_per_minute": 80.0,
    "unique_pages_visited": 1.0,
    "avg_time_between_events": 0.8,
    "cart_to_purchase_ratio": 0.0,
    "session_duration_seconds": 200.0,
    "event_type_diversity": 2,
    "has_payment": 0,
    "signup_to_purchase_speed": 0.0,
    "page_revisit_ratio": 0.0,
}


class ShopSentryUser(HttpUser):
    """Simulates a client hitting the prediction API."""

    # Wait between 0 and 0.1s between tasks per user. Aggressive.
    wait_time = between(0, 0.1)

    @task(5)
    def predict_via_session_id(self):
        """Most common path — session_id only, Feast does lookup."""
        sid = random.choice(SESSION_IDS)
        self.client.post("/predict/anomaly", json={"session_id": sid})

    @task(2)
    def predict_with_explicit_features(self):
        """Sometimes the caller provides features directly (bypasses Feast)."""
        # Mix normal/bot features ~50/50
        features = random.choice([NORMAL_FEATURES, BOT_FEATURES])
        self.client.post(
            "/predict/anomaly",
            json={"session_id": f"explicit_{random.randint(1, 10000)}", "features": features},
        )

    @task(1)
    def health_check(self):
        """Periodic health probe."""
        self.client.get("/health")