"""Live scorer: stand-in for the checkout service that calls the fraud API.

Watches the offline parquet folder Spark writes to. For every new batch file,
it scores each session through the API (session_id -> Feast/Redis lookup),
then appends the prediction plus the true label to a predictions log.
Monitoring reads that log to track drift and F1 over time.
"""
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scorer")

API_URL = os.getenv("API_URL", "http://localhost:8000")
FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "feature_repo/feature_repo/data/offline_features"))
PREDICTIONS_DIR = Path(os.getenv("PREDICTIONS_DIR", "feature_repo/feature_repo/data/predictions"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "5"))
STATE_FILE = PREDICTIONS_DIR / "_scored_files.txt"


def load_scored_files() -> set[str]:
    """Names of batch files already scored, so a restart doesn't re-score them."""
    if STATE_FILE.exists():
        return set(STATE_FILE.read_text().splitlines())
    return set()


def mark_scored(name: str) -> None:
    with STATE_FILE.open("a") as f:
        f.write(name + "\n")


def score_session(session_id: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_URL}/predict/anomaly", json={"session_id": session_id}, timeout=5
        )
        if r.status_code == 200:
            return r.json()
        logger.warning(f"API returned {r.status_code} for {session_id}")
    except requests.RequestException as e:
        logger.warning(f"API call failed for {session_id}: {e}")
    return None


def score_file(path: Path) -> None:
    df = pd.read_parquet(path).dropna(subset=["session_id"])
    # Spark runs in update mode, so one file can hold the same session twice.
    # Keep the latest row per session.
    df = df.sort_values("event_timestamp").drop_duplicates("session_id", keep="last")

    rows = []
    for rec in df.itertuples(index=False):
        result = score_session(rec.session_id)
        if result is None:
            continue
        rows.append({
            "scored_at": datetime.now(timezone.utc),
            "session_id": rec.session_id,
            "event_timestamp": rec.event_timestamp,
            "agent_type": rec.agent_type,  # ground truth from the simulator
            "true_label": int(rec.agent_type in ("bot", "fraud")),
            "anomaly_score": result["anomaly_score"],
            "is_anomaly": int(result["is_anomaly"]),
            "model_version": result["model_version"],
            "source_file": path.name,
        })

    if rows:
        out = PREDICTIONS_DIR / f"pred_{path.stem}.parquet"
        pd.DataFrame(rows).to_parquet(out, index=False)
    logger.info(f"{path.name}: scored {len(rows)}/{len(df)} sessions")


def main() -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    scored = load_scored_files()
    logger.info(f"Scorer started. API={API_URL}, watching {FEATURES_DIR}")

    while True:
        if FEATURES_DIR.exists():
            for path in sorted(FEATURES_DIR.glob("batch_*.parquet")):
                if path.name in scored:
                    continue
                try:
                    score_file(path)
                except Exception as e:  # keep the loop alive on a bad file
                    logger.error(f"Failed on {path.name}: {e}")
                scored.add(path.name)
                mark_scored(path.name)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()