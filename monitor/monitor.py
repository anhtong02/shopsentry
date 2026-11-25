"""Model performance monitor.

Every POLL_SECONDS, reads the scorer's predictions (prediction + true label),
takes the most recent WINDOW_SIZE sessions, and publishes live metrics at
http://<host>:8001/metrics for Prometheus to collect:

  shopsentry_model_precision / _recall / _f1   (anomaly class, i.e. bot or fraud)
  shopsentry_flag_rate{agent_type=...}         (share of each agent type flagged)
  shopsentry_window_sessions                   (sessions in the current window)
"""
import logging
import os
import time
from pathlib import Path

import pandas as pd
from prometheus_client import Gauge, start_http_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("monitor")

PREDICTIONS_DIR = Path(os.getenv("PREDICTIONS_DIR", "feature_repo/feature_repo/data/predictions"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "2000"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "10"))
PORT = int(os.getenv("MONITOR_PORT", "8001"))

PRECISION = Gauge("shopsentry_model_precision", "Precision on anomalies, recent window")
RECALL = Gauge("shopsentry_model_recall", "Recall on anomalies, recent window")
F1 = Gauge("shopsentry_model_f1", "F1 on anomalies, recent window")
FLAG_RATE = Gauge("shopsentry_flag_rate", "Share of sessions flagged, by agent type", ["agent_type"])
WINDOW_N = Gauge("shopsentry_window_sessions", "Sessions in the current window")


def load_recent() -> pd.DataFrame:
    files = sorted(PREDICTIONS_DIR.glob("pred_*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("scored_at").tail(WINDOW_SIZE)


def precision_recall_f1(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def update() -> None:
    df = load_recent()
    if df.empty:
        logger.info("No predictions yet")
        return

    p, r, f = precision_recall_f1(df["true_label"], df["is_anomaly"])
    PRECISION.set(p)
    RECALL.set(r)
    F1.set(f)
    WINDOW_N.set(len(df))
    for agent, rate in df.groupby("agent_type")["is_anomaly"].mean().items():
        FLAG_RATE.labels(agent).set(rate)
    logger.info(f"n={len(df)} precision={p:.3f} recall={r:.3f} f1={f:.3f}")


def main() -> None:
    start_http_server(PORT)
    logger.info(f"Monitor serving metrics on :{PORT}/metrics, window={WINDOW_SIZE}")
    while True:
        try:
            update()
        except Exception as e:  # keep running on a bad read
            logger.error(f"Update failed: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()