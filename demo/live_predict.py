"""Live demo: streams session data through the API and shows real-time predictions.

Reads sessions from offline parquet (simulating "new traffic"), fires them at
the running API, and renders a live-updating terminal dashboard:
  - Total processed
  - Anomalies caught (with breakdown by agent_type)
  - Recent predictions stream (color-coded)

Usage (with API running on localhost:8000):
    python -m demo.live_predict
    python -m demo.live_predict --speed 0.05    # 20 sessions/sec
    python -m demo.live_predict --shuffle       # randomize order
    python -m demo.live_predict --max 500       # only first 500 sessions
"""
import argparse
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import glob

import pandas as pd
import requests


# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[H"  # clear screen + move cursor to top


def load_sessions(parquet_path: str = "feature_repo/feature_repo/data/offline_features",
                  shuffle: bool = False,
                  max_sessions: int | None = None) -> pd.DataFrame:
    """Load sessions from offline parquet — these have known agent_type for ground truth."""
    if not Path(parquet_path).exists():
        raise FileNotFoundError(
            f"Parquet not found at {parquet_path}. "
            "Run the simulator + Spark first to generate data."
        )
    files = sorted(glob.glob(f"{parquet_path}/*.parquet"))
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Skipping {f}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["session_id"])
    if shuffle:
        df = df.sample(frac=1, random_state=None).reset_index(drop=True)
    if max_sessions:
        df = df.head(max_sessions)
    return df


def predict(session_id: str, api_url: str, timeout: float = 5.0) -> dict | None:
    """Call the prediction API for a single session."""
    try:
        r = requests.post(
            f"{api_url}/predict/anomaly",
            json={"session_id": session_id},
            timeout=timeout,
        )
        if r.status_code == 200:
            return r.json()
        return None
    except requests.RequestException:
        return None


def render_dashboard(stats: dict, recent: list[dict], total: int) -> None:
    """Re-render the terminal dashboard."""
    print(CLEAR, end="")

    # Header
    print(f"{BOLD}{BLUE}╔══════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{BLUE}║  ShopSentry — Live Anomaly Detection                              ║{RESET}")
    print(f"{BOLD}{BLUE}╚══════════════════════════════════════════════════════════════════╝{RESET}")
    print()

    # Top-line stats
    processed = stats["processed"]
    anomalies = stats["anomalies"]
    rate = (anomalies / processed * 100) if processed > 0 else 0
    api_failures = stats["api_failures"]

    print(f"  {BOLD}Sessions processed:{RESET} {processed:,} / {total:,}")
    print(f"  {BOLD}Anomalies caught:  {RESET} {RED}{anomalies}{RESET}  ({rate:.1f}% of traffic)")
    if api_failures:
        print(f"  {BOLD}API failures:      {RESET} {YELLOW}{api_failures}{RESET}")
    print()

    # Per-class breakdown vs ground truth (the cool part)
    print(f"  {BOLD}Detection breakdown vs ground truth:{RESET}")
    print(f"    {'Agent type':<12} {'seen':>6} {'caught':>8} {'recall':>8}")
    for atype in ("normal", "churning", "bot", "fraud"):
        seen = stats["by_type"][atype]
        caught = stats["caught_by_type"][atype]
        if atype in ("bot", "fraud"):
            recall = (caught / seen * 100) if seen > 0 else 0
            recall_str = f"{recall:.1f}%"
            color = GREEN if recall >= 90 else (YELLOW if recall >= 50 else RED)
        else:
            # For normals/churning, "caught" is false-positive count (worse if higher)
            fp = (caught / seen * 100) if seen > 0 else 0
            recall_str = f"{fp:.1f}% FP"
            color = GREEN if fp <= 10 else (YELLOW if fp <= 25 else RED)
        print(f"    {atype:<12} {seen:>6} {caught:>8} {color}{recall_str:>8}{RESET}")
    print()

    # Recent stream
    print(f"  {BOLD}Recent predictions:{RESET}")
    print(f"    {'time':<10} {'session_id':<14} {'truth':<10} {'score':>6} {'verdict':<12}")
    for row in recent[-10:]:
        ts = row["time"]
        sid = row["session_id"]
        truth = row["agent_type"]
        score = row["score"]
        is_anom = row["is_anomaly"]

        # Color logic: green = correct call, red = mistake
        actually_anom = truth in ("bot", "fraud")
        correct = (is_anom == actually_anom)

        if is_anom:
            verdict = "🚨 ANOMALY"
            verdict_color = GREEN if correct else RED
        else:
            verdict = "✓ normal "
            verdict_color = GREEN if correct else RED

        truth_color = RED if actually_anom else GRAY
        print(f"    {GRAY}{ts}{RESET}  {sid:<14} {truth_color}{truth:<10}{RESET}"
              f" {score:>6.3f} {verdict_color}{verdict}{RESET}")
    print()

    # Footer
    print(f"  {GRAY}Press Ctrl+C to stop. Updating every prediction.{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000",
                        help="API base URL")
    parser.add_argument("--speed", type=float, default=0.1,
                        help="Seconds between predictions (lower = faster)")
    parser.add_argument("--shuffle", action="store_true",
                        help="Randomize session order")
    parser.add_argument("--max", type=int, default=None,
                        help="Limit number of sessions to process")
    args = parser.parse_args()

    print(f"Loading sessions...")
    df = load_sessions(shuffle=args.shuffle, max_sessions=args.max)
    print(f"Loaded {len(df)} sessions. Hitting API at {args.api}")
    time.sleep(1)

    stats = {
        "processed": 0,
        "anomalies": 0,
        "api_failures": 0,
        "by_type": defaultdict(int),
        "caught_by_type": defaultdict(int),
    }
    recent: list[dict] = []
    total = len(df)

    try:
        for _, row in df.iterrows():
            sid = row["session_id"]
            atype = row.get("agent_type") or "unknown"

            result = predict(sid, args.api)

            if result is None:
                stats["api_failures"] += 1
                continue

            stats["processed"] += 1
            stats["by_type"][atype] += 1
            is_anom = result["is_anomaly"]
            if is_anom:
                stats["anomalies"] += 1
                stats["caught_by_type"][atype] += 1

            recent.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "session_id": sid,
                "agent_type": atype,
                "score": result["anomaly_score"],
                "is_anomaly": is_anom,
            })

            render_dashboard(stats, recent, total)
            time.sleep(args.speed)

    except KeyboardInterrupt:
        pass

    # Final summary
    print(f"\n{BOLD}Final stats:{RESET}")
    print(f"  Processed: {stats['processed']:,}")
    print(f"  Anomalies caught: {stats['anomalies']}")
    if stats["by_type"]["bot"]:
        bot_recall = stats["caught_by_type"]["bot"] / stats["by_type"]["bot"] * 100
        print(f"  Bot recall: {bot_recall:.1f}% ({stats['caught_by_type']['bot']}/{stats['by_type']['bot']})")
    if stats["by_type"]["fraud"]:
        fraud_recall = stats["caught_by_type"]["fraud"] / stats["by_type"]["fraud"] * 100
        print(f"  Fraud recall: {fraud_recall:.1f}% ({stats['caught_by_type']['fraud']}/{stats['by_type']['fraud']})")


if __name__ == "__main__":
    main()