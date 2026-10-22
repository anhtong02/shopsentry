"""Live demo with a rich terminal dashboard.

Streams deduped sessions through the running API and shows, live:
  - a progress bar
  - catch rate per agent type vs ground truth (bars)
  - a confusion summary (caught, missed, false alarms)
  - API latency
  - a scrolling feed of recent predictions

Usage (API running on localhost:8000):
    python -m demo.live_predict_rich
    python -m demo.live_predict_rich --shuffle --max 300 --speed 0.03
    python -m demo.live_predict_rich --via feast      # look up features in Redis by session_id

By default it sends the session's features directly, so only the API (and MLflow,
for model loading) need to be running. --via feast also needs Redis + Feast.
"""
import argparse
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime

import requests
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from models.data_loader import feature_cols, load_training_data

session = requests.Session()
ANOMALY_TYPES = ("bot", "fraud")
AGENT_TYPES = ("normal", "churning", "bot", "fraud")
console = Console()


def predict(row: dict, api_url: str, via: str) -> tuple[dict | None, float]:
    """Call the API once. Returns (response_json or None, latency_ms)."""
    if via == "feast":
        payload = {"session_id": row["session_id"]}
    else:
        payload = {
            "session_id": row["session_id"],
            "features": {c: float(row[c]) for c in feature_cols},
        }
        payload["features"]["has_payment"] = int(row["has_payment"])
    t0 = time.perf_counter()
    try:
        r = session.post(f"{api_url}/predict/anomaly", json=payload, timeout=5)
        latency = (time.perf_counter() - t0) * 1000
        return (r.json() if r.status_code == 200 else None), latency
    except requests.RequestException:
        return None, (time.perf_counter() - t0) * 1000


def bar(fraction: float, width: int = 20, good_high: bool = True) -> Text:
    """A small colored bar. good_high=False flips colors (for false-alarm rates)."""
    filled = int(round(fraction * width))
    score = fraction if good_high else 1 - fraction
    color = "green" if score >= 0.9 else "yellow" if score >= 0.6 else "red"
    return Text("█" * filled, style=color) + Text("░" * (width - filled), style="grey35")


def header(model_version: str) -> Panel:
    title = Text("SHOPSENTRY", style="bold cyan") + Text("  live anomaly detection", style="grey70")
    sub = Text(f"model {model_version}", style="grey50")
    return Panel(Align.center(Group(Align.center(title), Align.center(sub))), border_style="cyan")


def detection_table(stats: dict) -> Panel:
    t = Table(expand=True, box=None, padding=(0, 1))
    t.add_column("Agent type", style="bold")
    t.add_column("Seen", justify="right")
    t.add_column("Flagged", justify="right")
    t.add_column("Rate", justify="right")
    t.add_column("", min_width=20)
    for a in AGENT_TYPES:
        seen = stats["seen"][a]
        flagged = stats["flagged"][a]
        frac = flagged / seen if seen else 0.0
        if a in ANOMALY_TYPES:
            label = f"{frac:.0%} caught"
            b = bar(frac, good_high=True)
            name = Text(a, style="red")
        else:
            label = f"{frac:.0%} false alarm"
            b = bar(frac, good_high=False)
            name = Text(a, style="grey70")
        t.add_row(name, str(seen), str(flagged), label, b)
    return Panel(t, title="Detection vs ground truth", border_style="blue")


def summary_panel(stats: dict) -> Panel:
    tp, fn, fp = stats["tp"], stats["fn"], stats["fp"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lat = stats["latency"]
    p50 = statistics.median(lat) if lat else 0.0
    p95 = sorted(lat)[int(len(lat) * 0.95) - 1] if len(lat) >= 20 else p50

    t = Table.grid(padding=(0, 2))
    t.add_column(style="grey70")
    t.add_column(justify="right")
    t.add_row("Caught", Text(str(tp), style="bold green"))
    t.add_row("Missed", Text(str(fn), style="bold red" if fn else "green"))
    t.add_row("False alarms", Text(str(fp), style="bold yellow" if fp else "green"))
    t.add_row("", "")
    t.add_row("Precision", f"{precision:.2f}")
    t.add_row("Recall", f"{recall:.2f}")
    t.add_row("F1", Text(f"{f1:.2f}", style="bold cyan"))
    t.add_row("", "")
    t.add_row("Latency p50", f"{p50:.0f} ms")
    t.add_row("Latency p95", f"{p95:.0f} ms")
    if stats["api_failures"]:
        t.add_row("API failures", Text(str(stats["api_failures"]), style="bold red"))
    return Panel(t, title="Scoreboard", border_style="magenta")


def feed_panel(recent: deque) -> Panel:
    t = Table(expand=True, box=None, padding=(0, 1))
    t.add_column("Time", style="grey50", no_wrap=True)
    t.add_column("Session", no_wrap=True)
    t.add_column("Truth")
    t.add_column("Score", justify="right")
    t.add_column("Verdict")
    for r in recent:
        actually_bad = r["agent_type"] in ANOMALY_TYPES
        correct = r["is_anomaly"] == actually_bad
        truth = Text(r["agent_type"], style="red" if actually_bad else "grey70")
        if r["is_anomaly"]:
            verdict = Text("FLAGGED", style="bold white on red" if correct else "bold black on yellow")
        else:
            verdict = Text("ok", style="green" if correct else "bold white on red")
        if not correct:
            verdict += Text(" (wrong)", style="yellow")
        score_style = "red" if r["score"] > 0.5 else "green"
        t.add_row(r["time"], str(r["session_id"])[:12], truth,
                  Text(f"{r['score']:.3f}", style=score_style), verdict)
    return Panel(t, title="Live feed", border_style="green")


def build_layout(progress: Progress, stats: dict, recent: deque, model_version: str) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(header(model_version), size=4),
        Layout(Panel(progress, border_style="cyan"), size=3),
        Layout(name="middle", size=9),
        Layout(feed_panel(recent), name="feed"),
    )
    layout["middle"].split_row(
        Layout(detection_table(stats), ratio=2),
        Layout(summary_panel(stats), ratio=1),
    )
    return layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--speed", type=float, default=0.05, help="Seconds between requests")
    parser.add_argument("--shuffle", action="store_true", help="Randomize session order")
    parser.add_argument("--max", type=int, default=300, help="Sessions to stream")
    parser.add_argument("--via", choices=["features", "feast"], default="features",
                        help="Send features directly, or look them up in Feast by session_id")
    args = parser.parse_args()

    with console.status("[cyan]Loading sessions..."):
        df, _ = load_training_data()
        if args.shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        df = df.head(args.max)

    try:
        info = requests.get(f"{args.api}/", timeout=3).json()
        model_version = f"api v{info.get('version', '?')}"
    except requests.RequestException:
        console.print(f"[bold red]Can't reach the API at {args.api}.[/] Start it with: uvicorn api.main:app --port 8000")
        return

    stats: dict = {
        "seen": defaultdict(int), "flagged": defaultdict(int),
        "tp": 0, "fn": 0, "fp": 0, "api_failures": 0, "latency": [],
    }
    recent: deque = deque(maxlen=12)

    progress = Progress(
        TextColumn("[bold cyan]Streaming sessions"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        expand=True,
    )
    task = progress.add_task("stream", total=len(df))

    with Live(build_layout(progress, stats, recent, model_version),
              console=console, refresh_per_second=12, screen=True) as live:
        try:
            for row in df.to_dict("records"):
                result, latency = predict(row, args.api, args.via)
                progress.advance(task)
                if result is None:
                    stats["api_failures"] += 1
                    continue

                model_version = result.get("model_version", model_version)
                atype = row.get("agent_type") or "unknown"
                is_anom = bool(result["is_anomaly"])
                actually_bad = atype in ANOMALY_TYPES

                stats["seen"][atype] += 1
                stats["flagged"][atype] += int(is_anom)
                stats["latency"].append(latency)
                stats["tp"] += int(is_anom and actually_bad)
                stats["fn"] += int(not is_anom and actually_bad)
                stats["fp"] += int(is_anom and not actually_bad)

                recent.appendleft({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "session_id": row["session_id"],
                    "agent_type": atype,
                    "score": float(result["anomaly_score"]),
                    "is_anomaly": is_anom,
                })
                live.update(build_layout(progress, stats, recent, model_version))
                time.sleep(args.speed)
            time.sleep(2)
        except KeyboardInterrupt:
            pass

    console.print(summary_panel(stats))


if __name__ == "__main__":
    main()
