import pandas as pd
import glob

feature_cols = [
    "events_per_minute",
    "unique_pages_visited",
    "avg_time_between_events",
    "cart_to_purchase_ratio",
    "session_duration_seconds",
    "event_type_diversity",
    "has_payment",
    "signup_to_purchase_speed",
    "page_revisit_ratio"
]

def load_training_data() -> tuple[pd.DataFrame, list[str]]:
    files = sorted(glob.glob("feature_repo/feature_repo/data/offline_features/*.parquet"))
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Skipping {f}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    df["label"] = df["agent_type"].apply(lambda x: 1 if x in ["bot", "fraud"] else 0)
    df[feature_cols] = df[feature_cols].fillna(0)

    return df, feature_cols