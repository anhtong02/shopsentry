import pandas as pd

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

def load_training_data():
    df = pd.read_parquet("feature_repo/feature_repo/data/offline_features")
    df["label"] = df["agent_type"].apply(lambda x: 1 if x in ["bot", "fraud"] else 0)
    df[feature_cols] = df[feature_cols].fillna(0)

    return df, feature_cols