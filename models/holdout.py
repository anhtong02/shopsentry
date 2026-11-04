"""Held-out sessions: the test split the served XGBoost never trained on.

The training pipeline (XGBoostClassifier.fit_evaluate) splits sessions 70/30,
stratified on the true label, with random_state=1. The 30% test split is
saved here so the demo can replay only sessions the model has not seen.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

HOLDOUT_PATH = Path("models/holdout_session_ids.csv")
TEST_SIZE = 0.3     # must match XGBoostClassifier(test_size=...)
RANDOM_STATE = 1    # must match XGBoostClassifier(random_state=...)


def save_holdout_ids(test_df: pd.DataFrame) -> Path:
    """Write the test split's session_ids so the demo can replay them later."""
    HOLDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_df[["session_id"]].to_csv(HOLDOUT_PATH, index=False)
    return HOLDOUT_PATH


def split_train_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The exact 70/30 split XGBoostClassifier.fit_evaluate uses (same size, stratify, seed).

    Labelers fit on the train part only, so nothing about the held-out sessions
    (e.g. the autoencoder's verified-normal sample) leaks into training.
    """
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE,
    )
    return train_df, test_df


def holdout_sessions(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return only held-out sessions, and how they were found.

    Uses the saved id list when it exists. Otherwise rebuilds the same split the
    training pipeline used, which is exact as long as the offline feature data
    hasn't changed since the model was trained.
    """
    if HOLDOUT_PATH.exists():
        ids = set(pd.read_csv(HOLDOUT_PATH)["session_id"])
        return df[df["session_id"].isin(ids)].reset_index(drop=True), "saved id list"

    _, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE,
    )
    return test_df.reset_index(drop=True), "rebuilt split (assumes data unchanged since training)"