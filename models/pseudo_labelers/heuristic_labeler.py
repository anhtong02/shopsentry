"""Heuristic pseudo-labeler. Domain rules derived from EDA findings."""
import pandas as pd

from .base import BaseLabeler


class HeuristicLabeler(BaseLabeler):
    """
    Rules derived from looking at the data:
      - bots: events_per_minute > 30 (normal users averaged ~3-5/min)
      - fraud: session_duration < 10s AND has_payment == 1
                (no human checks out in under 10 seconds)
    These rules act as a stand-in for SME-defined heuristics.
    """
    name = "heuristic"

    def __init__(self, bot_epm_threshold: float = 30.0, fraud_duration_threshold: float = 10.0):
        self.bot_epm = bot_epm_threshold
        self.fraud_duration = fraud_duration_threshold

    def fit_predict(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        is_bot_like = df["events_per_minute"] > self.bot_epm
        is_fraud_like = (
            (df["session_duration_seconds"] < self.fraud_duration)
            & (df["has_payment"] == 1)
        )

        out = df.copy()
        out["pseudo_label"] = (is_bot_like | is_fraud_like).astype(int)
        # Score: rough confidence based on how strongly rules fire
        # (higher epm above threshold OR more extreme short duration)
        epm_score = (df["events_per_minute"] / self.bot_epm).clip(upper=10)
        dur_score = ((self.fraud_duration / df["session_duration_seconds"].clip(lower=0.1))
                     * df["has_payment"]).clip(upper=10)
        out["score"] = epm_score.combine(dur_score, max)
        return out