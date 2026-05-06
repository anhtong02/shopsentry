"""Ensemble pseudo-labeler. Combines multiple labelers via majority vote."""
import pandas as pd

from .base import BaseLabeler


class EnsembleLabeler(BaseLabeler):
    """
    Takes a list of (already-fitted) labeler outputs and combines them.
    A row is flagged anomaly if at least `min_votes` labelers agreed.
    """
    name = "ensemble"

    def __init__(self, labelers: list[BaseLabeler], min_votes: int = 2):
        self.labelers = labelers
        self.min_votes = min_votes
        self.name = f"ensemble_{min_votes}of{len(labelers)}"

    def fit_predict(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        votes = pd.DataFrame(index=df.index)
        for labeler in self.labelers:
            result = labeler.fit_predict(df, feature_cols)
            votes[labeler.name] = result["pseudo_label"].values

        out = df.copy()
        out["score"] = votes.sum(axis=1)
        out["pseudo_label"] = (out["score"] >= self.min_votes).astype(int)
        # Attach individual votes for diagnostics
        for col in votes.columns:
            out[f"vote_{col}"] = votes[col].values
        return out