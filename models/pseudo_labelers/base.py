"""
Base class for pseudo-labelers.
 
Every labeler must implement fit_predict() that takes a DataFrame
and returns a pseudo_label series (0=normal, 1=anomaly) plus an
optional score series for ensemble voting and analysis.
"""

from abc import ABC, abstractmethod
import pandas as pd
 
 
class BaseLabeler(ABC):
    name: str = "base"
 
    @abstractmethod
    def fit_predict(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
            pseudo_label : int (0 or 1)
            score        : float (higher = more anomalous)
        Indexed the same as input df.
        """
        pass
 
    def save(self, df: pd.DataFrame, path: str | None = None) -> str:
        """Save pseudo-labels to parquet for downstream classifiers."""
        path = path or f"data/pseudo_labels_{self.name}.parquet"
        df.to_parquet(path, index=False)
        print(f"[{self.name}] saved {df['pseudo_label'].sum()} flagged / {len(df)} total → {path}")
        return path
 