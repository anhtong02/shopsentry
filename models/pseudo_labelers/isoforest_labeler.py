"""Isolation Forest pseudo-labeler."""
import pandas as pd
from sklearn.ensemble import IsolationForest
 
from .base import BaseLabeler
 
 
class IsoForestLabeler(BaseLabeler):
    name = "isoforest"
 
    def __init__(self, contamination: float = 0.15, random_state: int = 53):
        self.contamination = contamination
        self.random_state = random_state
 
    def fit_predict(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        X = df[feature_cols]
        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,
        )
        model.fit(X)
 
        # decision_function: higher = more normal. Negate so higher = more anomalous.
        scores = -model.decision_function(X)
        preds = (model.predict(X) == -1).astype(int)
 
        out = df.copy()
        out["pseudo_label"] = preds
        out["score"] = scores
        return out