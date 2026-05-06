"""XGBoost classifier — trains on pseudo-labels, evaluates against true labels."""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class XGBoostClassifier:
    """
    Wraps XGBoost training + evaluation. Takes pseudo-labels for training,
    evaluates against true labels held out of the training data.
    """
    def __init__(
        self,
        n_estimators: int = 70,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state: int = 1,
        test_size: float = 0.3,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )
        self.random_state = random_state
        self.test_size = test_size
        self.model: xgb.XGBClassifier | None = None
        self.feature_importances_: pd.Series | None = None

    def fit_evaluate(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        pseudo_label_col: str = "pseudo_label",
        true_label_col: str = "label",
    ) -> dict:
        X = df[feature_cols]
        y_pseudo = df[pseudo_label_col]
        y_true = df[true_label_col]

        # Stratify on TRUE labels so test set reflects real class distribution
        X_tr, X_te, yp_tr, yp_te, yt_tr, yt_te = train_test_split(
            X, y_pseudo, y_true,
            test_size=self.test_size, stratify=y_true,
            random_state=self.random_state,
        )

        # scale_pos_weight from PSEUDO labels (production-realistic)
        neg = (yp_tr == 0).sum()
        pos = (yp_tr == 1).sum()
        scale = neg / pos if pos else 1.0

        self.model = xgb.XGBClassifier(**self.params, scale_pos_weight=scale)
        self.model.fit(X_tr, yp_tr)

        preds = self.model.predict(X_te)
        report = classification_report(yt_te, preds, output_dict=True)

        # Per-agent breakdown on test set
        test_df = df.loc[X_te.index].copy()
        test_df["prediction"] = preds
        bot_recall = self._class_recall(test_df, "bot")
        fraud_recall = self._class_recall(test_df, "fraud")

        self.feature_importances_ = pd.Series(
            self.model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)

        return {
            "precision_anomaly": report["1"]["precision"],
            "recall_anomaly": report["1"]["recall"],
            "f1_anomaly": report["1"]["f1-score"],
            "bot_recall": bot_recall,
            "fraud_recall": fraud_recall,
            "scale_pos_weight": scale,
            "n_pseudo_pos": int(pos),
            "n_pseudo_neg": int(neg),
            "classification_report": classification_report(yt_te, preds),
            "test_df": test_df,
        }

    @staticmethod
    def _class_recall(test_df: pd.DataFrame, agent_type: str) -> float:
        sub = test_df[test_df["agent_type"] == agent_type]
        if len(sub) == 0:
            return 0.0
        return float((sub["prediction"] == 1).sum() / len(sub))