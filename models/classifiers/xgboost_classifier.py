"""XGBoost classifier — trains on pseudo-labels, evaluates against true labels."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from models.pseudo_labelers.autoencoder_labeler import AutoencoderLabeler


class XGBoostClassifier:
    """
    Wraps XGBoost training + evaluation. Two modes:
    - fit_evaluate(): single train/test split (original path)
    - fit_evaluate_cv(): nested 5-fold CV (each fold trains its own autoencoder + xgboost)

    Both paths scale features with StandardScaler so training matches inference.
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
        self.scaler: StandardScaler | None = None
        self.feature_importances_: pd.Series | None = None

    # ---------- Single split path ----------
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

        X_tr, X_te, yp_tr, yp_te, yt_tr, yt_te = train_test_split(
            X, y_pseudo, y_true,
            test_size=self.test_size, stratify=y_true,
            random_state=self.random_state,
        )

        # Scale on training fold, apply to test
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        neg = (yp_tr == 0).sum()
        pos = (yp_tr == 1).sum()
        scale = neg / pos if pos else 1.0

        self.model = xgb.XGBClassifier(**self.params, scale_pos_weight=scale)
        self.model.fit(X_tr_scaled, yp_tr)
        self.scaler = scaler

        preds = self.model.predict(X_te_scaled)
        report = classification_report(yt_te, preds, output_dict=True)

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

    # ---------- Nested CV path ----------
    def fit_evaluate_cv(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        n_folds: int = 5,
        true_label_col: str = "label",
        agent_type_col: str = "agent_type",
    ) -> dict:
        """
        Nested 5-fold CV. Each fold:
        - Split data, stratified on agent_type
        - Train fresh autoencoder on train fold's normals
        - Generate pseudo-labels on train fold
        - Scale train fold, train XGBoost
        - Apply same scaler to test fold, evaluate against true labels
        Returns mean + std of metrics across all folds.
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        all_metrics = []
        fold_idx = 0

        for train_idx, test_idx in skf.split(df, df[agent_type_col]):
            fold_idx += 1
            print(f"  Fold {fold_idx}/{n_folds}...")

            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            # Fresh autoencoder per fold
            ae = AutoencoderLabeler(random_state=self.random_state)
            ae.fit(train_df, feature_cols)
            labeled_train = ae.predict(train_df)

            yp_tr = labeled_train["pseudo_label"]
            X_tr = labeled_train[feature_cols]

            # Scale train fold, apply same scaler to test fold
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)

            neg = (yp_tr == 0).sum()
            pos = (yp_tr == 1).sum()
            scale = neg / pos if pos else 1.0

            model = xgb.XGBClassifier(**self.params, scale_pos_weight=scale)
            model.fit(X_tr_scaled, yp_tr)

            X_te = test_df[feature_cols]
            yt_te = test_df[true_label_col]
            X_te_scaled = scaler.transform(X_te)
            preds = model.predict(X_te_scaled)
            report = classification_report(yt_te, preds, output_dict=True)

            test_df_copy = test_df.copy()
            test_df_copy["prediction"] = preds
            bot_recall = self._class_recall(test_df_copy, "bot")
            fraud_recall = self._class_recall(test_df_copy, "fraud")

            all_metrics.append({
                "fold": fold_idx,
                "precision_anomaly": report["1"]["precision"],
                "recall_anomaly": report["1"]["recall"],
                "f1_anomaly": report["1"]["f1-score"],
                "bot_recall": bot_recall,
                "fraud_recall": fraud_recall,
            })

            print(f"    F1={report['1']['f1-score']:.3f}, bot={bot_recall:.2%}, fraud={fraud_recall:.2%}")

        keys = ["precision_anomaly", "recall_anomaly", "f1_anomaly", "bot_recall", "fraud_recall"]
        summary = {}
        for k in keys:
            values = [m[k] for m in all_metrics]
            summary[f"{k}_mean"] = float(np.mean(values))
            summary[f"{k}_std"] = float(np.std(values))

        summary["per_fold"] = all_metrics
        summary["n_folds"] = n_folds
        return summary

    @staticmethod
    def _class_recall(test_df: pd.DataFrame, agent_type: str) -> float:
        sub = test_df[test_df["agent_type"] == agent_type]
        if len(sub) == 0:
            return 0.0
        return float((sub["prediction"] == 1).sum() / len(sub))