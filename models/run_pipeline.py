"""
Pipeline orchestrator. Runs every pseudo-labeler, trains XGBoost on each,
evaluates against true labels, logs everything to MLflow.

Usage:
    python -m models.run_pipeline
    python -m models.run_pipeline --labelers autoencoder heuristic
    python -m models.run_pipeline --skip-oracle
    python -m models.run_pipeline --cv               # use 5-fold CV (autoencoder only)
"""
import argparse
import mlflow
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from models.data_loader import load_training_data
from models.pseudo_labelers.base import BaseLabeler
from models.pseudo_labelers.isoforest_labeler import IsoForestLabeler
from models.pseudo_labelers.autoencoder_labeler import AutoencoderLabeler
from models.pseudo_labelers.heuristic_labeler import HeuristicLabeler
from models.pseudo_labelers.ensemble import EnsembleLabeler
from models.classifiers.xgboost_classifier import XGBoostClassifier
import mlflow.tensorflow
import mlflow.sklearn
import mlflow.xgboost
from typing import Callable
from mlflow.models.signature import infer_signature


LABELER_REGISTRY: dict[str, Callable[[], BaseLabeler]] = {
    "isoforest": IsoForestLabeler,
    "autoencoder": AutoencoderLabeler,
    "autoencoder_contaminated": lambda: AutoencoderLabeler(train_on_normal_only=False),
    "heuristic": HeuristicLabeler,
}


def evaluate_labeler_quality(labeled_df: pd.DataFrame) -> dict:
    report = classification_report(
        labeled_df["label"], labeled_df["pseudo_label"], output_dict=True
    )
    return {
        "labeler_precision": report["1"]["precision"],
        "labeler_recall": report["1"]["recall"],
        "labeler_f1": report["1"]["f1-score"],
    }


def run_one(name: str, labeler: BaseLabeler, df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Original single-split path."""
    print(f"\n{'='*60}\nLABELER: {name}\n{'='*60}")

    labeled_df = labeler.fit_predict(df, feature_cols)
    labeler.save(labeled_df)
    label_quality = evaluate_labeler_quality(labeled_df)
    print(f"Pseudo-label quality vs truth: {label_quality}")

    clf = XGBoostClassifier()
    metrics = clf.fit_evaluate(labeled_df, feature_cols)

    print(f"\nXGBoost trained on {name} pseudo-labels:")
    print(metrics["classification_report"])
    print(f"Bot recall: {metrics['bot_recall']:.2%}")
    print(f"Fraud recall: {metrics['fraud_recall']:.2%}")

    with mlflow.start_run(run_name=f"xgb_on_{name}"):
        mlflow.log_param("labeler", name)
        mlflow.log_param("classifier", "xgboost")
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("scale_pos_weight", metrics["scale_pos_weight"])
        mlflow.log_param("eval_method", "single_split")

        if hasattr(labeler, "_fitted_model") and hasattr(labeler, "_fitted_scaler"):
            mlflow.tensorflow.log_model(labeler._fitted_model, name="autoencoder")
            mlflow.log_param("anomaly_threshold", labeler._threshold)

        # Save the XGBoost's scaler (fit on training fold) — this is what API needs
        if clf.scaler is not None:
            mlflow.sklearn.log_model(clf.scaler, name="scaler")

        if clf.model is not None:
            sample_input = np.zeros((1, len(feature_cols)))
            sample_output = clf.model.predict(sample_input)
            sig = infer_signature(sample_input, sample_output)
            mlflow.xgboost.log_model(clf.model, name="xgboost", signature=sig)

        for k in ("labeler_precision", "labeler_recall", "labeler_f1"):
            mlflow.log_metric(k, label_quality[k])
        for k in ("precision_anomaly", "recall_anomaly", "f1_anomaly",
                  "bot_recall", "fraud_recall"):
            mlflow.log_metric(k, metrics[k])


def run_autoencoder_cv(df: pd.DataFrame, feature_cols: list[str], n_folds: int = 5) -> None:
    """
    Nested 5-fold CV path. After CV summary, trains a FINAL model on all data
    (with proper scaling) for MLflow registration.
    """
    print(f"\n{'='*60}\nAUTOENCODER + XGBOOST — {n_folds}-FOLD NESTED CV\n{'='*60}")

    clf = XGBoostClassifier()
    cv_results = clf.fit_evaluate_cv(df, feature_cols, n_folds=n_folds)

    print(f"\nCV Summary across {n_folds} folds:")
    print(f"  F1 anomaly:    {cv_results['f1_anomaly_mean']:.3f} ± {cv_results['f1_anomaly_std']:.3f}")
    print(f"  Bot recall:    {cv_results['bot_recall_mean']:.2%} ± {cv_results['bot_recall_std']:.2%}")
    print(f"  Fraud recall:  {cv_results['fraud_recall_mean']:.2%} ± {cv_results['fraud_recall_std']:.2%}")

    # ---------- Final model on all data (for MLflow registration) ----------
    print(f"\nTraining final model on all data (for registration)...")

    final_ae = AutoencoderLabeler()
    final_ae.fit(df, feature_cols)
    labeled_df = final_ae.predict(df)

    # Scaler that XGBoost will use (separate from autoencoder's internal scaler)
    final_scaler = StandardScaler()
    X_full_scaled = final_scaler.fit_transform(labeled_df[feature_cols])

    yp = labeled_df["pseudo_label"]
    neg = (yp == 0).sum()
    pos = (yp == 1).sum()
    scale = neg / pos if pos else 1.0

    final_xgb = xgb.XGBClassifier(
        n_estimators=70, max_depth=3, learning_rate=0.1,
        scale_pos_weight=scale,
    )
    final_xgb.fit(X_full_scaled, yp)

    # Sanity check final model on each agent type
    print(f"\nFinal model performance on training data:")
    for atype in ["normal", "churning", "bot", "fraud"]:
        sub = labeled_df[labeled_df["agent_type"] == atype]
        if len(sub) == 0:
            continue
        sub_scaled = final_scaler.transform(sub[feature_cols])
        preds = final_xgb.predict(sub_scaled)
        if atype in ("bot", "fraud"):
            print(f"  {atype:10s}: {preds.sum()}/{len(sub)} caught = {preds.mean()*100:.1f}% recall")
        else:
            print(f"  {atype:10s}: {preds.sum()}/{len(sub)} flagged = {preds.mean()*100:.1f}% FP")

    with mlflow.start_run(run_name="xgb_on_autoencoder_cv"):
        mlflow.log_param("labeler", "autoencoder")
        mlflow.log_param("classifier", "xgboost")
        mlflow.log_param("eval_method", f"{n_folds}_fold_nested_cv")
        mlflow.log_param("n_features", len(feature_cols))

        for k in ("f1_anomaly", "bot_recall", "fraud_recall", "precision_anomaly", "recall_anomaly"):
            mlflow.log_metric(f"cv_{k}_mean", cv_results[f"{k}_mean"])
            mlflow.log_metric(f"cv_{k}_std", cv_results[f"{k}_std"])

        for fold_metrics in cv_results["per_fold"]:
            fold_idx = fold_metrics["fold"]
            for k in ("f1_anomaly", "bot_recall", "fraud_recall"):
                mlflow.log_metric(f"fold_{fold_idx}_{k}", fold_metrics[k])

        # Register artifacts: autoencoder (for drift later), XGBoost's scaler, XGBoost
        mlflow.tensorflow.log_model(final_ae._fitted_model, name="autoencoder")
        mlflow.log_param("anomaly_threshold", final_ae._threshold)

        mlflow.sklearn.log_model(final_scaler, name="scaler")

        sample_input = np.zeros((1, len(feature_cols)))
        sample_output = final_xgb.predict(sample_input)
        sig = infer_signature(sample_input, sample_output)
        mlflow.xgboost.log_model(final_xgb, name="xgboost", signature=sig)


def run_oracle(df: pd.DataFrame, feature_cols: list[str]) -> None:
    print(f"\n{'='*60}\nORACLE (true labels) — upper bound\n{'='*60}")

    oracle_df = df.copy()
    oracle_df["pseudo_label"] = oracle_df["label"]

    clf = XGBoostClassifier()
    metrics = clf.fit_evaluate(oracle_df, feature_cols)

    print(metrics["classification_report"])
    print(f"Bot recall: {metrics['bot_recall']:.2%}")
    print(f"Fraud recall: {metrics['fraud_recall']:.2%}")

    with mlflow.start_run(run_name="xgb_oracle_true_labels"):
        mlflow.log_param("labeler", "oracle_true_labels")
        mlflow.log_param("classifier", "xgboost")
        for k in ("precision_anomaly", "recall_anomaly", "f1_anomaly",
                  "bot_recall", "fraud_recall"):
            mlflow.log_metric(k, metrics[k])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labelers", nargs="+", default=list(LABELER_REGISTRY.keys()),
        help="Which labelers to run",
    )
    parser.add_argument("--skip-ensemble", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--cv", action="store_true",
                        help="Use 5-fold nested CV for autoencoder (skips other labelers)")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("anomaly_detection_pipeline")

    df, feature_cols = load_training_data()
    print(f"Loaded {len(df)} sessions, {len(feature_cols)} features")
    print(f"True anomaly rate: {df['label'].mean():.2%}")

    if args.cv:
        run_autoencoder_cv(df, feature_cols, n_folds=args.n_folds)
        print("\n✅ CV pipeline complete. Check MLflow UI at http://localhost:5000")
        return

    fitted_labelers = []
    for name in args.labelers:
        if name not in LABELER_REGISTRY:
            print(f"Unknown labeler: {name}, skipping")
            continue
        labeler = LABELER_REGISTRY[name]()
        run_one(name, labeler, df, feature_cols)
        fitted_labelers.append(labeler)

    if not args.skip_ensemble and len(fitted_labelers) >= 2:
        ensemble = EnsembleLabeler(fitted_labelers, min_votes=2)
        run_one(ensemble.name, ensemble, df, feature_cols)

    if not args.skip_oracle:
        run_oracle(df, feature_cols)

    print("\n✅ Pipeline complete. Check MLflow UI at http://localhost:5000")


if __name__ == "__main__":
    main()