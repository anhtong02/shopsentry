"""
Pipeline orchestrator. Runs every pseudo-labeler, trains XGBoost on each,
evaluates against true labels, logs everything to MLflow.

Usage:
    python -m models.run_pipeline
    python -m models.run_pipeline --labelers autoencoder heuristic
    python -m models.run_pipeline --skip-oracle
"""
import argparse
import mlflow
import pandas as pd
from sklearn.metrics import classification_report
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

LABELER_REGISTRY: dict[str, type[BaseLabeler]] = {
    "isoforest": IsoForestLabeler,
    "autoencoder": AutoencoderLabeler,
    "autoencoder_contaminated": lambda: AutoencoderLabeler(train_on_normal_only=False),

    "heuristic": HeuristicLabeler,
}


def evaluate_labeler_quality(labeled_df: pd.DataFrame) -> dict:
    """How good are the pseudo-labels themselves vs ground truth?"""
    report = classification_report(
        labeled_df["label"], labeled_df["pseudo_label"], output_dict=True
    )
    return {
        "labeler_precision": report["1"]["precision"],
        "labeler_recall": report["1"]["recall"],
        "labeler_f1": report["1"]["f1-score"],
    }


def run_one(name: str, labeler: BaseLabeler, df: pd.DataFrame, feature_cols: list[str]) -> None:
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
    
        # Log labeler artifacts (autoencoder + scaler)
        if hasattr(labeler, "_fitted_model"):
            mlflow.tensorflow.log_model(labeler._fitted_model, artifact_path="autoencoder")
            mlflow.sklearn.log_model(labeler._fitted_scaler, artifact_path="scaler")
            mlflow.log_param("anomaly_threshold", labeler._threshold)
    
        # Log XGBoost model
        if clf.model is not None:
            mlflow.xgboost.log_model(clf.model, artifact_path="xgboost")
    
        for k in ("labeler_precision", "labeler_recall", "labeler_f1"):
            mlflow.log_metric(k, label_quality[k])
        for k in ("precision_anomaly", "recall_anomaly", "f1_anomaly",
                "bot_recall", "fraud_recall"):
            mlflow.log_metric(k, metrics[k])


def run_oracle(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Train XGBoost on TRUE labels — upper bound on what's achievable."""
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
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("anomaly_detection_pipeline")

    df, feature_cols = load_training_data()
    print(f"Loaded {len(df)} sessions, {len(feature_cols)} features")
    print(f"True anomaly rate: {df['label'].mean():.2%}")

    # Run individual labelers
    fitted_labelers = []
    for name in args.labelers:
        if name not in LABELER_REGISTRY:
            print(f"Unknown labeler: {name}, skipping")
            continue
        labeler = LABELER_REGISTRY[name]()
        run_one(name, labeler, df, feature_cols)
        fitted_labelers.append(labeler)

    # Ensemble of all selected labelers
    if not args.skip_ensemble and len(fitted_labelers) >= 2:
        ensemble = EnsembleLabeler(fitted_labelers, min_votes=2)
        run_one(ensemble.name, ensemble, df, feature_cols)

    # Oracle baseline (upper bound)
    if not args.skip_oracle:
        run_oracle(df, feature_cols)

    print("\n✅ Pipeline complete. Check MLflow UI at http://localhost:5000")


if __name__ == "__main__":
    main()