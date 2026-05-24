import logging
from dataclasses import dataclass
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)

FEATURE_ORDER = [
    "events_per_minute",
    "unique_pages_visited",
    "avg_time_between_events",
    "cart_to_purchase_ratio",
    "session_duration_seconds",
    "event_type_diversity",
    "has_payment",
    "signup_to_purchase_speed",
    "page_revisit_ratio",
]

@dataclass
class LoadedModels:
    scaler: Any
    classifier: Any
    classifier_version: str

    def predict(self, features: np.ndarray) -> tuple[float, bool]:
        """features: 2D array shape (1, n_features). Returns (anomaly_score, is_anomaly)."""
        scaled = self.scaler.transform(features)
        # XGBClassifier.predict_proba returns shape (n, 2): [P(normal), P(anomaly)]
        proba = self.classifier.predict_proba(scaled)
        anomaly_score = float(proba[0, 1])
        return anomaly_score, anomaly_score > 0.5

def load_from_registry(
        classifier_name: str = "shopsentry_classifier",
        scaler_name: str = "shopsentry_scaler",
        version: str = "6",
        tracking_uri: str = "http://localhost:5000",
) -> LoadedModels:
        
        mlflow.set_tracking_uri(tracking_uri)
        classifier_uri = f"models:/{classifier_name}/{version}"
        scaler_uri = f"models:/{scaler_name}/{version}"
        logger.info(f"Loading classifier from {classifier_uri}")
        classifier = mlflow.xgboost.load_model(classifier_uri)
 
        logger.info(f"Loading scaler from {scaler_uri}")
        scaler = mlflow.sklearn.load_model(scaler_uri)
 
        logger.info("Models loaded successfully")
        return LoadedModels(
        scaler=scaler,
        classifier=classifier,
        classifier_version=f"{classifier_name}:v{version}",
        )
