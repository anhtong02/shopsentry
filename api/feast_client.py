"""Feast online store client. Fetches session features from Redis."""
import logging
from pathlib import Path

from feast import FeatureStore

logger = logging.getLogger(__name__)

FEATURE_REFS = [
    "session_features:events_per_minute",
    "session_features:unique_pages_visited",
    "session_features:avg_time_between_events",
    "session_features:cart_to_purchase_ratio",
    "session_features:session_duration_seconds",
    "session_features:event_type_diversity",
    "session_features:has_payment",
    "session_features:signup_to_purchase_speed",
    "session_features:page_revisit_ratio",
]

# Default values when Feast/Redis is down — biased toward "looks normal"
# so we don't generate false-positive anomaly alerts during outages.
DEFAULT_FEATURES = {
    "events_per_minute": 5.0,
    "unique_pages_visited": 3.0,
    "avg_time_between_events": 15.0,
    "cart_to_purchase_ratio": 0.0,
    "session_duration_seconds": 60.0,
    "event_type_diversity": 2,
    "has_payment": 0,
    "signup_to_purchase_speed": 0.0,
    "page_revisit_ratio": 0.0,
}


class FeastClient:
    def __init__(self, repo_path: str = "feature_repo/feature_repo"):
        self.repo_path = Path(repo_path)
        self._store: FeatureStore | None = None
        try:
            self._store = FeatureStore(repo_path=str(self.repo_path))
            logger.info(f"Feast store initialized from {self.repo_path}")
        except Exception as e:
            logger.warning(f"Feast init failed: {e}. Will use defaults on every request.")

    def get_features(self, session_id: str) -> tuple[dict, bool]:
        """
        Returns (features_dict, used_defaults).
        used_defaults=True means Feast/Redis was unavailable — caller should log & possibly alert.
        """
        if self._store is None:
            return dict(DEFAULT_FEATURES), True

        try:
            response = self._store.get_online_features(
                features=FEATURE_REFS,
                entity_rows=[{"session_id": session_id}],
            ).to_dict()

            # response is {"feature_name": [val_for_row_0], ...}
            features = {}
            for full_name in FEATURE_REFS:
                short_name = full_name.split(":")[1]
                val = response.get(short_name, [None])[0]
                if val is None:
                    # Feature missing for this session_id — fall back
                    logger.warning(f"Missing {short_name} for session {session_id}, using default")
                    val = DEFAULT_FEATURES[short_name]
                features[short_name] = val
            return features, False

        except Exception as e:
            logger.error(f"Feast lookup failed for {session_id}: {e}. Using defaults.")
            return dict(DEFAULT_FEATURES), True