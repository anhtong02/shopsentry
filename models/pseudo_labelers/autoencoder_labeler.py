"""Autoencoder pseudo-labeler. Trains on normal-only, flags high reconstruction error."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.layers import Dense
import tensorflow as tf
from .base import BaseLabeler


class AutoencoderLabeler(BaseLabeler):
    name = "autoencoder"

    def __init__(
        self,
        bottleneck_dim: int = 3,
        threshold_percentile: float = 95.0,
        epochs: int = 120,
        batch_size: int = 32,
        random_state: int = 1,
        train_on_normal_only: bool = True,  

    ):
        self.train_on_normal_only = train_on_normal_only
        self.name = "autoencoder" if train_on_normal_only else "autoencoder_contaminated"
        self.bottleneck_dim = bottleneck_dim
        self.threshold_percentile = threshold_percentile
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        tf.keras.utils.set_random_seed(random_state)

    def fit_predict(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        # CRITICAL: only train on rows we BELIEVE are normal.
        # In production with no labels, you'd assume the bulk of traffic is normal
        # and accept some contamination. Here we use the simulator's labels for
        # CLEAN training, but the pseudo_label OUTPUT is what XGBoost will use.
        if self.train_on_normal_only:
            train_pool = df[df["label"] == 0]
        else:
            train_pool = df

        train_df, val_df = train_test_split(
        train_pool, test_size=0.2, random_state=self.random_state)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df[feature_cols])
        val_scaled = scaler.transform(val_df[feature_cols])

        n_features = len(feature_cols)
        model = keras.Sequential([
            Dense(6, activation="relu", input_shape=(n_features,)),
            Dense(self.bottleneck_dim, activation="relu"),
            Dense(6, activation="relu"),
            Dense(n_features, activation="linear"),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            train_scaled, train_scaled,
            validation_data=(val_scaled, val_scaled),
            epochs=self.epochs, batch_size=self.batch_size, verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        )

        # Threshold from validation reconstruction errors (held out from training)
        val_recon = model.predict(val_scaled, verbose=0)
        val_errors = np.mean((val_scaled - val_recon) ** 2, axis=1)
        threshold = np.percentile(val_errors, self.threshold_percentile)

        # Score the FULL dataset
        all_scaled = scaler.transform(df[feature_cols])
        all_recon = model.predict(all_scaled, verbose=0)
        all_errors = np.mean((all_scaled - all_recon) ** 2, axis=1)

        out = df.copy()
        out["score"] = all_errors
        out["pseudo_label"] = (all_errors > threshold).astype(int)

        self._fitted_model = model
        self._fitted_scaler = scaler
        self._threshold = threshold

        return out