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

        # Fit state (set by fit())
        self._fitted_model = None
        self._fitted_scaler = None
        self._threshold = None
        self._feature_cols = None

    # ---------- NEW: separate fit and predict ----------
    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        """Train autoencoder on (normal-only) data. Sets internal state."""
        tf.keras.utils.set_random_seed(self.random_state)

        if self.train_on_normal_only:
            train_pool = df[df["label"] == 0]
        else:
            train_pool = df

        train_df, val_df = train_test_split(
            train_pool, test_size=0.2, random_state=self.random_state
        )

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df[feature_cols])
        val_scaled = scaler.transform(val_df[feature_cols])

        n_features = len(feature_cols)
        model = keras.Sequential([
            Dense(6, activation="relu", input_shape=(n_features,)), #encoder layer 1
            Dense(self.bottleneck_dim, activation="relu"), #bottleneck middle layer
            Dense(6, activation="relu"), #decoder 
            Dense(n_features, activation="linear"), #rescontruction
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            train_scaled, train_scaled,
            validation_data=(val_scaled, val_scaled),
            epochs=self.epochs, batch_size=self.batch_size, verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        )

        # Threshold from validation set
        val_recon = model.predict(val_scaled, verbose=0)
        val_errors = np.mean((val_scaled - val_recon) ** 2, axis=1)
        threshold = np.percentile(val_errors, self.threshold_percentile)

        self._fitted_model = model
        self._fitted_scaler = scaler
        self._threshold = threshold
        self._feature_cols = feature_cols

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply trained autoencoder to any data. Returns df with score + pseudo_label."""
        if self._fitted_model is None or self._fitted_scaler is None:
            raise RuntimeError("Must call fit() before predict()")

        scaled = self._fitted_scaler.transform(df[self._feature_cols])
        recon = self._fitted_model.predict(scaled, verbose=0)
        errors = np.mean((scaled - recon) ** 2, axis=1)

        out = df.copy()
        out["score"] = errors
        out["pseudo_label"] = (errors > self._threshold).astype(int)
        return out

    # ---------- KEEP: convenience wrapper for old non-CV path ----------
    def fit_predict(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Train on full df, then score full df. Original behavior."""
        self.fit(df, feature_cols)
        return self.predict(df)