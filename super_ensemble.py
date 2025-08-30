# src/project/super_ensemble.py
# SuperEnsemble: uczeń–student (stacking) nad bazowymi modelami ML/NN.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from .predictor import Predictor
from .model_registry import requires_tft_input

LOGGER = logging.getLogger(__name__)


class SuperEnsemble:
    """
    Meta-model (student) uczący się na predykcjach modeli bazowych (uczniów).

    - base_predictors: lista bazowych modeli (Predictor). Jeśli None, trzeba podać później.
    - student_type: "ridge" (domyślne, regresja liniowa z regularyzacją) lub "mlp" (Keras MLP).
    """

    def __init__(self, base_predictors: list[Predictor] | None = None, student_type: str = "ridge"):
        self.base_predictors: list[Predictor] = base_predictors or []
        self.student_type = student_type
        self.student: Any | None = None
        self.horizon: int | None = None

    # ————————————————————————————————
    def _stack_preds(self, X_pack) -> np.ndarray:
        """
        Zwraca macierz predykcji bazowych modeli: shape=(n, n_models * horizon).
        """
        mats = []
        for pr in self.base_predictors:
            y_pred = pr.predict(X_pack, inverse_scale=False, verbose=0)
            mats.append(y_pred)  # (n, h)
        if not mats:
            raise ValueError("Brak bazowych predictorów w SuperEnsemble")
        stacked = np.concatenate(mats, axis=1)  # (n, n_models*h)
        if self.horizon is None:
            self.horizon = mats[0].shape[1]
        return stacked

    # ————————————————————————————————
    def fit(self, X_pack, Y: np.ndarray, *, val_fraction: float = 0.2, epochs: int = 10, batch_size: int = 64) -> None:
        """
        Uczy studenta na predykcjach bazowych modeli (stacking).
        - student_type="ridge": Ridge regression na predykcjach → Y.
        - student_type="mlp": prosty MLP w Kerasie, mapujący stacked preds → Y.
        """
        if not isinstance(Y, np.ndarray) or Y.ndim != 2:
            raise ValueError("Y musi być tablicą 2D (n, horizon)")

        X_stacked = self._stack_preds(X_pack)  # (n, n_models*h)
        n, total_h = X_stacked.shape
        h = self.horizon or Y.shape[1]

        if self.student_type == "ridge":
            self.student = Ridge(alpha=1.0)
            self.student.fit(X_stacked, Y)
            LOGGER.info("SuperEnsemble (ridge) wytrenowany na %d próbkach", n)

        elif self.student_type == "mlp":
            try:
                import tensorflow as tf  # noqa
                from tensorflow import keras
                from tensorflow.keras import layers
            except ImportError:
                raise RuntimeError("student_type='mlp' wymaga tensorflow.keras")

            inp = keras.Input(shape=(total_h,), name="stacked_inputs")
            x = layers.Dense(64, activation="relu")(inp)
            x = layers.Dropout(0.2)(x)
            out = layers.Dense(h)(x)
            mdl = keras.Model(inp, out)
            mdl.compile(optimizer="adam", loss="mse")
            mdl.fit(X_stacked, Y, validation_split=val_fraction, epochs=epochs, batch_size=batch_size, verbose=0)
            self.student = mdl
            LOGGER.info("SuperEnsemble (mlp) wytrenowany na %d próbkach", n)

        else:
            raise ValueError(f"Nieznany student_type={self.student_type}")

    # ————————————————————————————————
    def predict(self, X_pack) -> np.ndarray:
        """
        Predykcja przez studenta. Zwraca macierz (n, horizon).
        """
        if self.student is None:
            raise RuntimeError("SuperEnsemble nie został jeszcze wytrenowany (student=None)")
        X_stacked = self._stack_preds(X_pack)

        if isinstance(self.student, Ridge):
            return self.student.predict(X_stacked)
        else:
            # zakładamy Keras MLP
            y_pred = self.student.predict(X_stacked, verbose=0)
            return np.asarray(y_pred, dtype="float32")
