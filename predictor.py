# src/project/predictor.py
# Jednolite wykonywanie predykcji: ładowanie bundli, przygotowanie wejść, ujednolicenie wyjść.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .model_registry import requires_tft_input, standardize_preds
from .model_saver_loader import load_model_bundle

LOGGER = logging.getLogger(__name__)


def _has_keras_predict(obj: Any) -> bool:
    """Próbuje rozpoznać, czy obiekt wygląda jak model Keras (ma .predict)."""
    return hasattr(obj, "predict") and callable(getattr(obj, "predict"))


def _inverse_scale_if_possible(y_scaled: np.ndarray, scalers: dict[str, Any] | None) -> np.ndarray:
    """
    Odskaluje predykcję (n, h) przy użyciu `scalers["target"]`, jeśli dostępny.
    """
    if scalers is None or not isinstance(scalers, dict):
        return y_scaled
    tscaler = scalers.get("target")
    if tscaler is None:
        return y_scaled
    try:
        # inverse_transform oczekuje 2D, zwraca 2D
        inv = tscaler.inverse_transform(y_scaled.astype(float))
        return inv.astype(y_scaled.dtype, copy=False)
    except Exception as e:
        LOGGER.warning("Nie udało się odskalić predykcji (pomijam): %s", e)
        return y_scaled


def prepare_model_input(
    model_type: str,
    X_pack: np.ndarray | dict[str, np.ndarray],
) -> np.ndarray | dict[str, np.ndarray]:
    """
    Zwraca wejście w formie wymaganej przez dany model:
      - TFT → dict {'observed_past': ... , 'known_future': ...}
      - standard (LSTM/GRU/CNN-LSTM/TRANSFORMER) → ndarray (n, seq, feat)
        (jeśli przekazano dict, użyje 'standard' lub 'observed_past').
    """
    if requires_tft_input(model_type):
        if not isinstance(X_pack, dict) or "observed_past" not in X_pack or "known_future" not in X_pack:
            raise ValueError("Dla TFT wymagane X_pack={'observed_past', 'known_future'}.")
        return {"observed_past": X_pack["observed_past"], "known_future": X_pack["known_future"]}
    # standard
    if isinstance(X_pack, dict):
        arr = X_pack.get("standard") or X_pack.get("observed_past")
        if arr is None:
            raise ValueError("Dla modeli standardowych oczekiwano ndarray lub dict z 'standard'/'observed_past'.")
        return arr
    return X_pack


class Predictor:
    """
    Wykonuje predykcje na wytrenowanym modelu.

    Atrybuty:
      - model: obiekt modelu (Keras lub sklearn z API .predict)
      - scalers: optional dict zawierający 'target' do odskalowania
      - params: dict parametrów zapisanych przy treningu (musi zawierać 'model_type')
      - model_type: str – typ modelu (np. 'LSTM', 'TFT')
      - horizon: int – liczba kroków prognozy (wnioskowana z Y podczas treningu lub params)
    """

    def __init__(self, model: Any, scalers: dict[str, Any] | None, params: dict[str, Any] | None):
        if params is None or "model_type" not in params:
            raise ValueError("Brak wymaganych parametrów modelu (params['model_type']).")
        self.model: Any = model
        self.scalers: dict[str, Any] | None = scalers
        self.params: dict[str, Any] = params
        self.model_type: str = str(params["model_type"])
        # `horizon` może być zapisany w params; jeśli nie, oszacujemy przy pierwszej predykcji
        self.horizon: int | None = int(params.get("forecast_horizon", params.get("horizon", 0))) or None

    # ------------- Fabryka --------------

    @classmethod
    def load(cls, load_dir: str | Path, base_filename: str | None = None) -> "Predictor":
        """
        Wczytuje (model, scalers, params) z katalogu. Jeśli base_filename=None → autodetekcja.
        """
        model, scalers, params = load_model_bundle(base_filename, load_dir)
        if model is None or params is None:
            raise FileNotFoundError(f"Nie udało się wczytać bundla z {load_dir!s} (base={base_filename!r}).")
        return cls(model=model, scalers=scalers, params=params)

    # ------------- Predykcja ------------

    def predict(
        self,
        X_pack: np.ndarray | dict[str, np.ndarray],
        *,
        inverse_scale: bool = True,
        batch_size: int = 1024,
        verbose: int = 0,
    ) -> np.ndarray:
        """
        Zwraca predykcje o kształcie `(n, horizon)`.

        - Dobiera właściwy typ wejścia (ndarray vs. dict) na podstawie `self.model_type`.
        - Wykonuje `model.predict(...)` (Keras/sklearn-kompatybilny).
        - Ujednolica wynik do `(n, h)` (np. jeśli 1D → `[:, None]`).
        - Gdy `inverse_scale=True` i dostępny `scalers["target"]`, wynik jest odskalowany.
        """
        X_in = prepare_model_input(self.model_type, X_pack)

        # Keras/sklearn: preferuj .predict
        if not _has_keras_predict(self.model):
            raise TypeError("Załadowany model nie udostępnia metody .predict(...)")

        try:
            if isinstance(X_in, dict):
                # Keras przyjmuje dict wejść (pasujące nazwy Input layers)
                y_pred = self.model.predict(X_in, batch_size=batch_size, verbose=verbose)
            else:
                y_pred = self.model.predict(X_in, batch_size=batch_size, verbose=verbose)
        except Exception as e:
            LOGGER.exception("Błąd predykcji: %s", e)
            raise

        y2d = standardize_preds(y_pred)

        # zapamiętaj `horizon`, jeśli nie znany
        if self.horizon is None:
            self.horizon = int(y2d.shape[1])

        if inverse_scale:
            y2d = _inverse_scale_if_possible(y2d, self.scalers)

        return y2d
