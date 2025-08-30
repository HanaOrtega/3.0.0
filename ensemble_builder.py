# src/project/ensemble_builder.py
# Budowanie prostych ensemble z wielu modeli (średnia, mediana, wagi).

from __future__ import annotations

import logging
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .predictor import Predictor
from .model_registry import standardize_preds

LOGGER = logging.getLogger(__name__)


def load_predictors(paths: list[str | Path]) -> list[Predictor]:
    """
    Ładuje kilka bundli modeli i zwraca listę obiektów Predictor.
    """
    preds: list[Predictor] = []
    for p in paths:
        try:
            pr = Predictor.load(Path(p).parent, Path(p).stem)
            preds.append(pr)
        except Exception as e:
            LOGGER.error("Nie udało się załadować bundla %s: %s", p, e)
            raise
    return preds


def build_ensemble(
    predictors: list[Predictor],
    X_pack: np.ndarray | dict[str, np.ndarray],
    *,
    method: str = "mean",
    weights: list[float] | None = None,
    inverse_scale: bool = True,
) -> np.ndarray:
    """
    Łączy predykcje wielu modeli w jedną.

    Args:
        predictors: lista obiektów Predictor (wcześniej wytrenowanych i załadowanych)
        X_pack: dane wejściowe (ndarray lub dict dla TFT)
        method: 'mean', 'median' lub 'weighted'
        weights: lista wag dla metody 'weighted' (musi sumować się do 1)
        inverse_scale: czy odskalać predykcje zgodnie ze scalerem modelu (domyślnie True)

    Returns:
        np.ndarray shape=(n, horizon)
    """
    if not predictors:
        raise ValueError("Lista predictorów jest pusta")

    preds = []
    for pr in predictors:
        arr = pr.predict(X_pack, inverse_scale=inverse_scale, verbose=0)
        preds.append(standardize_preds(arr))

    preds_arr = np.stack(preds, axis=0)  # shape=(n_models, n, horizon)

    if method == "mean":
        ens = preds_arr.mean(axis=0)
    elif method == "median":
        ens = np.median(preds_arr, axis=0)
    elif method == "weighted":
        if weights is None or len(weights) != len(predictors):
            raise ValueError("Dla metody 'weighted' podaj listę wag o długości = liczba modeli")
        w = np.array(weights, dtype=float)
        if not np.isclose(w.sum(), 1.0):
            raise ValueError("Wagi muszą sumować się do 1.0")
        # ważona średnia po osi modeli
        ens = np.tensordot(w, preds_arr, axes=(0, 0))
    else:
        raise ValueError(f"Nieznana metoda ensemble: {method}")

    LOGGER.info("Zbudowano ensemble (%s) z %d modeli → kształt %s", method, len(predictors), ens.shape)
    return ens
