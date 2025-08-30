# src/project/model_registry.py
# Rejestr typów modeli i narzędzia do standaryzacji wejść/wyjść.

from __future__ import annotations

import enum
import logging
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class ModelType(str, enum.Enum):
    """Obsługiwane typy modeli w projekcie."""

    LSTM = "LSTM"
    GRU = "GRU"
    CNN_LSTM = "CNN-LSTM"
    TRANSFORMER = "TRANSFORMER"
    TFT = "TFT"

    @classmethod
    def list(cls) -> list[str]:
        """Lista dopuszczalnych nazw modeli jako stringi."""
        return [m.value for m in cls]


# ————————————————————————————————————————————————————————————————————————
# Funkcje pomocnicze
# ————————————————————————————————————————————————————————————————————————

def requires_tft_input(model_type: str | ModelType) -> bool:
    """
    Czy dany model wymaga wejścia w formacie `dict` (TFT)?
    Pozostałe przyjmują `np.ndarray`.
    """
    mt = ModelType(model_type) if not isinstance(model_type, ModelType) else model_type
    return mt == ModelType.TFT


def standardize_preds(preds: Any) -> np.ndarray:
    """
    Standaryzuje predykcje modeli do formatu `(n_samples, horizon)`.

    Obsługuje:
    - `np.ndarray`: akceptowane bez zmian, wymuszamy 2D
    - listy: konwertowane do `np.ndarray`
    - dict: jeśli model TFT zwróci słownik, wybieramy `main` lub pierwszy klucz

    Raises:
        ValueError: jeśli format predykcji jest nieobsługiwany
    """
    if preds is None:
        raise ValueError("Predykcje są None")

    arr: np.ndarray
    if isinstance(preds, np.ndarray):
        arr = preds
    elif isinstance(preds, (list, tuple)):
        arr = np.asarray(preds)
    elif isinstance(preds, dict):
        if "main" in preds:
            arr = np.asarray(preds["main"])
        else:
            # fallback: bierzemy pierwszą wartość
            key, val = next(iter(preds.items()))
            LOGGER.warning("Predykcja dict → używam klucza '%s'", key)
            arr = np.asarray(val)
    else:
        raise ValueError(f"Nieobsługiwany format predykcji: {type(preds)}")

    if arr.ndim == 1:
        arr = arr[:, None]  # (n,) → (n,1)

    if arr.ndim != 2:
        raise ValueError(f"Predykcje muszą być 2D, a są {arr.shape}")

    return arr


def validate_model_name(name: str) -> ModelType:
    """
    Waliduje nazwę modelu względem `ModelType`.
    Obsługuje aliasy (case-insensitive).
    """
    if not name:
        raise ValueError("Nazwa modelu nie może być pusta")

    upper = name.strip().upper().replace("_", "-")
    for mt in ModelType:
        if upper == mt.value.upper():
            return mt
    raise ValueError(f"Nieznany typ modelu: {name}. Dozwolone: {ModelType.list()}")
