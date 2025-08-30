# src/project/model_trainer.py
# Jednolity trening modeli Keras z obsługą TFT (dict) i standardowych modeli (ndarray).

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

try:
    from tensorflow import keras
except Exception as e:  # pragma: no cover - środowiska bez TF
    raise ImportError(
        "TensorFlow/Keras są wymagane przez model_trainer.py. "
        "Zainstaluj tensorflow>=2.12."
    ) from e

from model_builder import build_model
from model_registry import requires_tft_input

LOGGER = logging.getLogger(__name__)


# ————————————————————————————————————————————————————————————————————————
# Pomocnicze struktury i funkcje
# ————————————————————————————————————————————————————————————————————————

@dataclass(frozen=True)
class TrainerConfig:
    """Konfiguracja treningu o sensownych domyślnych wartościach."""
    epochs: int = 20
    batch_size: int = 64
    patience: int = 5
    min_delta: float = 1e-4
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    shuffle: bool = True
    checkpoint_dir: str | None = None  # jeśli podane → zapisuj najlepszy model (val loss)


def _as_trainer_config(params: Dict[str, Any]) -> TrainerConfig:
    t = params.get("trainer", {}) or {}
    return TrainerConfig(
        epochs=int(params.get("epochs", t.get("epochs", 20))),
        batch_size=int(params.get("batch_size", t.get("batch_size", 64))),
        patience=int(t.get("patience", 5)),
        min_delta=float(t.get("min_delta", 1e-4)),
        reduce_lr_patience=int(t.get("reduce_lr_patience", 3)),
        reduce_lr_factor=float(t.get("reduce_lr_factor", 0.5)),
        shuffle=bool(t.get("shuffle", True)),
        checkpoint_dir=t.get("checkpoint_dir"),
    )


def _ensure_shapes_match(
    X: np.ndarray | Dict[str, np.ndarray],
    Y: np.ndarray,
    *,
    is_tft: bool,
) -> None:
    """Podstawowe sanity checks kształtów wejść/wyjść."""
    if not isinstance(Y, np.ndarray) or Y.ndim != 2:
        raise ValueError(f"Y musi mieć shape (n, horizon) 2D, otrzymano {getattr(Y, 'shape', None)}")
    n_y = Y.shape[0]
    if is_tft:
        if not isinstance(X, dict) or "observed_past" not in X or "known_future" not in X:
            raise ValueError("Dla TFT X musi być dict z 'observed_past' i 'known_future'.")
        n_obs = X["observed_past"].shape[0]
        n_kf = X["known_future"].shape[0]
        if n_obs != n_y or n_kf != n_y:
            raise ValueError(f"Niezgodne rozmiary: len(Y)={n_y}, observed={n_obs}, known_future={n_kf}")
    else:
        if isinstance(X, dict):
            X = X.get("standard") or X.get("observed_past")
            if X is None:
                raise ValueError("Dla modeli standardowych X musi być ndarray lub dict z 'standard'/'observed_past'.")
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            raise ValueError(f"X (standard) musi mieć shape (n, seq, feat), otrzymano {getattr(X, 'shape', None)}")
        if X.shape[0] != n_y:
            raise ValueError(f"Niezgodne rozmiary: len(Y)={n_y}, len(X)={X.shape[0]}")


def _materialize_standard_X(X: np.ndarray | Dict[str, np.ndarray]) -> np.ndarray:
    """Zwróć ndarray dla standardowych modeli, akceptując paczkę/dict."""
    if isinstance(X, dict):
        Xn = X.get("standard") or X.get("observed_past")
        if Xn is None:
            raise ValueError("Oczekiwano ndarray lub dict z 'standard'/'observed_past'.")
        return Xn
    return X


def make_validation_split(
    X: np.ndarray | Dict[str, np.ndarray],
    Y: np.ndarray,
    val_fraction: float = 0.15,
) -> Tuple[np.ndarray | Dict[str, np.ndarray], np.ndarray, np.ndarray | Dict[str, np.ndarray], np.ndarray]:
    """
    Dzieli (X, Y) na (X_tr, Y_tr, X_val, Y_val). Zachowuje strukturę TFT (dict).
    Podział jest „od końca” – typowe dla szeregów czasowych.
    """
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction musi być w (0,1).")
    n = Y.shape[0]
    split = max(1, int(round(n * (1.0 - val_fraction))))
    if isinstance(X, dict):
        X_tr = {k: v[:split] for k, v in X.items()}
        X_val = {k: v[split:] for k, v in X.items()}
    else:
        X_tr, X_val = X[:split], X[split:]
    Y_tr, Y_val = Y[:split], Y[split:]
    return X_tr, Y_tr, X_val, Y_val


def _callbacks(cfg: TrainerConfig, ckpt_prefix: str | None = None) -> list[Any]:
    cbs: list[Any] = []
    cbs.append(
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            min_delta=cfg.min_delta,
            restore_best_weights=True,
        )
    )
    cbs.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=cfg.reduce_lr_patience,
            factor=cfg.reduce_lr_factor,
            min_lr=1e-6,
        )
    )
    if cfg.checkpoint_dir:
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        fname = "best.keras" if ckpt_prefix is None else f"{ckpt_prefix}_best.keras"
        cbs.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(Path(cfg.checkpoint_dir) / fname),
                monitor="val_loss",
                save_best_only=True,
            )
        )
    return cbs


def _history_to_dict(h: "keras.callbacks.History") -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for k, v in (h.history or {}).items():
        out[str(k)] = [float(x) for x in v]
    return out


# ————————————————————————————————————————————————————————————————————————
# Public API: trening
# ————————————————————————————————————————————————————————————————————————

def train_model(
    X_train: np.ndarray | Dict[str, np.ndarray],
    Y_train: np.ndarray,
    X_val: np.ndarray | Dict[str, np.ndarray],
    Y_val: np.ndarray,
    params: Dict[str, Any],
    *,
    is_incremental: bool = False,
) -> Tuple["keras.Model", Dict[str, Any], Dict[str, list[float]]]:
    """
    Trenuje model zgodnie z parametrami i zwraca (model, finalne_parametry, historia).

    Rozpoznawane klucze `params` (wycinek):
      - model_type: str (LSTM|GRU|CNN-LSTM|TRANSFORMER|TFT) [wymagane]
      - nazwa_modelu: str (alias używany w logach/checkpointach)
      - epochs, batch_size, trainer.{patience,min_delta,reduce_lr_*...,checkpoint_dir}
      - loss, optimizer, lr, momentum
      - units, filters, heads, ff_dim, blocks, dropout, batchnorm, layers

    `is_incremental` – jeżeli True, dopasowuje istniejący model z `params["prebuilt_model"]`
    (jeśli podany), w przeciwnym razie buduje od zera.
    """
    if "model_type" not in params:
        raise ValueError("params['model_type'] jest wymagane.")

    model_name = str(params.get("nazwa_modelu", params["model_type"]))
    is_tft = requires_tft_input(params["model_type"])

    # sanity checks kształtów wejść
    _ensure_shapes_match(X_train, Y_train, is_tft=is_tft)
    _ensure_shapes_match(X_val, Y_val, is_tft=is_tft)

    # Budowa lub reuse modelu
    t0 = time.time()
    if is_incremental and params.get("prebuilt_model") is not None:
        model = params["prebuilt_model"]
        LOGGER.info("Kontynuuję trening istniejącego modelu: %s", model_name)
    else:
        input_example = X_train if is_tft else _materialize_standard_X(X_train)
        model = build_model(params["model_type"], X_train, Y_train.shape[1], params=params)
        LOGGER.info("Zbudowano model %s w %.2fs", model_name, time.time() - t0)

    # Parametry treningu
    tcfg = _as_trainer_config(params)
    cbs = _callbacks(tcfg, ckpt_prefix=model_name)

    # Dane wejściowe: pozostaw X bez zmian (dla TFT dict, dla standardu ndarray)
    x_tr = X_train if is_tft else _materialize_standard_X(X_train)
    x_va = X_val if is_tft else _materialize_standard_X(X_val)

    hist = model.fit(
        x=x_tr,
        y=Y_train,
        validation_data=(x_va, Y_val),
        epochs=tcfg.epochs,
        batch_size=tcfg.batch_size,
        shuffle=tcfg.shuffle if not is_tft else False,  # dla szeregów czasowych i TFT zwykle False
        verbose=0,
        callbacks=cbs,
    )

    history_dict = _history_to_dict(hist)
    eff_params = copy.deepcopy(params)
    eff_params["trainer"] = {
        "epochs": tcfg.epochs,
        "batch_size": tcfg.batch_size,
        "patience": tcfg.patience,
        "min_delta": tcfg.min_delta,
        "reduce_lr_patience": tcfg.reduce_lr_patience,
        "reduce_lr_factor": tcfg.reduce_lr_factor,
        "shuffle": tcfg.shuffle if not is_tft else False,
        "checkpoint_dir": tcfg.checkpoint_dir,
    }

    LOGGER.info(
        "Trening zakończony: %s | best_val=%.6f | epoki=%d | czas=%.2fs",
        model_name,
        float(np.min(history_dict.get("val_loss", [np.inf]))),
        len(history_dict.get("loss", [])),
        time.time() - t0,
    )

    return model, eff_params, history_dict
