# src/project/model_builder.py
# Budowa modeli sekwencyjnych (Keras): LSTM, GRU, CNN-LSTM, TRANSFORMER, TFT.

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:  # pragma: no cover - środowiska bez TF
    raise ImportError(
        "TensorFlow/Keras są wymagane przez model_builder.py. "
        "Zainstaluj tensorflow>=2.12."
    ) from e

from model_registry import ModelType, requires_tft_input

LOGGER = logging.getLogger(__name__)


# ————————————————————————————————————————————————————————————————————————
# Pomocnicze: inferencja sygnatury wejść i sanity checks
# ————————————————————————————————————————————————————————————————————————

def _shape_wo_batch(arr: np.ndarray) -> Tuple[int, ...]:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Spodziewano się np.ndarray, otrzymano {type(arr)}")
    if arr.ndim < 2:
        raise ValueError(f"Wejście musi mieć ≥2 wymiary (batch, ...). Otrzymano shape={arr.shape}")
    return tuple(arr.shape[1:])


def infer_input_signature(
    input_example: np.ndarray | Dict[str, np.ndarray],
    model_type: str | ModelType,
) -> Dict[str, Tuple[int, ...]]:
    """
    Zwraca sygnatury wejść (bez batch):
      - dla modeli standardowych: {"standard": (seq_len, n_features)}
      - dla TFT: {"observed_past": (seq_len, n_obs), "known_future": (horizon, n_known)}
    """
    mt = ModelType(model_type) if not isinstance(model_type, ModelType) else model_type
    if requires_tft_input(mt):
        if not isinstance(input_example, dict) or \
           "observed_past" not in input_example or "known_future" not in input_example:
            raise ValueError("Dla TFT wymagane jest wejście dict z kluczami: 'observed_past', 'known_future'.")
        sig = {
            "observed_past": _shape_wo_batch(input_example["observed_past"]),
            "known_future": _shape_wo_batch(input_example["known_future"]),
        }
        if sig["observed_past"][0] <= 0 or sig["known_future"][0] <= 0:
            raise ValueError("Długości sekwencji/horyzontu muszą być dodatnie.")
        return sig
    # standard
    if isinstance(input_example, dict):
        # dopuszczamy paczkę – wybierz "standard" albo "observed_past"
        x = input_example.get("standard") or input_example.get("observed_past")
        if x is None:
            raise ValueError("Dla modeli standardowych oczekiwano ndarray lub dict z 'standard'/'observed_past'.")
        return {"standard": _shape_wo_batch(x)}
    return {"standard": _shape_wo_batch(input_example)}


# ————————————————————————————————————————————————————————————————————————
# Bloki/warstwy wielokrotnego użytku
# ————————————————————————————————————————————————————————————————————————

def _maybe_batchnorm(x: "keras.Tensor", use_bn: bool) -> "keras.Tensor":
    return layers.BatchNormalization()(x) if use_bn else x


def _transformer_encoder_block(x: "keras.Tensor", d_model: int, num_heads: int, ff_dim: int, dropout: float, use_bn: bool) -> "keras.Tensor":
    # pre-norm
    h = layers.LayerNormalization()(x)
    h = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(h, h)
    h = layers.Dropout(dropout)(h)
    x = layers.Add()([x, h])

    h = layers.LayerNormalization()(x)
    h = layers.Dense(ff_dim, activation="relu")(h)
    h = layers.Dropout(dropout)(h)
    h = layers.Dense(d_model)(h)
    x = layers.Add()([x, h])
    return _maybe_batchnorm(x, use_bn)


def _cnn_backbone(x: "keras.Tensor", filters: int, dropout: float, use_bn: bool) -> "keras.Tensor":
    x = layers.Conv1D(filters=filters, kernel_size=3, padding="causal", activation="relu")(x)
    x = _maybe_batchnorm(x, use_bn)
    x = layers.Conv1D(filters=filters, kernel_size=3, padding="causal", activation="relu")(x)
    x = _maybe_batchnorm(x, use_bn)
    x = layers.Dropout(dropout)(x)
    return x


# ————————————————————————————————————————————————————————————————————————
# Budowniczowie modeli
# ————————————————————————————————————————————————————————————————————————

def _build_lstm(input_shape: Tuple[int, int], horizon: int, p: Dict[str, Any]) -> "keras.Model":
    units = int(p.get("units", 64))
    dropout = float(p.get("dropout", 0.1))
    layers_count = int(p.get("layers", 1))
    use_bn = bool(p.get("batchnorm", False))

    inp = keras.Input(shape=input_shape, name="inputs")
    x = inp
    for i in range(max(1, layers_count) - 1):
        x = layers.LSTM(units, return_sequences=True)(x)
        x = layers.Dropout(dropout)(x)
        x = _maybe_batchnorm(x, use_bn)
    x = layers.LSTM(units)(x)
    x = layers.Dropout(dropout)(x)
    x = _maybe_batchnorm(x, use_bn)
    out = layers.Dense(horizon, name="prediction")(x)
    return keras.Model(inp, out, name="LSTM")


def _build_gru(input_shape: Tuple[int, int], horizon: int, p: Dict[str, Any]) -> "keras.Model":
    units = int(p.get("units", 64))
    dropout = float(p.get("dropout", 0.1))
    layers_count = int(p.get("layers", 1))
    use_bn = bool(p.get("batchnorm", False))

    inp = keras.Input(shape=input_shape, name="inputs")
    x = inp
    for i in range(max(1, layers_count) - 1):
        x = layers.GRU(units, return_sequences=True)(x)
        x = layers.Dropout(dropout)(x)
        x = _maybe_batchnorm(x, use_bn)
    x = layers.GRU(units)(x)
    x = layers.Dropout(dropout)(x)
    x = _maybe_batchnorm(x, use_bn)
    out = layers.Dense(horizon, name="prediction")(x)
    return keras.Model(inp, out, name="GRU")


def _build_cnn_lstm(input_shape: Tuple[int, int], horizon: int, p: Dict[str, Any]) -> "keras.Model":
    filters = int(p.get("filters", 64))
    units = int(p.get("units", 64))
    dropout = float(p.get("dropout", 0.1))
    use_bn = bool(p.get("batchnorm", False))

    inp = keras.Input(shape=input_shape, name="inputs")
    x = _cnn_backbone(inp, filters=filters, dropout=dropout, use_bn=use_bn)
    x = layers.LSTM(units)(x)
    x = layers.Dropout(dropout)(x)
    x = _maybe_batchnorm(x, use_bn)
    out = layers.Dense(horizon, name="prediction")(x)
    return keras.Model(inp, out, name="CNN_LSTM")


def _build_transformer(input_shape: Tuple[int, int], horizon: int, p: Dict[str, Any]) -> "keras.Model":
    d_model = int(p.get("d_model", 64))
    num_heads = int(p.get("heads", 4))
    ff_dim = int(p.get("ff_dim", 128))
    blocks = int(p.get("blocks", 2))
    dropout = float(p.get("dropout", 0.1))
    use_bn = bool(p.get("batchnorm", False))

    inp = keras.Input(shape=input_shape, name="inputs")  # (seq, feat)
    # dopasuj kanały do d_model jeśli trzeba
    x = layers.Dense(d_model)(inp)
    for _ in range(max(1, blocks)):
        x = _transformer_encoder_block(x, d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout, use_bn=use_bn)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(horizon, name="prediction")(x)
    return keras.Model(inp, out, name="TRANSFORMER")


def _build_tft(sig: Dict[str, Tuple[int, ...]], horizon: int, p: Dict[str, Any]) -> "keras.Model":
    """
    Minimalny „TFT-like” (proxy) z dwoma wejściami:
      - observed_past: (seq_len, n_obs)
      - known_future:  (horizon, n_known)
    Realne TFT może być podpięte tutaj, zachowując nazwy wejść i wyjść.
    """
    units = int(p.get("units", 64))
    dropout = float(p.get("dropout", 0.1))
    use_bn = bool(p.get("batchnorm", False))

    inp_obs = keras.Input(shape=sig["observed_past"], name="observed_past")   # (seq, n_obs)
    inp_kf  = keras.Input(shape=sig["known_future"],  name="known_future")    # (horizon, n_known)

    # Enkodowanie przeszłości
    x = layers.LSTM(units, return_sequences=True)(inp_obs)
    x = layers.Dropout(dropout)(x)
    x = _maybe_batchnorm(x, use_bn)
    x = layers.LSTM(units)(x)

    # Enkodowanie znanej przyszłości (proste uśrednienie informacji)
    k = layers.TimeDistributed(layers.Dense(units, activation="relu"))(inp_kf)
    k = layers.GlobalAveragePooling1D()(k)

    h = layers.Concatenate()([x, k])
    h = layers.Dropout(dropout)(h)
    h = _maybe_batchnorm(h, use_bn)
    out = layers.Dense(horizon, name="prediction")(h)

    return keras.Model([inp_obs, inp_kf], out, name="TFT_PROXY")


# ————————————————————————————————————————————————————————————————————————
# Kompilacja i główne API
# ————————————————————————————————————————————————————————————————————————

def _compile(model: "keras.Model", params: Dict[str, Any]) -> "keras.Model":
    loss = str(params.get("loss", "mse"))
    lr = float(params.get("lr", 1e-3))
    opt_name = str(params.get("optimizer", "adam")).lower()

    if opt_name == "adam":
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=lr)
    elif opt_name == "sgd":
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=float(params.get("momentum", 0.0)))
    else:
        LOGGER.warning("Nieznany optimizer '%s' – używam Adam.", opt_name)
        opt = keras.optimizers.Adam(learning_rate=lr)

    metrics = ["mae", "mse"]
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model


def build_model(
    model_type: str | ModelType,
    input_example: np.ndarray | Dict[str, np.ndarray],
    horizon: int,
    params: Dict[str, Any] | None = None,
) -> "keras.Model":
    """
    Buduje i kompiluje model Keras o żądanym typie.

    Args:
        model_type: jeden z ModelType (LSTM, GRU, CNN-LSTM, TRANSFORMER, TFT)
        input_example: przykładowe wejście:
            - dla standardowych modeli: ndarray (batch, seq_len, n_features) lub dict z kluczem 'standard'/'observed_past'
            - dla TFT: dict {'observed_past': (batch, seq_len, n_obs), 'known_future': (batch, horizon, n_known)}
        horizon: liczba kroków predykcji (wyjście ma shape (batch, horizon))
        params: słownik hiperparametrów (units, dropout, lr, loss, optimizer, ...)

    Returns:
        Skompilowany `keras.Model` z wyjściem nazwanym 'prediction'.
    """
    if horizon <= 0:
        raise ValueError("Horizon musi być dodatni.")
    p = params or {}
    mt = ModelType(model_type) if not isinstance(model_type, ModelType) else model_type

    sig = infer_input_signature(input_example, mt)
    LOGGER.info("Buduję model %s; sygnatura wejść: %s; horizon=%d", mt.value, sig, horizon)

    if mt == ModelType.LSTM:
        model = _build_lstm(sig["standard"], horizon, p)
    elif mt == ModelType.GRU:
        model = _build_gru(sig["standard"], horizon, p)
    elif mt == ModelType.CNN_LSTM:
        model = _build_cnn_lstm(sig["standard"], horizon, p)
    elif mt == ModelType.TRANSFORMER:
        model = _build_transformer(sig["standard"], horizon, p)
    elif mt == ModelType.TFT:
        model = _build_tft(sig, horizon, p)
    else:  # pragma: no cover - zabezpieczenie Enum
        raise ValueError(f"Nieobsługiwany typ modelu: {mt}")

    model = _compile(model, p)
    return model
