# src/project/data_preprocessing.py
# Przygotowanie danych: cechy, skalowanie, selekcja, sekwencje (standard + TFT), augmentacja.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
    PowerTransformer,
    QuantileTransformer,
)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

LOGGER = logging.getLogger(__name__)

# ————————————————————————————————————————————————————————————————————————
# Konfiguracja i pomocnicze typy
# ————————————————————————————————————————————————————————————————————————

_SCALERS = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
    "robust": RobustScaler,
    "maxabs": MaxAbsScaler,
    "power": PowerTransformer,       # Box-Cox/Yeo-Johnson
    "quantile": QuantileTransformer, # rank-gauss
}


@dataclass(frozen=True)
class PreprocessConfig:
    """Konfiguracja przygotowania danych (wartości domyślne)."""
    sequence_length: int = 30
    forecast_horizon: int = 14
    scaler: str = "minmax"  # jeden z: _SCALERS
    feature_selection_method: str | None = None  # np. "rfe" lub None
    feature_selection_top_k: int = 30
    include_sp500: bool = True
    augment_noise_level: float = 0.0  # 0.01 => 1% szumu
    augment_times: int = 0            # ile replik augmentować
    dropna: bool = True               # usuń wiersze z NaN po inżynierii
    tft_enable: bool = True           # dokłada strukturę wejść dla TFT


# ————————————————————————————————————————————————————————————————————————
# Budowa cech i łączenie z S&P 500
# ————————————————————————————————————————————————————————————————————————

def _add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje podstawowe cechy techniczne (bez TA-Lib, brak zależności)."""
    out = df.copy()
    # zwroty proste i logarytmiczne
    out["ret_1"] = out["close"].pct_change()
    out["logret_1"] = np.log(out["close"]).diff()
    # zmienność krocząca
    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()
    # momentum
    out["mom_5"] = out["close"] / out["close"].shift(5) - 1.0
    out["mom_10"] = out["close"] / out["close"].shift(10) - 1.0
    # średnie kroczące
    out["sma_5"] = out["close"].rolling(5).mean()
    out["sma_10"] = out["close"].rolling(10).mean()
    out["sma_20"] = out["close"].rolling(20).mean()
    # różnice high/low/close
    out["hl_spread"] = (out["high"] - out["low"]) / out["close"]
    out["oc_spread"] = (out["open"] - out["close"]) / out["close"]
    # kierunek
    out["dir"] = np.sign(out["ret_1"]).fillna(0.0)
    return out


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje cechy czasowe (część można oznaczyć jako 'known_future' dla TFT)."""
    out = df.copy()
    idx = out.index
    out["dow"] = idx.dayofweek           # 0..6
    out["dom"] = idx.day                 # 1..31
    out["month"] = idx.month             # 1..12
    out["is_month_start"] = idx.is_month_start.astype(int)
    out["is_month_end"] = idx.is_month_end.astype(int)
    return out


def _merge_sp500(df: pd.DataFrame, sp500_df: pd.DataFrame | None) -> pd.DataFrame:
    if sp500_df is None or sp500_df.empty:
        return df
    s = sp500_df.copy()
    # minimalny zestaw: close i dzienne zwroty
    s = s.rename(columns={c: f"sp500_{c}" for c in s.columns})
    s["sp500_ret_1"] = s["sp500_close"].pct_change()
    return df.join(s[["sp500_close", "sp500_ret_1"]], how="left")


def _build_features(
    raw_df: pd.DataFrame,
    *,
    include_sp500: bool,
    sp500_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Główny pipeline cech (bez skalowania)."""
    if raw_df is None or raw_df.empty:
        raise ValueError("Brak danych wejściowych (raw_df).")
    # standaryzacja indeksu i sortowanie
    df = raw_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    # podstawowe cechy
    df = _add_technical_features(df)
    df = _add_time_features(df)
    # join sp500
    if include_sp500:
        df = _merge_sp500(df, sp500_df)
    return df


# ————————————————————————————————————————————————————————————————————————
# Selekcja cech (opcjonalna)
# ————————————————————————————————————————————————————————————————————————

def _select_features_rfe(df: pd.DataFrame, target_col: str, top_k: int) -> list[str]:
    """RFE nad RandomForest – szybka, odporna metoda wyboru najważniejszych cech."""
    feats = [c for c in df.columns if c != target_col]
    X = df[feats].replace([np.inf, -np.inf], np.nan).dropna()
    y = df.loc[X.index, target_col]
    if len(X) < max(200, top_k * 5):
        # za mało danych – pomiń RFE
        LOGGER.warning("Za mało danych do RFE (%d wierszy). Pomijam selekcję cech.", len(X))
        return feats[:top_k]
    est = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=est, n_features_to_select=min(top_k, len(feats)), step=0.1)
    rfe.fit(X, y)
    selected = list(np.array(feats)[rfe.support_])
    return selected


def _maybe_select_features(
    df: pd.DataFrame,
    method: str | None,
    top_k: int,
    target_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    feats_all = [c for c in df.columns if c != target_col]
    if method is None:
        return df, feats_all
    method = method.lower()
    if method == "rfe":
        sel = _select_features_rfe(df, target_col, top_k)
        return df[sel + [target_col]], sel
    LOGGER.warning("Nieznana metoda selekcji cech: %s (pomijam)", method)
    return df, feats_all


# ————————————————————————————————————————————————————————————————————————
# Skalowanie
# ————————————————————————————————————————————————————————————————————————

def _get_scaler(name: str):
    key = (name or "minmax").lower()
    if key not in _SCALERS:
        raise ValueError(f"Nieznany scaler: {name}. Dozwolone: {list(_SCALERS)}")
    return _SCALERS[key]()


def _fit_transform_scalers(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    scaler_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dopasuj scalery i przeskaluj cechy oraz (opcjonalnie) target."""
    X_scaler = _get_scaler(scaler_name)
    T_scaler = _get_scaler(scaler_name)

    df_scaled = df.copy()
    df_scaled[feature_cols] = X_scaler.fit_transform(df_scaled[feature_cols].astype(float))
    df_scaled[[target_col]] = T_scaler.fit_transform(df_scaled[[target_col]].astype(float))

    scalers = {"features": X_scaler, "target": T_scaler}
    return df_scaled, scalers


# ————————————————————————————————————————————————————————————————————————
# Sekwencje (standard + TFT)
# ————————————————————————————————————————————————————————————————————————

def _make_supervised_sequences(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Buduje okna wejściowe (n, seq_len, n_feat) i cele (n, horizon) -> standard dla LSTM/GRU/CNN/Transformer.

    Target to przyszłe wartości `target_col` w horyzoncie [t+1..t+horizon] (point-forecast).
    """
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    vals = df[feature_cols + [target_col]].values.astype(float)
    n_total = len(df)
    n_feat = len(feature_cols)

    for t in range(seq_len, n_total - horizon + 1):
        x_win = vals[t - seq_len : t, :n_feat]
        y_win = vals[t : t + horizon, -1]
        if np.isnan(x_win).any() or np.isnan(y_win).any():
            continue
        X_list.append(x_win)
        y_list.append(y_win)

    if not X_list:
        raise ValueError("Nie udało się zbudować sekwencji: brak pełnych okien bez NaN.")
    X = np.stack(X_list, axis=0)              # (n, seq_len, n_feat)
    Y = np.stack(y_list, axis=0)              # (n, horizon)
    return X, Y


def _split_tft_features(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """
    Dzieli cechy na:
    - observed_past: te, które znamy tylko w przeszłości (ceny, wolumeny itp.)
    - known_future: cechy 'kalendarzowe' (dow, dom, month, is_month_*), które znamy z wyprzedzeniem
    """
    known_future_keys = {"dow", "dom", "month", "is_month_start", "is_month_end"}
    obs, known = [], []
    for c in feature_cols:
        (known if any(k in c for k in known_future_keys) else obs).append(c)
    # zapewnij przynajmniej jedną znaną przyszłość (jeśli brak)
    if not known:
        known = ["dow", "dom", "month"]
    return obs, known


def _make_tft_sequences(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
    horizon: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Buduje słownik wejść dla TFT:
    {
      'observed_past': (n, seq_len, n_obs),
      'known_future':  (n, horizon, n_known)  # cechy kalendarzowe „patrzące w przód”
    }
    oraz Y: (n, horizon)
    """
    observed_cols, known_cols = _split_tft_features(feature_cols)
    # dane „observed” – jak w standardowych sekwencjach
    X_obs, Y = _make_supervised_sequences(
        df, feature_cols=observed_cols, target_col=target_col, seq_len=seq_len, horizon=horizon
    )
    # dane „known_future” – przyszłe cechy kalendarzowe od t..t+horizon-1 (tu zrobimy [t+1..t+horizon])
    K_list: list[np.ndarray] = []
    idx = df.index
    df_known = df[known_cols].astype(float).values

    n_total = len(df)
    for t in range(seq_len, n_total - horizon + 1):
        k_win = df_known[t : t + horizon, :]
        if np.isnan(k_win).any():
            k_win = np.nan_to_num(k_win)
        K_list.append(k_win)

    X_known = np.stack(K_list, axis=0)  # (n, horizon, n_known)
    X_tft = {"observed_past": X_obs, "known_future": X_known}
    return X_tft, Y


# ————————————————————————————————————————————————————————————————————————
# Augmentacja
# ————————————————————————————————————————————————————————————————————————

def _augment_gaussian(X: np.ndarray, Y: np.ndarray, *, noise_level: float, times: int) -> tuple[np.ndarray, np.ndarray]:
    """Dodaje biały szum do cech wejściowych (nie do targetu)."""
    if noise_level <= 0.0 or times <= 0:
        return X, Y
    Xs, Ys = [X], [Y]
    scale = noise_level
    for _ in range(times):
        noise = np.random.normal(loc=0.0, scale=scale, size=X.shape).astype(X.dtype)
        Xs.append(X + noise)
        Ys.append(Y.copy())
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


# ————————————————————————————————————————————————————————————————————————
# Główne API: prepare_data
# ————————————————————————————————————————————————————————————————————————

def prepare_data(
    raw_df: pd.DataFrame,
    params: Dict[str, Any],
    *,
    sp500_df: pd.DataFrame | None = None,
) -> tuple[dict[str, object], np.ndarray, dict[str, object], list[str], str, str]:
    """
    Główny punkt wejścia do przygotowania danych.

    Args:
        raw_df: dane bazowe instrumentu (index: DatetimeIndex, kolumny co najmniej: open, high, low, close, volume)
        params: słownik ustawień; rozpoznawane klucze:
            - sequence_length: int (domyślnie 30)
            - forecast_horizon: int (domyślnie 14)
            - scaler: str (minmax|standard|robust|maxabs|power|quantile) – domyślnie minmax
            - feature_selection: {"method": "rfe", "top_k": 30} (opcjonalnie)
            - include_sp500: bool (domyślnie True)
            - augment: {"noise_level": float, "times": int}
            - tft: {"enable": bool}
        sp500_df: opcjonalne dane indeksu referencyjnego (np. ^GSPC)

    Returns:
        (X_pack_all, Y_all, scalers, selected_features, actual_start_date, actual_end_date)
    """
    cfg = PreprocessConfig(
        sequence_length=int(params.get("sequence_length", 30)),
        forecast_horizon=int(params.get("forecast_horizon", 14)),
        scaler=str(params.get("scaler", "minmax")).lower(),
        feature_selection_method=(params.get("feature_selection", {}) or {}).get("method"),
        feature_selection_top_k=int((params.get("feature_selection", {}) or {}).get("top_k", 30)),
        include_sp500=bool(params.get("include_sp500", True)),
        augment_noise_level=float((params.get("augment", {}) or {}).get("noise_level", 0.0)),
        augment_times=int((params.get("augment", {}) or {}).get("times", 0)),
        dropna=bool(params.get("dropna", True)),
        tft_enable=bool((params.get("tft", {}) or {}).get("enable", True)),
    )

    LOGGER.info("\n--- Rozpoczęcie: prepare_data ---")
    # 1) Budowa cech
    df = _build_features(raw_df, include_sp500=cfg.include_sp500, sp500_df=sp500_df)

    # 2) Czyszczenie
    df = df.replace([np.inf, -np.inf], np.nan)
    if cfg.dropna:
        # usuń początkowe wiersze wymagane przez rolling/shift (NaN)
        before = len(df)
        df = df.dropna()
        LOGGER.info("  Usunięto %d wierszy z powodu NaN/Inf.", before - len(df))
    if df.empty:
        raise ValueError("Brak danych po czyszczeniu; sprawdź wejście i parametry.")

    # 3) Target: przewidujemy `close` (po skalowaniu)
    target_col = "close"

    # 4) (Opcjonalnie) selekcja cech
    df_sel, feature_cols = _maybe_select_features(
        df, method=cfg.feature_selection_method, top_k=cfg.feature_selection_top_k, target_col=target_col
    )

    # 5) Skalowanie
    df_scaled, scalers = _fit_transform_scalers(
        df_sel, feature_cols=feature_cols, target_col=target_col, scaler_name=cfg.scaler
    )

    # 6) Sekwencje – standard
    X_std, Y = _make_supervised_sequences(
        df_scaled, feature_cols=feature_cols, target_col=target_col,
        seq_len=cfg.sequence_length, horizon=cfg.forecast_horizon
    )

    # 7) Sekwencje – TFT (opcjonalnie)
    if cfg.tft_enable:
        X_tft, Y_tft = _make_tft_sequences(
            df_scaled, feature_cols=feature_cols, target_col=target_col,
            seq_len=cfg.sequence_length, horizon=cfg.forecast_horizon
        )
        # sanity: Y musi być spójne
        if not np.allclose(Y, Y_tft):
            LOGGER.warning("  Y między standard i TFT minimalnie się różni (zaokrąglenie) – używam Y ze standard.")
        X_pack_all: dict[str, object] = {"standard": X_std, "tft": X_tft}
    else:
        X_pack_all = {"standard": X_std}

    # 8) Augmentacja (tylko standard; TFT pozostawiamy bez szumu, aby nie psuć struktury known_future)
    if cfg.augment_noise_level > 0.0 and cfg.augment_times > 0:
        X_aug, Y_aug = _augment_gaussian(X_std, Y, noise_level=cfg.augment_noise_level, times=cfg.augment_times)
        X_pack_all["standard"] = X_aug
        Y = Y_aug

    # 9) Metadane zakresu
    actual_start_date = df_scaled.index.min().date().isoformat()
    actual_end_date = df_scaled.index.max().date().isoformat()

    LOGGER.info("  Utworzono sekwencje: Y=%s, X standard=%s, TFT=%s",
                Y.shape, X_pack_all["standard"].__class__.__name__,
                "tak" if "tft" in X_pack_all else "nie")
    LOGGER.info("--- Zakończono: prepare_data ---\n")

    selected_features = feature_cols
    return X_pack_all, Y, scalers, selected_features, actual_start_date, actual_end_date
