# main.py
# Główny pipeline: wczytanie configu YAML, dane, (auto)tuning, trening, zapis modeli, ensemble.

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np

# ——— Importy projektowe (pliki w katalogu głównym projektu) ———
from config_models import load_and_validate_config, generate_experiments, resolve_output_paths  # type: ignore
from data_loader import pobierz_dane_akcji # type: ignore
from data_loader import pobierz_dane_sp500 # type: ignore
from data_preprocessing import prepare_data  # type: ignore
from autotune import autotune_keras  # type: ignore
from model_trainer import train_model, make_validation_split  # type: ignore
from model_saver_loader import save_model_bundle  # type: ignore
from predictor import Predictor  # type: ignore
from ensemble_builder import build_ensemble  # type: ignore
from super_ensemble import SuperEnsemble  # type: ignore
from model_registry import validate_model_name  # type: ignore
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger(__name__)


# ————————————————————————————————————————————————————————————————————————
# Pomocnicze
# ————————————————————————————————————————————————————————————————————————

def setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config_dynamic(path_like: str | Path):
    cfg_path = Path(path_like)
    if not cfg_path.is_absolute():
        cfg_path = BASE_DIR / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku konfiguracyjnego: {cfg_path}")
    return load_and_validate_config(cfg_path)

def build_outputs(symbol: str, cfg, loss: str, probe: str) -> dict[str, Path]:
    """Zwraca słownik ścieżek wyjściowych i tworzy katalogi."""
    paths = resolve_output_paths(cfg.output_dir, symbol, loss, probe)
    # podfoldery na artefakty
    (paths["run_dir"] / "zapisane_modele").mkdir(parents=True, exist_ok=True)
    (paths["run_dir"] / "tuner_dir").mkdir(parents=True, exist_ok=True)
    (paths["run_dir"] / "wykresy").mkdir(parents=True, exist_ok=True)
    return paths


def maybe_autotune(
    X: np.ndarray | dict[str, np.ndarray],
    Y: np.ndarray,
    params: dict[str, Any],
    *,
    autotune_cfg: dict[str, Any] | None,
    tuner_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Jeśli włączony autotuning w configu, przeprowadza szybki tuning (Optuna/KerasTuner),
    zwraca najlepsze parametry scalone z bazowymi.
    """
    auto = autotune_cfg or {}
    enabled = bool(auto.get("enabled", False))
    if not enabled:
        return params

    engine = str(auto.get("engine", "optuna"))
    n_trials = int(auto.get("n_trials", 20))
    timeout = auto.get("timeout")
    direction = str(auto.get("direction", "minimize"))
    epochs = int(auto.get("epochs", 8)) if "epochs" in auto else 8
    batch_size = int(auto.get("batch_size", params.get("trainer", {}).get("batch_size", 64)))

    LOGGER.info("Autotune włączony (%s): trials=%s, timeout=%s", engine, n_trials, timeout)
    res = autotune_keras(
        X, Y,
        base_params={"model_type": params["model_type"], **params},
        tuner=engine,
        n_trials=n_trials,
        timeout=timeout,
        direction=direction,
        epochs=epochs,
        batch_size=batch_size,
    )
    best = dict(params)
    best.update(res.best_params or {})
    LOGGER.info("Autotune: najlepszy wynik=%.6f; zaktualizowano parametry.", res.best_score)
    return best


def _save_bundle(model, scalers, params: dict[str, Any], base_name: str, save_dir: Path) -> None:
    ok = save_model_bundle(model, scalers, params, base_filename=base_name, save_dir=save_dir)
    if not ok:
        LOGGER.warning("Nie udało się zapisać bundla modelu: %s", base_name)


def _materialize_standard_X(X: np.ndarray | dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(X, dict):
        return X.get("standard") or X.get("observed_past")
    return X


def _fetch_data(symbol: str, cfg) -> tuple[np.ndarray | dict[str, np.ndarray], np.ndarray, dict, list[str], dict]:
    """
    Pobiera dane, przygotowuje je i zwraca pakiet do trenowania.
    Zwracane: (X_pack, Y, scalers, selected_features, meta)
    """
    # Zakres dat – opcjonalny w YAML; gdy brak, load_symbol_history sam zadba o pełny zakres
    data_dates = getattr(cfg, "data", None)
    start_date = getattr(data_dates, "start_date", None) if data_dates else None
    end_date = getattr(data_dates, "end_date", None) if data_dates else None

    LOGGER.info("Wczytywanie danych (%s)...", symbol)
    raw_df = pobierz_dane_akcji(symbol, data_start=start_date, data_end=end_date)
    print(f"Loaded raw_df for {symbol}: {raw_df.shape}")
    sp500_df = pobierz_dane_sp500(data_start=start_date, data_end=end_date)
    print(f"Loaded sp500_df: {sp500_df.shape}")

    params_for_prep = {
        "sequence_length": cfg.data.sequence_length,
        "forecast_horizon": cfg.data.forecast_horizon,
        "scaler": cfg.data.scaler,
        "feature_selection": cfg.data.feature_selection,
        "include_sp500": cfg.data.include_sp500,
        "augment": cfg.data.augment,
        "dropna": cfg.data.dropna,
        "tft": cfg.data.tft,
    }
    X_pack, Y, scalers, selected_features, d0, d1 = prepare_data(raw_df, params_for_prep, sp500_df=sp500_df)
    meta = {"date_range": (d0, d1), "features": selected_features}
    return X_pack, Y, scalers, selected_features, meta


def run_single_experiment(exp: dict, cfg) -> dict:
    """
    Wykonuje pojedynczy eksperyment dla danego modelu.
    exp: {"symbol": str, "params": dict, "paths": dict[Path] }
    """
    symbol = exp["symbol"]
    params = dict(exp["params"])
    paths = exp["paths"]
    model_name = str(params.get("nazwa_modelu", params.get("model_type", "MODEL")))
    model_type = validate_model_name(params["model_type"]).value

    # 1) Dane
    X_pack, Y, scalers, selected_features, meta = _fetch_data(symbol, cfg)

    # 2) Train/Val split
    X_tr, Y_tr, X_val, Y_val = make_validation_split(X_pack, Y, val_fraction=0.15)

    # 3) (Opcjonalny) autotuning
    params = maybe_autotune(
        X_tr, Y_tr, params,
        autotune_cfg=exp.get("autotune") or {},
        tuner_dir=paths["run_dir"] / "tuner_dir",
    )

    # 4) Trening
    model, eff_params, history = train_model(X_tr, Y_tr, X_val, Y_val, params)

    # 5) Zapis bundla
    base_name = f"{model_name}"
    _save_bundle(model, scalers, eff_params, base_name, paths["run_dir"] / "zapisane_modele")

    # 6) Prosta ewaluacja na walidacji (np. MAE/MSE)
    y_pred_val = model.predict(_materialize_standard_X(X_val) if model_type != "TFT" else X_val, verbose=0)
    y_pred_val = np.asarray(y_pred_val)
    # ujednolicenie do (n,h)
    if y_pred_val.ndim == 1:
        y_pred_val = y_pred_val[:, None]
    elif y_pred_val.ndim > 2:
        y_pred_val = y_pred_val.reshape((y_pred_val.shape[0], -1))
    mse = float(np.mean((y_pred_val - Y_val) ** 2))
    mae = float(np.mean(np.abs(y_pred_val - Y_val)))

    LOGGER.info("Model %s: val_mse=%.6f | val_mae=%.6f", model_name, mse, mae)

    return {
        "model_name": model_name,
        "params": eff_params,
        "metrics": {"val_mse": mse, "val_mae": mae},
        "paths": paths,
    }


# ————————————————————————————————————————————————————————————————————————
# CLI
# ————————————————————————————————————————————————————————————————————————

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline treningu i ensemble dla modeli sekwencyjnych.")
    p.add_argument("--config", type=str, default="configs/config.yaml", help="Ścieżka do pliku YAML.")
    p.add_argument("--symbol", type=str, default="GOOGL", help="Ticker instrumentu (np. GOOGL).")
    p.add_argument("--loss", type=str, default="mse", help="Funkcja straty (np. mse|mae).")
    p.add_argument("--probe", type=str, default="rmse", help="Metryka sondy/probe (opisowa, do nazwy katalogu).")
    p.add_argument("--ensemble", action="store_true", help="Zbuduj prosty ensemble (średnia) z zapisanych modeli.")
    p.add_argument("--super-ensemble", action="store_true", help="Wytrenuj i użyj SuperEnsemble (uczeń-student).")
    p.add_argument("--verbosity", type=int, default=1, help="Poziom logów: 0=WARNING,1=INFO,2=DEBUG")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbosity)

    t0 = time.time()
    try:
        cfg = load_config_dynamic(args.config)
    except Exception as e:
        LOGGER.critical("Krytyczny błąd: %s", e)
        return 2

    # Przygotuj listę eksperymentów (po jednym na model)
    try:
        paths = build_outputs(args.symbol, cfg, args.loss, args.probe)
        experiments = generate_experiments(cfg, symbol=args.symbol, loss=args.loss, probe_metric=args.probe)
        # wstrzyknij sekcję autotune do każdego exp
        for e in experiments:
            e["autotune"] = cfg.autotune
            e["paths"] = paths
    except Exception as e:
        LOGGER.critical("Błąd generowania eksperymentów: %s", e)
        return 2

    results: list[dict] = []
    for exp in experiments:
        try:
            LOGGER.info("\n  ~~~ Start eksperymentu: %s ~~~", exp["params"]["model_type"])
            res = run_single_experiment(exp, cfg)
            results.append(res)
        except Exception as e:
            LOGGER.error("Eksperyment (%s) nie powiódł się: %s\n%s",
                         exp["params"]["model_type"], e, traceback.format_exc())

    # ——— Ensemble (opcjonalny) ———
    if args.ensemble or args.super_ensemble:
        # wczytaj predictorów z zapisanych bundli
        pred_list: list[Predictor] = []
        save_dir = paths["run_dir"] / "zapisane_modele"
        for r in results:
            base = r["model_name"]
            try:
                pred = Predictor.load(save_dir, base_filename=base)
                pred_list.append(pred)
            except Exception as e:
                LOGGER.warning("Pominięto model w ensemble (brak bundla?): %s", e)

        # Odśwież dane walidacyjne (bez utraty deterministyczności)
        X_pack, Y, scalers, selected_features, meta = _fetch_data(args.symbol, cfg)
        _, _, X_val, Y_val = make_validation_split(X_pack, Y, val_fraction=0.15)

        if args.ensemble and pred_list:
            y_ens = build_ensemble(pred_list, X_val, method="mean", inverse_scale=True)
            mse = float(np.mean((y_ens - Y_val) ** 2))
            mae = float(np.mean(np.abs(y_ens - Y_val)))
            LOGGER.info("ENSEMBLE(mean): val_mse=%.6f | val_mae=%.6f", mse, mae)

        if args.super_ensemble and pred_list:
            se = SuperEnsemble(base_predictors=pred_list, student_type="ridge")
            se.fit(X_val, Y_val)
            y_se = se.predict(X_val)
            mse = float(np.mean((y_se - Y_val) ** 2))
            mae = float(np.mean(np.abs(y_se - Y_val)))
            LOGGER.info("SUPER_ENSEMBLE(ridge): val_mse=%.6f | val_mae=%.6f", mse, mae)

    LOGGER.info("\n### KONIEC. Całkowity czas: %.2fs ###", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
