# src/project/autotune.py
# Automatyczne dostrajanie hiperparametrów (Optuna/KerasTuner) dla modeli Keras.

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from model_builder import build_model
from model_registry import requires_tft_input
from model_trainer import make_validation_split

LOGGER = logging.getLogger(__name__)


# ————————————————————————————————————————————————————————————————————————
# Wynik autotuningu
# ————————————————————————————————————————————————————————————————————————

@dataclass(frozen=True)
class AutoTuneResult:
    best_params: dict[str, object]
    best_score: float
    trials: list[dict[str, object]]


# ————————————————————————————————————————————————————————————————————————
# Przestrzenie wyszukiwania – wspólne dla Optuna i KerasTuner
# ————————————————————————————————————————————————————————————————————————

def _suggest_space(trial_like: Any, model_type: str) -> dict[str, Any]:
    """
    Buduje słownik hiperparametrów na podstawie „trial_like”.
    'trial_like' musi udostępniać metody:
      - suggest_int(name, low, high, step=1)
      - suggest_float(name, low, high, log=False)
      - suggest_categorical(name, choices)
    Używane zarówno przez Optuna (trial) jak i „emulację” w KerasTuner.
    """
    mt = model_type.upper()
    p: dict[str, Any] = {}

    # wspólne
    p["dropout"] = trial_like.suggest_float("dropout", 0.0, 0.5)
    p["batchnorm"] = trial_like.suggest_categorical("batchnorm", [False, True])
    p["optimizer"] = trial_like.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])
    p["lr"] = trial_like.suggest_float("lr", 1e-4, 1e-2, log=True)
    p["loss"] = trial_like.suggest_categorical("loss", ["mse", "mae", "huber"])  # huber dostępny w Keras 2.12+

    if mt in ("LSTM", "GRU"):
        p["units"] = trial_like.suggest_int("units", 16, 256, step=16)
        p["layers"] = trial_like.suggest_int("layers", 1, 3)
    elif mt == "CNN-LSTM":
        p["filters"] = trial_like.suggest_int("filters", 16, 128, step=16)
        p["units"] = trial_like.suggest_int("units", 16, 256, step=16)
    elif mt == "TRANSFORMER":
        p["d_model"] = trial_like.suggest_int("d_model", 32, 256, step=32)
        p["heads"] = trial_like.suggest_categorical("heads", [2, 4, 8])
        p["ff_dim"] = trial_like.suggest_int("ff_dim", 64, 512, step=64)
        p["blocks"] = trial_like.suggest_int("blocks", 1, 4)
    elif mt == "TFT":
        p["units"] = trial_like.suggest_int("units", 16, 256, step=16)
    else:
        raise ValueError(f"Nieznany model_type: {model_type}")

    return p


# ————————————————————————————————————————————————————————————————————————
# Ewaluacja jednego kandydata
# ————————————————————————————————————————————————————————————————————————

def _eval_candidate(
    params_base: dict[str, Any],
    model_params: dict[str, Any],
    X_tr: np.ndarray | dict[str, np.ndarray],
    Y_tr: np.ndarray,
    X_val: np.ndarray | dict[str, np.ndarray],
    Y_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
) -> tuple[float, dict[str, Any]]:
    """
    Buduje model z połączonych parametrów, trenuje kilka epok i zwraca `val_loss` oraz finalne parametry.
    """
    p = dict(params_base)  # kopia
    p.update(model_params)

    # budowa i kompilacja
    model = build_model(p["model_type"], X_tr, Y_tr.shape[1], p)

    # mini-trening (bez callbacków – intencjonalnie krótkie i stabilne)
    try:
        if requires_tft_input(p["model_type"]):
            history = model.fit(
                X_tr, Y_tr, validation_data=(X_val, Y_val),
                epochs=epochs, batch_size=batch_size,
                shuffle=False, verbose=0,
            )
        else:
            history = model.fit(
                X_tr, Y_tr, validation_data=(X_val, Y_val),
                epochs=epochs, batch_size=batch_size,
                shuffle=True, verbose=0,
            )
        hist = history.history or {}
        val_losses = hist.get("val_loss") or []
        score = float(val_losses[-1]) if val_losses else float("inf")
    except Exception as e:
        LOGGER.warning("Błąd w treningu kandydata (%s): %s", p.get("model_type"), e)
        score = float("inf")

    return score, p


# ————————————————————————————————————————————————————————————————————————
# Główne API: autotune_keras
# ————————————————————————————————————————————————————————————————————————

def autotune_keras(
    X: np.ndarray | dict[str, np.ndarray],
    Y: np.ndarray,
    *,
    base_params: dict[str, object],
    X_val: np.ndarray | dict[str, np.ndarray] | None = None,
    Y_val: np.ndarray | None = None,
    tuner: str = "optuna",          # "optuna" | "kerastuner"
    n_trials: int = 20,
    timeout: int | None = None,
    direction: str = "minimize",
    val_fraction: float = 0.15,
    epochs: int = 8,
    batch_size: int = 64,
) -> AutoTuneResult:
    """
    Automatyczny tuning hiperparametrów dla modeli Keras.

    - Jeśli `X_val`, `Y_val` nie są podane, dzieli dane na train/val przez `make_validation_split`.
    - W razie braku żądanego „silnika” (np. Optuna nie zainstalowana) spróbuje alternatywy,
      a jeśli żadna nie jest dostępna – wykona **pseudo-tuning** (1 trial = parametry bazowe).
    """
    if "model_type" not in base_params:
        raise ValueError("base_params['model_type'] jest wymagane.")
    if not isinstance(Y, np.ndarray) or Y.ndim != 2:
        raise ValueError("Y musi mieć shape (n, horizon) 2D.")

    # Podział na train/val – jeśli potrzeba
    if X_val is None or Y_val is None:
        X_tr, Y_tr, X_va, Y_va = make_validation_split(X, Y, val_fraction=val_fraction)
    else:
        X_tr, Y_tr, X_va, Y_va = X, Y, X_val, Y_val

    model_type = str(base_params["model_type"]).upper()

    trials_log: List[dict[str, Any]] = []
    best_score = float("inf") if direction == "minimize" else -float("inf")
    best_params: dict[str, Any] = dict(base_params)

    start_time = time.time()

    def _maybe_update_best(score: float, params: dict[str, Any]) -> None:
        nonlocal best_score, best_params
        better = score < best_score if direction == "minimize" else score > best_score
        if better:
            best_score = score
            best_params = dict(params)

    # ——— ścieżka: OPTUNA ———
    if tuner.lower() == "optuna":
        try:
            import optuna  # type: ignore

            def objective(trial: "optuna.trial.Trial") -> float:
                model_params = _suggest_space(trial, model_type)
                score, eff = _eval_candidate(
                    base_params, model_params, X_tr, Y_tr, X_va, Y_va,
                    epochs=epochs, batch_size=batch_size
                )
                trials_log.append({"score": score, "params": eff})
                _maybe_update_best(score, eff)
                return score

            study = optuna.create_study(direction="minimize" if direction == "minimize" else "maximize")
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            # Optuna już zaktualizowała best_* w trakcie objective
            LOGGER.info("Optuna: zakończono tuning. Best score=%.6f", best_score)
            return AutoTuneResult(best_params=best_params, best_score=best_score, trials=trials_log)

        except Exception as e:
            LOGGER.warning("Optuna niedostępna lub błąd (%s). Próba KerasTuner.", e)

    # ——— ścieżka: KERASTUNER ———
    if tuner.lower() == "kerastuner":
        try:
            import keras_tuner as kt  # type: ignore

            class _AdapterHP:
                """Adapter HyperParameters → trial_like API."""
                def __init__(self, hp: "kt.HyperParameters") -> None:
                    self.hp = hp
                def suggest_int(self, name, low, high, step=1):
                    return self.hp.Int(name, min_value=low, max_value=high, step=step)
                def suggest_float(self, name, low, high, log=False):
                    if log:
                        return self.hp.Float(name, min_value=low, max_value=high, sampling="log")
                    return self.hp.Float(name, min_value=low, max_value=high)
                def suggest_categorical(self, name, choices):
                    return self.hp.Choice(name, values=list(choices))

            def _build_from_hp(hp: "kt.HyperParameters"):
                model_params = _suggest_space(_AdapterHP(hp), model_type)
                params = dict(base_params)
                params.update(model_params)
                model = build_model(params["model_type"], X_tr, Y_tr.shape[1], params)
                model.fit(
                    X_tr, Y_tr, validation_data=(X_va, Y_va),
                    epochs=epochs, batch_size=batch_size,
                    shuffle=not requires_tft_input(params["model_type"]),
                    verbose=0,
                )
                return model

            tuner_obj = kt.RandomSearch(
                _build_from_hp,
                objective="val_loss" if direction == "minimize" else kt.Objective("val_loss", direction="max"),
                max_trials=n_trials,
                overwrite=True,
                executions_per_trial=1,
                directory=None,  # w pamięci
                project_name="autotune",
            )
            tuner_obj.search(X_tr, Y_tr, validation_data=(X_va, Y_va), verbose=0)

            # wyciągnij najlepsze hp i wynik
            best_hp = tuner_obj.get_best_hyperparameters(1)[0]
            # zbuduj finalny słownik parametrów
            best_model_params = _suggest_space(_AdapterHP(best_hp), model_type)
            best_params = dict(base_params)
            best_params.update(best_model_params)
            # raport z prób
            for t in tuner_obj.oracle.trials.values():
                trials_log.append({"score": float(t.score) if t.score is not None else float("inf"),
                                   "params": dict(base_params) | {k: t.hyperparameters.values.get(k) for k in t.hyperparameters.values}})
            # wynik
            # KerasTuner może nie mieć jednoznacznego „best_score” przed retreningiem; przyjmij najniższy ze zebranych
            best_score = min([x["score"] for x in trials_log]) if trials_log else float("inf")
            LOGGER.info("KerasTuner: zakończono tuning. Best score=%.6f", best_score)
            return AutoTuneResult(best_params=best_params, best_score=best_score, trials=trials_log)

        except Exception as e:
            LOGGER.warning("KerasTuner niedostępny lub błąd (%s). Spadam do pseudo-tuningu.", e)

    # ——— fallback: pseudo-tuning (1 trial z parametrami bazowymi) ———
    LOGGER.warning("Brak dostępnych narzędzi tuningu. Wykonuję pojedynczą ewaluację bazowych parametrów.")
    score, eff = _eval_candidate(base_params, {}, X_tr, Y_tr, X_va, Y_va, epochs=epochs, batch_size=batch_size)
    trials_log.append({"score": score, "params": eff})
    best_params = eff
    best_score = score
    return AutoTuneResult(best_params=best_params, best_score=best_score, trials=trials_log)
