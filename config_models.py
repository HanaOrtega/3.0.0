# src/project/config_models.py
# Obsługa konfiguracji: wczytywanie YAML, walidacja, generowanie eksperymentów.

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from model_registry import ModelType, validate_model_name

LOGGER = logging.getLogger(__name__)


# ————————————————————————————————————————————————————————————————————————
# Dataclasses konfiguracji
# ————————————————————————————————————————————————————————————————————————

@dataclass(frozen=True)
class ModelSpec:
    """Opis pojedynczego modelu do uruchomienia."""
    name: str                          # np. "LSTM"
    params: dict[str, Any] = field(default_factory=dict)
    enable: bool = True                # pozwala wyłączyć dany model bez usuwania z pliku


@dataclass(frozen=True)
class TrainerSpec:
    """Ustawienia treningu wspólne dla modeli."""
    epochs: int = 20
    batch_size: int = 64
    patience: int = 5
    min_delta: float = 1e-4
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    shuffle: bool = True
    checkpoint_dir: str | None = None


@dataclass(frozen=True)
class DataSpec:
    """Ustawienia danych oraz przygotowania cech."""
    sequence_length: int = 30
    forecast_horizon: int = 14
    scaler: str = "minmax"
    feature_selection: dict[str, Any] = field(default_factory=dict)  # {"method": "rfe", "top_k": 30}
    include_sp500: bool = True
    augment: dict[str, Any] = field(default_factory=dict)            # {"noise_level": 0.0,"times": 0}
    dropna: bool = True
    tft: dict[str, Any] = field(default_factory=lambda: {"enable": True})


@dataclass(frozen=True)
class Config:
    """Główny obiekt konfiguracji."""
    output_dir: Path
    cache_dir: Path
    data: DataSpec
    trainer: TrainerSpec
    models: list[ModelSpec]
    autotune: dict[str, Any] = field(default_factory=dict)  # {"enabled": false, "engine": "optuna", "n_trials": 20}

    def enabled_models(self) -> list[ModelSpec]:
        return [m for m in self.models if m.enable]


# ————————————————————————————————————————————————————————————————————————
# Wczytywanie i walidacja YAML
# ————————————————————————————————————————————————————————————————————————

def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Plik konfiguracyjny '{path.as_posix()}' nie został znaleziony!")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Plik konfiguracyjny musi zawierać słownik na poziomie root.")
    return data


def _coalesce_path(value: str | None, default_rel: str, base: Path) -> Path:
    if value:
        p = Path(os.path.expandvars(os.path.expanduser(value)))
        return p
    return (base / default_rel).resolve()


def _parse_models(raw_models: Iterable[dict[str, Any] | str]) -> list[ModelSpec]:
    out: list[ModelSpec] = []
    for item in raw_models or []:
        if isinstance(item, str):
            mt = validate_model_name(item)
            out.append(ModelSpec(name=mt.value))
        elif isinstance(item, dict):
            name = item.get("name") or item.get("type") or item.get("model") or ""
            mt = validate_model_name(str(name))
            params = dict(item.get("params", {}))
            enable = bool(item.get("enable", True))
            out.append(ModelSpec(name=mt.value, params=params, enable=enable))
        else:
            raise ValueError(f"Nieprawidłowa definicja modelu: {item!r}")
    if not out:
        # domyślny zestaw
        out = [ModelSpec(name=mt.value) for mt in (ModelType.TRANSFORMER, ModelType.GRU, ModelType.LSTM, ModelType.CNN_LSTM, ModelType.TFT)]
    return out


def _parse_data(raw: dict[str, Any] | None) -> DataSpec:
    raw = raw or {}
    return DataSpec(
        sequence_length=int(raw.get("sequence_length", 30)),
        forecast_horizon=int(raw.get("forecast_horizon", 14)),
        scaler=str(raw.get("scaler", "minmax")),
        feature_selection=dict(raw.get("feature_selection", {})),
        include_sp500=bool(raw.get("include_sp500", True)),
        augment=dict(raw.get("augment", {})),
        dropna=bool(raw.get("dropna", True)),
        tft=dict(raw.get("tft", {"enable": True})),
    )


def _parse_trainer(raw: dict[str, Any] | None) -> TrainerSpec:
    raw = raw or {}
    return TrainerSpec(
        epochs=int(raw.get("epochs", 20)),
        batch_size=int(raw.get("batch_size", 64)),
        patience=int(raw.get("patience", 5)),
        min_delta=float(raw.get("min_delta", 1e-4)),
        reduce_lr_patience=int(raw.get("reduce_lr_patience", 3)),
        reduce_lr_factor=float(raw.get("reduce_lr_factor", 0.5)),
        shuffle=bool(raw.get("shuffle", True)),
        checkpoint_dir=raw.get("checkpoint_dir"),
    )


def load_and_validate_config(path_like: str | os.PathLike[str]) -> Config:
    """
    Wczytuje YAML, waliduje podstawowe pola i zwraca `Config`.
    Minimalny YAML:
    ---
    output_dir: wyniki_dla_symboli
    cache_dir: cache
    models: ["LSTM", "GRU"]
    """
    path = Path(path_like)
    data = _read_yaml(path)

    base = path.parent.resolve()
    out_dir = _coalesce_path(data.get("output_dir"), "wyniki_dla_symboli", base)
    cache_dir = _coalesce_path(data.get("cache_dir"), "cache", base)

    models = _parse_models(data.get("models", []))
    data_spec = _parse_data(data.get("data"))
    trainer_spec = _parse_trainer(data.get("trainer"))
    autotune_raw = dict(data.get("autotune", {}))

    cfg = Config(
        output_dir=out_dir,
        cache_dir=cache_dir,
        data=data_spec,
        trainer=trainer_spec,
        models=models,
        autotune=autotune_raw,
    )

    # sanity: katalogi
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    if cfg.trainer.checkpoint_dir:
        Path(cfg.trainer.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    LOGGER.info("Konfiguracja wczytana. Modele aktywne: %s", [m.name for m in cfg.enabled_models()])
    return cfg


# ————————————————————————————————————————————————————————————————————————
# Generowanie eksperymentów
# ————————————————————————————————————————————————————————————————————————

def resolve_output_paths(base_dir: str | Path, symbol: str, loss: str, probe_metric: str) -> dict[str, Path]:
    """
    Buduje i tworzy folder wyjściowy: `<base>/<SYMBOL>/<loss_xxx-probe_yyy>/`.
    """
    base_dir = Path(base_dir)
    symbol_dir = base_dir / symbol
    run_name = f"loss_{loss}-probe_{probe_metric}"
    run_dir = symbol_dir / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return {
        "symbol_dir": symbol_dir,
        "run_dir": run_dir,
        "checkpoints": run_dir / "checkpoints",
    }


def _merge_params(global_data: DataSpec, trainer: TrainerSpec, model: ModelSpec) -> dict[str, Any]:
    """
    Tworzy słownik parametrów przekazywany dalej do pipeline'u.
    """
    p: dict[str, Any] = {
        # to, co konsumuje data_preprocessing
        "sequence_length": global_data.sequence_length,
        "forecast_horizon": global_data.forecast_horizon,
        "scaler": global_data.scaler,
        "feature_selection": global_data.feature_selection,
        "include_sp500": global_data.include_sp500,
        "augment": global_data.augment,
        "dropna": global_data.dropna,
        "tft": global_data.tft,
        # to, co konsumują trenujący/builder
        "model_type": model.name,
        "trainer": {
            "epochs": trainer.epochs,
            "batch_size": trainer.batch_size,
            "patience": trainer.patience,
            "min_delta": trainer.min_delta,
            "reduce_lr_patience": trainer.reduce_lr_patience,
            "reduce_lr_factor": trainer.reduce_lr_factor,
            "shuffle": trainer.shuffle,
            "checkpoint_dir": trainer.checkpoint_dir,
        },
    }
    # parametry specyficzne modelu nadpisują domyślne
    p.update(model.params or {})
    return p


def generate_experiments(
    cfg: Config,
    *,
    symbol: str,
    loss: str,
    probe_metric: str,
) -> list[dict[str, Any]]:
    """
    Zwraca listę eksperymentów (słowników parametrów) – po jednym na każdy aktywny model.

    Każdy element zawiera: `params`, `output_paths`, `symbol`, `loss`, `probe_metric`.
    """
    out: list[dict[str, Any]] = []
    paths = resolve_output_paths(cfg.output_dir, symbol, loss, probe_metric)

    for model in cfg.enabled_models():
        params = _merge_params(cfg.data, cfg.trainer, model)
        params["loss"] = loss
        params["probe_metric"] = probe_metric
        params["nazwa_modelu"] = model.name
        out.append(
            {
                "symbol": symbol,
                "params": params,
                "paths": paths,
            }
        )

    if not out:
        raise ValueError("Brak aktywnych modeli do uruchomienia.")
    return out
