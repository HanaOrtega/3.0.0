# src/project/model_saver_loader.py
# Zapis i wczytywanie modeli (Keras/sklearn), scalerów, parametrów i metadanych.

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

import joblib

LOGGER = logging.getLogger(__name__)

# ————————————————————————————————————————————————————————————————————————
# Pomocnicze wyjątki
# ————————————————————————————————————————————————————————————————————————

class ModelIOError(RuntimeError):
    """Błąd operacji I/O modelu (zapis/wczytanie)."""


# ————————————————————————————————————————————————————————————————————————
# Wersje i metadane
# ————————————————————————————————————————————————————————————————————————

def _pkg_version(pkg_name: str) -> str | None:
    try:
        import importlib.metadata as im
        return im.version(pkg_name)
    except Exception:
        return None


def _gather_meta(model_type: str, base_filename: str, dest_dir: Path) -> dict[str, Any]:
    return {
        "model_type": model_type,
        "base_filename": base_filename,
        "saved_at_unix": int(time.time()),
        "versions": {
            "tensorflow": _pkg_version("tensorflow"),
            "keras": _pkg_version("keras"),
            "scikit-learn": _pkg_version("scikit-learn"),
            "joblib": _pkg_version("joblib"),
            "numpy": _pkg_version("numpy"),
        },
        "files_dir": dest_dir.as_posix(),
    }


# ————————————————————————————————————————————————————————————————————————
# Detekcja typu modelu i ścieżek
# ————————————————————————————————————————————————————————————————————————

def _is_keras_model(obj: Any) -> bool:
    try:
        from tensorflow import keras  # type: ignore
    except Exception:
        return False
    return isinstance(obj, keras.Model)


def _is_sklearn_estimator(obj: Any) -> bool:
    try:
        from sklearn.base import BaseEstimator  # type: ignore
    except Exception:
        return False
    return isinstance(obj, BaseEstimator)


def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass


# ————————————————————————————————————————————————————————————————————————
# Public API: Zapis
# ————————————————————————————————————————————————————————————————————————

def save_model_bundle(
    model: Any,
    scalers: dict[str, Any] | None,
    params: dict[str, Any] | None,
    base_filename: str,
    save_dir: str | Path,
    *,
    keras_format: Literal["keras", "h5", None] = None,
) -> bool:
    """
    Zapisuje model i artefakty w katalogu `save_dir/base_filename/`.

    - Keras:  `model.keras` (domyślnie) lub `model.h5`
    - Sklearn: `model.joblib`
    - Scalery: `scalers.joblib` (opcjonalnie)
    - Parametry: `params.json` (opcjonalnie)
    - Metadane: `meta.json` (z wersjami pakietów)

    Zwraca True gdy zapis się powiódł, False w przeciwnym razie.

    Wyjątki niekrytyczne (np. brak scalerów) są logowane jako WARNING.
    """

    if not base_filename or any(sep in base_filename for sep in ("/", "\\", "..")):
        raise ValueError(f"Nieprawidłowa nazwa bazowa: {base_filename!r}")

    dest_dir = Path(save_dir) / base_filename
    _ensure_dir(dest_dir)

    ok = True
    model_type: str

    try:
        if _is_keras_model(model):
            model_type = "keras"
            from tensorflow import keras as _keras  # lazy import

            # wybór formatu
            fmt = keras_format or "keras"
            if fmt not in {"keras", "h5"}:
                raise ValueError("keras_format musi być jednym z: 'keras', 'h5'")

            out_path = dest_dir / ("model.keras" if fmt == "keras" else "model.h5")
            LOGGER.info("Zapisuję model Keras → %s", out_path.name)
            # Uwaga: tf/keras zapisuje atomowo do pliku tymczasowego wewnętrznie
            model.save(out_path)

        elif _is_sklearn_estimator(model):
            model_type = "sklearn"
            out_path = dest_dir / "model.joblib"
            LOGGER.info("Zapisuję model sklearn → %s", out_path.name)
            joblib.dump(model, out_path)
        else:
            raise ModelIOError(
                f"Nieobsługiwany typ modelu: {type(model)!r}. Oczekiwano Keras lub sklearn."
            )

        # scalers (opcjonalnie)
        if scalers:
            sc_path = dest_dir / "scalers.joblib"
            LOGGER.info("Zapisuję scalery → %s", sc_path.name)
            joblib.dump(scalers, sc_path)
        else:
            LOGGER.warning("Brak scalerów do zapisania (pomijam).")

        # params (opcjonalnie)
        if params:
            prm_path = dest_dir / "params.json"
            LOGGER.info("Zapisuję parametry → %s", prm_path.name)
            _atomic_write_json(prm_path, params)
        else:
            LOGGER.warning("Brak parametrów do zapisania (pomijam).")

        # meta (zawsze)
        meta_path = dest_dir / "meta.json"
        meta = _gather_meta(model_type=model_type, base_filename=base_filename, dest_dir=dest_dir)
        _atomic_write_json(meta_path, meta)
        LOGGER.info("✅ Bundle zapisany w: %s", dest_dir.as_posix())
        return ok

    except Exception as e:
        LOGGER.exception("❌ Nie udało się zapisać bundle: %s", e)
        return False


# ————————————————————————————————————————————————————————————————————————
# Public API: Odczyt
# ————————————————————————————————————————————————————————————————————————

def _detect_base_dir(base_filename: str | None, load_dir: Path) -> Path | None:
    if base_filename:
        base = load_dir / base_filename
        return base if base.exists() else None

    # autodetekcja: wybierz pierwszy katalog z plikiem modelu
    candidates: list[Path] = []
    for p in sorted(load_dir.iterdir()):
        if not p.is_dir():
            continue
        if any((p / fn).exists() for fn in ("model.keras", "model.h5", "model.joblib")):
            candidates.append(p)
    return candidates[0] if candidates else None


def load_model_bundle(
    base_filename: str | None,
    load_dir: str | Path,
    *,
    compile_keras: bool = True,
    is_keras_type: bool = True,
    is_sklearn_type: bool = True,
) -> tuple[Any | None, dict[str, Any] | None, dict[str, Any] | None]:
    """
    Wczytuje model + scalery + parametry z `load_dir/base_filename/` lub autodetekcji.

    :param base_filename: nazwa bazowa katalogu; jeśli None → autodetekcja w `load_dir`
    :param load_dir: katalog nadrzędny zawierający modele
    :param compile_keras: czy kompilować model Keras przy wczytywaniu
    :param is_keras_type: zezwól na szukanie formatu Keras
    :param is_sklearn_type: zezwól na szukanie formatu sklearn
    :return: (model | None, scalers | None, params | None)
    """
    root = Path(load_dir)
    base_dir = _detect_base_dir(base_filename, root)
    if base_dir is None:
        LOGGER.error("Nie znaleziono katalogu bundle w %s (base=%r)", root.as_posix(), base_filename)
        return None, None, None

    model: Any | None = None
    scalers: dict[str, Any] | None = None
    params: dict[str, Any] | None = None

    # 1) Model
    try:
        if is_keras_type and (base_dir / "model.keras").exists() or (base_dir / "model.h5").exists():
            from tensorflow import keras as _keras  # lazy import
            keras_path = base_dir / ("model.keras" if (base_dir / "model.keras").exists() else "model.h5")
            LOGGER.info("Wczytuję model Keras z %s", keras_path.name)
            model = _keras.models.load_model(keras_path, compile=compile_keras)
        elif is_sklearn_type and (base_dir / "model.joblib").exists():
            skl_path = base_dir / "model.joblib"
            LOGGER.info("Wczytuję model sklearn z %s", skl_path.name)
            model = joblib.load(skl_path)
        else:
            LOGGER.warning("Nie znaleziono pliku modelu w %s", base_dir.as_posix())
    except Exception as e:
        LOGGER.exception("Błąd wczytywania modelu: %s", e)

    # 2) Scalery (opcjonalnie)
    try:
        sc_path = base_dir / "scalers.joblib"
        if sc_path.exists():
            scalers = joblib.load(sc_path)
    except Exception as e:
        LOGGER.warning("Błąd wczytywania scalerów: %s", e)

    # 3) Parametry (opcjonalnie)
    try:
        prm_path = base_dir / "params.json"
        if prm_path.exists():
            params = json.loads(prm_path.read_text(encoding="utf-8"))
    except Exception as e:
        LOGGER.warning("Błąd wczytywania parametrów: %s", e)

    return model, scalers, params
