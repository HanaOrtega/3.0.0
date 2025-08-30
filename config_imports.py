# src/project/config_imports.py
"""Konfiguracja środowiska i zależności projektu ML."""

from __future__ import annotations

import logging
import os
import random
import warnings
from typing import Optional

LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Skonfiguruj logowanie globalne."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    LOGGER.info("--- Inicjalizacja modułu config_imports ---")


def configure_warnings() -> None:
    """Skonfiguruj filtrowanie ostrzeżeń (ignoruj FutureWarning)."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    LOGGER.info("Ostrzeżenia skonfigurowane: FutureWarning ignorowany.")


def configure_gpu() -> None:
    """Próba konfiguracji GPU dla TensorFlow."""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            LOGGER.info(
                "Konfiguracja GPU zakończona: Fizyczne=%d, Logiczne=%d",
                len(gpus),
                len(logical_gpus),
            )
        else:
            LOGGER.warning("Nie znaleziono GPU. Używam CPU.")
    except ImportError:
        LOGGER.warning("TensorFlow nie jest zainstalowany, GPU nie skonfigurowano.")
    except Exception as e:
        LOGGER.error("Błąd konfiguracji GPU: %s", e)


def set_global_seed(seed: int = 42) -> None:
    """Ustaw globalny seed dla powtarzalności."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass
    LOGGER.info("Ustawiono globalny seed na wartość %d", seed)


def initialize_environment(seed: int = 42, log_level: int = logging.INFO) -> None:
    """Pełna inicjalizacja środowiska (logi, ostrzeżenia, GPU, seed)."""
    configure_logging(log_level)
    configure_warnings()
    configure_gpu()
    set_global_seed(seed)
    LOGGER.info("--- Moduł config_imports załadowany pomyślnie ---")
