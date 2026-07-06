"""Automatyczna selekcja cech, wzorowana na `data/preprocessing_modules/custom_features.py`
z projektu JuggleLab - z jedną świadomą różnicą: tam filtr obejmował też korelację
cech z targetem (etykietą), co jest formą przecieku danych (dobór cech na
podstawie tego, co dopiero mamy przewidzieć, myląco podnosi ocenę modelu w
walidacji krzyżowej). Tutaj filtr patrzy WYŁĄCZNIE na same cechy (wariancja
i wzajemna korelacja) - nie dotyka etykiety, więc jest w pełni bezpieczny
względem przecieku, spójnie z resztą podejścia w tym projekcie (purged CV,
embargo)."""

import numpy as np
import pandas as pd


def select_features(
    X: pd.DataFrame,
    variance_threshold: float = 1e-8,
    corr_threshold: float = 0.95,
) -> list:
    """Zwraca listę kolumn po odrzuceniu cech prawie stałych (zerowa wariancja)
    i nadmiarowych (silnie skorelowanych z inną, już zachowaną cechą)."""
    variances = X.var()
    keep = variances[variances > variance_threshold].index.tolist()
    if len(keep) <= 1:
        return keep

    corr = X[keep].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    to_drop = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        correlated = upper.index[upper[col] > corr_threshold].tolist()
        to_drop.update(c for c in correlated if c not in to_drop)

    return [c for c in keep if c not in to_drop]
