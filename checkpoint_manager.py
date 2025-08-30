# src/project/checkpoint_manager.py
# Bezpieczne zarządzanie checkpointami: atomowe zapisy, metadane, listowanie.

from __future__ import annotations

import json
import logging
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)

_DEFAULT_SUFFIX = ".pkl"
_META_SUFFIX = ".json"


class CheckpointError(RuntimeError):
    """Błąd ogólny operacji na checkpointach."""


class CheckpointManager:
    """
    Zarządza zapisem/odczytem checkpointów w katalogu `root_dir`.

    - Zapisy atomowe: zapis do pliku tymczasowego + `replace` -> brak częściowych plików.
    - Metadane: opcjonalny plik JSON obok `.pkl` (np. typ, wersja, timestamp, własne pola).
    - Wzorce: nazwy dowolne, np. `fold_0_prepared_data`, `tuning_trial_12`.

    Przykład:
        >>> from pathlib import Path
        >>> cm = CheckpointManager(Path("wyniki/checkpoints"))
        >>> cm.save("fold_0_prepared_data", {"x": 1})
        PosixPath('wyniki/checkpoints/fold_0_prepared_data.pkl')
        >>> cm.exists("fold_0_prepared_data")
        True
        >>> data = cm.load("fold_0_prepared_data")
        >>> data["x"]
        1
    """

    def __init__(self, root_dir: Path | str, create: bool = True) -> None:
        self.root_dir = Path(root_dir)
        if create:
            self.root_dir.mkdir(parents=True, exist_ok=True)
        if not self.root_dir.exists():
            raise CheckpointError(f"Katalog checkpointów nie istnieje: {self.root_dir}")
        LOGGER.info("CheckpointManager zainicjalizowany w katalogu: %s", self.root_dir.as_posix())

    # ---------- Ścieżki i walidacja ----------

    def _sanitize_name(self, name: str) -> str:
        """Prosta walidacja nazwy (bez separatorów katalogów)."""
        if not name or any(sep in name for sep in ("/", "\\", "..")):
            raise ValueError(f"Nieprawidłowa nazwa checkpointu: {name!r}")
        return name

    def _paths_for(self, name: str, suffix: str = _DEFAULT_SUFFIX) -> tuple[Path, Path]:
        """Zwraca ścieżkę docelową i tymczasową dla bezpiecznego zapisu."""
        base = self.root_dir / f"{name}{suffix}"
        tmp = self.root_dir / f".{name}{suffix}.tmp"
        return base, tmp

    # ---------- Operacje publiczne ----------

    def save(self, name: str, obj: Any, *, metadata: dict[str, Any] | None = None) -> Path:
        """
        Zapisuje obiekt jako pickle (`.pkl`) + opcjonalnie metadane (`.json`) obok.

        :param name: alias bez rozszerzenia, np. 'fold_0_prepared_data'
        :param obj: dowolny obiekt picklowalny
        :param metadata: dodatkowe metadane do zapisania jako JSON
        :return: ścieżka do finalnego pliku `.pkl`
        """
        name = self._sanitize_name(name)
        dst, tmp = self._paths_for(name, _DEFAULT_SUFFIX)

        # atomowy zapis pickle
        try:
            with tmp.open("wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(dst)
        finally:
            if tmp.exists():
                # w wypadku wyjątków spróbuj posprzątać tmp
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass

        # metadane (nie krytyczne dla powodzenia)
        if metadata is not None:
            mpath, mtmp = self._paths_for(name, _META_SUFFIX)
            enriched = {
                "name": name,
                "path": str(dst),
                "saved_at": int(time.time()),
                **metadata,
            }
            try:
                with mtmp.open("w", encoding="utf-8") as f:
                    json.dump(enriched, f, ensure_ascii=False, indent=2)
                mtmp.replace(mpath)
            finally:
                if mtmp.exists():
                    try:
                        mtmp.unlink(missing_ok=True)
                    except Exception:
                        pass

        LOGGER.info("✅ Zapisano checkpoint: %s", dst.as_posix())
        return dst

    def load(self, name: str) -> Any:
        """
        Wczytuje obiekt z `.pkl`. Rzuca CheckpointError jeśli brak pliku.

        :param name: alias bez rozszerzenia
        """
        name = self._sanitize_name(name)
        path = self.root_dir / f"{name}{_DEFAULT_SUFFIX}"
        if not path.exists():
            raise CheckpointError(f"Checkpoint nie istnieje: {path}")
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise CheckpointError(f"Nie udało się wczytać checkpointu: {path} -> {e}") from e

    def exists(self, name: str) -> bool:
        """Czy istnieje plik `.pkl` o danej nazwie."""
        try:
            name = self._sanitize_name(name)
        except ValueError:
            return False
        return (self.root_dir / f"{name}{_DEFAULT_SUFFIX}").exists()

    def delete(self, name: str) -> bool:
        """Usuwa `.pkl` i `.json` jeśli istnieją. Zwraca True jeśli coś usunięto."""
        name = self._sanitize_name(name)
        removed = False
        for suffix in (_DEFAULT_SUFFIX, _META_SUFFIX):
            p = self.root_dir / f"{name}{suffix}"
            if p.exists():
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                    removed = True
                except Exception as e:
                    LOGGER.warning("Nie udało się usunąć %s: %s", p, e)
        if removed:
            LOGGER.info("🗑️  Usunięto checkpoint '%s'", name)
        return removed

    def list(self, prefix: str | None = None, suffix: str = _DEFAULT_SUFFIX) -> list[Path]:
        """Zwraca listę ścieżek checkpointów (domyślnie `.pkl`), opcjonalnie filtrując po prefiksie."""
        out: list[Path] = []
        for p in sorted(self.root_dir.glob(f"*{suffix}")):
            if prefix is None or p.name.startswith(prefix):
                out.append(p)
        return out

    def latest(self, prefix: str | None = None) -> Path | None:
        """
        Zwraca najnowszy checkpoint (po czasie modyfikacji) o danym prefiksie,
        lub None jeśli brak.
        """
        files = self.list(prefix=prefix, suffix=_DEFAULT_SUFFIX)
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)
