"""Bramka jakości danych przed trenowaniem/backtestem (uproszczona wersja
quality_dataset_policy.py / cv_requirements.py z JuggleLab: tam to rozbudowany
system raportowania, tu - kilka twardych/miękkich sprawdzeń, żeby nie trenować
po cichu na danych zbyt krótkich, nieaktualnych albo dziurawych)."""

from dataclasses import dataclass, field

import pandas as pd


class DataQualityError(RuntimeError):
    pass


@dataclass
class QualityReport:
    warnings: list = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def print_warnings(self) -> None:
        for message in self.warnings:
            print(f"UWAGA (jakość danych): {message}")


def validate_ohlcv(
    df: pd.DataFrame,
    min_rows: int = 100,
    max_staleness_days: int = 7,
    gap_ratio_threshold: float = 10.0,
) -> QualityReport:
    """Sprawdza podstawową jakość danych OHLCV przed trenowaniem modelu.

    Rzuca DataQualityError, jeśli danych jest stanowczo za mało do sensownego
    treningu/walidacji (twardy próg). Pozostałe problemy (świeżość, dziury w
    danych) trafiają jako ostrzeżenia do QualityReport - trening może się
    odbyć, ale użytkownik powinien o nich wiedzieć.
    """
    report = QualityReport()

    if len(df) < min_rows:
        raise DataQualityError(
            f"Za mało danych do wiarygodnego treningu/walidacji: {len(df)} świec "
            f"(wymagane min. {min_rows}). Zwiększ --period albo wybierz płynniejszy ticker."
        )

    last_date = df.index[-1]
    now = pd.Timestamp.now(tz=last_date.tz)
    staleness_days = (now - last_date).days
    if staleness_days > max_staleness_days:
        report.warn(
            f"Ostatnia świeca pochodzi sprzed {staleness_days} dni ({last_date.date()}) - "
            "ticker może być nieaktywny/wycofany z giełdy, albo dane są nieaktualne."
        )

    gaps = df.index.to_series().diff().dropna()
    if len(gaps) > 5:
        median_gap = gaps.median()
        max_gap = gaps.max()
        if median_gap.total_seconds() > 0 and max_gap / median_gap > gap_ratio_threshold:
            report.warn(
                f"Wykryto nietypowo dużą przerwę w danych ({max_gap} vs typowe {median_gap}) - "
                "sprawdź, czy w historii nie brakuje fragmentu (np. zawieszenie notowań)."
            )

    return report
