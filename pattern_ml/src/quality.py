"""Bramka jakości danych przed trenowaniem/backtestem (uproszczona wersja
quality_dataset_policy.py / cv_requirements.py z JuggleLab: tam to rozbudowany
system raportowania z artefaktami JSON/Excel, tu - lekki deklaratywny silnik
reguł nad kilkoma policzonymi metrykami, żeby nie trenować po cichu na danych
zbyt krótkich, nieaktualnych albo dziurawych).

Deklaratywność (`MetricRule` + `check_rules`) ułatwia dopisywanie nowych,
prostych sprawdzeń bez rozrastania `validate_ohlcv` o kolejne ify - ale bez
pełnego rozbudowanego "silnika reguł" z JuggleLab, bo przy skali tego projektu
(jeden użytkownik, kilka reguł) taka złożoność nie miałaby uzasadnienia.
"""

import operator
from dataclasses import dataclass, field

import pandas as pd

_OPS = {"lt": operator.lt, "lte": operator.le, "gt": operator.gt, "gte": operator.ge}


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


@dataclass
class MetricRule:
    """Deklaratywna reguła: ostrzeż, jeśli `metric` (klucz w słowniku metryk)
    spełnia `op` względem `threshold`. `message` może użyć `{value}` do
    wstawienia zmierzonej wartości."""
    metric: str
    op: str  # "lt", "lte", "gt", "gte"
    threshold: float
    message: str


def check_rules(metrics: dict, rules: list) -> list:
    """Zwraca listę komunikatów ostrzeżeń dla reguł, które się spełniły."""
    warnings = []
    for rule in rules:
        value = metrics.get(rule.metric)
        if value is None:
            continue
        if _OPS[rule.op](value, rule.threshold):
            warnings.append(rule.message.format(value=value))
    return warnings


def _compute_metrics(df: pd.DataFrame) -> dict:
    last_date = df.index[-1]
    now = pd.Timestamp.now(tz=last_date.tz)
    staleness_days = (now - last_date).days

    gaps = df.index.to_series().diff().dropna()
    gap_ratio = 0.0
    if len(gaps) > 5:
        median_gap = gaps.median()
        if median_gap.total_seconds() > 0:
            gap_ratio = gaps.max() / median_gap

    return {"staleness_days": staleness_days, "gap_ratio": gap_ratio, "n_rows": len(df)}


def default_rules(max_staleness_days: int = 7, gap_ratio_threshold: float = 10.0) -> list:
    return [
        MetricRule(
            metric="staleness_days", op="gt", threshold=max_staleness_days,
            message=(
                f"Ostatnia świeca pochodzi sprzed {{value}} dni - ticker może być "
                "nieaktywny/wycofany z giełdy, albo dane są nieaktualne."
            ),
        ),
        MetricRule(
            metric="gap_ratio", op="gt", threshold=gap_ratio_threshold,
            message=(
                "Wykryto nietypowo dużą przerwę w danych (stosunek do typowego "
                "odstępu: {value:.1f}x) - sprawdź, czy w historii nie brakuje "
                "fragmentu (np. zawieszenie notowań)."
            ),
        ),
    ]


def validate_ohlcv(
    df: pd.DataFrame,
    min_rows: int = 100,
    max_staleness_days: int = 7,
    gap_ratio_threshold: float = 10.0,
    extra_rules: list | None = None,
) -> QualityReport:
    """Sprawdza podstawową jakość danych OHLCV przed trenowaniem modelu.

    Rzuca DataQualityError, jeśli danych jest stanowczo za mało do sensownego
    treningu/walidacji (twardy próg, jakościowo różny od miękkich ostrzeżeń -
    stąd poza silnikiem reguł). Pozostałe problemy (świeżość, dziury w danych)
    ocenia deklaratywnie `check_rules` i trafiają jako ostrzeżenia do
    QualityReport - trening może się odbyć, ale użytkownik powinien o nich wiedzieć.
    """
    if len(df) < min_rows:
        raise DataQualityError(
            f"Za mało danych do wiarygodnego treningu/walidacji: {len(df)} świec "
            f"(wymagane min. {min_rows}). Zwiększ --period albo wybierz płynniejszy ticker."
        )

    metrics = _compute_metrics(df)
    rules = default_rules(max_staleness_days, gap_ratio_threshold) + (extra_rules or [])

    report = QualityReport()
    for message in check_rules(metrics, rules):
        report.warn(message)
    return report
