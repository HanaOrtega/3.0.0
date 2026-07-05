"""Wizualizacja: świece + wskaźniki + formacje świecowe + sygnał ML + prognoza przyszłości."""

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from .patterns import BEARISH, BULLISH

SIGNAL_COLORS = {"LONG": "#1a9850", "SHORT": "#d73027", "NEUTRALNY": "#999999"}


def _marker_series(df: pd.DataFrame, pat: pd.DataFrame, columns, offset_dir: int) -> pd.Series:
    """Buduje serię do rysowania znaczników formacji (offset_dir=-1 pod świecą, +1 nad świecą)."""
    hit = pat[columns].any(axis=1)
    offset = (df["High"] - df["Low"]).rolling(14).mean().bfill() * 0.5
    if offset_dir < 0:
        y = df["Low"] - offset
    else:
        y = df["High"] + offset
    series = pd.Series(np.nan, index=df.index)
    series[hit] = y[hit]
    return series


def _infer_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    diffs = index.to_series().diff().dropna()
    return diffs.mode().iloc[0] if not diffs.empty else pd.Timedelta(days=1)


def _forecast_cone(last_date, step, last_close, quantile_prices, horizon):
    """Buduje serię dat przyszłych + ścieżki dolna/mediana/górna (rosnący 'stożek' niepewności)."""
    future_dates = [last_date + step * i for i in range(1, horizon + 1)]

    low_target = quantile_prices[0.1]
    median_target = quantile_prices[0.5]
    high_target = quantile_prices[0.9]

    median_path = [last_close]
    low_path = [last_close]
    high_path = [last_close]
    for i in range(1, horizon + 1):
        frac = i / horizon
        spread_frac = np.sqrt(frac)  # niepewność rośnie wraz z odległością w czasie
        median_val = last_close + (median_target - last_close) * frac
        low_path.append(median_val - (median_target - low_target) * spread_frac)
        high_path.append(median_val + (high_target - median_target) * spread_frac)
        median_path.append(median_val)

    return future_dates, low_path, median_path, high_path


def plot_chart(
    df: pd.DataFrame,
    features: pd.DataFrame,
    pat: pd.DataFrame,
    ticker: str,
    signal: str,
    proba: dict,
    quantile_prices: dict | None = None,
    horizon: int = 5,
    last_n: int = 150,
    save_path: str | None = None,
):
    """Rysuje wykres świecowy z SMA/Bollingerem, RSI, MACD, wolumenem, formacjami,
    sygnałem ML oraz - jeśli podano `quantile_prices` - prognozowanym stożkiem ceny
    (dolna/mediana/górna granica) rozciągniętym w przyszłość na osi X."""
    plot_df = df.iloc[-last_n:]
    feat = features.loc[plot_df.index]
    pat_slice = pat.loc[plot_df.index]

    step = _infer_step(plot_df.index)
    n_future = horizon if quantile_prices else 0
    if n_future:
        future_dates = [plot_df.index[-1] + step * i for i in range(1, n_future + 1)]
        future_blank = pd.DataFrame(
            {col: np.nan for col in plot_df.columns}, index=pd.DatetimeIndex(future_dates)
        )
        ext_df = pd.concat([plot_df, future_blank])
    else:
        ext_df = plot_df

    def _extend(series: pd.Series) -> pd.Series:
        return series.reindex(ext_df.index)

    bullish_markers = _marker_series(plot_df, pat_slice, BULLISH, offset_dir=-1)
    bearish_markers = _marker_series(plot_df, pat_slice, BEARISH, offset_dir=+1)

    addplots = [
        mpf.make_addplot(_extend(feat["sma20"]), color="#377eb8", width=1.0, panel=0),
        mpf.make_addplot(_extend(feat["sma50"]), color="#ff7f00", width=1.0, panel=0),
        mpf.make_addplot(_extend(feat["bb_high"]), color="#999999", width=0.7, linestyle="--", panel=0),
        mpf.make_addplot(_extend(feat["bb_low"]), color="#999999", width=0.7, linestyle="--", panel=0),
        mpf.make_addplot(
            _extend(bullish_markers), type="scatter", markersize=70, marker="^",
            color="#1a9850", panel=0,
        ),
        mpf.make_addplot(
            _extend(bearish_markers), type="scatter", markersize=70, marker="v",
            color="#d73027", panel=0,
        ),
        mpf.make_addplot(_extend(feat["rsi"]), color="#6a3d9a", panel=2, ylabel="RSI"),
        mpf.make_addplot(_extend(feat["macd"]), color="#1f78b4", panel=3, ylabel="MACD"),
        mpf.make_addplot(_extend(feat["macd_signal"]), color="#e31a1c", panel=3),
    ]

    fig, axes = mpf.plot(
        ext_df,
        type="candle",
        style="yahoo",
        addplot=addplots,
        volume=True,
        volume_panel=1,
        panel_ratios=(6, 1.5, 1.5, 1.5),
        figratio=(16, 10),
        figscale=1.2,
        returnfig=True,
        title=f"\n{ticker} — formacje świecowe + sygnał ML",
    )

    ax_rsi = axes[2] if len(axes) > 2 else None
    if ax_rsi is not None:
        ax_rsi.axhline(70, color="red", linestyle=":", linewidth=0.8)
        ax_rsi.axhline(30, color="green", linestyle=":", linewidth=0.8)

    color = SIGNAL_COLORS.get(signal, "#333333")
    ax_price = axes[0]

    if quantile_prices:
        last_close = plot_df["Close"].iloc[-1]
        n_hist = len(plot_df)
        _, low_path, median_path, high_path = _forecast_cone(
            plot_df.index[-1], step, last_close, quantile_prices, n_future
        )
        x_forecast = list(range(n_hist - 1, n_hist - 1 + n_future + 1))

        ax_price.axvline(n_hist - 1, color="#666666", linestyle=":", linewidth=1.0)
        ax_price.fill_between(x_forecast, low_path, high_path, color=color, alpha=0.15, zorder=2)
        ax_price.plot(x_forecast, median_path, color=color, linewidth=2.0, linestyle="--", zorder=3)
        ax_price.plot(
            x_forecast[-1], median_path[-1], marker="o", markersize=6, color=color, zorder=4,
        )
        ax_price.annotate(
            f"prognoza +{n_future}: {median_path[-1]:.2f}\n"
            f"({low_path[-1]:.2f} - {high_path[-1]:.2f})",
            xy=(x_forecast[-1], median_path[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8,
            color=color,
            fontweight="bold",
            va="center",
        )

    proba_txt = "  |  ".join(f"{k}: {v * 100:.1f}%" for k, v in proba.items())
    ax_price.annotate(
        f"Sygnał ML: {signal}\n{proba_txt}",
        xy=(0.01, 0.02),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.85),
    )

    last_price = plot_df["Close"].iloc[-1]
    arrow_symbol = {"LONG": "↑", "SHORT": "↓", "NEUTRALNY": "→"}[signal]
    ax_price.annotate(
        arrow_symbol,
        xy=(len(plot_df) - 1, last_price),
        xytext=(len(plot_df) - 1, last_price),
        fontsize=28,
        color=color,
        fontweight="bold",
        ha="center",
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
