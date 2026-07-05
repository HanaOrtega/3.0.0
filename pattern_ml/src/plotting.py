"""Wizualizacja: świece + wskaźniki + formacje świecowe + sygnał ML."""

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


def plot_chart(
    df: pd.DataFrame,
    features: pd.DataFrame,
    pat: pd.DataFrame,
    ticker: str,
    signal: str,
    proba: dict,
    last_n: int = 150,
    save_path: str | None = None,
):
    """Rysuje wykres świecowy z SMA/Bollingerem, RSI, MACD, wolumenem, formacjami i sygnałem ML."""
    plot_df = df.iloc[-last_n:]
    feat = features.loc[plot_df.index]
    pat_slice = pat.loc[plot_df.index]

    bullish_markers = _marker_series(plot_df, pat_slice, BULLISH, offset_dir=-1)
    bearish_markers = _marker_series(plot_df, pat_slice, BEARISH, offset_dir=+1)

    addplots = [
        mpf.make_addplot(feat["sma20"], color="#377eb8", width=1.0, panel=0),
        mpf.make_addplot(feat["sma50"], color="#ff7f00", width=1.0, panel=0),
        mpf.make_addplot(feat["bb_high"], color="#999999", width=0.7, linestyle="--", panel=0),
        mpf.make_addplot(feat["bb_low"], color="#999999", width=0.7, linestyle="--", panel=0),
        mpf.make_addplot(
            bullish_markers, type="scatter", markersize=70, marker="^",
            color="#1a9850", panel=0,
        ),
        mpf.make_addplot(
            bearish_markers, type="scatter", markersize=70, marker="v",
            color="#d73027", panel=0,
        ),
        mpf.make_addplot(feat["rsi"], color="#6a3d9a", panel=2, ylabel="RSI"),
        mpf.make_addplot(feat["macd"], color="#1f78b4", panel=3, ylabel="MACD"),
        mpf.make_addplot(feat["macd_signal"], color="#e31a1c", panel=3),
    ]

    fig, axes = mpf.plot(
        plot_df,
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
    proba_txt = "  |  ".join(f"{k}: {v * 100:.1f}%" for k, v in proba.items())
    axes[0].annotate(
        f"Sygnał ML: {signal}\n{proba_txt}",
        xy=(0.99, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.85),
    )

    last_price = plot_df["Close"].iloc[-1]
    arrow_symbol = {"LONG": "↑", "SHORT": "↓", "NEUTRALNY": "→"}[signal]
    axes[0].annotate(
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
