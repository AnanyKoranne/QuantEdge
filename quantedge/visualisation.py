"""
Visualization utilities for QuantEdge.

Purpose:
    Centralize all plotting functions and ensure a consistent dark theme across
    charts saved into the results directory.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from .config import Config
from .factors import compute_composite
from .portfolio import BacktestResult


# Dark theme palette (kept consistent with original strategy.py).
BG = "#0d1117"
AX_BG = "#161b22"
GRID = "#30363d"
GOLD, GREEN, RED, BLUE, GRAY = "#FFD700", "#00FF88", "#FF4444", "#00BFFF", "#8b949e"


def style_ax(ax: plt.Axes) -> None:
    """Apply dark theme styling to an axis."""
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=GRAY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save and close a figure with correct facecolor."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    """Compute Spearman rank correlation without SciPy.

    Notes:
        Spearman correlation is Pearson correlation of the rank-transformed data.
        This avoids pandas' SciPy dependency for `method="spearman"`.
    """
    rx = x.rank(method="average")
    ry = y.rank(method="average")
    # Use numpy for speed and to avoid method dispatch.
    a = rx.to_numpy(dtype=float)
    b = ry.to_numpy(dtype=float)
    if len(a) < 2:
        return float("nan")
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    denom = float(np.sqrt(np.nansum(a * a) * np.nansum(b * b)))
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(np.nansum(a * b) / denom)


def plot_results(result: BacktestResult, cfg: Config, save_path: str) -> None:
    """Plot a 2x2 dashboard: cumulative, drawdown, rolling Sharpe, monthly heatmap."""
    r, bm = result.portfolio_returns, result.benchmark_returns
    cum = (1.0 + r).cumprod()
    cum_bm = (1.0 + bm).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    roll_sh = r.rolling(63).apply(lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0)

    monthly = r.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    mdf = monthly.to_frame("ret")
    mdf["yr"] = mdf.index.year
    mdf["mo"] = mdf.index.month_name().str[:3]
    pivot = mdf.pivot(index="yr", columns="mo", values="ret")
    mo_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=[m for m in mo_order if m in pivot.columns])

    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    fig.suptitle(
        f"{cfg.strategy_name}  ·  Momentum ({cfg.weight_momentum:.0%}) + Mean Reversion ({cfg.weight_mean_rev:.0%})",
        fontsize=15,
        color="white",
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1)
    ax1.plot(cum.index, cum.values, color=GOLD, lw=1.8, label="L/S Strategy")
    ax1.plot(cum_bm.index, cum_bm.values, color=GRAY, lw=1.2, ls="--", label="EW Benchmark")
    ax1.fill_between(cum.index, cum_bm.values, cum.values, where=cum.values >= cum_bm.values, alpha=0.12, color=GREEN)
    ax1.fill_between(cum.index, cum_bm.values, cum.values, where=cum.values < cum_bm.values, alpha=0.12, color=RED)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}x"))
    ax1.set_title("Cumulative Returns vs Benchmark", color="white", fontsize=11)
    ax1.legend(facecolor=AX_BG, labelcolor="white", fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2)
    ax2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.55)
    ax2.plot(dd.index, dd.values, color=RED, lw=0.8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.set_title("Drawdown", color="white", fontsize=11)

    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3)
    ax3.plot(roll_sh.index, roll_sh.values, color=BLUE, lw=1.2)
    ax3.axhline(0, color=GRAY, ls="--", lw=0.8)
    ax3.axhline(1, color=GREEN, ls=":", lw=0.8, alpha=0.7, label="Sharpe = 1")
    ax3.fill_between(roll_sh.index, 0, roll_sh.values, where=roll_sh.values >= 0, alpha=0.15, color=GREEN)
    ax3.fill_between(roll_sh.index, 0, roll_sh.values, where=roll_sh.values < 0, alpha=0.15, color=RED)
    ax3.set_title("Rolling 63-Day Sharpe Ratio", color="white", fontsize=11)
    ax3.legend(facecolor=AX_BG, labelcolor="white", fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4)
    sns.heatmap(
        pivot,
        ax=ax4,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        center=0,
        linewidths=0.3,
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8},
    )
    ax4.set_title("Monthly Returns Heatmap", color="white", fontsize=11)
    ax4.set_xlabel("")
    ax4.set_ylabel("")
    ax4.tick_params(colors=GRAY, labelsize=8)

    save_fig(fig, save_path)


def plot_factor_ic(prices: pd.DataFrame, cfg: Config, save_path: str) -> None:
    """Plot monthly Information Coefficient (Spearman) and cumulative IC."""
    scores = compute_composite(prices, cfg)
    fwd_ret = prices.pct_change(cfg.rebal_freq).shift(-cfg.rebal_freq)

    records: List[tuple[pd.Timestamp, float]] = []
    for date in scores.index[cfg.momentum_lookback : -cfg.rebal_freq : cfg.rebal_freq]:
        s = scores.loc[date].dropna()
        f = fwd_ret.loc[date].dropna()
        common = s.index.intersection(f.index)
        if len(common) < cfg.min_cross_section:
            continue
        records.append((date, spearman_corr(s[common], f[common])))

    ic_df = pd.DataFrame(records, columns=["date", "IC"]).set_index("date")
    ic_cum = ic_df["IC"].cumsum()
    ic_std = float(ic_df["IC"].std()) if len(ic_df) else 0.0
    icir = float(ic_df["IC"].mean() / ic_std) if ic_std > 0 else 0.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor=BG)
    fig.suptitle("Information Coefficient (IC) Analysis", fontsize=13, color="white", fontweight="bold")
    for ax in axes:
        style_ax(ax)

    bar_colors = [GREEN if v > 0 else RED for v in ic_df["IC"]]
    axes[0].bar(ic_df.index, ic_df["IC"], color=bar_colors, alpha=0.8, width=15)
    axes[0].axhline(0, color=GRAY, lw=0.8)
    if len(ic_df):
        axes[0].axhline(
            ic_df["IC"].mean(),
            color=GOLD,
            ls="--",
            lw=1.2,
            label=f"Mean IC = {ic_df['IC'].mean():.3f}  |  ICIR = {icir:.2f}",
        )
        axes[0].legend(facecolor=AX_BG, labelcolor="white", fontsize=8)
    axes[0].set_title("Monthly IC (Spearman)", color="white", fontsize=10)

    axes[1].plot(ic_cum.index, ic_cum.values, color=GOLD, lw=1.5)
    axes[1].fill_between(ic_cum.index, ic_cum.values, alpha=0.2, color=GOLD)
    axes[1].set_title("Cumulative IC (upward slope = persistent predictive signal)", color="white", fontsize=10)

    fig.tight_layout()
    save_fig(fig, save_path)


def plot_factor_decay(prices: pd.DataFrame, cfg: Config, save_path: str) -> pd.DataFrame:
    """Plot IC decay vs holding period for the composite signal.

    For each holding period (h), compute cross-sectional Spearman rank IC between
    today's composite score and the forward return over horizon (h).
    The plotted value is the time-series average IC for each horizon.

    Args:
        prices: Adjusted close prices.
        cfg: Backtest configuration.
        save_path: Output path for the bar chart.

    Returns:
        DataFrame with columns ['horizon', 'ic_mean', 'ic_std', 'icir'].
    """
    horizons = [1, 5, 10, 21, 42, 63]
    scores = compute_composite(prices, cfg)

    rows = []
    for h in horizons:
        fwd = prices.pct_change(h).shift(-h)
        ics: List[float] = []
        # Use the same warmup start as the strategy so scores are defined.
        for date in scores.index[cfg.momentum_lookback : -h]:
            s = scores.loc[date].dropna()
            f = fwd.loc[date].dropna()
            common = s.index.intersection(f.index)
            if len(common) < cfg.min_cross_section:
                continue
            ics.append(spearman_corr(s[common], f[common]))

        ic_mean = float(np.mean(ics)) if len(ics) else float("nan")
        ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else float("nan")
        icir = float(ic_mean / ic_std) if np.isfinite(ic_mean) and np.isfinite(ic_std) and ic_std > 0 else float("nan")
        rows.append({"horizon": h, "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir})

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.8), facecolor=BG)
    style_ax(ax)
    ax.set_title("Factor Decay: IC vs Forward Return Horizon", color="white", fontsize=12, fontweight="bold")
    colors = [GREEN if (v >= 0 or not np.isfinite(v)) else RED for v in df["ic_mean"].values]
    ax.bar(df["horizon"].astype(str), df["ic_mean"], color=colors, alpha=0.85)
    ax.axhline(0, color=GRAY, lw=0.9)
    ax.set_xlabel("Holding period (trading days)", color=GRAY)
    ax.set_ylabel("Mean Spearman IC", color=GRAY)
    for i, v in enumerate(df["ic_mean"].values):
        if np.isfinite(v):
            ax.text(i, v + (0.002 if v >= 0 else -0.002), f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", color="white", fontsize=8)

    save_fig(fig, save_path)
    return df


def plot_sweep_heatmap(
    sweep_df: pd.DataFrame,
    save_path: str,
    *,
    x_col: str = "weight_momentum",
    y_col: str = "momentum_lookback",
    value_col: str = "sharpe",
) -> None:
    """Generate a heatmap for sweep results.

    Note:
        The sweep grid includes additional dimensions. For a 2D heatmap, we
        aggregate by taking the maximum Sharpe observed for each (y, x) pair.
    """
    agg = sweep_df.groupby([y_col, x_col], as_index=False)[value_col].max()
    pivot = agg.pivot(index=y_col, columns=x_col, values=value_col).sort_index()

    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=BG)
    style_ax(ax)
    ax.set_title("Sharpe Ratio Sensitivity (max across other params)", color="white", fontsize=11, fontweight="bold")
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"shrink": 0.85},
    )
    ax.set_xlabel("weight_momentum", color=GRAY)
    ax.set_ylabel("momentum_lookback", color=GRAY)

    save_fig(fig, save_path)

