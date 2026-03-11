"""
PDF tearsheet generation for QuantEdge.

Purpose:
    Create a single-page PDF report combining key metrics and charts into a
    compact, shareable summary of a backtest run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

from .config import Config
from .metrics import PerfMetrics, compute_metrics, compute_max_drawdown
from .visualisation import BG, AX_BG, GRAY, GOLD, GREEN, RED, style_ax, save_fig


@dataclass(frozen=True)
class TearsheetInputs:
    """Inputs required to generate a tearsheet."""

    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    universe_size: int
    start_date: str
    end_date: str
    avg_turnover: float | None = None


def _plot_cumulative(strategy: pd.Series, benchmark: pd.Series, path: str, title: str) -> None:
    cum = (1.0 + strategy).cumprod()
    cum_bm = (1.0 + benchmark).cumprod()

    fig, ax = plt.subplots(figsize=(7.2, 2.6), facecolor=BG)
    style_ax(ax)
    ax.plot(cum.index, cum.values, color=GOLD, lw=1.6, label="Strategy")
    ax.plot(cum_bm.index, cum_bm.values, color=GRAY, lw=1.0, ls="--", label="Benchmark")
    ax.set_title(title, color="white", fontsize=10, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}x"))
    ax.legend(facecolor=AX_BG, labelcolor="white", fontsize=7, loc="upper left")
    save_fig(fig, path, dpi=160)


def _plot_drawdown(strategy: pd.Series, path: str, title: str) -> None:
    cum = (1.0 + strategy).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()

    fig, ax = plt.subplots(figsize=(7.2, 2.1), facecolor=BG)
    style_ax(ax)
    ax.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.6)
    ax.plot(dd.index, dd.values, color=RED, lw=0.8)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    save_fig(fig, path, dpi=160)


def _plot_monthly_heatmap(strategy: pd.Series, path: str, title: str) -> None:
    monthly = strategy.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    mdf = monthly.to_frame("ret")
    mdf["yr"] = mdf.index.year
    mdf["mo"] = mdf.index.month_name().str[:3]
    pivot = mdf.pivot(index="yr", columns="mo", values="ret")
    mo_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=[m for m in mo_order if m in pivot.columns])

    fig, ax = plt.subplots(figsize=(7.2, 2.6), facecolor=BG)
    style_ax(ax)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold")
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        center=0,
        linewidths=0.25,
        annot_kws={"size": 6},
        cbar_kws={"shrink": 0.75},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(colors=GRAY, labelsize=7)
    save_fig(fig, path, dpi=170)


def _safe_series(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    return s.sort_index().dropna()


def generate_tearsheet_pdf(
    cfg: Config,
    inputs: TearsheetInputs,
    save_path: str,
    *,
    charts_dir: Optional[str] = None,
) -> str:
    """Generate a single-page PDF tearsheet.

    Args:
        cfg: Backtest configuration.
        inputs: TearsheetInputs (returns and metadata).
        save_path: PDF output path.
        charts_dir: Directory for intermediate chart PNGs (defaults to cfg.results_dir).

    Returns:
        The PDF path written.
    """
    charts_dir = charts_dir or cfg.results_dir
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    strat = _safe_series(inputs.strategy_returns)
    bm = _safe_series(inputs.benchmark_returns.reindex(strat.index).fillna(0.0))

    strat_m = compute_metrics(strat)
    bm_m = compute_metrics(bm)
    avg_turnover = float(inputs.avg_turnover) if inputs.avg_turnover is not None else float("nan")

    cum_png = os.path.join(charts_dir, "tearsheet_cumulative.png")
    dd_png = os.path.join(charts_dir, "tearsheet_drawdown.png")
    hm_png = os.path.join(charts_dir, "tearsheet_monthly_heatmap.png")

    _plot_cumulative(strat, bm, cum_png, "Cumulative Returns (Strategy vs Benchmark)")
    _plot_drawdown(strat, dd_png, "Drawdown")
    _plot_monthly_heatmap(strat, hm_png, "Monthly Returns Heatmap")

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # Background (approx dark) by drawing filled rectangle.
    pdf.set_fill_color(13, 17, 23)
    pdf.rect(0, 0, 210, 297, style="F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(10, 8)
    pdf.cell(190, 8, cfg.strategy_name, ln=1)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(200, 200, 200)
    pdf.set_x(10)
    pdf.cell(
        190,
        6,
        f"Date range: {inputs.start_date} -> {inputs.end_date}    |    Universe size: {inputs.universe_size}",
        ln=1,
    )

    # Key metrics table.
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_xy(10, 24)
    pdf.cell(95, 6, "Key Metrics", ln=1)

    def _fmt_pct(x: float) -> str:
        return "—" if not np.isfinite(x) else f"{x:.1%}"

    def _fmt_num(x: float) -> str:
        return "—" if not np.isfinite(x) else f"{x:.2f}"

    rows = [
        ("Sharpe", _fmt_num(strat_m.sharpe)),
        ("Ann Return", _fmt_pct(strat_m.annual_return)),
        ("Max DD", _fmt_pct(strat_m.max_drawdown)),
        ("Calmar", _fmt_num(strat_m.calmar)),
        ("Turnover", "—" if not np.isfinite(avg_turnover) else f"{avg_turnover:.1%}"),
    ]

    pdf.set_font("Helvetica", "", 10)
    x0, y0 = 10, 31
    pdf.set_xy(x0, y0)
    pdf.set_draw_color(48, 54, 61)
    pdf.set_fill_color(22, 27, 34)
    for i, (k, v) in enumerate(rows):
        pdf.set_xy(x0, y0 + i * 6.2)
        pdf.cell(45, 6.2, k, border=1, fill=True)
        pdf.cell(35, 6.2, v, border=1, fill=False)
        pdf.set_xy(x0 + 80, y0 + i * 6.2)
        # also show benchmark for context (subset)
        if k == "Sharpe":
            bmv = _fmt_num(bm_m.sharpe)
        elif k == "Ann Return":
            bmv = _fmt_pct(bm_m.annual_return)
        elif k == "Max DD":
            bmv = _fmt_pct(bm_m.max_drawdown)
        elif k == "Calmar":
            bmv = _fmt_num(bm_m.calmar)
        else:
            bmv = ""
        pdf.cell(25, 6.2, "BM", border=1, fill=True)
        pdf.cell(25, 6.2, bmv, border=1, fill=False)

    # Strategy Summary bullets.
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_xy(10, 68)
    pdf.cell(95, 6, "Strategy Summary", ln=1)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(230, 230, 230)
    bullets = [
        "Dual-factor composite: 12-1 momentum + short-term mean reversion pullback.",
        "Dollar-neutral decile long/short portfolio with monthly rebalancing.",
        "Transaction costs modeled via turnover-based one-way bps deduction.",
    ]
    y = 74
    for b in bullets:
        pdf.set_xy(12, y)
        pdf.multi_cell(190, 4.5, f"- {b}")
        y += 7.5

    # Charts on the right/lower area.
    pdf.set_text_color(255, 255, 255)
    # Place charts (x, y, width). Heights auto from image ratio.
    pdf.image(cum_png, x=105, y=24, w=95)
    pdf.image(dd_png, x=105, y=88, w=95)
    pdf.image(hm_png, x=10, y=118, w=190)

    pdf.output(save_path)
    return save_path


def load_returns_from_results(results_dir: str) -> Tuple[pd.Series, pd.Series]:
    """Load strategy and benchmark daily returns from `results/`.

    Expects:
        - results/daily_returns.csv with column 'return'
        - results/benchmark_returns.csv with column 'return'
    """
    strat_path = os.path.join(results_dir, "daily_returns.csv")
    bm_path = os.path.join(results_dir, "benchmark_returns.csv")
    strat_df = pd.read_csv(strat_path, index_col=0)
    bm_df = pd.read_csv(bm_path, index_col=0) if os.path.exists(bm_path) else None
    strat = pd.Series(strat_df.iloc[:, 0].values, index=pd.to_datetime(strat_df.index), name="strategy")
    if bm_df is None:
        bm = pd.Series(0.0, index=strat.index, name="benchmark")
    else:
        bm = pd.Series(bm_df.iloc[:, 0].values, index=pd.to_datetime(bm_df.index), name="benchmark")
    return strat, bm


def load_avg_turnover_from_results(results_dir: str) -> float | None:
    """Load average turnover from `results/turnover.csv` if present."""
    path = os.path.join(results_dir, "turnover.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    s = pd.Series(df.iloc[:, 0].values, index=pd.to_datetime(df.index))
    s = s.dropna()
    if len(s) == 0:
        return None
    return float(s.mean())

