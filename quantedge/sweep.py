"""
Parameter sweep utilities for QuantEdge.

Purpose:
    Run sensitivity analysis over a parameter grid and save results + summary
    charts to the results directory.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .config import Config
from .data import ensure_results_dir, simulate_universe
from .metrics import compute_metrics
from .portfolio import compute_turnover_summary, run_backtest
from .visualisation import plot_sweep_heatmap


def run_parameter_sweep(
    base_cfg: Config,
    *,
    prices: pd.DataFrame | None = None,
    results_dir: str | None = None,
    save_csv_path: str | None = None,
    save_heatmap_path: str | None = None,
) -> pd.DataFrame:
    """Run the backtest across a grid of parameters and save results.

    Grid (as requested):
        - momentum_lookback: [126, 189, 252]
        - momentum_skip: [0, 10, 21]
        - mean_rev_lookback: [3, 5, 10]
        - weight_momentum: [0.5, 0.75, 1.0]

    Notes:
        - weight_mean_rev is set to (1 - weight_momentum) to preserve a convex
          combination of the two z-scored factors.
        - Uses the same simulated prices across all runs unless prices provided.

    Args:
        base_cfg: Base config to copy/modify.
        prices: Optional price matrix to reuse across parameter combinations.
        results_dir: Output directory (defaults to base_cfg.results_dir).
        save_csv_path: CSV output path (defaults to results/parameter_sweep.csv).
        save_heatmap_path: Heatmap output path (defaults to results/parameter_sweep_heatmap.png).

    Returns:
        DataFrame of sweep results (one row per parameter combination).
    """
    results_dir = results_dir or base_cfg.results_dir
    ensure_results_dir(results_dir)
    save_csv_path = save_csv_path or f"{results_dir}/parameter_sweep.csv"
    save_heatmap_path = save_heatmap_path or f"{results_dir}/parameter_sweep_heatmap.png"

    prices = prices if prices is not None else simulate_universe(base_cfg)

    grid = {
        "momentum_lookback": [126, 189, 252],
        "momentum_skip": [0, 10, 21],
        "mean_rev_lookback": [3, 5, 10],
        "weight_momentum": [0.5, 0.75, 1.0],
    }

    rows: List[Dict[str, float]] = []
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        wm = float(params["weight_momentum"])
        cfg = replace(
            base_cfg,
            momentum_lookback=int(params["momentum_lookback"]),
            momentum_skip=int(params["momentum_skip"]),
            mean_rev_lookback=int(params["mean_rev_lookback"]),
            weight_momentum=wm,
            weight_mean_rev=1.0 - wm,
        )

        result = run_backtest(prices, cfg)
        perf = compute_metrics(result.portfolio_returns)
        avg_to = compute_turnover_summary(result.turnover_history)
        rows.append(
            {
                **params,
                "weight_mean_rev": cfg.weight_mean_rev,
                "ann_return": perf.annual_return,
                "sharpe": perf.sharpe,
                "max_drawdown": perf.max_drawdown,
                "avg_turnover": avg_to,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    df.to_csv(save_csv_path, index=False)
    plot_sweep_heatmap(df, save_heatmap_path)
    return df


def print_top_n(df: pd.DataFrame, n: int = 5) -> None:
    """Print top N parameter combinations by Sharpe."""
    cols = [
        "sharpe",
        "ann_return",
        "max_drawdown",
        "avg_turnover",
        "momentum_lookback",
        "momentum_skip",
        "mean_rev_lookback",
        "weight_momentum",
        "weight_mean_rev",
    ]
    top = df.head(n)[cols].copy()
    # Friendly formatting
    top["ann_return"] = top["ann_return"].map(lambda x: f"{x:.1%}")
    top["max_drawdown"] = top["max_drawdown"].map(lambda x: f"{x:.1%}")
    top["avg_turnover"] = top["avg_turnover"].map(lambda x: f"{x:.1%}")
    top["sharpe"] = top["sharpe"].map(lambda x: f"{x:.2f}")
    print("\nTop parameter combinations by Sharpe:")
    print(top.to_string(index=False))

