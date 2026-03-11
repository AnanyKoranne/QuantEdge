"""
Portfolio construction and backtesting engine for QuantEdge.

Purpose:
    Implement the long/short decile portfolio construction and the monthly
    rebalancing backtest with transaction costs, preserving core logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .config import Config
from .factors import compute_composite


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    weights_history: pd.DataFrame
    turnover_history: pd.Series


def build_weights(scores: pd.Series, cfg: Config) -> pd.Series:
    """Build dollar-neutral long/short weights: top decile long, bottom decile short.

    Core logic preserved from original implementation.

    Args:
        scores: Cross-sectional scores for one date.
        cfg: Backtest configuration.

    Returns:
        Weight vector aligned to scores.index.
    """
    valid = scores.dropna()
    if len(valid) < cfg.min_cross_section:
        return pd.Series(0.0, index=scores.index)

    n = len(valid)
    n_long = max(1, int(n * cfg.top_decile))
    n_short = max(1, int(n * cfg.bot_decile))
    ranked = valid.rank(ascending=False)

    w = pd.Series(0.0, index=valid.index)
    w[ranked <= n_long] = 1.0 / n_long
    w[ranked > (n - n_short)] = -1.0 / n_short
    return w.reindex(scores.index, fill_value=0.0)


def run_backtest(prices: pd.DataFrame, cfg: Config) -> BacktestResult:
    """Run a monthly rebalancing backtest with transaction costs.

    Core logic preserved from original `strategy.py`:
        - compute composite score
        - equal-weight benchmark = mean of daily returns across assets
        - rebalance at `momentum_lookback` then every `rebal_freq` days
        - between rebalances, hold last weights
        - transaction costs applied on rebalance days based on turnover

    Args:
        prices: Adjusted close prices (date index, tickers as columns).
        cfg: Backtest configuration.

    Returns:
        BacktestResult containing strategy returns, benchmark returns, weights, turnover.
    """
    scores = compute_composite(prices, cfg)
    daily_ret = prices.pct_change()
    benchmark = daily_ret.mean(axis=1)

    rebal_dates = prices.index[cfg.momentum_lookback :: cfg.rebal_freq]
    weights_hist = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    turnover_log: Dict[pd.Timestamp, float] = {}
    prev_w = pd.Series(0.0, index=prices.columns)

    for date in rebal_dates:
        new_w = build_weights(scores.loc[date], cfg)
        turnover_log[date] = float((new_w - prev_w).abs().sum() / 2)
        loc = prices.index.get_loc(date)
        end_loc = min(loc + cfg.rebal_freq, len(prices))
        weights_hist.loc[prices.index[loc:end_loc]] = new_w.values
        prev_w = new_w

    port_gross = (weights_hist * daily_ret).sum(axis=1)
    tc = pd.Series(0.0, index=prices.index)
    for d, to in turnover_log.items():
        tc.loc[d] = to * cfg.tc_bps / 10_000

    start = prices.index[cfg.momentum_lookback]
    return BacktestResult(
        portfolio_returns=(port_gross - tc).loc[start:],
        benchmark_returns=benchmark.loc[start:],
        weights_history=weights_hist,
        turnover_history=pd.Series(turnover_log).sort_index(),
    )


def compute_turnover_summary(turnover_history: pd.Series) -> float:
    """Compute average turnover for logging/metrics."""
    if len(turnover_history) == 0:
        return float("nan")
    return float(turnover_history.mean())

