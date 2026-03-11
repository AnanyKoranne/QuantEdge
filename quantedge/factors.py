"""
Factor construction for QuantEdge.

Purpose:
    Compute the dual-factor signals used by the strategy:
    - 12-1 momentum (lookback with skip)
    - short-term mean reversion (negative short-term return)
    - weighted composite score
"""

from __future__ import annotations

import pandas as pd

from .config import Config


def xscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score (row-wise).

    Args:
        df: DataFrame where each row is a cross-section (date) and columns are assets.

    Returns:
        Row-wise z-scored DataFrame.
    """
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def compute_momentum(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute momentum factor as a cross-sectional z-score.

    Formula (preserved from original):
        raw = prices.shift(momentum_skip) / prices.shift(momentum_lookback) - 1

    Args:
        prices: Adjusted close prices.
        cfg: Backtest configuration.

    Returns:
        Momentum z-scores by date and ticker.
    """
    raw = prices.shift(cfg.momentum_skip) / prices.shift(cfg.momentum_lookback) - 1
    return xscore(raw)


def compute_mean_reversion(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute mean-reversion factor as a cross-sectional z-score.

    Formula (preserved from original):
        raw = -(prices / prices.shift(mean_rev_lookback) - 1)

    Args:
        prices: Adjusted close prices.
        cfg: Backtest configuration.

    Returns:
        Mean-reversion z-scores by date and ticker.
    """
    raw = -(prices / prices.shift(cfg.mean_rev_lookback) - 1)
    return xscore(raw)


def compute_composite(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute weighted composite score from momentum and mean reversion.

    Args:
        prices: Adjusted close prices.
        cfg: Backtest configuration.

    Returns:
        Composite z-score by date and ticker.
    """
    return cfg.weight_momentum * compute_momentum(prices, cfg) + cfg.weight_mean_rev * compute_mean_reversion(
        prices, cfg
    )

