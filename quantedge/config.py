"""
Configuration for the QuantEdge backtesting project.

Purpose:
    Centralize constants and runtime parameters into a single typed dataclass so
    modules can remain pure and parameterized without changing core math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Config:
    """Backtest configuration.

    Attributes:
        seed: Random seed for simulation.
        n_stocks: Number of simulated stocks.
        n_years: Number of simulated years (252 trading days per year).
        start: Start date (YYYY-MM-DD) for real data or simulation index start.
        end: End date (YYYY-MM-DD) for real data (simulation ignores and uses n_years).
        tickers: Default ticker universe for real-data mode.
        momentum_lookback: Momentum lookback window in trading days.
        momentum_skip: Skip window (trading days) for momentum calculation.
        mean_rev_lookback: Mean reversion lookback window in trading days.
        top_decile: Fraction of universe to long.
        bot_decile: Fraction of universe to short.
        weight_momentum: Weight on momentum z-score.
        weight_mean_rev: Weight on mean-reversion z-score.
        rebal_freq: Rebalance frequency in trading days.
        tc_bps: One-way transaction costs in basis points applied on turnover.
        min_cross_section: Minimum number of stocks required to form a portfolio.
        results_dir: Output directory for CSVs/charts/PDF.
        strategy_name: Human-readable strategy name for reporting.
    """

    seed: int = 42
    n_stocks: int = 100
    n_years: int = 5
    start: str = "2019-01-02"
    end: str = "2024-01-01"
    tickers: Sequence[str] = (
        # Hardcoded: top ~30 S&P 500 constituents by market cap (common large caps).
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "GOOG",
        "META",
        "BRK-B",
        "LLY",
        "AVGO",
        "JPM",
        "TSLA",
        "V",
        "XOM",
        "UNH",
        "MA",
        "WMT",
        "COST",
        "PG",
        "JNJ",
        "ORCL",
        "HD",
        "ABBV",
        "KO",
        "BAC",
        "MRK",
        "CRM",
        "PEP",
        "CSCO",
        "TMO",
    )
    momentum_lookback: int = 252
    momentum_skip: int = 21
    mean_rev_lookback: int = 5
    top_decile: float = 0.10
    bot_decile: float = 0.10
    weight_momentum: float = 0.75
    weight_mean_rev: float = 0.25
    rebal_freq: int = 21
    tc_bps: float = 10.0
    min_cross_section: int = 20
    results_dir: str = "results"
    strategy_name: str = "Dual-Factor L/S (Momentum + Mean Reversion)"

