"""
quantedge package.

Purpose:
    A small, self-contained equity long/short research backtesting toolkit for
    dual-factor signals (momentum + mean reversion) with plotting and reporting.
"""

from .config import Config
from .portfolio import BacktestResult, run_backtest
from .data import simulate_universe, load_universe

__all__ = [
    "Config",
    "BacktestResult",
    "run_backtest",
    "simulate_universe",
    "load_universe",
]

