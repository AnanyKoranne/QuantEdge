"""
Data utilities for QuantEdge.

Purpose:
    Provide both (1) a simulated equity universe for reproducible demos and
    (2) a real-data universe loader using Yahoo Finance adjusted close prices.
"""

from __future__ import annotations

import logging
from typing import Iterable, List

import numpy as np
import pandas as pd
import yfinance as yf

from .config import Config

logger = logging.getLogger(__name__)


def simulate_universe(cfg: Config) -> pd.DataFrame:
    """Simulate an equity universe using a simple factor return model.

    Model:
        r_i(t) = beta_i * r_mkt(t) + alpha_i + eps_i(t)

    Args:
        cfg: Backtest configuration.

    Returns:
        DataFrame of simulated adjusted close prices (business days x tickers).
    """
    np.random.seed(cfg.seed)
    n_days = cfg.n_years * 252
    dates = pd.bdate_range(start=cfg.start, periods=n_days)
    tickers = [f"STK{i:03d}" for i in range(cfg.n_stocks)]

    market_ret = np.random.normal(0.00028, 0.011, n_days)
    betas = np.random.uniform(0.7, 1.3, cfg.n_stocks)
    alphas = np.random.normal(0.0002, 0.0003, cfg.n_stocks)
    vols = np.random.uniform(0.008, 0.018, cfg.n_stocks)

    idio = np.random.normal(0, 1, (n_days, cfg.n_stocks)) * vols
    returns = market_ret[:, None] * betas[None, :] + alphas[None, :] + idio
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _normalize_yahoo_ticker(ticker: str) -> str:
    """Normalize ticker for Yahoo Finance symbols."""
    # Many class shares use '.' in market notation but '-' in Yahoo.
    return ticker.replace(".", "-").upper().strip()


def load_universe(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Load adjusted close prices for a ticker universe using yfinance.

    Fallback behavior:
        - If download fails or a ticker has no usable data, a warning is logged
          and that ticker is dropped.

    Args:
        tickers: List of tickers (Yahoo symbols preferred; '.' will be normalized to '-').
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        DataFrame of adjusted close prices (business days x tickers) with missing tickers dropped.
    """
    norm = [_normalize_yahoo_ticker(t) for t in tickers]
    if len(norm) == 0:
        raise ValueError("tickers must be non-empty")

    try:
        df = yf.download(
            tickers=" ".join(norm),
            start=start,
            end=end,
            auto_adjust=False,
            actions=False,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception as e:
        raise RuntimeError(f"yfinance download failed: {e}") from e

    if df is None or len(df) == 0:
        raise RuntimeError("yfinance returned empty dataframe")

    # yfinance returns either:
    # - MultiIndex columns: (field, ticker) when multiple tickers
    # - Single Index columns: fields when single ticker
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close" in df.columns.get_level_values(0)) is False:
            raise RuntimeError("yfinance data missing 'Adj Close'")
        adj = df["Adj Close"].copy()
    else:
        if "Adj Close" not in df.columns:
            raise RuntimeError("yfinance data missing 'Adj Close'")
        # Single ticker: make it 2D
        t0 = norm[0]
        adj = df[["Adj Close"]].rename(columns={"Adj Close": t0})

    # Drop tickers that are entirely missing or too sparse to be useful.
    kept = []
    for t in adj.columns:
        s = adj[t].dropna()
        if len(s) < 50:
            logger.warning("Dropping ticker %s due to insufficient data", t)
            continue
        kept.append(t)

    dropped = sorted(set(adj.columns) - set(kept))
    for t in dropped:
        logger.warning("Dropped ticker %s (missing/insufficient data)", t)

    out = adj[kept].copy()
    out = out.sort_index()
    # For cross-sectional signals, align on shared dates; forward-fill small gaps then drop residual NaNs.
    out = out.ffill(limit=3)
    out = out.dropna(how="any")
    if out.shape[1] == 0:
        raise RuntimeError("No tickers remain after cleaning")
    return out


def ensure_results_dir(results_dir: str) -> None:
    """Create results directory if missing."""
    import os

    os.makedirs(results_dir, exist_ok=True)


def load_prices(cfg: Config, use_real_data: bool) -> pd.DataFrame:
    """Load prices based on runtime mode (simulated vs real)."""
    if use_real_data:
        return load_universe(list(cfg.tickers), cfg.start, cfg.end)
    return simulate_universe(cfg)


def set_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for scripts."""
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

