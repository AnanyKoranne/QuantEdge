"""
Real data loader (standalone).

Purpose:
    Provide a standalone `load_universe()` function that fetches adjusted close
    prices via yfinance, with a fallback that drops missing tickers.

Notes:
    The package implementation lives in `quantedge.data.load_universe`; this
    module is a small, runnable wrapper to satisfy the project interface.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from quantedge.data import load_universe as _load_universe


def load_universe(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close prices for a universe using yfinance.

    Args:
        tickers: List of tickers.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        Adjusted close price DataFrame (dates x tickers). Tickers with missing
        data are dropped (with warnings logged).
    """
    return _load_universe(tickers=tickers, start=start, end=end)


if __name__ == "__main__":
    # Minimal smoke test when run as a script.
    df = load_universe(["AAPL", "MSFT", "BRK-B"], start="2020-01-01", end="2021-01-01")
    print(df.tail())

