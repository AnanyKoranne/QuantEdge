"""
Legacy entry point (compatibility wrapper).

Purpose:
    This repository has been refactored into the `quantedge/` package with a new
    CLI entry point at `main.py`. This file remains as a thin wrapper so existing
    workflows that call `python strategy.py` continue to work.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from quantedge.config import Config
from quantedge.data import set_logging
from main import run_backtest_mode

def main() -> None:
    """Compatibility wrapper CLI (prefer `python main.py --mode backtest`)."""
    set_logging()
    p = argparse.ArgumentParser(description="Legacy wrapper for QuantEdge (use main.py)")
    p.add_argument("--use-real-data", action="store_true", help="Use yfinance adjusted close data (vs simulated).")
    p.add_argument("--start", type=str, default=Config().start)
    p.add_argument("--end", type=str, default=Config().end)
    args = p.parse_args()

    cfg = Config(start=args.start, end=args.end)
    run_backtest_mode(cfg, use_real_data=args.use_real_data)


if __name__ == "__main__":
    main()