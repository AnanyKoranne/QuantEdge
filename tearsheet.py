"""
Tearsheet generator (standalone).

Purpose:
    Generate a single-page PDF tearsheet from existing results saved in
    `results/` (daily_returns.csv, benchmark_returns.csv, turnover.csv).
"""

from __future__ import annotations

import os

from quantedge.config import Config
from quantedge.data import ensure_results_dir, set_logging
from quantedge.tearsheet import (
    TearsheetInputs,
    generate_tearsheet_pdf,
    load_avg_turnover_from_results,
    load_returns_from_results,
)


def main() -> None:
    """Generate a tearsheet PDF from existing results files."""
    set_logging()
    cfg = Config()
    ensure_results_dir(cfg.results_dir)

    strat, bm = load_returns_from_results(cfg.results_dir)
    avg_to = load_avg_turnover_from_results(cfg.results_dir)
    pdf_path = os.path.join(cfg.results_dir, "tearsheet.pdf")

    inputs = TearsheetInputs(
        strategy_returns=strat,
        benchmark_returns=bm,
        universe_size=int(os.environ.get("QUANTEDGE_UNIVERSE_SIZE", "0")) or 0,
        start_date=str(strat.index.min().date()),
        end_date=str(strat.index.max().date()),
        avg_turnover=avg_to,
    )
    generate_tearsheet_pdf(cfg, inputs, pdf_path)
    print(f"[OK] Generated tearsheet -> {pdf_path}")


if __name__ == "__main__":
    main()

