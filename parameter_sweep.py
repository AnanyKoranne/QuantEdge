"""
Parameter sweep runner (standalone).

Purpose:
    Run a grid search over key strategy parameters, save results to CSV, render
    a Sharpe heatmap, and print the top combinations.
"""

from __future__ import annotations

from quantedge.config import Config
from quantedge.sweep import print_top_n, run_parameter_sweep
from quantedge.data import set_logging


def main() -> None:
    """Run the parameter sweep on the simulated universe."""
    set_logging()
    cfg = Config()
    df = run_parameter_sweep(cfg, results_dir=cfg.results_dir)
    print_top_n(df, n=5)


if __name__ == "__main__":
    main()

