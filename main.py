"""
Project entry point for QuantEdge.

Purpose:
    Provide an argparse CLI to run:
    - backtest (with optional real data)
    - parameter sweep
    - tearsheet-only generation from existing CSV results
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

import pandas as pd

from quantedge.config import Config
from quantedge.data import ensure_results_dir, load_prices, set_logging
from quantedge.metrics import compute_metrics, format_metrics_table
from quantedge.portfolio import compute_turnover_summary, run_backtest
from quantedge.sweep import print_top_n, run_parameter_sweep
from quantedge.tearsheet import (
    TearsheetInputs,
    generate_tearsheet_pdf,
    load_avg_turnover_from_results,
    load_returns_from_results,
)
from quantedge.visualisation import plot_factor_decay, plot_factor_ic, plot_results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="QuantEdge: dual-factor long/short equity backtest toolkit")
    p.add_argument("--mode", choices=["backtest", "sweep", "tearsheet"], required=True)

    p.add_argument("--use-real-data", action="store_true", help="Use yfinance adjusted close data (vs simulated).")
    p.add_argument("--start", type=str, default=Config().start, help="Start date YYYY-MM-DD (real data).")
    p.add_argument("--end", type=str, default=Config().end, help="End date YYYY-MM-DD (real data).")

    # Optional overrides (useful for ad-hoc runs).
    p.add_argument("--momentum-lookback", type=int, default=None)
    p.add_argument("--momentum-skip", type=int, default=None)
    p.add_argument("--mean-rev-lookback", type=int, default=None)
    p.add_argument("--weight-momentum", type=float, default=None)
    p.add_argument("--rebal-freq", type=int, default=None)
    return p.parse_args()


def build_cfg(args: argparse.Namespace) -> Config:
    """Build Config from defaults + CLI overrides."""
    cfg = Config(start=args.start, end=args.end)
    if args.momentum_lookback is not None:
        cfg = replace(cfg, momentum_lookback=args.momentum_lookback)
    if args.momentum_skip is not None:
        cfg = replace(cfg, momentum_skip=args.momentum_skip)
    if args.mean_rev_lookback is not None:
        cfg = replace(cfg, mean_rev_lookback=args.mean_rev_lookback)
    if args.weight_momentum is not None:
        wm = float(args.weight_momentum)
        cfg = replace(cfg, weight_momentum=wm, weight_mean_rev=1.0 - wm)
    if args.rebal_freq is not None:
        cfg = replace(cfg, rebal_freq=args.rebal_freq)
    return cfg


def run_backtest_mode(cfg: Config, use_real_data: bool) -> None:
    """Run full backtest pipeline and generate all charts + tearsheet."""
    ensure_results_dir(cfg.results_dir)

    print("\n" + "=" * 66)
    print("  QUANTEDGE BACKTEST  |  Dual-Factor Momentum + Mean Reversion")
    print("=" * 66)

    print("\n[1/5] Loading data...")
    prices = load_prices(cfg, use_real_data=use_real_data)
    print(f"      {len(prices.columns)} stocks  -  {len(prices)} trading days")

    print("[2/5] Running backtest...")
    result = run_backtest(prices, cfg)

    strat_m = compute_metrics(result.portfolio_returns)
    bm_m = compute_metrics(result.benchmark_returns)
    print(format_metrics_table(strat_m, bm_m))
    avg_to = compute_turnover_summary(result.turnover_history)
    print(f"  {'Avg Monthly Turnover':<20} {avg_to:>13.1%} {'-':>14}")

    print("\n[3/5] Generating charts...")
    plot_results(result, cfg, save_path=os.path.join(cfg.results_dir, "performance.png"))
    plot_factor_ic(prices, cfg, save_path=os.path.join(cfg.results_dir, "factor_ic.png"))
    plot_factor_decay(prices, cfg, save_path=os.path.join(cfg.results_dir, "factor_decay.png"))

    print("[4/5] Saving data...")
    result.portfolio_returns.to_csv(os.path.join(cfg.results_dir, "daily_returns.csv"), header=["return"])
    result.benchmark_returns.to_csv(os.path.join(cfg.results_dir, "benchmark_returns.csv"), header=["return"])
    result.turnover_history.to_csv(os.path.join(cfg.results_dir, "turnover.csv"), header=["turnover"])
    print("  [OK] Returns    -> results/daily_returns.csv")
    print("  [OK] Benchmark  -> results/benchmark_returns.csv")
    print("  [OK] Turnover   -> results/turnover.csv")

    print("[5/5] Generating PDF tearsheet...")
    pdf_path = os.path.join(cfg.results_dir, "tearsheet.pdf")
    inputs = TearsheetInputs(
        strategy_returns=result.portfolio_returns,
        benchmark_returns=result.benchmark_returns,
        universe_size=len(prices.columns),
        start_date=str(result.portfolio_returns.index.min().date()),
        end_date=str(result.portfolio_returns.index.max().date()),
        avg_turnover=avg_to,
    )
    generate_tearsheet_pdf(cfg, inputs, pdf_path)
    print(f"  [OK] Tearsheet  -> {pdf_path}")

    print("\nComplete. All outputs in ./results/\n")


def run_sweep_mode(cfg: Config) -> None:
    """Run parameter sweep on simulated universe and save outputs."""
    ensure_results_dir(cfg.results_dir)
    df = run_parameter_sweep(cfg, results_dir=cfg.results_dir)
    print_top_n(df, n=5)
    print(f"\n[OK] Saved sweep CSV -> {cfg.results_dir}/parameter_sweep.csv")
    print(f"[OK] Saved heatmap  -> {cfg.results_dir}/parameter_sweep_heatmap.png\n")


def run_tearsheet_mode(cfg: Config) -> None:
    """Generate tearsheet only from existing results CSV files."""
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


def main() -> None:
    set_logging()
    args = parse_args()
    cfg = build_cfg(args)

    if args.mode == "backtest":
        run_backtest_mode(cfg, use_real_data=args.use_real_data)
    elif args.mode == "sweep":
        run_sweep_mode(cfg)
    else:
        run_tearsheet_mode(cfg)


if __name__ == "__main__":
    main()

