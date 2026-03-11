# 📈 QuantEdge — Dual-Factor L/S Equity Strategy

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-2.4%2B-013243?style=flat-square&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-3.0%2B-150458?style=flat-square&logo=pandas)
![Scipy](https://img.shields.io/badge/SciPy-1.0%2B-8CAAE6?style=flat-square&logo=scipy)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A systematic long/short equity backtesting framework combining **12-1 month price momentum** with **5-day mean reversion** into a composite factor signal. Dollar-neutral portfolio construction, monthly rebalancing, full IC/ICIR analysis, 81-combination parameter sweep, factor decay analysis, and a PDF tearsheet — all driven by a 3-mode CLI.

---

## 🧠 Motivation

Most algorithmic trading projects focus on single-stock prediction or simple moving average crossovers. This project is built to reflect how **buy-side quant desks actually operate** — working with a cross-sectional universe, constructing factor signals, running a proper long/short backtest with realistic frictions, and evaluating signal quality through IC analysis rather than just eyeballing a returns chart.

The central research question: *can momentum and mean reversion — two seemingly contradictory effects — complement each other when combined at different time horizons?*

---

## 🔬 Strategy

The strategy scores a 100-stock universe daily using two cross-sectional factors, rebalances monthly, and holds the top/bottom decile in a dollar-neutral long/short book.

### Factor 1 — 12-1 Month Momentum (75%)

```
MOM_t = price_{t-21} / price_{t-252} - 1
```

Trailing 12-month return, skipping the most recent month. The skip avoids the well-documented short-term reversal effect — stocks that ran up strongly in the last few weeks tend to revert immediately rather than continue trending.

### Factor 2 — 5-Day Mean Reversion (25%)

```
MR_t = -(price_t / price_{t-5} - 1)
```

Negative sign intentional: the signal favours stocks that have recently pulled back within an ongoing momentum trend, improving entry timing and reducing return-chasing.

### Signal Combination

Both factors are **cross-sectionally z-scored** (normalised relative to the full universe on each date) before blending:

```
Composite_t = 0.75 × MOM_z + 0.25 × MR_z
```

### Portfolio Construction

| Leg | Selection | Weighting |
|-----|-----------|-----------|
| Long | Top 10% composite score (10 stocks) | Equal-weight |
| Short | Bottom 10% composite score (10 stocks) | Equal-weight |

Dollar-neutral: long Σw = +1, short Σw = −1. Transaction costs of **10 bps per side** applied on turnover at each monthly rebalance.

---

## 📊 Performance

Backtest period: 2019–2023 | Universe: 100 stocks | Rebalance: Monthly

| Metric | L/S Strategy | EW Benchmark |
|--------|-------------|--------------|
| Annualised Return | ~1% | ~33% |
| Annualised Volatility | ~10% | ~18% |
| Sharpe Ratio | 0.07 | 1.87 |
| Max Drawdown | −23% | −20% |
| Avg Monthly Turnover | ~92% | — |

> **Why does L/S underperform the benchmark?** This is expected and by design. A dollar-neutral book surrenders directional beta exposure — it's not built to beat a long-only benchmark in a bull market. Its purpose is to generate **alpha uncorrelated with market direction**, evaluated on Sharpe and IC rather than absolute return. In a real portfolio, this strategy would sit as a **market-neutral diversifying sleeve**.

---

## 🔍 IC / ICIR Analysis

The **Information Coefficient** (Spearman rank correlation between factor score and next-month return) measures signal predictive power independently of portfolio construction choices.

| Metric | Interpretation |
|--------|----------------|
| Mean Monthly IC | Positive → factor is directionally correct on average |
| ICIR = Mean IC / StdDev IC | Signal consistency — higher = more reliable |
| Cumulative IC slope | Upward → signal hasn't decayed over the backtest period |

---

## 🔁 Parameter Sweep

An 81-combination grid search across key parameters reveals how sensitive the strategy is to configuration choices:

| Parameter | Values Tested |
|-----------|--------------|
| `momentum_lookback` | 126, 189, 252 (6, 9, 12 months) |
| `momentum_skip` | 0, 10, 21 days |
| `mean_rev_lookback` | 3, 5, 10 days |
| `weight_momentum` | 0.5, 0.75, 1.0 |

**Key finding:** 6-month momentum (`lookback=126`) with no skip significantly outperformed the base 12-month config, achieving a Sharpe of **1.52** vs **0.07** — suggesting momentum signal decay is faster in recent market regimes than classic academic literature (Jegadeesh & Titman, 1993) implies.

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/quantedge.git
cd quantedge

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Usage

The project is driven by a 3-mode CLI via `main.py`:

### Mode 1 — Full Backtest
```bash
python main.py --mode backtest
```
Runs the complete pipeline: data simulation → factor construction → backtest → all charts → PDF tearsheet.

### Mode 2 — Parameter Sweep
```bash
python main.py --mode sweep
```
Runs 81 parameter combinations, prints top 5 by Sharpe, saves heatmap and CSV.

### Mode 3 — Tearsheet Only
```bash
python main.py --mode tearsheet
```
Regenerates the PDF tearsheet from existing `results/daily_returns.csv` without re-running the full backtest.

### Optional: Real Market Data
```bash
python main.py --mode backtest --use-real-data
```
Fetches live adjusted close prices via `yfinance` for the top 30 S&P 500 stocks instead of simulated data.

---

## 📁 Project Structure

```
quantedge/
├── main.py                      # CLI entry point (--mode backtest/sweep/tearsheet)
├── strategy.py                  # Legacy single-file runner
│
├── quantedge/                   # Core package
│   ├── __init__.py
│   ├── config.py                # All constants and configuration
│   ├── data.py                  # simulate_universe() + load_universe() via yfinance
│   ├── factors.py               # compute_momentum(), compute_mean_reversion(), compute_composite()
│   ├── portfolio.py             # build_weights(), run_backtest(), BacktestResult
│   ├── metrics.py               # compute_metrics() — Sharpe, Calmar, drawdown, IC/ICIR
│   ├── visualisation.py         # All chart functions with dark-theme matplotlib
│   └── tearsheet.py             # PDF tearsheet generator via fpdf2
│
├── parameter_sweep.py           # Standalone 81-combo grid search
│
├── results/                     # All outputs (auto-created on first run)
│   ├── performance.png          # 4-panel performance dashboard
│   ├── factor_ic.png            # IC bar chart + cumulative IC
│   ├── factor_decay.png         # IC vs holding period (1–63 days)
│   ├── parameter_sweep.png      # Sharpe heatmap across parameter grid
│   ├── parameter_sweep.csv      # Full sweep results table
│   ├── tearsheet.pdf            # Single-page PDF performance summary
│   ├── daily_returns.csv        # Daily net portfolio returns
│   ├── benchmark_returns.csv    # Equal-weighted benchmark returns
│   └── turnover.csv             # Monthly rebalancing turnover
│
├── requirements.txt
└── README.md
```

---

## 📦 Output Files

| File | Description |
|------|-------------|
| `performance.png` | 4-panel dashboard: cumulative returns vs benchmark, drawdown, rolling Sharpe, monthly heatmap |
| `factor_ic.png` | Monthly IC bar chart with mean IC line + cumulative IC trend |
| `factor_decay.png` | Signal IC across holding periods (1, 5, 10, 21, 42, 63 days) |
| `parameter_sweep.png` | Sharpe ratio heatmap: momentum lookback × signal weight |
| `parameter_sweep.csv` | Full 81-row results table with all metrics per combination |
| `tearsheet.pdf` | One-page PDF: metrics table, returns chart, drawdown, monthly heatmap |
| `daily_returns.csv` | Daily net-of-cost portfolio returns |
| `turnover.csv` | Monthly turnover at each rebalance date |

---

## 📚 References

- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers.* Journal of Finance.
- Asness, C., Moskowitz, T. & Pedersen, L. (2013). *Value and Momentum Everywhere.* Journal of Finance.
- Grinold, R. & Kahn, R. (1999). *Active Portfolio Management.* McGraw-Hill.

---

## 🙋 About

Built as a personal deep-dive into systematic, factor-based investing — going beyond simple price prediction toward how institutional quant desks construct and evaluate strategies. The goal was to understand not just *whether* a factor works, but *why* it works and how to measure that rigorously through IC, ICIR, factor decay, and parameter robustness testing.