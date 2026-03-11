"""
Performance metrics for QuantEdge.

Purpose:
    Compute standard strategy performance metrics from a daily return series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerfMetrics:
    """Numeric performance metrics (annualization assumes 252 trading days)."""

    total_return: float
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float


def compute_max_drawdown(returns: pd.Series) -> float:
    """Compute max drawdown from daily returns."""
    cum = (1.0 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return float(dd.min())


def compute_metrics(returns: pd.Series) -> PerfMetrics:
    """Compute performance metrics.

    Args:
        returns: Daily returns series.

    Returns:
        PerfMetrics with numeric values.
    """
    ann = 252
    returns = returns.dropna()
    if len(returns) == 0:
        return PerfMetrics(
            total_return=float("nan"),
            annual_return=float("nan"),
            annual_vol=float("nan"),
            sharpe=float("nan"),
            max_drawdown=float("nan"),
            calmar=float("nan"),
        )

    total_ret = float((1.0 + returns).prod() - 1.0)
    ann_ret = float((1.0 + total_ret) ** (ann / len(returns)) - 1.0)
    ann_vol = float(returns.std() * np.sqrt(ann))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else float("nan")
    max_dd = compute_max_drawdown(returns)
    calmar = float(ann_ret / abs(max_dd)) if max_dd != 0 else float("nan")
    return PerfMetrics(
        total_return=total_ret,
        annual_return=ann_ret,
        annual_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
    )


def format_metrics_table(strategy: PerfMetrics, benchmark: PerfMetrics) -> str:
    """Format a console-friendly metrics table."""
    rows = [
        ("Total Return", f"{strategy.total_return:.1%}", f"{benchmark.total_return:.1%}"),
        ("Ann. Return", f"{strategy.annual_return:.1%}", f"{benchmark.annual_return:.1%}"),
        ("Ann. Volatility", f"{strategy.annual_vol:.1%}", f"{benchmark.annual_vol:.1%}"),
        ("Sharpe Ratio", f"{strategy.sharpe:.2f}", f"{benchmark.sharpe:.2f}"),
        ("Max Drawdown", f"{strategy.max_drawdown:.1%}", f"{benchmark.max_drawdown:.1%}"),
        ("Calmar Ratio", f"{strategy.calmar:.2f}", f"{benchmark.calmar:.2f}"),
    ]
    out = []
    out.append(f"\n{'Metric':<22} {'L/S Strategy':>14} {'EW Benchmark':>14}")
    out.append("-" * 52)
    for k, v1, v2 in rows:
        out.append(f"  {k:<20} {v1:>14} {v2:>14}")
    out.append("-" * 52)
    return "\n".join(out)


def metrics_dict_for_sweep(metrics: PerfMetrics, avg_turnover: float) -> Dict[str, float]:
    """Flatten metrics for parameter sweep output CSV."""
    return {
        "ann_return": metrics.annual_return,
        "sharpe": metrics.sharpe,
        "max_drawdown": metrics.max_drawdown,
        "avg_turnover": avg_turnover,
        "calmar": metrics.calmar,
    }

