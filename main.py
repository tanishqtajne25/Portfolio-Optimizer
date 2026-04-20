"""
NiftyOpt — NIFTY 50 Portfolio Optimizer
========================================
A portfolio optimizer that uses three algorithms from the DAA course:
  1. Greedy         — fast approximate selection by return/risk ratio  O(n log n)
  2. 0/1 Knapsack   — exact DP solution with 1D space optimization      O(n × W)
  3. Floyd-Warshall — all-pairs diversification via distance matrix      O(n³)

This is a SIMULATION on historical data — not a live trading system.

Usage:
    # Launch Streamlit UI (recommended)
    streamlit run ui/app.py

    # Run CLI demo
    python main.py

    # Run individual modules
    python data/fetch_stocks.py
    python algorithms/greedy.py
    python algorithms/knapsack_dp.py
    python algorithms/floyd_warshall.py
    python algorithms/compare.py

    # Run tests
    pytest tests/
"""

import sys
import os

# Add project root to path so all imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fetch_stocks import get_stock_data
from algorithms.greedy import greedy_select
from algorithms.knapsack_dp import knapsack_dp
from algorithms.floyd_warshall import diversify_portfolio
from algorithms.compare import compare_across_budgets, print_comparison_table

# ── Stock Universe ─────────────────────────────────────────────────────────────
NIFTY_SYMBOLS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "HINDUNILVR", "ITC", "LT", "SBIN", "BHARTIARTL",
    "WIPRO", "HCLTECH", "ASIANPAINT", "MARUTI", "BAJFINANCE",
    "KOTAKBANK", "AXISBANK", "NESTLEIND", "TITAN", "SUNPHARMA"
]

BUDGET = 50_000


def print_portfolio(label: str, result: dict) -> None:
    print(f"\n── {label} ────────────────────────────────────────────")
    print(f"  Selected : {[t.replace('.NS','') for t in result['selected']]}")
    print(f"  Invested : ₹{result['total_invested']:,.2f}")
    print(f"  Remaining: ₹{result['remaining_budget']:,.2f}")
    print(f"  Return   : {result['total_return']:.2%}")


if __name__ == "__main__":
    print("=" * 60)
    print("  NiftyOpt — NIFTY 50 Portfolio Optimizer")
    print("  (Simulation on historical data)")
    print("=" * 60)

    # ── Load Data ──────────────────────────────────────────────────────────────
    print(f"\nLoading data for {len(NIFTY_SYMBOLS)} NIFTY stocks…")
    metrics, log_returns = get_stock_data(NIFTY_SYMBOLS, use_cache=True)
    print(f"Loaded {len(metrics)} stocks.\n")

    # ── Greedy ─────────────────────────────────────────────────────────────────
    greedy_result = greedy_select(metrics, BUDGET)
    print_portfolio(f"Greedy Portfolio  (Budget: ₹{BUDGET:,.0f})", greedy_result)

    # ── DP ─────────────────────────────────────────────────────────────────────
    dp_result = knapsack_dp(metrics, BUDGET)
    print_portfolio(f"DP Portfolio      (Budget: ₹{BUDGET:,.0f})", dp_result)

    # ── Diversification on DP portfolio ────────────────────────────────────────
    div_result = diversify_portfolio(dp_result["selected"], metrics, log_returns)
    print(f"\n── Diversification (Floyd-Warshall, threshold=0.85) ─────────")
    if div_result["penalized"]:
        for p in div_result["penalized"]:
            print(f"  Removed {p['removed'].replace('.NS',''):15s} | {p['reason']}")
    else:
        print("  No highly correlated pairs found.")
    print(f"  Diversified portfolio: {[t.replace('.NS','') for t in div_result['diversified']]}")

    # ── Comparison ─────────────────────────────────────────────────────────────
    results = compare_across_budgets(metrics, [20_000, 50_000, 1_00_000, 2_00_000])
    print_comparison_table(results)

    print("\n\nTo launch the full interactive UI, run:")
    print("  streamlit run ui/app.py\n")
