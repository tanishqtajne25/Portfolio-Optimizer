import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_stocks import get_stock_data
from algorithms.greedy import greedy_select
from algorithms.knapsack_dp import knapsack_dp

# ── Greedy vs DP Comparison ───────────────────────────────────────────────────
#
# Purpose: Empirically demonstrate the trade-off between Greedy and DP
# across different budget levels.
#
# Key insight to verify experimentally:
#   "Greedy performs close to optimal when budget is large (many options fit),
#    but deviates significantly when budget is tight (item combinations matter)."
#
# Why does this happen?
#   With a large budget, Greedy can fit most of the high-ratio stocks anyway,
#   so the greedy ordering closely matches the DP selection.
#   With a tight budget, the exact combinations of stock prices matter a lot —
#   Greedy may pick an expensive stock that wastes budget, while DP finds a
#   combination of cheaper stocks with higher total return.


def compare_at_budget(metrics: dict, budget: float) -> dict:
    """
    Run both Greedy and DP for a single budget level and return their results
    side-by-side for comparison.

    Args:
        metrics (dict): {ticker: {expected_return, risk, price}}
        budget  (float): Budget in ₹

    Returns:
        dict with keys: budget, greedy_return, dp_return, improvement_pct,
                        greedy_selected, dp_selected, greedy_invested, dp_invested
    """
    greedy_result = greedy_select(metrics, budget)
    dp_result     = knapsack_dp(metrics, budget)

    greedy_ret = greedy_result["total_return"]
    dp_ret     = dp_result["total_return"]

    # Improvement = how much better DP is over Greedy, as a percentage.
    # We add a small epsilon to avoid division by zero when both return 0.
    if abs(greedy_ret) < 1e-9:
        improvement_pct = 0.0
    else:
        improvement_pct = ((dp_ret - greedy_ret) / abs(greedy_ret)) * 100

    return {
        "budget"          : budget,
        "greedy_return"   : greedy_ret,
        "dp_return"       : dp_ret,
        "improvement_pct" : round(improvement_pct, 2),
        "greedy_selected" : greedy_result["selected"],
        "dp_selected"     : dp_result["selected"],
        "greedy_invested" : greedy_result["total_invested"],
        "dp_invested"     : dp_result["total_invested"],
    }


def compare_across_budgets(metrics: dict, budgets: list[float]) -> list[dict]:
    """
    Run Greedy vs DP comparison across multiple budget levels.

    Args:
        metrics (dict)      : {ticker: {expected_return, risk, price}}
        budgets (list[float]): Budget levels in ₹ (e.g. [20000, 50000, 100000, 200000])

    Returns:
        List of comparison dicts, one per budget level.
    """
    results = []
    for budget in budgets:
        row = compare_at_budget(metrics, budget)
        results.append(row)
    return results


def print_comparison_table(results: list[dict]) -> None:
    """
    Pretty-print the comparison table to the terminal.
    This is what you show in your project demo or README.
    """
    header = f"{'Budget':>12}  {'Greedy Return':>14}  {'DP Return':>10}  {'Improvement':>12}  {'Greedy N':>9}  {'DP N':>6}"
    print("\n── Greedy vs DP Comparison ─────────────────────────────────────────────────")
    print(header)
    print("-" * 72)

    for r in results:
        print(
            f"₹{r['budget']:>10,.0f}"
            f"  {r['greedy_return']:>13.2%}"
            f"  {r['dp_return']:>9.2%}"
            f"  {r['improvement_pct']:>+11.2f}%"
            f"  {len(r['greedy_selected']):>9}"
            f"  {len(r['dp_selected']):>6}"
        )

    print("\nNotes:")
    print("  'Return' = sum of annualized expected log returns of selected stocks.")
    print("  'N' = number of stocks selected.")
    print("  Improvement = (DP - Greedy) / |Greedy| × 100%.")
    print("  Positive improvement means DP found a better combination.")

    # Highlight the key insight
    low_budget  = results[0]["improvement_pct"]
    high_budget = results[-1]["improvement_pct"]
    print(f"\n── Key Insight ──────────────────────────────────────────────────────────────")
    print(f"  At tight budget  (₹{results[0]['budget']:,.0f}): DP improves on Greedy by {low_budget:+.2f}%")
    print(f"  At large budget  (₹{results[-1]['budget']:,.0f}): DP improves on Greedy by {high_budget:+.2f}%")
    if abs(low_budget) > abs(high_budget):
        print("  → Greedy deviates more at tight budgets, confirming expected behavior.")
    else:
        print("  → Gap is modest here — may vary with stock universe and price distribution.")


# ── Manual Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    NIFTY_SYMBOLS = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "HINDUNILVR", "ITC", "LT", "SBIN", "BHARTIARTL",
        "WIPRO", "HCLTECH", "ASIANPAINT", "MARUTI", "BAJFINANCE",
        "KOTAKBANK", "AXISBANK", "NESTLEIND", "TITAN", "SUNPHARMA"
    ]

    metrics, _ = get_stock_data(NIFTY_SYMBOLS, use_cache=True)

    BUDGETS = [20_000, 50_000, 1_00_000, 2_00_000]

    results = compare_across_budgets(metrics, BUDGETS)
    print_comparison_table(results)

    # Also print which stocks each algorithm picked at ₹50,000
    r50k = results[1]
    print(f"\n── Portfolio Breakdown at ₹50,000 ──────────────────────────────────────────")
    print(f"Greedy selected : {r50k['greedy_selected']}")
    print(f"DP     selected : {r50k['dp_selected']}")
    overlap = set(r50k["greedy_selected"]) & set(r50k["dp_selected"])
    print(f"Common stocks   : {sorted(overlap)}")
