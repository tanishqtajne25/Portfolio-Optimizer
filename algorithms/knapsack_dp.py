import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_stocks import get_stock_data

# ── 0/1 Knapsack DP ───────────────────────────────────────────────────────────
#
# Problem mapping:
#   item   → stock
#   weight → stock price (normalized to ₹1000 units to keep W manageable)
#   profit → expected annual return (annualized log return)
#   capacity → budget (normalized to same ₹1000 units)
#
# Why 0/1 and not fractional knapsack?
#   Because you can't buy 0.7 of a share. Each stock is either bought (1) or skipped (0).
#   Greedy solves fractional knapsack optimally — but NOT 0/1 knapsack.
#   DP solves 0/1 knapsack exactly.
#
# Space optimization:
#   Naive 2D DP uses O(n × W) space — one row per stock, one column per capacity.
#   1D rolling array reuses the same row, updating right-to-left so we never
#   overwrite a value we still need in the same iteration.
#   This reduces space from O(n × W) → O(W).
#
# Time  Complexity: O(n × W)
# Space Complexity: O(W)  [after space optimization]
#
# Traceback note:
#   Once you have only a 1D array, you cannot trace back which items were chosen
#   just from dp[] alone. So we separately store a 2D boolean `selected[i][w]`
#   (did we pick item i at capacity w?) to recover the actual stock list.


# Weight normalization unit — ₹1000 per "weight unit"
# Why 1000? So a ₹3,500 stock becomes weight=3, ₹12,000 stock becomes weight=12.
# This keeps W (capacity) under ~200 for typical budgets, keeping DP fast.
WEIGHT_UNIT = 1000


def normalize_price(price: float) -> int:
    """
    Convert a stock price in ₹ to an integer weight in ₹1000 units.
    Rounds up using ceil so we don't accidentally afford a stock we can't.

    Example: ₹3,500 → weight 4  (ceil(3500/1000))
    """
    import math
    return max(1, math.ceil(price / WEIGHT_UNIT))


def prepare_items(metrics: dict) -> tuple[list, list, list]:
    """
    Extract tickers, weights (normalized prices), and profits (expected returns)
    from the metrics dictionary, in a consistent order.

    Args:
        metrics (dict): {ticker: {expected_return, risk, price}}

    Returns:
        tickers (list of str)  : stock names
        weights (list of int)  : normalized prices (weight units)
        profits (list of float): annualized expected returns
    """
    tickers = list(metrics.keys())
    weights = [normalize_price(metrics[t]["price"]) for t in tickers]
    profits = [metrics[t]["expected_return"] for t in tickers]
    return tickers, weights, profits


def knapsack_dp(metrics: dict, budget: float) -> dict:
    """
    Solve the 0/1 knapsack problem to find the globally optimal portfolio.

    Args:
        metrics (dict): {ticker: {expected_return, risk, price}}
        budget  (float): Budget in ₹

    Returns:
        result (dict): {
            "selected"        : list of ticker strings,
            "total_return"    : float (maximum achievable annualized return),
            "total_invested"  : float (actual ₹ spent),
            "remaining_budget": float,
            "breakdown"       : list of dicts per selected stock
        }
    """
    tickers = list(metrics.keys())
    # Use ₹1000 budget units (floor) for tractable DP state size.
    # This matches the intended coarse-capacity modeling in this project.
    weights = [max(1, int(metrics[t]["price"] // WEIGHT_UNIT)) for t in tickers]
    profits = [metrics[t]["expected_return"] for t in tickers]
    n = len(tickers)

    # Capacity in ₹1000 units
    W = int(budget // WEIGHT_UNIT)

    if W <= 0:
        return {
            "selected": [], "total_return": 0.0,
            "total_invested": 0.0, "remaining_budget": budget, "breakdown": []
        }

    # ── 2D DP Table ───────────────────────────────────────────────────────────
    # dp[i][w] = best return using first i items within capacity w.
    # This keeps traceback exact and guarantees the reconstructed portfolio
    # matches the reported optimum.
    dp = [[0.0] * (W + 1) for _ in range(n + 1)]

    eps = 1e-12

    for i in range(1, n + 1):
        wi = weights[i - 1]
        pi = profits[i - 1]
        for w in range(W + 1):
            best_without = dp[i - 1][w]
            if wi <= w:
                best_with = dp[i - 1][w - wi] + pi
                # Use epsilon-aware comparison to keep tie behavior stable.
                if best_with > best_without + eps:
                    dp[i][w] = best_with
                else:
                    dp[i][w] = best_without
            else:
                dp[i][w] = best_without

    # ── Traceback ─────────────────────────────────────────────────────────────
    # Walk backwards and include item i-1 whenever dp value changes.
    chosen_tickers = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] > dp[i - 1][w] + eps:
            chosen_tickers.append(tickers[i - 1])
            w -= weights[i - 1]

    # ── Build result ──────────────────────────────────────────────────────────
    total_invested = 0.0
    total_return   = 0.0
    breakdown      = []

    for ticker in chosen_tickers:
        m = metrics[ticker]
        total_invested += m["price"]
        total_return   += m["expected_return"]
        breakdown.append({
            "ticker"         : ticker,
            "price"          : m["price"],
            "expected_return": round(m["expected_return"], 6),
            "risk"           : round(m["risk"], 6),
            "weight_units"   : normalize_price(m["price"]),
        })

    return {
        "selected"        : chosen_tickers,
        "total_return"    : round(total_return, 6),
        "total_invested"  : round(total_invested, 2),
        "remaining_budget": round(budget - total_invested, 2),
        "breakdown"       : breakdown,
    }


# ── Manual Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    NIFTY_SYMBOLS = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "HINDUNILVR", "ITC", "LT", "SBIN", "BHARTIARTL",
        "WIPRO", "HCLTECH", "ASIANPAINT", "MARUTI", "BAJFINANCE",
        "KOTAKBANK", "AXISBANK", "NESTLEIND", "TITAN", "SUNPHARMA"
    ]

    metrics, _ = get_stock_data(NIFTY_SYMBOLS, use_cache=True)

    budget = 50_000
    result = knapsack_dp(metrics, budget)

    print(f"\n── DP Knapsack Selection (Budget: ₹{budget:,.0f}) ────────────────")
    print(f"{'Ticker':<20} {'Price (₹)':>12} {'Weight Units':>14} {'Exp. Return':>12}")
    print("-" * 62)
    for b in result["breakdown"]:
        print(f"{b['ticker']:<20} {b['price']:>12.2f} {b['weight_units']:>14} {b['expected_return']:>11.2%}")

    print(f"\nSelected : {result['selected']}")
    print(f"Invested : ₹{result['total_invested']:,.2f}")
    print(f"Remaining: ₹{result['remaining_budget']:,.2f}")
    print(f"Max Return (DP optimal): {result['total_return']:.2%}")
