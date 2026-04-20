import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_stocks import get_stock_data

# ── Greedy Stock Selection ────────────────────────────────────────────────────
#
# Strategy: Sort stocks by return-to-risk ratio (Sharpe-like score) descending.
# Then greedily pick stocks in that order until the budget is exhausted.
#
# Why Greedy works well here as a baseline:
#   - It is O(n log n) — extremely fast
#   - The return/risk ratio is a good proxy for "value per rupee of risk"
#   - But it is NOT guaranteed to be globally optimal — it can miss combinations
#     that fit better within the budget (the DP does that)
#
# Time Complexity : O(n log n)  — dominated by the sort
# Space Complexity: O(n)        — for the sorted list


def compute_return_risk_ratio(metrics: dict) -> list[tuple]:
    """
    For each stock, compute the return-to-risk ratio (annualized return / volatility).
    This is similar to the Sharpe ratio but without a risk-free rate subtracted.

    Args:
        metrics (dict): {ticker: {expected_return, risk, price}}

    Returns:
        List of (ticker, ratio, expected_return, risk, price) sorted by ratio descending.
        Stocks with zero or negative risk are excluded to avoid division by zero.
    """
    scored = []
    for ticker, m in metrics.items():
        if m["risk"] <= 0:
            print(f"[greedy] Skipping {ticker} — risk is zero or negative.")
            continue

        ratio = m["expected_return"] / m["risk"]
        scored.append((ticker, ratio, m["expected_return"], m["risk"], m["price"]))

    # Sort highest ratio first — these are the most "efficient" stocks
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def greedy_select(metrics: dict, budget: float) -> dict:
    """
    Greedily select stocks within a given budget using the return/risk ratio.

    A stock is either fully bought (1 share) or skipped entirely.
    No fractional shares — just like a real retail investor buying whole shares.

    Args:
        metrics (dict): {ticker: {expected_return, risk, price}}
        budget  (float): Maximum total spend in ₹

    Returns:
        result (dict): {
            "selected"        : list of ticker strings,
            "total_return"    : float (sum of annualized expected returns),
            "total_invested"  : float (total ₹ spent),
            "remaining_budget": float,
            "breakdown"       : list of dicts per selected stock
        }
    """
    scored = compute_return_risk_ratio(metrics)

    selected    = []
    total_invested = 0.0
    total_return   = 0.0
    breakdown      = []

    for ticker, ratio, exp_return, risk, price in scored:
        if total_invested + price <= budget:
            selected.append(ticker)
            total_invested += price
            total_return   += exp_return

            breakdown.append({
                "ticker"         : ticker,
                "price"          : price,
                "expected_return": round(exp_return, 6),
                "risk"           : round(risk, 6),
                "ratio"          : round(ratio, 4),
            })

    return {
        "selected"        : selected,
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
    result = greedy_select(metrics, budget)

    print(f"\n── Greedy Selection (Budget: ₹{budget:,.0f}) ─────────────────────")
    print(f"{'Ticker':<20} {'Price (₹)':>12} {'Exp. Return':>12} {'Ratio':>8}")
    print("-" * 56)
    for b in result["breakdown"]:
        print(f"{b['ticker']:<20} {b['price']:>12.2f} {b['expected_return']:>11.2%} {b['ratio']:>8.4f}")

    print(f"\nSelected : {result['selected']}")
    print(f"Invested : ₹{result['total_invested']:,.2f}")
    print(f"Remaining: ₹{result['remaining_budget']:,.2f}")
    print(f"Total Expected Return (sum): {result['total_return']:.2%}")
