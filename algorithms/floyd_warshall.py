import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_stocks import get_stock_data

# ── Floyd-Warshall Diversification ────────────────────────────────────────────
#
# The core idea: two highly correlated stocks move together — owning both
# doesn't reduce your risk the way two uncorrelated stocks would.
# We want to penalize (remove) stocks that are too similar to others
# already in the portfolio.
#
# Why Floyd-Warshall?
#   We model stocks as nodes in a graph. Edge weight between stock A and B
#   is their "distance" — how DIFFERENT they are. We want to find the most
#   diversified (most distant) portfolio.
#   Floyd-Warshall gives us ALL-PAIRS shortest distances in one pass,
#   letting us see every pair's relationship, not just adjacent ones.
#
# CRITICAL — the distance transform:
#   Correlation is a SIMILARITY metric (1 = identical, 0 = unrelated, -1 = opposite)
#   Floyd-Warshall needs a DISTANCE metric (0 = identical, larger = more different)
#   So we transform:  distance = 1 - correlation
#   This maps: correlation 1.0 → distance 0.0  (same stock, zero distance)
#              correlation 0.0 → distance 1.0  (unrelated)
#              correlation -1  → distance 2.0  (perfectly opposite)
#
# Time  Complexity: O(n³) — feasible for n ≤ 50 stocks
# Space Complexity: O(n²) — the distance matrix


CORRELATION_THRESHOLD = 0.85   # pairs with correlation above this are "too similar"
INF = float("inf")


# ── Step 1: Build Correlation + Distance Matrices ─────────────────────────────

def build_distance_matrix(log_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise Pearson correlation between all stock log return series,
    then convert to a distance matrix using the (1 - correlation) transform.

    Args:
        log_returns (pd.DataFrame): Daily log returns, columns = tickers

    Returns:
        corr_matrix (pd.DataFrame): Raw Pearson correlations
        dist_matrix (pd.DataFrame): Distance = 1 - correlation (values in [0, 2])
    """
    corr_matrix = log_returns.corr(method="pearson")
    dist_matrix = 1 - corr_matrix    # distance transform — THIS is the key step
    return corr_matrix, dist_matrix


# ── Step 2: Floyd-Warshall All-Pairs Shortest Path ───────────────────────────

def floyd_warshall(dist_matrix: pd.DataFrame) -> np.ndarray:
    """
    Run Floyd-Warshall on the distance matrix to find the shortest path
    between every pair of stocks.

    Shortest path here means: the most "similar" multi-hop route between
    two stocks. A short path A→C→B means A and B are likely correlated
    through a common third stock C.

    Args:
        dist_matrix (pd.DataFrame): n×n distance matrix (1 - correlation)

    Returns:
        fw_dist (np.ndarray): n×n matrix of all-pairs shortest distances
    """
    n = len(dist_matrix)
    # Make a float copy — we'll modify it in-place
    fw_dist = dist_matrix.values.astype(float).copy()

    # Floyd-Warshall triple loop:
    #   For every intermediate node k, check if going i→k→j is shorter than i→j.
    #   If yes, update the distance.
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if fw_dist[i][k] + fw_dist[k][j] < fw_dist[i][j]:
                    fw_dist[i][j] = fw_dist[i][k] + fw_dist[k][j]

    return fw_dist


# ── Step 3: Identify Highly Correlated Pairs ──────────────────────────────────

def find_correlated_pairs(
    corr_matrix: pd.DataFrame,
    selected_tickers: list,
    threshold: float = CORRELATION_THRESHOLD
) -> list[tuple]:
    """
    Among the selected portfolio stocks, find pairs whose Pearson correlation
    exceeds the threshold. These pairs need diversification enforcement.

    Args:
        corr_matrix      (pd.DataFrame): Full pairwise correlation matrix
        selected_tickers (list)        : Stocks in the current portfolio
        threshold        (float)       : Correlation cutoff (default 0.85)

    Returns:
        List of (ticker_a, ticker_b, correlation_value) tuples, sorted by
        correlation descending. Only includes pairs where both tickers are
        present in corr_matrix.
    """
    # Filter corr_matrix to only the selected stocks that exist in it
    available = [t for t in selected_tickers if t in corr_matrix.columns]
    sub = corr_matrix.loc[available, available]

    pairs = []
    tickers = list(sub.columns)
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):   # upper triangle only, no duplicates
            corr_val = sub.iloc[i, j]
            if corr_val > threshold:
                pairs.append((tickers[i], tickers[j], round(corr_val, 4)))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


# ── Step 4: Enforce Diversification ──────────────────────────────────────────

def enforce_diversification(
    selected_tickers: list,
    metrics: dict,
    corr_matrix: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD
) -> tuple[list, list]:
    """
    Remove stocks that are too correlated with others already in the portfolio.

    Strategy: Find all pairs with correlation > threshold. For each such pair,
    drop the stock with the LOWER return/risk ratio — it's the less efficient one.
    Repeat until no over-correlated pairs remain.

    Args:
        selected_tickers (list)        : Current portfolio tickers
        metrics          (dict)        : {ticker: {expected_return, risk, price}}
        corr_matrix      (pd.DataFrame): Full pairwise correlation matrix
        threshold        (float)       : Correlation cutoff

    Returns:
        cleaned    (list): Tickers after removing over-correlated stocks
        penalized  (list): Tickers that were removed, with reason
    """
    current    = list(selected_tickers)   # working copy
    penalized  = []

    while True:
        pairs = find_correlated_pairs(corr_matrix, current, threshold)
        if not pairs:
            break   # no more over-correlated pairs — done

        # Penalize the worst pair first (highest correlation)
        ticker_a, ticker_b, corr_val = pairs[0]

        # Compute return/risk ratio for both — remove the weaker one
        ratio_a = metrics[ticker_a]["expected_return"] / max(metrics[ticker_a]["risk"], 1e-9)
        ratio_b = metrics[ticker_b]["expected_return"] / max(metrics[ticker_b]["risk"], 1e-9)

        to_remove = ticker_b if ratio_a >= ratio_b else ticker_a
        to_keep   = ticker_a if to_remove == ticker_b else ticker_b

        current.remove(to_remove)
        penalized.append({
            "removed"    : to_remove,
            "kept"       : to_keep,
            "correlation": corr_val,
            "reason"     : f"Correlation {corr_val:.2%} > threshold {threshold:.0%}. "
                           f"Kept {to_keep} (higher return/risk ratio)."
        })

    return current, penalized


# ── Public Interface ──────────────────────────────────────────────────────────

def diversify_portfolio(
    selected_tickers: list,
    metrics: dict,
    log_returns: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD
) -> dict:
    """
    Full diversification pipeline:
      1. Build correlation + distance matrices
      2. Run Floyd-Warshall for all-pairs shortest distances
      3. Identify correlated pairs in the selected portfolio
      4. Enforce diversification by penalizing weak correlated stocks

    Args:
        selected_tickers (list)        : Portfolio from greedy or DP
        metrics          (dict)        : {ticker: {expected_return, risk, price}}
        log_returns      (pd.DataFrame): Daily log returns (all stocks)
        threshold        (float)       : Correlation cutoff

    Returns:
        result (dict): {
            "original"        : original selected tickers,
            "diversified"     : tickers after diversification,
            "penalized"       : list of removed stocks with reasons,
            "corr_matrix"     : pd.DataFrame,
            "fw_distances"    : np.ndarray (all-pairs shortest distances),
            "total_return"    : float (return of diversified portfolio),
            "total_invested"  : float
        }
    """
    # Only compute correlation for tickers that are present in log_returns
    available_cols = [t for t in log_returns.columns if t in selected_tickers or True]
    corr_matrix, dist_matrix = build_distance_matrix(log_returns[available_cols])

    fw_distances = floyd_warshall(dist_matrix)

    cleaned, penalized = enforce_diversification(
        selected_tickers, metrics, corr_matrix, threshold
    )

    # Recompute totals for the diversified portfolio
    total_return   = sum(metrics[t]["expected_return"] for t in cleaned if t in metrics)
    total_invested = sum(metrics[t]["price"]           for t in cleaned if t in metrics)

    return {
        "original"      : selected_tickers,
        "diversified"   : cleaned,
        "penalized"     : penalized,
        "corr_matrix"   : corr_matrix,
        "fw_distances"  : fw_distances,
        "total_return"  : round(total_return, 6),
        "total_invested": round(total_invested, 2),
    }


# ── Manual Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    NIFTY_SYMBOLS = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "HINDUNILVR", "ITC", "LT", "SBIN", "BHARTIARTL",
        "WIPRO", "HCLTECH", "ASIANPAINT", "MARUTI", "BAJFINANCE",
        "KOTAKBANK", "AXISBANK", "NESTLEIND", "TITAN", "SUNPHARMA"
    ]

    metrics, log_returns = get_stock_data(NIFTY_SYMBOLS, use_cache=True)

    # Test with a sample selection — in real use, this comes from greedy or DP
    sample_selected = list(metrics.keys())[:10]
    print(f"Testing diversification on: {sample_selected}\n")

    result = diversify_portfolio(sample_selected, metrics, log_returns)

    print("── Penalized (over-correlated) stocks ──────────────────────────")
    if result["penalized"]:
        for p in result["penalized"]:
            print(f"  Removed: {p['removed']:20s}  |  {p['reason']}")
    else:
        print("  None — all selected stocks are well-diversified.")

    print(f"\nOriginal  portfolio: {result['original']}")
    print(f"Diversified portfolio: {result['diversified']}")
    print(f"Total return (diversified): {result['total_return']:.2%}")
    print(f"Total invested (diversified): ₹{result['total_invested']:,.2f}")

    print(f"\nCorrelation matrix shape : {result['corr_matrix'].shape}")
    print(f"Floyd-Warshall dist shape: {result['fw_distances'].shape}")
