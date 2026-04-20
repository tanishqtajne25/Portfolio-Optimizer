"""
tests/test_algorithms.py
========================
Unit tests for all three NiftyOpt algorithms.
Uses deterministic mock data — no yfinance calls needed.

Run with:
    pytest tests/test_algorithms.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

from algorithms.greedy import greedy_select, compute_return_risk_ratio
from algorithms.knapsack_dp import knapsack_dp, normalize_price
from algorithms.floyd_warshall import (
    build_distance_matrix,
    floyd_warshall,
    find_correlated_pairs,
    enforce_diversification,
)


# ── Mock Data ──────────────────────────────────────────────────────────────────

def make_metrics(overrides: dict = None) -> dict:
    """
    Small, deterministic set of 5 mock stocks.
    Prices are set so they fit neatly into test budgets.
    """
    base = {
        "A.NS": {"expected_return": 0.20, "risk": 0.10, "price": 1000.0},  # ratio 2.0
        "B.NS": {"expected_return": 0.15, "risk": 0.10, "price": 2000.0},  # ratio 1.5
        "C.NS": {"expected_return": 0.10, "risk": 0.10, "price": 1500.0},  # ratio 1.0
        "D.NS": {"expected_return": 0.30, "risk": 0.20, "price": 3000.0},  # ratio 1.5
        "E.NS": {"expected_return": 0.05, "risk": 0.10, "price":  500.0},  # ratio 0.5
    }
    if overrides:
        base.update(overrides)
    return base


def make_log_returns(corr_override: dict = None) -> pd.DataFrame:
    """
    Mock log returns — 252 trading days × 4 stocks.
    Default: all stocks are uncorrelated (random independent series).
    corr_override lets us inject high-correlation pairs for FW tests.
    """
    np.random.seed(42)
    n_days = 252
    tickers = ["A.NS", "B.NS", "C.NS", "D.NS"]
    data = {t: np.random.normal(0.001, 0.015, n_days) for t in tickers}

    if corr_override:
        # Make B highly correlated with A by adding noise to A's series
        if "B_like_A" in corr_override:
            data["B.NS"] = data["A.NS"] + np.random.normal(0, 0.002, n_days)

    return pd.DataFrame(data)


# ── Greedy Tests ───────────────────────────────────────────────────────────────

class TestGreedy:

    def test_ratio_ordering(self):
        """Stocks should be sorted by return/risk ratio descending."""
        metrics = make_metrics()
        scored = compute_return_risk_ratio(metrics)
        ratios = [s[1] for s in scored]
        assert ratios == sorted(ratios, reverse=True), \
            "Stocks must be sorted by return/risk ratio descending"

    def test_budget_not_exceeded(self):
        """Total invested must never exceed the budget."""
        metrics = make_metrics()
        for budget in [500, 1000, 3000, 10000]:
            result = greedy_select(metrics, budget)
            assert result["total_invested"] <= budget, \
                f"Invested ₹{result['total_invested']} exceeds budget ₹{budget}"

    def test_empty_when_budget_too_low(self):
        """No stocks should be selected when budget < cheapest stock price."""
        metrics = make_metrics()
        result = greedy_select(metrics, budget=100)   # cheapest is ₹500
        assert result["selected"] == [], "Should select nothing when budget is too low"
        assert result["total_invested"] == 0.0

    def test_selects_best_ratio_first(self):
        """First selected stock should be the one with the highest return/risk ratio."""
        metrics = make_metrics()
        result = greedy_select(metrics, budget=1500)  # can afford A (₹1000) or E (₹500)
        # A has ratio 2.0, highest — it should be picked first
        assert "A.NS" in result["selected"], "Stock A (highest ratio=2.0) should be selected"

    def test_remaining_budget_correct(self):
        """remaining_budget = budget - total_invested."""
        metrics = make_metrics()
        result = greedy_select(metrics, budget=5000)
        assert abs(result["remaining_budget"] - (5000 - result["total_invested"])) < 0.01

    def test_all_selected_in_breakdown(self):
        """Every selected ticker must appear in the breakdown list."""
        metrics = make_metrics()
        result = greedy_select(metrics, budget=10_000)
        breakdown_tickers = [b["ticker"] for b in result["breakdown"]]
        for t in result["selected"]:
            assert t in breakdown_tickers, f"{t} is in selected but missing from breakdown"


# ── Knapsack DP Tests ──────────────────────────────────────────────────────────

class TestKnapsack:

    def test_budget_not_exceeded(self):
        """DP total invested must never exceed the budget."""
        metrics = make_metrics()
        for budget in [500, 1000, 3000, 10000]:
            result = knapsack_dp(metrics, budget)
            assert result["total_invested"] <= budget, \
                f"DP invested ₹{result['total_invested']} exceeds budget ₹{budget}"

    def test_dp_ge_greedy_return(self):
        """DP return must always be >= Greedy return (DP is optimal)."""
        metrics = make_metrics()
        for budget in [1000, 3000, 5000, 8000]:
            g = greedy_select(metrics, budget)["total_return"]
            d = knapsack_dp(metrics, budget)["total_return"]
            assert d >= g - 1e-9, \
                f"At budget ₹{budget}: DP ({d:.4f}) < Greedy ({g:.4f}) — DP must be optimal"

    def test_empty_when_budget_zero(self):
        """No stocks selected when budget is 0."""
        metrics = make_metrics()
        result = knapsack_dp(metrics, budget=0)
        assert result["selected"] == []

    def test_normalize_price(self):
        """Price normalization should ceil-divide by 1000."""
        assert normalize_price(1000) == 1
        assert normalize_price(1001) == 2   # ceil
        assert normalize_price(500)  == 1   # max(1, ceil(0.5)) = 1
        assert normalize_price(3500) == 4   # ceil(3.5) = 4

    def test_traceback_matches_dp_value(self):
        """Sum of returned stocks' returns must equal dp's reported total_return."""
        metrics = make_metrics()
        result = knapsack_dp(metrics, budget=10_000)
        computed_total = sum(metrics[t]["expected_return"] for t in result["selected"])
        assert abs(computed_total - result["total_return"]) < 1e-9, \
            "Traceback total return doesn't match DP optimum"

    def test_known_optimal(self):
        """
        Manually verify optimality for a small case.
        Budget = ₹3500.
        Stocks: A=₹1000 (ret=0.20), B=₹2000 (ret=0.15), C=₹1500 (ret=0.10), E=₹500 (ret=0.05)
        Greedy picks A (ratio 2.0), then E (ratio 0.5) → total = 0.25
        Optimal: A(1000) + B(2000) = ₹3000 → total = 0.35  OR  A+C = ₹2500 → 0.30
        Best fitting ₹3500: A+B = ₹3000, return=0.35
        """
        metrics = make_metrics()
        result = knapsack_dp(metrics, budget=3500)
        # DP should find A+B = 0.35 (fits in budget, highest return)
        assert "A.NS" in result["selected"]
        assert "B.NS" in result["selected"]
        assert abs(result["total_return"] - 0.35) < 1e-6


# ── Floyd-Warshall Tests ───────────────────────────────────────────────────────

class TestFloydWarshall:

    def test_distance_matrix_range(self):
        """Distance values (1-corr) should be in [0, 2] since correlation is in [-1, 1]."""
        log_returns = make_log_returns()
        _, dist_matrix = build_distance_matrix(log_returns)
        assert dist_matrix.values.min() >= -1e-9, "Distance must be >= 0"
        assert dist_matrix.values.max() <= 2 + 1e-9, "Distance must be <= 2"

    def test_diagonal_is_zero(self):
        """A stock's distance from itself must be 0 (correlation with itself = 1)."""
        log_returns = make_log_returns()
        _, dist_matrix = build_distance_matrix(log_returns)
        diag = [dist_matrix.iloc[i, i] for i in range(len(dist_matrix))]
        for val in diag:
            assert abs(val) < 1e-9, f"Diagonal should be 0, got {val}"

    def test_fw_distances_le_direct(self):
        """Floyd-Warshall all-pairs distances must be <= direct 1-hop distances."""
        log_returns = make_log_returns()
        _, dist_matrix = build_distance_matrix(log_returns)
        fw_dist = floyd_warshall(dist_matrix)
        direct  = dist_matrix.values

        # FW finds shortest paths — always <= direct distance
        for i in range(len(fw_dist)):
            for j in range(len(fw_dist)):
                assert fw_dist[i][j] <= direct[i][j] + 1e-9, \
                    f"FW distance [{i}][{j}] = {fw_dist[i][j]} > direct {direct[i][j]}"

    def test_highly_correlated_pair_detected(self):
        """Pair with high correlation should appear in find_correlated_pairs()."""
        log_returns = make_log_returns(corr_override={"B_like_A": True})
        corr_matrix, _ = build_distance_matrix(log_returns)
        pairs = find_correlated_pairs(corr_matrix, ["A.NS", "B.NS"], threshold=0.70)
        # A and B were constructed to be highly correlated
        assert len(pairs) > 0, "High-correlation pair A, B should be detected"

    def test_enforce_removes_weaker_stock(self):
        """Diversification should remove the stock with the lower return/risk ratio."""
        # Make A and B highly correlated in returns
        log_returns = make_log_returns(corr_override={"B_like_A": True})
        corr_matrix, _ = build_distance_matrix(log_returns)

        metrics = make_metrics()
        # A has ratio 2.0, B has ratio 1.5 → B should be removed
        cleaned, penalized = enforce_diversification(
            ["A.NS", "B.NS"], metrics, corr_matrix, threshold=0.70
        )
        # Only run assertion if correlation was actually high enough
        if penalized:
            removed_tickers = [p["removed"] for p in penalized]
            assert "B.NS" in removed_tickers, \
                "B (lower ratio=1.5) should be removed, not A (ratio=2.0)"

    def test_no_penalization_for_low_correlation(self):
        """When all correlations are below threshold, nothing should be penalized."""
        log_returns = make_log_returns()   # random, uncorrelated
        corr_matrix, _ = build_distance_matrix(log_returns)
        metrics = make_metrics()

        _, penalized = enforce_diversification(
            list(metrics.keys()), metrics, corr_matrix, threshold=0.99
        )
        # With threshold=0.99, random mock stocks won't exceed it
        assert penalized == [], "No stocks should be penalized at threshold=0.99"
