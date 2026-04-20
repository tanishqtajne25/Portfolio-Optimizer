"""
tests/test_data.py
==================
Unit tests for the data fetching and metrics computation pipeline.
All tests use mock/synthetic data — no real yfinance API calls.

Run with:
    pytest tests/test_data.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
import json
import tempfile

from data.fetch_stocks import compute_metrics, save_cache, load_cache


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_mock_prices(n_stocks: int = 5, n_days: int = 252) -> pd.DataFrame:
    """
    Generate synthetic price data using geometric Brownian motion.
    Each stock starts at ₹1000 and drifts randomly.
    """
    np.random.seed(7)
    tickers = [f"MOCK{i}.NS" for i in range(n_stocks)]
    data = {}
    for t in tickers:
        daily_returns = np.random.normal(0.001, 0.015, n_days)
        price_series  = 1000 * np.exp(np.cumsum(daily_returns))  # GBM path
        data[t] = price_series

    dates = pd.date_range(end="2024-12-31", periods=n_days, freq="B")
    return pd.DataFrame(data, index=dates)


# ── compute_metrics Tests ──────────────────────────────────────────────────────

class TestComputeMetrics:

    def test_returns_correct_keys(self):
        """Each stock entry must have expected_return, risk, and price."""
        prices = make_mock_prices()
        metrics, _ = compute_metrics(prices)

        for ticker, m in metrics.items():
            assert "expected_return" in m, f"{ticker} missing expected_return"
            assert "risk"            in m, f"{ticker} missing risk"
            assert "price"           in m, f"{ticker} missing price"

    def test_all_tickers_present(self):
        """Metrics must contain all columns from the price DataFrame."""
        prices = make_mock_prices(n_stocks=5)
        metrics, _ = compute_metrics(prices)
        assert set(metrics.keys()) == set(prices.columns)

    def test_risk_is_positive(self):
        """Annualized volatility must always be > 0 for non-flat price series."""
        prices = make_mock_prices()
        metrics, _ = compute_metrics(prices)
        for ticker, m in metrics.items():
            assert m["risk"] > 0, f"{ticker} has non-positive risk: {m['risk']}"

    def test_price_is_last_closing_price(self):
        """Price in metrics must equal the last row of the prices DataFrame."""
        prices = make_mock_prices()
        metrics, _ = compute_metrics(prices)
        for ticker in prices.columns:
            expected_price = round(float(prices[ticker].iloc[-1]), 2)
            assert metrics[ticker]["price"] == expected_price, \
                f"{ticker}: metrics price {metrics[ticker]['price']} != last close {expected_price}"

    def test_log_returns_shape(self):
        """
        Log returns should have (n_days - 1) rows and same columns as prices.
        The first row is dropped because log(P1/P0) needs P0.
        """
        n_days = 252
        prices = make_mock_prices(n_days=n_days)
        _, log_returns = compute_metrics(prices)
        assert log_returns.shape == (n_days - 1, prices.shape[1]), \
            f"Expected shape ({n_days-1}, {prices.shape[1]}), got {log_returns.shape}"

    def test_log_returns_no_nan(self):
        """Log returns must not contain NaN values after dropna()."""
        prices = make_mock_prices()
        _, log_returns = compute_metrics(prices)
        assert not log_returns.isnull().any().any(), "Log returns contain NaN values"

    def test_annualization_factor(self):
        """
        Manually verify annualization:
        Expected return = daily_mean * 252
        Risk = daily_std * sqrt(252)
        """
        prices = make_mock_prices(n_stocks=1)
        metrics, log_returns = compute_metrics(prices)

        ticker = list(prices.columns)[0]
        daily_mean = log_returns[ticker].mean()
        daily_std  = log_returns[ticker].std()

        expected_ann_return = daily_mean * 252
        expected_ann_risk   = daily_std  * (252 ** 0.5)

        assert abs(metrics[ticker]["expected_return"] - expected_ann_return) < 1e-9
        assert abs(metrics[ticker]["risk"]            - expected_ann_risk)   < 1e-9


# ── Cache Tests ────────────────────────────────────────────────────────────────

class TestCache:

    def test_save_and_load_roundtrip(self):
        """
        save_cache() followed by load_cache() must return identical data.
        Uses a temporary directory so we don't pollute the real data/ folder.
        """
        prices   = make_mock_prices()
        metrics, log_returns = compute_metrics(prices)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily override cache paths
            import data.fetch_stocks as fs
            original_metrics_path    = fs.METRICS_CACHE
            original_log_path        = fs.LOG_RETURNS_CACHE

            fs.METRICS_CACHE    = os.path.join(tmpdir, "stocks_cache.json")
            fs.LOG_RETURNS_CACHE = os.path.join(tmpdir, "log_returns_cache.csv")

            save_cache(metrics, log_returns)
            loaded_metrics, loaded_log_returns = load_cache()

            # Restore original paths
            fs.METRICS_CACHE    = original_metrics_path
            fs.LOG_RETURNS_CACHE = original_log_path

        assert loaded_metrics is not None, "load_cache returned None for metrics"
        assert loaded_log_returns is not None, "load_cache returned None for log_returns"

        # Verify metrics round-trips correctly
        for ticker in metrics:
            assert abs(metrics[ticker]["expected_return"] - loaded_metrics[ticker]["expected_return"]) < 1e-9
            assert abs(metrics[ticker]["risk"]            - loaded_metrics[ticker]["risk"])            < 1e-9
            assert abs(metrics[ticker]["price"]           - loaded_metrics[ticker]["price"])           < 0.01

    def test_load_cache_returns_none_when_missing(self):
        """
        load_cache() must return (None, None) when cache files don't exist.
        """
        import data.fetch_stocks as fs
        original_metrics_path    = fs.METRICS_CACHE
        original_log_path        = fs.LOG_RETURNS_CACHE

        fs.METRICS_CACHE    = "/tmp/nonexistent_metrics_12345.json"
        fs.LOG_RETURNS_CACHE = "/tmp/nonexistent_log_12345.csv"

        result = load_cache()

        fs.METRICS_CACHE    = original_metrics_path
        fs.LOG_RETURNS_CACHE = original_log_path

        assert result == (None, None), "load_cache must return (None, None) for missing files"
