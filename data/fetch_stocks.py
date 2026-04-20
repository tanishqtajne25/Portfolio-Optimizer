import yfinance as yf
import pandas as pd
import numpy as np
import json
import os

CACHE_DIR = "data"
METRICS_CACHE = os.path.join(CACHE_DIR, "stocks_cache.json")
LOG_RETURNS_CACHE = os.path.join(CACHE_DIR, "log_returns_cache.csv")
DATA_COMPLETENESS_THRESHOLD = 0.95


def fetch_stock_data(symbols: list, period: str = "1y") -> pd.DataFrame:
    symbols_ns = [s + ".NS" for s in symbols]

    raw = yf.download(symbols_ns, period=period, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError("yfinance returned no data. Check symbols or internet.")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": symbols_ns[0]})

    min_rows = int(DATA_COMPLETENESS_THRESHOLD * len(prices))
    before = set(prices.columns)
    prices = prices.dropna(axis=1, thresh=min_rows)
    dropped = before - set(prices.columns)
    if dropped:
        print(f"[fetch_stocks] Dropped (insufficient data): {dropped}")

    prices = prices.ffill().bfill()

    if prices.empty:
        raise ValueError("No stocks survived quality filter. Try a shorter period.")

    print(f"[fetch_stocks] Loaded {prices.shape[1]} stocks × {prices.shape[0]} days.")
    return prices


def compute_metrics(prices: pd.DataFrame) -> tuple:
    log_returns = np.log(prices / prices.shift(1)).dropna()

    metrics = {}
    for col in prices.columns:
        metrics[col] = {
            "expected_return": float(log_returns[col].mean() * 252),
            "risk":            float(log_returns[col].std() * np.sqrt(252)),
            "price":           round(float(prices[col].iloc[-1]), 2),
        }

    return metrics, log_returns


def save_cache(metrics: dict, log_returns: pd.DataFrame) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(METRICS_CACHE, "w") as f:
        json.dump(metrics, f, indent=4)
    log_returns.to_csv(LOG_RETURNS_CACHE)
    print(f"[fetch_stocks] Cache saved → {METRICS_CACHE}, {LOG_RETURNS_CACHE}")


def load_cache() -> tuple:
    if not os.path.exists(METRICS_CACHE) or not os.path.exists(LOG_RETURNS_CACHE):
        return None, None
    with open(METRICS_CACHE, "r") as f:
        metrics = json.load(f)
    log_returns = pd.read_csv(LOG_RETURNS_CACHE, index_col=0, parse_dates=True)
    print("[fetch_stocks] Loaded from cache.")
    return metrics, log_returns


def get_stock_data(symbols: list, use_cache: bool = True) -> tuple:
    if use_cache:
        metrics, log_returns = load_cache()
        if metrics is not None and log_returns is not None:
            return metrics, log_returns

    prices = fetch_stock_data(symbols)
    metrics, log_returns = compute_metrics(prices)
    save_cache(metrics, log_returns)
    return metrics, log_returns


if __name__ == "__main__":
    NIFTY_SYMBOLS = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "HINDUNILVR", "ITC", "LT", "SBIN", "BHARTIARTL",
        "WIPRO", "HCLTECH", "ASIANPAINT", "MARUTI", "BAJFINANCE",
        "KOTAKBANK", "AXISBANK", "NESTLEIND", "TITAN", "SUNPHARMA"
    ]

    metrics, log_returns = get_stock_data(NIFTY_SYMBOLS, use_cache=False)

    print(f"\n{'Ticker':<20} {'Exp. Return':>12} {'Risk':>10} {'Price (₹)':>12}")
    print("-" * 56)
    for ticker, m in metrics.items():
        print(f"{ticker:<20} {m['expected_return']:>11.2%} {m['risk']:>9.2%} {m['price']:>12.2f}")

    print(f"\nLog returns shape: {log_returns.shape}")