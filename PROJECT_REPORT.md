# NiftyOpt Project Report

Date: April 20, 2026
Repository: DAA-Project

## 1) Project Overview

NiftyOpt is a portfolio optimization simulation built on historical NIFTY stock data. The project demonstrates how multiple Design and Analysis of Algorithms (DAA) techniques can solve different investment selection problems under constraints.

Core idea:
- Input: a budget and a stock universe with historical statistics
- Output: selected stocks and portfolio metrics
- Methods:
  - Greedy algorithm for fast approximation
  - 0/1 Knapsack Dynamic Programming (DP) for exact combinational optimization under a modeled budget capacity
  - Floyd-Warshall for diversification analysis using all-pairs relationships

Important scope note:
- This is a historical simulation and educational optimizer.
- It is not a live trading system and not financial advice.

## 2) Project Structure and Responsibilities

- main.py
  - CLI entry point
  - Runs data loading, Greedy, DP, diversification step, and comparison

- data/fetch_stocks.py
  - Fetches data from Yahoo Finance (yfinance)
  - Computes expected return, risk (volatility), and latest price
  - Handles local caching

- algorithms/greedy.py
  - Computes return/risk score and selects stocks in descending score order
  - Fast baseline method

- algorithms/knapsack_dp.py
  - Solves stock selection as a 0/1 knapsack problem
  - Uses DP table + traceback to recover selected stocks

- algorithms/floyd_warshall.py
  - Converts correlation to distance and computes all-pairs shortest paths
  - Identifies highly correlated pairs and enforces diversification penalties

- algorithms/compare.py
  - Compares Greedy and DP across multiple budgets

- ui/app.py, ui/charts.py, ui/warnings.py
  - Streamlit UI, plotting, and warnings/summary displays

- tests/test_algorithms.py
  - Unit tests for Greedy, DP, and Floyd-Warshall components

- tests/test_data.py
  - Unit tests for metric calculation and cache pipeline

## 3) End-to-End Workflow

1. User selects budget and algorithm mode in UI (or uses CLI defaults).
2. Data loader pulls cached metrics if available; otherwise fetches from Yahoo Finance.
3. Metrics are prepared per stock:
   - expected_return (annualized from daily log returns)
   - risk (annualized volatility)
   - latest price
4. Optimizer runs:
   - Greedy and/or DP to select stocks
5. Optional diversification step:
   - Correlation matrix is converted to distance matrix
   - Floyd-Warshall helps analyze all-pairs relationships
   - Highly correlated pairs can trigger penalty/removal decisions
6. Results are displayed:
   - selected stocks
   - invested amount
   - expected return summary
   - comparison and charts

## 4) Financial Metrics Explained

### 4.1 Expected Return

Expected return here is annualized mean log return:

expected_return = mean(daily_log_returns) * 252

Interpretation:
- Positive value: historical trend over the selected lookback period was upward on average.
- Negative value: historical trend over that period was downward on average.

### 4.2 Risk / Volatility

Risk is annualized standard deviation of daily log returns:

risk = std(daily_log_returns) * sqrt(252)

Interpretation:
- Higher risk means more variability/uncertainty in returns.
- Lower risk means more stable return behavior.

### 4.3 Return-to-Risk Ratio

Used by Greedy ranking:

ratio = expected_return / risk

Interpretation:
- Larger ratio means more return per unit of volatility.
- If expected return is negative, ratio is negative and usually unattractive.

## 5) Why Negative Expected Returns Matter

Negative expected return stocks are a key reason for portfolio behavior that can look surprising.

If many stocks have negative expected returns in the cached dataset:
- Optimizers that maximize return will avoid them whenever possible.
- Increasing budget does not necessarily force more stock purchases.
- Invested amount can plateau even at higher budget values.

This is not automatically a bug. It is consistent with objective-driven optimization under current data.

## 6) Algorithm Details

### 6.1 Greedy (algorithms/greedy.py)

Objective:
- Quickly build a portfolio by sorting stocks on return/risk ratio and selecting while budget allows.

Complexity:
- Time: O(n log n)
- Space: O(n)

Pros:
- Fast and easy to explain.

Limitations:
- Not globally optimal for 0/1 selection problems.
- Can miss better combinations that DP can find.

### 6.2 0/1 Knapsack DP (algorithms/knapsack_dp.py)

Objective:
- Find the best combination under capacity constraints.

Model mapping:
- item -> stock
- weight -> modeled price unit
- value -> expected return score
- capacity -> modeled budget units

Implementation points:
- Uses DP table for exact optimization under the chosen discretization.
- Uses traceback to recover selected stocks.
- Includes stable tie handling to keep reconstruction deterministic.

Complexity:
- Time: O(n * W)
- Space: O(n * W)
- W depends on budget capacity units used by the model.

### 6.3 Floyd-Warshall Diversification (algorithms/floyd_warshall.py)

Purpose:
- Analyze pairwise and transitive relationship structure among selected stocks.

Critical transformation:
- Correlation is similarity, not distance.
- Convert to distance:

distance = 1 - correlation

Range logic:
- correlation 1.0 -> distance 0
- correlation 0.0 -> distance 1
- correlation -1.0 -> distance 2

Complexity:
- Time: O(n^3)

Use in this project:
- Detect highly correlated pairs
- Apply penalization/removal logic to improve diversification quality

## 7) Root Cause of "Same Invested Amount" Across Higher Budgets

Observed behavior:
- Total invested can remain unchanged even when budget slider increases.

Primary reasons:
1. Objective function prioritizes return score, not full capital utilization.
2. Current cached dataset contains multiple negative expected return stocks.
3. After selecting attractive stocks, adding more can reduce objective quality.

Therefore:
- A flat invested amount at higher budgets can be mathematically valid under current objective and data.

## 8) Solved Test Cases (Current Status)

Latest run:
- Command: python -m pytest tests -q
- Result: 27 passed in 0.83s

### 8.1 tests/test_algorithms.py

Greedy tests solved:
- test_ratio_ordering
- test_budget_not_exceeded
- test_empty_when_budget_too_low
- test_selects_best_ratio_first
- test_remaining_budget_correct
- test_all_selected_in_breakdown

Knapsack DP tests solved:
- test_budget_not_exceeded
- test_dp_ge_greedy_return
- test_empty_when_budget_zero
- test_normalize_price
- test_traceback_matches_dp_value
- test_known_optimal

Floyd-Warshall tests solved:
- test_distance_matrix_range
- test_diagonal_is_zero
- test_fw_distances_le_direct
- test_highly_correlated_pair_detected
- test_enforce_removes_weaker_stock
- test_no_penalization_for_low_correlation

### 8.2 tests/test_data.py

Compute metrics tests solved:
- test_returns_correct_keys
- test_all_tickers_present
- test_risk_is_positive
- test_price_is_last_closing_price
- test_log_returns_shape
- test_log_returns_no_nan
- test_annualization_factor

Cache tests solved:
- test_save_and_load_roundtrip
- test_load_cache_returns_none_when_missing

## 9) Reliability and Validation Summary

Validated successfully:
- Algorithm unit correctness (Greedy, DP, Floyd-Warshall)
- Data metric computations and annualization behavior
- Cache save/load behavior
- CLI end-to-end execution path

Known practical constraints:
- Historical data quality and period sensitivity
- No transaction costs, slippage, tax, liquidity, or execution latency
- Single-period static allocation, not dynamic rebalancing

## 10) How to Run

From repository root:

1. Install dependencies
   - pip install -r requirements.txt

2. Run UI
   - python -m streamlit run ui/app.py

3. Run CLI
   - python main.py

4. Run tests
   - python -m pytest tests -v

## 11) Suggested Future Improvements

1. Optimize expected rupee profit instead of pure return sum to improve budget utilization behavior.
2. Add explicit minimum deployment constraints (for example at least 80 percent budget usage).
3. Add objective variants in UI so users can choose return-maximization vs utilization-aware optimization.
4. Add transaction cost and slippage modeling.
5. Add backtesting over rolling windows for robustness.
6. Add integration tests for UI-level behavior and budget sensitivity checks.

## 12) Final Assessment

Yes, the project is implemented successfully as a DAA-based portfolio optimizer simulation.

- Greedy works as fast approximation.
- DP works as exact combinational optimizer under the chosen model.
- Floyd-Warshall works for diversification analysis on transformed correlation distances.
- The full automated test suite currently passes.

The main non-intuitive behavior (flat invested amount for larger budgets) is explained by the current objective and current data characteristics, especially negative expected returns.
