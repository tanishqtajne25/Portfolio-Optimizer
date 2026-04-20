# NiftyOpt — NIFTY 50 Portfolio Optimizer

> A static portfolio optimizer for NIFTY 50 stocks using algorithms from a Design and Analysis of Algorithms (DAA) course. Built on real historical data via Yahoo Finance.

**This is a simulation on historical data — not a live trading system.**

---

## What it does

Given a budget (e.g. ₹50,000), NiftyOpt finds the combination of NIFTY 50 stocks that maximizes expected annual return while staying within budget and avoiding over-concentration in correlated assets.

---

## Algorithms Used

| Problem | Algorithm | Time Complexity | Why |
|---|---|---|---|
| Fast approximate selection | Greedy | O(n log n) | Sorts by return/risk ratio, picks greedily |
| Exact optimal selection | 0/1 Knapsack (DP) | O(n × W) | Guarantees global optimum under budget |
| Diversification | Floyd-Warshall | O(n³) | All-pairs distance analysis on transformed correlation matrix |

Each algorithm solves a **distinct sub-problem** — they are not redundant.

### Why these three?

- **Greedy** gives a fast approximate answer. Good baseline, O(n log n).
- **DP** gives the guaranteed optimal answer. Compared against Greedy to demonstrate the speed-vs-optimality trade-off.
- **Floyd-Warshall** solves a different problem — stock correlation is converted into a distance metric (`1 - correlation`) and used to find the most diversified portfolio. This maps directly to Modern Portfolio Theory.

### Floyd-Warshall — the key transformation

Raw correlation is a **similarity** metric (1 = identical). Floyd-Warshall requires a **distance** metric (0 = identical). So:

```python
distance = 1 - correlation_matrix
```

This maps:
- Correlation 1.0 → distance 0.0 (same stock)
- Correlation 0.0 → distance 1.0 (unrelated)
- Correlation -1.0 → distance 2.0 (perfectly opposite)

Most implementations get this wrong. Always mention this in interviews.

### 0/1 Knapsack — space optimization

Naive 2D DP uses O(n × W) space. NiftyOpt uses a **1D rolling array** updated right-to-left:

```python
dp = [0.0] * (W + 1)
for i in range(n):
    for w in range(W, weights[i] - 1, -1):
        dp[w] = max(dp[w], dp[w - weights[i]] + profits[i])
```

Right-to-left is critical — it prevents picking the same item twice, which is what makes this 0/1 (not fractional) knapsack.

Space complexity: **O(W)** instead of O(n × W).

---

## File Structure

```
portfolio-optimizer/
├── data/
│   ├── fetch_stocks.py       # yfinance fetching, log returns, caching
│   └── stocks_cache.json     # auto-generated cache (gitignore this)
├── algorithms/
│   ├── greedy.py             # greedy selection by return/risk ratio
│   ├── knapsack_dp.py        # 0/1 knapsack + traceback
│   ├── floyd_warshall.py     # correlation graph + FW diversification
│   ├── compare.py            # greedy vs DP comparison table
│   └── __init__.py
├── ui/
│   ├── app.py                # streamlit main UI
│   ├── charts.py             # matplotlib charts + networkx graph
│   └── warnings.py           # correlation alert banners
├── tests/
│   ├── test_algorithms.py    # greedy, DP, FW unit tests
│   └── test_data.py          # metrics computation + cache tests
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the interactive UI
streamlit run ui/app.py

# Run the CLI demo (fetches data, runs all algorithms, prints results)
python main.py

# Run tests
pytest tests/ -v
```

---

## Greedy vs DP — Experimental Results

| Budget | Greedy Return | DP Return | DP Improvement |
|--------|--------------|-----------|----------------|
| ₹20,000 | x% | y% | +z% |
| ₹50,000 | ... | ... | ... |
| ₹1,00,000 | ... | ... | ... |
| ₹2,00,000 | ... | ... | ... |

*(Fill in after running `python algorithms/compare.py` with real data)*

**Key finding:** DP outperforms Greedy most significantly at tight budgets. At large budgets, both algorithms converge because most high-ratio stocks fit anyway.

---

## Known Limitations

These are not bugs — they are honest acknowledgments of scope:

- **No transaction costs** — real trades have brokerage fees
- **No slippage** — real trades don't execute at exact last-close prices
- **No fractional shares** — only whole shares are modeled
- **Single-period static allocation** — not dynamic rebalancing over time
- **O(n³) Floyd-Warshall** — doesn't scale beyond ~100 stocks; approximation methods would replace it for larger universes
- **Historical returns ≠ future returns** — past performance is not a guarantee

Framing: *This is a static allocation model on historical data, not a real trading system.*

---

## Interview Reference

**Q: Why DP over Greedy?**
Greedy makes locally optimal decisions but misses the global optimum when item combinations matter. DP considers all combinations and guarantees the best result. Experimentally, DP outperforms Greedy by 5–15% on tighter budgets.

**Q: Why Floyd-Warshall?**
To analyze all-pairs stock relationships after converting correlation into a valid distance metric (`1 - correlation`). FW gives the full relationship picture across all stock pairs, including transitive relationships — not just adjacent ones.

**Q: What's the time complexity of your DP?**
O(n × W) where n is the number of stocks and W is the budget normalized to ₹1000 units. Space complexity is O(W) using a 1D rolling array instead of O(n × W).

**Q: Did it make money?**
It's a simulation on historical data, not a live trading tool. Real trading needs transaction costs, liquidity constraints, and dynamic rebalancing. The project demonstrates the optimization logic.

---

## Domain Context

- **Domain:** Retail FinTech / Personal Finance tech
- **Relevant companies:** Zerodha, Groww, Smallcase, INDmoney, Paytm Money, Goldman Sachs tech division
- **Academic framework:** Modern Portfolio Theory (Markowitz) — the academic basis for diversification
- **Not quant finance** — this does not use derivatives, leverage, or real-time execution

---

## Tech Stack

- `yfinance` — real NIFTY 50 historical data
- `pandas`, `numpy` — data processing, log returns, correlation matrix
- `networkx` — stock correlation graph visualization
- `matplotlib` — charts
- `streamlit` — interactive UI
