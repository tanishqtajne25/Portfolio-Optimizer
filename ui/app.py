import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data.fetch_stocks import get_stock_data
from algorithms.greedy import greedy_select
from algorithms.knapsack_dp import knapsack_dp
from algorithms.floyd_warshall import diversify_portfolio
from algorithms.compare import compare_across_budgets
from ui.charts import (
    plot_return_risk,
    plot_comparison,
    plot_correlation_network,
    plot_allocation_pie,
)
from ui.warnings import (
    show_correlation_warnings,
    show_correlated_pairs_table,
    show_diversification_summary,
)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NiftyOpt — Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
)

# ── Stock Universe ─────────────────────────────────────────────────────────────
NIFTY_SYMBOLS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "HINDUNILVR", "ITC", "LT", "SBIN", "BHARTIARTL",
    "WIPRO", "HCLTECH", "ASIANPAINT", "MARUTI", "BAJFINANCE",
    "KOTAKBANK", "AXISBANK", "NESTLEIND", "TITAN", "SUNPHARMA"
]


# ── Data Loading (cached so it doesn't re-fetch on every widget interaction) ──
@st.cache_data(show_spinner="Fetching stock data from Yahoo Finance…")
def load_data(use_cache: bool = True):
    return get_stock_data(NIFTY_SYMBOLS, use_cache=use_cache)


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ NiftyOpt Settings")

budget = st.sidebar.slider(
    label="Budget (₹)",
    min_value=10_000,
    max_value=3_00_000,
    value=50_000,
    step=5_000,
    format="₹%d",
)

algorithm = st.sidebar.radio(
    "Optimization Algorithm",
    options=["Greedy", "DP (0/1 Knapsack)", "Both (Compare)"],
    index=1,
)

apply_diversification = st.sidebar.toggle(
    "Apply Diversification (Floyd-Warshall)",
    value=True,
)

corr_threshold = st.sidebar.slider(
    "Correlation Threshold for Penalization",
    min_value=0.50,
    max_value=0.99,
    value=0.85,
    step=0.05,
    help="Stocks with pairwise correlation above this value will be penalized.",
)

refresh_data = st.sidebar.button("🔄 Refresh Data from Yahoo Finance")

st.sidebar.markdown("---")
st.sidebar.caption(
    "**NiftyOpt** is a simulation on historical NIFTY 50 data.\n"
    "Not financial advice. Not a live trading system."
)


# ── Main Content ───────────────────────────────────────────────────────────────
st.title("📈 NiftyOpt — NIFTY 50 Portfolio Optimizer")
st.caption(
    "Finds the optimal combination of stocks that maximizes expected return "
    "within your budget. Uses Greedy, 0/1 Knapsack DP, and Floyd-Warshall "
    "diversification — all on real historical data."
)

# Load data
metrics, log_returns = load_data(use_cache=not refresh_data)

# Display stock universe stats
with st.expander("📋 Stock Universe Overview"):
    rows = []
    for ticker, m in metrics.items():
        rows.append({
            "Ticker"           : ticker.replace(".NS", ""),
            "Price (₹)"        : f"₹{m['price']:,.2f}",
            "Expected Return"  : f"{m['expected_return']:.2%}",
            "Risk (Volatility)": f"{m['risk']:.2%}",
            "Return/Risk"      : f"{m['expected_return']/max(m['risk'], 1e-9):.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ── Run Algorithms ─────────────────────────────────────────────────────────────

def run_and_display(algo_name: str, result: dict, metrics: dict, log_returns, corr_threshold: float):
    """Shared display logic for both Greedy and DP results."""
    st.subheader(f"{'🟢' if algo_name == 'Greedy' else '🟣'} {algo_name} Portfolio")

    col1, col2, col3 = st.columns(3)
    col1.metric("Stocks Selected",        len(result["selected"]))
    col2.metric("Total Invested",         f"₹{result['total_invested']:,.0f}")
    col3.metric("Expected Return (sum)",  f"{result['total_return']:.2%}")

    if not result["selected"]:
        st.warning("No stocks selected — budget may be too low for any stock.")
        return result

    # Diversification
    if apply_diversification and result["selected"]:
        with st.spinner("Running Floyd-Warshall diversification…"):
            div_result = diversify_portfolio(
                result["selected"], metrics, log_returns, threshold=corr_threshold
            )

        show_diversification_summary(div_result["original"], div_result["diversified"])
        show_correlation_warnings(div_result["penalized"])
        show_correlated_pairs_table(
            div_result["corr_matrix"], result["selected"], threshold=0.70
        )

        # Update result to diversified portfolio for charts
        display_tickers = div_result["diversified"]
        corr_matrix     = div_result["corr_matrix"]
    else:
        display_tickers = result["selected"]
        _, dist_m       = None, None
        corr_matrix     = log_returns.corr()

    # Breakdown table
    with st.expander("📊 Stock Breakdown"):
        rows = []
        for t in display_tickers:
            m = metrics.get(t, {})
            rows.append({
                "Ticker"          : t.replace(".NS", ""),
                "Price (₹)"       : f"₹{m.get('price', 0):,.2f}",
                "Expected Return" : f"{m.get('expected_return', 0):.2%}",
                "Risk"            : f"{m.get('risk', 0):.2%}",
                "Return/Risk"     : f"{m.get('expected_return', 0)/max(m.get('risk', 1e-9), 1e-9):.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Charts
    tab1, tab2, tab3 = st.tabs(["📊 Return vs Risk", "🥧 Allocation", "🕸️ Network Graph"])

    with tab1:
        fig = plot_return_risk(metrics, display_tickers)
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        fig = plot_allocation_pie(metrics, display_tickers)
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        fig = plot_correlation_network(corr_matrix, display_tickers, threshold=0.5)
        st.pyplot(fig)
        plt.close(fig)

    return result


# ── Algorithm Routing ──────────────────────────────────────────────────────────

if algorithm in ("Greedy", "Both (Compare)"):
    greedy_result = greedy_select(metrics, budget)
    run_and_display("Greedy", greedy_result, metrics, log_returns, corr_threshold)

if algorithm in ("DP (0/1 Knapsack)", "Both (Compare)"):
    dp_result = knapsack_dp(metrics, budget)
    run_and_display("DP (0/1 Knapsack)", dp_result, metrics, log_returns, corr_threshold)

# ── Comparison Dashboard ───────────────────────────────────────────────────────

if algorithm == "Both (Compare)":
    st.markdown("---")
    st.subheader("📉 Greedy vs DP — Multi-Budget Comparison")
    st.caption(
        "Greedy performs close to optimal at large budgets. "
        "DP outperforms Greedy more noticeably at tighter budget constraints."
    )

    comparison_budgets = [20_000, 50_000, 1_00_000, 2_00_000]

    with st.spinner("Running comparison across budget levels…"):
        comparison_results = compare_across_budgets(metrics, comparison_budgets)

    # Summary table
    table_rows = []
    for r in comparison_results:
        table_rows.append({
            "Budget"         : f"₹{r['budget']:,.0f}",
            "Greedy Return"  : f"{r['greedy_return']:.2%}",
            "DP Return"      : f"{r['dp_return']:.2%}",
            "Improvement"    : f"{r['improvement_pct']:+.2f}%",
            "Greedy Stocks"  : len(r["greedy_selected"]),
            "DP Stocks"      : len(r["dp_selected"]),
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    fig = plot_comparison(comparison_results)
    st.pyplot(fig)
    plt.close(fig)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "**NiftyOpt** | Historical simulation only | "
    "Data: Yahoo Finance via yfinance | "
    "Algorithms: Greedy · 0/1 Knapsack DP · Floyd-Warshall | "
    "Not financial advice."
)
