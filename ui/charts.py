import matplotlib
matplotlib.use("Agg")   # non-interactive backend — required for Streamlit

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
import pandas as pd

# ── Chart Helpers for NiftyOpt UI ─────────────────────────────────────────────
#
# All functions return a matplotlib Figure object.
# Streamlit renders it with st.pyplot(fig).
# Always call plt.close(fig) after st.pyplot() to free memory.


def _strip_ns(ticker: str) -> str:
    """Remove .NS suffix for display labels."""
    return ticker.replace(".NS", "")


# ── 1. Return / Risk Bar Chart ─────────────────────────────────────────────────

def plot_return_risk(metrics: dict, selected_tickers: list) -> plt.Figure:
    """
    Horizontal bar chart: expected return and risk for each selected stock.
    Two bars per stock — return (green) and risk (red).

    Args:
        metrics          (dict): Full metrics dict
        selected_tickers (list): Stocks to display

    Returns:
        matplotlib Figure
    """
    if not selected_tickers:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No stocks selected.", ha="center", va="center")
        return fig

    labels  = [_strip_ns(t) for t in selected_tickers]
    returns = [metrics[t]["expected_return"] * 100 for t in selected_tickers]
    risks   = [metrics[t]["risk"]            * 100 for t in selected_tickers]

    x      = np.arange(len(labels))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    bars_r = ax.bar(x - width / 2, returns, width, label="Expected Return (%)", color="#2ecc71", alpha=0.85)
    bars_v = ax.bar(x + width / 2, risks,   width, label="Risk / Volatility (%)",  color="#e74c3c", alpha=0.85)

    ax.set_xlabel("Stock", fontsize=11)
    ax.set_ylabel("Annualized % (log return basis)", fontsize=11)
    ax.set_title("Expected Return vs Risk — Selected Portfolio", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ── 2. Greedy vs DP Comparison Bar Chart ──────────────────────────────────────

def plot_comparison(comparison_results: list[dict]) -> plt.Figure:
    """
    Grouped bar chart: Greedy return vs DP return at each budget level.

    Args:
        comparison_results (list[dict]): Output of compare_across_budgets()

    Returns:
        matplotlib Figure
    """
    budgets      = [f"₹{r['budget'] / 1000:.0f}k" for r in comparison_results]
    greedy_rets  = [r["greedy_return"] * 100 for r in comparison_results]
    dp_rets      = [r["dp_return"]     * 100 for r in comparison_results]
    improvements = [r["improvement_pct"]     for r in comparison_results]

    x     = np.arange(len(budgets))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: grouped bars — return at each budget
    ax1.bar(x - width / 2, greedy_rets, width, label="Greedy", color="#3498db", alpha=0.85)
    ax1.bar(x + width / 2, dp_rets,     width, label="DP (0/1 Knapsack)", color="#9b59b6", alpha=0.85)
    ax1.set_title("Total Expected Return by Budget", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(budgets)
    ax1.set_ylabel("Sum of Annualized Returns (%)")
    ax1.set_xlabel("Budget")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Right: improvement line — how much better DP is
    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in improvements]
    ax2.bar(budgets, improvements, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("DP Improvement over Greedy (%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Improvement (%)")
    ax2.set_xlabel("Budget")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Greedy vs 0/1 Knapsack DP — Portfolio Optimizer Comparison",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── 3. Correlation Network Graph (NetworkX) ────────────────────────────────────

def plot_correlation_network(
    corr_matrix: pd.DataFrame,
    selected_tickers: list,
    threshold: float = 0.5
) -> plt.Figure:
    """
    Draw a correlation network graph using NetworkX.
    - Nodes = stocks
    - Edges = drawn only if |correlation| > threshold
    - Edge color: green = positive, red = negative correlation
    - Edge thickness = correlation strength
    - Selected portfolio stocks are highlighted in gold

    Args:
        corr_matrix      (pd.DataFrame): Full pairwise correlation matrix
        selected_tickers (list)        : Stocks in the current portfolio (highlighted)
        threshold        (float)       : Only show edges above this correlation

    Returns:
        matplotlib Figure
    """
    G = nx.Graph()

    tickers = list(corr_matrix.columns)
    G.add_nodes_from(tickers)

    edge_colors = []
    edge_widths = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                G.add_edge(tickers[i], tickers[j], weight=abs(corr_val))
                edge_colors.append("#27ae60" if corr_val > 0 else "#e74c3c")
                edge_widths.append(abs(corr_val) * 3)   # thickness ∝ correlation

    fig, ax = plt.subplots(figsize=(12, 8))

    pos = nx.spring_layout(G, seed=42, k=1.5)

    # Node colors: gold for portfolio stocks, light blue for others
    node_colors = [
        "#f39c12" if t in selected_tickers else "#aed6f1"
        for t in G.nodes()
    ]
    node_sizes = [
        600 if t in selected_tickers else 350
        for t in G.nodes()
    ]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_labels(
        G, pos,
        labels={t: _strip_ns(t) for t in G.nodes()},
        font_size=7, font_weight="bold", ax=ax
    )

    if G.edges():
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.6,
            ax=ax
        )

    # Legend
    legend_handles = [
        mpatches.Patch(color="#f39c12",  label="In portfolio"),
        mpatches.Patch(color="#aed6f1",  label="Not selected"),
        mpatches.Patch(color="#27ae60",  label="Positive correlation"),
        mpatches.Patch(color="#e74c3c",  label="Negative correlation"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8)
    ax.set_title(
        f"Stock Correlation Network  (edges shown for |corr| > {threshold})",
        fontsize=12, fontweight="bold"
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


# ── 4. Portfolio Allocation Pie Chart ─────────────────────────────────────────

def plot_allocation_pie(metrics: dict, selected_tickers: list) -> plt.Figure:
    """
    Pie chart showing capital allocation (by stock price) in the portfolio.

    Args:
        metrics          (dict): Full metrics dict
        selected_tickers (list): Stocks in the portfolio

    Returns:
        matplotlib Figure
    """
    if not selected_tickers:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No stocks selected.", ha="center", va="center")
        return fig

    labels = [_strip_ns(t) for t in selected_tickers]
    sizes  = [metrics[t]["price"] for t in selected_tickers]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        startangle=140, pctdistance=0.82,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2}
    )
    ax.set_title("Capital Allocation by Stock Price", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
