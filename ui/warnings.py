import streamlit as st
import pandas as pd

# ── Correlation Warnings UI ────────────────────────────────────────────────────
#
# These functions render Streamlit warning/info banners when the optimizer
# detects stocks in the portfolio that are highly correlated.
# This is the "user-facing" output of the Floyd-Warshall diversification step.


def show_correlation_warnings(penalized: list[dict]) -> None:
    """
    Display warning banners for stocks that were removed during diversification.

    Each banner explains:
      - Which stock was removed
      - Which stock it was correlated with (and was kept)
      - The correlation value
      - Why the kept stock was preferred

    Args:
        penalized (list[dict]): Output of enforce_diversification() — list of
                                {removed, kept, correlation, reason} dicts
    """
    if not penalized:
        st.success("✅ Portfolio is well-diversified. No highly correlated pairs found.")
        return

    st.warning(
        f"⚠️ **Diversification Alert** — {len(penalized)} stock(s) were penalized "
        f"due to high correlation with others in the portfolio."
    )

    for p in penalized:
        st.error(
            f"🔴 **{p['removed'].replace('.NS', '')}** removed  "
            f"(correlation = {p['correlation']:.0%} with **{p['kept'].replace('.NS', '')}**)  \n"
            f"_{p['reason']}_"
        )


def show_correlated_pairs_table(
    corr_matrix: pd.DataFrame,
    selected_tickers: list,
    threshold: float = 0.7
) -> None:
    """
    Display an expandable table of all correlated pairs in the portfolio,
    above a lower threshold — useful for user awareness even if not penalized.

    Args:
        corr_matrix      (pd.DataFrame): Full pairwise correlation matrix
        selected_tickers (list)        : Portfolio stocks
        threshold        (float)       : Show pairs above this correlation (default 0.7)
    """
    available = [t for t in selected_tickers if t in corr_matrix.columns]
    if len(available) < 2:
        return

    sub = corr_matrix.loc[available, available]
    pairs = []
    tickers = list(sub.columns)

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            corr_val = sub.iloc[i, j]
            if corr_val > threshold:
                pairs.append({
                    "Stock A"    : tickers[i].replace(".NS", ""),
                    "Stock B"    : tickers[j].replace(".NS", ""),
                    "Correlation": f"{corr_val:.2%}",
                    "Risk Level" : "🔴 High" if corr_val > 0.85 else "🟡 Moderate",
                })

    if not pairs:
        return

    with st.expander(f"📊 Correlated Pairs in Portfolio (correlation > {threshold:.0%})"):
        st.dataframe(pd.DataFrame(pairs), use_container_width=True, hide_index=True)
        st.caption(
            "High correlation means these stocks tend to move together. "
            "Holding both provides less diversification benefit than it appears."
        )


def show_diversification_summary(original: list, diversified: list) -> None:
    """
    Show a simple before/after summary of the diversification step.

    Args:
        original    (list): Portfolio before diversification
        diversified (list): Portfolio after removing correlated stocks
    """
    removed = set(original) - set(diversified)
    if not removed:
        return

    cols = st.columns(3)
    cols[0].metric("Original Stocks",     len(original))
    cols[1].metric("After Diversification", len(diversified))
    cols[2].metric("Removed (Correlated)", len(removed))
