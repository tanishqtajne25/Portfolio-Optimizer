import json
import math
import os
import sys
import urllib.parse
import uuid
import textwrap

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.compare import compare_across_budgets
from algorithms.floyd_warshall import diversify_portfolio
from algorithms.greedy import greedy_select
from algorithms.knapsack_dp import knapsack_dp
from data.fetch_stocks import get_stock_data


st.set_page_config(page_title="NiftyOpt Premium Dashboard", page_icon="📈", layout="wide")


NIFTY_SYMBOLS = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "HINDUNILVR", "ITC", "LT", "SBIN", "BHARTIARTL",
        "WIPRO", "HCLTECH", "ASIANPAINT", "MARUTI", "BAJFINANCE",
        "KOTAKBANK", "AXISBANK", "NESTLEIND", "TITAN", "SUNPHARMA",
]

NAV_ITEMS = ["Dashboard", "Portfolio", "Algorithms", "Diversification", "Compare", "Settings"]


@st.cache_data(show_spinner="Fetching stock data from Yahoo Finance...")
def load_data(use_cache: bool = True):
        return get_stock_data(NIFTY_SYMBOLS, use_cache=use_cache)


def _ticker_name(ticker: str) -> str:
        return ticker.replace(".NS", "")


def _html_block(html: str) -> str:
    return textwrap.dedent(html).strip()


def _table_height(selected: list[str]) -> int:
    height = 100 + len(selected) * 52
    return max(200, min(600, height))


def _sparkline_svg(values: list[float], width: int = 96, height: int = 30, color: str = "#7C3AED") -> str:
        if not values:
                values = [0.0, 0.0]
        arr = np.array(values, dtype=float)
        if np.allclose(arr.max(), arr.min()):
                arr = arr + np.linspace(0, 1e-6, len(arr))

        x = np.linspace(0, width, len(arr))
        y = (arr - arr.min()) / (arr.max() - arr.min())
        y = (height - 2) - y * (height - 4)
        points = " ".join(f"{xx:.2f},{yy:.2f}" for xx, yy in zip(x, y))
        return (
                f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
                f"xmlns='http://www.w3.org/2000/svg'>"
                f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{points}'/></svg>"
        )


def _countup_html(value: float, prefix: str = "", suffix: str = "", decimals: int = 2, color: str = "#ffffff") -> str:
        uid = f"count_{uuid.uuid4().hex[:8]}"
        return f"""
        <div id="{uid}" style="font-family:'DM Mono',monospace;font-size:30px;font-weight:700;color:{color};letter-spacing:-0.02em;">{prefix}0{suffix}</div>
        <script>
            (function() {{
                const target = {value:.8f};
                const el = document.getElementById('{uid}');
                const duration = 1100;
                const start = performance.now();
                function frame(now) {{
                    const progress = Math.min((now - start) / duration, 1);
                    const eased = 1 - Math.pow(1 - progress, 3);
                    const current = target * eased;
                    el.textContent = '{prefix}' + current.toLocaleString(undefined, {{minimumFractionDigits:{decimals}, maximumFractionDigits:{decimals}}}) + '{suffix}';
                    if (progress < 1) requestAnimationFrame(frame);
                }}
                requestAnimationFrame(frame);
            }})();
        </script>
        """


def _chartjs_line(labels: list[str], series_a: list[float], series_b: list[float], title_a: str, title_b: str, height: int = 220):
        chart_id = f"chart_{uuid.uuid4().hex[:8]}"
        payload_labels = json.dumps(labels)
        payload_a = json.dumps(series_a)
        payload_b = json.dumps(series_b)
        components.html(
                f"""
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <div style="height:{height}px;"><canvas id="{chart_id}"></canvas></div>
                <script>
                    (function() {{
                        const ctx = document.getElementById('{chart_id}').getContext('2d');
                        new Chart(ctx, {{
                            type: 'line',
                            data: {{
                                labels: {payload_labels},
                                datasets: [
                                    {{
                                        label: '{title_a}',
                                        data: {payload_a},
                                        borderColor: '#7C3AED',
                                        borderWidth: 2,
                                        fill: false,
                                        tension: 0.35,
                                        pointRadius: 0
                                    }},
                                    {{
                                        label: '{title_b}',
                                        data: {payload_b},
                                        borderColor: '#0D9488',
                                        borderWidth: 2,
                                        fill: false,
                                        tension: 0.35,
                                        pointRadius: 0
                                    }}
                                ]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {{legend: {{display: true, labels: {{boxWidth: 14}}}}}},
                                scales: {{
                                    x: {{grid: {{display: false}}, ticks: {{maxTicksLimit: 6, color: '#6B7280'}}}},
                                    y: {{grid: {{color: '#EEF0F2'}}, ticks: {{color: '#6B7280'}}}}
                                }}
                            }}
                        }});
                    }})();
                </script>
                """,
                height=height + 30,
        )


def _chartjs_area(labels: list[str], values: list[float], height: int = 300):
        chart_id = f"area_{uuid.uuid4().hex[:8]}"
        payload_labels = json.dumps(labels)
        payload_vals = json.dumps(values)
        components.html(
                f"""
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <div style="height:{height}px;"><canvas id="{chart_id}"></canvas></div>
                <script>
                    (function() {{
                        const ctx = document.getElementById('{chart_id}').getContext('2d');
                        const gradient = ctx.createLinearGradient(0, 0, 0, 260);
                        gradient.addColorStop(0, 'rgba(124,58,237,0.35)');
                        gradient.addColorStop(1, 'rgba(124,58,237,0.03)');

                        new Chart(ctx, {{
                            type: 'line',
                            data: {{
                                labels: {payload_labels},
                                datasets: [{{
                                    label: 'Portfolio Value',
                                    data: {payload_vals},
                                    borderColor: '#7C3AED',
                                    borderWidth: 2.5,
                                    fill: true,
                                    backgroundColor: gradient,
                                    tension: 0.35,
                                    pointRadius: 0,
                                    pointHoverRadius: 4
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                interaction: {{mode: 'index', intersect: false}},
                                plugins: {{
                                    legend: {{display: false}},
                                    tooltip: {{
                                        callbacks: {{
                                            label: function(context) {{
                                                return ' Value: ₹' + Number(context.parsed.y).toLocaleString(undefined, {{maximumFractionDigits: 0}});
                                            }}
                                        }}
                                    }}
                                }},
                                scales: {{
                                    x: {{grid: {{display: false}}, ticks: {{maxTicksLimit: 7, color: '#6B7280'}}}},
                                    y: {{grid: {{color: '#EEF0F2'}}, ticks: {{callback: (v) => '₹' + Number(v).toLocaleString(), color: '#6B7280'}}}}
                                }}
                            }}
                        }});
                    }})();
                </script>
                """,
                height=height + 30,
        )


def _inject_styles():
        st.markdown(
                """
                <style>
                    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

                    :root {
                        --bg: #F4F5F7;
                        --surface: #ffffff;
                        --text: #1F2937;
                        --muted: #6B7280;
                        --border: #E5E7EB;
                        --primary: #7C3AED;
                        --teal: #0D9488;
                        --success: #10B981;
                        --danger: #EF4444;
                        --warning: #F59E0B;
                    }

                    html, body, [class*="css"]  {
                        font-family: 'Plus Jakarta Sans', sans-serif;
                        color: var(--text);
                        font-size: 14px;
                    }

                    .stApp {
                        background: var(--bg);
                    }

                    [data-testid="stHeader"], #MainMenu, footer {
                        visibility: hidden;
                        height: 0;
                    }

                    [data-testid="stSidebar"] {
                        background: #ffffff;
                        width: 220px !important;
                        min-width: 220px !important;
                        border-right: 1px solid var(--border);
                    }

                    [data-testid="stSidebar"] > div:first-child {
                        padding-top: 16px;
                    }

                    .block-container {
                        padding-top: 18px;
                        padding-bottom: 28px;
                        max-width: 1250px;
                    }

                    .card {
                        background: var(--surface);
                        border: 1px solid var(--border);
                        border-radius: 14px;
                        padding: 22px;
                        box-shadow: 0 8px 24px rgba(17, 24, 39, 0.06);
                        transition: transform .2s ease, box-shadow .2s ease;
                    }

                    .card:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 14px 30px rgba(17, 24, 39, 0.10);
                    }

                    .card,
                    .card * {
                        color: #111827 !important;
                    }

                    .kpi-gradient,
                    .kpi-gradient *,
                    .kpi-charcoal,
                    .kpi-charcoal * {
                        color: #111827 !important;
                    }

                    .mono, .mono * {
                        font-family: 'DM Mono', monospace !important;
                    }

                    .headerbar {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        gap: 16px;
                        margin-bottom: 16px;
                    }

                    .header-title {
                        font-size: 22px;
                        font-weight: 800;
                        line-height: 1.2;
                        color: var(--text);
                        margin-bottom: 3px;
                    }

                    .muted {
                        color: var(--muted);
                        font-size: 13px;
                    }

                    .search-pill {
                        background: #fff;
                        border: 1px solid var(--border);
                        border-radius: 999px;
                        padding: 10px 14px;
                        min-width: 300px;
                        color: #9AA0A6;
                        font-size: 13px;
                    }

                    .bell {
                        width: 38px;
                        height: 38px;
                        border-radius: 999px;
                        border: 1px solid var(--border);
                        display: inline-flex;
                        align-items: center;
                        justify-content: center;
                        background: #fff;
                        font-size: 16px;
                    }

                    .ticker-row {
                        display: flex;
                        gap: 16px;
                        overflow-x: auto;
                        padding-bottom: 4px;
                        margin-bottom: 16px;
                    }

                    .ticker-card {
                        min-width: 180px;
                        border-radius: 14px;
                        border: 1px solid rgba(229, 231, 235, 0.9);
                        padding: 14px;
                        box-shadow: 0 6px 16px rgba(17, 24, 39, 0.06);
                    }

                    .ticker-card .ticker-name {
                        color: #4B5563 !important;
                    }

                    .ticker-card .ticker-price {
                        color: #111827 !important;
                    }

                    .ticker-name {
                        font-size: 12px;
                        color: #4B5563;
                        font-weight: 700;
                    }

                    .ticker-price {
                        font-size: 18px;
                        font-weight: 700;
                        margin-top: 4px;
                        margin-bottom: 2px;
                    }

                    .badge {
                        border-radius: 999px;
                        display: inline-flex;
                        align-items: center;
                        padding: 5px 10px;
                        font-size: 12px;
                        font-weight: 700;
                        border: 1px solid transparent;
                    }

                    .badge.success { background: rgba(16,185,129,.13); color: #047857; }
                    .badge.danger { background: rgba(239,68,68,.12); color: #B91C1C; }
                    .badge.warning { background: rgba(245,158,11,.16); color: #92400E; }
                    .badge.primary { background: rgba(124,58,237,.13); color: var(--primary); }

                    .kpi-gradient {
                        background: linear-gradient(135deg, rgba(124,58,237,.16) 0%, rgba(167,139,250,.20) 100%);
                        color: #111827;
                    }

                    .kpi-charcoal {
                        background: linear-gradient(135deg, rgba(15,23,42,.06) 0%, rgba(59,130,246,.08) 100%);
                        color: #111827;
                    }

                    .kpi-white {
                        background: #ffffff;
                        color: var(--text);
                    }

                    .kpi-white .muted {
                        color: #6B7280;
                    }

                    .kpi-white .mono {
                        color: #111827;
                    }

                    .kpi-white .badge.primary {
                        color: #6D28D9;
                    }

                    .kpi-sub {
                        color: #6B7280;
                        font-size: 12px;
                    }

                    .table-wrap {
                        border: 1px solid #F0F2F5;
                        border-radius: 12px;
                        overflow: hidden;
                        margin-top: 12px;
                    }

                    table.breakdown {
                        width: 100%;
                        border-collapse: collapse;
                        font-size: 12px;
                    }

                    table.breakdown thead th {
                        text-align: left;
                        padding: 12px;
                        background: #F8FAFC;
                        color: #4B5563;
                        border-bottom: 1px solid #EEF2F7;
                        white-space: nowrap;
                    }

                    table.breakdown tbody td {
                        padding: 12px;
                        border-bottom: 1px solid #F2F4F8;
                        vertical-align: middle;
                    }

                    table.breakdown tbody tr:hover {
                        background: rgba(124,58,237,0.06);
                    }

                    .ticker-pill {
                        display: inline-flex;
                        align-items: center;
                        border-radius: 999px;
                        font-size: 12px;
                        font-weight: 700;
                        background: rgba(124,58,237,.12);
                        color: #6D28D9;
                        padding: 5px 10px;
                    }

                    .bar-bg {
                        background: #F3F4F6;
                        border-radius: 999px;
                        overflow: hidden;
                        width: 120px;
                        height: 7px;
                        margin-top: 5px;
                    }

                    .bar-fill.green { background: #10B981; height: 100%; }
                    .bar-fill.red { background: #EF4444; height: 100%; }

                    .settings-card .stSlider, .settings-card .stRadio, .settings-card .stToggle {
                        background: #fff;
                    }

                    .stButton > button {
                        border-radius: 12px;
                        font-weight: 700;
                        min-height: 40px;
                    }

                    .stButton > button,
                    .stButton > button p,
                    .stButton > button span {
                        color: #111827 !important;
                    }

                    .run-btn button {
                        width: 100%;
                        background: rgba(124,58,237,.14) !important;
                        color: #4C1D95 !important;
                        border: 1px solid rgba(124,58,237,.24) !important;
                    }

                    .refresh-btn button {
                        width: 100%;
                        border: 1px solid rgba(124,58,237,.24) !important;
                        color: #7C3AED !important;
                        background: transparent !important;
                        box-shadow: none !important;
                    }

                    [data-testid="stSidebar"] .refresh-btn button:hover {
                        background: rgba(124,58,237,.06) !important;
                        border-color: #6D28D9 !important;
                        color: #6D28D9 !important;
                    }

                    div[data-baseweb="radio"] > div {
                        gap: 10px;
                    }

                    div[data-baseweb="radio"] label {
                        color: #111827 !important;
                    }

                    div[data-baseweb="radio"] label,
                    div[data-baseweb="radio"] label span,
                    div[data-baseweb="radio"] label p,
                    div[data-baseweb="radio"] [role="radio"] {
                        color: #111827 !important;
                        opacity: 1 !important;
                    }

                    div[data-baseweb="radio"] * {
                        color: #111827 !important;
                    }

                    div[data-testid="stRadio"] * {
                        color: #111827 !important;
                        opacity: 1 !important;
                    }

                    div[data-testid="stRadio"] label,
                    div[data-testid="stRadio"] label *,
                    div[data-testid="stRadio"] [role="radio"],
                    div[data-testid="stRadio"] [role="radio"] * {
                        color: #111827 !important;
                        opacity: 1 !important;
                    }

                    div[data-baseweb="radio"] [aria-checked="true"] {
                        color: #111827 !important;
                    }

                    .tab-like [role="radiogroup"] {
                        gap: 8px;
                    }

                    .tab-like [role="radiogroup"] label,
                    .tab-like [role="radiogroup"] label span,
                    .tab-like [role="radiogroup"] label p {
                        color: #111827 !important;
                        opacity: 1 !important;
                    }

                    .tab-like [role="radiogroup"] * {
                        color: #111827 !important;
                    }

                    .tab-like {
                        background: #ffffff;
                    }

                    .nav-item {
                        display: block;
                        padding: 9px 10px;
                        margin-bottom: 6px;
                        border-radius: 10px;
                        color: #4B5563;
                        text-decoration: none;
                        font-size: 13px;
                        font-weight: 600;
                        border-left: 3px solid transparent;
                    }

                    .nav-item.active {
                        border-left: 3px solid #7C3AED;
                        background: rgba(124,58,237,0.10);
                        color: #5B21B6;
                    }

                    .analytics-tabs {
                        display: inline-flex;
                        gap: 16px;
                        border-bottom: 1px solid #EEF2F7;
                        padding-bottom: 8px;
                        margin-bottom: 12px;
                    }

                    .analytics-tabs .active {
                        color: #7C3AED;
                        border-bottom: 2px solid #7C3AED;
                        padding-bottom: 6px;
                        font-weight: 700;
                    }
                </style>
                """,
                unsafe_allow_html=True,
        )


def _render_sidebar(active_nav: str) -> bool:
        gradient_logo = (
                "<span style='display:inline-flex;width:18px;height:18px;border-radius:5px;"
                "background:linear-gradient(135deg,#7C3AED 0%,#0D9488 100%);margin-right:8px;'></span>"
        )
        st.sidebar.markdown(
                (
                        "<div style='padding:2px 2px 12px 2px;'>"
                        f"<div style='display:flex;align-items:center;font-weight:800;font-size:20px;color:#111827;'>{gradient_logo}NiftyOpt</div>"
                        "<div style='margin-left:27px;margin-top:2px;font-size:11px;color:#6B7280;'>NIFTY 50 · DSA Project</div>"
                        "</div>"
                ),
                unsafe_allow_html=True,
        )

        nav_icons = {
                "Dashboard": "▦",
                "Portfolio": "◈",
                "Algorithms": "◎",
                "Diversification": "◇",
                "Compare": "↹",
                "Settings": "⚙",
        }

        nav_html = "<div style='margin-top:10px;margin-bottom:18px;'>"
        for item in NAV_ITEMS:
                cls = "nav-item active" if item == active_nav else "nav-item"
                qp = urllib.parse.quote(item)
                nav_html += f"<a class='{cls}' href='?nav={qp}'>{nav_icons[item]}&nbsp;&nbsp;{item}</a>"
        nav_html += "</div>"
        st.sidebar.markdown(nav_html, unsafe_allow_html=True)

        st.sidebar.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
        st.sidebar.markdown("<div class='refresh-btn'>", unsafe_allow_html=True)
        refresh = st.sidebar.button("↻ Refresh Data", use_container_width=True)
        st.sidebar.markdown("</div>", unsafe_allow_html=True)

        st.sidebar.markdown(
                "<div class='mono' style='margin-top:14px;font-size:10px;color:#6B7280;'>Simulation only. Not financial advice.</div>",
                unsafe_allow_html=True,
        )
        return refresh


def _compute_result(metrics: dict, log_returns: pd.DataFrame, algo: str, budget: int, apply_div: bool, threshold: float):
        if algo == "Greedy":
                base = greedy_select(metrics, budget)
        else:
                base = knapsack_dp(metrics, budget)

        selected = base["selected"]
        div_payload = None
        if apply_div and selected:
                div_payload = diversify_portfolio(selected, metrics, log_returns, threshold=threshold)
                selected = div_payload["diversified"]

        invested = float(sum(metrics[t]["price"] for t in selected))
        expected_return = float(sum(metrics[t]["expected_return"] for t in selected))

        return {
                "base": base,
                "selected": selected,
                "invested": round(invested, 2),
                "expected_return": round(expected_return, 6),
                "div": div_payload,
        }


def _table_html(metrics: dict, selected: list[str], budget: int, status_badge: str, status_class: str) -> str:
        if not selected:
                return "<div class='muted'>No stocks selected for the chosen budget.</div>"

        max_ret = max([metrics[t]["expected_return"] for t in selected], default=1e-9)
        max_risk = max([metrics[t]["risk"] for t in selected], default=1e-9)
        rows = []

        for t in selected:
                m = metrics[t]
                ratio = m["expected_return"] / max(m["risk"], 1e-9)
                shares = max(1, int((budget / max(len(selected), 1)) // max(m["price"], 1)))
                amount = shares * m["price"]
                ret_w = max(5, int((m["expected_return"] / max(max_ret, 1e-9)) * 100))
                risk_w = max(5, int((m["risk"] / max(max_risk, 1e-9)) * 100))
                ratio_color = "#10B981" if ratio > 1.2 else ("#EF4444" if ratio < 1.0 else "#F59E0B")

                rows.append(
                        f"""
                        <tr>
                            <td><span class='ticker-pill'>{_ticker_name(t)}</span></td>
                            <td class='mono'>₹{m['price']:,.2f}</td>
                            <td>
                                <div class='mono' style='font-size:12px;color:#047857;'>{m['expected_return']:.2%}</div>
                                <div class='bar-bg'><div class='bar-fill green' style='width:{ret_w}%'></div></div>
                            </td>
                            <td>
                                <div class='mono' style='font-size:12px;color:#B91C1C;'>{m['risk']:.2%}</div>
                                <div class='bar-bg'><div class='bar-fill red' style='width:{risk_w}%'></div></div>
                            </td>
                            <td class='mono' style='color:{ratio_color};font-weight:700;'>{ratio:.2f}</td>
                            <td class='mono'>{shares}</td>
                            <td class='mono'>₹{amount:,.0f}</td>
                        </tr>
                        """
                )

        return f"""
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    font-family: 'Plus Jakarta Sans', sans-serif;
                    color: #111827;
                    background: transparent;
                }}

                .table-shell {{
                    color: #111827;
                }}

                .table-shell .muted {{
                    color: #6B7280;
                }}

                .table-shell .badge {{
                    border-radius: 999px;
                    display: inline-flex;
                    align-items: center;
                    padding: 5px 10px;
                    font-size: 12px;
                    font-weight: 700;
                    border: 1px solid transparent;
                }}

                .table-shell .badge.success {{ background: rgba(16,185,129,.13); color: #047857; }}
                .table-shell .badge.warning {{ background: rgba(245,158,11,.16); color: #92400E; }}

                .table-shell .ticker-pill {{
                    display: inline-flex;
                    align-items: center;
                    border-radius: 999px;
                    font-size: 12px;
                    font-weight: 700;
                    background: rgba(124,58,237,.12);
                    color: #6D28D9;
                    padding: 5px 10px;
                }}

                .table-shell .table-wrap {{
                    border: 1px solid #F0F2F5;
                    border-radius: 12px;
                    overflow: hidden;
                    margin-top: 12px;
                    background: #fff;
                }}

                .table-shell table.breakdown {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 12px;
                }}

                .table-shell table.breakdown thead th {{
                    text-align: left;
                    padding: 12px;
                    background: #F8FAFC;
                    color: #4B5563;
                    border-bottom: 1px solid #EEF2F7;
                    white-space: nowrap;
                }}

                .table-shell table.breakdown tbody td {{
                    padding: 12px;
                    border-bottom: 1px solid #F2F4F8;
                    vertical-align: middle;
                    color: #111827;
                }}

                .table-shell table.breakdown tbody tr:hover {{
                    background: rgba(124,58,237,0.06);
                }}

                .table-shell .bar-bg {{
                    background: #F3F4F6;
                    border-radius: 999px;
                    overflow: hidden;
                    width: 120px;
                    height: 7px;
                    margin-top: 5px;
                }}

                .table-shell .bar-fill.green {{ background: #10B981; height: 100%; }}
                .table-shell .bar-fill.red {{ background: #EF4444; height: 100%; }}

                .table-shell .mono {{
                    font-family: 'DM Mono', monospace;
                    color: #111827;
                }}
            </style>
            <div class='table-shell' style='display:flex;justify-content:space-between;align-items:center;gap:8px;'>
                <div style='font-size:18px;font-weight:800;'>Stock Breakdown - Selected Portfolio</div>
                <span class='badge {status_class}'>{status_badge}</span>
            </div>
            <div class='table-shell table-wrap'>
                <table class='breakdown'>
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Price</th>
                            <th>Expected Return</th>
                            <th>Risk / Volatility</th>
                            <th>Return/Risk</th>
                            <th>Shares</th>
                            <th>Amount Invested</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rows)}
                    </tbody>
                </table>
            </div>
        """


_inject_styles()

for key, val in {
        "budget": 50000,
        "algo_main": "DP Knapsack",
        "apply_div": True,
        "corr_threshold": 0.85,
        "analytics_horizon": "1M",
}.items():
        if key not in st.session_state:
                st.session_state[key] = val


active_nav = st.query_params.get("nav", "Dashboard")
if active_nav not in NAV_ITEMS:
        active_nav = "Dashboard"

refresh_data = _render_sidebar(active_nav)
metrics, log_returns = load_data(use_cache=not refresh_data)

header_col1, header_col2 = st.columns([3, 2])
with header_col1:
        st.markdown(
                _html_block("""
                <div class='headerbar'>
                    <div>
                        <div class='header-title'>Hello, Investor 👋</div>
                        <div class='muted'>NIFTY 50 Portfolio · DP Knapsack + Floyd-Warshall Diversification</div>
                    </div>
                </div>
                """),
                unsafe_allow_html=True,
        )
with header_col2:
        st.markdown(
            _html_block("""
            <div style='display:flex;justify-content:flex-end;align-items:center;gap:10px;margin-top:6px;'>
                <div class='search-pill'>Search stocks and more...</div>
                <div class='bell'>🔔</div>
            </div>
            """),
            unsafe_allow_html=True,
        )


tickers_for_strip = sorted(metrics.items(), key=lambda kv: kv[1]["expected_return"], reverse=True)[:6]
pastel = ["#EEE9FF", "#EAFBF5", "#FFF3E8", "#EAF3FF", "#F3EDFF", "#E9FBFA"]
strip_html = "<div class='ticker-row'>"
for idx, (ticker, m) in enumerate(tickers_for_strip):
        series = log_returns[ticker].dropna().tail(24).tolist() if ticker in log_returns.columns else [0.0, 0.0]
        pct_1d = (series[-1] * 100) if series else 0.0
        spark = _sparkline_svg(np.cumsum(series).tolist(), color="#7C3AED" if pct_1d >= 0 else "#EF4444")
        change_cls = "success" if pct_1d >= 0 else "danger"
        strip_html += _html_block(f"""
            <div class='ticker-card' style='background:{pastel[idx % len(pastel)]};'>
                <div class='ticker-name'>{_ticker_name(ticker)}</div>
                <div class='ticker-price mono'>₹{m['price']:,.2f}</div>
                <span class='badge {change_cls}'>{pct_1d:+.2f}%</span>
                <div style='margin-top:8px;'>{spark}</div>
            </div>
        """)
strip_html += "</div>"
st.markdown(strip_html, unsafe_allow_html=True)


settings_col_left, settings_col_right = st.columns([2, 1])

with settings_col_left:
        st.markdown("<div class='card settings-card'>", unsafe_allow_html=True)
        show_settings = active_nav == "Settings"
        if show_settings:
                st.markdown("<div style='font-size:18px;font-weight:800;margin-bottom:8px;'>Settings</div>", unsafe_allow_html=True)
                with st.form("settings_form"):
                        budget = st.slider("Budget (₹)", min_value=10_000, max_value=5_00_000, value=int(st.session_state["budget"]), step=5_000)
                        algo = st.radio(
                                "Algorithm",
                                ["Greedy", "DP Knapsack", "Both"],
                                index=["Greedy", "DP Knapsack", "Both"].index(st.session_state["algo_main"]),
                                horizontal=True,
                        )
                        st.markdown("<div class='muted'>Greedy: fast approximation · DP Knapsack: exact optimizer · Both: side-by-side output</div>", unsafe_allow_html=True)
                        apply_div = st.toggle("Apply diversification", value=bool(st.session_state["apply_div"]))
                        corr_threshold = st.session_state["corr_threshold"]
                        if apply_div:
                                corr_threshold = st.slider("Correlation threshold", min_value=0.5, max_value=1.0, value=float(st.session_state["corr_threshold"]), step=0.05)

                        st.markdown("<div class='run-btn'>", unsafe_allow_html=True)
                        run_clicked = st.form_submit_button("Run Optimizer", use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                if run_clicked:
                        st.session_state["budget"] = budget
                        st.session_state["algo_main"] = algo
                        st.session_state["apply_div"] = apply_div
                        st.session_state["corr_threshold"] = corr_threshold
        else:
                st.markdown(
                    _html_block("""
                    <div style='font-size:16px;font-weight:700;margin-bottom:6px;color:#111827;'>Quick View</div>
                    <div class='muted' style='color:#6B7280;'>Use the Settings nav item to change budget, algorithm, and diversification controls.</div>
                    """),
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)


algo_for_cards = st.session_state["algo_main"]
budget_for_cards = int(st.session_state["budget"])
apply_div_for_cards = bool(st.session_state["apply_div"])
corr_for_cards = float(st.session_state["corr_threshold"])

dp_payload = _compute_result(metrics, log_returns, "DP Knapsack", budget_for_cards, apply_div_for_cards, corr_for_cards)
greedy_payload = _compute_result(metrics, log_returns, "Greedy", budget_for_cards, apply_div_for_cards, corr_for_cards)

if algo_for_cards == "Greedy":
    primary_payload = greedy_payload
elif algo_for_cards == "Both":
    primary_payload = dp_payload if dp_payload["expected_return"] >= greedy_payload["expected_return"] else greedy_payload
else:
    primary_payload = dp_payload


with settings_col_right:
    top_stock = None
    if primary_payload["selected"]:
        top_stock = max(
            primary_payload["selected"],
            key=lambda t: metrics[t]["expected_return"] / max(metrics[t]["risk"], 1e-9),
        )
    top_price = metrics[top_stock]["price"] if top_stock else 0.0
    top_ratio = (metrics[top_stock]["expected_return"] / max(metrics[top_stock]["risk"], 1e-9)) if top_stock else 0.0
    mini_trend = _sparkline_svg(
        np.cumsum(log_returns[top_stock].dropna().tail(24).tolist()).tolist() if top_stock else [0, 0],
        color="#10B981",
    )
    st.markdown(
        _html_block(f"""
        <div class='card kpi-white'>
            <div class='muted' style='font-weight:700;color:#6B7280;'>Top Performer (Return/Risk)</div>
            <div style='margin-top:8px;display:flex;align-items:center;justify-content:space-between;'>
                <span class='badge primary'>{_ticker_name(top_stock) if top_stock else 'N/A'}</span>
                <div class='mono' style='font-size:13px;color:#6B7280;'>₹{top_price:,.2f}</div>
            </div>
            <div class='mono' style='margin-top:8px;font-size:24px;color:#111827;font-weight:700;'>{top_ratio:.2f}x</div>
            <div style='margin-top:6px;'>{mini_trend}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


k1, k2, k3 = st.columns([1, 1, 1], gap="medium")
with k1:
    spark = _sparkline_svg(
        np.cumsum(
            log_returns[primary_payload["selected"][0]].dropna().tail(20).tolist()
            if primary_payload["selected"] else [0.0, 0.0]
        ).tolist(),
        color="#D1FAE5",
    )
    st.markdown("<div class='card kpi-gradient'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;font-weight:700;color:#6B7280;'>Total Expected Return</div>", unsafe_allow_html=True)
    components.html(_countup_html(primary_payload["expected_return"] * 100, suffix="%", color="#111827"), height=58)
    st.markdown("<span class='badge success'>+ Optimized</span>", unsafe_allow_html=True)
    st.markdown(f"<div style='margin-top:10px;opacity:.95'>{spark}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with k2:
    utilization = (primary_payload["invested"] / budget_for_cards * 100) if budget_for_cards else 0
    st.markdown("<div class='card kpi-charcoal'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;font-weight:700;color:#6B7280;'>Total Invested</div>", unsafe_allow_html=True)
    components.html(_countup_html(primary_payload["invested"], prefix="₹", decimals=0, color="#111827"), height=58)
    st.markdown(f"<div class='kpi-sub'>Budget utilization: {utilization:.1f}%</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='margin-top:10px;display:flex;justify-content:flex-end;'><span class='badge primary' style='background:rgba(124,58,237,.12);color:#6D28D9;'>→</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with k3:
        st.markdown(
                _html_block(f"""
                <div class='card kpi-white'>
                    <div class='muted' style='font-weight:700;color:#6B7280;'>Top Stock by Return/Risk</div>
                    <div style='margin-top:8px;display:flex;align-items:center;justify-content:space-between;'>
                        <span class='badge primary'>{_ticker_name(top_stock) if top_stock else 'N/A'}</span>
                        <span class='mono' style='font-size:13px;'>₹{top_price:,.2f}</span>
                    </div>
                    <div class='mono' style='font-size:28px;font-weight:700;margin-top:8px;color:#111827;'>{top_ratio:.2f}</div>
                    <div>{mini_trend}</div>
                </div>
                """),
                unsafe_allow_html=True,
        )


left, right = st.columns([2, 1], gap="medium")

with left:
    st.markdown("<div class='card tab-like'>", unsafe_allow_html=True)
    algo_toggle = st.radio("Algorithm Toggle", ["DP Knapsack", "Greedy", "Both"], horizontal=True, label_visibility="collapsed")

    if algo_toggle == "Greedy":
        table_payload = greedy_payload
    elif algo_toggle == "DP Knapsack":
        table_payload = dp_payload
    else:
        table_payload = dp_payload if dp_payload["expected_return"] >= greedy_payload["expected_return"] else greedy_payload

    penalized_count = len(table_payload["div"]["penalized"]) if table_payload["div"] else 0
    status_badge = "✓ Well Diversified" if penalized_count == 0 else "⚠ Correlated Pairs Found"
    status_class = "success" if penalized_count == 0 else "warning"

    components.html(
        _table_html(metrics, table_payload["selected"], budget_for_cards, status_badge, status_class),
        height=_table_height(table_payload["selected"]),
        scrolling=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    selected_for_snapshot = table_payload["selected"]
    if selected_for_snapshot:
        portfolio_daily = log_returns[selected_for_snapshot].mean(axis=1)
    else:
        portfolio_daily = log_returns.mean(axis=1)

    index_daily = log_returns.mean(axis=1)
    port_vol = float(portfolio_daily.std() * math.sqrt(252))
    sharpe = float((portfolio_daily.mean() * 252) / max(port_vol, 1e-9))
    beta = float(portfolio_daily.cov(index_daily) / max(index_daily.var(), 1e-9))
    max_dd = float((np.exp(portfolio_daily.cumsum()) / np.exp(portfolio_daily.cumsum()).cummax() - 1).min())
    nifty_level = float(22000 * np.exp(index_daily.cumsum().iloc[-1]))

    snap_labels = [d.strftime("%d %b") for d in portfolio_daily.tail(40).index]
    snap_port = (100 * np.exp(portfolio_daily.tail(40).cumsum())).tolist()
    snap_idx = (100 * np.exp(index_daily.tail(40).cumsum())).tolist()

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:17px;font-weight:800;'>Snapshot</div>", unsafe_allow_html=True)
    st.markdown(_html_block(f"""
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;'>
            <div><div class='muted'>Nifty 50 Index</div><div class='mono' style='font-size:16px;font-weight:700;'>₹{nifty_level:,.0f}</div></div>
            <div><div class='muted'>Portfolio Beta</div><div class='mono' style='font-size:16px;font-weight:700;'>{beta:.2f}</div></div>
            <div><div class='muted'>Estimated Sharpe</div><div class='mono' style='font-size:16px;font-weight:700;'>{sharpe:.2f}</div></div>
            <div><div class='muted'>Max Drawdown</div><div class='mono' style='font-size:16px;font-weight:700;color:#EF4444;'>{max_dd:.2%}</div></div>
        </div>
        """), unsafe_allow_html=True)
    _chartjs_line(snap_labels, snap_port, snap_idx, "Portfolio", "Nifty 50", height=170)
    st.markdown("</div>", unsafe_allow_html=True)

    all_tickers = set(metrics.keys())
    watch = list(sorted(all_tickers - set(selected_for_snapshot)))[:7]
    st.markdown("<div class='card' style='margin-top:16px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:17px;font-weight:800;color:#111827;'>Watchlist</div>", unsafe_allow_html=True)
    wl_rows = []
    for t in watch:
        last_move = float(log_returns[t].dropna().iloc[-1] * 100) if t in log_returns.columns else 0.0
        color = "#10B981" if last_move >= 0 else "#EF4444"
        wl_rows.append(
            f"""
            <div style='display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid #F1F5F9;'>
              <div style='display:flex;align-items:center;gap:10px;'>
                                <div style='width:28px;height:28px;border-radius:999px;background:rgba(124,58,237,.14);display:flex;align-items:center;justify-content:center;font-weight:700;color:#6D28D9;font-size:12px;'>{_ticker_name(t)[0]}</div>
                <div>
                                    <div style='font-size:13px;font-weight:700;color:#111827;'>{_ticker_name(t)}</div>
                                    <div class='muted' style='font-size:12px;color:#6B7280;'>{_ticker_name(t)}</div>
                </div>
              </div>
              <div style='text-align:right;'>
                                <div class='mono' style='font-size:12px;color:#111827;'>₹{metrics[t]['price']:,.0f}</div>
                <div class='mono' style='font-size:12px;color:{color};'>{last_move:+.2f}%</div>
              </div>
                            <div style='margin-left:8px;color:#9CA3AF;'>☆</div>
            </div>
            """
        )
    st.markdown(_html_block("".join(wl_rows)), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='card' style='margin-top:16px;'>", unsafe_allow_html=True)
st.markdown("<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:10px;'><div style='font-size:18px;font-weight:800;color:#111827;'>Portfolio Analytics</div></div>", unsafe_allow_html=True)

horizons = ["1D", "5D", "1M", "3M", "6M", "1Y", "Max"]
horizon = st.radio("Analytics Horizon", horizons, horizontal=True, label_visibility="collapsed", index=horizons.index(st.session_state.get("analytics_horizon", "1M")))
st.session_state["analytics_horizon"] = horizon

horizon_map = {"1D": 1, "5D": 5, "1M": 22, "3M": 66, "6M": 132, "1Y": 252, "Max": 9999}
n_points = horizon_map[horizon]

selected_for_analytics = table_payload["selected"] if table_payload["selected"] else list(metrics.keys())[:5]
port_daily = log_returns[selected_for_analytics].mean(axis=1)
if n_points < len(port_daily):
    port_daily = port_daily.tail(n_points)

start_capital = max(table_payload["invested"], 100000)
path = (start_capital * np.exp(port_daily.cumsum())).tolist()
labels = [d.strftime("%d %b") for d in port_daily.index]

curr_val = path[-1] if path else start_capital
pct_chg = ((curr_val / start_capital) - 1) * 100 if start_capital else 0
chg_color = "#10B981" if pct_chg >= 0 else "#EF4444"

st.markdown(_html_block(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;color:#111827;'>
            <div>
                <div class='muted' style='color:#6B7280;'>Current Portfolio Value</div>
                <div class='mono' style='font-size:24px;font-weight:800;color:#111827;'>₹{curr_val:,.0f}</div>
            </div>
            <div class='mono' style='font-size:16px;font-weight:700;color:{chg_color};'>{pct_chg:+.2f}%</div>
        </div>
        """), unsafe_allow_html=True)

_chartjs_area(labels, path, height=300)
st.markdown("</div>", unsafe_allow_html=True)


if active_nav == "Compare":
        compare_rows = compare_across_budgets(metrics, [20_000, 50_000, 100_000, 200_000])
        st.markdown("<div class='card' style='margin-top:16px;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:17px;font-weight:800;margin-bottom:8px;'>Greedy vs DP Comparison</div>", unsafe_allow_html=True)
        comp_df = pd.DataFrame([
                {
                        "Budget": f"₹{r['budget']:,.0f}",
                        "Greedy Return": f"{r['greedy_return']:.2%}",
                        "DP Return": f"{r['dp_return']:.2%}",
                        "DP Improvement": f"{r['improvement_pct']:+.2f}%",
                }
                for r in compare_rows
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
