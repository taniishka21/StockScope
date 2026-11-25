"""
StockScope — Stock Market Risk & Performance Analyzer
=====================================================
A Python 3.14 + Streamlit interactive web application for analyzing
historical risk, return, and performance of Indian NSE equities.

Designed as a Data Science / CSE College Mini-Project.
"""

import os
from typing import Dict
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# StockScope project modules
import data
import analysis
import risk_model

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StockScope — Stock Market Risk & Performance Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, academic mini-project typography and cards
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
        color: #1E3A8A;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: left;
    }
    .metric-card-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-card-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .risk-badge-low {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-mod {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #78350F;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. DATA CACHING LAYER
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cached_stock_data(ticker_symbol: str, period_key: str) -> pd.DataFrame:
    """
    Retrieves and caches historical stock data for 1 hour to prevent redundant API calls.
    """
    return data.get_stock_data(ticker_symbol, period=period_key)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_cached_metadata(ticker_symbol: str) -> Dict[str, str]:
    """
    Retrieves and caches descriptive company metadata.
    """
    return data.get_stock_metadata(ticker_symbol)


# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
# Optional project logo
if os.path.exists("assets/logo.png"):
    st.sidebar.image("assets/logo.png", use_container_width=True)

st.sidebar.title("📈 StockScope")
st.sidebar.caption("Indian Equities Quantitative Risk Analyzer")
st.sidebar.markdown("---")

# Stock selector with friendly names
friendly_stock_name = st.sidebar.selectbox(
    "Select Stock (NSE)",
    options=list(data.SUPPORTED_TICKERS.keys()),
    index=0,
    help="Select an Indian blue-chip stock listed on the National Stock Exchange (NSE).",
)
ticker = data.SUPPORTED_TICKERS[friendly_stock_name]

# Timeframe period selector
period_display = st.sidebar.selectbox(
    "Analysis Timeframe",
    options=list(data.PERIOD_MAPPING.keys()),
    index=2,  # Default: 1 Year
    help="Choose the historical duration for calculating risk and return metrics.",
)
period_code = data.PERIOD_MAPPING[period_display]

# Benchmark parameters
risk_free_rate_pct = st.sidebar.slider(
    "Risk-Free Rate Benchmark (%)",
    min_value=4.0,
    max_value=9.0,
    value=6.0,
    step=0.25,
    help="Annual risk-free benchmark yield (e.g. 10-Yr Indian Government Sovereign Bond yield ~6%-7%).",
)
risk_free_rate = risk_free_rate_pct / 100.0

st.sidebar.markdown("---")
# Stock info summary in sidebar
meta = fetch_cached_metadata(ticker)
st.sidebar.markdown(f"**Asset:** {meta.get('shortName', friendly_stock_name)}")
st.sidebar.markdown(f"**Symbol:** `{ticker}`")
st.sidebar.markdown(f"**Sector:** {meta.get('sector', 'N/A')}")
st.sidebar.markdown(f"**Exchange:** {meta.get('exchange', 'NSE')}")

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Academic Mini-Project**\n\n"
    "Built with **Python 3.14 + Streamlit** for quantitative financial analysis."
)


# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD CONTENT
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">StockScope — Stock Market Risk & Performance Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-title">Evaluating historical risk, returns, and drawdown dynamics for <strong>{meta.get("shortName", friendly_stock_name)} ({ticker})</strong> over <strong>{period_display}</strong>.</div>',
    unsafe_allow_html=True,
)

# Fetch stock data with robust error handling
try:
    with st.spinner(f"Retrieving market data for {ticker}..."):
        raw_df = fetch_cached_stock_data(ticker, period_code)
except Exception as exc:
    st.error(
        f"⚠️ **Unable to retrieve stock data for `{ticker}`.**\n\n"
        f"Details: {exc}\n\n"
        "Please verify your internet connection or try a different stock/period from the sidebar."
    )
    st.stop()

if raw_df is None or raw_df.empty or len(raw_df) < 5:
    st.warning("⚠️ Insufficient historical data returned for the selected stock and timeframe.")
    st.stop()

# Add moving averages and calculate quantitative metrics
df = analysis.add_moving_averages(raw_df, windows=(20, 50))
assessment = risk_model.evaluate_stock_risk(df, risk_free_rate=risk_free_rate)

current_price = assessment["current_price"]
period_ret = assessment["period_return"]
ann_ret = assessment["annualized_return"]
ann_vol = assessment["annualized_volatility"]
sharpe = assessment["sharpe_ratio"]
max_dd = assessment["max_drawdown"]
overall_risk = assessment["overall_risk_level"]
comp_score = assessment["composite_risk_score"]

# Daily price difference
prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else current_price
daily_change = current_price - prev_close
daily_pct = (daily_change / prev_close) if prev_close > 0 else 0.0

# -----------------------------------------------------------------------------
# 5. FOUR PRIMARY METRICS CARDS
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    delta_str = f"{daily_change:+.2f} ({daily_pct:+.2%}) today"
    st.metric(
        label="Current Close Price",
        value=f"₹{current_price:,.2f}",
        delta=delta_str,
    )

with col2:
    period_color = "normal" if period_ret >= 0 else "inverse"
    st.metric(
        label=f"Period Return ({period_display})",
        value=f"{period_ret:+.2%}",
        delta=f"Annualized: {ann_ret:+.2%}",
        delta_color=period_color,
    )

with col3:
    st.metric(
        label="Annualized Volatility (σ)",
        value=f"{ann_vol:.2%}",
        delta=f"Risk: {analysis.get_risk_level(ann_vol)}",
        delta_color="off",
    )

with col4:
    sharpe_badge = "Excellent" if sharpe > 1.0 else ("Positive" if sharpe > 0 else "Sub-optimal")
    st.metric(
        label=f"Sharpe Ratio (Rf={risk_free_rate:.1%})",
        value=f"{sharpe:.2f}",
        delta=sharpe_badge,
        delta_color="normal" if sharpe > 0 else "inverse",
    )

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. TABBED SECTIONS (PERFORMANCE, RISK, DATA)
# -----------------------------------------------------------------------------
tab_perf, tab_risk, tab_data = st.tabs(["📊 Performance", "🛡️ Risk Assessment", "📋 Historical Data"])

# -----------------------------------------------------------------------------
# TAB 1: PERFORMANCE
# -----------------------------------------------------------------------------
with tab_perf:
    st.subheader(f"Historical Price & Trend Analysis ({period_display})")

    # Interactive Plotly Candlestick / Line Chart with MAs and Volume
    fig_price = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{friendly_stock_name} Price Trend & Moving Averages", "Trading Volume"),
    )

    # Close Price line
    fig_price.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#2563EB", width=2),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Close:</b> ₹%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # MA20 line
    if "MA20" in df.columns:
        fig_price.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA20"],
                mode="lines",
                name="20-Day MA (Short-term)",
                line=dict(color="#10B981", width=1.5, dash="dot"),
                hovertemplate="<b>MA20:</b> ₹%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # MA50 line
    if "MA50" in df.columns and len(df) >= 20:
        fig_price.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA50"],
                mode="lines",
                name="50-Day MA (Medium-term)",
                line=dict(color="#F59E0B", width=1.5, dash="dash"),
                hovertemplate="<b>MA50:</b> ₹%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Volume Bar Chart
    volume_colors = [
        "#10B981" if c >= o else "#EF4444"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig_price.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=volume_colors,
            hovertemplate="<b>Volume:</b> %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig_price.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig_price.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig_price.update_yaxes(title_text="Shares", row=2, col=1)

    st.plotly_chart(fig_price, use_container_width=True)

    # Daily Return Analysis Charts
    col_ret1, col_ret2 = st.columns(2)

    returns = analysis.calculate_returns(df)

    with col_ret1:
        st.subheader("Daily Returns Fluctuations")
        fig_ret = go.Figure()
        fig_ret.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns * 100,
                mode="lines",
                name="Daily Return %",
                line=dict(color="#059669", width=1.2),
                hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Return:</b> %{y:+.2f}%<extra></extra>",
            )
        )
        fig_ret.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig_ret.update_layout(
            template="plotly_white",
            height=320,
            margin=dict(l=30, r=30, t=30, b=30),
            yaxis_title="Daily Return (%)",
            xaxis_title="Date",
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    with col_ret2:
        st.subheader("Daily Return Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=returns * 100,
                nbinsx=35,
                name="Returns",
                marker_color="#3B82F6",
                opacity=0.75,
                hovertemplate="<b>Range:</b> %{x:.2f}%<br><b>Count:</b> %{y}<extra></extra>",
            )
        )
        fig_hist.update_layout(
            template="plotly_white",
            height=320,
            margin=dict(l=30, r=30, t=30, b=30),
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: RISK ASSESSMENT
# -----------------------------------------------------------------------------
with tab_risk:
    st.subheader("Multi-Factor Risk Classification")

    # Risk badge formatting
    if overall_risk == "LOW":
        badge_html = '<span class="risk-badge-low">LOW RISK</span>'
    elif overall_risk == "MODERATE":
        badge_html = '<span class="risk-badge-mod">MODERATE RISK</span>'
    else:
        badge_html = '<span class="risk-badge-high">HIGH RISK</span>'

    st.markdown(
        f"### Overall Risk Rating: {badge_html} &nbsp; "
        f"<small style='color:#6B7280;'>(Composite Score: {comp_score:.2f} / 3.00)</small>",
        unsafe_allow_html=True,
    )

    # 4 Quantitative Risk Metrics
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    with r_col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-card-label">Annualized Return</div>'
            f'<div class="metric-card-value">{ann_ret:+.2%}</div></div>',
            unsafe_allow_html=True,
        )
    with r_col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-card-label">Annualized Volatility</div>'
            f'<div class="metric-card-value">{ann_vol:.2%}</div></div>',
            unsafe_allow_html=True,
        )
    with r_col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-card-label">Sharpe Ratio</div>'
            f'<div class="metric-card-value">{sharpe:.2f}</div></div>',
            unsafe_allow_html=True,
        )
    with r_col4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-card-label">Maximum Drawdown</div>'
            f'<div class="metric-card-value" style="color: #DC2626;">{max_dd:.2%}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Underwater / Drawdown Chart
    st.subheader("Historical Underwater Drawdown Curve")
    st.caption("Illustrates the percentage decline from the historical high-water mark peak over time.")

    drawdown_series = analysis.calculate_drawdown_series(df)
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=drawdown_series.index,
            y=drawdown_series * 100,
            mode="lines",
            name="Drawdown",
            line=dict(color="#DC2626", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(220, 38, 38, 0.2)",
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>",
        )
    )
    fig_dd.add_hline(
        y=max_dd * 100,
        line_dash="dot",
        line_color="#991B1B",
        annotation_text=f"Max Drawdown: {max_dd:.1%}",
        annotation_position="bottom right",
    )
    fig_dd.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=40, t=20, b=30),
        yaxis_title="Drawdown (%)",
        xaxis_title="Date",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Educational Rationale Expander
    with st.expander("🔍 Click to view Detailed Risk Rationale & Methodology", expanded=True):
        st.markdown(assessment["risk_explanation"])
        st.markdown(
            r"""
            ---
            **Scoring Framework Methodology:**
            * **Annualized Volatility (40% Weight):** Measures total variance. Low (<18%), Moderate (18%–28%), High (>28%).
            * **Maximum Drawdown (40% Weight):** Measures worst-case capital loss from peak. Mild (<15%), Moderate (15%–30%), Severe (>30%).
            * **Sharpe Ratio (20% Weight):** Evaluates excess return over sovereign risk-free rate ($R_f=6\%$).
            """
        )


# -----------------------------------------------------------------------------
# TAB 3: DATA & EXPORT
# -----------------------------------------------------------------------------
with tab_data:
    st.subheader(f"Historical Market Quotes for {ticker}")
    st.caption(f"Showing {len(df)} trading sessions sorted chronologically.")

    # Clean display copy
    display_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    if "MA20" in df.columns:
        display_df["MA20"] = df["MA20"]
    if "MA50" in df.columns:
        display_df["MA50"] = df["MA50"]

    # Format numeric columns for presentation
    formatted_df = display_df.copy()
    formatted_df["Volume"] = formatted_df["Volume"].apply(lambda v: f"{int(v):,}")
    for col in ["Open", "High", "Low", "Close"]:
        formatted_df[col] = formatted_df[col].apply(lambda p: f"₹{p:,.2f}")
    if "MA20" in formatted_df.columns:
        formatted_df["MA20"] = formatted_df["MA20"].apply(lambda p: f"₹{p:,.2f}" if pd.notnull(p) else "N/A")
    if "MA50" in formatted_df.columns:
        formatted_df["MA50"] = formatted_df["MA50"].apply(lambda p: f"₹{p:,.2f}" if pd.notnull(p) else "N/A")

    st.dataframe(formatted_df.iloc[::-1], use_container_width=True, height=380)

    # CSV Download Button
    csv_bytes = display_df.to_csv(index=True).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download {ticker} Data (CSV)",
        data=csv_bytes,
        file_name=f"{ticker}_{period_code}_historical_data.csv",
        mime="text/csv",
        help="Export the displayed historical stock prices and moving averages to a CSV file.",
    )


# -----------------------------------------------------------------------------
# 7. REGULATORY & EDUCATIONAL DISCLAIMER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ Financial & Educational Disclaimer:</strong><br>
        This application is for educational and analytical purposes only. Historical performance
        and risk metrics do not guarantee future results. This application does not provide
        investment advice or buy/sell recommendations. Always conduct independent research or
        consult a SEBI-registered financial advisor before making investment decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
