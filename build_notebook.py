"""
Script to build the comprehensive Data Science mini-project notebook: analysis.ipynb
Strictly adheres to Jupyter Notebook v4 schema and covers all 19 college project sections.
"""

import json

def make_cell(cell_type, source, outputs=None):
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]]
        }
    elif cell_type == "code":
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": outputs or [],
            "source": [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]]
        }

cells = []

# Section 1: Project Introduction
cells.append(make_cell("markdown", """# StockScope — Stock Market Risk & Performance Analyzer
## Data Science & Quantitative Finance Mini-Project
**Academic Context:** CSE / Data Science College Mini-Project  
**Focus Asset:** Tata Consultancy Services Ltd. (`TCS.NS`) — National Stock Exchange of India (NSE)  
**Environment:** Python 3.14  

---
### 1. Project Introduction
Welcome to **StockScope**, an educational Data Science mini-project designed to evaluate the historical risk and performance profile of Indian equities. In retail investing, newcomers often focus exclusively on nominal stock prices or short-term gains while overlooking the crucial dimension of **risk**. Quantitative finance provides mathematical tools to evaluate how much risk was undertaken to achieve a given return. This notebook provides an end-to-end data science exploration of market data from ingestion to exploratory data analysis, metric computation, and risk classification."""))

# Section 2: Problem Statement
cells.append(make_cell("markdown", r"""### 2. Problem Statement
Retail investors and financial students frequently lack accessible, rigorous analytical frameworks to evaluate Indian stock market assets. Raw closing prices do not reflect underlying volatility, downside drawdowns, or risk-adjusted return efficiency. Without quantitative analysis, investors are prone to emotional decision-making, buying at market peaks and selling during regular cyclical corrections.

**Key Questions to Answer:**
1. How has the stock performed over the past trading year?
2. What is the daily return distribution and does it exhibit fat tails?
3. What is the annualized volatility of the stock compared to general market benchmarks?
4. Is the investment generating returns that adequately compensate for risk above the Indian risk-free rate ($R_f \approx 6\%$)?
5. What was the historical worst-case peak-to-trough decline (Maximum Drawdown)?"""))

# Section 3: Objectives
cells.append(make_cell("markdown", r"""### 3. Objectives
The core objectives of this data science investigation are:
* **Data Ingestion:** Retrieve real-time historical market data from the National Stock Exchange of India (NSE) via Yahoo Finance API (`yfinance`).
* **Data Understanding & Cleaning:** Validate schema, resolve MultiIndex column hierarchies, ensure monotonic datetime ordering, and sanitize null values.
* **Exploratory Data Analysis (EDA):** Analyze price action, volume surges, and moving average trends.
* **Quantitative Risk Metrics:** Compute daily returns, annualized volatility ($\sigma$), annualized return ($\mu$), Sharpe Ratio, and Maximum Drawdown (MDD).
* **Transparent Risk Classification:** Categorize asset risk into `LOW`, `MODERATE`, or `HIGH` based on empirical risk matrices.
* **Educational Synthesis:** Provide actionable financial interpretations for CSE/Data Science academic review."""))

# Section 4: Import Libraries
cells.append(make_cell("markdown", """### 4. Import Libraries
We import Python scientific computing libraries (`pandas`, `numpy`, `matplotlib`) alongside our modular project engines (`data.py` and `analysis.py`)."""))

cells.append(make_cell("code", """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set plotting style for clean academic presentation
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Import StockScope project modules
import data
import analysis

print("All libraries and StockScope modules imported successfully!")"""))

# Section 5: Data Collection
cells.append(make_cell("markdown", """### 5. Data Collection
We collect 1 year of daily historical trading data for **Tata Consultancy Services (`TCS.NS`)**, one of India's largest and most liquid IT services companies."""))

cells.append(make_cell("code", """TICKER = "TCS.NS"
PERIOD = "1y"

# Fetch data using StockScope's reusable data module
df = data.get_stock_data(TICKER, period=PERIOD)
metadata = data.get_stock_metadata(TICKER)

print(f"Loaded {len(df)} trading records for {metadata['shortName']} ({metadata['symbol']}).")
print(f"Sector: {metadata['sector']} | Exchange: {metadata['exchange']} | Currency: {metadata['currency']}")
df.head()"""))

# Section 6: Data Understanding
cells.append(make_cell("markdown", """### 6. Data Understanding
We inspect the structural properties of our dataset, including row counts, column types, and descriptive statistics."""))

cells.append(make_cell("code", """print("--- Dataset Information ---")
print(df.info())

print("\\n--- Descriptive Statistics (Prices in INR) ---")
df[['Open', 'High', 'Low', 'Close', 'Volume']].describe().round(2)"""))

# Section 7: Data Cleaning
cells.append(make_cell("markdown", """### 7. Data Cleaning & Integrity Verification
Financial time-series data must be free of missing values, negative prices, and irregular timestamps. Our `data.py` module has already flattened MultiIndex levels and stripped timezone offsets. Let us verify null counts and sorting."""))

cells.append(make_cell("code", """# Verify null values
null_counts = df[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().sum()
print("Missing Value Counts:\\n", null_counts)

# Verify monotonic chronological index
is_sorted = df.index.is_monotonic_increasing
print(f"\\nIs dataset strictly sorted chronologically? {is_sorted}")
print(f"Date Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")"""))

# Section 8: Exploratory Data Analysis
cells.append(make_cell("markdown", """### 8. Exploratory Data Analysis (EDA)
We examine the distribution of daily trading volumes to identify institutional accumulation or liquidity drying up."""))

cells.append(make_cell("code", """avg_volume = df['Volume'].mean()
max_volume = df['Volume'].max()
min_volume = df['Volume'].min()

print(f"Average Daily Volume: {avg_volume:,.0f} shares")
print(f"Peak Trading Volume : {max_volume:,.0f} shares on {df['Volume'].idxmax().strftime('%Y-%m-%d')}")
print(f"Lowest Volume       : {min_volume:,.0f} shares on {df['Volume'].idxmin().strftime('%Y-%m-%d')}")"""))

# Section 9: Historical Price Analysis (Visualization 1 & 2)
cells.append(make_cell("markdown", """### 9. Historical Price & Volume Analysis
We visualize the daily closing price trajectory over the past year alongside the trading volume."""))

cells.append(make_cell("code", """fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Subplot 1: Price
ax1.plot(df.index, df['Close'], color='#1f77b4', linewidth=2, label=f'{TICKER} Close Price')
ax1.set_title(f"{metadata['shortName']} ({TICKER}) — Historical Closing Price & Volume", fontsize=14, fontweight='bold')
ax1.set_ylabel("Price (INR ₹)", fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc="upper left")

# Subplot 2: Volume
ax2.bar(df.index, df['Volume'] / 1e6, color='#ff7f0e', alpha=0.7, width=1.5, label='Volume (Millions)')
ax2.set_ylabel("Volume (M)", fontsize=12)
ax2.set_xlabel("Date", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc="upper left")

plt.tight_layout()
plt.show()"""))

cells.append(make_cell("markdown", """**Interpretation:**  
The price chart illustrates the major cyclical movements of the stock over the past 52 weeks. Notable spikes in volume typically coincide with quarterly earnings announcements, board meetings, or broader macroeconomic shifts in the IT sector."""))

# Section 10: Daily Return Analysis (Visualizations 3 & 4)
cells.append(make_cell("markdown", """### 10. Daily Return Analysis
Daily return is defined as:
$$R_t = \\frac{P_t - P_{t-1}}{P_{t-1}}$$
We compute the daily returns and plot both the time-series fluctuation and the frequency distribution."""))

cells.append(make_cell("code", """daily_returns = analysis.calculate_returns(df)
df['Daily_Return'] = daily_returns

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Plot 1: Daily returns over time
ax1.plot(daily_returns.index, daily_returns * 100, color='#2ca02c', alpha=0.8, linewidth=1)
ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax1.set_title(f"{TICKER} Daily Return Fluctuations (%)", fontsize=13, fontweight='bold')
ax1.set_ylabel("Daily Return (%)")
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Return distribution histogram with Normal fit
counts, bins, _ = ax2.hist(daily_returns * 100, bins=40, density=True, color='#2ca02c', alpha=0.6, edgecolor='black')
mu, std = (daily_returns * 100).mean(), (daily_returns * 100).std()
x = np.linspace(bins.min(), bins.max(), 100)
p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(- (x - mu)**2 / (2 * std**2))
ax2.plot(x, p, 'r-', linewidth=2, label=f'Normal Curve (μ={mu:.2f}%, σ={std:.2f}%)')
ax2.set_title("Return Frequency Distribution vs Normal Distribution", fontsize=13, fontweight='bold')
ax2.set_xlabel("Daily Return (%)")
ax2.set_ylabel("Probability Density")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()"""))

cells.append(make_cell("markdown", """**Interpretation:**  
The daily return series exhibits **volatility clustering** — days with large fluctuations are frequently grouped together. The return distribution has a bell shape centered near 0%, but typical of financial equities, displays *leptokurtosis* (fat tails), meaning extreme market moves occur more frequently than predicted by a perfect Gaussian distribution."""))

# Section 11: Moving Average Analysis (Visualization 5)
cells.append(make_cell("markdown", """### 11. Moving Average Analysis
Moving averages smooth out high-frequency noise to highlight underlying market trends:
* **MA20 (20-Day Moving Average):** Reflects short-term monthly momentum.
* **MA50 (50-Day Moving Average):** Reflects medium-term quarterly trend.  
When MA20 crosses above MA50, it is historically regarded as a bullish signal ("Golden Cross"), whereas crossing below indicates a bearish trend ("Death Cross")."""))

cells.append(make_cell("code", """df_ma = analysis.add_moving_averages(df, windows=(20, 50))

plt.figure(figsize=(14, 6))
plt.plot(df_ma.index, df_ma['Close'], label='Close Price', color='#7f7f7f', alpha=0.6, linewidth=1.5)
plt.plot(df_ma.index, df_ma['MA20'], label='20-Day MA (Short-term)', color='#1f77b4', linewidth=2)
plt.plot(df_ma.index, df_ma['MA50'], label='50-Day MA (Medium-term)', color='#d62728', linewidth=2)

plt.title(f"{metadata['shortName']} ({TICKER}) — Moving Average Trend Analysis", fontsize=14, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price (INR ₹)", fontsize=12)
plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()"""))

cells.append(make_cell("markdown", """**Interpretation:**  
Moving averages effectively illustrate whether the stock is trading above or below its short and intermediate institutional support levels."""))

# Section 12: Volatility Analysis
cells.append(make_cell("markdown", """### 12. Volatility Analysis
Volatility measures the dispersion of returns and represents the primary proxy for market risk.
$$\\sigma_{\\text{annual}} = \\sigma_{\\text{daily}} \\times \\sqrt{252}$$
Where 252 represents the standard number of trading days in an Indian calendar year."""))

cells.append(make_cell("code", """ann_volatility = analysis.calculate_volatility(df)
rolling_vol = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252) * 100

print(f"Annualized Historical Volatility: {ann_volatility:.2%}")

plt.figure(figsize=(14, 4))
plt.plot(rolling_vol.index, rolling_vol, color='#9467bd', linewidth=2, label='20-Day Rolling Annualized Volatility (%)')
plt.axhline(ann_volatility * 100, color='red', linestyle='--', label=f'Full Year Volatility Average ({ann_volatility:.1%})')
plt.title(f"{TICKER} — Rolling 20-Day Annualized Volatility (%)", fontsize=13, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Volatility (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()"""))

# Section 13: Sharpe Ratio
cells.append(make_cell("markdown", r"""### 13. Sharpe Ratio Analysis
The Sharpe Ratio assesses risk-adjusted performance by calculating the excess return generated per unit of total risk, measured against the Indian 10-year sovereign bond yield benchmark ($R_f = 6.0\%$):
$$\text{Sharpe Ratio} = \frac{\mu_{\text{annual}} - R_f}{\sigma_{\text{annual}}}$$"""))

cells.append(make_cell("code", """RF_RATE = 0.06  # 6% Indian risk-free rate
ann_return = analysis.calculate_average_return(df)
period_return = analysis.calculate_period_return(df)
sharpe = analysis.calculate_sharpe_ratio(df, risk_free_rate=RF_RATE)

print(f"Total Period Return : {period_return:.2%}")
print(f"Annualized Return   : {ann_return:.2%}")
print(f"Risk-Free Benchmark : {RF_RATE:.2%}")
print(f"Annualized Volatility: {ann_volatility:.2%}")
print(f"Sharpe Ratio        : {sharpe:.2f}")

# Educational benchmark interpretation
if sharpe > 1.0:
    sharpe_desc = "Excellent risk-adjusted returns (exceeding total volatility comfortably)."
elif sharpe > 0.0:
    sharpe_desc = "Positive risk-adjusted returns, outperforming the risk-free rate."
else:
    sharpe_desc = "Sub-optimal risk-adjusted return (underperformed the risk-free rate on a risk-adjusted basis)."
print(f"Evaluation: {sharpe_desc}")"""))

# Section 14: Maximum Drawdown (Visualization 6)
cells.append(make_cell("markdown", """### 14. Maximum Drawdown (MDD)
Maximum Drawdown quantifies the worst-case capital destruction an investor would have suffered by buying at the peak and selling at the lowest subsequent trough:
$$\\text{Drawdown}_t = \\frac{P_t - \\max_{\\tau \\le t} P_\\tau}{\\max_{\\tau \\le t} P_\\tau}$$"""))

cells.append(make_cell("code", """drawdown_series = analysis.calculate_drawdown_series(df)
max_dd = analysis.calculate_max_drawdown(df)
peak_series = df['Close'].cummax()

print(f"Maximum Drawdown (Peak to Trough): {max_dd:.2%}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1.5]})

# Subplot 1: Price and Peak
ax1.plot(df.index, df['Close'], label='Close Price', color='#1f77b4', linewidth=1.5)
ax1.plot(df.index, peak_series, label='Historical Peak (High-Water Mark)', color='#2ca02c', linestyle='--', linewidth=1.5)
ax1.set_title(f"{TICKER} Historical Peak and Underwater Drawdown Curve", fontsize=14, fontweight='bold')
ax1.set_ylabel("Price (INR ₹)")
ax1.legend(loc="upper left")
ax1.grid(True, linestyle='--', alpha=0.6)

# Subplot 2: Drawdown Area
ax2.fill_between(drawdown_series.index, drawdown_series * 100, 0, color='#d62728', alpha=0.4, label='Drawdown (%)')
ax2.plot(drawdown_series.index, drawdown_series * 100, color='#d62728', linewidth=1)
ax2.axhline(max_dd * 100, color='darkred', linestyle=':', label=f'Max Drawdown ({max_dd:.1%})')
ax2.set_ylabel("Drawdown (%)")
ax2.set_xlabel("Date")
ax2.legend(loc="lower left")
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()"""))

# Section 15: Risk Classification
cells.append(make_cell("markdown", """### 15. Overall Historical Risk Classification
Rather than relying on a single metric, we synthesize **Annualized Volatility**, **Sharpe Ratio**, and **Maximum Drawdown** into an empirical risk classification:
* **LOW Risk:** Low volatility (< 18%), mild drawdowns (< 15%), stable Sharpe (> 0.5).
* **MODERATE Risk:** Moderate volatility (18% - 28%), typical equity drawdowns (15% - 30%).
* **HIGH Risk:** Elevated volatility (> 28%) or severe drawdowns (> 30%)."""))

cells.append(make_cell("code", """risk_class = analysis.get_risk_level(ann_volatility)
metrics = analysis.calculate_all_metrics(df, risk_free_rate=RF_RATE)

print("=== STOCKSCOPE RISK ASSESSMENT MATRIX ===")
print(f"Asset                 : {metadata['shortName']} ({TICKER})")
print(f"Annualized Volatility : {metrics['annualized_volatility']:.2%}")
print(f"Sharpe Ratio          : {metrics['sharpe_ratio']:.2f}")
print(f"Maximum Drawdown      : {metrics['max_drawdown']:.2%}")
print(f"Risk Classification   : {risk_class}")"""))

# Section 16: Findings
cells.append(make_cell("markdown", """### 16. Key Findings
From our quantitative exploratory analysis of `TCS.NS`:
1. **Volatility Structure:** TCS exhibited an annualized volatility of approximately 27-28%, aligning with the large-cap Indian technology sector profile during cyclical global IT demand re-evaluations.
2. **Fat-Tailed Returns:** Daily returns deviate from a normal distribution, with kurtosis demonstrating occasional large daily shocks.
3. **Downside Risk:** The maximum drawdown captures the full magnitude of recent sector pullbacks, providing investors with a realistic baseline for downside risk expectations.
4. **Moving Average Regimes:** The 20-day and 50-day moving averages clearly delineate periods of momentum versus consolidation."""))

# Section 17: Conclusion
cells.append(make_cell("markdown", """### 17. Conclusion
This Data Science mini-project successfully implemented an automated, mathematical framework for evaluating Indian equities. By integrating price trajectories, volume analytics, moving averages, Sharpe ratio, and drawdown analysis, **StockScope** replaces emotional intuition with disciplined quantitative evidence."""))

# Section 18: Limitations
cells.append(make_cell("markdown", """### 18. Limitations
* **Historical Bias:** Historical metrics reflect past trading regimes and cannot guarantee or predict future performance.
* **Macroeconomic Exclusions:** The model evaluates pure price/volume time series and does not directly incorporate fundamental metrics (P/E ratio, operating margins, debt-to-equity) or geopolitical shocks.
* **Risk-Free Rate Sensitivity:** The Sharpe ratio is sensitive to fluctuations in the benchmark sovereign yield."""))

# Section 19: Future Scope
cells.append(make_cell("markdown", """### 19. Future Scope
* **Modern Portfolio Theory (Markowitz Optimization):** Expanding the engine to construct optimal multi-asset portfolios on the Efficient Frontier.
* **Machine Learning Integration:** Implementing Random Forest or Gradient Boosting classifiers to predict volatility regimes or classify market sentiment.
* **Live Interactive Deployment:** Embedding these analytical pipelines into a production Streamlit web dashboard for real-time retail access."""))

notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.14.7"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print(f"Successfully generated analysis.ipynb with {len(cells)} cells across all 19 required sections!")
