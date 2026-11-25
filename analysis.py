"""
StockScope - Financial Analysis Module (analysis.py)
===================================================
Provides quantitative financial calculations for analyzing historical
performance and risk metrics of stocks.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

# Benchmark constants
TRADING_DAYS_PER_YEAR: int = 252
DEFAULT_RISK_FREE_RATE: float = 0.06  # 6.0% Indian benchmark risk-free rate (approx. RBI repo/G-Sec yield)


def calculate_returns(data: pd.DataFrame) -> pd.Series:
    """
    Computes daily percentage returns based on the closing prices.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.

    Returns:
        pd.Series: Daily return series with initial NaN dropped.
    """
    if data is None or data.empty or "Close" not in data.columns:
        return pd.Series(dtype=float, name="Daily_Return")

    close = pd.to_numeric(data["Close"], errors="coerce")
    daily_returns = close.pct_change().dropna()
    daily_returns.name = "Daily_Return"
    return daily_returns


def calculate_period_return(data: pd.DataFrame) -> float:
    """
    Calculates the total percentage return over the entire dataset duration.

    Formula:
        (Final Close - Initial Close) / Initial Close

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.

    Returns:
        float: Total period return as a decimal (e.g. 0.15 for +15%).
    """
    if data is None or len(data) < 2 or "Close" not in data.columns:
        return 0.0

    initial_price = float(data["Close"].iloc[0])
    final_price = float(data["Close"].iloc[-1])

    if initial_price <= 0 or np.isnan(initial_price) or np.isnan(final_price):
        return 0.0

    return float((final_price - initial_price) / initial_price)


def calculate_average_return(data: pd.DataFrame) -> float:
    """
    Calculates the annualized average return from daily returns.

    Formula:
        Annualized Return = Mean(Daily Returns) * 252

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.

    Returns:
        float: Annualized average return.
    """
    returns = calculate_returns(data)
    if returns.empty or len(returns) < 2:
        return 0.0

    mean_daily = float(returns.mean())
    if np.isnan(mean_daily):
        return 0.0

    return float(mean_daily * TRADING_DAYS_PER_YEAR)


def calculate_volatility(data: pd.DataFrame) -> float:
    """
    Calculates the annualized historical volatility (standard deviation of returns).

    Formula:
        Annualized Volatility = Std(Daily Returns) * sqrt(252)

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.

    Returns:
        float: Annualized volatility as a decimal (e.g. 0.22 for 22%).
    """
    returns = calculate_returns(data)
    if returns.empty or len(returns) < 2:
        return 0.0

    daily_std = float(returns.std(ddof=1))
    if np.isnan(daily_std) or daily_std <= 0:
        return 0.0

    return float(daily_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def calculate_sharpe_ratio(
    data: pd.DataFrame, risk_free_rate: float = DEFAULT_RISK_FREE_RATE
) -> float:
    """
    Calculates the annualized Sharpe Ratio.

    Formula:
        Sharpe Ratio = (Annualized Return - Risk Free Rate) / Annualized Volatility

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        risk_free_rate (float): Annual risk-free interest rate (default: 0.06).

    Returns:
        float: Sharpe ratio. Returns 0.0 if volatility is zero or data is insufficient.
    """
    annual_return = calculate_average_return(data)
    volatility = calculate_volatility(data)

    if volatility <= 1e-8 or np.isnan(volatility) or np.isnan(annual_return):
        return 0.0

    excess_return = annual_return - risk_free_rate
    sharpe = excess_return / volatility

    if np.isnan(sharpe) or np.isinf(sharpe):
        return 0.0

    return float(sharpe)


def calculate_drawdown_series(data: pd.DataFrame) -> pd.Series:
    """
    Calculates the percentage drawdown series from historical peak prices.

    Formula:
        Drawdown_t = (Price_t - Peak_t) / Peak_t

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.

    Returns:
        pd.Series: Drawdown values over time as negative decimals (e.g. -0.10 for -10%).
    """
    if data is None or data.empty or "Close" not in data.columns:
        return pd.Series(dtype=float, name="Drawdown")

    close = pd.to_numeric(data["Close"], errors="coerce")
    rolling_max = close.cummax()

    # Prevent division by zero or negative peaks
    valid_mask = rolling_max > 0
    drawdown = pd.Series(0.0, index=close.index, name="Drawdown")
    drawdown.loc[valid_mask] = (close.loc[valid_mask] - rolling_max.loc[valid_mask]) / rolling_max.loc[valid_mask]
    return drawdown


def calculate_max_drawdown(data: pd.DataFrame) -> float:
    """
    Calculates the maximum drawdown (worst peak-to-trough drop).

    Formula:
        Max Drawdown = Min( (Price_t - Peak_t) / Peak_t )

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.

    Returns:
        float: Maximum drawdown as a negative decimal (e.g. -0.25 for -25%).
    """
    drawdown = calculate_drawdown_series(data)
    if drawdown.empty:
        return 0.0

    mdd = float(drawdown.min())
    if np.isnan(mdd):
        return 0.0

    return mdd


def add_moving_averages(
    data: pd.DataFrame, windows: Tuple[int, int] = (20, 50)
) -> pd.DataFrame:
    """
    Appends moving average columns (MA20 and MA50) to the DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        windows (Tuple[int, int]): Window lengths (default: 20 and 50 periods).

    Returns:
        pd.DataFrame: New DataFrame copy with 'MA20' and 'MA50' columns added.
    """
    if data is None or data.empty or "Close" not in data.columns:
        return data.copy() if data is not None else pd.DataFrame()

    df = data.copy()
    close = pd.to_numeric(df["Close"], errors="coerce")

    w1, w2 = windows
    df[f"MA{w1}"] = close.rolling(window=w1, min_periods=1).mean()
    df[f"MA{w2}"] = close.rolling(window=w2, min_periods=1).mean()

    return df


def calculate_beta(
    data: pd.DataFrame, market_data: pd.DataFrame
) -> Optional[float]:
    """
    Calculates the stock's Beta relative to a benchmark market index (e.g. Nifty 50).

    Formula:
        Beta = Cov(R_stock, R_market) / Var(R_market)

    Parameters:
        data (pd.DataFrame): Stock DataFrame with 'Close' column.
        market_data (pd.DataFrame): Benchmark market DataFrame with 'Close' column.

    Returns:
        Optional[float]: Beta coefficient, or None if insufficient overlap.
    """
    if data is None or market_data is None or data.empty or market_data.empty:
        return None

    stock_returns = calculate_returns(data)
    market_returns = calculate_returns(market_data)

    aligned = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    aligned.columns = ["Stock", "Market"]

    if len(aligned) < 10:
        return None

    market_var = float(aligned["Market"].var(ddof=1))
    if market_var <= 1e-9 or np.isnan(market_var):
        return None

    covariance = float(aligned["Stock"].cov(aligned["Market"]))
    beta = covariance / market_var

    if np.isnan(beta) or np.isinf(beta):
        return None

    return float(beta)


def get_risk_level(volatility: float) -> str:
    """
    Classifies risk based on annualized volatility thresholds calibrated for Indian equities.

    Thresholds:
        - Volatility < 18%: LOW
        - Volatility 18% to 28%: MODERATE
        - Volatility > 28%: HIGH

    Parameters:
        volatility (float): Annualized volatility as a decimal (e.g. 0.20 for 20%).

    Returns:
        str: 'LOW', 'MODERATE', or 'HIGH'.
    """
    if np.isnan(volatility) or volatility < 0.18:
        return "LOW"
    elif volatility <= 0.28:
        return "MODERATE"
    else:
        return "HIGH"


def calculate_all_metrics(
    data: pd.DataFrame, risk_free_rate: float = DEFAULT_RISK_FREE_RATE
) -> Dict[str, Any]:
    """
    Convenience function that calculates all key financial and risk metrics.

    Parameters:
        data (pd.DataFrame): Cleaned stock DataFrame.
        risk_free_rate (float): Benchmark annual risk-free rate.

    Returns:
        dict: Consolidated dictionary of all computed metrics.
    """
    if data is None or data.empty or "Close" not in data.columns:
        return {
            "current_price": 0.0,
            "period_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "risk_level": "UNKNOWN",
            "trading_days": 0,
        }

    current_price = float(data["Close"].iloc[-1])
    period_return = calculate_period_return(data)
    annualized_return = calculate_average_return(data)
    volatility = calculate_volatility(data)
    sharpe = calculate_sharpe_ratio(data, risk_free_rate=risk_free_rate)
    max_dd = calculate_max_drawdown(data)
    risk_level = get_risk_level(volatility)

    return {
        "current_price": current_price,
        "period_return": period_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "risk_level": risk_level,
        "trading_days": len(data),
    }
