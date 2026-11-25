"""
StockScope - Data Ingestion Module (data.py)
============================================
Handles fetching, cleaning, and validating historical market data
for Indian stocks (NSE) using yfinance.
"""

from typing import Dict, Optional
import pandas as pd
import yfinance as yf

# Standard curated Indian stocks list (NSE tickers)
SUPPORTED_TICKERS: Dict[str, str] = {
    "TCS": "TCS.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
}

# User-friendly period display to yfinance period parameter mapping
PERIOD_MAPPING: Dict[str, str] = {
    "1 Month": "1mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
}

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def normalize_period(period: str) -> str:
    """
    Normalizes a human-readable period label to a yfinance-compatible period string.

    Parameters:
        period (str): Friendly label (e.g. '1 Year') or yfinance string (e.g. '1y').

    Returns:
        str: yfinance-compatible period string.
    """
    return PERIOD_MAPPING.get(period, period)


def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Retrieves historical price and volume data for a given ticker from Yahoo Finance.

    Parameters:
        ticker (str): NSE ticker symbol (e.g. 'TCS.NS', 'RELIANCE.NS').
        period (str): Timeframe period string ('1mo', '6mo', '1y', '2y', '5y')
                      or friendly display label ('1 Month', '1 Year', etc.).

    Returns:
        pd.DataFrame: Cleaned DataFrame with DatetimeIndex and columns:
                      ['Open', 'High', 'Low', 'Close', 'Volume'] (and 'Adj Close' if present).

    Raises:
        ValueError: If the ticker is invalid, the dataset is empty,
                    or required columns are missing.
        RuntimeError: If network or API communication fails.
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError("A valid non-empty ticker symbol string must be provided.")

    ticker_clean = ticker.strip().upper()
    yf_period = normalize_period(period)

    try:
        ticker_obj = yf.Ticker(ticker_clean)
        df: pd.DataFrame = ticker_obj.history(period=yf_period, auto_adjust=False)

        # Fallback to yf.download if history returned empty
        if df is None or df.empty:
            df = yf.download(ticker_clean, period=yf_period, progress=False, auto_adjust=False)

    except Exception as exc:
        raise RuntimeError(
            f"Unable to retrieve stock data for '{ticker_clean}'. "
            "Please check your internet connection and try again."
        ) from exc

    if df is None or df.empty:
        raise ValueError(
            f"No market data found for ticker '{ticker_clean}' over period '{period}'. "
            "Verify the ticker symbol or try a different time period."
        )

    # Clean and flatten yfinance MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        first_level = [str(c) for c in df.columns.get_level_values(0)]
        if any(col in REQUIRED_COLUMNS for col in first_level):
            df.columns = pd.Index([c[0] for c in df.columns], name=None)
        else:
            df.columns = pd.Index([c[-1] for c in df.columns], name=None)

    # Standardize column naming
    df.columns = [str(col).strip().title() for col in df.columns]

    # Validate that all required price and volume columns are present
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Data retrieved for '{ticker_clean}' is missing required columns: {missing_cols}"
        )

    # Ensure index is datetime and strip timezone info to prevent comparison/plotting issues
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Drop any records with missing essential price or volume data
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    if df.empty:
        raise ValueError(
            f"Historical dataset for '{ticker_clean}' contains only missing values after cleaning."
        )

    # Ensure chronological ascending sort order
    df = df.sort_index(ascending=True)

    # Ensure proper numeric typing
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)

    return df


def get_stock_metadata(ticker: str) -> Dict[str, Optional[str]]:
    """
    Safely retrieves descriptive metadata for a given stock ticker.

    Parameters:
        ticker (str): Stock ticker symbol.

    Returns:
        dict: Basic company metadata including name, sector, currency, and exchange.
    """
    default_info = {
        "symbol": ticker,
        "shortName": ticker,
        "sector": "N/A",
        "currency": "INR",
        "exchange": "NSE",
    }
    try:
        ticker_obj = yf.Ticker(ticker)
        fast_info = getattr(ticker_obj, "fast_info", None)
        info = getattr(ticker_obj, "info", {})

        short_name = (
            info.get("shortName")
            or info.get("longName")
            or getattr(fast_info, "currency", None)
            or ticker
        )
        sector = info.get("sector", "Indian Equities")
        currency = getattr(fast_info, "currency", info.get("currency", "INR"))
        exchange = getattr(fast_info, "exchange", info.get("exchange", "NSE"))

        return {
            "symbol": ticker,
            "shortName": short_name,
            "sector": sector,
            "currency": currency or "INR",
            "exchange": exchange or "NSE",
        }
    except Exception:
        return default_info
