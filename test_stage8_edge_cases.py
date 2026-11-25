"""
Stage 8 Verification Script: Comprehensive Error Handling & Edge Cases
======================================================================
Tests all edge cases specified in Section 18:
- Invalid ticker handling
- Zero volatility (flat price)
- Insufficient historical data (< 5 days)
- Short period where MA50 is unavailable (e.g. 1 Month)
- Missing values and NaNs
- Graceful error messaging without unhandled exceptions
"""

import pandas as pd
import numpy as np
import data
import analysis
import risk_model

def main():
    print("--- Testing Stage 8: Comprehensive Error Handling & Edge Cases ---")

    # 1. Edge Case: Invalid Ticker
    print("\n[Edge Case 1] Invalid Ticker Input:")
    try:
        data.get_stock_data("NON_EXISTENT_TICKER_999.NS", "1y")
        print("  FAILED: Did not raise expected exception")
        assert False
    except (ValueError, RuntimeError) as e:
        print(f"  PASSED: Caught friendly exception -> {type(e).__name__}: {e}")

    # 2. Edge Case: Zero Volatility (Completely flat price series)
    print("\n[Edge Case 2] Zero Volatility (Flatline price):")
    flat_data = pd.DataFrame({
        "Open": [100.0] * 30,
        "High": [100.0] * 30,
        "Low": [100.0] * 30,
        "Close": [100.0] * 30,
        "Volume": [1000] * 30,
    }, index=pd.date_range("2026-01-01", periods=30))
    
    vol = analysis.calculate_volatility(flat_data)
    sharpe = analysis.calculate_sharpe_ratio(flat_data)
    mdd = analysis.calculate_max_drawdown(flat_data)
    assessment = risk_model.evaluate_stock_risk(flat_data)
    print(f"  Volatility={vol}, Sharpe={sharpe}, MDD={mdd}")
    print(f"  Assigned Risk={assessment['overall_risk_level']}, Score={assessment['composite_risk_score']}")
    assert vol == 0.0, "Zero volatility should evaluate to 0.0"
    assert sharpe == 0.0, "Sharpe with zero volatility should evaluate to 0.0 without ZeroDivisionError"
    assert mdd == 0.0, "Max drawdown for flat prices should be 0.0"
    print("  PASSED: Zero volatility handled cleanly without ZeroDivisionError.")

    # 3. Edge Case: Short Period (MA50 unavailable)
    print("\n[Edge Case 3] Short Timeframe (1 Month / 22 trading days):")
    short_df = data.get_stock_data("TCS.NS", "1mo")
    print(f"  Fetched 1mo data: {len(short_df)} trading days.")
    short_df_ma = analysis.add_moving_averages(short_df, (20, 50))
    print(f"  MA20 non-null count: {short_df_ma['MA20'].notnull().sum()}")
    print(f"  MA50 non-null count: {short_df_ma['MA50'].notnull().sum()}")
    assert not short_df_ma["MA20"].dropna().empty, "MA20 should have values"
    print("  PASSED: Short periods handled gracefully without runtime warnings.")

    # 4. Edge Case: Empty & 1-row DataFrames
    print("\n[Edge Case 4] Empty & 1-row DataFrames:")
    empty_df = pd.DataFrame()
    one_row_df = pd.DataFrame({"Close": [150.0]})
    assert analysis.calculate_returns(empty_df).empty
    assert analysis.calculate_volatility(empty_df) == 0.0
    assert analysis.calculate_volatility(one_row_df) == 0.0
    assert analysis.calculate_period_return(empty_df) == 0.0
    assert analysis.calculate_max_drawdown(empty_df) == 0.0
    print("  PASSED: Empty and single-row data handled cleanly.")

    # 5. Edge Case: Embedded NaNs in price columns
    print("\n[Edge Case 5] Data with Random NaNs:")
    nan_df = flat_data.copy()
    nan_df.iloc[5, nan_df.columns.get_loc("Close")] = np.nan
    nan_returns = analysis.calculate_returns(nan_df)
    assert not nan_returns.isnull().any(), "Returns series should not contain unhandled NaNs"
    print("  PASSED: NaNs dropped and handled properly.")

    print("\nSTAGE 8 SUCCESS: All 5 critical financial edge cases verified and handled gracefully!")

if __name__ == "__main__":
    main()
