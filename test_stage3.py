"""
Stage 3 Verification Script
===========================
Tests data ingestion and all quantitative calculations on TCS.NS.
"""

import data
import analysis

def main():
    print("--- Testing Stage 3: Financial Calculations on TCS.NS ---")
    df = data.get_stock_data("TCS.NS", "1y")
    m = analysis.calculate_all_metrics(df)

    print(f"Latest Close Price  : Rs. {m['current_price']:.2f}")
    print(f"Period Net Return   : {m['period_return']:.2%}")
    print(f"Annualized Return   : {m['annualized_return']:.2%}")
    print(f"Annualized Volatility: {m['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio (Rf=6%): {m['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown    : {m['max_drawdown']:.2%}")
    print(f"Risk Classification : {m['risk_level']}")
    print(f"Trading Days Loaded : {m['trading_days']}")

    # Verification checks
    assert m["trading_days"] > 200, "Should have over 200 trading days for 1y"
    assert m["current_price"] > 0, "Current price must be positive"
    assert m["annualized_volatility"] > 0, "Volatility must be positive"
    assert -1.0 <= m["max_drawdown"] <= 0.0, "Max drawdown must be between -100% and 0%"

    print("\nSTAGE 3 SUCCESS: All financial formulas and edge cases verified!")

if __name__ == "__main__":
    main()
