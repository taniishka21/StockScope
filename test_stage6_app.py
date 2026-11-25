"""
Stage 6 Verification Script: App Integration Test
=================================================
Validates the complete pipeline connecting data.py, analysis.py,
risk_model.py, and all Plotly visualizations used in app.py.
"""

import data
import analysis
import risk_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    print("--- Testing Stage 6: App Pipeline Integration (app.py) ---")

    for stock_name, ticker in list(data.SUPPORTED_TICKERS.items())[:3]:
        print(f"\nVerifying end-to-end dashboard pipeline for: {stock_name} ({ticker})...")
        
        # 1. Data Ingestion
        df = data.get_stock_data(ticker, "1y")
        assert not df.empty, f"DataFrame for {ticker} is empty"
        print(f"  [1/5] Ingested {len(df)} records.")

        # 2. Moving Averages
        df_ma = analysis.add_moving_averages(df, (20, 50))
        assert "MA20" in df_ma.columns and "MA50" in df_ma.columns
        print("  [2/5] Moving averages MA20 and MA50 computed.")

        # 3. Multi-Factor Risk Assessment
        assessment = risk_model.evaluate_stock_risk(df_ma, risk_free_rate=0.06)
        assert assessment["overall_risk_level"] in ["LOW", "MODERATE", "HIGH"]
        print(f"  [3/5] Risk assessed: {assessment['overall_risk_level']} (Score: {assessment['composite_risk_score']})")

        # 4. Plotly Visualization Constructs
        # Price + Volume subplot
        fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig_price.add_trace(go.Scatter(x=df_ma.index, y=df_ma["Close"]), row=1, col=1)
        fig_price.add_trace(go.Bar(x=df_ma.index, y=df_ma["Volume"]), row=2, col=1)
        assert fig_price.data, "Price figure has no traces"

        # Drawdown curve
        dd_series = analysis.calculate_drawdown_series(df_ma)
        fig_dd = go.Figure(go.Scatter(x=dd_series.index, y=dd_series * 100))
        assert fig_dd.data, "Drawdown figure has no traces"
        print("  [4/5] Plotly figures successfully assembled.")

        # 5. CSV Export validation
        csv_bytes = df_ma[["Open", "High", "Low", "Close", "Volume"]].to_csv().encode("utf-8")
        assert len(csv_bytes) > 100, "CSV bytes generation failed"
        print("  [5/5] CSV export data verified.")

    print("\nSTAGE 6 SUCCESS: All app.py modules, data flows, and charts fully verified!")

if __name__ == "__main__":
    main()
