"""
Stage 5 Verification Script: Risk Model
======================================
Tests the transparent multi-factor risk scoring engine and classification.
"""

import data
import risk_model

def main():
    print("--- Testing Stage 5: Risk Model Module (risk_model.py) ---")

    # 1. Test deterministic boundaries
    # Low Risk Case: Vol=12%, MDD=-8%, Sharpe=1.5
    score_low, sub_low = risk_model.calculate_risk_score(0.12, -0.08, 1.5)
    class_low = risk_model.classify_risk(score_low)
    print(f"Low Risk Test Scenario     : Score={score_low}, Class={class_low}")
    assert class_low == "LOW", f"Expected LOW, got {class_low}"

    # Moderate Risk Case: Vol=22%, MDD=-20%, Sharpe=0.6
    score_mod, sub_mod = risk_model.calculate_risk_score(0.22, -0.20, 0.6)
    class_mod = risk_model.classify_risk(score_mod)
    print(f"Moderate Risk Test Scenario: Score={score_mod}, Class={class_mod}")
    assert class_mod == "MODERATE", f"Expected MODERATE, got {class_mod}"

    # High Risk Case: Vol=35%, MDD=-38%, Sharpe=-0.4
    score_high, sub_high = risk_model.calculate_risk_score(0.35, -0.38, -0.4)
    class_high = risk_model.classify_risk(score_high)
    print(f"High Risk Test Scenario    : Score={score_high}, Class={class_high}")
    assert class_high == "HIGH", f"Expected HIGH, got {class_high}"

    # 2. Test live evaluation on TCS.NS
    print("\nFetching TCS.NS data for live risk assessment...")
    df = data.get_stock_data("TCS.NS", "1y")
    assessment = risk_model.evaluate_stock_risk(df)

    print(f"Asset                     : TCS.NS")
    print(f"Annualized Volatility     : {assessment['annualized_volatility']:.2%}")
    print(f"Maximum Drawdown          : {assessment['max_drawdown']:.2%}")
    print(f"Sharpe Ratio              : {assessment['sharpe_ratio']:.2f}")
    print(f"Composite Risk Score      : {assessment['composite_risk_score']} / 3.00")
    print(f"Assigned Risk Level       : {assessment['overall_risk_level']}")
    print("\nGenerated Explanation:\n", assessment["risk_explanation"])

    # 3. Test ML feature extractor stub
    ml_stub = risk_model.MLRiskClassifier()
    feats = ml_stub.extract_features(df)
    print(f"\nML Feature Extractor Shape: {feats.shape} (Columns: {list(feats.columns)})")
    assert not feats.empty, "Feature extraction should produce non-empty DataFrame"

    print("\nSTAGE 5 SUCCESS: Risk model calculations, classification, and explanation verified!")

if __name__ == "__main__":
    main()
