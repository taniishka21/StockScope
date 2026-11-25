"""
StockScope - Risk Model Module (risk_model.py)
==============================================
Provides a transparent, multi-factor quantitative risk scoring engine
based on historical metrics: Annualized Volatility, Maximum Drawdown,
and Sharpe Ratio.

NOTE: This is a deterministic rule-based quantitative scoring framework.
It is NOT a black-box machine learning model and does not fabricate predictions.
The architecture provides an MLRiskClassifier interface prepared for future
Random Forest model extension.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import analysis


# Risk Threshold Benchmarks calibrated for Indian Equities (NSE)
VOLATILITY_THRESHOLDS = {
    "low": 0.18,       # Below 18% annualized volatility -> Low Volatility
    "moderate": 0.28,  # 18% to 28% -> Moderate Volatility, Above 28% -> High Volatility
}

DRAWDOWN_THRESHOLDS = {
    "low": 0.15,       # Under 15% peak-to-trough decline -> Mild Drawdown
    "moderate": 0.30,  # 15% to 30% -> Moderate Drawdown, Over 30% -> Severe Drawdown
}

SHARPE_THRESHOLDS = {
    "high_quality": 1.0,  # Sharpe > 1.0 -> Excellent risk-adjusted returns
    "moderate_quality": 0.0,  # 0.0 <= Sharpe <= 1.0 -> Outperforming risk-free benchmark
}

# Metric Weights in Composite Score
WEIGHTS = {
    "volatility": 0.40,
    "max_drawdown": 0.40,
    "sharpe_ratio": 0.20,
}


def calculate_risk_score(
    volatility: float, max_drawdown: float, sharpe_ratio: float
) -> Tuple[float, Dict[str, Any]]:
    """
    Computes a composite quantitative risk score ranging from 1.0 (Low Risk)
    to 3.0 (High Risk) based on three fundamental risk pillars.

    Pillar Scoring (1 = Low Risk, 2 = Moderate Risk, 3 = High Risk):
    1. Volatility:
       - < 18% : 1
       - 18% - 28% : 2
       - > 28% : 3
    2. Maximum Drawdown (absolute drop):
       - < 15% : 1
       - 15% - 30% : 2
       - > 30% : 3
    3. Sharpe Ratio:
       - > 1.0 : 1 (efficient risk-adjusted return)
       - 0.0 - 1.0 : 2 (moderate risk-adjusted return)
       - < 0.0 : 3 (negative excess return over risk-free rate)

    Parameters:
        volatility (float): Annualized volatility as a decimal (e.g. 0.22 for 22%).
        max_drawdown (float): Maximum drawdown as a negative decimal (e.g. -0.25).
        sharpe_ratio (float): Annualized Sharpe ratio.

    Returns:
        Tuple[float, Dict[str, Any]]:
            - composite_score (float): Value between 1.0 and 3.0
            - sub_scores (dict): Individual pillar ratings and normalized scores
    """
    # 1. Volatility sub-score
    if np.isnan(volatility) or volatility < VOLATILITY_THRESHOLDS["low"]:
        vol_score = 1.0
        vol_label = "Low"
    elif volatility <= VOLATILITY_THRESHOLDS["moderate"]:
        vol_score = 2.0
        vol_label = "Moderate"
    else:
        vol_score = 3.0
        vol_label = "High"

    # 2. Maximum Drawdown sub-score (evaluate absolute magnitude)
    abs_dd = abs(max_drawdown) if not np.isnan(max_drawdown) else 0.0
    if abs_dd < DRAWDOWN_THRESHOLDS["low"]:
        dd_score = 1.0
        dd_label = "Mild"
    elif abs_dd <= DRAWDOWN_THRESHOLDS["moderate"]:
        dd_score = 2.0
        dd_label = "Moderate"
    else:
        dd_score = 3.0
        dd_label = "Severe"

    # 3. Sharpe Ratio sub-score
    if np.isnan(sharpe_ratio) or sharpe_ratio < SHARPE_THRESHOLDS["moderate_quality"]:
        sharpe_score = 3.0
        sharpe_label = "Negative Excess Return"
    elif sharpe_ratio < SHARPE_THRESHOLDS["high_quality"]:
        sharpe_score = 2.0
        sharpe_label = "Moderate Return per Unit Risk"
    else:
        sharpe_score = 1.0
        sharpe_label = "High Return per Unit Risk"

    # Weighted Composite Score
    composite = (
        WEIGHTS["volatility"] * vol_score
        + WEIGHTS["max_drawdown"] * dd_score
        + WEIGHTS["sharpe_ratio"] * sharpe_score
    )

    sub_scores = {
        "volatility_score": vol_score,
        "volatility_label": vol_label,
        "drawdown_score": dd_score,
        "drawdown_label": dd_label,
        "sharpe_score": sharpe_score,
        "sharpe_label": sharpe_label,
        "composite_score": round(composite, 2),
    }

    return round(composite, 2), sub_scores


def classify_risk(risk_score: float) -> str:
    """
    Maps a numeric composite risk score to a categorical classification.

    Classification Boundaries:
        - Score < 1.60        -> 'LOW'
        - 1.60 <= Score <= 2.40 -> 'MODERATE'
        - Score > 2.40        -> 'HIGH'

    Parameters:
        risk_score (float): Composite risk score (1.0 to 3.0).

    Returns:
        str: 'LOW', 'MODERATE', or 'HIGH'.
    """
    if risk_score < 1.60:
        return "LOW"
    elif risk_score <= 2.40:
        return "MODERATE"
    else:
        return "HIGH"


def explain_risk(
    volatility: float,
    max_drawdown: float,
    sharpe_ratio: float,
    risk_level: str,
    composite_score: float,
) -> str:
    """
    Generates an educational, transparent rationale for the assigned risk classification.

    Parameters:
        volatility (float): Annualized volatility.
        max_drawdown (float): Maximum drawdown.
        sharpe_ratio (float): Sharpe ratio.
        risk_level (str): 'LOW', 'MODERATE', or 'HIGH'.
        composite_score (float): Composite risk score.

    Returns:
        str: Human-readable narrative explanation.
    """
    abs_dd = abs(max_drawdown)

    explanations = []

    # Volatility explanation
    if volatility < VOLATILITY_THRESHOLDS["low"]:
        explanations.append(
            f"Annualized volatility ({volatility:.1%}) is low (<18%), reflecting stable price behavior."
        )
    elif volatility <= VOLATILITY_THRESHOLDS["moderate"]:
        explanations.append(
            f"Annualized volatility ({volatility:.1%}) is moderate (18%–28%), typical of Indian blue-chip equities."
        )
    else:
        explanations.append(
            f"Annualized volatility ({volatility:.1%}) is elevated (>28%), signaling wide price fluctuations."
        )

    # Drawdown explanation
    if abs_dd < DRAWDOWN_THRESHOLDS["low"]:
        explanations.append(
            f"Maximum peak-to-trough drawdown ({max_drawdown:.1%}) is well-controlled (<15%)."
        )
    elif abs_dd <= DRAWDOWN_THRESHOLDS["moderate"]:
        explanations.append(
            f"Maximum historical drawdown ({max_drawdown:.1%}) represents standard market cyclicality (15%–30%)."
        )
    else:
        explanations.append(
            f"Maximum historical drawdown ({max_drawdown:.1%}) is severe (>30%), indicating significant capital contraction from peak levels."
        )

    # Sharpe ratio explanation
    if sharpe_ratio > SHARPE_THRESHOLDS["high_quality"]:
        explanations.append(
            f"Sharpe ratio ({sharpe_ratio:.2f}) indicates strong risk-adjusted returns exceeding the 6% Indian benchmark rate."
        )
    elif sharpe_ratio >= SHARPE_THRESHOLDS["moderate_quality"]:
        explanations.append(
            f"Sharpe ratio ({sharpe_ratio:.2f}) indicates acceptable risk-adjusted returns above the risk-free rate."
        )
    else:
        explanations.append(
            f"Sharpe ratio ({sharpe_ratio:.2f}) is negative, meaning historical return underperformed the 6% sovereign risk-free rate."
        )

    summary = (
        f"**Risk Classification: {risk_level}** (Composite Score: {composite_score:.2f} / 3.00)\n\n"
        + " ".join(explanations)
        + "\n\n*Note: Risk classification is based on historical volatility, Sharpe ratio, and maximum drawdown. "
        "It does not predict future stock performance.*"
    )

    return summary


def evaluate_stock_risk(
    data: pd.DataFrame, risk_free_rate: float = 0.06
) -> Dict[str, Any]:
    """
    High-level convenience pipeline that runs financial analysis and computes
    the complete multi-factor risk assessment.

    Parameters:
        data (pd.DataFrame): Stock historical price DataFrame.
        risk_free_rate (float): Benchmark risk-free interest rate (default: 0.06).

    Returns:
        dict: Complete risk summary including metrics, scores, and explanation.
    """
    metrics = analysis.calculate_all_metrics(data, risk_free_rate=risk_free_rate)

    composite_score, sub_scores = calculate_risk_score(
        volatility=metrics["annualized_volatility"],
        max_drawdown=metrics["max_drawdown"],
        sharpe_ratio=metrics["sharpe_ratio"],
    )

    risk_level = classify_risk(composite_score)

    explanation = explain_risk(
        volatility=metrics["annualized_volatility"],
        max_drawdown=metrics["max_drawdown"],
        sharpe_ratio=metrics["sharpe_ratio"],
        risk_level=risk_level,
        composite_score=composite_score,
    )

    return {
        **metrics,
        "composite_risk_score": composite_score,
        "sub_scores": sub_scores,
        "overall_risk_level": risk_level,
        "risk_explanation": explanation,
    }


class MLRiskClassifier:
    """
    Scikit-Learn Random Forest Classifier architecture stub.
    
    This class documents how machine learning can be integrated into future versions
    of StockScope for volatility regime prediction or multi-factor risk classification.
    """
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.is_fitted = False

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts multi-factor quantitative features for machine learning:
        - 20-day rolling volatility
        - 50-day rolling return
        - Volume ratio (Volume / 20-day Volume MA)
        - Price distance to 50-day MA
        """
        returns = df["Close"].pct_change()
        features = pd.DataFrame(index=df.index)
        features["rolling_vol_20"] = returns.rolling(20).std() * np.sqrt(252)
        features["rolling_ret_50"] = returns.rolling(50).mean() * 252
        features["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        features["price_to_ma50"] = df["Close"] / df["Close"].rolling(50).mean() - 1.0
        return features.dropna()

    def train_placeholder(self):
        """
        Placeholder demonstrating where Scikit-Learn RandomForestClassifier
        will be trained when historical cross-sectional labeled data is provided.
        """
        pass
