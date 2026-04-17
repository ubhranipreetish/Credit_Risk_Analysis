"""
Risk Explanation Module
========================

Extracts interpretable risk factors from trained ML pipelines.
Returns structured data (list of dicts) suitable for agent consumption
and LLM prompt construction.

For Decision Trees:  Uses global feature importances (Gini importance).
For Logistic Regression: Uses signed coefficients (positive = increases risk).

Usage:
    >>> from agent.risk_explainer import explain_risk
    >>> factors = explain_risk(pipeline, input_df, model_name="decision_tree", top_n=5)
    >>> for f in factors:
    ...     print(f"{f['feature']}: {f['importance']:.3f} ({f['direction']})")
"""

import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _clean_feature_name(name: str) -> str:
    """
    Strip sklearn ColumnTransformer prefixes and format for display.

    Examples:
        'num__person_age'  → 'person_age'
        'cat__loan_grade'  → 'loan_grade'
    """
    return name.replace("num__", "").replace("cat__", "")


def _explain_decision_tree(pipeline: Pipeline, top_n: int) -> List[dict]:
    """Extract feature importances from a Decision Tree classifier."""
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]
    importances = classifier.feature_importances_

    # Build sorted importance list
    feature_importance = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []
    for name, importance in feature_importance[:top_n]:
        if importance < 1e-6:
            continue  # Skip negligible features
        results.append({
            "feature": _clean_feature_name(name),
            "importance": round(float(importance), 4),
            "direction": "risk_factor",  # Tree importances don't indicate direction
        })

    return results


def _explain_logistic(pipeline: Pipeline, top_n: int) -> List[dict]:
    """Extract signed coefficients from a Logistic Regression classifier."""
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]
    coefficients = classifier.coef_[0]

    # Sort by absolute value (largest impact first)
    abs_coefs = np.abs(coefficients)
    sorted_indices = np.argsort(abs_coefs)[::-1]

    results = []
    for idx in sorted_indices[:top_n]:
        coef_val = float(coefficients[idx])
        if abs(coef_val) < 1e-6:
            continue
        results.append({
            "feature": _clean_feature_name(feature_names[idx]),
            "importance": round(abs(coef_val), 4),
            "direction": "increases_risk" if coef_val > 0 else "decreases_risk",
            "coefficient": round(coef_val, 4),
        })

    return results


def explain_risk(
    pipeline: Pipeline,
    input_df: pd.DataFrame,
    model_name: str = "decision_tree",
    top_n: int = 5,
) -> List[dict]:
    """
    Identify and return the top risk-driving features for a prediction.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline (preprocessor + classifier).
    input_df : pd.DataFrame
        Single-row borrower profile DataFrame (used for context/future SHAP).
    model_name : str
        One of 'decision_tree' or 'logistic'.
    top_n : int
        Number of top features to return.

    Returns
    -------
    List[dict]
        Ordered list of risk factors, each containing:
        - feature: str        — clean feature name
        - importance: float   — magnitude of importance/coefficient
        - direction: str      — 'increases_risk', 'decreases_risk', or 'risk_factor'
        - coefficient: float  — (logistic only) signed coefficient value
    """
    if model_name == "decision_tree":
        factors = _explain_decision_tree(pipeline, top_n)
    elif model_name == "logistic":
        factors = _explain_logistic(pipeline, top_n)
    else:
        logger.warning("Unknown model type '%s', defaulting to decision_tree", model_name)
        factors = _explain_decision_tree(pipeline, top_n)

    logger.info("Top %d risk factors: %s", top_n, [f["feature"] for f in factors])
    return factors
