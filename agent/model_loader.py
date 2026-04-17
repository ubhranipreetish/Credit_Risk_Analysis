"""
Model Loader & Risk Predictor
===============================

Loads trained sklearn pipelines (which embed the preprocessor) and provides
a clean prediction interface returning structured results.

The saved pipelines already contain the full preprocessing → classifier flow,
so raw DataFrames can be passed directly to predict().

Usage:
    >>> from agent.model_loader import predict_risk
    >>> result = predict_risk(input_df, model_name="decision_tree")
    >>> print(result)
    {'prediction': 1, 'label': 'High Risk', 'probability': 0.87, 'model_used': 'decision_tree'}
"""

import os
import logging
from typing import Optional

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

MODEL_REGISTRY = {
    "decision_tree": "decision_tree_pipeline.joblib",
    "logistic": "logistic_pipeline.joblib",
}

# ── Module-level cache ───────────────────────────────────────────────────────

_model_cache: dict[str, Pipeline] = {}
_encoder_cache: Optional[LabelEncoder] = None


def load_model(model_name: str = "decision_tree") -> Pipeline:
    """
    Load a trained sklearn pipeline by name.

    Parameters
    ----------
    model_name : str
        One of 'decision_tree' or 'logistic'.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The full preprocessing + classifier pipeline.

    Raises
    ------
    ValueError
        If model_name is not in the registry.
    FileNotFoundError
        If the model file does not exist on disk.
    """
    global _model_cache

    if model_name in _model_cache:
        return _model_cache[model_name]

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_path = os.path.join(MODEL_DIR, MODEL_REGISTRY[model_name])

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Please run the training pipeline first (Milestone 1)."
        )

    logger.info("Loading model '%s' from %s", model_name, model_path)
    pipeline = joblib.load(model_path)
    _model_cache[model_name] = pipeline
    return pipeline


def load_target_encoder() -> LabelEncoder:
    """
    Load the target label encoder used during training.

    Returns
    -------
    sklearn.preprocessing.LabelEncoder
        Encoder mapping between numeric predictions and original labels.

    Raises
    ------
    FileNotFoundError
        If the encoder file does not exist.
    """
    global _encoder_cache

    if _encoder_cache is not None:
        return _encoder_cache

    encoder_path = os.path.join(MODEL_DIR, "target_encoder.joblib")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Target encoder not found: {encoder_path}. "
            f"Please run the training pipeline first."
        )

    logger.info("Loading target encoder from %s", encoder_path)
    _encoder_cache = joblib.load(encoder_path)
    return _encoder_cache


def predict_risk(
    input_df: pd.DataFrame,
    model_name: str = "decision_tree",
) -> dict:
    """
    Run credit risk prediction on a validated borrower profile.

    Parameters
    ----------
    input_df : pd.DataFrame
        Single-row DataFrame with the 11 feature columns.
        Should be produced by `schema.validate_input()`.
    model_name : str
        Which model to use: 'decision_tree' or 'logistic'.

    Returns
    -------
    dict
        Structured prediction result:
        {
            "prediction": int,          # 0 or 1
            "label": str,               # "Low Risk" or "High Risk"
            "probability": float,       # probability of default (class 1)
            "model_used": str           # model name used
        }
    """
    pipeline = load_model(model_name)

    # Raw prediction (0 = no default, 1 = default)
    prediction = int(pipeline.predict(input_df)[0])

    # Default probability
    try:
        probability = float(pipeline.predict_proba(input_df)[0][1])
    except AttributeError:
        # Fallback if model doesn't support predict_proba
        logger.warning("Model '%s' does not support predict_proba", model_name)
        probability = float(prediction)

    # Human-readable label
    label = "High Risk" if prediction == 1 else "Low Risk"

    result = {
        "prediction": prediction,
        "label": label,
        "probability": round(probability, 4),
        "model_used": model_name,
    }

    logger.info("Prediction result: %s", result)
    return result
