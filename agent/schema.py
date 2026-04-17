"""
Input Schema Validation
========================

Defines the borrower profile data structure and validates incoming data
against the feature schema used during model training.

Usage:
    >>> from agent.schema import validate_input
    >>> input_df = validate_input({
    ...     "person_age": 30,
    ...     "person_income": 50000,
    ...     "person_emp_length": 5.0,
    ...     "loan_amnt": 10000,
    ...     "loan_int_rate": 10.5,
    ...     "loan_percent_income": 0.20,
    ...     "cb_person_cred_hist_length": 8,
    ...     "person_home_ownership": "RENT",
    ...     "loan_intent": "PERSONAL",
    ...     "loan_grade": "B",
    ...     "cb_person_default_on_file": "N"
    ... })
"""

from dataclasses import dataclass, asdict
from typing import Union

import pandas as pd


# ── Feature column order (must match training schema) ────────────────────────

NUMERICAL_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

FEATURE_COLUMNS = NUMERICAL_COLS + CATEGORICAL_COLS

# ── Valid categorical values ─────────────────────────────────────────────────

VALID_HOME_OWNERSHIP = {"RENT", "MORTGAGE", "OWN", "OTHER"}
VALID_LOAN_INTENT = {
    "PERSONAL", "EDUCATION", "MEDICAL",
    "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION",
}
VALID_LOAN_GRADE = {"A", "B", "C", "D", "E", "F", "G"}
VALID_DEFAULT_ON_FILE = {"Y", "N"}


@dataclass
class BorrowerProfile:
    """Structured representation of a borrower's loan application."""

    person_age: int
    person_income: int
    person_emp_length: float
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: int
    person_home_ownership: str
    loan_intent: str
    loan_grade: str
    cb_person_default_on_file: str

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a single-row DataFrame with correct column order."""
        return pd.DataFrame([asdict(self)])[FEATURE_COLUMNS]


def validate_input(data: dict) -> pd.DataFrame:
    """
    Validate and convert raw input data into a model-ready DataFrame.

    Parameters
    ----------
    data : dict
        Raw borrower profile data. Must contain all 11 feature columns.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with correctly ordered and typed columns.

    Raises
    ------
    ValueError
        If required columns are missing, values are out of range,
        or categorical values are not in the allowed set.
    """
    # ── Check required columns ───────────────────────────────────────────
    missing = [col for col in FEATURE_COLUMNS if col not in data]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Expected columns: {FEATURE_COLUMNS}"
        )

    # ── Type coercion ────────────────────────────────────────────────────
    try:
        validated = {
            "person_age": int(data["person_age"]),
            "person_income": int(data["person_income"]),
            "person_emp_length": float(data["person_emp_length"]),
            "loan_amnt": int(data["loan_amnt"]),
            "loan_int_rate": float(data["loan_int_rate"]),
            "loan_percent_income": float(data["loan_percent_income"]),
            "cb_person_cred_hist_length": int(data["cb_person_cred_hist_length"]),
            "person_home_ownership": str(data["person_home_ownership"]).upper(),
            "loan_intent": str(data["loan_intent"]).upper(),
            "loan_grade": str(data["loan_grade"]).upper(),
            "cb_person_default_on_file": str(data["cb_person_default_on_file"]).upper(),
        }
    except (TypeError, ValueError) as e:
        raise ValueError(f"Type conversion error: {e}") from e

    # ── Range validations ────────────────────────────────────────────────
    errors = []

    if not (18 <= validated["person_age"] <= 100):
        errors.append(f"person_age must be 18-100, got {validated['person_age']}")
    if validated["person_income"] < 0:
        errors.append(f"person_income must be >= 0, got {validated['person_income']}")
    if validated["person_emp_length"] < 0:
        errors.append(f"person_emp_length must be >= 0, got {validated['person_emp_length']}")
    if validated["loan_amnt"] <= 0:
        errors.append(f"loan_amnt must be > 0, got {validated['loan_amnt']}")
    if not (0.0 <= validated["loan_int_rate"] <= 50.0):
        errors.append(f"loan_int_rate must be 0-50, got {validated['loan_int_rate']}")
    if validated["loan_percent_income"] < 0:
        errors.append(f"loan_percent_income must be >= 0, got {validated['loan_percent_income']}")
    if validated["cb_person_cred_hist_length"] < 0:
        errors.append(f"cb_person_cred_hist_length must be >= 0, got {validated['cb_person_cred_hist_length']}")

    # ── Categorical validations ──────────────────────────────────────────
    if validated["person_home_ownership"] not in VALID_HOME_OWNERSHIP:
        errors.append(
            f"person_home_ownership must be one of {VALID_HOME_OWNERSHIP}, "
            f"got '{validated['person_home_ownership']}'"
        )
    if validated["loan_intent"] not in VALID_LOAN_INTENT:
        errors.append(
            f"loan_intent must be one of {VALID_LOAN_INTENT}, "
            f"got '{validated['loan_intent']}'"
        )
    if validated["loan_grade"] not in VALID_LOAN_GRADE:
        errors.append(
            f"loan_grade must be one of {VALID_LOAN_GRADE}, "
            f"got '{validated['loan_grade']}'"
        )
    if validated["cb_person_default_on_file"] not in VALID_DEFAULT_ON_FILE:
        errors.append(
            f"cb_person_default_on_file must be one of {VALID_DEFAULT_ON_FILE}, "
            f"got '{validated['cb_person_default_on_file']}'"
        )

    if errors:
        raise ValueError("Validation errors:\n  - " + "\n  - ".join(errors))

    # ── Build DataFrame with correct column order ────────────────────────
    profile = BorrowerProfile(**validated)
    return profile.to_dataframe()
