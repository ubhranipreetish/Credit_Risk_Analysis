from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class AnalyzeRequest(BaseModel):
    person_age: int = Field(..., ge=18, le=100)
    person_income: int = Field(..., ge=0)
    person_emp_length: float = Field(..., ge=0)
    loan_amnt: int = Field(..., gt=0)
    loan_int_rate: float = Field(..., ge=0, le=50)
    loan_percent_income: float = Field(..., ge=0)
    cb_person_cred_hist_length: int = Field(..., ge=0)
    person_home_ownership: Literal["RENT", "MORTGAGE", "OWN", "OTHER"]
    loan_intent: Literal[
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    cb_person_default_on_file: Literal["Y", "N"]


class RiskPayload(BaseModel):
    prediction: int
    label: str
    probability: float
    model_used: str


class DecisionPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    borrower_profile_summary: str = Field(
        alias="Borrower Profile Summary",
        validation_alias=AliasChoices(
            "Borrower Profile Summary",
            "Profile Summary",
            "profile_summary",
            "borrower_profile_summary",
        ),
    )
    risk_analysis: str = Field(
        alias="Risk Analysis",
        validation_alias=AliasChoices("Risk Analysis", "risk_analysis"),
    )
    lending_decision: Literal["APPROVE", "REJECT", "CONDITIONAL"] = Field(
        alias="Lending Decision",
        validation_alias=AliasChoices(
            "Lending Decision",
            "Decision",
            "decision",
            "lending_decision",
        ),
    )
    confidence: float = Field(
        alias="Confidence",
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("Confidence", "Confidence Score", "confidence", "confidence_score"),
    )
    regulatory_references: List[str] = Field(
        default_factory=list,
        alias="Regulatory References",
        validation_alias=AliasChoices(
            "Regulatory References",
            "Regulatory Sources",
            "regulatory_references",
            "regulatory_sources",
        ),
    )
    disclaimer: str = Field(
        alias="Disclaimer",
        validation_alias=AliasChoices("Disclaimer", "disclaimer"),
    )

    @field_validator("lending_decision", mode="before")
    @classmethod
    def _normalize_lending_decision(cls, value: Any) -> str:
        if value is None:
            return "CONDITIONAL"

        normalized = str(value).strip().upper()
        if normalized in {"APPROVE", "APPROVED"}:
            return "APPROVE"
        if normalized in {"REJECT", "REJECTED"}:
            return "REJECT"
        if normalized in {"CONDITIONAL", "CONDITION"}:
            return "CONDITIONAL"
        return "CONDITIONAL"

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: Any) -> float:
        if value is None:
            return 0.5

        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "high":
                return 0.85
            if lowered == "medium":
                return 0.60
            if lowered == "low":
                return 0.35
            try:
                return float(value)
            except ValueError:
                return 0.5

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.5

    @field_validator("regulatory_references", mode="before")
    @classmethod
    def _normalize_regulatory_references(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, (tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        reference = str(value).strip()
        return [reference] if reference else []


class AnalysisWarning(BaseModel):
    code: str
    message: str


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    status: Literal["success", "degraded"]
    steps_completed: List[str]
    risk_tier: Optional[str]
    confidence_score: float
    reasoning_passes: int
    risk: RiskPayload
    decision: DecisionPayload
    warnings: List[AnalysisWarning] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def normalize_decision_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a canonical decision payload with the new field names only."""
    model = DecisionPayload.model_validate(payload or {})
    return model.model_dump(by_alias=True)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    checks: Dict[str, bool]
    version: str
