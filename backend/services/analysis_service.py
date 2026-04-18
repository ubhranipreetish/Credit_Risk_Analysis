import os
from typing import Dict, List

from agent.schema import validate_input
from agent.workflow import run_workflow
from backend.core.exceptions import InvalidInputError, WorkflowExecutionError
from backend.schemas.analyze import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalysisWarning,
    DecisionPayload,
    RiskPayload,
    normalize_decision_payload,
)


class AnalysisService:
    def __init__(self, rag_index_dir: str = "rag/faiss_index"):
        self.rag_index_dir = rag_index_dir

    def analyze(self, payload: AnalyzeRequest) -> AnalyzeResponse:
        input_data = payload.model_dump()

        # Defensive validation using the same schema checker used by the workflow.
        try:
            validate_input(input_data)
        except ValueError as exc:
            raise InvalidInputError(str(exc)) from exc

        try:
            result = run_workflow(input_data=input_data, debug=False, max_passes=2)
        except Exception as exc:
            raise WorkflowExecutionError(f"Workflow execution failed: {exc}") from exc

        if result.get("error"):
            error_message = str(result["error"])
            if "Validation failed" in error_message:
                raise InvalidInputError(error_message)
            raise WorkflowExecutionError(error_message)

        decision_payload = normalize_decision_payload(result.get("decision") or {})
        warnings = self._collect_warnings(result, decision_payload)

        risk_payload = result.get("risk") or {}

        response_status = "degraded" if warnings else "success"
        return AnalyzeResponse(
            status=response_status,
            steps_completed=result.get("steps_completed") or [],
            risk_tier=result.get("risk_tier"),
            confidence_score=float(result.get("confidence_score") or 0.0),
            reasoning_passes=int(result.get("reasoning_passes") or 0),
            risk=RiskPayload(**risk_payload),
            decision=DecisionPayload.model_validate(decision_payload),
            warnings=warnings,
            metadata={
                "workflow_engine": "langgraph_compiled",
                "max_passes": 2,
            },
        )

    def _collect_warnings(self, result: Dict, decision_payload: Dict) -> List[AnalysisWarning]:
        warnings: List[AnalysisWarning] = []
        steps = result.get("steps_completed") or []

        if any(
            step in steps
            for step in ("rag:NO_INDEX", "rag_deep:NO_INDEX", "rag:FAILED", "rag_deep:FAILED", "rag:EMPTY", "rag_deep:EMPTY")
        ):
            warnings.append(
                AnalysisWarning(
                    code="missing_rag_index",
                    message="RAG context is unavailable; decision generated with empty retrieval context.",
                )
            )

        disclaimer = str(decision_payload.get("Disclaimer", ""))
        if "FALLBACK MODE" in disclaimer:
            if not os.environ.get("GROQ_API_KEY"):
                message = (
                    "LLM is in fallback mode because GROQ_API_KEY is not configured. "
                    "Set GROQ_API_KEY to enable live LLM reasoning."
                )
            else:
                message = "LLM reasoning failed and fallback decision logic was used."
            warnings.append(
                AnalysisWarning(
                    code="llm_fallback",
                    message=message,
                )
            )

        return warnings
