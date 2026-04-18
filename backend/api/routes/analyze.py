from fastapi import APIRouter, Depends

from backend.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from backend.services.analysis_service import AnalysisService

router = APIRouter()


def get_analysis_service() -> AnalysisService:
    return AnalysisService()


@router.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze_credit_risk(
    payload: AnalyzeRequest,
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalyzeResponse:
    return service.analyze(payload)
