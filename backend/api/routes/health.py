from fastapi import APIRouter, Depends

from backend.schemas.analyze import HealthResponse
from backend.services.health_service import HealthService

router = APIRouter()


def get_health_service() -> HealthService:
    return HealthService()


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health(
    service: HealthService = Depends(get_health_service),
) -> HealthResponse:
    return service.check()
