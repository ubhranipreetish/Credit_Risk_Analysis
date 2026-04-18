import os

from backend.schemas.analyze import HealthResponse


class HealthService:
    def __init__(self, model_dir: str = "models", rag_index_dir: str = "rag/faiss_index"):
        self.model_dir = model_dir
        self.rag_index_dir = rag_index_dir

    def check(self) -> HealthResponse:
        checks = {
            "logistic_model": os.path.exists(os.path.join(self.model_dir, "logistic_pipeline.joblib")),
            "decision_tree_model": os.path.exists(os.path.join(self.model_dir, "decision_tree_pipeline.joblib")),
            "target_encoder": os.path.exists(os.path.join(self.model_dir, "target_encoder.joblib")),
            "rag_index": os.path.exists(os.path.join(self.rag_index_dir, "index.faiss")),
            "rag_metadata": os.path.exists(os.path.join(self.rag_index_dir, "chunks_metadata.json")),
            "groq_api_key": bool(os.environ.get("GROQ_API_KEY")),
        }

        status = "ok" if all(checks.values()) else "degraded"
        return HealthResponse(
            status=status,
            service="credit-risk-agent-backend",
            checks=checks,
            version="1.0.0",
        )
