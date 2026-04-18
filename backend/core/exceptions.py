class BackendError(Exception):
    """Base exception for backend service errors."""

    status_code = 500
    error_code = "backend_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidInputError(BackendError):
    status_code = 422
    error_code = "invalid_input"


class WorkflowExecutionError(BackendError):
    status_code = 500
    error_code = "workflow_execution_error"


class MissingRAGIndexError(BackendError):
    status_code = 503
    error_code = "missing_rag_index"
