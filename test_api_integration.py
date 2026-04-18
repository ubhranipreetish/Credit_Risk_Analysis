"""High-value API integration tests for analysis endpoint.

Mocking strategy:
- Startup RAG initialization is monkeypatched to avoid file/index side effects.
- Retrieval is monkeypatched at `agent.rag.retrieve_docs` so tests control grounding input.
- LLM path is either:
  1) monkeypatched `generate_decision` for schema integrity checks, or
  2) real `generate_decision` with mocked Groq client to validate grounding rewrite.
- No external API calls are made.
"""

from __future__ import annotations

import json
from typing import Dict, Iterator, List

import pytest
from fastapi.testclient import TestClient


REQUIRED_DECISION_KEYS = {
    "Borrower Profile Summary",
    "Risk Analysis",
    "Lending Decision",
    "Confidence",
    "Regulatory References",
    "Disclaimer",
}


@pytest.fixture
def sample_payload() -> Dict[str, object]:
    return {
        "person_age": 30,
        "person_income": 65000,
        "person_emp_length": 5.0,
        "loan_amnt": 12000,
        "loan_int_rate": 11.0,
        "loan_percent_income": 12000 / 65000,
        "cb_person_cred_hist_length": 8,
        "person_home_ownership": "RENT",
        "loan_intent": "PERSONAL",
        "loan_grade": "C",
        "cb_person_default_on_file": "N",
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    # Keep app startup deterministic and non-blocking for tests.
    import backend.main as backend_main

    monkeypatch.setattr(
        backend_main,
        "ensure_faiss_index_ready",
        lambda *args, **kwargs: {
            "ready": False,
            "mode": "safe_empty",
            "vectors": 0,
            "metadata_entries": 0,
            "index_dir": "rag/faiss_index",
            "reason": "test-safe-mode",
        },
    )
    monkeypatch.setattr(backend_main, "initialize_retriever_safe_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_main, "preload_retriever_index", lambda *args, **kwargs: None)

    with TestClient(backend_main.app) as test_client:
        yield test_client


def _assert_exact_decision_schema(response_json: Dict[str, object]) -> Dict[str, object]:
    assert "decision" in response_json, "Response must contain 'decision' object"
    decision = response_json["decision"]
    assert isinstance(decision, dict), "'decision' must be a JSON object"

    assert set(decision.keys()) == REQUIRED_DECISION_KEYS, (
        "Decision schema mismatch. "
        f"Expected exactly {sorted(REQUIRED_DECISION_KEYS)}, got {sorted(decision.keys())}"
    )
    return decision


def test_response_schema_integrity(
    client: TestClient,
    sample_payload: Dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call main analysis endpoint and assert exact decision keys."""
    import agent.rag
    import agent.llm_reasoner

    monkeypatch.setattr(
        agent.rag,
        "retrieve_docs",
        lambda *args, **kwargs: [
            {
                "text": "RBI guideline on affordability checks and prudent underwriting.",
                "source": "RBI_Guidelines.pdf",
                "page": 2,
                "score": 0.9,
                "chunk_id": 1,
            }
        ],
    )

    monkeypatch.setattr(
        agent.llm_reasoner,
        "generate_decision",
        lambda risk, explanation, retrieved_docs: {
            "Borrower Profile Summary": "Borrower has stable income and moderate debt burden.",
            "Risk Analysis": "Risk is manageable with standard underwriting controls.",
            "Lending Decision": "APPROVE",
            "Confidence": 0.81,
            "Regulatory References": ["RBI_Guidelines.pdf"],
            "Disclaimer": "AI-assisted recommendation; human approval required.",
        },
    )

    resp = client.post("/analyze", json=sample_payload)
    assert resp.status_code == 200, resp.text

    response_json = resp.json()
    decision = _assert_exact_decision_schema(response_json)

    assert decision["Lending Decision"] in {"APPROVE", "REJECT", "CONDITIONAL"}
    assert isinstance(decision["Confidence"], float)
    assert 0.0 <= decision["Confidence"] <= 1.0


def test_regulatory_grounding_rewrites_hallucinated_references(
    client: TestClient,
    sample_payload: Dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock retrieval + LLM output and verify references are grounded to retrieved data only."""
    import agent.rag
    import agent.llm_reasoner

    retrieved_docs: List[Dict[str, object]] = [
        {
            "text": "RBI guidelines require affordability and risk governance controls.",
            "source": "RBI_Guidelines.pdf",
            "page": 1,
            "score": 0.92,
            "chunk_id": 7,
        },
        {
            "text": "Risk management framework under RBI policy with conservative lending caps.",
            "source": "RBI_Risk_Management.pdf",
            "page": 3,
            "score": 0.88,
            "chunk_id": 8,
        },
    ]

    monkeypatch.setattr(agent.rag, "retrieve_docs", lambda *args, **kwargs: retrieved_docs)

    # Force the real LLM pipeline without network by mocking Groq client response.
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class _FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeChoice:
        def __init__(self, content: str) -> None:
            self.message = _FakeMessage(content)

    class _FakeCompletion:
        def __init__(self, content: str) -> None:
            self.choices = [_FakeChoice(content)]

    class _FakeCompletions:
        @staticmethod
        def create(**kwargs):
            llm_json = {
                "Borrower Profile Summary": "Borrower is moderately risky with manageable leverage.",
                "Risk Analysis": "Analysis cites one valid and one hallucinated source.",
                "Lending Decision": "CONDITIONAL",
                "Confidence": 0.67,
                "Regulatory References": ["Imaginary Circular 2025", "RBI Guidelines"],
                "Disclaimer": "AI-assisted recommendation; human approval required.",
            }
            return _FakeCompletion(json.dumps(llm_json))

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(agent.llm_reasoner, "_get_groq_client", lambda: _FakeClient())

    resp = client.post("/analyze", json=sample_payload)
    assert resp.status_code == 200, resp.text

    response_json = resp.json()
    decision = _assert_exact_decision_schema(response_json)

    returned_refs = decision["Regulatory References"]
    retrieved_sources = {doc["source"] for doc in retrieved_docs}

    # Critical grounding assertion: all returned references must come from retrieval sources,
    # unless the system explicitly returns no-verifiable-reference fallback.
    if returned_refs != ["No verifiable regulatory reference found"]:
        assert set(returned_refs).issubset(retrieved_sources), (
            f"Ungrounded references detected: {returned_refs}, retrieved={sorted(retrieved_sources)}"
        )


def test_no_groq_fallback_still_returns_complete_schema(
    client: TestClient,
    sample_payload: Dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disable GROQ_API_KEY and ensure fallback response remains valid and complete."""
    import agent.rag

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(agent.rag, "retrieve_docs", lambda *args, **kwargs: [])

    resp = client.post("/analyze", json=sample_payload)
    assert resp.status_code == 200, resp.text

    response_json = resp.json()
    decision = _assert_exact_decision_schema(response_json)

    assert isinstance(decision["Confidence"], float)
    assert 0.0 <= decision["Confidence"] <= 1.0
    assert decision["Lending Decision"] in {"APPROVE", "REJECT", "CONDITIONAL"}
    assert isinstance(decision["Disclaimer"], str) and len(decision["Disclaimer"].strip()) > 0
