"""
Agent Package — Milestone 2: Agentic AI Lending Decision Support System
========================================================================

This package provides modular components for an AI-powered lending decision
pipeline. Designed for future integration with LangGraph agent workflows.

Components:
    - schema:          Input validation & borrower profile data structure
    - model_loader:    ML model loading & risk prediction
    - risk_explainer:  Feature importance-based risk explanation
    - rag:             Retrieval-Augmented Generation pipeline (FAISS + sentence-transformers)
    - llm_reasoner:    LLM-powered lending decision generation (Groq / Llama 3)
    - workflow:        LangGraph agent workflow connecting all nodes
"""

from agent.schema import validate_input, BorrowerProfile
from agent.model_loader import predict_risk, load_model, load_target_encoder
from agent.risk_explainer import explain_risk
from agent.rag import build_rag_index, retrieve_docs
from agent.llm_reasoner import generate_decision, build_query
from agent.workflow import run_agent, run_workflow, build_agent, AgentState, WorkflowOutput

__all__ = [
    "validate_input",
    "BorrowerProfile",
    "predict_risk",
    "load_model",
    "load_target_encoder",
    "explain_risk",
    "build_rag_index",
    "retrieve_docs",
    "generate_decision",
    "build_query",
    "run_agent",
    "run_workflow",
    "build_agent",
    "AgentState",
    "WorkflowOutput",
]
