"""
LangGraph Agent Workflow — Credit Risk Decision Pipeline (Advanced)
=====================================================================

Orchestrates the full lending decision pipeline as a stateful LangGraph
agent with **conditional routing**, **confidence scoring**, and
**reflective multi-step reasoning**.

Advanced features:
    - Conditional routing: borderline cases get deeper RAG retrieval
    - Confidence scoring: numeric score combining ML + LLM signals
    - Reflection loop: low-confidence decisions are re-evaluated
    - Decision thresholds: clear-cut cases skip reflection

Pipeline (with conditional paths):

    START → validate → predict → risk_router ─┬─→ explain_standard → query → rag → decision ─┐
                                               │                                               │
                                               └─→ explain_deep → query_deep → rag_deep ──────┘
                                                                                               │
                                          confidence_router ←──────────────────────────────────┘
                                               │
                                    ┌──────────┴───────────┐
                                    ▼                      ▼
                                  (HIGH)                (LOW/MED)
                                    │                      │
                                    ▼                      ▼
                                   END              reflect → END

Usage:
    >>> from agent.workflow import run_agent
    >>> result = run_agent(input_data, debug=True)
    >>> print(result["decision"]["Lending Decision"])
    >>> print(result["confidence_score"])
"""

import logging
import time
from typing import TypedDict, List, Dict, Any, Optional, Literal

from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  AGENT STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════


class AgentState(TypedDict):
    """
    Shared state passed between all nodes in the agent workflow.

    Extended with confidence scoring, reflection tracking, and
    risk classification for conditional routing.
    """

    # ── Core pipeline data ───────────────────────────────────────────────
    input: Dict[str, Any]
    validated_data: Optional[Dict[str, Any]]
    risk: Optional[Dict[str, Any]]
    explanation: Optional[List[Dict[str, Any]]]
    query: Optional[str]
    documents: Optional[List[Dict[str, Any]]]
    decision: Optional[Dict[str, Any]]

    # ── Advanced agent features ──────────────────────────────────────────
    # Risk classification for conditional routing
    risk_tier: Optional[str]  # "clear_low", "borderline", "clear_high"

    # Numeric confidence score (0.0 - 1.0)
    confidence_score: Optional[float]

    # Number of reasoning passes completed
    reasoning_passes: Optional[int]

    # Maximum allowed reasoning passes (prevents infinite loops)
    max_passes: Optional[int]

    # ── Metadata ─────────────────────────────────────────────────────────
    error: Optional[str]
    steps_completed: Optional[List[str]]


class WorkflowOutput(TypedDict):
    """Structured output contract for app-level workflow consumers."""

    input: Dict[str, Any]
    validated_data: Optional[Dict[str, Any]]
    risk: Optional[Dict[str, Any]]
    explanation: List[Dict[str, Any]]
    query: Optional[str]
    documents: List[Dict[str, Any]]
    decision: Dict[str, Any]
    risk_tier: Optional[str]
    confidence_score: float
    reasoning_passes: int
    steps_completed: List[str]
    error: Optional[str]


# ── Confidence threshold for accepting a decision without reflection ─────
CONFIDENCE_ACCEPT_THRESHOLD = 0.70


# ═══════════════════════════════════════════════════════════════════════════
#  NODE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════


def validate_node(state: AgentState) -> dict:
    """Node 1: Input Validation — validates borrower data against schema."""
    logger.info("🔍 [Node 1] Validating borrower input...")

    try:
        from agent.schema import validate_input

        input_data = state["input"]
        validated_df = validate_input(input_data)
        validated_dict = validated_df.to_dict(orient="records")[0]

        logger.info("✅ Input validation passed — %d features", len(validated_dict))

        return {
            "validated_data": validated_dict,
            "steps_completed": (state.get("steps_completed") or []) + ["validate"],
        }

    except Exception as e:
        logger.error("❌ Input validation failed: %s", e)
        return {
            "error": f"Validation failed: {e}",
            "steps_completed": (state.get("steps_completed") or []) + ["validate:FAILED"],
        }


def predict_node(state: AgentState) -> dict:
    """Node 2: Risk Prediction — ML model inference + risk tier classification."""
    logger.info("🧠 [Node 2] Running ML risk prediction...")

    if state.get("error"):
        logger.warning("⚠️  Skipping prediction — upstream error")
        return {}

    try:
        import pandas as pd
        from agent.model_loader import predict_risk
        from agent.llm_reasoner import BORDERLINE_RANGE

        validated_df = pd.DataFrame([state["validated_data"]])
        risk = predict_risk(validated_df, model_name="logistic")

        # ── Classify risk tier for conditional routing ────────────────
        probability = risk["probability"]
        low_threshold, high_threshold = BORDERLINE_RANGE

        if probability < low_threshold:
            risk_tier = "clear_low"
        elif probability > high_threshold:
            risk_tier = "clear_high"
        else:
            risk_tier = "borderline"

        logger.info(
            "✅ Prediction: %s (p=%.4f) → tier: %s",
            risk["label"], probability, risk_tier,
        )

        return {
            "risk": risk,
            "risk_tier": risk_tier,
            "steps_completed": (state.get("steps_completed") or []) + ["predict"],
        }

    except Exception as e:
        logger.error("❌ Risk prediction failed: %s", e)
        return {
            "error": f"Prediction failed: {e}",
            "risk": {"prediction": -1, "label": "Unknown", "probability": 0.0, "model_used": "error"},
            "risk_tier": "borderline",
            "steps_completed": (state.get("steps_completed") or []) + ["predict:FAILED"],
        }


# ── Conditional Router: Risk Tier ────────────────────────────────────────────

def risk_router(state: AgentState) -> str:
    """
    Conditional edge: route based on risk tier.

    - clear_low / clear_high → standard path (quick analysis)
    - borderline → deep path (extended RAG retrieval + more factors)
    """
    tier = state.get("risk_tier", "borderline")

    if tier == "borderline":
        logger.info("🔀 Risk router → DEEP analysis path (borderline case)")
        return "deep"
    else:
        logger.info("🔀 Risk router → STANDARD path (%s)", tier)
        return "standard"


def explain_standard_node(state: AgentState) -> dict:
    """Node 3a: Standard risk explanation — top 5 features."""
    logger.info("📊 [Node 3a] Standard risk explanation (top 5 features)...")

    if state.get("error"):
        return {"explanation": []}

    try:
        import pandas as pd
        from agent.model_loader import load_model
        from agent.risk_explainer import explain_risk

        pipeline = load_model("decision_tree")
        validated_df = pd.DataFrame([state["validated_data"]])
        explanation = explain_risk(pipeline, validated_df, model_name="decision_tree", top_n=5)

        logger.info("✅ %d risk factors identified", len(explanation))

        return {
            "explanation": explanation,
            "steps_completed": (state.get("steps_completed") or []) + ["explain_standard"],
        }

    except Exception as e:
        logger.error("❌ Standard explanation failed: %s", e)
        return {
            "explanation": [],
            "steps_completed": (state.get("steps_completed") or []) + ["explain_standard:FAILED"],
        }


def explain_deep_node(state: AgentState) -> dict:
    """
    Node 3b: Deep risk explanation — top 8 features + both models.

    Triggered for borderline cases where more detail helps the LLM reason.
    """
    logger.info("📊 [Node 3b] DEEP risk explanation (top 8 features, both models)...")

    if state.get("error"):
        return {"explanation": []}

    try:
        import pandas as pd
        from agent.model_loader import load_model
        from agent.risk_explainer import explain_risk

        validated_df = pd.DataFrame([state["validated_data"]])

        # Get explanations from BOTH models for a richer picture
        dt_pipeline = load_model("decision_tree")
        dt_factors = explain_risk(dt_pipeline, validated_df, model_name="decision_tree", top_n=8)

        try:
            lr_pipeline = load_model("logistic")
            lr_factors = explain_risk(lr_pipeline, validated_df, model_name="logistic", top_n=5)
            # Tag logistic factors with their model source
            for f in lr_factors:
                f["model_source"] = "logistic_regression"
        except Exception:
            lr_factors = []

        # Merge: DT factors first (primary model), then LR factors for contrast
        combined = dt_factors + [f for f in lr_factors if f["feature"] not in
                                  {d["feature"] for d in dt_factors}]

        logger.info("✅ DEEP: %d factors (%d DT + %d LR unique)",
                     len(combined), len(dt_factors), len(combined) - len(dt_factors))

        return {
            "explanation": combined,
            "steps_completed": (state.get("steps_completed") or []) + ["explain_deep"],
        }

    except Exception as e:
        logger.error("❌ Deep explanation failed: %s", e)
        return {
            "explanation": [],
            "steps_completed": (state.get("steps_completed") or []) + ["explain_deep:FAILED"],
        }


def query_node(state: AgentState) -> dict:
    """Node 4: Build RAG query from risk + explanation."""
    logger.info("🔧 [Node 4] Building RAG query...")

    try:
        from agent.llm_reasoner import build_query

        risk = state.get("risk") or {}
        explanation = state.get("explanation") or []
        query = build_query(risk, explanation)

        logger.info("✅ Query: '%s'", query[:80])

        return {
            "query": query,
            "steps_completed": (state.get("steps_completed") or []) + ["query"],
        }

    except Exception as e:
        logger.error("❌ Query build failed: %s", e)
        return {
            "query": "credit risk borrower loan approval regulations guidelines",
            "steps_completed": (state.get("steps_completed") or []) + ["query:FALLBACK"],
        }


def rag_node(state: AgentState) -> dict:
    """Node 5a: Standard RAG retrieval — top 5 chunks."""
    logger.info("📚 [Node 5] Retrieving regulatory documents (top 5)...")

    try:
        from agent.rag import retrieve_docs

        query = state.get("query", "credit risk guidelines")
        documents = retrieve_docs(
            query=query,
            top_k=5,
            index_dir="rag/faiss_index",
            fail_on_empty=False,
        )

        logger.info("✅ Retrieved %d chunks (top score: %.4f)",
                     len(documents), documents[0]["score"] if documents else 0.0)

        return {
            "documents": documents,
            "steps_completed": (state.get("steps_completed") or []) + (["rag"] if documents else ["rag:EMPTY"]),
        }

    except FileNotFoundError:
        logger.warning("⚠️  FAISS index not found — no regulatory context")
        return {
            "documents": [],
            "steps_completed": (state.get("steps_completed") or []) + ["rag:NO_INDEX"],
        }

    except Exception as e:
        logger.error("❌ RAG retrieval failed: %s", e)
        return {
            "documents": [],
            "steps_completed": (state.get("steps_completed") or []) + ["rag:FAILED"],
        }


def rag_deep_node(state: AgentState) -> dict:
    """
    Node 5b: Deep RAG retrieval — top 8 chunks for borderline cases.

    More regulatory context helps the LLM make better borderline decisions.
    """
    logger.info("📚 [Node 5b] DEEP RAG retrieval (top 8 chunks for borderline case)...")

    try:
        from agent.rag import retrieve_docs

        query = state.get("query", "credit risk guidelines")
        documents = retrieve_docs(
            query=query,
            top_k=8,
            index_dir="rag/faiss_index",
            fail_on_empty=False,
        )

        logger.info("✅ DEEP: Retrieved %d chunks (top score: %.4f)",
                     len(documents), documents[0]["score"] if documents else 0.0)

        return {
            "documents": documents,
            "steps_completed": (state.get("steps_completed") or []) + (["rag_deep"] if documents else ["rag_deep:EMPTY"]),
        }

    except FileNotFoundError:
        logger.warning("⚠️  FAISS index not found — no regulatory context")
        return {
            "documents": [],
            "steps_completed": (state.get("steps_completed") or []) + ["rag_deep:NO_INDEX"],
        }

    except Exception as e:
        logger.error("❌ Deep RAG retrieval failed: %s", e)
        return {
            "documents": [],
            "steps_completed": (state.get("steps_completed") or []) + ["rag_deep:FAILED"],
        }


def decision_node(state: AgentState) -> dict:
    """
    Node 6: LLM Decision Generation + Confidence Scoring.

    Generates a lending decision and computes a multi-signal confidence score
    to determine if the decision needs reflective review.
    """
    logger.info("🤖 [Node 6] Generating lending decision via LLM...")

    if state.get("error"):
        logger.warning("⚠️  Skipping decision generation due to upstream error: %s", state.get("error"))
        return {
            "decision": {},
            "confidence_score": 0.0,
            "reasoning_passes": (state.get("reasoning_passes") or 0) + 1,
            "steps_completed": (state.get("steps_completed") or []) + ["decision:SKIPPED"],
        }

    try:
        from agent.llm_reasoner import generate_decision, compute_confidence_score

        risk = state.get("risk") or {}
        explanation = state.get("explanation") or []
        documents = state.get("documents") or []

        decision = generate_decision(risk, explanation, documents)

        # Compute multi-signal confidence score
        confidence_score = compute_confidence_score(decision, risk)

        current_passes = (state.get("reasoning_passes") or 0) + 1

        logger.info(
            "✅ Decision: %s | Confidence: %.2f | Score: %.4f | Pass: %d",
            decision.get("Lending Decision"), float(decision.get("Confidence", 0.0)),
            confidence_score, current_passes,
        )

        return {
            "decision": decision,
            "confidence_score": confidence_score,
            "reasoning_passes": current_passes,
            "steps_completed": (state.get("steps_completed") or []) + ["decision"],
        }

    except Exception as e:
        logger.error("❌ Decision generation failed: %s", e)
        risk = state.get("risk") or {}
        probability = risk.get("probability", 0.0)
        return {
            "decision": {
                "Borrower Profile Summary": "Unable to generate full analysis due to an error.",
                "Risk Analysis": f"ML model predicted {risk.get('label', 'Unknown')} risk.",
                "Lending Decision": "REJECT" if probability >= 0.65 else ("APPROVE" if probability <= 0.30 else "CONDITIONAL"),
                "Confidence": 0.2,
                "Regulatory References": [],
                "Disclaimer": "AI-generated recommendation in fallback mode. Human review mandatory.",
            },
            "confidence_score": 0.25,
            "reasoning_passes": (state.get("reasoning_passes") or 0) + 1,
            "steps_completed": (state.get("steps_completed") or []) + ["decision:FALLBACK"],
        }


# ── Conditional Router: Confidence ───────────────────────────────────────────

def confidence_router(state: AgentState) -> str:
    """
    Conditional edge: decide whether to accept the decision or reflect.

    - High confidence (≥ threshold) → accept and finish
    - Low confidence (< threshold) → reflect for iterative review
    - Already at max passes → accept regardless (prevent infinite loops)
    """
    confidence_score = state.get("confidence_score", 0.0)
    passes = state.get("reasoning_passes", 1)
    max_passes = state.get("max_passes", 2)

    if passes >= max_passes:
        logger.info(
            "🔀 Confidence router → ACCEPT (max passes %d/%d reached, score=%.4f)",
            passes, max_passes, confidence_score,
        )
        return "accept"

    if confidence_score >= CONFIDENCE_ACCEPT_THRESHOLD:
        logger.info(
            "🔀 Confidence router → ACCEPT (score %.4f ≥ threshold %.2f)",
            confidence_score, CONFIDENCE_ACCEPT_THRESHOLD,
        )
        return "accept"
    else:
        logger.info(
            "🔀 Confidence router → REFLECT (score %.4f < threshold %.2f, pass %d/%d)",
            confidence_score, CONFIDENCE_ACCEPT_THRESHOLD, passes, max_passes,
        )
        return "reflect"


def reflect_node(state: AgentState) -> dict:
    """
    Node 7: Reflective Re-evaluation.

    Triggered when the initial decision has low confidence. The LLM
    re-examines the decision with a critical "reviewer" persona,
    checking for logical gaps and missed evidence.
    """
    logger.info("🔄 [Node 7] Reflective re-evaluation (pass %d)...",
                (state.get("reasoning_passes") or 1))

    try:
        from agent.llm_reasoner import reflect_on_decision, compute_confidence_score

        initial_decision = state.get("decision") or {}
        risk = state.get("risk") or {}
        explanation = state.get("explanation") or []
        documents = state.get("documents") or []

        # Run reflection
        revised_decision = reflect_on_decision(
            initial_decision, risk, explanation, documents,
        )

        # Re-compute confidence score for the revised decision
        new_confidence = compute_confidence_score(revised_decision, risk)
        new_passes = (state.get("reasoning_passes") or 1) + 1

        # Log whether the decision was upheld or overridden
        original = initial_decision.get("Lending Decision", "Unknown")
        revised = revised_decision.get("Lending Decision", "Unknown")
        if original != revised:
            logger.info(
                "🔄 Decision OVERRIDDEN: %s → %s (new confidence: %.4f)",
                original, revised, new_confidence,
            )
        else:
            logger.info(
                "🔄 Decision UPHELD: %s (new confidence: %.4f)",
                revised, new_confidence,
            )

        return {
            "decision": revised_decision,
            "confidence_score": new_confidence,
            "reasoning_passes": new_passes,
            "steps_completed": (state.get("steps_completed") or []) + ["reflect"],
        }

    except Exception as e:
        logger.error("❌ Reflection failed: %s", e)
        return {
            "steps_completed": (state.get("steps_completed") or []) + ["reflect:FAILED"],
            "reasoning_passes": (state.get("reasoning_passes") or 1) + 1,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════


def build_agent() -> Any:
    """
    Construct and compile the advanced LangGraph agent workflow.

    Graph topology:

        START → validate → predict → risk_router
                                         │
                          ┌──────────────┴──────────────┐
                          ▼                              ▼
                  explain_standard                explain_deep
                          │                              │
                        query                          query
                          │                              │
                         rag                          rag_deep
                          │                              │
                          └──────────────┬──────────────┘
                                         ▼
                                      decision
                                         │
                                  confidence_router
                                   │             │
                                   ▼             ▼
                                 (accept)    (reflect)
                                   │             │
                                   ▼             ▼
                                  END           END

    Returns
    -------
    CompiledGraph
        A compiled LangGraph with conditional routing and reflection.
    """
    graph = StateGraph(AgentState)

    # ── Register all nodes ───────────────────────────────────────────────
    graph.add_node("validate", validate_node)
    graph.add_node("predict", predict_node)
    graph.add_node("explain_standard", explain_standard_node)
    graph.add_node("explain_deep", explain_deep_node)
    graph.add_node("query", query_node)
    graph.add_node("rag", rag_node)
    graph.add_node("rag_deep", rag_deep_node)
    graph.add_node("decision", decision_node)
    graph.add_node("reflect", reflect_node)

    # ── Sequential edges ─────────────────────────────────────────────────
    graph.add_edge(START, "validate")
    graph.add_edge("validate", "predict")

    # ── Conditional: Risk-based routing ──────────────────────────────────
    graph.add_conditional_edges(
        "predict",
        risk_router,
        {
            "standard": "explain_standard",
            "deep": "explain_deep",
        },
    )

    # ── Standard path ────────────────────────────────────────────────────
    graph.add_edge("explain_standard", "query")
    graph.add_edge("query", "rag")
    graph.add_edge("rag", "decision")

    # ── Deep path (borderline cases) ─────────────────────────────────────
    graph.add_edge("explain_deep", "query")  # Reuse same query node
    # But we override: deep explain goes to query, which goes to rag_deep
    # Actually, since query is shared, we need deep path to go to rag_deep
    # Let's handle this: after query, if deep path was taken, go to rag_deep
    # We'll route after explain_deep → query is shared, then we need another router.
    # Simpler: explain_deep → query_deep (same logic) → rag_deep → decision
    # But we want to reuse query_node. Let's use a conditional after query.

    # Actually, the simplest clean approach: both paths converge at "query",
    # then use a conditional edge from "query" to decide rag vs rag_deep.

    # Remove the direct edge from query → rag and replace with conditional
    # We need to restructure slightly. Let me do it properly:

    # Re-do: remove conflicting edges and use proper conditional routing

    # The graph builder is additive, so let me restructure:
    # Instead of shared query node, let's keep it simple and correct.

    # ── Conditional: After decision, check confidence ────────────────────
    graph.add_conditional_edges(
        "decision",
        confidence_router,
        {
            "accept": END,
            "reflect": "reflect",
        },
    )

    # ── Reflection always ends ───────────────────────────────────────────
    graph.add_edge("reflect", "decision")

    # ── Compile ──────────────────────────────────────────────────────────
    compiled = graph.compile()
    logger.info(
        "🏗️  Advanced agent compiled: 9 nodes, conditional routing + reflection loop"
    )

    return compiled


def _rebuild_clean_graph() -> Any:
    """
    Build the graph with clean, non-conflicting edges.

    This is the actual graph builder that handles the fact that
    both standard and deep paths need to converge at the decision node.
    """
    graph = StateGraph(AgentState)

    # ── Register all nodes ───────────────────────────────────────────────
    graph.add_node("validate", validate_node)
    graph.add_node("predict", predict_node)
    graph.add_node("explain_standard", explain_standard_node)
    graph.add_node("explain_deep", explain_deep_node)
    graph.add_node("query", query_node)
    graph.add_node("rag", rag_node)
    graph.add_node("rag_deep", rag_deep_node)
    graph.add_node("decision", decision_node)
    graph.add_node("reflect", reflect_node)

    # ── Entry ────────────────────────────────────────────────────────────
    graph.add_edge(START, "validate")
    graph.add_edge("validate", "predict")

    # ── Conditional: Risk tier routing ───────────────────────────────────
    graph.add_conditional_edges(
        "predict",
        risk_router,
        {
            "standard": "explain_standard",
            "deep": "explain_deep",
        },
    )

    # ── Standard path: explain → query → rag → decision ─────────────────
    graph.add_edge("explain_standard", "query")

    # ── Deep path: explain_deep → query (shared) ────────────────────────
    graph.add_edge("explain_deep", "query")

    # ── After query, route based on risk tier for RAG depth ──────────────
    def rag_router(state: AgentState) -> str:
        tier = state.get("risk_tier", "borderline")
        if tier == "borderline":
            return "deep"
        return "standard"

    graph.add_conditional_edges(
        "query",
        rag_router,
        {
            "standard": "rag",
            "deep": "rag_deep",
        },
    )

    # ── Both RAG paths converge at decision ──────────────────────────────
    graph.add_edge("rag", "decision")
    graph.add_edge("rag_deep", "decision")

    # ── Conditional: Confidence-based routing ────────────────────────────
    graph.add_conditional_edges(
        "decision",
        confidence_router,
        {
            "accept": END,
            "reflect": "reflect",
        },
    )

    # ── Reflection loops back to decision (for re-scoring) ───────────────
    graph.add_edge("reflect", "decision")

    # ── Compile ──────────────────────────────────────────────────────────
    compiled = graph.compile()
    logger.info(
        "🏗️  Advanced agent compiled: 9 nodes, 2 conditional routers, reflection loop"
    )

    return compiled


# Override build_agent to use the clean version
def build_agent() -> Any:
    """Build and compile the advanced LangGraph agent with conditional routing."""
    return _rebuild_clean_graph()


# ═══════════════════════════════════════════════════════════════════════════
#  RUN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════


def run_agent(input_data: dict, debug: bool = False, max_passes: int = 2) -> dict:
    """
    Execute the full lending decision agent for a borrower profile.

    Parameters
    ----------
    input_data : dict
        Raw borrower profile data with all 11 feature fields.
    debug : bool
        If True, prints intermediate state after each node.
    max_passes : int
        Maximum reasoning passes (1 = single-pass, 2+ = reflective).
        Default 2 allows one reflection if confidence is low.

    Returns
    -------
    dict
        Final agent state containing:
        - decision: Structured lending decision
        - confidence_score: Numeric confidence (0-1)
        - risk_tier: "clear_low", "borderline", or "clear_high"
        - reasoning_passes: How many LLM reasoning passes were made
        - steps_completed: Full audit trail of executed nodes
    """
    logger.info("=" * 60)
    logger.info("🚀 Starting Credit Risk Agent Workflow (Advanced)")
    logger.info("=" * 60)

    start_time = time.time()
    agent = build_agent()

    # ── Initialize state ─────────────────────────────────────────────────
    initial_state: AgentState = {
        "input": input_data,
        "validated_data": None,
        "risk": None,
        "explanation": None,
        "query": None,
        "documents": None,
        "decision": None,
        "risk_tier": None,
        "confidence_score": None,
        "reasoning_passes": 0,
        "max_passes": max_passes,
        "error": None,
        "steps_completed": [],
    }

    # ── Execute ──────────────────────────────────────────────────────────
    if debug:
        logger.info("🐛 Debug mode — streaming intermediate states")
        final_state = initial_state.copy()

        for step_output in agent.stream(initial_state):
            for node_name, state_update in step_output.items():
                final_state.update(state_update)
                logger.info("─── 🐛 After '%s' ───", node_name)
                _debug_print_state(node_name, final_state)
    else:
        final_state = agent.invoke(initial_state)

    # ── Finalize ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    steps = final_state.get("steps_completed", [])

    logger.info("=" * 60)
    logger.info(
        "✅ Agent complete in %.2fs | %d steps | %d reasoning passes",
        elapsed, len(steps), final_state.get("reasoning_passes", 0),
    )
    logger.info("   Path: %s", " → ".join(steps))
    logger.info("   Risk tier: %s", final_state.get("risk_tier", "N/A"))
    logger.info("   Confidence: %.4f", final_state.get("confidence_score", 0.0))

    if final_state.get("decision"):
        decision = final_state["decision"]
        logger.info("   Decision: %s (%.2f confidence)",
                     decision.get("Lending Decision"), float(decision.get("Confidence", 0.0)))
        if decision.get("Reflection"):
            logger.info("   Reflection: %s", str(decision.get("Reflection"))[:100])

    logger.info("=" * 60)

    return final_state


def _debug_print_state(node_name: str, state: dict):
    """Pretty-print relevant state fields after a node executes."""
    import json

    node_outputs = {
        "validate": ["validated_data"],
        "predict": ["risk", "risk_tier"],
        "explain_standard": ["explanation"],
        "explain_deep": ["explanation"],
        "query": ["query"],
        "rag": ["documents"],
        "rag_deep": ["documents"],
        "decision": ["decision", "confidence_score", "reasoning_passes"],
        "reflect": ["decision", "confidence_score", "reasoning_passes"],
    }

    fields = node_outputs.get(node_name, [])
    for field in fields:
        value = state.get(field)
        if value is None:
            print(f"  {field}: None")
        elif isinstance(value, (int, float)):
            print(f"  {field}: {value}")
        elif isinstance(value, str):
            print(f"  {field}: \"{value[:100]}{'...' if len(value) > 100 else ''}\"")
        elif isinstance(value, list):
            print(f"  {field}: [{len(value)} items]")
            for i, item in enumerate(value[:2]):
                preview = json.dumps(item, default=str)
                print(f"    [{i}]: {preview[:120]}{'...' if len(preview) > 120 else ''}")
            if len(value) > 2:
                print(f"    ... +{len(value) - 2} more")
        elif isinstance(value, dict):
            preview = json.dumps(value, indent=2, default=str)
            if len(preview) > 300:
                preview = preview[:300] + "\n    ..."
            print(f"  {field}: {preview}")

    if state.get("error"):
        print(f"  ⚠️  error: {state['error']}")

    print()


def _normalize_workflow_output(input_data: dict, final_state: dict) -> WorkflowOutput:
    """Normalize graph state into a stable output payload for app integration."""
    return {
        "input": input_data,
        "validated_data": final_state.get("validated_data"),
        "risk": final_state.get("risk"),
        "explanation": final_state.get("explanation") or [],
        "query": final_state.get("query"),
        "documents": final_state.get("documents") or [],
        "decision": final_state.get("decision") or {},
        "risk_tier": final_state.get("risk_tier"),
        "confidence_score": float(final_state.get("confidence_score") or 0.0),
        "reasoning_passes": int(final_state.get("reasoning_passes") or 0),
        "steps_completed": final_state.get("steps_completed") or [],
        "error": final_state.get("error"),
    }


def run_workflow(input_data: dict, debug: bool = False, max_passes: int = 2) -> WorkflowOutput:
    """
    Stable workflow entrypoint for application callers.

    This function executes the compiled LangGraph and returns a normalized,
    structured payload suitable for UI/API consumers.
    """
    final_state = run_agent(input_data=input_data, debug=debug, max_passes=max_passes)
    return _normalize_workflow_output(input_data=input_data, final_state=final_state)
