"""
LLM Reasoning Node — Credit Risk Decision Engine
===================================================

Combines ML risk prediction, feature-based explanation, and retrieved
financial regulations to generate a structured lending decision report.

Uses Groq API with Llama 3 70B as the reasoning backbone. The LLM acts
as a senior credit risk analyst — conservative, source-backed, and
explainable in its recommendations.

Pipeline:
    build_query()  →  retrieve_docs()  →  format_prompt()  →  LLM call  →  parse_response()

Usage:
    >>> from agent.llm_reasoner import generate_decision
    >>> decision = generate_decision(
    ...     risk={"prediction": 1, "label": "High Risk", "probability": 0.87},
    ...     explanation=[{"feature": "loan_percent_income", "importance": 0.42, "direction": "increases_risk"}],
    ...     retrieved_docs=[{"text": "...", "source": "RBI_Guidelines.pdf", "page": 3, "score": 0.82}]
    ... )
"""

import os
import json
import logging
import re
from typing import List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from a deterministic project-root .env path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOTENV_PATH = os.path.join(_PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=_DOTENV_PATH, override=False)

# ── Configuration ────────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.2
GROQ_MAX_TOKENS = 2048

CANONICAL_OUTPUT_FIELDS = (
    "Borrower Profile Summary",
    "Risk Analysis",
    "Lending Decision",
    "Confidence",
    "Regulatory References",
    "Disclaimer",
)

FIELD_ALIASES = {
    "Borrower Profile Summary": ("Borrower Profile Summary", "Profile Summary", "profile_summary"),
    "Risk Analysis": ("Risk Analysis", "risk_analysis"),
    "Lending Decision": ("Lending Decision", "Decision", "decision", "lending_decision"),
    "Confidence": ("Confidence", "Confidence Score", "confidence_score"),
    "Regulatory References": (
        "Regulatory References",
        "Regulatory Sources",
        "regulatory_references",
        "regulatory_sources",
    ),
    "Disclaimer": ("Disclaimer", "disclaimer"),
}

SUPPORTED_LENDING_DECISIONS = {"APPROVE", "REJECT", "CONDITIONAL"}

# ── System Prompt ────────────────────────────────────────────────────────────
# Defines the LLM's role and behavioral constraints.

SYSTEM_PROMPT = """You are a senior credit risk analyst working in a bank's lending division.

Your role is to evaluate borrower profiles and provide structured lending decisions.

RULES YOU MUST FOLLOW:
1. Use the ML risk prediction as the PRIMARY signal for your decision.
2. Use retrieved financial regulations as SUPPORTING EVIDENCE — cite them where relevant.
3. Provide logical, explainable reasoning that a compliance officer could audit.
4. Be CONSERVATIVE in lending decisions — when in doubt, err on the side of caution.
5. NEVER fabricate regulations, statistics, or sources. Only reference what is provided.
6. If the ML model flags a borrower as High Risk with probability > 0.5, your default stance should be to Reject unless there are strong mitigating factors.
7. Always structure your output as valid JSON matching the required format exactly.

You must respond ONLY with a JSON object — no markdown, no extra text, no code blocks."""


# ── Output JSON Schema ──────────────────────────────────────────────────────
# The exact structure the LLM must return.

OUTPUT_SCHEMA = {
    "Borrower Profile Summary": "Brief summary of the borrower's profile",
    "Risk Analysis": "Risk analysis aligned with ML prediction and regulations",
    "Lending Decision": "APPROVE or REJECT or CONDITIONAL",
    "Confidence": "Numeric confidence score between 0.0 and 1.0",
    "Regulatory References": ["List of regulation documents referenced"],
    "Disclaimer": "This is an AI-generated recommendation and should not be used as the sole basis for lending decisions. Final approval must involve human review and comply with all applicable regulations.",
}

REQUIRED_OUTPUT_FIELDS = list(OUTPUT_SCHEMA.keys())

# ── Confidence Thresholds ────────────────────────────────────────────────────
# Used by the workflow to decide whether to accept or re-evaluate a decision.

CONFIDENCE_LEVELS = {
    "high": 0.85,
    "medium": 0.60,
    "low": 0.35,
}

# Probability ranges that are considered "borderline" and may need deeper review
BORDERLINE_RANGE = (0.30, 0.65)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def _get_groq_client():
    """
    Initialize and return a Groq API client.

    Reads the API key from environment variable GROQ_API_KEY.

    Returns
    -------
    groq.Groq
        Authenticated Groq client instance.

    Raises
    ------
    EnvironmentError
        If GROQ_API_KEY is not set.
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError(
            "groq package is required for LLM reasoning. "
            "Install with: pip install groq"
        )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set. "
            "Add it to your .env file or export it in your shell."
        )

    return Groq(api_key=api_key)


def build_query(risk: dict, explanation: list) -> str:
    """
    Construct a semantically rich query for RAG retrieval.

    Combines the risk level, probability, and top risk factors into a
    natural language query that will match relevant regulatory content.

    Parameters
    ----------
    risk : dict
        Risk prediction result from predict_risk().
    explanation : list
        Risk factors from explain_risk().

    Returns
    -------
    str
        A natural language query for document retrieval.

    Examples
    --------
    >>> query = build_query(
    ...     {"label": "High Risk", "probability": 0.87},
    ...     [{"feature": "loan_percent_income", "importance": 0.42, "direction": "increases_risk"}]
    ... )
    >>> print(query)
    'high risk borrower with 87.0% default probability key risk factors loan percent income ...'
    """
    # Start with the risk level
    risk_label = risk.get("label", "Unknown Risk").lower()
    probability = risk.get("probability", 0.0)
    query_parts = [
        f"{risk_label} borrower",
        f"with {probability * 100:.1f}% default probability",
    ]

    # Add top risk factors (up to 3) for semantic relevance
    if explanation:
        top_factors = explanation[:3]
        factor_names = [
            f["feature"].replace("_", " ") for f in top_factors
        ]
        query_parts.append(f"key risk factors: {', '.join(factor_names)}")

    # Add regulatory context keywords based on risk level
    if probability > 0.5:
        query_parts.append("loan rejection criteria NPA classification guidelines")
    else:
        query_parts.append("loan approval criteria risk management guidelines")

    query = " ".join(query_parts)
    logger.info("Built RAG query: '%s'", query[:100])
    return query


def _format_risk_section(risk: dict) -> str:
    """Format the risk prediction as readable text for the LLM prompt."""
    return (
        f"- Risk Label: {risk.get('label', 'Unknown')}\n"
        f"- Default Probability: {risk.get('probability', 0.0):.1%}\n"
        f"- Prediction (0=No Default, 1=Default): {risk.get('prediction', 'N/A')}\n"
        f"- Model Used: {risk.get('model_used', 'Unknown')}"
    )


def _format_explanation_section(explanation: list) -> str:
    """Format the risk explanation as readable text for the LLM prompt."""
    if not explanation:
        return "No feature-level explanation available."

    lines = []
    for i, factor in enumerate(explanation, 1):
        feature = factor.get("feature", "unknown").replace("_", " ").title()
        importance = factor.get("importance", 0.0)
        direction = factor.get("direction", "unknown")

        # Make the direction human-readable
        direction_text = {
            "increases_risk": "↑ Increases Default Risk",
            "decreases_risk": "↓ Decreases Default Risk",
            "risk_factor": "⚠ Key Risk Factor",
        }.get(direction, direction)

        lines.append(
            f"  {i}. {feature} (importance: {importance:.4f}) — {direction_text}"
        )

    return "\n".join(lines)


def _format_docs_section(retrieved_docs: list) -> str:
    """Format retrieved regulatory documents as context for the LLM prompt."""
    if not retrieved_docs:
        return "No regulatory documents were retrieved. Base your analysis on the ML prediction and risk factors only."

    lines = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.get("source", "Unknown")
        page = doc.get("page", "?")
        score = doc.get("score", 0.0)
        text = doc.get("text", "").strip()

        # Truncate very long chunks to stay within context window
        if len(text) > 600:
            text = text[:600] + "..."

        lines.append(
            f"--- Regulation {i} (Source: {source}, Page: {page}, Relevance: {score:.2f}) ---\n"
            f"{text}"
        )

    return "\n\n".join(lines)


def _normalize_decision_payload(payload: dict) -> dict:
    """Map legacy decision payload keys into the canonical response schema."""
    source = dict(payload or {})
    normalized = {}

    for canonical_field, aliases in FIELD_ALIASES.items():
        value = None
        for alias in aliases:
            if alias in source:
                value = source[alias]
                break
        normalized[canonical_field] = value

    summary = normalized["Borrower Profile Summary"]
    if not isinstance(summary, str) or not summary.strip():
        normalized["Borrower Profile Summary"] = "Information not available."

    risk_analysis = normalized["Risk Analysis"]
    if not isinstance(risk_analysis, str) or not risk_analysis.strip():
        normalized["Risk Analysis"] = "Information not available."

    decision_value = str(normalized["Lending Decision"] or "CONDITIONAL").strip().upper()
    if decision_value == "APPROVED":
        decision_value = "APPROVE"
    elif decision_value == "REJECTED":
        decision_value = "REJECT"
    elif decision_value not in SUPPORTED_LENDING_DECISIONS:
        decision_value = "CONDITIONAL"
    normalized["Lending Decision"] = decision_value

    raw_confidence = normalized["Confidence"]
    if isinstance(raw_confidence, str):
        score_from_label = CONFIDENCE_LEVELS.get(raw_confidence.strip().lower())
        if score_from_label is not None:
            confidence = score_from_label
        else:
            try:
                confidence = float(raw_confidence)
            except ValueError:
                confidence = 0.5
    else:
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.5
    normalized["Confidence"] = max(0.0, min(confidence, 1.0))

    regulatory_references = normalized["Regulatory References"]
    if regulatory_references is None:
        normalized["Regulatory References"] = []
    elif isinstance(regulatory_references, list):
        normalized["Regulatory References"] = [
            str(reference).strip() for reference in regulatory_references if str(reference).strip()
        ]
    elif isinstance(regulatory_references, (tuple, set)):
        normalized["Regulatory References"] = [
            str(reference).strip() for reference in regulatory_references if str(reference).strip()
        ]
    else:
        normalized["Regulatory References"] = [str(regulatory_references).strip()]

    disclaimer = normalized["Disclaimer"]
    if not isinstance(disclaimer, str) or not disclaimer.strip():
        normalized["Disclaimer"] = (
            "This is an AI-generated recommendation and should not be used as the sole basis for lending decisions. "
            "Final approval must involve human review and comply with all applicable regulations."
        )

    return {field: normalized[field] for field in CANONICAL_OUTPUT_FIELDS}


def _normalize_reference_text(text: str) -> str:
    """Normalize reference text for deterministic matching against retrieved chunks."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _validate_regulatory_references(references: list, retrieved_docs: list) -> list:
    """Keep only references grounded in retrieved chunks and rewrite to canonical sources."""
    if not references or not retrieved_docs:
        return ["No verifiable regulatory reference found"]

    normalized_docs = []
    for doc in retrieved_docs:
        source = str(doc.get("source", "")).strip()
        text = str(doc.get("text", "")).strip()
        if not source and not text:
            continue
        normalized_docs.append((source, _normalize_reference_text(source), _normalize_reference_text(text)))

    grounded_references = []
    seen = set()
    for reference in references:
        reference_text = str(reference).strip()
        if not reference_text:
            continue

        normalized_reference = _normalize_reference_text(reference_text)
        canonical_source = None

        for source, normalized_source, normalized_text in normalized_docs:
            if not source:
                continue

            if (
                normalized_reference == normalized_source
                or normalized_reference in normalized_source
                or normalized_source in normalized_reference
                or normalized_reference in normalized_text
            ):
                canonical_source = source
                break

        if canonical_source and canonical_source not in seen:
            grounded_references.append(canonical_source)
            seen.add(canonical_source)

    if not grounded_references:
        return ["No verifiable regulatory reference found"]

    return grounded_references


def _build_user_prompt(risk: dict, explanation: list, retrieved_docs: list) -> str:
    """
    Construct the full user prompt sent to the LLM.

    Assembles all evidence (prediction, explanation, regulations) into a
    structured prompt that guides the LLM to produce the required JSON output.

    Parameters
    ----------
    risk : dict
        Risk prediction result.
    explanation : list
        Risk factor explanations.
    retrieved_docs : list
        Retrieved regulatory document chunks.

    Returns
    -------
    str
        Complete prompt string.
    """
    risk_section = _format_risk_section(risk)
    explanation_section = _format_explanation_section(explanation)
    docs_section = _format_docs_section(retrieved_docs)

    prompt = f"""Analyze the following borrower risk profile and generate a lending decision.

═══════════════════════════════════════════
SECTION 1: ML RISK PREDICTION
═══════════════════════════════════════════
{risk_section}

═══════════════════════════════════════════
SECTION 2: KEY RISK DRIVERS (from ML model)
═══════════════════════════════════════════
{explanation_section}

═══════════════════════════════════════════
SECTION 3: RELEVANT FINANCIAL REGULATIONS
═══════════════════════════════════════════
{docs_section}

═══════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════
Based on ALL the above evidence, produce a lending decision.

Respond with ONLY a valid JSON object in this EXACT format:
{{
    "Borrower Profile Summary": "Brief summary of the borrower's financial profile based on the risk factors",
    "Risk Analysis": "Detailed analysis combining ML prediction with regulatory context",
    "Lending Decision": "APPROVE" or "REJECT" or "CONDITIONAL",
    "Confidence": 0.0,
    "Regulatory References": ["list", "of", "source documents referenced"],
    "Disclaimer": "This is an AI-generated recommendation and should not be used as the sole basis for lending decisions. Final approval must involve human review and comply with all applicable regulations."
}}

IMPORTANT:
- Respond with ONLY the JSON object. No markdown, no code fences, no extra text.
- The "Lending Decision" field must be exactly "APPROVE", "REJECT", or "CONDITIONAL".
- Use "CONDITIONAL" when evidence is borderline and requires manual review/conditions.
- "Confidence" must be numeric and between 0.0 and 1.0.
- The "Regulatory References" field must list actual document names from the regulations provided.
- If no regulations were provided, use an empty list for "Regulatory References"."""

    return prompt


def _extract_json_from_response(raw_text: str) -> dict:
    """
    Parse JSON from the LLM response, handling common formatting issues.

    The LLM may wrap JSON in markdown code blocks or add extra text.
    This function robustly extracts the JSON object.

    Parameters
    ----------
    raw_text : str
        Raw text response from the LLM.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    ValueError
        If no valid JSON can be extracted.
    """
    text = raw_text.strip()

    # Attempt 1: Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Strip markdown code fences (```json ... ``` or ``` ... ```)
    code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Attempt 3: Find the first { ... } block in the text
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from LLM response: {text[:200]}...")


def _build_fallback_response(risk: dict, explanation: list, retrieved_docs: list) -> dict:
    """
    Generate a structured fallback response when the LLM call fails.

    Uses rule-based logic to produce a conservative decision based on
    the ML prediction alone, without LLM reasoning.

    Parameters
    ----------
    risk : dict
        Risk prediction result.
    explanation : list
        Risk factor explanations.
    retrieved_docs : list
        Retrieved regulatory documents.

    Returns
    -------
    dict
        Structured decision report matching the output schema.
    """
    probability = risk.get("probability", 0.0)
    label = risk.get("label", "Unknown")
    prediction = risk.get("prediction", 0)

    # Conservative rule-based decision
    if prediction == 1 or probability > 0.5:
        decision = "REJECT"
    elif probability > 0.35:
        decision = "REJECT"  # Conservative: borderline cases get rejected
    else:
        decision = "APPROVE"

    # Build risk drivers summary
    if explanation:
        drivers = "; ".join(
            f"{f['feature'].replace('_', ' ').title()} "
            f"(importance: {f['importance']:.3f})"
            for f in explanation[:3]
        )
    else:
        drivers = "No detailed risk factor breakdown available."

    # Collect sources
    sources = list({doc.get("source", "Unknown") for doc in (retrieved_docs or [])})

    fallback = {
        "Borrower Profile Summary": (
            f"Borrower classified as {label} with a {probability:.1%} "
            f"probability of default based on ML model analysis."
        ),
        "Risk Analysis": (
            f"The ML model predicts a {probability:.1%} default probability. "
            f"This assessment is based on the borrower's financial profile "
            f"including income, loan characteristics, and credit history. "
            f"{'The elevated risk level warrants caution.' if probability > 0.3 else 'The risk level is within acceptable bounds.'}"
        ),
        "Lending Decision": decision,
        "Confidence": 0.4 if decision == "CONDITIONAL" else 0.35,
        "Regulatory References": sources if sources else [],
        "Disclaimer": (
            "This is an AI-generated recommendation and should not be used as the "
            "sole basis for lending decisions. Final approval must involve human review "
            "and comply with all applicable regulations. "
            "[FALLBACK MODE: LLM reasoning was unavailable; this decision is based on "
            "rule-based logic applied to the ML prediction.]"
        ),
    }
    return _finalize_decision_output(fallback, risk, retrieved_docs)


def _ml_recommended_decision(risk: dict) -> str:
    """Derive a deterministic recommendation from ML prediction and probability."""
    prediction = int(risk.get("prediction", 0))
    probability = float(risk.get("probability", 0.0))

    # Conservative triage policy aligned with end-sem requirements.
    if prediction == 1 and probability >= BORDERLINE_RANGE[1]:
        return "REJECT"
    if prediction == 0 and probability <= BORDERLINE_RANGE[0]:
        return "APPROVE"
    return "CONDITIONAL"


def _reconcile_decision_with_ml(decision: dict, risk: dict) -> dict:
    """
    Enforce consistency between LLM decision and ML primary signal.

    This avoids contradictions and guarantees production-safe outputs where
    the final decision aligns with model risk scoring policy.
    """
    aligned = dict(decision)
    ml_decision = _ml_recommended_decision(risk)
    current_decision = str(aligned.get("Lending Decision", "CONDITIONAL")).strip().upper()
    if current_decision == "APPROVED":
        current_decision = "APPROVE"
    elif current_decision == "REJECTED":
        current_decision = "REJECT"
    elif current_decision not in SUPPORTED_LENDING_DECISIONS:
        current_decision = "CONDITIONAL"

    if current_decision != ml_decision:
        probability = float(risk.get("probability", 0.0))
        aligned["Lending Decision"] = ml_decision
        aligned["Confidence"] = 0.3 if ml_decision != "CONDITIONAL" else 0.4
        aligned["Risk Analysis"] = (
            f"{aligned.get('Risk Analysis', 'Risk analysis available.')} "
            f"Final verdict aligned to ML primary signal (probability={probability:.1%})."
        )

    return aligned


def _finalize_decision_output(decision: dict, risk: dict, retrieved_docs: list) -> dict:
    """Return a schema-complete, ML-aligned, JSON-safe decision payload."""
    validated = _validate_decision_output(decision)
    validated["Regulatory References"] = _validate_regulatory_references(
        validated.get("Regulatory References") or [],
        retrieved_docs,
    )
    aligned = _reconcile_decision_with_ml(validated, risk)

    # Round-trip through JSON to guarantee serializable, valid JSON-compatible output.
    return json.loads(json.dumps(aligned, ensure_ascii=False, default=str))


def _validate_decision_output(parsed: dict) -> dict:
    """
    Validate and normalize the parsed LLM output against the required schema.

    Ensures all required fields are present and the Decision field contains
    a valid value. Fills in defaults for missing fields.

    Parameters
    ----------
    parsed : dict
        Parsed JSON from LLM response.

    Returns
    -------
    dict
        Validated and normalized decision report.
    """
    normalized = _normalize_decision_payload(parsed)

    # Keep only required contract keys to enforce exact output format.
    normalized = {field: normalized[field] for field in CANONICAL_OUTPUT_FIELDS}

    # Normalize the Lending Decision field.
    decision = str(normalized["Lending Decision"]).strip().upper()
    if decision == "APPROVED":
        decision = "APPROVE"
    elif decision == "REJECTED":
        decision = "REJECT"
    elif decision not in SUPPORTED_LENDING_DECISIONS:
        logger.warning(
            "Invalid Lending Decision value '%s' — defaulting to 'CONDITIONAL'",
            normalized["Lending Decision"],
        )
        decision = "CONDITIONAL"
    normalized["Lending Decision"] = decision

    # Normalize Confidence to numeric range [0.0, 1.0]
    raw_confidence = normalized.get("Confidence", 0.5)
    if isinstance(raw_confidence, str):
        score_from_label = CONFIDENCE_LEVELS.get(raw_confidence.strip().lower())
        if score_from_label is not None:
            confidence_score = score_from_label
        else:
            try:
                confidence_score = float(raw_confidence)
            except ValueError:
                confidence_score = 0.5
    else:
        try:
            confidence_score = float(raw_confidence)
        except (TypeError, ValueError):
            confidence_score = 0.5

    if not (0.0 <= confidence_score <= 1.0):
        logger.warning(
            "Invalid Confidence value '%s' — clamping to [0.0, 1.0]",
            raw_confidence,
        )
    normalized["Confidence"] = max(0.0, min(confidence_score, 1.0))

    # Ensure Regulatory References is a list
    if not isinstance(normalized["Regulatory References"], list):
        normalized["Regulatory References"] = [str(normalized["Regulatory References"])]

    references = []
    for reference in normalized["Regulatory References"]:
        reference_text = str(reference).strip()
        if reference_text and reference_text not in references:
            references.append(reference_text)
    normalized["Regulatory References"] = references

    # Ensure text fields are non-empty strings.
    for field in ["Borrower Profile Summary", "Risk Analysis", "Disclaimer"]:
        value = normalized.get(field, "")
        if not isinstance(value, str) or not value.strip():
            normalized[field] = "Information not available."

    return normalized


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════


def generate_decision(
    risk: dict,
    explanation: list,
    retrieved_docs: list,
    model: str = GROQ_MODEL,
    temperature: float = GROQ_TEMPERATURE,
    max_tokens: int = GROQ_MAX_TOKENS,
) -> dict:
    """
    Generate a structured lending decision by combining ML prediction,
    risk explanation, and retrieved regulations through LLM reasoning.

    This is the core reasoning node of the agentic pipeline. It:
    1. Formats all evidence into a structured prompt
    2. Calls the Groq LLM (Llama 3 70B) for reasoning
    3. Parses and validates the JSON response
    4. Falls back to rule-based logic on failure

    Parameters
    ----------
    risk : dict
        Risk prediction from predict_risk():
        {"prediction": int, "label": str, "probability": float, "model_used": str}
    explanation : list
        Risk factors from explain_risk():
        [{"feature": str, "importance": float, "direction": str}, ...]
    retrieved_docs : list
        Retrieved regulation chunks from retrieve_docs():
        [{"text": str, "source": str, "page": int, "score": float}, ...]
    model : str
        Groq model identifier (default: "llama3-70b-8192").
    temperature : float
        Sampling temperature (default: 0.2 for conservative outputs).
    max_tokens : int
        Maximum tokens in response (default: 2048).

    Returns
    -------
    dict
        Structured lending decision report:
        {
            "Borrower Profile Summary": str,
            "Risk Analysis": str,
            "Lending Decision": "APPROVE" or "REJECT" or "CONDITIONAL",
            "Confidence": float,
            "Regulatory References": List[str],
            "Disclaimer": str
        }
    """
    # ── Sanitize inputs (robustness) ─────────────────────────────────────
    if risk is None:
        risk = {"prediction": -1, "label": "Unknown", "probability": 0.0, "model_used": "none"}
    if explanation is None:
        explanation = []
    if retrieved_docs is None:
        retrieved_docs = []

    # ── Build prompt ─────────────────────────────────────────────────────
    user_prompt = _build_user_prompt(risk, explanation, retrieved_docs)
    logger.info(
        "Calling LLM (model=%s, temp=%.2f) with %d risk factors and %d retrieved docs",
        model, temperature, len(explanation), len(retrieved_docs),
    )

    # Always attempt LLM reasoning whenever an API key exists.
    has_api_key = bool(os.environ.get("GROQ_API_KEY"))

    if not has_api_key:
        logger.warning("GROQ_API_KEY not found — using fallback response")
        return _build_fallback_response(risk, explanation, retrieved_docs)

    # ── Call LLM ─────────────────────────────────────────────────────────
    try:
        client = _get_groq_client()

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            stream=False,
        )

        raw_response = chat_completion.choices[0].message.content
        logger.info("LLM response received (%d chars)", len(raw_response))
        logger.debug("Raw LLM response: %s", raw_response[:500])

    except Exception as e:
        logger.error("LLM call failed: %s — using fallback response", e)
        return _build_fallback_response(risk, explanation, retrieved_docs)

    # ── Parse response ───────────────────────────────────────────────────
    try:
        parsed = _extract_json_from_response(raw_response)
        validated = _finalize_decision_output(parsed, risk, retrieved_docs)
        logger.info(
            "Lending Decision: %s (Confidence: %.2f)",
            validated["Lending Decision"],
            float(validated.get("Confidence", 0.0)),
        )
        return validated

    except (ValueError, json.JSONDecodeError) as e:
        logger.error(
            "Failed to parse LLM response as JSON: %s — using fallback", e
        )
        return _build_fallback_response(risk, explanation, retrieved_docs)


def compute_confidence_score(decision: dict, risk: dict) -> float:
    """
    Compute a numeric confidence score (0.0–1.0) for the lending decision.

    Combines three signals:
    1. LLM self-reported confidence (High/Medium/Low)
    2. ML probability alignment (does the LLM agree with the ML model?)
    3. Borderline detection (is the ML probability in the ambiguous zone?)

    Parameters
    ----------
    decision : dict
        Structured decision from generate_decision().
    risk : dict
        Risk prediction from predict_risk().

    Returns
    -------
    float
        Confidence score between 0.0 (no confidence) and 1.0 (full confidence).
    """
    score = 0.0

    decision = _normalize_decision_payload(decision)

    # ── Signal 1: Model-reported Confidence Score (40% weight) ───────────
    try:
        llm_confidence_score = float(decision.get("Confidence", 0.5))
    except (TypeError, ValueError):
        llm_confidence_score = 0.5
    llm_confidence_score = max(0.0, min(llm_confidence_score, 1.0))
    score += 0.40 * llm_confidence_score

    # ── Signal 2: ML-LLM alignment (35% weight) ─────────────────────────
    # If the LLM decision aligns with what the ML probability suggests,
    # confidence is higher.
    probability = risk.get("probability", 0.5)
    llm_decision = decision.get("Lending Decision", "REJECT")

    if llm_decision == "REJECT" and probability >= BORDERLINE_RANGE[1]:
        alignment = 1.0  # Both say risky
    elif llm_decision == "APPROVE" and probability <= BORDERLINE_RANGE[0]:
        alignment = 1.0  # Both say safe
    elif llm_decision == "CONDITIONAL" and BORDERLINE_RANGE[0] < probability < BORDERLINE_RANGE[1]:
        alignment = 1.0
    elif llm_decision == "CONDITIONAL":
        alignment = 0.5
    else:
        alignment = 0.2  # Contradictory with ML policy
    score += 0.35 * alignment

    # ── Signal 3: Borderline detection (25% weight) ──────────────────────
    # Clear-cut cases (very low or very high probability) get high scores.
    low, high = BORDERLINE_RANGE
    if probability < low or probability > high:
        borderline_score = 1.0  # Clear case
    else:
        # Linear interpolation: center of range = 0.0, edges = 1.0
        center = (low + high) / 2
        distance = abs(probability - center) / (center - low)
        borderline_score = min(distance, 1.0)
    score += 0.25 * borderline_score

    final_score = round(min(max(score, 0.0), 1.0), 4)
    logger.info(
        "Confidence score: %.4f (reported=%.2f, alignment=%.2f, borderline=%.2f)",
        final_score, llm_confidence_score, alignment, borderline_score,
    )
    return final_score


# ── Reflection Prompt ────────────────────────────────────────────────────────

REFLECTION_SYSTEM_PROMPT = """You are a senior credit risk reviewer performing a second-pass review of a lending decision.

Your colleague (another credit analyst) has already made a decision. Your job is to:

1. CRITICALLY EXAMINE the initial decision for logical gaps, missed risks, or unsupported claims.
2. CHECK if the decision is consistent with the ML prediction and regulations cited.
3. CONSIDER if there are alternative interpretations of the evidence.
4. DECIDE whether to UPHOLD or OVERRIDE the initial decision.
5. If you override, provide clear justification for why the original was wrong.
6. Assign a NEW confidence level based on your deeper review.

Be rigorous but fair. Override only if there is a genuine error in reasoning.
Respond with ONLY a valid JSON object — no markdown, no extra text."""


def reflect_on_decision(
    initial_decision: dict,
    risk: dict,
    explanation: list,
    retrieved_docs: list,
    model: str = GROQ_MODEL,
    temperature: float = 0.3,
) -> dict:
    """
    Perform a reflective second-pass review of a lending decision.

    This function implements iterative reasoning — the LLM re-examines its
    own decision with a critical eye, checking for logical gaps and
    consistency with the evidence.

    Parameters
    ----------
    initial_decision : dict
        The first-pass decision from generate_decision().
    risk : dict
        Risk prediction from ML model.
    explanation : list
        Risk factor explanations.
    retrieved_docs : list
        Retrieved regulatory documents.
    model : str
        Groq model to use.
    temperature : float
        Slightly higher temperature (0.3) to allow critical thinking.

    Returns
    -------
    dict
        Revised decision with updated confidence and justification.
        Contains an additional "Reflection" field documenting the review.
    """
    initial_decision = _normalize_decision_payload(initial_decision)

    logger.info("🔄 Reflecting on initial decision: %s (Confidence: %s)",
                initial_decision.get("Lending Decision"), initial_decision.get("Confidence"))

    risk_section = _format_risk_section(risk)
    explanation_section = _format_explanation_section(explanation)
    docs_section = _format_docs_section(retrieved_docs)

    reflection_prompt = f"""You are reviewing a colleague's lending decision. Critically examine it.

═══════════════════════════════════════════
ORIGINAL EVIDENCE
═══════════════════════════════════════════

ML RISK PREDICTION:
{risk_section}

KEY RISK DRIVERS:
{explanation_section}

RELEVANT REGULATIONS:
{docs_section}

═══════════════════════════════════════════
COLLEAGUE'S INITIAL DECISION
═══════════════════════════════════════════

Lending Decision: {initial_decision.get('Lending Decision', 'UNKNOWN')}
Confidence: {initial_decision.get('Confidence', 'Unknown')}
Borrower Profile Summary: {initial_decision.get('Borrower Profile Summary', 'N/A')}
Risk Analysis: {initial_decision.get('Risk Analysis', 'N/A')}

═══════════════════════════════════════════
YOUR REVIEW TASK
═══════════════════════════════════════════

1. Is the initial decision logically consistent with the ML prediction?
2. Are there any risks or mitigating factors the initial analysis missed?
3. Are the cited regulations applied correctly?
4. Should the decision be UPHELD or OVERRIDDEN?

Respond with ONLY a valid JSON object:
{{
    "Borrower Profile Summary": "Updated summary incorporating your review insights",
    "Risk Analysis": "Deeper analysis after second-pass review",
    "Lending Decision": "APPROVE" or "REJECT" or "CONDITIONAL" (your final verdict),
    "Confidence": 0.0,
    "Regulatory References": ["source documents referenced"],
    "Disclaimer": "This is an AI-generated recommendation reviewed through a multi-step reasoning process. Final approval must involve human review and comply with all applicable regulations."
}}"""

    try:
        client = _get_groq_client()

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                {"role": "user", "content": reflection_prompt},
            ],
            model=model,
            temperature=temperature,
            max_tokens=GROQ_MAX_TOKENS,
            top_p=1,
            stream=False,
        )

        raw_response = chat_completion.choices[0].message.content
        logger.info("Reflection response received (%d chars)", len(raw_response))

        parsed = _extract_json_from_response(raw_response)
        validated = _finalize_decision_output(parsed, risk, retrieved_docs)

        logger.info(
            "🔄 Reflection result: %s (Confidence: %.2f) | %s",
            validated["Lending Decision"], float(validated.get("Confidence", 0.0)),
            "UPHELD" if validated["Lending Decision"] == initial_decision.get("Lending Decision") else "OVERRIDDEN",
        )

        return validated

    except Exception as e:
        logger.error("Reflection failed: %s — keeping initial decision", e)
        return _finalize_decision_output(initial_decision, risk, retrieved_docs)
