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

# Load environment variables from .env
load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.2
GROQ_MAX_TOKENS = 2048

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
    "Profile Summary": "Brief summary of the borrower's financial profile",
    "Risk Analysis": "Detailed analysis of default risk based on ML prediction and regulations",
    "Key Risk Drivers": "Top factors driving the risk assessment, explained in plain language",
    "Decision": "Approve or Reject",
    "Confidence": "High, Medium, or Low — how confident you are in this decision",
    "Justification": "Clear reasoning linking evidence to decision",
    "Sources": ["List of regulation documents referenced"],
    "Disclaimer": "This is an AI-generated recommendation and should not be used as the sole basis for lending decisions. Final approval must involve human review and comply with all applicable regulations."
}

# ── Confidence Thresholds ────────────────────────────────────────────────────
# Used by the workflow to decide whether to accept or re-evaluate a decision.

CONFIDENCE_LEVELS = {
    "high": 0.85,
    "medium": 0.60,
    "low": 0.30,
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
    "Profile Summary": "Brief summary of the borrower's financial profile based on the risk factors",
    "Risk Analysis": "Detailed analysis combining ML prediction with regulatory context",
    "Key Risk Drivers": "Top factors driving the assessment, explained in plain business language",
    "Decision": "Approve" or "Reject",
    "Confidence": "High", "Medium", or "Low" — how confident you are in this decision given the evidence,
    "Justification": "Clear reasoning linking the evidence to your decision",
    "Sources": ["list", "of", "source documents referenced"],
    "Disclaimer": "This is an AI-generated recommendation and should not be used as the sole basis for lending decisions. Final approval must involve human review and comply with all applicable regulations."
}}

IMPORTANT:
- Respond with ONLY the JSON object. No markdown, no code fences, no extra text.
- The "Decision" field must be exactly "Approve" or "Reject".
- The "Confidence" field must be exactly "High", "Medium", or "Low".
- "High" = clear-cut case, strong evidence supports the decision.
- "Medium" = some ambiguity, but evidence leans in one direction.
- "Low" = borderline case, decision could reasonably go either way.
- The "Sources" field must list actual document names from the regulations provided.
- If no regulations were provided, use an empty list for Sources."""

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
        decision = "Reject"
    elif probability > 0.35:
        decision = "Reject"  # Conservative: borderline cases get rejected
    else:
        decision = "Approve"

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

    return {
        "Profile Summary": (
            f"Borrower classified as {label} with a {probability:.1%} "
            f"probability of default based on ML model analysis."
        ),
        "Risk Analysis": (
            f"The ML model predicts a {probability:.1%} default probability. "
            f"This assessment is based on the borrower's financial profile "
            f"including income, loan characteristics, and credit history. "
            f"{'The elevated risk level warrants caution.' if probability > 0.3 else 'The risk level is within acceptable bounds.'}"
        ),
        "Key Risk Drivers": drivers,
        "Decision": decision,
        "Justification": (
            f"Based on the ML risk prediction ({label}, p={probability:.1%}), "
            f"the recommendation is to {decision.lower()} this loan application. "
            f"{'The default probability exceeds the acceptable threshold for approval.' if decision == 'Reject' else 'The default probability is within acceptable limits for standard lending criteria.'} "
            f"Note: This is a fallback assessment generated without LLM reasoning due to a service error."
        ),
        "Sources": sources if sources else [],
        "Disclaimer": (
            "This is an AI-generated recommendation and should not be used as the "
            "sole basis for lending decisions. Final approval must involve human review "
            "and comply with all applicable regulations. "
            "[FALLBACK MODE: LLM reasoning was unavailable; this decision is based on "
            "rule-based logic applied to the ML prediction.]"
        ),
    }


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
    required_fields = [
        "Profile Summary", "Risk Analysis", "Key Risk Drivers",
        "Decision", "Confidence", "Justification", "Sources", "Disclaimer",
    ]

    # Ensure all fields exist
    for field in required_fields:
        if field not in parsed:
            logger.warning("Missing field '%s' in LLM output — using default", field)
            if field == "Sources":
                parsed[field] = []
            elif field == "Decision":
                parsed[field] = "Reject"  # Conservative default
            elif field == "Confidence":
                parsed[field] = "Low"  # Default to low if missing
            elif field == "Disclaimer":
                parsed[field] = (
                    "This is an AI-generated recommendation and should not be used "
                    "as the sole basis for lending decisions. Final approval must involve "
                    "human review and comply with all applicable regulations."
                )
            else:
                parsed[field] = "Information not available."

    # Normalize the Decision field
    decision = str(parsed["Decision"]).strip().title()
    if decision not in ("Approve", "Reject"):
        logger.warning(
            "Invalid Decision value '%s' — defaulting to 'Reject' (conservative)",
            parsed["Decision"],
        )
        decision = "Reject"
    parsed["Decision"] = decision

    # Normalize the Confidence field
    confidence = str(parsed.get("Confidence", "Low")).strip().title()
    if confidence not in ("High", "Medium", "Low"):
        logger.warning(
            "Invalid Confidence value '%s' — defaulting to 'Low'",
            parsed.get("Confidence"),
        )
        confidence = "Low"
    parsed["Confidence"] = confidence

    # Ensure Sources is a list
    if not isinstance(parsed["Sources"], list):
        parsed["Sources"] = [str(parsed["Sources"])]

    return parsed


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
            "Profile Summary": str,
            "Risk Analysis": str,
            "Key Risk Drivers": str,
            "Decision": "Approve" or "Reject",
            "Justification": str,
            "Sources": List[str],
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
        validated = _validate_decision_output(parsed)
        logger.info("Decision: %s (Confidence: %s)", validated["Decision"], validated.get("Confidence", "N/A"))
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

    # ── Signal 1: LLM's self-reported confidence (40% weight) ────────────
    llm_confidence = decision.get("Confidence", "Low").lower()
    confidence_map = {"high": 1.0, "medium": 0.6, "low": 0.25}
    score += 0.40 * confidence_map.get(llm_confidence, 0.25)

    # ── Signal 2: ML-LLM alignment (35% weight) ─────────────────────────
    # If the LLM decision aligns with what the ML probability suggests,
    # confidence is higher.
    probability = risk.get("probability", 0.5)
    llm_decision = decision.get("Decision", "Reject")

    if llm_decision == "Reject" and probability > 0.5:
        alignment = 1.0  # Both say risky
    elif llm_decision == "Approve" and probability < 0.5:
        alignment = 1.0  # Both say safe
    elif llm_decision == "Approve" and probability > 0.5:
        alignment = 0.2  # LLM overrides ML — low alignment
    elif llm_decision == "Reject" and probability < 0.3:
        alignment = 0.3  # LLM is overly conservative
    else:
        alignment = 0.5  # Borderline zone
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
        "Confidence score: %.4f (LLM=%s, alignment=%.2f, borderline=%.2f)",
        final_score, llm_confidence, alignment, borderline_score,
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
    logger.info("🔄 Reflecting on initial decision: %s (Confidence: %s)",
                initial_decision.get("Decision"), initial_decision.get("Confidence"))

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

Decision: {initial_decision.get('Decision', 'Unknown')}
Confidence: {initial_decision.get('Confidence', 'Unknown')}
Profile Summary: {initial_decision.get('Profile Summary', 'N/A')}
Risk Analysis: {initial_decision.get('Risk Analysis', 'N/A')}
Justification: {initial_decision.get('Justification', 'N/A')}

═══════════════════════════════════════════
YOUR REVIEW TASK
═══════════════════════════════════════════

1. Is the initial decision logically consistent with the ML prediction?
2. Are there any risks or mitigating factors the initial analysis missed?
3. Are the cited regulations applied correctly?
4. Should the decision be UPHELD or OVERRIDDEN?

Respond with ONLY a valid JSON object:
{{
    "Profile Summary": "Updated summary incorporating your review insights",
    "Risk Analysis": "Deeper analysis after second-pass review",
    "Key Risk Drivers": "Refined risk drivers after critical examination",
    "Decision": "Approve" or "Reject" (your final verdict),
    "Confidence": "High", "Medium", or "Low" (your confidence after review),
    "Justification": "Full justification incorporating review findings",
    "Reflection": "What you found in your review — did you agree or disagree with the original? Why?",
    "Sources": ["source documents referenced"],
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
        validated = _validate_decision_output(parsed)

        # Preserve the reflection field if present
        if "Reflection" in parsed:
            validated["Reflection"] = parsed["Reflection"]
        else:
            validated["Reflection"] = "Review completed — no specific critique noted."

        # Mark as reviewed
        validated["review_pass"] = "reflected"

        logger.info(
            "🔄 Reflection result: %s (Confidence: %s) | %s",
            validated["Decision"], validated.get("Confidence"),
            "UPHELD" if validated["Decision"] == initial_decision.get("Decision") else "OVERRIDDEN",
        )

        return validated

    except Exception as e:
        logger.error("Reflection failed: %s — keeping initial decision", e)
        initial_decision["Reflection"] = f"Reflection attempt failed: {e}. Initial decision preserved."
        initial_decision["review_pass"] = "reflection_failed"
        return initial_decision
