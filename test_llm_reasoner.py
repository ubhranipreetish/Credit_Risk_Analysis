#!/usr/bin/env python3
"""
End-to-End LLM Reasoning Test
================================

Tests the full pipeline including the LLM reasoning node:
    1. Validate borrower input
    2. Predict risk (ML model)
    3. Explain risk factors
    4. Build RAG query from risk + explanation
    5. Retrieve relevant regulations
    6. Generate structured lending decision (LLM)

Run:
    python test_llm_reasoner.py
"""

import os
import sys
import json
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_llm")


DIVIDER = "=" * 70
SECTION = "-" * 50


def print_header(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ── Sample Borrower Profiles ────────────────────────────────────────────────

HIGH_RISK_BORROWER = {
    "person_age": 22,
    "person_income": 25000,
    "person_emp_length": 0.5,
    "loan_amnt": 18000,
    "loan_int_rate": 18.5,
    "loan_percent_income": 0.72,
    "cb_person_cred_hist_length": 2,
    "person_home_ownership": "RENT",
    "loan_intent": "PERSONAL",
    "loan_grade": "E",
    "cb_person_default_on_file": "Y",
}

LOW_RISK_BORROWER = {
    "person_age": 35,
    "person_income": 85000,
    "person_emp_length": 8.0,
    "loan_amnt": 10000,
    "loan_int_rate": 7.5,
    "loan_percent_income": 0.12,
    "cb_person_cred_hist_length": 12,
    "person_home_ownership": "MORTGAGE",
    "loan_intent": "HOMEIMPROVEMENT",
    "loan_grade": "A",
    "cb_person_default_on_file": "N",
}


def run_full_pipeline(borrower_data: dict, label: str):
    """Run the complete pipeline for a single borrower profile."""
    from agent.schema import validate_input
    from agent.model_loader import predict_risk, load_model
    from agent.risk_explainer import explain_risk
    from agent.llm_reasoner import generate_decision, build_query
    from agent.rag import retrieve_docs

    print_header(f"PIPELINE: {label}")

    # Step 1: Validate
    print(f"\n📋 Input Profile:")
    for k, v in borrower_data.items():
        print(f"   {k:30s}: {v}")
    input_df = validate_input(borrower_data)
    print(f"\n✅ Input validated")

    # Step 2: Predict
    risk = predict_risk(input_df, model_name="decision_tree")
    print(f"✅ Risk Prediction: {risk['label']} (p={risk['probability']:.4f})")

    # Step 3: Explain
    pipeline = load_model("decision_tree")
    explanation = explain_risk(pipeline, input_df, model_name="decision_tree", top_n=5)
    print(f"✅ Risk Explanation: {len(explanation)} factors")
    for f in explanation[:3]:
        print(f"   - {f['feature']:30s} importance={f['importance']:.4f}")

    # Step 4: Build query and retrieve docs
    query = build_query(risk, explanation)
    print(f"\n🔍 RAG Query: \"{query[:80]}...\"")
    retrieved_docs = retrieve_docs(query=query, top_k=5, index_dir="rag/faiss_index")
    print(f"✅ Retrieved {len(retrieved_docs)} regulation chunks")

    # Step 5: Generate LLM decision
    print(f"\n🤖 Calling LLM for reasoning...")
    print(SECTION)
    decision = generate_decision(risk, explanation, retrieved_docs)

    # Print the structured decision
    print(f"\n{'█' * 70}")
    print(f"  LENDING DECISION REPORT")
    print(f"{'█' * 70}")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"{'█' * 70}")

    return decision


def main():
    print("\n" + "█" * 70)
    print("  MILESTONE 2 — LLM REASONING NODE — END-TO-END TEST")
    print("█" * 70)

    try:
        # Test 1: High risk borrower (should be rejected)
        decision_high = run_full_pipeline(HIGH_RISK_BORROWER, "HIGH RISK BORROWER")

        print("\n" + "=" * 70)

        # Test 2: Low risk borrower (should be approved)
        decision_low = run_full_pipeline(LOW_RISK_BORROWER, "LOW RISK BORROWER")

        # Summary
        print_header("TEST SUMMARY")
        print(f"  High Risk Borrower → Decision: {decision_high['Lending Decision']}")
        print(f"  Low Risk Borrower  → Decision: {decision_low['Lending Decision']}")

        # Validate decisions make sense
        if decision_high["Lending Decision"] == "REJECT":
            print(f"  ✅ High risk correctly rejected")
        else:
            print(f"  ⚠️  High risk was approved — may need review")

        if decision_low["Lending Decision"] == "APPROVE":
            print(f"  ✅ Low risk correctly approved")
        else:
            print(f"  ⚠️  Low risk was rejected — LLM may be overly conservative")

        print(f"\n  🎉 LLM REASONING NODE TEST COMPLETE")
        print(DIVIDER)

    except Exception as e:
        logger.error("Test failed: %s", e, exc_info=True)
        print(f"\n  ❌ TEST FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
