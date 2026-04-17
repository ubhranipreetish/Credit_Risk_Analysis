#!/usr/bin/env python3
"""
End-to-End RAG Pipeline Test
==============================

Smoke test that validates the full Milestone 2 pipeline:
    1. Schema validation — create and validate a borrower profile
    2. Risk prediction  — predict risk using Decision Tree
    3. Risk explanation  — get top risk drivers
    4. RAG build        — build FAISS index from rag/rag_docs/
    5. RAG retrieval    — retrieve docs for a sample query

Run:
    python test_rag_pipeline.py
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
logger = logging.getLogger("test_pipeline")


# ── Formatting helpers ───────────────────────────────────────────────────────

DIVIDER = "=" * 70
SECTION = "-" * 50


def print_header(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def print_json(data, indent=2):
    print(json.dumps(data, indent=indent, default=str))


# ── Test Data ────────────────────────────────────────────────────────────────

SAMPLE_BORROWER = {
    "person_age": 28,
    "person_income": 55000,
    "person_emp_length": 4.0,
    "loan_amnt": 12000,
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.22,
    "cb_person_cred_hist_length": 6,
    "person_home_ownership": "RENT",
    "loan_intent": "PERSONAL",
    "loan_grade": "C",
    "cb_person_default_on_file": "N",
}

RAG_QUERY = "high risk borrower loan approval rules"
DOCS_DIR = "rag/rag_docs"
INDEX_DIR = "rag/faiss_index"


# ── Step 1: Schema Validation ───────────────────────────────────────────────

def test_schema_validation():
    print_header("STEP 1: Input Schema Validation")
    from agent.schema import validate_input

    print(f"Input data:")
    print_json(SAMPLE_BORROWER)
    print(SECTION)

    input_df = validate_input(SAMPLE_BORROWER)

    print(f"✅ Validation passed")
    print(f"   DataFrame shape: {input_df.shape}")
    print(f"   Columns: {list(input_df.columns)}")
    print(f"   Dtypes:\n{input_df.dtypes.to_string()}")
    return input_df


# ── Step 2: Risk Prediction ─────────────────────────────────────────────────

def test_risk_prediction(input_df):
    print_header("STEP 2: Risk Prediction (Decision Tree)")
    from agent.model_loader import predict_risk

    result = predict_risk(input_df, model_name="decision_tree")

    print(f"✅ Prediction complete")
    print_json(result)
    return result


# ── Step 3: Risk Explanation ─────────────────────────────────────────────────

def test_risk_explanation(input_df):
    print_header("STEP 3: Risk Explanation")
    from agent.model_loader import load_model
    from agent.risk_explainer import explain_risk

    pipeline = load_model("decision_tree")
    factors = explain_risk(pipeline, input_df, model_name="decision_tree", top_n=5)

    print(f"✅ Top {len(factors)} risk drivers identified:")
    for i, factor in enumerate(factors, 1):
        direction = factor.get("direction", "unknown")
        importance = factor["importance"]
        print(f"   {i}. {factor['feature']:30s}  importance={importance:.4f}  ({direction})")

    return factors


# ── Step 4: RAG Index Build ──────────────────────────────────────────────────

def test_rag_build():
    print_header("STEP 4: RAG Index Build")
    from agent.rag import build_rag_index

    print(f"📂 Documents directory: {DOCS_DIR}")
    print(f"📦 Index output:        {INDEX_DIR}")
    print(SECTION)

    num_chunks = build_rag_index(
        docs_dir=DOCS_DIR,
        index_dir=INDEX_DIR,
        chunk_size=500,
        chunk_overlap=100,
    )

    print(f"\n✅ FAISS index built successfully")
    print(f"   Total chunks indexed: {num_chunks}")

    # Verify files exist
    index_file = os.path.join(INDEX_DIR, "index.faiss")
    meta_file = os.path.join(INDEX_DIR, "chunks_metadata.json")
    print(f"   Index file:    {index_file} ({os.path.getsize(index_file):,} bytes)")
    print(f"   Metadata file: {meta_file} ({os.path.getsize(meta_file):,} bytes)")

    return num_chunks


# ── Step 5: RAG Retrieval ────────────────────────────────────────────────────

def test_rag_retrieval():
    print_header("STEP 5: RAG Retrieval")
    from agent.rag import retrieve_docs

    print(f'🔍 Query: "{RAG_QUERY}"')
    print(SECTION)

    results = retrieve_docs(query=RAG_QUERY, top_k=5, index_dir=INDEX_DIR)

    print(f"\n✅ Retrieved {len(results)} relevant chunks:\n")
    for i, result in enumerate(results, 1):
        print(f"  ── Result {i} ──")
        print(f"  Score:  {result['score']:.4f}")
        print(f"  Source: {result['source']} (page {result['page']})")
        print(f"  Text:   {result['text'][:200]}...")
        print()

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 70)
    print("  MILESTONE 2 — AGENTIC AI LENDING PIPELINE — END-TO-END TEST")
    print("█" * 70)

    try:
        # Step 1: Validate input
        input_df = test_schema_validation()

        # Step 2: Predict risk
        prediction = test_risk_prediction(input_df)

        # Step 3: Explain risk
        factors = test_risk_explanation(input_df)

        # Step 4: Build RAG index
        num_chunks = test_rag_build()

        # Step 5: Retrieve documents
        results = test_rag_retrieval()

        # ── Summary ──────────────────────────────────────────────────────
        print_header("PIPELINE SUMMARY")
        print(f"  ✅ Schema Validation:  PASSED")
        print(f"  ✅ Risk Prediction:    {prediction['label']} (p={prediction['probability']:.4f})")
        print(f"  ✅ Risk Explanation:   {len(factors)} factors identified")
        print(f"  ✅ RAG Index Build:    {num_chunks} chunks indexed")
        print(f"  ✅ RAG Retrieval:      {len(results)} docs retrieved")
        print(f"\n  🎉 ALL STEPS PASSED — Pipeline ready for LLM integration")
        print(DIVIDER)

    except Exception as e:
        logger.error("Pipeline test failed: %s", e, exc_info=True)
        print(f"\n  ❌ PIPELINE TEST FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
