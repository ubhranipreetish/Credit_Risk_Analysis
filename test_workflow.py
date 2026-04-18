#!/usr/bin/env python3
"""
LangGraph Agent Workflow Test
===============================

Tests the full LangGraph agent pipeline with debug mode enabled,
showing intermediate state after each node.

Run:
    python test_workflow.py
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

DIVIDER = "=" * 70

HIGH_RISK = {
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

LOW_RISK = {
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

BORDERLINE_RISK = {
    "person_age": 25,
    "person_income": 32000,
    "person_emp_length": 1.0,
    "loan_amnt": 14000,
    "loan_int_rate": 15.5,
    "loan_percent_income": 0.44,
    "cb_person_cred_hist_length": 3,
    "person_home_ownership": "RENT",
    "loan_intent": "MEDICAL",
    "loan_grade": "D",
    "cb_person_default_on_file": "Y",
}


def test_agent(input_data: dict, label: str, debug: bool = True):
    from agent.workflow import run_agent

    print(f"\n{'█' * 70}")
    print(f"  AGENT TEST: {label}")
    print(f"{'█' * 70}")

    result = run_agent(input_data, debug=debug)

    # Print final decision report
    decision = result.get("decision", {})
    print(f"\n{DIVIDER}")
    print(f"  FINAL DECISION REPORT")
    print(DIVIDER)
    print(json.dumps(decision, indent=2, ensure_ascii=False))

    print(f"\n  Steps completed: {' → '.join(result.get('steps_completed', []))}")
    print(f"  Decision: {decision.get('Lending Decision', 'N/A')}")
    print(DIVIDER)

    return result


def main():
    print(f"\n{'█' * 70}")
    print(f"  LANGGRAPH AGENT WORKFLOW — END-TO-END TEST")
    print(f"{'█' * 70}")

    # Test 1: High risk with debug mode
    r1 = test_agent(HIGH_RISK, "HIGH RISK BORROWER (debug=False)", debug=False)

    # Test 2: Low risk without debug
    r2 = test_agent(LOW_RISK, "LOW RISK BORROWER (debug=False)", debug=False)
    
    # Test 3: Borderline risk with debug mode to show reflection loop
    r3 = test_agent(BORDERLINE_RISK, "BORDERLINE RISK BORROWER (debug=True)", debug=True)

    # Summary
    print(f"\n{DIVIDER}")
    print(f"  TEST SUMMARY")
    print(DIVIDER)
    print(f"  High Risk      → {r1['decision']['Lending Decision']}  {'✅' if r1['decision']['Lending Decision'] == 'REJECT' else '⚠️'}")
    print(f"  Low Risk       → {r2['decision']['Lending Decision']}  {'✅' if r2['decision']['Lending Decision'] == 'APPROVE' else '⚠️'}")
    print(f"  Borderline     → {r3['decision']['Lending Decision']} (score: {r3.get('confidence_score')}, passes: {r3.get('reasoning_passes')})")
    print(f"\n  🎉 LANGGRAPH AGENT WORKFLOW TEST COMPLETE")
    print(DIVIDER)


if __name__ == "__main__":
    main()
