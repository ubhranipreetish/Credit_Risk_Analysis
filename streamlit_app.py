import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from agent.model_loader import predict_risk
from agent.schema import validate_input
from backend.core.exceptions import BackendError
from backend.schemas.analyze import AnalyzeRequest
from backend.services.analysis_service import AnalysisService


st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")
st.title("Credit Risk Analyzer")
st.caption("Milestone 1 shows the classical ML scoring layer. Milestone 2 adds the agentic lending workflow with RAG and structured reasoning.")

if not os.environ.get("GROQ_API_KEY"):
    st.info(
        "LLM fallback mode is active: GROQ_API_KEY is not set. "
        "Add GROQ_API_KEY to your environment or a project .env file to enable full LLM reasoning."
    )

service = AnalysisService()


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    metrics_path = Path("reports/metrics.json")
    if not metrics_path.exists():
        return {}

    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_borrower_input(prefix: str = "") -> dict:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")
        person_age = st.number_input(f"Age{prefix}", min_value=18, max_value=100, value=30)
        person_income = st.number_input(f"Annual Income ($){prefix}", min_value=0, value=50000, step=1000)
        person_emp_length = st.number_input(f"Employment Length (years){prefix}", min_value=0.0, value=5.0, step=0.5)
        person_home_ownership = st.selectbox(f"Home Ownership{prefix}", ["RENT", "MORTGAGE", "OWN", "OTHER"])

    with col2:
        st.subheader("Loan Information")
        loan_amnt = st.number_input(f"Loan Amount ($){prefix}", min_value=1, value=10000, step=500)
        loan_int_rate = st.number_input(f"Interest Rate (%) {prefix}".strip(), min_value=0.0, max_value=50.0, value=10.0, step=0.1)
        loan_intent = st.selectbox(
            f"Loan Intent{prefix}",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        )
        loan_grade = st.selectbox(f"Loan Grade{prefix}", ["A", "B", "C", "D", "E", "F", "G"])

    st.subheader("Credit History")
    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        cb_person_cred_hist_length = st.number_input(f"Credit History Length (years){prefix}", min_value=0, value=5)
    with ch_col2:
        cb_person_default_on_file = st.selectbox(f"Historical Default on File{prefix}", ["N", "Y"])

    loan_percent_income = (loan_amnt / person_income) if person_income > 0 else 0.0

    return {
        "person_age": int(person_age),
        "person_income": int(person_income),
        "person_emp_length": float(person_emp_length),
        "loan_amnt": int(loan_amnt),
        "loan_int_rate": float(loan_int_rate),
        "loan_percent_income": float(loan_percent_income),
        "cb_person_cred_hist_length": int(cb_person_cred_hist_length),
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "cb_person_default_on_file": cb_person_default_on_file,
    }


def build_request(input_data: dict) -> AnalyzeRequest:
    return AnalyzeRequest(**input_data)


def render_metrics_snapshot() -> None:
    metrics = load_metrics()
    if not metrics:
        st.info("Milestone 1 evaluation metrics are not available yet. Run the training pipeline to populate reports/metrics.json.")
        return

    st.subheader("Evaluation Snapshot")
    for model_name, model_metrics in metrics.items():
        st.markdown(f"**{model_name}**")
        score_cols = st.columns(3)
        score_cols[0].metric("Accuracy", f"{model_metrics.get('Accuracy', 0.0):.4f}")
        score_cols[1].metric("ROC-AUC", f"{model_metrics.get('ROC_AUC', 0.0):.4f}")
        score_cols[2].metric("Confusion Matrix", "Available")

        confusion_matrix = model_metrics.get("Confusion_Matrix")
        if confusion_matrix:
            matrix_frame = pd.DataFrame(
                confusion_matrix,
                index=["Actual Good (0)", "Actual Bad (1)"],
                columns=["Predicted Good (0)", "Predicted Bad (1)"],
            )
            st.dataframe(matrix_frame, use_container_width=True)


def render_milestone_1_tab() -> None:
    st.subheader("Milestone 1: Machine Learning Credit Risk Scoring")
    st.write(
        "This tab shows the classical scoring system: borrower features go through schema validation, the trained sklearn pipeline predicts default risk, and the saved evaluation metrics summarize model quality."
    )

    input_data = collect_borrower_input(" (M1)")
    model_name = st.radio(
        "Scoring Model",
        ["Logistic Regression", "Decision Tree"],
        horizontal=True,
        key="milestone1_model_choice",
    )

    if st.button("Run ML Credit Scoring", type="primary", key="milestone1_run"):
        try:
            validated_request = build_request(input_data)
            validated_df = validate_input(validated_request.model_dump())
            backend_model_name = "logistic" if model_name == "Logistic Regression" else "decision_tree"
            prediction = predict_risk(validated_df, model_name=backend_model_name)

            decision_cols = st.columns(3)
            decision_cols[0].metric("Predicted Class", prediction["label"])
            decision_cols[1].metric("Default Probability", f"{prediction['probability']:.2%}")
            decision_cols[2].metric("Model Used", prediction["model_used"].replace("_", " ").title())

            if prediction["label"] == "High Risk":
                st.error("The ML model predicts elevated default risk.")
            else:
                st.success("The ML model predicts lower default risk.")

            st.write("**Feature-ready borrower profile**")
            st.dataframe(validated_df, use_container_width=True)

            render_metrics_snapshot()

        except Exception as exc:
            st.error(f"Milestone 1 scoring failed: {exc}")


def render_agentic_report(result) -> None:
    st.markdown("---")
    st.subheader("Decision")
    decision = result.decision.lending_decision
    if decision == "REJECT":
        st.error("REJECT")
    elif decision == "CONDITIONAL":
        st.warning("CONDITIONAL")
    else:
        st.success("APPROVE")

    st.metric("Confidence", f"{result.decision.confidence:.2f}")

    report_cols = st.columns(2)
    with report_cols[0]:
        st.write("**Borrower Profile Summary**")
        st.write(result.decision.borrower_profile_summary)
        st.write("**Risk Analysis**")
        st.write(result.decision.risk_analysis)
    with report_cols[1]:
        st.write("**Regulatory References**")
        if result.decision.regulatory_references:
            for source in result.decision.regulatory_references:
                st.write(f"- {source}")
        else:
            st.write("- None")
        st.write("**Disclaimer**")
        st.write(result.decision.disclaimer)


def render_milestone_2_tab() -> None:
    st.subheader("Milestone 2: Agentic Lending Decision Support")
    st.write(
        "This tab runs the LangGraph-based workflow. The ML risk prediction is combined with explanations, regulatory retrieval, and LLM reasoning to produce a structured lending recommendation."
    )

    input_data = collect_borrower_input(" (M2)")

    if st.button("Run Agent Workflow", type="primary", key="milestone2_run"):
        try:
            req = build_request(input_data)

            with st.spinner("Running agent workflow..."):
                result = service.analyze(req)

            st.metric("Workflow Status", result.status.title())
            st.metric("Reasoning Passes", str(result.reasoning_passes))
            render_agentic_report(result)

            if result.warnings:
                st.markdown("---")
                st.subheader("Warnings")
                for warning in result.warnings:
                    if warning.code == "llm_fallback" and not os.environ.get("GROQ_API_KEY"):
                        st.info(f"{warning.code}: {warning.message}")
                    else:
                        st.warning(f"{warning.code}: {warning.message}")

            with st.expander("Workflow trace"):
                st.write(result.steps_completed)
                st.write({
                    "risk_tier": result.risk_tier,
                    "confidence_score": result.confidence_score,
                    "metadata": result.metadata,
                })

        except BackendError as exc:
            st.error(f"Request failed: {exc.message}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")


tab_m1, tab_m2 = st.tabs(["Milestone 1: ML Scoring", "Milestone 2: Agentic Decision Support"])

with tab_m1:
    render_milestone_1_tab()

with tab_m2:
    render_milestone_2_tab()
