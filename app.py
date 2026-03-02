import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")

st.title("Intelligent Credit Risk Scoring")
st.markdown("Enter borrower details below to predict credit risk using the trained Decision Tree model.")

@st.cache_resource
def load_models():
    try:
        dt_pipeline = joblib.load("models/decision_tree_pipeline.joblib")
        target_encoder = joblib.load("models/target_encoder.joblib")
        return dt_pipeline, target_encoder
    except FileNotFoundError:
        st.error("Models not found. Please train the models first by running `src/train.py`.")
        st.stop()

model, encoder = load_models()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, value=5.0)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])

with col2:
    st.subheader("Loan Details")
    loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=10.0)
    loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

st.subheader("Credit History")
col3, col4 = st.columns(2)
with col3:
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5)
with col4:
    cb_person_default_on_file = st.selectbox("Historical Default on File", ["Y", "N"])

if loan_amnt > 0 and person_income > 0:
    loan_percent_income = loan_amnt / person_income
else:
    loan_percent_income = 0.0

if st.button("Evaluate Credit Risk"):
    input_df = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "cb_person_default_on_file": cb_person_default_on_file
    }])
    
    prediction = model.predict(input_df)
    status = encoder.inverse_transform(prediction)[0]
    
    st.markdown("---")
    st.subheader("Assessment Result")
    
    if status == 1:
        st.error("⚠️ **High Risk**: The profile matches historical patterns of default.")
    else:
        st.success("**Low Risk**: The profile suggests successful repayment.")
    
    try:
        proba = model.predict_proba(input_df)[0]
        st.info(f"Probability of Default: {proba[1]:.1%}")
    except Exception:
        pass
