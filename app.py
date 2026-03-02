import os
import sys
import time
import joblib
import pandas as pd
import streamlit as st


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.explain import get_logistic_coefficients, get_decision_tree_importance

st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")

st.title("Intelligent Credit Risk Scoring")
st.markdown("Enter borrower details below to predict credit risk using our machine learning models.")

@st.cache_resource
def load_models():
    try:
        dt_pipeline = joblib.load("models/decision_tree_pipeline.joblib")
        lr_pipeline = joblib.load("models/logistic_pipeline.joblib")
        target_encoder = joblib.load("models/target_encoder.joblib")
        return dt_pipeline, lr_pipeline, target_encoder
    except FileNotFoundError:
        st.error("Models not found. Please train the models first by running `src/train.py`.")
        st.stop()

dt_pipeline, lr_pipeline, encoder = load_models()

model_choice = st.selectbox("Select Model", ["Decision Tree", "Logistic Regression"])
if model_choice == "Decision Tree":
    model = dt_pipeline
else:
    model = lr_pipeline

with st.expander("📊 Model Performance (Test Set Metrics)", expanded=False):
    st.markdown(f"**Selected Model:** {model_choice}")
    
    metrics = {
        "Logistic Regression": {
            "Accuracy": 0.8494, 
            "ROC-AUC": 0.8592, 
            "Confusion Matrix": [[7278, 364], [1108, 1025]]
        },
        "Decision Tree": {
            "Accuracy": 0.9099, 
            "ROC-AUC": 0.8963, 
            "Confusion Matrix": [[7626, 16], [865, 1268]]
        }
    }
    m = metrics[model_choice]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{m['Accuracy'] * 100:.2f}%")
    c2.metric("ROC-AUC", f"{m['ROC-AUC'] * 100:.2f}%")
    
    st.markdown("**Confusion Matrix:**")
    cm_df = pd.DataFrame(
        m["Confusion Matrix"], 
        columns=["Predicted Good (0)", "Predicted Bad (1)"], 
        index=["Actual Good (0)", "Actual Bad (1)"]
    )
    st.dataframe(cm_df)

st.markdown("---")

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
    with st.spinner("Analyzing risk profile..."):
        time.sleep(0.5) 
        
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
            st.success("✅ **Low Risk**: The profile suggests successful repayment.")
        
        try:
            proba = model.predict_proba(input_df)[0][1] 
            
            st.markdown(f"**Probability of Default:** {proba:.1%}")
            
            progress_color = "red" if proba > 0.5 else "green"
            st.markdown(
                f"""
                <div style="width: 100%; background-color: #e0e0e0; border-radius: 5px;">
                    <div style="width: {proba*100}%; background-color: {progress_color}; height: 24px; border-radius: 5px;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            
        except Exception as e:
            st.warning("Probability estimation not available for this model.")
        
        st.subheader("Key Risk Drivers")
        st.markdown("The top 3 features influencing this prediction:")
        
        def clean_feature_name(name):
            return name.replace('num__', '').replace('cat__', '').replace('_', ' ').title()
        
        if model_choice == "Decision Tree":
            importance_df = get_decision_tree_importance(model)
            top_features = importance_df.head(3)
            for _, row in top_features.iterrows():
                feat = clean_feature_name(row['Feature'])
                val = row['Importance']
                st.markdown(f"- **{feat}** (Relative Importance: {val:.1%})")
        else:
            coef_df = get_logistic_coefficients(model)
            top_features = coef_df.head(3)
            for _, row in top_features.iterrows():
                feat = clean_feature_name(row['Feature'])
                impact = row['Absolute Impact']
                raw_coef = row['Coefficient']
                direction = "Increases Risk ⬆️" if raw_coef > 0 else "Decreases Risk ⬇️"
                st.markdown(f"- **{feat}** ({direction}, Impact Magnitude: {impact:.2f})")
