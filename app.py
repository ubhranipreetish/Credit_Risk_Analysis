import os
import sys
import time
import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.explain import get_logistic_coefficients, get_decision_tree_importance
from src.preprocessing import build_preprocessor
from src.model_builder import build_logistic_pipeline, build_decision_tree_pipeline

def auto_train():
    model_dir = "models"
    if (os.path.exists(f"{model_dir}/decision_tree_pipeline.joblib") and
            os.path.exists(f"{model_dir}/logistic_pipeline.joblib")):
        return
    os.makedirs(model_dir, exist_ok=True)
    with st.spinner("First run detected — training models, please wait…"):
        df = pd.read_csv("data/credit_risk_dataset.csv")
        numerical_cols   = ['person_age','person_income','person_emp_length','loan_amnt',
                             'loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
        categorical_cols = ['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']
        X = df.drop(columns=["loan_status"])
        y = df["loan_status"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)
        enc = LabelEncoder()
        y_train = enc.fit_transform(y_train)
        preprocessor  = build_preprocessor(numerical_cols, categorical_cols)
        lr_pipe = build_logistic_pipeline(preprocessor)
        dt_pipe = build_decision_tree_pipeline(preprocessor)
        lr_pipe.fit(X_train, y_train)
        dt_pipe.fit(X_train, y_train)
        joblib.dump(lr_pipe, f"{model_dir}/logistic_pipeline.joblib")
        joblib.dump(dt_pipe, f"{model_dir}/decision_tree_pipeline.joblib")
        joblib.dump(enc,     f"{model_dir}/target_encoder.joblib")

auto_train()

st.set_page_config(page_title="Credit Risk Analyzer", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a4e, #0f3460);
    background-size: 400% 400%;
    animation: aurora 16s ease infinite;
    color: #e2e8f0;
}
@keyframes aurora {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}

.stApp::before {
    content: ''; position: fixed; top: -160px; left: -160px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none; z-index: 0;
    animation: orb1 13s ease-in-out infinite;
}
.stApp::after {
    content: ''; position: fixed; bottom: -130px; right: -130px;
    width: 520px; height: 520px;
    background: radial-gradient(circle, rgba(236,72,153,0.13) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none; z-index: 0;
    animation: orb2 17s ease-in-out infinite;
}
@keyframes orb1 { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(45px,55px) scale(1.1)} }
@keyframes orb2 { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(-35px,-45px) scale(1.08)} }

h1 { font-family:'Space Grotesk',sans-serif !important; font-weight:700 !important; color:#fff !important; letter-spacing:-0.5px !important; margin-bottom:0 !important; }
h2, h3 { font-family:'Space Grotesk',sans-serif !important; color:#c7d2fe !important; font-weight:600 !important; }

section[data-testid="stSidebar"] { display:none; }

input[type="number"],
div[data-baseweb="input"] input {
    background: #ffffff !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    border-radius: 12px !important;
    color: #1e293b !important;
    font-size: 1rem !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
input[type="number"]:hover,
input[type="number"]:focus {
    border-color: rgba(129,140,248,0.65) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.18) !important;
}

div[data-baseweb="select"] > div {
    background: rgba(30, 27, 75, 0.85) !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
    backdrop-filter: blur(8px);
}
div[data-baseweb="select"] > div:hover {
    border-color: rgba(129,140,248,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
div[data-baseweb="select"] span {
    color: #e2e8f0 !important;
}
ul[data-testid="stSelectboxVirtualDropdown"] {
    background: #1e1b4b !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
}
ul[data-testid="stSelectboxVirtualDropdown"] li {
    color: #e2e8f0 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600; font-size: 1rem;
    border: none; border-radius: 14px;
    padding: 0.75rem 2rem; width: 100%;
    box-shadow: 0 4px 24px rgba(99,102,241,0.35);
    transition: transform 0.25s ease, box-shadow 0.25s ease, background 0.3s ease;
    margin-top: 8px;
}
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.6);
    background: linear-gradient(135deg, #818cf8, #a78bfa);
}
.stButton > button:active { transform: translateY(0); }

details {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    padding: 6px 12px !important;
    backdrop-filter: blur(10px);
    transition: box-shadow 0.3s;
}
details:hover { box-shadow: 0 4px 24px rgba(99,102,241,0.15) !important; }
summary { color:#c7d2fe !important; font-weight:600 !important; font-size:1rem !important; }

div[data-testid="stAlertContainer"] > div {
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    backdrop-filter: blur(10px);
    padding: 1rem 1.25rem !important;
    animation: slideUp 0.4s ease both;
}
@keyframes slideUp {
    from { opacity:0; transform:translateY(12px); }
    to   { opacity:1; transform:translateY(0); }
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
    backdrop-filter: blur(8px);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
div[data-testid="stMetric"]:hover { transform:translateY(-3px); box-shadow:0 6px 24px rgba(99,102,241,0.2); }
div[data-testid="stMetricValue"] { color:#a5b4fc !important; font-family:'Space Grotesk',sans-serif !important; font-size:1.8rem !important; font-weight:700 !important; }
div[data-testid="stMetricLabel"] { color:#94a3b8 !important; font-size:0.82rem !important; font-weight:500 !important; text-transform:uppercase; letter-spacing:0.5px; }

label, .stSelectbox label, .stNumberInput label { color:#94a3b8 !important; font-size:0.88rem !important; font-weight:500 !important; }

.stDataFrame { border-radius:14px !important; overflow:hidden; border:1px solid rgba(255,255,255,0.1) !important; }

hr { border-color:rgba(255,255,255,0.1) !important; margin:1.5rem 0 !important; }

div[data-testid="stSpinner"] > div { border-top-color:#818cf8 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; padding:2.5rem 0 1.5rem 0;'>
    <div style='display:inline-block; background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(139,92,246,0.15));
                border:1px solid rgba(99,102,241,0.35); border-radius:999px; padding:6px 22px;
                color:#a5b4fc; font-size:0.82rem; font-weight:600; letter-spacing:1.5px; margin-bottom:18px;'>
        ✦&nbsp; ML-POWERED LENDING INTELLIGENCE
    </div><br>
    <span style='font-family:Space Grotesk,sans-serif; font-size:3rem; font-weight:700;
                 background:linear-gradient(135deg,#e0e7ff,#c7d2fe,#a5b4fc);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        Credit Risk Analyzer
    </span>
    <p style='color:#94a3b8; font-size:1.1rem; margin-top:12px; font-weight:400;'>
        Instant borrower risk profiling powered by machine learning
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        dt  = joblib.load("models/decision_tree_pipeline.joblib")
        lr  = joblib.load("models/logistic_pipeline.joblib")
        enc = joblib.load("models/target_encoder.joblib")
        return dt, lr, enc
    except FileNotFoundError:
        st.error("Models not found. Run `src/train.py` first.")
        st.stop()

dt_pipeline, lr_pipeline, encoder = load_models()

sel_col, _ = st.columns([1, 2])
with sel_col:
    model_choice = st.selectbox("🤖  Select Model", ["Decision Tree", "Logistic Regression"])
model = dt_pipeline if model_choice == "Decision Tree" else lr_pipeline

with st.expander("📊  Model Performance — Test Set Metrics", expanded=False):
    ALL_METRICS = {
        "Logistic Regression": {"Accuracy": 0.8494, "ROC-AUC": 0.8592,
                                 "CM": [[7278, 364], [1108, 1025]]},
        "Decision Tree":       {"Accuracy": 0.9099, "ROC-AUC": 0.8963,
                                 "CM": [[7626,  16], [ 865, 1268]]}
    }
    m = ALL_METRICS[model_choice]
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Accuracy", f"{m['Accuracy']*100:.2f}%")
    mc2.metric("ROC-AUC",  f"{m['ROC-AUC']*100:.2f}%")
    st.markdown("<br>", unsafe_allow_html=True)
    cm_df = pd.DataFrame(m["CM"],
                          columns=["Predicted Good (0)", "Predicted Bad (1)"],
                          index=["Actual Good (0)", "Actual Bad (1)"])
    st.dataframe(cm_df, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 👤 &nbsp; Personal Information")
    person_age            = st.number_input("Age",                           min_value=18,  max_value=100,     value=30)
    person_income         = st.number_input("Annual Income ($)",             min_value=0,                      value=50000)
    person_emp_length     = st.number_input("Employment Length (years)",     min_value=0.0,                    value=5.0)
    person_home_ownership = st.selectbox  ("Home Ownership",                ["RENT", "MORTGAGE", "OWN", "OTHER"])

with col2:
    st.markdown("### 💳 &nbsp; Loan Details")
    loan_amnt     = st.number_input("Loan Amount ($)",   min_value=0,   max_value=1_000_000, value=10000)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0,      value=10.0)
    loan_intent   = st.selectbox  ("Loan Intent",       ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"])
    loan_grade    = st.selectbox  ("Loan Grade",        ["A","B","C","D","E","F","G"])

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📁 &nbsp; Credit History")
col3, col4 = st.columns(2, gap="large")
with col3:
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5)
with col4:
    cb_person_default_on_file  = st.selectbox("Historical Default on File", ["N", "Y"])

loan_percent_income = (loan_amnt / person_income) if person_income > 0 else 0.0

st.markdown("<br>", unsafe_allow_html=True)

if st.button("⚡ &nbsp; Evaluate Credit Risk"):
    with st.spinner("Running inference…"):
        time.sleep(0.65)

        input_df = pd.DataFrame([{
            "person_age":               person_age,
            "person_income":            person_income,
            "person_emp_length":        person_emp_length,
            "loan_amnt":                loan_amnt,
            "loan_int_rate":            loan_int_rate,
            "loan_percent_income":      loan_percent_income,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
            "person_home_ownership":    person_home_ownership,
            "loan_intent":              loan_intent,
            "loan_grade":               loan_grade,
            "cb_person_default_on_file": cb_person_default_on_file
        }])

        prediction = model.predict(input_df)
        status     = encoder.inverse_transform(prediction)[0]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 &nbsp; Assessment Result")

    if status == 1:
        st.error("⚠️  **High Risk** — This profile matches historical patterns of loan default.")
        verdict_color = "#ef4444"
    else:
        st.success("✅  **Low Risk** — This profile suggests a high likelihood of successful repayment.")
        verdict_color = "#22c55e"

    try:
        proba   = model.predict_proba(input_df)[0][1]
        bar_pct = int(proba * 100)
        st.markdown(f"""
        <div style='margin:1.4rem 0 0.5rem 0;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                <span style='color:#94a3b8; font-size:0.9rem; font-weight:500;'>Default Probability</span>
                <span style='color:{verdict_color}; font-size:1rem; font-weight:700;'>{proba:.1%}</span>
            </div>
            <div style='width:100%; height:10px; background:rgba(255,255,255,0.08); border-radius:999px; overflow:hidden;'>
                <div style='
                    height:100%; width:{bar_pct}%;
                    background: linear-gradient(90deg, #22c55e 0%, #eab308 50%, #ef4444 100%);
                    background-size:{max(bar_pct,1)*4}% 100%;
                    border-radius:999px;
                    transition: width 1s cubic-bezier(0.34,1.56,0.64,1);
                '></div>
            </div>
            <div style='display:flex; justify-content:space-between; margin-top:5px;'>
                <span style='color:#64748b; font-size:0.75rem;'>Low Risk</span>
                <span style='color:#64748b; font-size:0.75rem;'>High Risk</span>
            </div>
        </div>""", unsafe_allow_html=True)
    except Exception:
        st.warning("Probability estimation not available for this model.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔍 &nbsp; Key Risk Drivers")

    def clean(n):
        return n.replace("num__","").replace("cat__","").replace("_"," ").title()

    if model_choice == "Decision Tree":
        for _, row in get_decision_tree_importance(model).head(3).iterrows():
            feat  = clean(row["Feature"])
            val   = row["Importance"]
            bar_w = min(int(val * 380), 100)
            st.markdown(f"""
            <div style='margin-bottom:14px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:5px;'>
                    <span style='color:#e2e8f0; font-weight:500; font-size:0.95rem;'>{feat}</span>
                    <span style='color:#a5b4fc; font-weight:700;'>{val:.1%}</span>
                </div>
                <div style='height:7px; background:rgba(255,255,255,0.07); border-radius:999px; overflow:hidden;'>
                    <div style='height:100%; width:{bar_w}%;
                         background:linear-gradient(90deg,#6366f1,#8b5cf6);
                         border-radius:999px;'></div>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        for _, row in get_logistic_coefficients(model).head(3).iterrows():
            feat      = clean(row["Feature"])
            impact    = row["Absolute Impact"]
            is_risk   = row["Coefficient"] > 0
            direction = "⬆️  Increases Risk" if is_risk else "⬇️  Reduces Risk"
            color     = "#f87171" if is_risk else "#4ade80"
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                         border-radius:14px; padding:12px 18px; margin-bottom:12px;
                         transition: background 0.3s;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#e2e8f0; font-weight:500; font-size:0.95rem;'>{feat}</span>
                    <span style='color:{color}; font-size:0.88rem; font-weight:600;'>{direction}</span>
                </div>
                <span style='color:#64748b; font-size:0.8rem;'>Impact magnitude: {impact:.3f}</span>
            </div>""", unsafe_allow_html=True)
