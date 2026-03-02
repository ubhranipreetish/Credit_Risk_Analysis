import os
import sys
import json
import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.explain import get_logistic_coefficients, get_decision_tree_importance
from src.preprocessing import build_preprocessor
from src.model_builder import build_tuned_logistic_pipeline, build_tuned_decision_tree_pipeline
from src.data_loader import load_data
from src.evaluate import evaluate_model, save_metrics_json, generate_report_images
from src import NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL

def auto_train():
    model_dir = "models"
    report_dir = "reports"
    if (os.path.exists(f"{model_dir}/decision_tree_pipeline.joblib") and
            os.path.exists(f"{model_dir}/logistic_pipeline.joblib") and
            os.path.exists(f"{report_dir}/metrics.json")):
        return
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    with st.spinner("First run detected — training models with hyperparameter tuning, please wait…"):
        df = load_data(clean=True)
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)
        enc = LabelEncoder()
        y_train_enc = enc.fit_transform(y_train)
        y_test_enc = enc.transform(y_test)
        preprocessor = build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS)
        lr_pipe = build_tuned_logistic_pipeline(preprocessor, X_train, y_train_enc)
        dt_pipe = build_tuned_decision_tree_pipeline(preprocessor, X_train, y_train_enc)
        lr_metrics = evaluate_model("Logistic Regression", lr_pipe, X_test, y_test_enc)
        dt_metrics = evaluate_model("Decision Tree", dt_pipe, X_test, y_test_enc)
        all_metrics = {"Logistic Regression": lr_metrics, "Decision Tree": dt_metrics}
        save_metrics_json(all_metrics, f"{report_dir}/metrics.json")
        generate_report_images({"Logistic Regression": lr_pipe, "Decision Tree": dt_pipe},
                               X_test, y_test_enc, report_dir)
        joblib.dump(lr_pipe, f"{model_dir}/logistic_pipeline.joblib")
        joblib.dump(dt_pipe, f"{model_dir}/decision_tree_pipeline.joblib")
        joblib.dump(enc,     f"{model_dir}/target_encoder.joblib")

st.set_page_config(page_title="Credit Risk Analyzer", layout="wide", initial_sidebar_state="collapsed")

auto_train()

# -- Styling --
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #0d1117, #161b22);
    color: #c9d1d9;
}

h1, h2, h3 { color: #e6edf3 !important; font-weight: 600 !important; }

section[data-testid="stSidebar"] { display: none; }

/* inputs */
input[type="number"], div[data-baseweb="input"] input {
    background: #fff !important;
    border: 1px solid #3a4a5c !important;
    border-radius: 8px !important;
    color: #1e293b !important;
}
input[type="number"]:focus {
    border-color: #5b8def !important;
    box-shadow: 0 0 0 2px rgba(91,141,239,0.2) !important;
}

/* selects */
div[data-baseweb="select"] > div {
    background: rgba(22, 27, 34, 0.9) !important;
    border: 1px solid #3a4a5c !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
}
div[data-baseweb="select"] span { color: #c9d1d9 !important; }
ul[data-testid="stSelectboxVirtualDropdown"] {
    background: #161b22 !important;
    border: 1px solid #3a4a5c !important;
    border-radius: 8px !important;
}
ul[data-testid="stSelectboxVirtualDropdown"] li { color: #c9d1d9 !important; }

/* buttons */
.stButton > button {
    background: #238636;
    color: #fff;
    font-weight: 600;
    border: none; border-radius: 8px;
    padding: 0.7rem 1.8rem; width: 100%;
    transition: background 0.2s ease;
    margin-top: 6px;
}
.stButton > button:hover { background: #2ea043; }

/* expander */
details {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    padding: 6px 12px !important;
}
summary { color: #e6edf3 !important; font-weight: 600 !important; }

/* alerts */
div[data-testid="stAlertContainer"] > div {
    border-radius: 10px !important;
    border: 1px solid #30363d !important;
    padding: 0.9rem 1.1rem !important;
}

/* metrics */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    padding: 0.9rem 1.1rem !important;
}
div[data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 1.6rem !important; font-weight: 700 !important; }
div[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.82rem !important; text-transform: uppercase; }

label, .stSelectbox label, .stNumberInput label { color: #8b949e !important; font-size: 0.88rem !important; font-weight: 500 !important; }

.stDataFrame { border-radius: 10px !important; overflow: hidden; border: 1px solid #30363d !important; }

hr { border-color: #30363d !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# -- Header --
st.markdown("<h1 style='text-align:center; margin-bottom:0;'>Credit Risk Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#8b949e; margin-top:4px;'>Evaluate borrower risk profiles using trained ML models</p>", unsafe_allow_html=True)

# -- Load Models --
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

# -- Model Selection --
sel_col, _ = st.columns([1, 2])
with sel_col:
    model_choice = st.selectbox("Select Model", ["Decision Tree", "Logistic Regression"])
model = dt_pipeline if model_choice == "Decision Tree" else lr_pipeline

# -- Metrics + Charts (always show all 4 images) --
with st.expander("Model Performance — Test Set Metrics", expanded=False):
    try:
        with open("reports/metrics.json", "r") as f:
            ALL_METRICS = json.load(f)
    except FileNotFoundError:
        ALL_METRICS = {}
        st.warning("Metrics not found. Please retrain models.")

    if model_choice in ALL_METRICS:
        m = ALL_METRICS[model_choice]
        mc1, mc2, _ = st.columns(3)
        mc1.metric("Accuracy", f"{m['Accuracy']*100:.2f}%")
        mc2.metric("ROC-AUC",  f"{m['ROC_AUC']*100:.2f}%")
        st.markdown("---")
        cm = m.get("Confusion_Matrix", [])
        if cm:
            cm_df = pd.DataFrame(cm,
                                  columns=["Predicted Good (0)", "Predicted Bad (1)"],
                                  index=["Actual Good (0)", "Actual Bad (1)"])
            st.dataframe(cm_df, use_container_width=True)

    # Always show all 4 report images
    st.markdown("---")
    st.markdown("#### Evaluation Charts")
    img_col1, img_col2 = st.columns(2)
    if os.path.exists("reports/confusion_matrix_logistic_regression.png"):
        img_col1.image("reports/confusion_matrix_logistic_regression.png",
                       caption="Logistic Regression — Confusion Matrix", use_container_width=True)
    if os.path.exists("reports/confusion_matrix_decision_tree.png"):
        img_col2.image("reports/confusion_matrix_decision_tree.png",
                       caption="Decision Tree — Confusion Matrix", use_container_width=True)
    img_col3, img_col4 = st.columns(2)
    if os.path.exists("reports/roc_curves.png"):
        img_col3.image("reports/roc_curves.png",
                       caption="ROC Curves Comparison", use_container_width=True)
    if os.path.exists("reports/model_comparison.png"):
        img_col4.image("reports/model_comparison.png",
                       caption="Accuracy vs ROC-AUC", use_container_width=True)

st.markdown("---")

# -- Input Form --
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### Personal Information")
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30,
                                 step=1, help="Borrower's current age (18-100)")
    person_income = st.number_input("Annual Income ($)", min_value=0, value=50000,
                                    step=5000, help="Total yearly income before taxes")
    person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, value=5.0,
                                        step=0.5, help="How long the borrower has been employed at current job")
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"],
                                         help="Current living situation of the borrower")

with col2:
    st.markdown("### Loan Details")
    loan_amnt = st.number_input("Loan Amount ($)", min_value=0, max_value=1_000_000, value=10000,
                                step=1000, help="Requested loan amount in dollars")
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=10.0,
                                    step=0.5, help="Annual interest rate on the loan")
    loan_intent = st.selectbox("Loan Intent",
                               ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
                               help="Purpose of the loan")
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"],
                              help="A = lowest risk/rate, G = highest risk/rate. "
                                   "Assigned by the lender based on credit score and history.")

st.markdown("---")
st.markdown("### Credit History")
col3, col4 = st.columns(2, gap="large")
with col3:
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5,
                                                  step=1, help="Number of years the borrower has had credit accounts")
with col4:
    cb_person_default_on_file = st.selectbox("Historical Default on File", ["N", "Y"],
                                             help="Whether the borrower has previously defaulted on a loan")

loan_percent_income = (loan_amnt / person_income) if person_income > 0 else 0.0

# -- Validation Warnings --
validation_msgs = []
if person_emp_length > (person_age - 18):
    validation_msgs.append(f"Employment length ({person_emp_length:.0f} yrs) exceeds possible working years for age {person_age}.")
if cb_person_cred_hist_length > (person_age - 18):
    validation_msgs.append(f"Credit history ({cb_person_cred_hist_length} yrs) exceeds possible history for age {person_age}.")
if loan_percent_income > 0.8:
    validation_msgs.append(f"Loan-to-income ratio is very high ({loan_percent_income:.0%}). Lenders typically cap at 40-50%.")
if person_income > 0 and person_income < 10000 and loan_amnt > 50000:
    validation_msgs.append("Loan amount seems disproportionately large relative to income.")
if person_age < 21 and person_home_ownership == "OWN":
    validation_msgs.append("Home ownership under age 21 is unusual and may affect prediction reliability.")

for msg in validation_msgs:
    st.warning(msg)

st.markdown("---")

# -- Evaluate Button --
run_clicked = st.button("Evaluate Credit Risk")

# -- Prediction --
if run_clicked:
    with st.spinner("Running inference…"):
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

    st.markdown("---")
    st.markdown("<div id='result-anchor'></div>", unsafe_allow_html=True)
    st.markdown("### Assessment Result")

    if status == 1:
        st.error("**High Risk** — This profile matches historical default patterns.")
        verdict_color = "#da3633"
    else:
        st.success("**Low Risk** — This profile suggests successful repayment is likely.")
        verdict_color = "#238636"

    # probability meter
    try:
        proba = model.predict_proba(input_df)[0][1]
        bar_pct = int(proba * 100)
        # Pick color based on risk level
        if proba < 0.3:
            prob_color = "#238636"
            risk_label = "Low"
        elif proba < 0.6:
            prob_color = "#d29922"
            risk_label = "Moderate"
        else:
            prob_color = "#da3633"
            risk_label = "High"
        st.markdown(f"""
        <div style='margin: 1.2rem 0;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                <span style='color:#8b949e; font-size:0.9rem;'>Default Probability</span>
                <span style='color:{prob_color}; font-weight:700; font-size:1.1rem;'>{proba:.1%} ({risk_label})</span>
            </div>
            <div style='position:relative; width:100%; height:20px; background:#21262d; border-radius:6px; overflow:hidden; border:1px solid #30363d;'>
                <!-- colored segments -->
                <div style='position:absolute; top:0; left:0; width:30%; height:100%; background:#16331d;'></div>
                <div style='position:absolute; top:0; left:30%; width:30%; height:100%; background:#2d2a0e;'></div>
                <div style='position:absolute; top:0; left:60%; width:40%; height:100%; background:#3d1214;'></div>
                <!-- marker -->
                <div style='position:absolute; top:0; left:{bar_pct}%; transform:translateX(-50%); width:3px; height:100%; background:#fff; box-shadow:0 0 4px rgba(255,255,255,0.5);'></div>
            </div>
            <div style='display:flex; justify-content:space-between; margin-top:4px; font-size:0.75rem; color:#484f58;'>
                <span>0%</span>
                <span>30%</span>
                <span>60%</span>
                <span>100%</span>
            </div>
            <div style='display:flex; justify-content:space-between; margin-top:1px; font-size:0.7rem;'>
                <span style='color:#238636;'>Low Risk</span>
                <span style='color:#d29922; text-align:center;'>Moderate</span>
                <span style='color:#da3633; text-align:right;'>High Risk</span>
            </div>
        </div>""", unsafe_allow_html=True)
    except Exception:
        st.warning("Probability estimation not available for this model.")

    st.markdown("---")
    st.markdown("### Top Risk Drivers")

    def format_feature(name):
        """Strip sklearn column prefixes and format for display."""
        return name.replace("num__", "").replace("cat__", "").replace("_", " ").title()

    if model_choice == "Decision Tree":
        for _, row in get_decision_tree_importance(model).head(3).iterrows():
            feat = format_feature(row["Feature"])
            val  = row["Importance"]
            bar_w = min(int(val * 380), 100)
            st.markdown(f"""
            <div style='margin-bottom:12px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                    <span style='color:#c9d1d9;'>{feat}</span>
                    <span style='color:#58a6ff; font-weight:600;'>{val:.1%}</span>
                </div>
                <div style='height:6px; background:rgba(255,255,255,0.06); border-radius:3px; overflow:hidden;'>
                    <div style='height:100%; width:{bar_w}%;
                         background:#58a6ff; border-radius:3px;'></div>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        for _, row in get_logistic_coefficients(model).head(3).iterrows():
            feat   = format_feature(row["Feature"])
            impact = row["Absolute Impact"]
            is_risk = row["Coefficient"] > 0
            direction = "Increases Risk" if is_risk else "Reduces Risk"
            color = "#da3633" if is_risk else "#238636"
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03); border:1px solid #30363d;
                         border-radius:8px; padding:10px 14px; margin-bottom:10px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#c9d1d9;'>{feat}</span>
                    <span style='color:{color}; font-size:0.85rem; font-weight:600;'>{direction}</span>
                </div>
                <span style='color:#484f58; font-size:0.8rem;'>Impact: {impact:.3f}</span>
            </div>""", unsafe_allow_html=True)

    # auto-scroll to results
    st.components.v1.html("""
    <script>
        function scrollToBottom() {
            try {
                // Try multiple Streamlit DOM selectors
                var selectors = [
                    'section.main .block-container',
                    'section.main',
                    '[data-testid="stAppViewContainer"]',
                    '.main'
                ];
                var doc = window.parent.document;
                for (var i = 0; i < selectors.length; i++) {
                    var el = doc.querySelector(selectors[i]);
                    if (el) {
                        el.scrollTop = el.scrollHeight;
                        break;
                    }
                }
            } catch(e) {
                // fallback: scroll the parent window itself
                try { window.parent.scrollTo(0, 999999); } catch(e2) {}
            }
        }
        // Run after a short delay to let Streamlit finish rendering
        setTimeout(scrollToBottom, 100);
        setTimeout(scrollToBottom, 500);
    </script>
    """, height=0)
