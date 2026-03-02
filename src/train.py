import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Allow imports from src/ regardless of working directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_data
from preprocessing import build_preprocessor
from model_builder import build_tuned_logistic_pipeline, build_tuned_decision_tree_pipeline
from evaluate import evaluate_model, save_metrics_json, generate_report_images
from explain import get_logistic_coefficients, get_decision_tree_importance

# Import shared column config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
DATA_PATH_OVERRIDE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "credit_risk_dataset.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# --- Load & Clean Data ---
import data_loader as dl
dl.DATA_PATH = DATA_PATH_OVERRIDE
df = load_data(clean=True)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42, stratify=y
)

target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(y_train)
y_test = target_encoder.transform(y_test)

preprocessor = build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS)

# --- Train with Hyperparameter Tuning ---
print("\n--- Training Logistic Regression with GridSearchCV ---")
log_reg_pipeline = build_tuned_logistic_pipeline(preprocessor, X_train, y_train)

print("\n--- Training Decision Tree with GridSearchCV ---")
dt_pipeline = build_tuned_decision_tree_pipeline(preprocessor, X_train, y_train)

# --- Evaluate ---
log_metrics = evaluate_model("Logistic Regression", log_reg_pipeline, X_test, y_test)
dt_metrics = evaluate_model("Decision Tree", dt_pipeline, X_test, y_test)

print("\nModel Comparison:")
print({
    "Logistic Regression": log_metrics,
    "Decision Tree": dt_metrics
})

# --- Save Metrics to JSON ---
all_metrics = {
    "Logistic Regression": log_metrics,
    "Decision Tree": dt_metrics
}
save_metrics_json(all_metrics, os.path.join(REPORT_DIR, "metrics.json"))

# --- Explainability ---
log_coef_df = get_logistic_coefficients(log_reg_pipeline)
dt_importance_df = get_decision_tree_importance(dt_pipeline)

print("\nTop 5 Logistic Regression Risk Drivers:")
print(log_coef_df.head(5))

print("\nTop 5 Decision Tree Risk Drivers:")
print(dt_importance_df.head(5))

# --- Generate Report Images ---
models = {
    "Logistic Regression": log_reg_pipeline,
    "Decision Tree": dt_pipeline
}
generate_report_images(models, X_test, y_test, REPORT_DIR)

# --- Save Models ---
joblib.dump(log_reg_pipeline, os.path.join(MODEL_DIR, "logistic_pipeline.joblib"))
joblib.dump(dt_pipeline, os.path.join(MODEL_DIR, "decision_tree_pipeline.joblib"))
joblib.dump(target_encoder, os.path.join(MODEL_DIR, "target_encoder.joblib"))

print("\nModels saved successfully.")
print(f"Reports saved to: {REPORT_DIR}")