import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_loader import load_data
from preprocessing import build_preprocessor
from model_builder import build_logistic_pipeline, build_decision_tree_pipeline
from evaluate import evaluate_model
from explain import get_logistic_coefficients, get_decision_tree_importance

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = load_data()

numerical_cols = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length'
]

categorical_cols = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

target_col = "loan_status"

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42, stratify=y
)

target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(y_train)
y_test = target_encoder.transform(y_test)

# Build pipelines
preprocessor = build_preprocessor(numerical_cols, categorical_cols)

log_reg_pipeline = build_logistic_pipeline(preprocessor)
dt_pipeline = build_decision_tree_pipeline(preprocessor)

# Train
log_reg_pipeline.fit(X_train, y_train)
dt_pipeline.fit(X_train, y_train)

# Evaluate
log_metrics = evaluate_model("Logistic Regression", log_reg_pipeline, X_test, y_test)
dt_metrics = evaluate_model("Decision Tree", dt_pipeline, X_test, y_test)

print("\nModel Comparison:")
print({
    "Logistic Regression": log_metrics,
    "Decision Tree": dt_metrics
})

# Explainability
log_coef_df = get_logistic_coefficients(log_reg_pipeline)
dt_importance_df = get_decision_tree_importance(dt_pipeline)

print("\nTop 5 Logistic Regression Risk Drivers:")
print(log_coef_df.head(5))

print("\nTop 5 Decision Tree Risk Drivers:")
print(dt_importance_df.head(5))

# Save models
joblib.dump(log_reg_pipeline, f"{MODEL_DIR}/logistic_pipeline.joblib")
joblib.dump(dt_pipeline, f"{MODEL_DIR}/decision_tree_pipeline.joblib")
joblib.dump(target_encoder, f"{MODEL_DIR}/target_encoder.joblib")

print("\nModels saved successfully.")