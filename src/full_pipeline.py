import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    FunctionTransformer,
    LabelEncoder
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import joblib


# --------------------------------------------
# 1. LOAD DATA
# --------------------------------------------

DATA_PATH = "data/credit_risk_dataset.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

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


# --------------------------------------------
# 2. TRAIN-TEST SPLIT
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.7,
    random_state=42,
    stratify=y
)

target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(y_train)
y_test = target_encoder.transform(y_test)


# --------------------------------------------
# 3. PREPROCESSING
# --------------------------------------------

log_transformer = FunctionTransformer(np.log1p, feature_names_out="one-to-one")

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("log", log_transformer),
    ("scaler", MinMaxScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_cols),
    ("cat", cat_pipeline, categorical_cols)
])


# --------------------------------------------
# 4. MODEL PIPELINES
# --------------------------------------------

log_reg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=42, solver="liblinear"))
])

dt_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))
])


# --------------------------------------------
# 5. TRAIN MODELS
# --------------------------------------------

log_reg_pipeline.fit(X_train, y_train)
dt_pipeline.fit(X_train, y_train)


# --------------------------------------------
# 6. EVALUATION FUNCTION
# --------------------------------------------

def evaluate_model(name, pipeline):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"\n===== {name} =====")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.show()

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob)
    }


log_metrics = evaluate_model("Logistic Regression", log_reg_pipeline)
dt_metrics = evaluate_model("Decision Tree", dt_pipeline)


# --------------------------------------------
# 7. MODEL COMPARISON
# --------------------------------------------

comparison_df = pd.DataFrame({
    "Logistic Regression": log_metrics,
    "Decision Tree": dt_metrics
})

print("\n===== MODEL COMPARISON =====")
print(comparison_df)


# --------------------------------------------
# 8. EXPLAINABILITY
# --------------------------------------------

feature_names = log_reg_pipeline.named_steps["preprocessor"].get_feature_names_out()

# Logistic Coefficients
log_model = log_reg_pipeline.named_steps["classifier"]
coefficients = log_model.coef_[0]

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute Impact": np.abs(coefficients)
}).sort_values(by="Absolute Impact", ascending=False)

print("\nTop 10 Logistic Regression Risk Drivers:")
print(coef_df.head(10))

# Decision Tree Importance
dt_model = dt_pipeline.named_steps["classifier"]
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": dt_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Decision Tree Risk Drivers:")
print(importance_df.head(10))