# --------------------------------------------
# CREDIT RISK - MODEL EXPLAINABILITY
# --------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, FunctionTransformer, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("data/credit_risk_dataset.csv")

# Define columns
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

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.7,
    random_state=42,
    stratify=y
)

# Encode target
target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(y_train)
y_test = target_encoder.transform(y_test)

# Preprocessing
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

# Logistic Regression Pipeline
log_reg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=42, solver="liblinear"))
])

# Decision Tree Pipeline
dt_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))
])

# Train models
log_reg_pipeline.fit(X_train, y_train)
dt_pipeline.fit(X_train, y_train)

# Get feature names
feature_names = log_reg_pipeline.named_steps["preprocessor"].get_feature_names_out()

# ------------------------------
# Logistic Regression Coefficients
# ------------------------------
log_model = log_reg_pipeline.named_steps["classifier"]
coefficients = log_model.coef_[0]

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute Impact": np.abs(coefficients)
}).sort_values(by="Absolute Impact", ascending=False)

print("\nTop 10 Logistic Regression Risk Drivers:")
print(coef_df.head(10))

# Plot
plt.figure()
plt.barh(coef_df["Feature"][:10], coef_df["Coefficient"][:10])
plt.title("Top 10 Logistic Regression Risk Drivers")
plt.gca().invert_yaxis()
plt.show()

# ------------------------------
# Decision Tree Feature Importance
# ------------------------------
dt_model = dt_pipeline.named_steps["classifier"]
importances = dt_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Decision Tree Risk Drivers:")
print(importance_df.head(10))

plt.figure()
plt.barh(importance_df["Feature"][:10], importance_df["Importance"][:10])
plt.title("Top 10 Decision Tree Feature Importance")
plt.gca().invert_yaxis()
plt.show()