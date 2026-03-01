import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/credit_risk_dataset.csv")

target_col = "loan_status"

X = df.drop(columns=[target_col])
y = df[target_col]

print("Feature Matrix Shape:", X.shape)
print("Target Vector Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.7,
    random_state=42,
    stratify=y
)

print("\nTraining Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting Target Distribution:")
print(y_test.value_counts(normalize=True))