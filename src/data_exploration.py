import pandas as pd

df = pd.read_csv("data/credit_risk_dataset.csv")

print("First 10 rows of dataset:")
print(df.head(10))

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nLoan Status Distribution:")
print(df['loan_status'].value_counts())