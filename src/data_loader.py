import pandas as pd

DATA_PATH = "data/credit_risk_dataset.csv"


def print_missing_value_report(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
        return
    print("\n===== Missing Value Report =====")
    for col, count in missing.items():
        pct = count / len(df) * 100
        print(f"  {col}: {count} ({pct:.2f}%)")
    print(f"  Total rows: {len(df)}")
    print("================================\n")


def clean_data(df):
    original_len = len(df)

    # Remove unrealistic age values (> 100 years)
    df = df[df["person_age"] <= 100]

    # Remove unrealistic employment length (> 60 years)
    df = df[df["person_emp_length"] <= 60]

    removed = original_len - len(df)
    if removed > 0:
        print(f"Outlier removal: dropped {removed} rows "
              f"(person_age > 100 or person_emp_length > 60)")

    return df.reset_index(drop=True)


def load_data(clean=True):
    df = pd.read_csv(DATA_PATH)
    print_missing_value_report(df)
    if clean:
        df = clean_data(df)
    return df