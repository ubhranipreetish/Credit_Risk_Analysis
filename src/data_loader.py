import pandas as pd

DATA_PATH = "data/credit_risk_dataset.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df