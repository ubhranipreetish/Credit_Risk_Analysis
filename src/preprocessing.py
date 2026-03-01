import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, FunctionTransformer

def build_preprocessor(numerical_cols, categorical_cols):

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

    return preprocessor