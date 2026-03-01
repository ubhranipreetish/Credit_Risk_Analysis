import pandas as pd
import numpy as np

def get_logistic_coefficients(pipeline):
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]

    coefficients = classifier.coef_[0]

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Absolute Impact": np.abs(coefficients)
    }).sort_values(by="Absolute Impact", ascending=False)

    return coef_df


def get_decision_tree_importance(pipeline):
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": classifier.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return importance_df