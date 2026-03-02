from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def build_logistic_pipeline(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, solver="saga", max_iter=2000))
    ])


def build_decision_tree_pipeline(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))
    ])


def build_tuned_logistic_pipeline(preprocessor, X_train, y_train):
    pipe = build_logistic_pipeline(preprocessor)
    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    }
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc",
                        n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Logistic Regression best params: {grid.best_params_}")
    print(f"Logistic Regression best ROC-AUC (CV): {grid.best_score_:.4f}")
    return grid.best_estimator_


def build_tuned_decision_tree_pipeline(preprocessor, X_train, y_train):
    pipe = build_decision_tree_pipeline(preprocessor)
    param_grid = {
        "classifier__max_depth": [3, 5, 7, 10],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    }
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc",
                        n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Decision Tree best params: {grid.best_params_}")
    print(f"Decision Tree best ROC-AUC (CV): {grid.best_score_:.4f}")
    return grid.best_estimator_