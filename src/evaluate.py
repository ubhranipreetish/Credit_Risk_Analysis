import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             roc_curve, ConfusionMatrixDisplay)


def evaluate_model(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {
        "Accuracy": round(acc, 4),
        "ROC_AUC": round(auc, 4),
        "Confusion_Matrix": cm.tolist()
    }


def save_metrics_json(metrics_dict, path="reports/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to {path}")


def generate_report_images(models_dict, X_test, y_test,
                           output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)

    # --- Confusion Matrix Heatmaps (one per model) ---
    for name, pipeline in models_dict.items():
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Good (0)", "Bad (1)"])
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"{name} — Confusion Matrix", fontsize=14, fontweight="bold")
        fig.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(output_dir, f"confusion_matrix_{safe_name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    # --- ROC Curves (both models on one plot) ---
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#6366f1", "#ef4444"]
    for i, (name, pipeline) in enumerate(models_dict.items()):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Accuracy Comparison Bar Chart ---
    names = []
    accuracies = []
    aucs = []
    for name, pipeline in models_dict.items():
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        names.append(name)
        accuracies.append(accuracy_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_prob))

    x = np.arange(len(names))
    width = 0.3
    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy",
                   color="#6366f1", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, aucs, width, label="ROC-AUC",
                   color="#8b5cf6", edgecolor="white", linewidth=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Accuracy vs ROC-AUC",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")