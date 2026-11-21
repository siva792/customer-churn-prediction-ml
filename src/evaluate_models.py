from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_curve,
)

from src.preprocessing import (
    clean_data,
    get_dataset_path,
    load_raw_data,
    prepare_features_and_target,
    split_and_preprocess,
)
from src.train_models import train_and_evaluate_models


def get_eval_charts_dir() -> Path:
    """Create and return directory for saving evaluation visualizations."""
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "reports" / "evaluation_charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_metrics_comparison(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Plot grouped bar chart comparing Accuracy, Precision, Recall, F1-Score, and ROC-AUC."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    plot_df = results_df.set_index("Model")[metrics]

    ax = plot_df.plot(kind="bar", figsize=(10, 6), rot=0, colormap="Set2")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8, fontweight="bold")

    plt.title("Model Evaluation Metrics Comparison (Test Set)")
    plt.xlabel("Classification Model")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "01_model_metrics_comparison.png", dpi=150)
    plt.close()


def plot_confusion_matrices(trained_models: dict, X_test_proc, y_test, out_dir: Path) -> None:
    """Plot side-by-side confusion matrices for all trained models."""
    fig, axes = plt.subplots(1, len(trained_models), figsize=(15, 4.5))
    if len(trained_models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, trained_models.items()):
        y_pred = model.predict(X_test_proc)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No (0)", "Yes (1)"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}\nConfusion Matrix")

    plt.tight_layout()
    plt.savefig(out_dir / "02_confusion_matrices.png", dpi=150)
    plt.close()


def plot_roc_curves(trained_models: dict, X_test_proc, y_test, out_dir: Path) -> None:
    """Plot ROC curves and AUC scores for all models on a single figure."""
    plt.figure(figsize=(8, 6))

    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test_proc)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = np.trapezoid(tpr, fpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier (AUC = 0.5000)")
    plt.title("ROC Curves Comparison (Test Set)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "03_roc_curves.png", dpi=150)
    plt.close()


def main() -> None:
    df = clean_data(load_raw_data(get_dataset_path()))
    X, y = prepare_features_and_target(df, target_col="Churn", drop_id=True)

    (
        _,
        _,
        y_train,
        y_test,
        X_train_proc,
        X_test_proc,
        _,
        _,
    ) = split_and_preprocess(X, y, test_size=0.2, random_state=42)

    trained_models, results_df = train_and_evaluate_models(
        X_train_proc, X_test_proc, y_train, y_test, random_state=42
    )

    out_dir = get_eval_charts_dir()
    plot_metrics_comparison(results_df, out_dir)
    plot_confusion_matrices(trained_models, X_test_proc, y_test, out_dir)
    plot_roc_curves(trained_models, X_test_proc, y_test, out_dir)

    print("=" * 75)
    print("MODULE 5: DETAILED MODEL EVALUATION")
    print("=" * 75)
    print(results_df.to_string(index=False))
    print()

    for name, model in trained_models.items():
        y_pred = model.predict(X_test_proc)
        print("-" * 75)
        print(f"Classification Report: {name}")
        print("-" * 75)
        print(
            classification_report(
                y_test, y_pred, target_names=["Retained (0)", "Churned (1)"], digits=4
            )
        )

    print(f"Saved evaluation charts to: {out_dir}")


if __name__ == "__main__":
    main()