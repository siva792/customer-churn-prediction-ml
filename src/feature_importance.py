from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing import (
    clean_data,
    get_dataset_path,
    load_raw_data,
    prepare_features_and_target,
    split_and_preprocess,
)
from src.train_models import train_and_evaluate_models


def get_fi_charts_dir() -> Path:
    """Create and return directory for saving feature importance visualizations."""
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "reports" / "feature_importance_charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_importance_tables(trained_models: dict, feature_names) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build sorted feature importance tables for Random Forest and Logistic Regression."""
    rf_model = trained_models["Random Forest"]
    rf_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": rf_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    lr_model = trained_models["Logistic Regression"]
    lr_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": lr_model.coef_[0],
            "Abs_Coefficient": abs(lr_model.coef_[0]),
        }
    ).sort_values("Abs_Coefficient", ascending=False)

    return rf_df.reset_index(drop=True), lr_df.reset_index(drop=True)


def plot_rf_top_features(rf_df: pd.DataFrame, out_dir: Path, top_n: int = 12) -> None:
    """Plot top N most important features from Random Forest."""
    top_df = rf_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_df["Feature"], top_df["Importance"], color="#2b8cbe")
    for bar in bars:
        val = bar.get_width()
        plt.annotate(
            f"{val:.4f}",
            (val, bar.get_y() + bar.get_height() / 2),
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
    plt.title(f"Top {top_n} Most Important Features — Random Forest")
    plt.xlabel("Gini Importance Score")
    plt.ylabel("Encoded Feature")
    plt.xlim(0, top_df["Importance"].max() * 1.18)
    plt.tight_layout()
    plt.savefig(out_dir / "01_random_forest_feature_importance.png", dpi=150)
    plt.close()


def plot_lr_top_coefficients(lr_df: pd.DataFrame, out_dir: Path, top_n: int = 12) -> None:
    """Plot top N directional coefficients from Logistic Regression."""
    top_df = lr_df.head(top_n).iloc[::-1]
    colors = ["#e34a33" if c > 0 else "#1b9e77" for c in top_df["Coefficient"]]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_df["Feature"], top_df["Coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    for bar in bars:
        val = bar.get_width()
        ha = "left" if val >= 0 else "right"
        offset = 0.02 if val >= 0 else -0.02
        plt.annotate(
            f"{val:+.3f}",
            (val + offset, bar.get_y() + bar.get_height() / 2),
            ha=ha,
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
    plt.title(f"Top {top_n} Directional Drivers — Logistic Regression (Red = Higher Churn, Green = Lower)")
    plt.xlabel("Model Coefficient (Log-Odds Impact)")
    plt.ylabel("Encoded Feature")
    plt.tight_layout()
    plt.savefig(out_dir / "02_logistic_regression_coefficients.png", dpi=150)
    plt.close()


def print_retention_recommendations() -> None:
    """Print practical retention recommendations tied to top churn drivers."""
    print("=" * 75)
    print("BUSINESS RETENTION RECOMMENDATIONS FROM KEY DRIVERS")
    print("=" * 75)
    print("1. Contract Incentives: Encourage month-to-month customers to switch to 1-year or")
    print("   2-year plans using discounts, waived installation/upgrade fees, or bonus perks.")
    print("2. Early-Tenure Onboarding: Prioritize new customers in their first 0-12 months with")
    print("   guided onboarding, proactive quality checks, and early loyalty rewards.")
    print("3. Support & Security Bundles: Offer trial/bundled OnlineSecurity and TechSupport for")
    print("   high-charge and Fiber optic customers to reduce dissatisfaction and churn.")
    print("4. Autopay Migration: Nudge Electronic check users toward automatic card or bank")
    print("   payments with clear billing reminders or minor recurring-payment benefits.")


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
        feature_names,
    ) = split_and_preprocess(X, y, test_size=0.2, random_state=42)

    trained_models, _ = train_and_evaluate_models(
        X_train_proc, X_test_proc, y_train, y_test, random_state=42
    )

    rf_df, lr_df = get_importance_tables(trained_models, feature_names)
    out_dir = get_fi_charts_dir()
    plot_rf_top_features(rf_df, out_dir, top_n=12)
    plot_lr_top_coefficients(lr_df, out_dir, top_n=12)

    print("=" * 75)
    print("MODULE 6: TOP 10 RANDOM FOREST FEATURE IMPORTANCES")
    print("=" * 75)
    print(rf_df.head(10).to_string(index=False))
    print()

    print("=" * 75)
    print("MODULE 6: TOP 10 LOGISTIC REGRESSION COEFFICIENTS (BY MAGNITUDE)")
    print("=" * 75)
    print(lr_df.head(10).to_string(index=False))
    print()

    print_retention_recommendations()
    print(f"\nSaved feature importance charts to: {out_dir}")


if __name__ == "__main__":
    main()