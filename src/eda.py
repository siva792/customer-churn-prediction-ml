from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocessing import clean_data, get_dataset_path, load_raw_data


def get_charts_dir() -> Path:
    """Create and return the directory for saving EDA visualizations."""
    project_root = Path(__file__).resolve().parent.parent
    charts_dir = project_root / "reports" / "eda_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    return charts_dir


def prepare_eda_df() -> pd.DataFrame:
    """Load and clean dataset, adding binary Churn indicator and tenure buckets."""
    df = clean_data(load_raw_data(get_dataset_path()))
    df["Churn_bin"] = df["Churn"].map({"No": 0, "Yes": 1})
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 mos", "13-24 mos", "25-48 mos", "49-72 mos"],
    )
    return df


def plot_1_overall_churn(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(7, 5))
    order = ["No", "Yes"]
    ax = sns.countplot(
        data=df,
        x="Churn",
        hue="Churn",
        order=order,
        hue_order=order,
        palette=["#2b8cbe", "#e34a33"],
        legend=False,
    )
    total = len(df)
    for p in ax.patches:
        count = int(p.get_height())
        pct = count / total * 100
        ax.annotate(
            f"{count:,}\n({pct:.1f}%)",
            (p.get_x() + p.get_width() / 2, count),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.title("1. Overall Customer Churn Distribution")
    plt.xlabel("Churn Status (No = Retained, Yes = Churned)")
    plt.ylabel("Number of Customers")
    plt.ylim(0, total * 0.85)
    plt.tight_layout()
    plt.savefig(out_dir / "01_overall_churn.png", dpi=150)
    plt.close()


def plot_2_gender_churn(df: pd.DataFrame, out_dir: Path) -> None:
    rates = (df.groupby("gender", observed=False)["Churn_bin"].mean() * 100).reset_index()
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=rates,
        x="gender",
        y="Churn_bin",
        hue="gender",
        palette=["#756bb1", "#31a354"],
        legend=False,
    )
    for p in ax.patches:
        val = p.get_height()
        ax.annotate(
            f"{val:.2f}%",
            (p.get_x() + p.get_width() / 2, val),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.title("2. Churn Rate by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Churn Rate (%)")
    plt.ylim(0, 40)
    plt.tight_layout()
    plt.savefig(out_dir / "02_gender_churn.png", dpi=150)
    plt.close()


def plot_3_contract_churn(df: pd.DataFrame, out_dir: Path) -> None:
    order = ["Month-to-month", "One year", "Two year"]
    rates = (
        df.groupby("Contract", observed=False)["Churn_bin"]
        .mean()
        .reindex(order)
        .mul(100)
        .reset_index()
    )
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=rates,
        x="Contract",
        y="Churn_bin",
        hue="Contract",
        order=order,
        hue_order=order,
        palette=["#d95f02", "#7570b3", "#1b9e77"],
        legend=False,
    )
    for p in ax.patches:
        val = p.get_height()
        ax.annotate(
            f"{val:.2f}%",
            (p.get_x() + p.get_width() / 2, val),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.title("3. Churn Rate by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Churn Rate (%)")
    plt.ylim(0, 52)
    plt.tight_layout()
    plt.savefig(out_dir / "03_contract_churn.png", dpi=150)
    plt.close()


def plot_4_tenure_churn(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.boxplot(
        data=df,
        x="Churn",
        y="tenure",
        hue="Churn",
        order=["No", "Yes"],
        hue_order=["No", "Yes"],
        ax=axes[0],
        palette=["#2b8cbe", "#e34a33"],
        legend=False,
    )
    axes[0].set_title("Tenure Distribution by Churn")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Tenure (Months)")

    grp_rates = (df.groupby("TenureGroup", observed=False)["Churn_bin"].mean() * 100).reset_index()
    sns.barplot(
        data=grp_rates,
        x="TenureGroup",
        y="Churn_bin",
        hue="TenureGroup",
        ax=axes[1],
        palette="Blues_r",
        legend=False,
    )
    for p in axes[1].patches:
        val = p.get_height()
        axes[1].annotate(
            f"{val:.1f}%",
            (p.get_x() + p.get_width() / 2, val),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axes[1].set_title("Churn Rate by Tenure Group")
    axes[1].set_xlabel("Tenure Group")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].set_ylim(0, 60)

    fig.suptitle("4. Tenure-Wise Churn Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "04_tenure_churn.png", dpi=150)
    plt.close()


def plot_5_monthly_charges_churn(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges",
        hue="Churn",
        order=["No", "Yes"],
        hue_order=["No", "Yes"],
        palette=["#2b8cbe", "#e34a33"],
        legend=False,
    )
    plt.title("5. Monthly Charges vs. Churn")
    plt.xlabel("Churn Status")
    plt.ylabel("Monthly Charges ($)")
    plt.tight_layout()
    plt.savefig(out_dir / "05_monthly_charges_churn.png", dpi=150)
    plt.close()


def plot_6_payment_method_churn(df: pd.DataFrame, out_dir: Path) -> None:
    rates = (
        df.groupby("PaymentMethod", observed=False)["Churn_bin"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=rates,
        y="PaymentMethod",
        x="Churn_bin",
        hue="PaymentMethod",
        palette="Reds_r",
        legend=False,
    )
    for p in ax.patches:
        val = p.get_width()
        ax.annotate(
            f"{val:.2f}%",
            (val, p.get_y() + p.get_height() / 2),
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    plt.title("6. Churn Rate by Payment Method")
    plt.xlabel("Churn Rate (%)")
    plt.ylabel("Payment Method")
    plt.xlim(0, 55)
    plt.tight_layout()
    plt.savefig(out_dir / "06_payment_method_churn.png", dpi=150)
    plt.close()


def plot_7_internet_service_churn(df: pd.DataFrame, out_dir: Path) -> None:
    rates = (
        df.groupby("InternetService", observed=False)["Churn_bin"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=rates,
        x="InternetService",
        y="Churn_bin",
        hue="InternetService",
        palette="Oranges_r",
        legend=False,
    )
    for p in ax.patches:
        val = p.get_height()
        ax.annotate(
            f"{val:.2f}%",
            (p.get_x() + p.get_width() / 2, val),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.title("7. Churn Rate by Internet Service")
    plt.xlabel("Internet Service Type")
    plt.ylabel("Churn Rate (%)")
    plt.ylim(0, 50)
    plt.tight_layout()
    plt.savefig(out_dir / "07_internet_service_churn.png", dpi=150)
    plt.close()


def plot_8_customer_retention(df: pd.DataFrame, out_dir: Path) -> None:
    retained = df[df["Churn"] == "No"]
    ct = pd.crosstab(retained["TenureGroup"], retained["Contract"])
    order_cols = [c for c in ["Month-to-month", "One year", "Two year"] if c in ct.columns]
    ct = ct[order_cols]

    plt.figure(figsize=(8, 5))
    ct.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="viridis")
    plt.title("8. Retained Customers by Tenure Group and Contract")
    plt.xlabel("Tenure Group")
    plt.ylabel("Retained Customers (Churn = No)")
    plt.legend(title="Contract")
    plt.tight_layout()
    plt.savefig(out_dir / "08_customer_retention.png", dpi=150)
    plt.close()


def plot_9_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn_bin"]
    corr = df[num_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("9. Correlation Heatmap (Numeric Features & Churn)")
    plt.tight_layout()
    plt.savefig(out_dir / "09_correlation_heatmap.png", dpi=150)
    plt.close()


def plot_10_important_factors(df: pd.DataFrame, out_dir: Path) -> None:
    factor_rows = []
    candidate_cats = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "OnlineSecurity",
        "TechSupport",
        "PaperlessBilling",
    ]
    for col in candidate_cats:
        grp = df.groupby(col, observed=False)["Churn_bin"].mean() * 100
        spread = grp.max() - grp.min()
        top_cat = grp.idxmax()
        factor_rows.append(
            {
                "Factor": f"{col} (max: {top_cat})",
                "ChurnSpread_pp": spread,
            }
        )

    factors_df = pd.DataFrame(factor_rows).sort_values("ChurnSpread_pp", ascending=True)
    plt.figure(figsize=(9, 5))
    bars = plt.barh(factors_df["Factor"], factors_df["ChurnSpread_pp"], color="#3182bd")
    for bar in bars:
        val = bar.get_width()
        plt.annotate(
            f"{val:.1f} pp",
            (val, bar.get_y() + bar.get_height() / 2),
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    plt.title("10. Important Categorical Factors (Churn Rate Spread Across Categories)")
    plt.xlabel("Difference Between Highest and Lowest Category Churn Rate (Percentage Points)")
    plt.ylabel("Feature (Highest-Churn Category)")
    plt.xlim(0, 50)
    plt.tight_layout()
    plt.savefig(out_dir / "10_important_factors.png", dpi=150)
    plt.close()


def print_eda_summary(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("EDA KEY FINDINGS SUMMARY")
    print("=" * 70)
    print(f"Total customers: {len(df):,}")
    print(f"Retained (No): {(df['Churn'] == 'No').sum():,} ({(df['Churn'] == 'No').mean() * 100:.2f}%)")
    print(f"Churned  (Yes): {(df['Churn'] == 'Yes').sum():,} ({(df['Churn'] == 'Yes').mean() * 100:.2f}%)\n")

    print("Churn Rate by Gender (%):")
    print((df.groupby("gender", observed=False)["Churn_bin"].mean() * 100).round(2), "\n")

    print("Churn Rate by Contract (%):")
    print((df.groupby("Contract", observed=False)["Churn_bin"].mean() * 100).round(2), "\n")

    print("Tenure Summary by Churn (months):")
    print(df.groupby("Churn", observed=False)["tenure"].agg(["mean", "median", "min", "max"]).round(2), "\n")

    print("MonthlyCharges Summary by Churn ($):")
    print(df.groupby("Churn", observed=False)["MonthlyCharges"].agg(["mean", "median", "std"]).round(2), "\n")

    print("Churn Rate by PaymentMethod (%):")
    print((df.groupby("PaymentMethod", observed=False)["Churn_bin"].mean() * 100).round(2), "\n")

    print("Churn Rate by InternetService (%):")
    print((df.groupby("InternetService", observed=False)["Churn_bin"].mean() * 100).round(2), "\n")

    num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn_bin"]
    print("Correlation with Churn_bin:")
    print(df[num_cols].corr()["Churn_bin"].round(4))


def main() -> None:
    df = prepare_eda_df()
    out_dir = get_charts_dir()
    print_eda_summary(df)

    plot_1_overall_churn(df, out_dir)
    plot_2_gender_churn(df, out_dir)
    plot_3_contract_churn(df, out_dir)
    plot_4_tenure_churn(df, out_dir)
    plot_5_monthly_charges_churn(df, out_dir)
    plot_6_payment_method_churn(df, out_dir)
    plot_7_internet_service_churn(df, out_dir)
    plot_8_customer_retention(df, out_dir)
    plot_9_correlation_heatmap(df, out_dir)
    plot_10_important_factors(df, out_dir)

    print(f"\nSaved all 10 EDA charts to: {out_dir}")


if __name__ == "__main__":
    main()