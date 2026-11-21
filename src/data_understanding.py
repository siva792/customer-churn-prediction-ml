from pathlib import Path
import pandas as pd


def get_dataset_path() -> Path:
    """Resolve the path to WA_Fn-UseC_-Telco-Customer-Churn.csv inside data/."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the telecom churn CSV dataset into a Pandas DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {csv_path}\n"
            "Please place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in the 'data/' folder."
        )
    return pd.read_csv(csv_path)


def explain_columns() -> dict[str, str]:
    """Return a clear description of all 21 columns in the dataset."""
    return {
        "customerID": "Unique identifier assigned to each telecom customer.",
        "gender": "Customer's gender (Male or Female).",
        "SeniorCitizen": "Whether the customer is a senior citizen (1 = Yes, 0 = No).",
        "Partner": "Whether the customer has a spouse or partner (Yes or No).",
        "Dependents": "Whether the customer has dependents such as children or parents (Yes or No).",
        "tenure": "Number of months the customer has stayed with the company.",
        "PhoneService": "Whether the customer has phone service (Yes or No).",
        "MultipleLines": "Whether the customer has multiple phone lines (Yes, No, or No phone service).",
        "InternetService": "Customer's internet service provider type (DSL, Fiber optic, or No).",
        "OnlineSecurity": "Whether the customer has online security add-on (Yes, No, or No internet service).",
        "OnlineBackup": "Whether the customer has online backup add-on (Yes, No, or No internet service).",
        "DeviceProtection": "Whether the customer has device protection plan (Yes, No, or No internet service).",
        "TechSupport": "Whether the customer has technical support plan (Yes, No, or No internet service).",
        "StreamingTV": "Whether the customer has streaming TV service (Yes, No, or No internet service).",
        "StreamingMovies": "Whether the customer has streaming movies service (Yes, No, or No internet service).",
        "Contract": "Contract term type (Month-to-month, One year, or Two year).",
        "PaperlessBilling": "Whether the customer uses paperless billing (Yes or No).",
        "PaymentMethod": "Customer's billing payment method (Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)).",
        "MonthlyCharges": "The amount charged to the customer monthly.",
        "TotalCharges": "The total amount charged to the customer over their tenure (stored as text initially due to blank entries).",
        "Churn": "Target variable indicating whether the customer left within the last month (Yes or No).",
    }


def inspect_dataset(df: pd.DataFrame) -> None:
    """Display dataset structure, sample rows, column explanations, and feature types."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("=" * 70)
    print("1. DATASET DIMENSIONS")
    print("=" * 70)
    print(f"Number of Rows (Customers): {df.shape[0]}")
    print(f"Number of Columns (Attributes): {df.shape[1]}\n")

    print("=" * 70)
    print("2. FIRST 5 RECORDS (df.head())")
    print("=" * 70)
    print(df.head(5))
    print()

    print("=" * 70)
    print("3. LAST 5 RECORDS (df.tail())")
    print("=" * 70)
    print(df.tail(5))
    print()

    print("=" * 70)
    print("4. COLUMN DATA TYPES & EXPLANATIONS")
    print("=" * 70)
    descriptions = explain_columns()
    for idx, col in enumerate(df.columns, start=1):
        dtype = df[col].dtype
        desc = descriptions.get(col, "No description available.")
        print(f"{idx:2d}. {col} ({dtype}): {desc}")
    print()

    # Identify feature groups
    target_col = "Churn"
    id_col = "customerID"

    # Numeric columns currently detected by Pandas
    raw_numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # Note: 'tenure' and 'MonthlyCharges' are continuous/count numeric features;
    # 'SeniorCitizen' is a binary 0/1 indicator; 'TotalCharges' should become numeric in Module 2.
    continuous_or_count_cols = ["tenure", "MonthlyCharges"]
    binary_flag_cols = ["SeniorCitizen"]

    categorical_cols = [
        col for col in df.columns if col not in raw_numeric_cols and col != target_col
    ]

    print("=" * 70)
    print("5. FEATURE IDENTIFICATION")
    print("=" * 70)
    print(f"Identifier Column (exclude from ML features): {id_col}")
    print(f"Pandas Numeric Columns (int64/float64): {raw_numeric_cols}")
    print(f"  - Core Numeric Features (tenure, MonthlyCharges): {continuous_or_count_cols}")
    print(f"  - Binary Numeric Flag (0/1): {binary_flag_cols}")
    print(f"  - Note: 'TotalCharges' contains numeric billing totals but is currently object dtype.")
    print(f"\nCategorical / Text Columns ({len(categorical_cols)}): {categorical_cols}\n")

    print("=" * 70)
    print("6. TARGET VARIABLE")
    print("=" * 70)
    print(f"Target Variable: '{target_col}'")
    print("Value Counts:")
    print(df[target_col].value_counts())
    print("\nPercentage Distribution:")
    print((df[target_col].value_counts(normalize=True) * 100).round(2))


def main() -> None:
    csv_path = get_dataset_path()
    df = load_data(csv_path)
    inspect_dataset(df)


if __name__ == "__main__":
    main()