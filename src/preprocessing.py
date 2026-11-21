from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_dataset_path() -> Path:
    """Resolve the path to WA_Fn-UseC_-Telco-Customer-Churn.csv inside data/."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """Load the raw telecom churn dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw dataset:
    - Strip whitespace from string columns
    - Convert TotalCharges to numeric and handle blank entries
    - Remove exact duplicate rows
    - Drop customerID for modeling preparation
    """
    cleaned = df.copy()

    # Strip extra spaces in text/object/string columns
    text_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    # Convert TotalCharges to numeric ('coerce' turns blank strings into NaN)
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    # For customers with tenure == 0, TotalCharges is naturally 0.0 when blank
    zero_tenure_blank = cleaned["TotalCharges"].isna() & (cleaned["tenure"] == 0)
    cleaned.loc[zero_tenure_blank, "TotalCharges"] = 0.0

    # Remove exact duplicate records if any exist
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned


def prepare_features_and_target(
    df: pd.DataFrame, target_col: str = "Churn", drop_id: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X (without customerID) and binary target y (Yes=1, No=0)."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    y_mapped = df[target_col].map({"No": 0, "Yes": 1})
    if y_mapped.isna().any():
        invalid_vals = df.loc[y_mapped.isna(), target_col].unique().tolist()
        raise ValueError(f"Unexpected values in '{target_col}': {invalid_vals}")

    cols_to_drop = [target_col]
    if drop_id and "customerID" in df.columns:
        cols_to_drop.append("customerID")

    X = df.drop(columns=cols_to_drop).copy()
    y = y_mapped.astype(int)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a leak-free ColumnTransformer that preprocesses numerical and
    categorical columns separately.
    """
    # Identify numerical vs categorical features from X
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    # Numeric pipeline: impute missing values (if any) using median, then scale
    # Note: scaling helps distance/linear models like Logistic Regression;
    # tree models can either use scaled or unscaled features cleanly.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: impute missing categories if any, then one-hot encode
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def split_and_preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Perform a stratified train-test split first to prevent data leakage,
    then fit the preprocessor strictly on X_train.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(X_train)

    # Fit ONLY on training data; transform both train and test sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_processed,
        X_test_processed,
        preprocessor,
        feature_names,
    )


def run_preprocessing_report() -> None:
    """Run data quality checks, cleaning, and leak-free train/test preparation."""
    csv_path = get_dataset_path()
    raw_df = load_raw_data(csv_path)

    print("=" * 70)
    print("1. MISSING VALUES & DUPLICATE CHECK (RAW DATA)")
    print("=" * 70)
    print("Explicit NaN counts per column:")
    print(raw_df.isna().sum()[raw_df.isna().sum() >= 0])
    print(f"\nExact duplicate rows: {raw_df.duplicated().sum()}")
    print(f"Duplicate customerIDs: {raw_df['customerID'].duplicated().sum()}")

    blank_mask = raw_df["TotalCharges"].astype(str).str.strip() == ""
    print(f"\nBlank/whitespace 'TotalCharges' rows found: {blank_mask.sum()}")
    if blank_mask.sum() > 0:
        print(
            raw_df.loc[
                blank_mask, ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
            ]
        )
    print()

    cleaned_df = clean_data(raw_df)
    print("=" * 70)
    print("2. AFTER CLEANING & TotalCharges CONVERSION")
    print("=" * 70)
    print(f"TotalCharges dtype after conversion: {cleaned_df['TotalCharges'].dtype}")
    print(f"Remaining NaNs in TotalCharges: {cleaned_df['TotalCharges'].isna().sum()}")
    print(f"Cleaned DataFrame shape: {cleaned_df.shape}\n")

    X, y = prepare_features_and_target(cleaned_df, target_col="Churn", drop_id=True)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_processed,
        X_test_processed,
        _,
        feature_names,
    ) = split_and_preprocess(X, y, test_size=0.2, random_state=42)

    print("=" * 70)
    print("3. LEAK-FREE SPLIT & ENCODED FEATURE MATRIX")
    print("=" * 70)
    print(f"Raw feature columns (excluding customerID & Churn): {X.shape[1]}")
    print(f"Training set rows: {X_train.shape[0]} | Testing set rows: {X_test.shape[0]}")
    print(
        f"Churn rate ('1') -> Train: {y_train.mean() * 100:.2f}% | "
        f"Test: {y_test.mean() * 100:.2f}%"
    )
    print(f"Processed feature matrix shape (X_train_processed): {X_train_processed.shape}")
    print(f"Processed feature matrix shape (X_test_processed):  {X_test_processed.shape}")
    print(f"Total encoded features generated: {len(feature_names)}")


if __name__ == "__main__":
    run_preprocessing_report()