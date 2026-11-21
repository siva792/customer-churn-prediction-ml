from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import (
    clean_data,
    get_dataset_path,
    load_raw_data,
    prepare_features_and_target,
    split_and_preprocess,
)


def build_models(random_state: int = 42) -> dict:
    """Create the three required classification models."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def train_and_evaluate_models(
    X_train_proc,
    X_test_proc,
    y_train,
    y_test,
    random_state: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """Train all three models on preprocessed features and compute test metrics."""
    models = build_models(random_state=random_state)
    trained_models = {}
    rows = []

    for name, model in models.items():
        model.fit(X_train_proc, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        rows.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )

    results_df = pd.DataFrame(rows)
    return trained_models, results_df


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

    _, results_df = train_and_evaluate_models(
        X_train_proc, X_test_proc, y_train, y_test, random_state=42
    )

    print("=" * 75)
    print("MODULE 4: MODEL TRAINING & INITIAL COMPARISON (TEST SET)")
    print("=" * 75)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()