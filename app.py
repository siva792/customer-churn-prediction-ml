import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve

from src.preprocessing import (
    clean_data,
    get_dataset_path,
    load_raw_data,
    prepare_features_and_target,
    split_and_preprocess,
)
from src.train_models import train_and_evaluate_models


st.set_page_config(
    page_title="Telco Customer Churn Prediction & Analysis",
    page_icon="📡",
    layout="wide",
)


@st.cache_data
def load_clean_dataset() -> pd.DataFrame:
    """Load and clean the Telco Customer Churn dataset once (21 cleaned columns)."""
    return clean_data(load_raw_data(get_dataset_path()))


@st.cache_resource
def prepare_ml_artifacts():
    """Split data without leakage, fit preprocessor on train set, and train models."""
    df = load_clean_dataset()
    X, y = prepare_features_and_target(df, target_col="Churn", drop_id=True)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_proc,
        X_test_proc,
        preprocessor,
        feature_names,
    ) = split_and_preprocess(X, y, test_size=0.2, random_state=42)

    trained_models, results_df = train_and_evaluate_models(
        X_train_proc, X_test_proc, y_train, y_test, random_state=42
    )
    return {
        "X": X,
        "X_test": X_test,
        "y_test": y_test,
        "X_test_proc": X_test_proc,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "models": trained_models,
        "results_df": results_df,
    }


def generate_retention_recommendations(customer_data: dict, churn_prob: float) -> list[str]:
    """Generate practical retention actions based on customer attributes and risk level."""
    recs = []
    if customer_data["Contract"] == "Month-to-month":
        recs.append(
            "Offer a discounted 1-year or 2-year contract upgrade with waived setup/upgrade fees."
        )
    if customer_data["tenure"] <= 12:
        recs.append(
            "Provide early-tenure onboarding support, proactive service checks, and first-year loyalty perks."
        )
    if customer_data["PaymentMethod"] == "Electronic check":
        recs.append(
            "Encourage switching from Electronic check to automatic bank transfer or credit card billing."
        )
    if customer_data["InternetService"] == "Fiber optic" and customer_data["OnlineSecurity"] == "No":
        recs.append(
            "Offer a complimentary or bundled OnlineSecurity trial for this Fiber optic customer."
        )
    if customer_data["InternetService"] == "Fiber optic" and customer_data["TechSupport"] == "No":
        recs.append(
            "Provide priority TechSupport access to resolve service issues quickly."
        )
    if customer_data["MonthlyCharges"] >= 75:
        recs.append(
            "Review monthly plan add-ons to ensure perceived value matches the higher monthly bill."
        )

    if not recs:
        if churn_prob >= 0.5:
            recs.append("Reach out with a personalized check-in and tailored loyalty offer.")
        else:
            recs.append("Customer profile currently shows low churn risk; maintain standard engagement.")
    return recs


def render_home_page(df: pd.DataFrame) -> None:
    st.title("📡 Customer Churn Prediction & Analysis System")
    st.markdown(
        "Interactive analytics and machine learning dashboard built on the **Telco Customer Churn** dataset "
        "(`WA_Fn-UseC_-Telco-Customer-Churn.csv`)."
    )

    total_customers = len(df)
    churned_customers = int((df["Churn"] == "Yes").sum())
    retained_customers = int((df["Churn"] == "No").sum())
    churn_pct = (churned_customers / total_customers) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned_customers:,}")
    col3.metric("Retained Customers", f"{retained_customers:,}")
    col4.metric("Churn Percentage", f"{churn_pct:.2f}%")

    st.divider()
    st.subheader("Dataset Preview & Quick Summary")
    st.dataframe(df.head(15), use_container_width=True)


def render_eda_page(df: pd.DataFrame) -> None:
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Filter customers interactively to explore churn patterns across contracts, billing, and tenure.")

    eda_df = df.copy()
    eda_df["Churn_bin"] = eda_df["Churn"].map({"No": 0, "Yes": 1})

    col_f1, col_f2, col_f3 = st.columns(3)
    contract_opts = ["All"] + sorted(eda_df["Contract"].unique().tolist())
    internet_opts = ["All"] + sorted(eda_df["InternetService"].unique().tolist())
    payment_opts = ["All"] + sorted(eda_df["PaymentMethod"].unique().tolist())

    sel_contract = col_f1.selectbox("Contract Type", contract_opts)
    sel_internet = col_f2.selectbox("Internet Service", internet_opts)
    sel_payment = col_f3.selectbox("Payment Method", payment_opts)

    filtered = eda_df.copy()
    if sel_contract != "All":
        filtered = filtered[filtered["Contract"] == sel_contract]
    if sel_internet != "All":
        filtered = filtered[filtered["InternetService"] == sel_internet]
    if sel_payment != "All":
        filtered = filtered[filtered["PaymentMethod"] == sel_payment]

    st.caption(f"Showing **{len(filtered):,}** customers matching current filters.")

    if filtered.empty:
        st.warning("No customers match the selected filter combination.")
        return

    row1_left, row1_right = st.columns(2)

    with row1_left:
        st.subheader("Contract-Wise Churn Rate (%)")
        c_rates = (filtered.groupby("Contract")["Churn_bin"].mean() * 100).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=c_rates.index, y=c_rates.values, hue=c_rates.index, palette="Set2", legend=False, ax=ax)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xlabel("Contract")
        ax.set_ylim(0, max(60, float(c_rates.max()) * 1.2))
        for i, v in enumerate(c_rates.values):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with row1_right:
        st.subheader("Payment Method Churn Rate (%)")
        p_rates = (filtered.groupby("PaymentMethod")["Churn_bin"].mean() * 100).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=p_rates.values, y=p_rates.index, hue=p_rates.index, palette="Reds", legend=False, ax=ax)
        ax.set_xlabel("Churn Rate (%)")
        ax.set_ylabel("Payment Method")
        ax.set_xlim(0, max(60, float(p_rates.max()) * 1.2))
        for i, v in enumerate(p_rates.values):
            ax.text(v + 1, i, f"{v:.1f}%", va="center", fontweight="bold", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    row2_left, row2_right = st.columns(2)

    with row2_left:
        st.subheader("Tenure Distribution by Churn")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=filtered,
            x="Churn",
            y="tenure",
            hue="Churn",
            order=["No", "Yes"],
            hue_order=["No", "Yes"],
            palette=["#2b8cbe", "#e34a33"],
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("Churn")
        ax.set_ylabel("Tenure (Months)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with row2_right:
        st.subheader("Customer Behavior: Monthly Charges vs. Tenure")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=filtered.sample(min(len(filtered), 1500), random_state=42),
            x="tenure",
            y="MonthlyCharges",
            hue="Churn",
            palette={"No": "#2b8cbe", "Yes": "#e34a33"},
            alpha=0.6,
            ax=ax,
        )
        ax.set_xlabel("Tenure (Months)")
        ax.set_ylabel("Monthly Charges ($)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_model_performance_page(artifacts: dict) -> None:
    st.title("🤖 Model Performance & Evaluation")
    results_df = artifacts["results_df"]
    trained_models = artifacts["models"]
    X_test_proc = artifacts["X_test_proc"]
    y_test = artifacts["y_test"]

    st.subheader("Model Comparison Table (Test Set)")
    st.dataframe(
        results_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]],
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "**Why not rely on Accuracy alone?** Because ~73.5% of customers do not churn, a trivial model "
        "predicting 'No' for everyone would score ~73.5% accuracy while missing all churners. "
        "F1-Score, Recall, Precision, and ROC-AUC give a clearer picture of churn detection."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Confusion Matrix")
        selected_model_name = st.selectbox("Select Model for Confusion Matrix", list(trained_models.keys()))
        model = trained_models[selected_model_name]
        y_pred = model.predict(X_test_proc)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No (0)", "Yes (1)"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{selected_model_name} Confusion Matrix")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.subheader("ROC Curves Comparison")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for name, clf in trained_models.items():
            y_prob = clf.predict_proba(X_test_proc)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = results_df.loc[results_df["Model"] == name, "ROC-AUC"].values[0]
            ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_score:.4f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("ROC Curves")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_prediction_page(artifacts: dict) -> None:
    st.title("🔮 Customer Churn Prediction")
    st.markdown("Enter customer details below and click **Predict Churn** to estimate churn risk.")

    trained_models = artifacts["models"]
    preprocessor = artifacts["preprocessor"]

    selected_model_name = st.selectbox(
        "Choose Prediction Model",
        list(trained_models.keys()),
        index=0,
    )

    with st.form("churn_prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_label = st.selectbox("Senior Citizen", ["No (0)", "Yes (1)"])
            SeniorCitizen = 1 if senior_label.startswith("Yes") else 0
            Partner = st.selectbox("Partner", ["No", "Yes"])
            Dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
            PhoneService = st.selectbox("Phone Service", ["Yes", "No"])

        with c2:
            MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            InternetService = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

        with c3:
            StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            PaymentMethod = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            MonthlyCharges = st.number_input(
                "Monthly Charges ($)", min_value=15.0, max_value=130.0, value=75.0, step=0.5
            )
            default_total = round(float(tenure) * float(MonthlyCharges), 2)
            TotalCharges = st.number_input(
                "Total Charges ($)", min_value=0.0, max_value=10000.0, value=default_total, step=5.0
            )

        submitted = st.form_submit_button("Predict Churn", type="primary")

    if submitted:
        input_record = {
            "gender": gender,
            "SeniorCitizen": int(SeniorCitizen),
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": int(tenure),
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": float(MonthlyCharges),
            "TotalCharges": float(TotalCharges),
        }

        input_df = pd.DataFrame([input_record])
        X_proc = preprocessor.transform(input_df)
        model = trained_models[selected_model_name]

        pred_class = int(model.predict(X_proc)[0])
        churn_prob = float(model.predict_proba(X_proc)[0, 1])

        st.divider()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if pred_class == 1:
                st.error("⚠️ Prediction: **Customer Likely to Churn (Yes)**")
            else:
                st.success("✅ Prediction: **Customer Likely to Stay (No)**")
        with res_col2:
            st.metric("Estimated Churn Probability", f"{churn_prob * 100:.2f}%")

        st.subheader("Recommended Retention Actions")
        for idx, rec in enumerate(generate_retention_recommendations(input_record, churn_prob), start=1):
            st.write(f"{idx}. {rec}")


def main() -> None:
    df = load_clean_dataset()
    artifacts = prepare_ml_artifacts()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Home Dashboard",
            "Exploratory Data Analysis",
            "Model Performance",
            "Customer Churn Prediction",
        ],
    )

    if page == "Home Dashboard":
        render_home_page(df)
    elif page == "Exploratory Data Analysis":
        render_eda_page(df)
    elif page == "Model Performance":
        render_model_performance_page(artifacts)
    elif page == "Customer Churn Prediction":
        render_prediction_page(artifacts)


if __name__ == "__main__":
    main()