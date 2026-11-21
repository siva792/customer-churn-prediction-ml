# Customer Churn Prediction and Analysis System Using Machine Learning

## 1. Project Abstract
Customer churn—when subscribers cancel or discontinue service—directly reduces recurring revenue and increases acquisition pressure in the telecommunications industry. This project builds an end-to-end **Customer Churn Prediction and Analysis System Using Machine Learning** in Python using the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset (7,043 customer records and 21 attributes). The system covers data understanding, leak-free preprocessing, 10 exploratory data analysis (EDA) visualizations, training and evaluating Logistic Regression, Decision Tree, and Random Forest classifiers, feature importance analysis, Joblib model persistence, and a four-page interactive Streamlit web application.

## 2. Problem Statement
Telecom operators manage large subscriber bases with varying contract durations, internet and add-on services, billing preferences, and monthly charges. Identifying customers who are likely to churn before they leave is difficult when relying on manual reporting or overall averages. Furthermore, because roughly three-quarters of customers in the dataset are retained (`73.46%` No vs. `26.54%` Yes), evaluating models using accuracy alone can mask weak detection of at-risk customers.

## 3. Objectives
* Load, inspect, and understand the structure, data types, and target distribution of `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
* Clean data quality issues—including converting `TotalCharges` to numeric and handling blank entries for zero-tenure customers—while preventing data leakage via stratified splitting and Scikit-learn pipelines.
* Perform exploratory data analysis across demographics, contracts, tenure, billing, internet services, retention, and feature correlations.
* Train and compare Logistic Regression, Decision Tree Classifier, and Random Forest Classifier models using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
* Extract key churn drivers and translate them into actionable customer retention recommendations.
* Persist trained models and preprocessing artifacts with Joblib and serve analytics and real-time predictions through a Streamlit dashboard.

## 4. Existing System
Traditional churn tracking often relies on static spreadsheets, periodic SQL/BI summaries, or reactive post-cancellation exit counts. These approaches typically show historical churn after revenue is already lost, lack individualized churn risk scoring at the customer level, and provide limited interactive exploration for retention teams.

## 5. Proposed System
The proposed system provides a modular, reproducible Python pipeline and interactive Streamlit web dashboard that:
* Standardizes data cleaning and leak-free feature preprocessing (`src/preprocessing.py`).
* Generates visual insights and churn driver rankings (`src/eda.py`, `src/evaluate_models.py`, `src/feature_importance.py`).
* Saves and loads fitted preprocessors and trained models via Joblib (`src/save_and_predict.py`).
* Offers an interactive web interface (`app.py`) with a **Home Dashboard**, **Exploratory Data Analysis**, **Model Performance**, and **Customer Churn Prediction** with tailored retention recommendations.

## 6. Technologies Used
* **Programming Language:** Python
* **Data & Numerical Libraries:** Pandas, NumPy
* **Visualization Libraries:** Matplotlib, Seaborn
* **Machine Learning & Persistence:** Scikit-learn, Joblib
* **Web Framework:** Streamlit
* **Development Tools:** VS Code, Jupyter Notebook

## 7. System Architecture
```text
WA_Fn-UseC_-Telco-Customer-Churn.csv (data/)
        │
        ▼
Data Understanding & Cleaning (src/data_understanding.py, src/preprocessing.py)
        │
        ├──▶ Exploratory Data Analysis Charts (src/eda.py -> reports/eda_charts/)
        │
        ▼
Stratified Train/Test Split & ColumnTransformer Pipeline
        │
        ├──▶ Model Training & Evaluation (src/train_models.py, src/evaluate_models.py)
        ├──▶ Feature Importance & Business Insights (src/feature_importance.py)
        └──▶ Model & Preprocessor Serialization (src/save_and_predict.py -> models/*.joblib)
                    │
                    ▼
        Streamlit Web Application (app.py)
        [Home | EDA | Model Performance | Churn Prediction]