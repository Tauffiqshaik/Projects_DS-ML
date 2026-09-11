# Projects_DS-ML

# Machine Learning Project Portfolio

A collection of end-to-end machine learning projects covering regression, classification, and business analytics use cases — from data cleaning and EDA through model tuning, evaluation, and business recommendations.

| # | Project | Type | Best Model | Key Metric |
|---|---|---|---|---|
| 1 | [House Price Prediction](#1-house-price-prediction-advanced-regression) | Regression | XGBoost (tuned) | Test R² = 0.908 |
| 2 | [Telecom Customer Churn](#2-telecom-customer-churn-analysis) | Classification | CatBoost (Optuna-tuned) | Test ROC-AUC = 0.4954* |
| 3 | [Home Loan Default Risk](#3-home-loan-default-risk-management) | Classification | Random Forest | Best among LR / DT / RF |
| 4 | [Portuguese Bank Marketing](#4-portuguese-bank-term-deposit-marketing) | Classification | Random Forest | Best among 5 models |

\* See project notes — this dataset was found to contain no genuine predictive signal.

---

## 1. House Price Prediction — Advanced Regression

**Notebook:** `PRCP-1020-House_Price_Prediction_Advanced-Regression.ipynb`

Predicts residential house sale prices in Ames, Iowa using the classic 80-feature Advanced Regression housing dataset.

**Highlights**
- Full EDA on numerical and categorical features; identified `OverallQual`, `GrLivArea`, `TotalBsmtSF`, `GarageCars`, `YearBuilt`, and `Neighborhood` as the strongest price drivers.
- Applied `log1p` transformation to the skewed target (`SalePrice`) to stabilize variance.
- Missing value imputation, outlier treatment, ordinal/binary/one-hot encoding, and feature selection.
- Models compared: Linear Regression, Ridge, Lasso, Random Forest, XGBoost.
- Hyperparameter tuning via `RandomizedSearchCV` (5-fold CV) on Random Forest and XGBoost.

**Results**

| Model | Train R² | Test R² | RMSE |
|---|---|---|---|
| Random Forest (tuned) | 0.967 | 0.855 | 26,298 |
| **XGBoost (tuned)** | **0.993** | **0.908** | **19,699** |

**Business takeaways:** overall build quality and usable living space matter more than raw bedroom count; neighborhood and construction year materially affect resale value.

---

## 2. Telecom Customer Churn Analysis

**Notebook:** `No_Churn_telecom_17_3865.ipynb`

Builds a churn-prediction pipeline for a telecom provider to support marketing retention offers and prioritized customer support outreach.

**Highlights**
- Data pulled from a MySQL source; cleaned negative usage values and dropped high-cardinality/leaky identifier columns (`customer_id`, `city`, `state`, `pincode`).
- Engineered `tenure_months` from registration date using a fixed reference date to avoid leakage.
- Leakage-free `sklearn` `Pipeline` + `ColumnTransformer` (median/mode imputation, scaling, one-hot encoding).
- Compared Logistic Regression, LightGBM, Random Forest, XGBoost, and CatBoost via Stratified K-Fold CV.
- Bayesian hyperparameter search with **Optuna** (20 trials) on CatBoost.
- Explainability via **LIME** (local, per-customer) and **Permutation Importance** (global).
- Defined a `CHURN_FLAG` (YES/NO) output and a 4-tier marketing risk framework (Critical / High / Medium / Low) with recommended actions and support SLAs.

**Critical finding:** statistical testing (Mann-Whitney U on all 7 numerical features, p > 0.05 for every one) plus near-identical churn rates across all categorical segments (~20% everywhere) showed the dataset's churn labels carry **no genuine predictive signal** — consistent with synthetic, randomly-labeled data. The final CatBoost model scored Test ROC-AUC ≈ 0.495 (random), which is the mathematically correct outcome given the data, not a modeling failure.

**Value delivered:** a fully production-ready, leakage-free ML framework (preprocessing → tuning → explainability → deployment) that is ready to plug into real customer data containing genuine churn drivers (complaints, billing disputes, dropped calls, plan changes).

---

## 3. Home Loan Default Risk Management

**Notebook:** `PRCP-1006_Home_Loan_Default-Risk_Management__3_.ipynb`

Predicts whether a home loan applicant is likely to default, to support credit risk decisions.

**Highlights**
- Handled missing values (mean/median for numerical, mode for categorical) and encoded categorical features.
- EDA revealed lower income, higher loan amounts, and weak credit history as strong default indicators.
- Feature scaling and train/test split for unbiased evaluation.
- Models compared: Logistic Regression, Decision Tree, Random Forest — evaluated with Accuracy, Confusion Matrix, ROC-AUC, Precision/Recall/F1.

**Result:** Random Forest gave the best generalization and was selected as the final model, with credit history, income stability, and loan-to-income ratio identified as the top risk drivers.

**Business takeaways:** prioritize applicants with strong credit history, apply risk-based interest rates, and monitor high-risk accounts proactively.

---

## 4. Portuguese Bank — Term Deposit Marketing

**Notebook:** `PRCP-1000_Portuguese_Bank_Marketing__3_.ipynb`

Predicts whether a bank customer will subscribe to a term deposit, using the UCI Bank Marketing dataset, to improve telemarketing efficiency.

**Highlights**
- EDA on an imbalanced target (few subscribers vs. many non-subscribers); correlation heatmap and boxplot-based outlier checks.
- Feature encoding and scaling, followed by a train/test split.
- Models compared: Logistic Regression, Decision Tree, Random Forest, KNN, SVM — with ROC curves and feature importance for the best model.

**Result:** Random Forest was the best-performing model. Call duration, previous campaign outcome, account balance, number of contacts, and housing loan status were the strongest predictors of subscription.

**Business takeaways:** prioritize customers with positive prior campaign outcomes, cap the number of repeat contacts, and focus on call quality over call volume.

---

## Repository Structure

```
.
├── PRCP-1020-House_Price_Prediction_Advanced-Regression.ipynb
├── No_Churn_telecom_17_3865.ipynb
├── PRCP-1006_Home_Loan_Default-Risk_Management__3_.ipynb
├── PRCP-1000_Portuguese_Bank_Marketing__3_.ipynb
└── README.md
```

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost` · `Optuna` · `LIME` · `Matplotlib` / `Seaborn`

## How to Run

1. Clone the repository and install dependencies (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `optuna`, `lime`, `matplotlib`, `seaborn`, `pymysql` if using the DB source).
2. Open the desired notebook in Jupyter / VS Code / Google Colab.
3. Run cells top-to-bottom — each notebook is self-contained, ending with a full written project report (EDA insights, model comparison, business recommendations).

## Author

Add your name, links (LinkedIn/portfolio), and contact details here.
