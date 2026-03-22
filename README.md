# 🛡️ Insurance Fraud Detection — Machine Learning Project

> Academic Project | Université Abdelmalek Essaâdi, FST Tanger
> **Author:** Adam Gouchi

---

## 📋 Overview

A machine learning pipeline to detect **fraudulent insurance claims** from a highly imbalanced dataset (~95% legitimate, ~5% fraudulent). The project covers the full ML workflow: exploratory data analysis, data cleaning, feature engineering, class balancing with SMOTE, and comparison of multiple classifiers.

The target variable is `CLAIM_STATUS` — `A` (Approved/Legitimate) vs `D` (Denied/Fraudulent).

---

## 🎯 Problem Statement

Insurance fraud is a rare but costly event. With only **5% of claims being fraudulent**, standard classifiers tend to ignore the minority class entirely. This project addresses that challenge through:

- Strategic feature engineering and imputation
- SMOTE oversampling (applied **only on training data** to avoid data leakage)
- Threshold tuning to maximize fraud recall
- Evaluation using Precision, Recall, F1-Score, and ROC-AUC

---

## 📊 Dataset

| Feature | Description |
|---|---|
| `CLAIM_STATUS` | Target — `A` (legitimate) / `D` (fraud) |
| `INSURANCE_TYPE` | Health, Motor, Property, Life, Travel |
| `PREMIUM_AMOUNT` | Monthly premium paid |
| `CLAIM_AMOUNT` | Amount claimed |
| `INCIDENT_SEVERITY` | Minor Loss / Major Loss / Total Loss |
| `AUTHORITY_CONTACTED` | Police / Ambulance / None |
| `POLICE_REPORT_AVAILABLE` | Binary flag |
| `ANY_INJURY` | Binary flag |
| `INCIDENT_HOUR_OF_THE_DAY` | Hour of incident |
| `CUSTOMER_EDUCATION_LEVEL` | Education of policyholder |
| `AGE`, `TENURE`, `NO_OF_FAMILY_MEMBERS` | Customer demographics |

**Class imbalance:**
```
A (Legitimate) → 94.97%
D (Fraudulent) → 5.03%
```

**Missing values handled:**
- `AUTHORITY_CONTACTED` — conditional mode imputation grouped by `INCIDENT_SEVERITY`
- `CUSTOMER_EDUCATION_LEVEL` — global mode imputation
- High-cardinality/ID columns dropped (SSN, AGENT_ID, VENDOR_ID, CITY, etc.)

---

## 🔧 ML Pipeline
```
Raw Data
   │
   ├── 1. EDA — class distribution, missing values, df.head()
   ├── 2. Cleaning — drop irrelevant columns (IDs, addresses, dates)
   ├── 3. Imputation — conditional + mode-based
   ├── 4. Encoding — pd.get_dummies() (one-hot encoding)
   ├── 5. Train/Test Split — 80/20, stratified
   ├── 6. Scaling — StandardScaler (fit on train only)
   ├── 7. SMOTE — sampling_strategy=0.3, k_neighbors=3 (train only)
   └── 8. Model Training + Evaluation
```

> ⚠️ **Data leakage prevention:** Scaling and SMOTE are applied **after** the train/test split and **only on training data**.

---

## 🤖 Models Compared

| Model | Accuracy | Fraud Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 76% | 19% | 0.500 |
| Decision Tree (max_depth=6) | 94% | 0% | 0.484 |
| Random Forest (50 trees) | 87% | 10% | 0.513 |

> **Key insight:** High accuracy is misleading with imbalanced data. A model that predicts everything as "legitimate" gets 95% accuracy but catches zero fraud. Recall on the fraud class is the metric that matters.

---

## 🛠️ Technologies

| Tool | Role |
|---|---|
| Python 3.13 | Core language |
| pandas | Data manipulation |
| scikit-learn | ML models, preprocessing, metrics |
| imbalanced-learn | SMOTE oversampling |
| matplotlib / seaborn | Visualization |
| Jupyter Notebook | Development environment |

## 📁 Project Structure
```
insurance-fraud-detection/
├── fraud_detection.ipynb    # Main notebook — full pipeline
├── insurance_data.csv       # Dataset
├── requirements.txt
└── README.md
```

---

## 🔍 Key Decisions & Lessons Learned

- **Why SMOTE on train only?** Applying SMOTE before splitting would expose the test set to synthetic data, causing data leakage and inflated metrics.
- **Why lower the threshold to 0.3?** The default 0.5 threshold misses almost all fraud. Lowering it increases recall at the cost of more false positives — an acceptable trade-off in fraud detection.
- **Why Logistic Regression underperforms?** Structural underfitting — the decision boundary between legitimate and fraudulent claims is non-linear, which LR cannot capture even with normalization and SMOTE.
- **Overfitting prevention:** `max_depth` in Decision Tree and Random Forest limits how deep the tree grows, preventing the model from memorizing training data.

---

## 📚 Academic Context

- **Module:** Data Mining / Machine Learning
- **Concepts:** Imbalanced classification, SMOTE, threshold tuning, model comparison
- **Level:** L3 — Data Analytics
- **Institution:** FST Tanger, Université Abdelmalek Essaâdi
- **Year:** 2025–2026
