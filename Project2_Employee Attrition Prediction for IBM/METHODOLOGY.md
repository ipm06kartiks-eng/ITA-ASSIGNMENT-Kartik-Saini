# Employee Attrition Prediction — Methodology

---

## Overview

This project implements a **hybrid machine learning architecture** combining unsupervised and supervised learning to predict and prevent employee attrition at IBM. Generic models fail because they treat all employees identically. Our hybrid approach first discovers employee types through clustering, then builds a specialized predictive model for each group. This document details the full technical methodology.

---

## Problem Formulation

**Input:** Employee HR data (demographics, job role, compensation, satisfaction scores, overtime, tenure)

**Output:** Binary attrition prediction — `Yes` (will leave) or `No` (will stay) + identification of high-risk employee segments

**Key Innovation:** Employees leave for different reasons. A junior employee leaving due to low pay requires a different intervention than a senior employee leaving due to work-life balance issues. By clustering employees first and then predicting attrition within context, the model captures these nuanced differences.

---

## Step 1 — Data Collection

The dataset used is the **IBM HR Analytics Employee Attrition & Performance** dataset, sourced from Kaggle.

| Attribute | Details |
|---|---|
| Total Records | 1,470 employees |
| Number of Features | 35 attributes |
| Target Variable | `Attrition` — Yes / No |
| Class Distribution | ~84% Stayed, ~16% Left (imbalanced) |
| Key Features | Age, OverTime, MonthlyIncome, JobSatisfaction, YearsAtCompany, Department, JobRole |

---

## Step 2 — Data Preprocessing

Raw HR data contains categorical variables and inconsistencies that must be resolved before applying any ML algorithm.

**Actions taken:**

- **Dropped irrelevant columns:** `EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber` — these are constant or non-informative.
- **Target encoding:** `Attrition` mapped to binary → `Yes = 1`, `No = 0`
- **Label encoding:** All categorical features (e.g., `Gender`, `Department`, `JobRole`, `MaritalStatus`) converted to integers using `LabelEncoder`
- **Feature scaling:** All numerical features standardized using `StandardScaler` (mean = 0, std = 1) so that no single feature dominates due to scale

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## Step 3 — Unsupervised Learning: K-Means Clustering

K-Means clustering is applied on the scaled feature set to discover **natural employee groupings** without using the attrition label.

### Choosing the Optimal K — Elbow Method

Inertia (Within-Cluster Sum of Squares) is plotted against increasing values of K. The point where inertia stops decreasing sharply — the "elbow" — indicates the optimal number of clusters. For this dataset, **K = 3** is selected.

```python
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
```

### What Each Cluster Represents

| Cluster | Likely Profile | Attrition Risk |
|---|---|---|
| 0 | Senior employees, high income, low overtime | Low |
| 1 | Junior/mid-level, high overtime, low satisfaction | High |
| 2 | Mid-level, moderate satisfaction, moderate tenure | Medium |

> *Exact cluster profiles will vary based on your run — inspect `cluster_distribution.png` for your results.*

### Cluster Visualization

- **`elbow_curve.png`** — Inertia vs K to confirm optimal cluster count
- **`cluster_distribution.png`** — Attrition counts per cluster
- **`cluster_pca.png`** — 2D PCA projection of all three clusters

---

## Step 4 — Building the Hybrid Feature Set

This is the **core innovation** of the model. The cluster label assigned to each employee is appended as a new feature to the original dataset.

```
Hybrid Features = Original 31 Features  +  Cluster Label (0, 1, or 2)
```

This allows the Random Forest to use structural group membership as context, improving its ability to distinguish at-risk employees within the same demographic or role.

---

## Step 5 — Supervised Learning: Random Forest Classifier

A **Random Forest Classifier** is trained on the hybrid feature set to predict whether an employee will leave.

### Why Random Forest?

- Handles both numerical and categorical features well
- Robust to outliers and overfitting through ensemble averaging
- Provides built-in feature importance scores
- Works well with imbalanced datasets when `class_weight='balanced'` is set

### Configuration

```python
rf_model = RandomForestClassifier(
    n_estimators=200,        # 200 decision trees
    max_depth=10,            # Limit tree depth to prevent overfitting
    min_samples_split=5,     # Minimum samples required to split a node
    class_weight='balanced', # Compensates for 84/16 class imbalance
    random_state=42
)
```

### Train / Test Split

The dataset is split **80% training / 20% testing** using stratified sampling to preserve the original class ratio in both splits.

---

## Step 6 — Model Evaluation

The trained model is evaluated on the held-out test set using the following metrics:

| Metric | What It Measures |
|---|---|
| **Accuracy** | Overall proportion of correct predictions |
| **Precision** | Of predicted leavers, how many actually left |
| **Recall** | Of actual leavers, how many were correctly identified |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Model's ability to separate the two classes across all thresholds |
| **Confusion Matrix** | Visual breakdown of TP, TN, FP, FN |

> **Note:** For attrition prediction, **Recall** is the most critical metric — missing an employee who is about to leave (False Negative) is more costly than a false alarm.

### Expected Performance

| Metric | Expected Score |
|---|---|
| Accuracy | ~87–89% |
| ROC-AUC | ~0.86–0.90 |

---

## Step 7 — Feature Importance Analysis

After training, the Random Forest's built-in feature importance scores are extracted and ranked. This identifies which attributes most strongly drive attrition predictions.

**Top expected drivers of attrition:**

- `OverTime` — Employees working overtime are significantly more likely to leave
- `MonthlyIncome` — Lower-paid employees show higher attrition risk
- `JobSatisfaction` — Low satisfaction scores correlate strongly with leaving
- `Age` — Younger employees tend to leave more often
- `YearsAtCompany` — Early-career tenure is a critical risk window
- `Cluster` — The engineered cluster feature appears in the top predictors, validating the hybrid approach

Output saved as **`feature_importance.png`**.

---

## Output Files Summary

| File | Description |
|---|---|
| `elbow_curve.png` | Inertia vs K — confirms optimal cluster count |
| `cluster_distribution.png` | Attrition counts per cluster |
| `cluster_pca.png` | 2D PCA projection of K-Means clusters |
| `confusion_matrix.png` | Heatmap of predicted vs actual attrition |
| `roc_curve.png` | ROC curve with AUC score |
| `feature_importance.png` | Top 15 features ranked by importance |

---

## Pipeline Summary

```
Raw HR Data (1,470 employees, 35 features)
        │
        ▼
  Data Preprocessing
  (Drop constants → Encode → Scale)
        │
        ▼
  K-Means Clustering (K=3)
  Discovers employee segments
        │
        ▼
  Hybrid Feature Set
  (Original features + Cluster label)
        │
        ▼
  Random Forest Classifier
  Trained on 80% data
        │
        ▼
  Evaluation on 20% test set
  (Accuracy, ROC-AUC, F1, Confusion Matrix)
        │
        ▼
  Feature Importance Analysis
  (Top attrition drivers identified)
```

---

## References

- IBM HR Analytics Employee Attrition & Performance Dataset — [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45, 5–32.
- MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations.* Proc. 5th Berkeley Symposium.
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.

---

*Author: Kartik Saini | IPM06107*
