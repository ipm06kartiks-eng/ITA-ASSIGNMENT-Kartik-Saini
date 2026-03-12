# Employee Attrition Prediction
## Updated Research Methodology
*K-Means Clustering + Comparative Model Analysis*

---

## 1. Overview

This document describes the revised methodology for predicting employee attrition. The updated pipeline introduces a structured sequence: starting with rigorous data verification, followed by unsupervised K-Means clustering to uncover hidden workforce segments, then applying three supervised classifiers — Random Forest, Logistic Regression, and Decision Tree — and concluding with a comparative evaluation that identifies Logistic Regression as the best-performing model.

---

## 2. Methodology Pipeline

### Step 1 — Data Verification

Before any modelling begins, the dataset is thoroughly inspected to ensure integrity and suitability for analysis. The following checks are performed:

- Inspect the first five rows of the dataset to confirm feature columns and data layout.
- Run `df.info()` to identify data types, null values, and overall shape.
- Examine the distribution of the target variable (`y` — Attrition) to detect class imbalance.

This step ensures downstream models are trained on clean, well-understood data.

---

### Step 2 — K-Means Clustering

K-Means clustering is applied to the scaled feature matrix (`X_scaled`) using the pre-determined optimal number of clusters (`OPTIMAL_K = 3`). The purpose is to segment the workforce into natural groupings before supervised learning.

- Cluster assignments (`cluster_labels`) are generated via `KMeans(n_clusters=3, random_state=42)`.
- Clusters are visualised using PCA-reduced 2D space (`X_pca`) via a scatter plot.
- Attrition rate per cluster is computed to identify high-risk employee segments.

The resulting `cluster_labels` feature is then appended to the feature set (`X_hybrid_with_clusters`), enriching the dataset with unsupervised structure for subsequent supervised models.

---

### Step 3 — Supervised Model Training

Three classifiers are trained and evaluated independently on the cluster-enriched dataset (`X_hybrid_with_clusters`). All models use the same 80/20 stratified train-test split.

**Random Forest Classifier**
- Ensemble of 100 decision trees (`n_estimators=100, random_state=42`).
- Produces class probabilities for ROC-AUC evaluation.
- Feature importances extracted and visualised (top 15 features).

**Logistic Regression**
- Linear classifier with `liblinear` solver (`max_iter=200, random_state=42`).
- Model coefficients used as a proxy for feature importance.
- Evaluated via confusion matrix, classification report, and ROC curve.

**Decision Tree Classifier**
- Single-tree classifier (`random_state=42`) for interpretability.
- Feature importances extracted and compared with other models.
- Evaluated with the same metrics suite as the other classifiers.

---

### Step 4 — Model Comparison

All three models are retrained in a unified comparison block and their performance metrics are collated into a summary table. The evaluation criteria include:

- Overall Accuracy
- AUC-ROC Score
- Precision for the Attrition class (Class 1)
- Recall for the Attrition class (Class 1)

ROC curves for all three models are overlaid in a single plot to enable visual comparison of discriminatory power.

---

## 3. Model Performance Results

The table below summarises the key metrics for each model evaluated on the held-out test set (20% of the dataset):

| Model                  | Accuracy | AUC    | Precision (Attrition) | Recall (Attrition) |
| :--------------------- | :------- | :----- | :-------------------- | :----------------- |
| **Logistic Regression**| 0.8810   | 0.8169 | 0.8750                | 0.2979             |
| **Random Forest**      | 0.8367   | 0.7579 | 0.4444                | 0.0851             |
| **Decision Tree**      | 0.7925   | 0.6009 | 0.3409                | 0.3191             |

---

## 4. Conclusion — Best Predictor

Based on the comparative evaluation, **Logistic Regression** is identified as the best-performing model for employee attrition prediction under the current methodology. The key reasons are:

- Highest overall accuracy (**88.10%**) across all three models.
- Best AUC score (**0.8169**), indicating superior ability to discriminate between attriting and non-attriting employees across all decision thresholds.
- Highest precision for the Attrition class (**0.8750**) — when the model flags an employee as at-risk, it is correct approximately 87.5% of the time.

A notable limitation across all models is the **low recall** for the attrition class, meaning that a significant proportion of employees who will actually leave are not identified. This is largely attributable to the class imbalance in the dataset (fewer attrition cases than non-attrition).

---

## 5. Recommended Next Steps

1. **Address Class Imbalance** — Apply SMOTE (Synthetic Minority Over-sampling Technique) or use `class_weight='balanced'` during model training to improve minority class detection.
2. **Hyperparameter Tuning** — Use `GridSearchCV` or `RandomizedSearchCV` to optimise Logistic Regression and Random Forest parameters (`C`, `max_depth`, etc.).
3. **Explore Gradient Boosting** — Evaluate XGBoost or LightGBM as they are typically better suited for imbalanced tabular classification tasks.
4. **Feature Engineering** — Derive new features (e.g., tenure ratios, satisfaction indices) in collaboration with HR domain experts.
5. **Cross-Validation** — Replace single train-test splits with k-fold cross-validation to obtain more robust and generalisable performance estimates.

---

*End of Methodology Document*
