# Predicting Hospital Readmission for Diabetic Patients using Machine Learning

## Overview

This project focuses on building a predictive model to identify whether a diabetic patient is likely to be readmitted to the hospital within 30 days of discharge.
It uses a real-world dataset of diabetic patients and applies various machine learning techniques such as preprocessing, outlier handling, SMOTE for class imbalance, and models like Logistic Regression, Random Forest, and XGBoost.

## Dataset
- *Source*: Diabetes 130-US Hospitals for Years 1999-2008.
- *Target*: readmitted_binary (0 or 1).
- *Features*: Numerical (e.g., time_in_hospital, number_inpatient), categorical (e.g., age, insulin, diag_1_group).
- *Size*: 10176 rows, 50 columns.

## Preprocessing
- *Outlier Handling*: Used IQR method (clipping values outside Q1 - 1.5*IQR and Q3 + 1.5*IQR).
- *Encoding*: One-hot encoding for categorical columns.
- *Scaling*: Applied StandardScaler to numerical columns.
- *Class Imbalance*: Used SMOTE (random_state=42, k_neighbors=5).

## Models
- *Random Forest*: n_estimators=100, class_weight='balanced', random_state=42.
- *Logistic Regression*: max_iter=1000, class_weight='balanced', random_state=42.
- *XGBoost*: eval_metric='logloss',  random_state=42.

## Evaluation
- *Metrics*: Focused on recall for class 1 (readmissions), F1-score, accuracy, and ROC-AUC due to imbalance.
