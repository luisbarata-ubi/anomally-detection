# Supplementary Tables for ADMXGB: Anomaly Detection in Microservices

This repository contains supplementary data tables from the study:

**"Classification of Anomalies in Microservices Using an XGBoost-based Approach with Data Balancing and Hyperparameter Tuning"**  
Authors: Luís M. Barata, Eurico Lopes, Pedro R. M. Inácio, Mário M. Freire

---

## 📄 Table 1 — Performance Comparison (WHT)

- **Filename:** `Table1_Performance_Comparison_WHT.csv`
- **Description:**  
  Mean train accuracy, test accuracy, and F1-score for various anomaly detection methods **without hyperparameter tuning** (WHT).  
  The values are reported for:
  - **IB**: Imbalanced dataset  
  - **BDO**: Balanced dataset using Oversampling  
  - **BDU**: Balanced dataset using Undersampling  
  - **BDH**: Balanced dataset using Hybrid sampling  

- **Columns:**
  - `Method`: Model evaluated
  - `Mean Train Accuracy`: Accuracy on training set, including [BDO, BDU, BDH] deltas
  - `Test Accuracy`: Accuracy on test set, with deltas
  - `F1-Score`: Mean F1-Score with deltas

---

