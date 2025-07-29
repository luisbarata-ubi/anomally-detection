# Supplementary Tables for ADMXGB: Anomaly Detection in Microservices

This repository contains supplementary data tables from the study:

**"Classification of Anomalies in Microservices Using an XGBoost-based Approach with Data Balancing and Hyperparameter Tuning"**  
Authors: Luís M. Barata, Eurico Lopes, Pedro R. M. Inácio, Mário M. Freire

## Repository Contents

This dataset includes performance evaluation results comparing imbalanced and balanced sampling methods across several machine learning models, under different conditions.

### Available CSV Tables

| Filename                                  | Description                                                                 |
|-------------------------------------------|-----------------------------------------------------------------------------|
| `Table1_Performance_Comparison_WHT.csv`   | Mean Train Accurac, Test Accuracy, F1-Score metrics.                        |
| `Table2_Precision_Recall_AUC_WHT.csv`     | Precision, Recall, and AUC metrics.                                         |
| `Table3_Execution_Memory_CPU_WHT.csv`     | Execution Time, Memory Usage, and CPU Usage metrics.                        |


### Structure of the CSV Files

Each CSV includes:
- The base performance on **Imbalanced (IB)** data.
- Differences when applying:
  - **BDO**: Balanced-oversampling,
  - **BDU**: Balanced-undersampling,
  - **BDH**: Balanced-hybrid sampling.

Example format (columns):
```
Method, Precision (IB [BDO, BDU, BDH]), Recall (IB [BDO, BDU, BDH]), AUC (IB [BDO, BDU, BDH])
```
---

### Also available as PDF file:

Table5.pdf
Table6.pdf
Table7.pdf
