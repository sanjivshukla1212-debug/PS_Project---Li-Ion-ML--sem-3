# Solid-State Lithium Superionic Conductor Screener

An end-to-end machine learning screening pipeline and FastAPI service to predict superionic lithium conductor suitability ($\sigma \ge 10^{-4}\text{ S/cm}$) from crystal structure files (`.cif`) or extracted Sendek structural descriptors.

---

## Overview

Screening solid-state electrolytes typically demands computationally expensive DFT (Density Functional Theory) calculations or lab synthesis. This project implements a lightweight, high-throughput screening tool built on crystallographic topology metrics and supervised machine learning:

1. **Crystallographic Feature Extraction:** Uses `pymatgen` to extract the 5 structural descriptors proposed by Sendek et al. (`SBI`, `AFC`, `LLSD`, `LLB`, `LASD`) directly from raw CIF inputs.
2. **Supervised ML Classification:** Evaluates candidate superionicity using an ensemble Random Forest classifier trained on experimental ionic conductivity benchmarks.
3. **Dual Metric Verification:** Reports both modern non-linear ML probabilities and the original 2017/2019 published linear logistic regression baseline for direct screening comparison.
4. **Interactive API:** Provides structured, production-ready REST endpoints (`/predict` and `/predict-cif`) via FastAPI with automated Swagger documentation.

---

## Model Benchmark & Evaluation

The classification engine was trained on 402 Li-containing crystal structures using group-aware splits (`GroupShuffleSplit` on normalized chemical formulas) to prevent data leakage across isomorphic compounds:

| Model | 5-Fold CV $F_1$ | Test Accuracy | Test Precision | Test Recall | Test $F_1$ | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Selected)** | **0.816** | **85.7%** | **77.4%** | **85.7%** | **0.814** | **0.866** |
| **XGBoost** | 0.808 | 85.7% | 75.8% | 89.3% | 0.820 | 0.866 |
| **Logistic Regression (Trained)** | 0.710 | 63.6% | 50.0% | 67.9% | 0.576 | 0.749 |

### Descriptor Importance (Random Forest Gini Impurity)
* **LASD** (Li–anion shortest distance): **32.3%**
* **SBI** (Sublattice bonding index): **18.3%**
* **LLB** (Li–Li shortest distance / Li-neighbor metric): **18.2%**
* **LLSD** (Li–Li shortest distance): **16.6%**
* **AFC** (Average framework coordination): **14.4%**

---

## Repository Structure

```text
├── descriptors.py             # Pymatgen-based 3D CIF parser & Sendek feature engine
├── main.py                    # FastAPI application, route handlers & response schemas
├── model.joblib               # Serialized Random Forest pipeline (Imputer + Estimator)
├── model_metadata.json        # Feature ordering, positive class definition & training metrics
├── project_dependencies.txt   # Pinned Python dependencies
├── test_pipeline.py           # Automated test suite (Pytest & FastAPI TestClient)
├── ProjectPipeline1.ipynb     # Model training, exploratory analysis & benchmark notebook
└── README.md                  # Project documentation
