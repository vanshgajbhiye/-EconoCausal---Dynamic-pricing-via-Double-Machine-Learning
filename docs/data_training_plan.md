# Data Training Plan — EconoCausal

## 1. Objective

The objective of the data training module is to prepare retail data and develop a machine learning pipeline to support dynamic pricing using Double Machine Learning (DML).

## 2. Responsibilities

- Understand the retail dataset and identify relevant features.
- Check missing values, duplicate records, and data types.
- Identify treatment, outcome, and confounding variables.
- Perform data cleaning and preprocessing.
- Prepare the dataset for causal effect estimation.
- Train and evaluate the model using appropriate metrics.
- Document experiments, findings, and limitations.

## 3. Proposed Workflow

1. Dataset inspection
2. Data quality checks
3. Data preprocessing
4. Feature identification
5. Treatment and outcome definition
6. Double Machine Learning model development
7. Model evaluation and validation
8. Documentation of results

## 4. Initial Data Requirements

The dataset should ideally contain customer or transaction features, discount information, purchase outcomes, and relevant confounding variables.

The exact columns and preprocessing steps will be finalized after inspecting the available dataset.

## 5. Validation Plan

The model will be evaluated using suitable metrics for the outcome type. Causal estimates will be checked for stability and interpreted carefully. Where appropriate, causal validation and refutation tests will be performed.

## 6. Daily Milestone Roadmap & Progress

- **Day 1**: Dataset exploration, schema validation, and Kaggle Retail Price Optimization data loading pipeline (`src/data/load_data.py`).
- **Day 2**: Anti-leakage causal preprocessor, scaling, one-hot encoding, and K-fold cross-fitting fold generator (`src/data/preprocess.py`).
- **Day 3**: Double Machine Learning (DML) engine with Robinson residualization, Neyman-orthogonal score, and statistical inference (`src/models/train_dml.py`). Baseline elasticity: $\hat{\theta} = -1.2678$ ($p = 0.0131$).
- **Day 4**: Heterogeneous Treatment Effect (CATE) engine (`src/models/cate.py`) and Causal Refutation suite (`src/models/refutation.py`) testing placebo treatments, random common causes, and subsampling (100% pass score).
- **Day 5**: Dynamic Pricing Optimizer & Counterfactual Uplift Simulation engine (`src/models/optimizer.py`, `notebooks/run_day5_simulation.py`, `tests/test_optimizer.py`). Evaluated pricing strategies with Amoroso-Robinson profit maximization, margin floors, and guardrails (+29.96% profit uplift across catalog).
- **Day 6 (Upcoming)**: End-to-End Orchestration & Inference Pipeline / CLI Interface for Batch and Real-Time Pricing Inference.

## 7. Current Status

Days 1 through 5 are fully implemented, verified with 51 passing unit and integration tests across the repository, and documented with empirical findings in `docs/experiment_log.md`. All working tree changes are verified.
