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
- **Day 6**: End-to-End Orchestration & Inference Pipeline / CLI Interface (`src/pipeline.py`, `src/cli.py`, `notebooks/run_day6_pipeline.py`, `tests/test_pipeline.py`, `tests/test_cli.py`). Built `EconoCausalPipeline` integrating anti-leakage preprocessing, DML + CATE estimation, 100% robust causal refutations, constrained dynamic pricing optimization, single-SKU real-time inference, and joblib model persistence (2.1 MB). CLI subcommands (`train`, `recommend`, `predict-item`, `simulate`) verified. Catalog simulation yields +31.16% (+$89,871.91) profit uplift and expands gross margins from 30.00% to 40.38%.
- **Day 7**: High-Performance REST API Microservice (FastAPI / Pydantic) (`src/api/main.py`, `src/api/routes.py`, `src/api/schemas.py`, `src/api/service.py`, `scripts/benchmark_api.py`, `tests/test_api.py`). Built production REST API microservice wrapping the serialized causal inference pipeline into asynchronous endpoints (`/health`, `/model/metadata`, `/predict/price`, `/predict/elasticity`, `/batch/recommend`, `/simulate/portfolio`). Engineered Pydantic V2 request/response contracts, CORS and request-timing middleware (`X-Process-Time-Ms`), and error handlers. Achieved core pipeline compute latency of 9.61 ms, batch throughput of 2,324.6 items/sec, and 14 new tests (83 tests passed total across repository).
- **Day 8**: Production Docker Containerization, Prometheus Observability & Cloud Deployment Architecture (`Dockerfile`, `.dockerignore`, `docker-compose.yml`, `src/api/metrics.py`, `deploy/prometheus/prometheus.yml`, `deploy/k8s/`, `.github/workflows/ci.yml`, `tests/test_observability.py`). Built multi-stage hardened Docker image running as non-root `appuser`, local container orchestration stack (FastAPI + Prometheus + Grafana), Prometheus metrics instrumentation (`/metrics` tracking request throughput, latency histograms, CATE elasticity, active requests, and pricing recommendation counts), production Kubernetes manifests (Deployment with 3 replicas, liveness/readiness probes, Service, ConfigMap, HPA autoscaler, Ingress), and automated GitHub Actions CI/CD pipeline. 11 new tests added; 94 passed out of 94 tests total (100% pass rate).
- **Day 9**: Interactive Streamlit Pricing Dashboard & Counterfactual What-If Visualizer (`src/dashboard/app.py`, `src/dashboard/components.py`, `scripts/run_dashboard.py`, `tests/test_dashboard.py`). Built full multi-tab enterprise executive application providing: (1) Executive KPI cockpit with portfolio profit (+$89,871.91, +31.16%), margin expansion (+1,038 bps), and 6-scenario waterfall comparisons; (2) Interactive SKU elasticity and microeconomic demand/profit curve visualizer in Plotly with Amoroso-Robinson optimal price indicators, competitor anchors, and guardrail bands; (3) Real-time counterfactual what-if sandbox with live sliders for price changes, COGS cost shocks (-30% to +50%), and elasticity overrides; (4) Game-theoretic competitor response simulator analyzing retaliation scenarios (Price Matching, Undercutting -5%/-10%, Cooperative Shift) with cross-price elasticity dynamics and risk ratings; (5) Searchable, filterable catalog batch recommender with timestamped CSV export; (6) Causal model governance and refutation audit scorecard. 8 new tests added; 102 passed out of 102 tests total (100% pass rate).
- **Day 10 (Upcoming)**: Continuous Causal MLOps, Confounder Drift Detection & Capstone Handover (`src/monitoring/drift.py`) — Population stability index (PSI) on confounder distributions, automated refutation re-auditing, and final capstone project handover documentation.

## 7. Current Status

Days 1 through 9 are fully implemented, verified with 102 passing unit and integration tests across the repository, and documented with empirical findings in `docs/experiment_log.md`. All working tree changes are verified.
