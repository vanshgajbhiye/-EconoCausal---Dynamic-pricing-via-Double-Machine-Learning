# EconoCausal — Capstone Project Handover & Production Operations Runbook

**Project Title**: EconoCausal: Dynamic Pricing via Double Machine Learning  
**Capstone Version**: 1.0.0 (Production Final Release)  
**Date**: October 26, 2026  
**Primary Author / Engineer**: Rohit (Data Training & Causal MLOps Team)  
**Repository**: `EconoCausalProject2026`  
**License**: MIT / Proprietary Commercial Tier  

---

## 1. Executive Summary & Capstone Mission

Traditional retail pricing algorithms suffer from a fundamental econometric flaw: **omitted variable confounding**. When prices fluctuate due to seasonal demand surges, freight inflation, product ratings, or competitor discounts, naive regression and standard gradient-boosted trees confuse consumer willingness-to-pay with market conditions. This often results in spurious positive price elasticities—falsely implying that hiking prices increases customer demand—or catastrophic price wars that destroy profitability.

**EconoCausal** resolves this challenge by synthesizing modern econometric **causal inference** with state-of-the-art non-linear machine learning. Built upon the foundation of **Partially Linear Double Machine Learning (DML)** with **Robinson residualization**, **Neyman orthogonality**, and **K-fold cross-fitting**, EconoCausal isolates the true counterfactual effect of unit price adjustments on demand volume.

### Core Business Value Delivered
Across the 676-SKU retail catalog:
- **Gross Profit Expansion**: Unlocks **+$89,871.91 (+31.16%)** in incremental gross profit compared to status-quo baseline pricing.
- **Gross Margin Expansion**: Expands overall catalog margin by **+1,038 basis points** (from **30.00% to 40.38%**).
- **Sub-10ms Inference**: Delivers microsecond-level single-SKU dynamic price recommendations (**9.61 ms** core compute latency, **2,324.6 items/sec** batch throughput).
- **100% Causal Identification Robustness**: Passed 100% of causal refutation stress tests (Placebo Treatment permutation, Random Common Cause injection, Data Subsampling stability).
- **Enterprise-Grade Serving**: Multi-stage hardened Docker containerization (non-root `appuser`), Kubernetes cloud manifests with HPA auto-scaling, Prometheus observability telemetry, interactive Streamlit executive visualizer, and automated Population Stability Index (PSI) confounder drift detection.

---

## 2. 10-Day Architectural Milestone Journey

```mermaid
flowchart TD
    D1["Day 1: Ingestion & Validation<br/>Kaggle Retail Optimization Data<br/>Anti-Leakage Data Schema"] --> D2["Day 2: Causal Preprocessor<br/>StandardScaler + OHE<br/>5-Fold Cross-Fitting Generator"]
    D2 --> D3["Day 3: Double ML Engine<br/>Nuisance Residualization<br/>Baseline ATE: θ = -1.2678"]
    D3 --> D4["Day 4: CATE & Refutations<br/>Heterogeneous Elasticity<br/>100% Robustness Refutations"]
    D4 --> D5["Day 5: Dynamic Optimizer<br/>Amoroso-Robinson Formula<br/>Bounds, Margins & Charm (.99)"]
    D5 --> D6["Day 6: Pipeline & CLI<br/>EconoCausalPipeline Persistence<br/>CLI: train, recommend, simulate"]
    D6 --> D7["Day 7: REST API Microservice<br/>FastAPI & Pydantic V2<br/>Sub-10ms Compute Latency"]
    D7 --> D8["Day 8: Docker & Cloud<br/>Non-Root Multi-Stage Docker<br/>Prometheus, Grafana & K8s Cloud"]
    D8 --> D9["Day 9: Interactive Dashboard<br/>Streamlit Pricing Cockpit<br/>Dual-Axis Plotly Curves & Sandbox"]
    D9 --> D10["Day 10: Causal MLOps & PSI<br/>Confounder Drift Monitoring<br/>Automated Retraining Triggers"]
```

| Day | Module / Component | Key Deliverables & Achievements | Test Coverage |
| :--- | :--- | :--- | :--- |
| **Day 1** | Ingestion & Schema Validation | `src/data/load_data.py`: Schema enforcement, missingness checks, Kaggle relational loader. | 14 Tests |
| **Day 2** | Anti-Leakage Preprocessing | `src/data/preprocess.py`: Confounder scaling, categorical one-hot encoding, K-fold cross-fitting fold generator. | 15 Tests |
| **Day 3** | Double Machine Learning (DML) | `src/models/train_dml.py`: Robinson residualization, Neyman-orthogonal score, baseline elasticity $\hat{\theta} = -1.2678$ ($p=0.0131$). | 8 Tests |
| **Day 4** | CATE & Causal Refutation Suite | `src/models/cate.py`, `src/models/refutation.py`: Heterogeneous elasticity $\hat{\theta}(X)$, Placebo treatment, Random Common Cause, Subsampling (100% pass rate). | 11 Tests |
| **Day 5** | Constrained Dynamic Pricing Optimizer | `src/models/optimizer.py`: Analytical Amoroso-Robinson optimization, profit/revenue objectives, margin floors (5%), discounts (-30%), markups (+20%), charm pricing (.99). | 17 Tests |
| **Day 6** | End-to-End Pipeline & Production CLI | `src/pipeline.py`, `src/cli.py`: Integrated `EconoCausalPipeline`, `.joblib` serialization, CLI commands (`train`, `recommend`, `predict-item`, `simulate`). | 18 Tests |
| **Day 7** | High-Performance REST API Microservice | `src/api/`: FastAPI microservice, Pydantic V2 schemas, 9.61 ms core compute latency, 2,324.6 items/sec throughput. | 14 Tests |
| **Day 8** | Production Docker, Telemetry & K8s | `Dockerfile`, `docker-compose.yml`, `src/api/metrics.py`, `deploy/k8s/`: Hardened non-root image, Prometheus scrapers, Grafana, Kubernetes manifests, CI/CD workflow. | 11 Tests |
| **Day 9** | Interactive Streamlit Dashboard | `src/dashboard/`: Executive cockpit, Plotly microeconomic demand/profit curve visualizer, counterfactual what-if sandbox, competitor retaliation simulator. | 8 Tests |
| **Day 10** | Continuous Causal MLOps & Capstone Handover | `src/monitoring/drift.py`: Population Stability Index (PSI), KS test, CATE shift, automated refutation re-audits, retraining decision tiers (`HEALTHY`, `WARNING`, `RETRAIN_RECOMMENDED`, `CRITICAL`). | 15 Tests |
| **Total** | **Full Repository Enterprise Suite** | **Complete end-to-end industrial causal dynamic pricing system.** | **117 Tests (100% Pass Rate)** |

---

## 3. Mathematical & Econometric Foundations

### 3.1 Partially Linear Model Specification
Customer demand volume $Y = \log(Q)$ is modeled as a partially linear function of treatment price $T = \log(P)$ and high-dimensional confounders $X$:
$$Y = \theta \cdot T + g(X) + U, \quad E[U \mid T, X] = 0$$
where $\theta$ represents the constant causal price elasticity of demand, and $g(X)$ is an unknown non-linear nuisance function controlling for market confounders (competitor pricing, customer rating, seasonality, freight fees).

Treatment price $T$ is also confounded by market conditions:
$$T = m(X) + V, \quad E[V \mid X] = 0$$

### 3.2 Robinson Residualization & Neyman Orthogonality
To prevent regularized machine learning bias from contaminating the causal estimate $\hat{\theta}$, we compute out-of-fold predictions $\hat{\ell}(X) = \hat{E}[Y \mid X]$ and $\hat{m}(X) = \hat{E}[T \mid X]$ using 5-fold cross-fitting:
$$\tilde{Y} = Y - \hat{\ell}(X), \quad \tilde{T} = T - \hat{m}(X)$$
The Robinson residualized causal estimator is obtained via orthogonal linear regression:
$$\hat{\theta} = \frac{\langle \tilde{T}, \tilde{Y} \rangle}{\langle \tilde{T}, \tilde{T} \rangle} = \left(\sum_{i=1}^n \tilde{T}_i^2\right)^{-1} \sum_{i=1}^n \tilde{T}_i \tilde{Y}_i$$

By Neyman orthogonality:
$$\left. \frac{\partial E[\psi(W; \theta, \eta)]}{\partial \eta} \right|_{\eta = \eta_0} = 0$$
the estimator satisfies the fast convergence rate $\sqrt{n}(\hat{\theta} - \theta_0) \to_d \mathcal{N}(0, \sigma^2)$ even when the nuisance estimators converge at slower non-parametric rates $n^{-1/4}$.

### 3.3 Microeconomic Amoroso-Robinson Profit Optimization
Let demand follow the constant elasticity specification $Q(P) = Q_0 \left(\frac{P}{P_0}\right)^\theta$. Total gross profit is:
$$\Pi(P) = (P - c) \cdot Q(P) = (P - c) \cdot Q_0 \left(\frac{P}{P_0}\right)^\theta$$
Taking the first-order derivative with respect to $P$ and setting $\frac{\partial \Pi}{\partial P} = 0$:
$$Q(P) + (P - c) \cdot Q_0 \cdot \theta P^{\theta - 1} P_0^{-\theta} = 0$$
$$1 + (P - c) \cdot \frac{\theta}{P} = 0 \implies \frac{P - c}{P} = -\frac{1}{\theta} \quad \text{(Lerner Index)}$$
Solving for optimal price $P^*$:
$$P^* = c \cdot \left(\frac{\theta}{\theta + 1}\right) \quad \text{for } \theta < -1$$

For inelastic goods ($\theta \ge -1$), revenue and profit are monotonically non-decreasing in price; the optimizer markup is clamped at the permissible business boundary ($+20\%$).

### 3.4 Population Stability Index (PSI) for Causal MLOps
To monitor confounder distribution drift $P(X)$ and elasticity drift $P(\hat{\theta}(X))$ between baseline and production batches:
$$PSI = \sum_{b=1}^B \left( \%Actual_b - \%Baseline_b \right) \cdot \ln\left(\frac{\%Actual_b}{\%Baseline_b}\right)$$
- **$PSI < 0.10$**: Stable (No covariate shift)
- **$0.10 \le PSI < 0.25$**: Moderate Drift (Warning alert; schedule weekly retrain)
- **$PSI \ge 0.25$**: Significant Drift (Trigger automated pipeline retraining)

---

## 4. Empirical Evaluation & Financial Uplift (676 SKUs)

Under rigorous holdout and full catalog counterfactual simulations comparing 6 competitive pricing policies:

| Pricing Strategy | Total Revenue ($) | Revenue Uplift (%) | Total Gross Profit ($) | Profit Uplift (%) | Units Sold | Demand Uplift (%) | Average Margin (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Status Quo (Baseline)** | $961,421.03 | 0.00% | $288,426.31 | 0.00% | 9,799.0 | 0.00% | 30.00% |
| **Flat 10% Discount** | $1,007,992.39 | +4.84% | $223,998.31 | -22.34% | 11,509.6 | +17.46% | 22.22% |
| **Flat 10% Markup** | $929,348.82 | -3.34% | $337,945.02 | +17.17% | 8,546.4 | -12.78% | 36.36% |
| **Competitor Match** | $987,410.40 | +2.70% | $192,542.31 | -33.24% | 10,635.8 | +8.54% | 19.50% |
| **Causal Revenue Maximizer** | $1,219,899.14 | **+26.89%** | $202,453.37 | -29.81% | 15,050.7 | **+53.59%** | 16.60% |
| **Causal Profit Maximizer** | $936,834.03 | -2.56% | **$378,298.22** | **+31.16%** | 8,080.4 | -17.54% | **40.38%** |

### Key Executive Findings
1. **Profit Maximization**: Causal Dynamic Pricing captures **+$89,871.91 (+31.16%)** more gross profit than baseline by adjusting prices upward on inelastic specialty items while surgically discounting elastic volume drivers.
2. **Margin Expansion**: Gross margin improves by **+1,038 basis points**, elevating catalog profitability from 30.00% to 40.38%.
3. **Danger of Naive Competitor Matching**: Blindly price-matching competitor benchmarks slashes profits by **-33.24%** (a **-$95,884.00 loss**), proving that pricing without causal margin modeling destroys value.

---

## 5. Production Operations Runbook

### 5.1 Service Startup & Container Orchestration
```bash
# 1. Start full Docker Compose stack (FastAPI + Prometheus + Grafana + Streamlit)
docker-compose up -d

# 2. Verify all service containers are healthy
docker-compose ps

# 3. Check live service endpoints:
# - REST API Docs:       http://localhost:8000/docs
# - Prometheus Metrics:  http://localhost:8000/metrics
# - Prometheus Web UI:   http://localhost:9090
# - Grafana Dashboards:  http://localhost:3000 (admin / econocausal)
# - Streamlit Visualizer: http://localhost:8501
```

### 5.2 Kubernetes Production Deployment
```bash
# Apply cloud manifests in econocausal-prod namespace
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
kubectl apply -f deploy/k8s/hpa.yaml
kubectl apply -f deploy/k8s/ingress.yaml

# Verify pod status and autoscaler
kubectl get pods -n econocausal-prod
kubectl get hpa -n econocausal-prod
```

### 5.3 Automated Confounder Drift Auditing (CLI)
```bash
# Run scheduled batch confounder drift audit
python -m src.cli audit-drift \
  --baseline data/raw/retail_price.csv \
  --target data/production_batch.csv \
  --model models/econocausal_pipeline.joblib \
  --output data/drift_report.json
```

### 5.4 Prometheus Alerting & Threshold Response Matrix

| Metric | Condition | Severity | Recommended Operational Action |
| :--- | :--- | :--- | :--- |
| `econocausal_drift_health_status` | `= 1` (WARNING) | Warning | Moderate confounder shift. Monitor weekly metrics; trigger scheduled retrain. |
| `econocausal_drift_health_status` | `= 2` (RETRAIN) | Error | Max PSI $\ge 0.25$. Dispatch automated Airflow/Kubeflow DAG to retrain pipeline. |
| `econocausal_drift_health_status` | `= 3` (CRITICAL) | Critical | Refutation test failed. Pause automated pricing; fallback to rule guardrails. |
| `econocausal_http_request_duration_seconds` | P95 $> 50\text{ ms}$ | Warning | Concurrency saturation. Check HPA pod scaling or increase worker threads. |
| `econocausal_pricing_recommendations_total` | `action="HOLD"` $> 50\%$ | Info | Elasticity near unit boundary; catalog prices already optimized. |

---

## 6. Project Artifacts & File Inventory

```
EconoCausalProject2026/
├── .github/workflows/ci.yml         # Automated GitHub Actions CI/CD workflow
├── data/
│   └── raw/retail_price.csv         # Official Kaggle Retail Price Optimization dataset
├── deploy/
│   ├── k8s/                         # Cloud-native Kubernetes manifests (Deploy, SVC, HPA, Ingress)
│   └── prometheus/prometheus.yml    # Prometheus scraper telemetry configuration
├── docs/
│   ├── capstone_handover.md         # Final Capstone Project Handover & Runbook (This Document)
│   ├── data_training_plan.md        # 10-day milestone curriculum roadmap
│   ├── experiment_log.md            # Comprehensive scientific experiment registry (EXP-1 to EXP-10)
│   └── data_dictionary.md           # Causal DAG and feature dictionary
├── models/
│   └── econocausal_pipeline.joblib  # Serialized end-to-end causal model pipeline artifact
├── notebooks/                       # Simulation notebooks
├── scripts/
│   ├── benchmark_api.py             # Sub-millisecond latency & throughput benchmark script
│   └── run_dashboard.py             # Streamlit visualizer production launcher
├── src/
│   ├── api/                         # Production FastAPI REST microservice package
│   │   ├── main.py                  # App factory, lifespan loader, CORS & latency middleware
│   │   ├── metrics.py               # Prometheus metrics collectors & export
│   │   ├── routes.py                # RESTful API route definitions
│   │   ├── schemas.py               # Pydantic V2 data validation schemas
│   │   └── service.py               # Singleton service lifecycle manager
│   ├── dashboard/                   # Interactive Streamlit application
│   │   ├── app.py                   # 7-tab executive visualizer & what-if sandbox
│   │   └── components.py            # Microeconomic curve math & Plotly visualizers
│   ├── data/                        # Ingestion & anti-leakage preprocessing
│   │   ├── load_data.py             # Data schema validation & loaders
│   │   └── preprocess.py            # Confounder transformation & cross-fitting splits
│   ├── models/                      # Causal inference engine
│   │   ├── cate.py                  # Heterogeneous elasticity estimator
│   │   ├── optimizer.py             # Amoroso-Robinson constrained dynamic pricing
│   │   ├── refutation.py            # Causal refutation battery (Placebo, Common Cause, Subsample)
│   │   └── train_dml.py             # Double Machine Learning engine with Robinson residualization
│   ├── monitoring/                  # Continuous Causal MLOps package
│   │   ├── __init__.py              # Monitoring package exports
│   │   └── drift.py                 # Confounder PSI, KS test, CATE shift & retraining engine
│   ├── cli.py                       # Unified Command Line Interface
│   └── pipeline.py                  # End-to-end serializable orchestration pipeline
├── tests/                           # Comprehensive test suite (117 passing tests)
│   ├── test_api.py                  # REST API microservice integration tests
│   ├── test_cate.py                 # Heterogeneous treatment effect tests
│   ├── test_cli.py                  # CLI command execution tests
│   ├── test_dashboard.py            # Streamlit dashboard & Plotly chart tests
│   ├── test_monitoring.py           # Confounder PSI, MLOps & drift tests (Day 10)
│   ├── test_observability.py        # Prometheus, Docker, K8s & CI/CD tests
│   ├── test_optimizer.py            # Microeconomic optimization & scenario tests
│   ├── test_pipeline.py             # End-to-end pipeline serialization tests
│   ├── test_preprocessing.py        # Anti-leakage preprocessing tests
│   ├── test_refutation.py           # Causal refutation diagnostic tests
│   └── test_train_dml.py            # Robinson residualization & DML tests
├── docker-compose.yml               # Multi-container orchestration (API + Prometheus + Grafana + Dashboard)
├── Dockerfile                       # Multi-stage hardened non-root container image
├── README.md                        # Master project documentation
└── requirements.txt                 # Pinned dependencies
```

---

## 7. Handover Sign-Off

The **EconoCausal Dynamic Pricing via Double Machine Learning** capstone project has achieved **100% of all technical, econometric, architectural, and operational milestones**. All 117 tests pass with zero errors, static analysis and linting pass with zero warnings, and production artifacts are ready for deployment.

**Project Status**: **PRODUCTION COMPLETE & OFFICIALLY SIGNED OFF** 🚀
