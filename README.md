# -EconoCausal---Dynamic-pricing-via-Double-Machine-Learning

📌 Project Overview

EconoCausal: Dynamic Pricing via Double Machine Learning is a machine learning project designed to improve pricing decisions using causal inference and advanced statistical learning techniques. The project aims to understand how changes in product prices affect customer demand and sales by analyzing historical data and estimating the causal impact of pricing decisions.

Traditional pricing strategies often rely on historical trends and basic statistical relationships, which may not accurately capture the actual effect of price changes. EconoCausal addresses this challenge by applying Double Machine Learning (DML) to estimate the causal relationship between product prices and business outcomes while accounting for other influencing factors.

🎯 Objectives

- Analyze historical pricing and sales data.
- Estimate the causal effect of price changes on product demand.
- Apply Double Machine Learning techniques for causal effect estimation.
- Identify pricing strategies that can help improve business performance.
- Support data-driven and evidence-based pricing decisions.
- Evaluate how different pricing decisions may affect revenue and demand.

⚙️ Methodology

The project follows these main steps:

1. Data Collection: Collect relevant historical pricing, sales, demand, and product-related data from suitable datasets.
2. Data Preprocessing: Clean the dataset, handle missing values, identify outliers, and prepare the data for analysis.
3. Exploratory Data Analysis: Explore relationships between product prices, sales, customer demand, and other relevant variables.
4. Feature Engineering: Identify and prepare important features that may influence pricing and demand.
5. Double Machine Learning: Apply DML to estimate the causal effect of price on the selected business outcome while controlling for relevant confounding variables.
6. Pricing Analysis: Use the estimated causal effects to understand how different pricing decisions may influence demand.
7. Evaluation: Evaluate model performance and compare pricing scenarios using appropriate statistical and business metrics.

🧠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Double Machine Learning
- Causal Inference
- Matplotlib
- Seaborn
- Jupyter Notebook

The final technology stack may vary depending on the implementation.

📊 Expected Outcomes

- Better understanding of the relationship between price and demand.
- Estimation of the causal impact of pricing changes.
- Data-driven insights for pricing decisions.
- Comparison of potential pricing strategies.
- A foundation for developing intelligent dynamic pricing systems.

🚀 Applications

- E-commerce platforms
- Retail businesses
- Online marketplaces
- Travel and hospitality
- Revenue management
- Demand forecasting and pricing analytics

🔮 Future Scope

The project can be extended by integrating real-time sales data, developing an interactive pricing dashboard, incorporating customer segmentation, and testing more advanced machine learning models. Future versions could also explore automated pricing recommendations and continuously updated pricing strategies.

👥 Team Collaboration

This project is developed collaboratively using GitHub for version control, code sharing, task management, and tracking contributions. Team members can work on separate components, such as data preprocessing, causal modeling, pricing optimization, and visualization.

## 🚀 Pipeline & CLI Quickstart

### 1. Run Complete Test Suite
```bash
python -m pytest tests/ -v
```

### 2. Train Causal Pipeline
```bash
python -m src.cli train --data data/raw/retail_price.csv --save-model models/econocausal_pipeline.joblib
```

### 3. Generate Catalog Price Recommendations
```bash
python -m src.cli recommend --model models/econocausal_pipeline.joblib --data data/raw/retail_price.csv --output data/pricing_recommendations.csv --objective profit
```

### 4. Real-Time Single-SKU Pricing Inference
```bash
python -m src.cli predict-item --model models/econocausal_pipeline.joblib --category health_beauty --price 105.00 --cost 70.00 --comp-price 108.00
```

### 5. Multi-Strategy Counterfactual Uplift Simulation
```bash
python -m src.cli simulate --model models/econocausal_pipeline.joblib --data data/raw/retail_price.csv
```

---

## 🚀 REST API Microservice (FastAPI / Pydantic)

EconoCausal provides an asynchronous REST API microservice delivering sub-15ms causal dynamic pricing inference, batch catalog optimization, and counterfactual scenario simulations.

### 1. Launch API Microservice
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation: `http://localhost:8000/docs`  
ReDoc alternative documentation: `http://localhost:8000/redoc`

### 2. Available API Endpoints
| HTTP Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Microservice identification and documentation links |
| `GET` | `/health` | Health & readiness probe (liveness, uptime, model status, baseline ATE) |
| `GET` | `/model/metadata` | Detailed training metadata, ATE confidence intervals, refutation score |
| `POST` | `/predict/price` | Real-time single-SKU dynamic pricing recommendation with profit/revenue optimization |
| `POST` | `/predict/elasticity` | Heterogeneous causal elasticity estimation $\hat{\theta}(X_i)$ |
| `POST` | `/batch/recommend` | High-throughput batch catalog dynamic pricing & portfolio aggregates |
| `POST` | `/simulate/portfolio` | Multi-strategy counterfactual scenario matrix simulation |

### 3. Example Request (Single-SKU Dynamic Pricing)
```bash
curl -X POST "http://localhost:8000/predict/price" \
  -H "Content-Type: application/json" \
  -d '{
    "item": {
      "product_id": "SKU_HEALTH_001",
      "unit_price": 59.90,
      "qty": 15.0,
      "product_category_name": "health_beauty",
      "comp_price": 58.50
    },
    "objective": "profit"
  }'
```

### 4. Run API Latency & Throughput Benchmarks
```bash
python scripts/benchmark_api.py
```


---

## 🐳 Production Docker & Cloud Deployment (Day 8)

### 1. Build & Run via Docker
```bash
# Build multi-stage hardened production image
docker build -t econocausal-pricing-api:1.0.0 .

# Run standalone container with healthcheck
docker run -d -p 8000:8000 --name econocausal-api econocausal-pricing-api:1.0.0

# Verify container health
curl http://localhost:8000/health
```

### 2. Full Stack Orchestration (FastAPI + Prometheus + Grafana)
```bash
# Start microservice, Prometheus scraper, and Grafana dashboard
docker-compose up -d

# Service Endpoints:
# - REST API Microservice:  http://localhost:8000/docs
# - Prometheus Metrics:     http://localhost:8000/metrics
# - Prometheus Web UI:      http://localhost:9090
# - Grafana Dashboards:     http://localhost:3000 (admin / econocausal)
```

### 3. Deploy to Kubernetes
```bash
# Apply production cloud manifests
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
kubectl apply -f deploy/k8s/hpa.yaml
kubectl apply -f deploy/k8s/ingress.yaml

# Check rollout status
kubectl get pods -n econocausal-prod
kubectl get hpa -n econocausal-prod
```

### 4. Continuous Integration & Delivery (CI/CD)
The repository includes automated GitHub Actions (`.github/workflows/ci.yml`) providing:
- Automated code linting & static analysis (`flake8`, `black`)
- Python 3.10 and 3.11 test matrix execution (`pytest tests/`)
- Multi-stage Docker container build with layer caching and smoke tests

---

📌 Conclusion

EconoCausal aims to combine causal inference and machine learning to support more reliable pricing decisions. By estimating the causal impact of price changes rather than relying only on correlations, the project seeks to provide useful insights into demand behavior and help businesses develop more effective pricing strategies.
