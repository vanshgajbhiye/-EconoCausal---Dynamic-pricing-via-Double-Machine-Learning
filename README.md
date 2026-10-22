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

📌 Conclusion

EconoCausal aims to combine causal inference and machine learning to support more reliable pricing decisions. By estimating the causal impact of price changes rather than relying only on correlations, the project seeks to provide useful insights into demand behavior and help businesses develop more effective pricing strategies.
