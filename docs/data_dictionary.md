# Data Dictionary — EconoCausal

This document defines the schema, data types, and causal roles of variables used across the EconoCausal data training pipeline. The variable definitions align with the project's pricing domain, the team's backend contract (`SalesRecord`), and the official Kaggle dataset: **Retail Price Optimization** (`retail_price.csv`).

Dataset Source: [https://www.kaggle.com/datasets/suddharshan/retail-price-optimization](https://www.kaggle.com/datasets/suddharshan/retail-price-optimization)

---

## 1. Causal Problem Formulation

The primary question for the Data Training module is:
> **"What is the causal impact of product price adjustments on customer purchasing demand (quantity sold), after controlling for confounding market competition, product characteristics, and temporal seasonality?"**

Under the Double Machine Learning (DML) / Neyman-orthogonal framework:
- **Outcome ($Y$)**: Customer demand / volume purchased (`qty`).
- **Treatment ($T$)**: Selling unit price (`unit_price`).
- **Confounders ($X$)**: Pre-treatment variables that simultaneously affect both pricing decisions and consumer demand (competitor pricing, past price, seasonality, customer footprint, freight costs).

---

## 2. Core Variable Schema (`retail_price.csv`)

The official dataset contains 676 monthly observations across multiple retail product categories with zero missing values.

| Column Name | Data Type | Role in Causal Graph | Description | Validation Constraints |
| :--- | :--- | :--- | :--- | :--- |
| `product_id` | String | Identifier / Group | Unique product SKU identifier (e.g., `bed1`, `garden1`) | Non-empty string |
| `product_category_name` | String | Group / Categorical Moderator | Category name (e.g. `bed_bath_table`, `health_beauty`) | Non-empty string |
| `month_year` | String | Date Identifier | Observation month and year (`DD-MM-YYYY`) | Valid date format |
| `unit_price` | Float | **Treatment ($T$)** | Unit selling price of the product | `unit_price > 0` |
| `qty` | Integer | **Outcome ($Y$)** | Demand volume: total units sold during the month | `qty >= 0` |
| `total_price` | Float | Secondary Business Metric | Gross sales revenue (`unit_price * qty`) | `total_price >= 0` |
| `comp_1` | Float | **Confounder ($X$)** | Competitor 1 unit price | `comp_1 > 0` |
| `comp_2` | Float | **Confounder ($X$)** | Competitor 2 unit price | `comp_2 > 0` |
| `comp_3` | Float | **Confounder ($X$)** | Competitor 3 unit price | `comp_3 > 0` |
| `ps1`, `ps2`, `ps3` | Float | **Confounder ($X$)** | Competitors 1, 2, 3 product review rating scores | `0.0 <= score <= 5.0` |
| `fp1`, `fp2`, `fp3` | Float | **Confounder ($X$)** | Competitors 1, 2, 3 freight / shipping prices | `freight >= 0` |
| `lag_price` | Float | **Confounder ($X$)** | Historical unit price from previous period | `lag_price > 0` |
| `freight_price` | Float | **Confounder ($X$)** | Shipping price charged for the product | `freight_price >= 0` |
| `customers` | Integer | **Confounder ($X$)** | Number of unique customer interactions in period | `customers >= 0` |
| `s` | Float | **Confounder ($X$)** | Sales seasonality / demand trend index | Continuous |
| `weekday` | Integer | **Confounder ($X$)** | Number of weekdays in the observation period | `weekday >= 0` |
| `weekend` | Integer | **Confounder ($X$)** | Number of weekend days in observation period | `weekend >= 0` |
| `holiday` | Integer | **Confounder ($X$)** | Number of holiday days in observation period | `holiday >= 0` |
| `month`, `year` | Integer | Temporal Context | Calendar month (1–12) and year | Valid month/year |
| `product_score` | Float | Product Attribute ($X$) | Average customer review score of the product | `1.0 <= score <= 5.0` |
| `product_weight_g` | Float | Product Attribute ($X$) | Product weight in grams | `product_weight_g > 0` |
| `volume` | Float | Product Attribute ($X$) | Product packaging volume | `volume > 0` |

---

## 3. Causal Role Justifications

### Treatment ($T$ = `unit_price`)
- Changes in unit price directly alter customer purchasing decisions according to price elasticity of demand:
  $$\epsilon = \frac{\partial \ln(Q)}{\partial \ln(P)} = \frac{\partial Q / Q}{\partial P / P}$$

### Outcome ($Y$ = `qty`)
- `qty` measures actual consumer demand (units sold). Demand is the immediate behavioral outcome of pricing interventions.

### Confounders ($X$)
- **Competitor Prices (`comp_1`, `comp_2`, `comp_3`)**: Competitor pricing shifts consumer cross-price elasticity (substitution effect) and triggers internal pricing adjustments.
- **Seasonality & Calendar (`s`, `holiday`, `month`, `weekday`)**: Holidays and peak seasons shift baseline demand curves and dictate promotional discount schedules.
- **`lag_price`**: Anchoring effect; historical pricing directly influences customer perception of whether the current price is a deal.
- **Product Attributes (`product_score`, `freight_price`)**: High-rated products or low-shipping items have different price sensitivities than low-rated bulky items.

### Leakage Prevention
- **`total_price`**: Mechanically computed as `unit_price * qty`. Must **NEVER** be included in the confounder matrix $X$ to prevent target leakage.

---

## 4. Missing Data & Handling Policy
- `retail_price.csv` has verified 0 null values across all 676 rows and 30 columns.
- Any future ingested records with non-positive `unit_price` or negative `qty` will be rejected by validation checks.
