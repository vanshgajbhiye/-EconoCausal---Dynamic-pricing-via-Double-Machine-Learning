# Data Dictionary — EconoCausal

This document defines the schema, data types, and causal roles of variables used across the EconoCausal data training pipeline. The variable definitions align with the project's pricing domain and backend contract (`SalesRecord`).

---

## 1. Causal Problem Formulation

The primary question for the Data Training module is:
> **"What is the causal impact of price adjustments (or discount rates) on customer purchasing volume (units sold), after controlling for confounding market, temporal, and promotional variables?"**

Under the Double Machine Learning (DML) / Neyman-orthogonal framework:
- **Outcome ($Y$)**: Customer demand / volume purchased.
- **Treatment ($T$)**: Price offered or percentage discount applied.
- **Confounders ($X$)**: Pre-treatment variables that simultaneously affect both pricing decisions and consumer demand.

---

## 2. Core Variable Schema

| Column Name | Data Type | Role in Causal Graph | Description | Validation Constraints |
| :--- | :--- | :--- | :--- | :--- |
| `product_id` | String / ID | Identifier / Stratification | Unique identifier for product/SKU | Non-empty string |
| `date` | Date / String | Temporal Identifier | Transaction or observation date | Valid ISO 8601 date |
| `price` | Float | **Treatment ($T_1$)** | Active unit selling price during observation window | `price > 0` |
| `discount_pct` | Float | **Treatment ($T_2$)** | Percentage discount off baseline price (0 to 100) | `0.0 <= discount_pct <= 100.0` |
| `units_sold` | Integer / Float | **Outcome ($Y$)** | Demand: quantity of units purchased | `units_sold >= 0` |
| `revenue` | Float | Secondary Business Outcome | Gross sales revenue (`price * units_sold`) | `revenue >= 0` |
| `competitor_price` | Float | **Confounder ($X$)** | Known competitor price for identical / substitute product | `competitor_price >= 0` |
| `is_promotion` | Boolean / Int | **Confounder ($X$)** | Indicator if broader marketing campaign was active (0 or 1) | Binary `{0, 1}` |
| `seasonality_index`| Float | **Confounder ($X$)** | Temporal demand multiplier (1.0 = baseline, >1.0 peak season) | `seasonality_index > 0` |
| `inventory_level` | Integer | **Confounder / Moderator ($X$)** | Available stock at start of observation period | `inventory_level >= 0` |
| `day_of_week` | Integer | **Confounder ($X$)** | Day of week (0 = Monday or Sunday convention, 0–6) | Integer `0` to `6` |

---

## 3. Causal Role Justifications

### Treatment ($T$)
- **Why `price` / `discount_pct`**: Dynamic pricing and discount allocation directly alter customer willingness to pay and demand elasticity.

### Outcome ($Y$)
- **Why `units_sold`**: Demand volume is the primary behavioral response to price variation. Price elasticity $\epsilon$ is directly derived as:
  $$\epsilon = \frac{\partial Y / Y}{\partial T / T}$$

### Covariates & Confounders ($X$)
- **`competitor_price`**: Changes in competitor prices influence our pricing strategy and consumer substitution choice.
- **`seasonality_index` & `day_of_week`**: Temporal cycles drive demand peaks and frequently determine promotional pricing calendars.
- **`is_promotion`**: Marketing campaigns often coincide with price drops and drive autonomous traffic increases.
- **`inventory_level`**: Scarcity triggers clearance markdowns or price surges, while also constraining maximum fulfillable demand.

### Variables Excluded from Confounders (Post-Treatment Leakage)
- Any downstream metrics collected *after* customer purchase (e.g., return rates, customer reviews, subsequent support tickets) must not be included in $X$ to prevent collider stratification bias.

---

## 4. Missing Data & Handling Policy
- **Imputation**: Missing values in continuous confounders ($X$) will be imputed using median/mean strategy from pre-treatment baselines.
- **Missing Treatment ($T$) or Outcome ($Y$)**: Rows with null or corrupt $T$ or $Y$ will be dropped rather than imputed.
- **Duplicate Records**: Deduped on `(product_id, date)`.
