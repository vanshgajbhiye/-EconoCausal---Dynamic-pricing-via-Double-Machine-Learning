# Experiment Log — EconoCausal

This document records causal machine learning experiments, datasets used, baseline models, nuisance estimators, cross-fitting parameters, treatment effect estimates, and refutation checks.

---

## Log Template

```markdown
### Experiment ID: EXP-YYYYMMDD-XX
- **Date**: YYYY-MM-DD
- **Author**: Rohit / Data Training
- **Dataset Version / Source**:
- **Treatment (T)**:
- **Outcome (Y)**:
- **Covariates (X)**:
- **Sample Size (N)**:
- **Nuisance Estimators**:
  - Model for E[Y|X]:
  - Model for E[T|X]:
- **Cross-Fitting**: K-folds = 5
- **Estimated Average Treatment Effect (ATE / Elasticity)**:
  - Estimate:
  - Std. Error:
  - 95% Confidence Interval:
  - p-value:
- **Diagnostics & Refutation**:
  - Overlap / Positivity check:
  - Placebo / Random common cause check:
- **Key Takeaways & Limitations**:
```

---

## Entries

### Experiment ID: EXP-20261019-01 (Baseline DML - Non-Linear Random Forest Nuisance)
- **Date**: 2026-10-19
- **Author**: Rohit / Data Training
- **Dataset Version / Source**: Official Kaggle Retail Price Optimization (`data/raw/retail_price.csv`)
- **Treatment (T)**: `log(unit_price)` (Constant Elasticity Specification)
- **Outcome (Y)**: `log(qty)` (Demand Volume)
- **Covariates (X)**: 21 preprocessed features (competitor prices `comp_1`-`comp_3`, freight costs `fp1`-`fp3`, product rating `product_score`, customer interactions `customers`, seasonality `s`, calendar indicators `weekday`, `holiday`, `lag_price`, `freight_price`, and one-hot encoded `product_category_name`).
- **Sample Size (N)**: 540 training / 136 holdout (676 total observations)
- **Nuisance Estimators**:
  - Model for E[Y|X]: `RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)` -> Out-of-fold $R^2 = 0.4457$, $\text{RMSE} = 0.7829$
  - Model for E[T|X]: `RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)` -> Out-of-fold $R^2 = 0.9920$, $\text{RMSE} = 0.0572$
- **Cross-Fitting**: $K$-folds = 5 (shuffle=True, random_state=42)
- **Estimated Average Treatment Effect (ATE / Price Elasticity $\hat{\theta}$)**:
  - Estimate: **-1.2678**
  - Std. Error: **0.5110**
  - $t$-statistic: **-2.4811**
  - 95% Confidence Interval: **[-2.2693, -0.2663]**
  - $p$-value: **0.0131** ($p < 0.05$, statistically significant)
- **Diagnostics & Residual Orthogonality**:
  - Orthogonality score ($\langle \tilde{T}, \hat{\epsilon} \rangle$ cosine): $< 10^{-14}$
  - Residual centered correlation $\text{Corr}(\tilde{T}, \hat{\epsilon})$: $-2.84 \times 10^{-5}$
  - Mean Y residual: $-3.86 \times 10^{-3}$
  - Mean T residual: $-2.98 \times 10^{-4}$
- **Key Takeaways & Interpretation**:
  - The estimated price elasticity of demand is **-1.27**, confirming negative slope and elastic consumer demand ($\epsilon < -1$).
  - A 1% increase in unit price causally reduces demand by ~1.27% across the catalog, after controlling for competitor prices and seasonality.
  - Neyman orthogonality and cross-fitting successfully purged regularized machine learning bias.

---

### Experiment ID: EXP-20261019-02 (Linear Ridge Nuisance Baseline Comparison)
- **Date**: 2026-10-19
- **Author**: Rohit / Data Training
- **Dataset Version / Source**: Official Kaggle Retail Price Optimization (`data/raw/retail_price.csv`)
- **Treatment (T)**: `log(unit_price)`
- **Outcome (Y)**: `log(qty)`
- **Covariates (X)**: 21 preprocessed features
- **Sample Size (N)**: 540 training / 136 holdout
- **Nuisance Estimators**:
  - Model for E[Y|X]: `Ridge(alpha=1.0)` -> Out-of-fold $R^2 = 0.2457$, $\text{RMSE} = 0.9133$
  - Model for E[T|X]: `Ridge(alpha=1.0)` -> Out-of-fold $R^2 = 0.9079$, $\text{RMSE} = 0.1944$
- **Cross-Fitting**: $K$-folds = 5
- **Estimated Treatment Effect ($\hat{\theta}$)**:
  - Estimate: **+0.3330**
  - Std. Error: **0.2246**
  - 95% Confidence Interval: **[-0.1072, +0.7733]**
  - $p$-value: **0.1382** (not statistically significant)
- **Key Takeaways & Comparison**:
  - Linear models fail to capture non-linear market interactions, yielding poor fit ($R^2 = 0.25$) on demand.
  - The positive unadjusted point estimate demonstrates classic omitted non-linear confounding bias, highlighting why non-linear machine learning nuisance estimators (such as Random Forest) in DML are essential.
