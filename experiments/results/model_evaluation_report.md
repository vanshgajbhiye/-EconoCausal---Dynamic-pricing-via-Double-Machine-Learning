# EconoCausal Model Evaluation & Reproducibility Audit

**Evaluation Status:** [PASSED]  
**Audit Timestamp:** `2026-11-05T17:52:46.335083+00:00`  
**Model Version:** `1.0.0` (`ec-0f5b28d678d3`)  
**Dataset Size:** `676` rows (Source: `data/raw/retail_price.csv`)  

---

## 1. Causal Identification & Double Machine Learning (DML)
- **Baseline ATE Elasticity:** `-1.2678`
- **Asymptotic Standard Error:** `0.5110`
- **95% Confidence Interval:** `[-2.2693, -0.2663]`
- **Statistical Significance (p-value):** `1.3097e-02`
- **Hypothesis Test (e < 0):** `CONFIRMED (Statistically Significant Inverted Demand)`

---

## 2. Heterogeneous Price Elasticity (CATE)
- **Mean Elasticity:** `-1.2724`
- **Median Elasticity (P50):** `-1.2234`
- **Std Dev:** `0.9020`
- **Min / Max:** `-4.5963` / `0.9133`
- **Elasticity Distribution:**
  - Highly Elastic (`|e| > 1.5`): `41.1%`
  - Elastic (`1.0 < |e| <= 1.5`): `20.4%`
  - Inelastic (`|e| <= 1.0`): `38.5%`

---

## 3. Causal Refutation & Sensitivity Diagnostics
- **Refutation Battery Pass Rate:** `100.0%`
- **Placebo Treatment Test:** `PASSED`
- **Random Common Cause Test:** `PASSED`
- **Data Subsetting Stability:** `PASSED`

---

## 4. Counterfactual Financial Uplift Simulation
- **Projected Revenue Uplift:** `-5.12%`
- **Projected Gross Profit Uplift:** `+18.63%`
- **Demand Volume Shift:** `-18.27%`
- **Portfolio Gross Margin:** `30.0% -> 37.5%`
- **Pricing Actions:** `82.1% Markup, 9.3% Discount, 8.6% Hold`

---

## 5. Inference Latency & Memory Performance
### Single-SKU Real-Time Inference (`predict_item`)
- **Median Latency (P50):** `126.55 ms`
- **Tail Latency (P95 / P99):** `169.14 ms` / `222.98 ms`
- **Throughput:** `7.5 items/sec`
- **Peak Memory:** `0.187 MiB`

### Representative Batch Scaling
| Batch Size | Batch Latency (ms) | Per-Item Latency (ms) | Throughput (items/sec) | Peak Memory (MiB) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 202.66 | 202.658 | 4.9 | 0.101 |
| 10 | 270.26 | 27.026 | 37.0 | 0.099 |
| 50 | 278.46 | 5.569 | 179.6 | 0.140 |
| 100 | 362.79 | 3.628 | 275.6 | 0.215 |
| 500 | 1051.85 | 2.104 | 475.4 | 0.943 |
| 676 | 1371.49 | 2.029 | 492.9 | 1.270 |

---

## 6. Reproducibility & Determinism Verification
- **Determinism Check:** `PASSED (Exact bitwise/numerical match across duplicate runs)`
- **Max Elasticity Absolute Difference:** `0.00e+00`
- **Max Price Absolute Difference:** `0.00e+00`
- **Identical Actions Across Runs:** `100%`

---

## 7. Version & Environment Compatibility Audit
- **Compatibility Status:** `COMPATIBLE`
- **Warnings:** `0`
- **Errors:** `0`

---
*Report generated deterministically by EconoCausal Evaluation Engine.*
