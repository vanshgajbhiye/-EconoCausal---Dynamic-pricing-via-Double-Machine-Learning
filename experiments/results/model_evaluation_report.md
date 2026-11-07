# EconoCausal Model Evaluation & Reproducibility Audit

**Evaluation Status:** [PASSED]  
**Audit Timestamp:** `2026-11-07T08:09:42.571622+00:00`  
**Model Version:** `1.0.0` (`ec-946a1c0a28b8`)  
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
- **Median Latency (P50):** `145.27 ms`
- **Tail Latency (P95 / P99):** `191.99 ms` / `247.74 ms`
- **Throughput:** `6.6 items/sec`
- **Peak Memory:** `0.206 MiB`

### Representative Batch Scaling
| Batch Size | Batch Latency (ms) | Per-Item Latency (ms) | Throughput (items/sec) | Peak Memory (MiB) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 197.60 | 197.599 | 5.1 | 0.105 |
| 10 | 213.33 | 21.333 | 46.9 | 0.099 |
| 50 | 272.98 | 5.460 | 183.2 | 0.139 |
| 100 | 428.60 | 4.286 | 233.3 | 0.214 |
| 676 | 1383.32 | 2.046 | 488.7 | 1.271 |

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
