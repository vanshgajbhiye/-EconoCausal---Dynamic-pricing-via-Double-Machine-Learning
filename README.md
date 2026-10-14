# EconoCausal — Dynamic Pricing via Double Machine Learning

> **Move beyond "Who is likely to buy?" to "Who will buy ONLY BECAUSE of the discount?"**

---

## 📌 Project Overview

**EconoCausal** is a **Seller's Causal Pricing Dashboard** for e-commerce. It helps sellers make smarter promotional decisions by using **Causal AI** instead of traditional predictive ML.

### The Problem

Standard ML models predict *correlation*, not *causation*. If a seller gives a ₹200 discount and a customer buys, was it *because of* the discount — or would they have bought anyway?

- A traditional model says: *"Users who got discounts bought more → give everyone a discount!"*
- **EconoCausal** asks: *"For THIS specific customer, would they buy IF AND ONLY IF we gave them the discount?"*

### The Solution

EconoCausal uses **Double Machine Learning (DML)** to calculate the **Individual Treatment Effect (ITE)** — the exact causal impact of a discount on each customer's purchase probability. It then uses **budget-constrained optimization** to allocate personalized discounts only to customers where the money actually makes a difference.

---

## 🎯 Core Concepts

### Causal Inference vs. Predictive ML

| Traditional ML (Predictive) | EconoCausal (Causal / Prescriptive) |
|---|---|
| "This shoe sells well at ₹2,000" | "If you drop the price by ₹200, sales increase by **exactly 15%** for this segment" |
| "Competitors are giving 10% off" | "A 10% discount will only persuade **23% of customers** — the rest would've bought anyway" |
| "Here's what others charge" | "Here's your **mathematically optimal** discount per customer, under a ₹50K budget" |

### Customer Segmentation by Causal Response

EconoCausal classifies every customer into one of four segments based on their **causal response** to discounts:

| Segment | Behavior | Action |
|---|---|---|
| 🟢 **Persuadables** | Buy *only because of* the discount | ✅ Target these — money well spent |
| 🔵 **Sure Things** | Buy regardless of discount | 🚫 Don't discount — they're buying anyway |
| 🔴 **Lost Causes** | Won't buy even with a discount | 🚫 Ignore — don't waste budget |
| ⚫ **Sleeping Dogs** | Discount makes them *less* likely to buy (suspicious) | ⚠️ Definitely don't discount |

### Customer Graduation (Lifecycle)

A key innovation: customers don't stay "Persuadable" forever. After repeated discount-driven purchases, they develop brand loyalty and **graduate** to "Sure Thing" status — at which point discounts are tapered off and the freed budget is reallocated to new Persuadables.

```
Persuadable (needs discount) ──[buys 3-4 times]──► Sure Thing (buys without discount)
         💸 Give discount                              🚫 Stop discount, save money
```

| Purchase # | Customer's ITE | Discount Given | Status |
|---|---|---|---|
| 1st | +54% (High) | ₹200 off | Persuadable |
| 2nd | +41% (High) | ₹200 off | Persuadable |
| 3rd | +22% (Medium) | ₹100 off | Tapering |
| 4th | +8% (Low) | ₹50 off | Tapering |
| 5th | +2% (Negligible) | ₹0 | 🎓 Graduated! |

The causal model re-runs periodically on updated data. As a customer's ITE naturally drops, the system auto-reduces their discount.

---

## 🚶 Full Walkthrough — A Seller's Journey

> **Scenario:** You are Rajesh, owner of an online shoe store. Diwali is coming. You sell **Bata Power 500** running shoes (MRP ₹2,499) and have a marketing budget of **₹50,000**. You want to run a promotion but don't want to waste money on customers who'd buy anyway.

### Step 1: Search Your Product

You open EconoCausal and search **"Bata Power 500"**. The dashboard shows market context:

| Metric | Value |
|---|---|
| Your Price | ₹2,499 |
| Avg. Competitor Price | ₹2,350 |
| Amazon Price | ₹2,299 |
| Flipkart Price | ₹2,399 |
| Season | Pre-Diwali (High demand) |
| Regions Trending In | Maharashtra, Tamil Nadu, Delhi |
| 30-Day Sales Velocity | 142 units/month |

### Step 2: Set Up a Campaign

You click **"Plan Campaign"** and enter:
- **Budget:** ₹50,000
- **Discount options to test:** ₹0, ₹100, ₹200, ₹300
- **Target audience:** 5,000 past visitors who browsed but didn't all buy

### Step 3: Causal Engine Analyzes Each Customer

The Double ML engine calculates the **Individual Treatment Effect (ITE)** for each of the 5,000 customers. Here are three examples:

#### Customer A — Priya (College Student, Mumbai)

| Factor | Value |
|---|---|
| Past purchases | 0 (never bought) |
| Browse count | 12 times in 2 weeks |
| Cart abandoned | 3 times |
| Income bracket | Low-Medium |
| **Without discount** | **8% chance of buying** |
| **With ₹200 discount** | **62% chance of buying** |
| **ITE** | **+54%** 🟢 |

**Verdict: PERSUADABLE** — She wants the shoes but can't justify full price. The discount *causes* her purchase.

#### Customer B — Amit (IT Professional, Bangalore)

| Factor | Value |
|---|---|
| Past purchases | 8 (loyal customer) |
| Browse count | 2 times |
| Cart status | Added, likely buying |
| Income bracket | High |
| **Without discount** | **88% chance of buying** |
| **With ₹200 discount** | **91% chance of buying** |
| **ITE** | **+3%** 🔴 |

**Verdict: SURE THING** — He's buying anyway. Giving him ₹200 off = throwing away ₹200.

#### Customer C — Deepa (Housewife, Jaipur)

| Factor | Value |
|---|---|
| Past purchases | 0 |
| Browse count | 1 (from a random ad) |
| Cart status | Never added |
| Income bracket | Medium |
| **Without discount** | **2% chance of buying** |
| **With ₹200 discount** | **5% chance of buying** |
| **ITE** | **+3%** 🔴 |

**Verdict: LOST CAUSE** — Not interested. Even with a discount, she won't buy.

### Step 4: See Customer Segmentation

The dashboard shows how the 5,000 customers break down:

| Segment | Count | % | Action |
|---|---|---|---|
| 🟢 Persuadables | 1,200 | 24% | **Target these** |
| 🔵 Sure Things | 1,800 | 36% | Don't discount |
| 🔴 Lost Causes | 1,700 | 34% | Ignore |
| ⚫ Sleeping Dogs | 300 | 6% | Avoid discounting |

### Step 5: Get Optimal Budget Allocation

The SciPy optimizer calculates the **mathematically optimal discount** for each Persuadable under the ₹50,000 budget:

| Customer | ITE at ₹100 | ITE at ₹200 | ITE at ₹300 | Optimal Discount |
|---|---|---|---|---|
| Priya | +32% | +54% | +58% | **₹200** (₹300 = diminishing returns) |
| Rahul | +45% | +48% | +49% | **₹100** (₹100 is enough) |
| Sneha | +10% | +38% | +61% | **₹300** (needs a big push) |

Final allocation:
- 400 customers → **₹100 off** = ₹40,000
- 45 customers → **₹200 off** = ₹9,000
- 3 customers → **₹300 off** = ₹900
- **Total: ₹49,900** (within budget ✅)

### Step 6: ROI Comparison

| Strategy | Budget | Extra Sales | Revenue | ROI |
|---|---|---|---|---|
| ❌ Blanket ₹100 to everyone | ₹50,000 | 89 units | ₹2.2L | 4.4x |
| ❌ Random targeting | ₹50,000 | 134 units | ₹3.3L | 6.6x |
| ✅ **EconoCausal Optimal** | ₹49,900 | **312 units** | **₹7.8L** | **15.6x** |

> *"This strategy saves ₹4,20,000 compared to blanket targeting and generates 3.5x more revenue."*

### Step 7: Uplift Curve (Visual Proof)

The dashboard displays a **Qini/Uplift curve** — X-axis is "% of customers targeted" (sorted by ITE), Y-axis is "incremental purchases caused". The EconoCausal curve rises steeply (Persuadables first), while random targeting follows a flat diagonal. This is mathematical proof the causal model works.

### Step 8: Graduation Tracker (Over Time)

After the campaign, as Persuadables become repeat buyers, the dashboard shows:

> *"42 customers graduated this month — saving ₹8,400/month."*
> *"₹8,400 reallocated to 67 new Persuadables."*

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     React Frontend (Dashboard)                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌───────────┐  │
│  │ Product   │  │ Customer     │  │ Uplift /  │  │ Allocation│  │
│  │ Search &  │  │ Segmentation │  │ Qini      │  │ Matrix &  │  │
│  │ Market    │  │ Pie Chart    │  │ Curves    │  │ Graduation│  │
│  │ Context   │  │              │  │ (Plotly)  │  │ Tracker   │  │
│  └──────────┘  └──────────────┘  └───────────┘  └───────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ REST API
┌───────────────────────────┴─────────────────────────────────────┐
│                     Node.js / Express Backend                   │
│         (API Gateway, Auth, Data Ingestion, Drift Detection)    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                     Python Causal ML Engine                     │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ DoWhy        │  │ EconML       │  │ SciPy Optimizer    │    │
│  │ Causal DAG   │  │ Double ML    │  │ Budget-Constrained │    │
│  │ & Refutation │  │ ITE Estimator│  │ Allocation         │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ Propensity   │  │ Customer     │  │ Data Drift         │    │
│  │ Score        │  │ Graduation   │  │ Detector           │    │
│  │ Matching     │  │ Tracker      │  │                    │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Modules

| Module | Tech | Purpose |
|---|---|---|
| **Causal Inference Engine** | Python, EconML, DoWhy | Double/Debiased ML to isolate true causation from confounders |
| **Propensity Score Matching** | Python, Scikit-learn | Balances observational data to simulate a randomized A/B test |
| **Budget Optimizer** | SciPy | Maximizes revenue under a fixed budget constraint |
| **Customer Graduation** | Python | Tracks ITE decay over time, tapers discounts for loyal customers |
| **Data Drift Detector** | Python | Warns if customer behavior shifts, invalidating the model |
| **REST API** | Node.js, Express | API gateway connecting frontend to the causal engine |
| **Dashboard** | React, Plotly.js | Product search, segmentation charts, uplift curves, allocation table |

### Feedback Loop

The system is not a one-shot tool — it's a **living system**:

```
1. Run Causal Model → get ITE per customer
2. Allocate discounts to high-ITE users (Persuadables)
3. Campaign runs, sales happen
4. New purchase data flows back in
5. Re-run Causal Model → ITE drops for repeat buyers
6. System auto-reduces/stops their discount (Graduation)
7. Freed budget → target NEW Persuadables
```

---

## 🧠 Technologies

| Layer | Stack |
|---|---|
| **Frontend** | React, Plotly.js, CSS |
| **Backend API** | Node.js, Express |
| **Causal ML** | Python, EconML, DoWhy, Scikit-learn, LightGBM |
| **Optimization** | SciPy |
| **Data** | Pandas, NumPy |
| **Visualization** | Plotly.js (frontend), Matplotlib/Seaborn (analysis) |

---

## 📊 Dashboard Views

1. **Product Search & Market Context** — Search any product, see competitor prices, regional trends, seasonal signals
2. **Campaign Planner** — Set budget, discount tiers, target audience
3. **Customer Segmentation** — Pie chart of Persuadables / Sure Things / Lost Causes / Sleeping Dogs
4. **Allocation Matrix** — Table showing each customer's optimal discount amount
5. **Uplift & Qini Curves** — Visual proof that causal targeting outperforms random/blanket strategies
6. **ROI Comparison** — Side-by-side of EconoCausal vs blanket vs random strategies
7. **Graduation Tracker** — Customers graduating from Persuadable → Sure Thing over time
8. **Natural Language Summaries** — e.g., *"This strategy saves ₹4,20,000 compared to blanket targeting"*

---

## 📅 Development Plan

| Week | Causal ML & Optimization | Frontend & Visualization |
|---|---|---|
| **Week 1** | Define Causal DAG with DoWhy for mock retail dataset (Confounders, Treatments, Outcomes) | Scaffold React app; product search view; budget input |
| **Week 2** | Train EconML Double ML models (Random Forest / LightGBM base); estimate ITE | Plotly.js Qini & Uplift curves; customer segmentation chart |
| **Mid-Review** | Causal Audit — Refutation Tests proving causal isolation | Dynamic filtering of thousands of ITE scores |
| **Week 3** | SciPy optimizer for budget-constrained allocation; Graduation tracker | Allocation Matrix UI; Graduation Tracker view |
| **Week 4** | REST API wrapping causal engine; Data drift detector | Polish: NL summaries, ROI comparison, final styling |

---

## 🚀 Quick Start

### Backend API (Node.js)

```bash
cd backend
npm install
npm run dev
```

Visit the health check at [http://localhost:5000/api/health](http://localhost:5000/api/health).

### Causal ML Engine (Python) — *Coming Soon*

```bash
cd causal_engine
pip install -r requirements.txt
python main.py
```

---

## 🔮 Future Scope

- **Real-time data integration** — Live sales feeds updating the model continuously
- **Multi-product campaigns** — Optimize across an entire product catalog simultaneously
- **A/B test validation** — Run actual A/B tests to validate causal predictions
- **Advanced customer lifecycle** — Model transitions between all four segments over time
- **Automated campaign execution** — Trigger discount emails/notifications directly from the dashboard

---

## 👥 Team Collaboration

This project is developed collaboratively using GitHub for version control, code sharing, task management, and tracking contributions.

---

## 📌 Conclusion

EconoCausal moves beyond predictive AI into the frontier of **Causal and Prescriptive AI**. Instead of asking *"Who is likely to buy?"*, it answers *"Who will buy ONLY BECAUSE of the discount?"* — enabling sellers to maximize ROI by targeting only the customers where discounts actually make a causal difference.

---

For the full system architecture report, see [`PROJECT_REPORT.md`](PROJECT_REPORT.md) and [`backend/README.md`](backend/README.md).
