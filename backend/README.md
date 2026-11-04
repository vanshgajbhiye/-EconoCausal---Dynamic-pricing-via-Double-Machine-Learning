# EconoCausal — Backend API

> **Node.js / Express REST API** powering the EconoCausal Dynamic Pricing platform.  
> Uses Double Machine Learning (DML) to separate causal price effects from confounders.

---

## Quick Start

```powershell
# 1. Install dependencies
cd backend
npm install

# 2. Configure environment (already set for local dev)
# backend/.env  →  MONGO_URI=mongodb://127.0.0.1:27017/econocausal

# 3. Seed the database from real ML data
node ..\database\seeds\seedFromCSV.cjs

# 4. Start the server
npm run dev          # nodemon (auto-restart on file change)
# or
node src/server.js   # plain node
```

Server starts at **http://localhost:5000**

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | HTTP port |
| `NODE_ENV` | `development` | `development` / `production` |
| `MONGO_URI` | `mongodb://127.0.0.1:27017/econocausal` | MongoDB connection string |
| `JWT_SECRET` | `econocausal_dev_secret_key_2026` | JWT signing key (future auth) |
| `CORS_ORIGIN` | `*` | Allowed CORS origins |

Copy `.env.example` → `.env` to get started.

---

## Folder Structure

```
backend/
├── src/
│   ├── server.js                   ← Express app entrypoint, mounts all routes
│   ├── config/
│   │   └── db.js                   ← MongoDB connect / disconnect
│   ├── middleware/
│   │   ├── errorHandler.js         ← Global 404 + error handler
│   │   └── upload.js               ← Multer CSV upload (memory storage, 10MB max)
│   ├── models/                     ← Mongoose schemas
│   │   ├── Product.js
│   │   ├── SalesRecord.js
│   │   └── PricingExperiment.js
│   ├── controllers/                ← Route handler logic
│   │   ├── productController.js
│   │   ├── salesController.js
│   │   ├── pricingController.js
│   │   ├── datasetController.js    ← CSV upload + in-memory store
│   │   └── analysisController.js  ← DML engine + experiment persistence
│   ├── routes/                     ← Express routers
│   │   ├── productRoutes.js
│   │   ├── salesRoutes.js
│   │   ├── pricingRoutes.js
│   │   ├── datasetRoutes.js
│   │   └── analysisRoutes.js
│   └── services/                   ← Business logic (no HTTP concerns)
│       ├── dmlSimulationService.js ← Core DML engine + Lerner pricing
│       ├── customerITEService.js   ← Individual Treatment Effect estimation
│       ├── budgetOptimizerService.js ← Greedy knapsack discount allocator
│       └── graduationService.js   ← Customer lifecycle / discount tapering
└── test/
    └── apiTest.cjs                 ← Integration test (all 7 endpoints)
```

---

## API Reference

### Health

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Server heartbeat, returns `{ status: "online" }` |

---

### Products  `/api/products`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/products` | List all products (52 from ML dataset) |
| `POST` | `/api/products` | Create a product |
| `GET` | `/api/products/:id` | Get product by ID |
| `PATCH` | `/api/products/:id/price` | Update current price `{ newPrice }` |
| `POST` | `/api/products/seed` | Re-seed products from in-memory defaults |

**GET /api/products response:**
```json
{
  "success": true,
  "count": 52,
  "data": [{ "_id": "...", "name": "BED1", "category": "bed bath table", "basePrice": 45.95 }]
}
```

---

### Sales  `/api/sales`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/sales` | Sales history (query: `?productId=...`) |
| `POST` | `/api/sales` | Record a sale `{ productId, price, unitsSold }` |
| `POST` | `/api/sales/seed` | Generate synthetic sales for a product |

---

### DML Pricing  `/api/pricing`

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/api/pricing/optimize` | `{ productId? }` | Run DML on a product's historical sales → returns elasticity + optimal price |
| `POST` | `/api/pricing/simulate` | `{ productId, pricePoints[] }` | Simulate revenue/demand at custom price points |
| `GET` | `/api/pricing/experiments` | — | List all stored DML experiments |

**POST /api/pricing/optimize response:**
```json
{
  "success": true,
  "data": {
    "causalTreatmentEffect": -2.14,
    "priceElasticity": -1.65,
    "confidenceInterval": { "lower": -1.882, "upper": -1.418 },
    "recommendedOptimalPrice": 66.57,
    "projectedProfitLiftPercentage": 12.4,
    "priceDemandCurve": [{ "price": 40, "predictedDemand": 120, "predictedRevenue": 4800 }]
  }
}
```

---

### Dataset Upload  `/api/datasets`

Upload a CSV from the frontend's **DatasetStudio** page.

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/api/datasets` | `multipart/form-data` — `file` (CSV) + `mapping` (JSON string) | Upload CSV, returns `dataset_id` |
| `GET` | `/api/datasets/:id` | — | Get dataset metadata + 10-row preview |

**POST /api/datasets request (FormData):**
```
file       = <CSV file>
mapping    = {"treatment":"unit_price","outcome":"qty","competitorPrice":"comp_1"}
```

**Response:**
```json
{
  "success": true,
  "dataset_id": "ds_1793809493536_7bbts",
  "headers": ["unit_price", "qty", "comp_1", ...],
  "rowCount": 676,
  "mapping": { "treatment": "unit_price", "outcome": "qty" }
}
```

---

### DML Analysis  `/api/analysis`

Run Double Machine Learning on an uploaded dataset.

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/api/analysis` | `{ dataset_id, treatment, outcome, covariates[], method }` | Run DML → returns ATE, CI, optimal price, demand curve |

**Request:**
```json
{
  "dataset_id": "ds_1793809493536_7bbts",
  "treatment": "unit_price",
  "outcome": "qty",
  "covariates": ["comp_1", "comp_2", "customers", "month", "holiday"],
  "method": "DML"
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "experiment_id": "exp_1793809493600",
    "average_treatment_effect": -3.158,
    "confidence_interval": [-1.882, -1.418],
    "price_elasticity": -1.65,
    "optimal_price": 66.57,
    "interpretation": "A 1% change in unit_price causally changes qty by -1.650%...",
    "price_demand_curve": [...]
  }
}
```

---

### Reports  `/api/reports`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/reports/:id` | Fetch a stored experiment by ID |

---

## In-Memory Fallback Mode

If MongoDB is **not running**, the backend still works:
- Products / sales use hardcoded in-memory seed data
- Dataset uploads stored in a `Map` (lost on server restart)
- Analysis results stored in memory

Start MongoDB at `localhost:27017` for persistent storage.

---

## Database

**Seed MongoDB from ML data (676 real rows):**
```powershell
node ..\database\seeds\seedFromCSV.cjs
```

**Verify what's in the DB:**
```powershell
node ..\database\seeds\verifyDB.cjs
```

Source: `ml/data/raw/retail_price.csv` — Kaggle Retail Price Optimization dataset.

---

## Run Tests

Start the server first, then in another terminal:
```powershell
node test/apiTest.cjs
```

Expected output — all 7 green:
```
✅  GET  /api/health
✅  GET  /api/products      → 52 products
✅  GET  /api/sales         → 100 records
✅  POST /api/pricing/optimize
✅  POST /api/datasets
✅  GET  /api/datasets/:id
✅  POST /api/analysis      → ATE: -3.158, P*: $66.57
```

---

## Services — What They Do

| Service | Purpose |
|---|---|
| `dmlSimulationService.js` | Core DML engine: residualizes price/demand, computes elasticity via Lerner rule, generates price-demand curve |
| `customerITEService.js` | Estimates Individual Treatment Effect per customer — classifies as Persuadable / Sure Thing / Lost Cause / Sleeping Dog |
| `budgetOptimizerService.js` | Greedy knapsack: allocates discount budget optimally across Persuadable customers |
| `graduationService.js` | Tracks ITE decay over repeated purchases, recommends discount tapering schedule |
