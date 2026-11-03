# EconoCausal — Database Layer

This folder owns everything database-related, **independent of the backend server**.

```
database/
├── config/
│   └── connection.js         ← shared Mongoose connect/disconnect (imported by backend)
├── schemas/
│   ├── Product.js            ← Product schema & model
│   ├── SalesRecord.js        ← Sales record schema & model (676 rows from ML data)
│   └── PricingExperiment.js  ← DML experiment results schema
└── seeds/
    └── seedFromCSV.cjs       ← Populates MongoDB from ml/data/raw/retail_price.csv
```

---

## Connection

- **URI:** `mongodb://127.0.0.1:27017/econocausal`  
- **Configured in:** `backend/.env` → `MONGO_URI`  
- **Connection logic:** `database/config/connection.js` (imported by `backend/src/server.js`)

---

## Seed the Database

Make sure MongoDB is running, then from the **repo root**:

```powershell
node database/seeds/seedFromCSV.cjs
```

Output:
```
Products     : 52   (unique product_id values from CSV)
SalesRecords : 676  (one record per product per month, 2017–2018)
```

---

## Schema → CSV Column Mapping

| Schema Field | CSV Column | Role in DML |
|---|---|---|
| `price` | `unit_price` | Treatment **T** |
| `unitsSold` | `qty` | Outcome **Y** |
| `competitorPrice` | `comp_1` | Confounder **X** |
| `comp2`, `comp3` | `comp_2`, `comp_3` | Confounders **X** |
| `customers` | `customers` | Confounder **X** |
| `seasonalityIndex` | derived from `weekday`/`weekend` | Confounder **X** |
| `isPromotionActive` | `holiday > 0` | Confounder **X** |
| `lagPrice` | `lag_price` | Instrument candidate |

---

## Re-seed Anytime

Data lost or changed? Just re-run the seed script — it wipes and re-inserts cleanly.

