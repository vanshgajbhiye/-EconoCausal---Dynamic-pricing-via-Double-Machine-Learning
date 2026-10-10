# ⚡ EconoCausal - Backend API

> **MERN Stack Backend** for *EconoCausal: Dynamic Pricing via Double Machine Learning*.  
> Provides RESTful endpoints for product catalog management, sales history tracking, causal price elasticity estimation, and dynamic pricing simulation.

---

## 🛠️ Tech Stack

- **Runtime**: Node.js (v22+)
- **Framework**: Express.js (ES Modules)
- **Database**: MongoDB with Mongoose ODM (with automatic in-memory fallback for local dev)
- **Security & Utilities**: Helmet, CORS, Morgan, Dotenv

---

## 📁 Project Architecture

```
d:/EconoCausal/backend/
├── src/
│   ├── config/
│   │   └── db.js                 # MongoDB connection manager with offline fallback
│   ├── controllers/
│   │   ├── productController.js  # Product CRUD & seed endpoints
│   │   ├── salesController.js    # Sales logging & synthetic dataset generator
│   │   └── pricingController.js  # Causal DML estimation & price curve simulator
│   ├── middleware/
│   │   └── errorHandler.js       # 404 & centralized error handling
│   ├── models/
│   │   ├── Product.js            # Product catalog & baseline prices
│   │   ├── SalesRecord.js        # Historical demand with confounder covariates
│   │   └── PricingExperiment.js  # Causal DML results, elasticity & optimal prices
│   ├── routes/
│   │   ├── productRoutes.js      # /api/products
│   │   ├── salesRoutes.js        # /api/sales
│   │   └── pricingRoutes.js      # /api/pricing
│   ├── services/
│   │   └── dmlSimulationService.js # Econometric DML & elasticity calculations
│   └── server.js                 # Express application entrypoint
├── .env.example
├── .env
├── package.json
└── README.md
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
cd backend
npm install
```

### 2. Configure Environment
Copy `.env.example` to `.env` (already created for local dev):
```env
PORT=5000
NODE_ENV=development
MONGO_URI=mongodb://127.0.0.1:27017/econocausal
```

### 3. Run in Development Mode
```bash
npm run dev
```

---

## 📡 API Reference

### 🩺 System
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status & timestamp |

### 📦 Products
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/products` | Retrieve all products |
| `GET` | `/api/products/:id` | Get single product by ID |
| `POST` | `/api/products` | Create a new product |
| `PATCH` | `/api/products/:id/price` | Update current product price |
| `POST` | `/api/products/seed` | Seed initial demo catalog |

### 📊 Sales & Historical Data
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/sales?productId=...` | Retrieve sales history |
| `POST` | `/api/sales` | Record an individual sales transaction |
| `POST` | `/api/sales/seed` | Generate synthetic sales dataset for DML training |

### 📈 Causal Inference & Dynamic Pricing
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/pricing/optimize` | Runs DML estimation on historical sales to compute causal elasticity, Lerner index optimal price, and confidence intervals |
| `POST` | `/api/pricing/simulate` | Simulates demand, revenue, and profit curves across custom price ranges |
| `GET` | `/api/pricing/experiments` | Retrieves past causal pricing optimization runs |

---

## 🔮 Causal DML Integration Note

The [DMLSimulationService](file:///d:/EconoCausal/backend/src/services/dmlSimulationService.js) currently implements Neyman-orthogonal residualization and econometric Lerner markup formulas directly in JavaScript. When you are ready to build the Python ML service with **EconML / DoubleML**, this service can seamlessly delegate to a Python FastAPI endpoint.
