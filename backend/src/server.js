import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import dotenv from 'dotenv';
import { connectDB } from './config/db.js';
import productRoutes  from './routes/productRoutes.js';
import salesRoutes    from './routes/salesRoutes.js';
import pricingRoutes  from './routes/pricingRoutes.js';
import datasetRoutes  from './routes/datasetRoutes.js';
import analysisRoutes, { getReport } from './routes/analysisRoutes.js';
import { notFound, errorHandler } from './middleware/errorHandler.js';

// Load environment variables
dotenv.config();

const app  = express();
const PORT = process.env.PORT || 5000;

// Connect to MongoDB
connectDB();

// ── Global Middleware ────────────────────────────────────────────
app.use(helmet());
app.use(cors({ origin: '*', credentials: true }));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

if (process.env.NODE_ENV !== 'test') {
  app.use(morgan('dev'));
}

// ── Health Check ─────────────────────────────────────────────────
app.get('/api/health', (_req, res) => {
  res.status(200).json({
    status:      'online',
    service:     'EconoCausal Backend API',
    version:     '1.1.0',
    timestamp:   new Date().toISOString(),
    environment: process.env.NODE_ENV || 'development',
  });
});

// ── API Routes ───────────────────────────────────────────────────
app.use('/api/products',  productRoutes);   // GET/POST /api/products
app.use('/api/sales',     salesRoutes);     // GET/POST /api/sales
app.use('/api/pricing',   pricingRoutes);   // POST /api/pricing/optimize|simulate
app.use('/api/datasets',  datasetRoutes);   // POST /api/datasets  (CSV upload)
app.use('/api/analysis',  analysisRoutes);  // POST /api/analysis  (run DML)
app.get('/api/reports/:id', getReport);     // GET  /api/reports/:id

// ── Error Handling ───────────────────────────────────────────────
app.use(notFound);
app.use(errorHandler);

// ── Start Server ─────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n${'='.repeat(54)}`);
  console.log(`  EconoCausal Backend  →  http://localhost:${PORT}`);
  console.log(`  Mode : ${process.env.NODE_ENV || 'development'}`);
  console.log(`  Health: http://localhost:${PORT}/api/health`);
  console.log(`  Routes: /api/products | /api/sales | /api/pricing`);
  console.log(`            /api/datasets | /api/analysis | /api/reports`);
  console.log(`${'='.repeat(54)}\n`);
});

export default app;
