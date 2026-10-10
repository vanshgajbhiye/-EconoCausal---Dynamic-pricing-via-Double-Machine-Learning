import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import dotenv from 'dotenv';
import { connectDB } from './config/db.js';
import productRoutes from './routes/productRoutes.js';
import salesRoutes from './routes/salesRoutes.js';
import pricingRoutes from './routes/pricingRoutes.js';
import { notFound, errorHandler } from './middleware/errorHandler.js';

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Connect to MongoDB
connectDB();

// Global Middlewares
app.use(helmet());
app.use(
  cors({
    origin: '*',
    credentials: true,
  })
);
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

if (process.env.NODE_ENV !== 'test') {
  app.use(morgan('dev'));
}

// System Health Check
app.get('/api/health', (req, res) => {
  res.status(200).json({
    status: 'online',
    service: 'EconoCausal Backend API',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV || 'development',
  });
});

// API Routes
app.use('/api/products', productRoutes);
app.use('/api/sales', salesRoutes);
app.use('/api/pricing', pricingRoutes);

// Error Handling
app.use(notFound);
app.use(errorHandler);

// Start Server
app.listen(PORT, () => {
  console.log(`\n======================================================`);
  console.log(`🚀 EconoCausal Backend Server running on http://localhost:${PORT}`);
  console.log(`📈 Mode: ${process.env.NODE_ENV || 'development'}`);
  console.log(`🩺 Health check: http://localhost:${PORT}/api/health`);
  console.log(`======================================================\n`);
});

export default app;
