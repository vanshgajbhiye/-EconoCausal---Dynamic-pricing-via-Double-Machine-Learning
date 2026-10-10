import express from 'express';
import {
  recordSale,
  getSalesHistory,
  seedSampleSalesData,
} from '../controllers/salesController.js';

const router = express.Router();

router.route('/')
  .get(getSalesHistory)
  .post(recordSale);

router.post('/seed', seedSampleSalesData);

export default router;
