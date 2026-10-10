import express from 'express';
import {
  runDMLPricingAnalysis,
  simulatePricingScenarios,
  getExperiments,
} from '../controllers/pricingController.js';

const router = express.Router();

router.post('/optimize', runDMLPricingAnalysis);
router.post('/simulate', simulatePricingScenarios);
router.get('/experiments', getExperiments);

export default router;
