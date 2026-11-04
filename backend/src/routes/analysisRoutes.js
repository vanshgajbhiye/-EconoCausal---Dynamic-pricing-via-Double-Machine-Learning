import express from 'express';
import { runAnalysis, getReport } from '../controllers/analysisController.js';

const router = express.Router();

// POST /api/analysis       — run DML on uploaded dataset
router.post('/', runAnalysis);

// GET  /api/reports/:id    — fetch a stored experiment/report
// (mounted separately at /api/reports in server.js)
export { getReport };

export default router;

