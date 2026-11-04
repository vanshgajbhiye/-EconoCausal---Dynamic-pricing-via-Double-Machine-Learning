/**
 * Analysis Controller
 * ===================
 * Runs DML causal pricing analysis on an uploaded dataset.
 *
 * POST /api/analysis
 * Body: { dataset_id, treatment, outcome, covariates[], method }
 *
 * Response shape matches what frontend AnalysisLab.jsx expects:
 *   { average_treatment_effect, confidence_interval, interpretation, ... }
 */

import { DMLSimulationService } from '../services/dmlSimulationService.js';
import { datasetStore }         from './datasetController.js';
import { PricingExperiment }    from '../models/PricingExperiment.js';
import { isConnectedToMongo }   from '../config/db.js';
import mongoose                 from 'mongoose';

// In-memory experiment store (fallback when MongoDB unavailable)
const memoryExperiments = new Map();

const isDbConnected = () => isConnectedToMongo && mongoose.connection.readyState === 1;

function num(v, fb = 0) {
  const n = parseFloat(v);
  return isNaN(n) ? fb : n;
}

// ──────────────────────────────────────────────────────────────────────
// POST /api/analysis
// ──────────────────────────────────────────────────────────────────────
export const runAnalysis = async (req, res, next) => {
  try {
    const {
      dataset_id,
      treatment  = 'unit_price',
      outcome    = 'qty',
      covariates = ['comp_1', 'comp_2', 'customers', 'month', 'holiday'],
      method     = 'DML',
    } = req.body;

    if (!dataset_id) {
      return res.status(400).json({ success: false, message: 'dataset_id is required' });
    }

    const dataset = datasetStore.get(dataset_id);
    if (!dataset) {
      return res.status(404).json({
        success: false,
        message: `Dataset ${dataset_id} not found. Upload a CSV first via POST /api/datasets`,
      });
    }

    const { rows, mapping } = dataset;

    // Resolve column names from mapping (if user mapped columns in UI)
    const treatmentCol = mapping.treatment || treatment;
    const outcomeCol   = mapping.outcome   || outcome;

    // Extract T (price) and Y (demand) arrays from CSV rows
    const T = rows.map(r => num(r[treatmentCol])).filter(v => v > 0);
    const Y = rows.map(r => num(r[outcomeCol])).filter(v => v >= 0);

    if (T.length < 10) {
      return res.status(400).json({
        success: false,
        message: `Treatment column "${treatmentCol}" has fewer than 10 valid numeric values. Check your mapping.`,
      });
    }

    // Build a synthetic product profile from dataset stats
    const avgPrice = T.reduce((s, v) => s + v, 0) / T.length;
    const avgDemand = Y.reduce((s, v) => s + v, 0) / (Y.length || 1);
    const comp1Col  = mapping.competitorPrice || 'comp_1';
    const avgComp   = rows.map(r => num(r[comp1Col])).reduce((s, v) => s + v, 0) / rows.length;

    // Run DML via existing simulation service
    const dmlResult = DMLSimulationService.estimateCausalElasticity({
      currentPrice:    avgPrice,
      basePrice:       avgPrice,
      costPrice:       avgPrice * 0.45,
      avgDemand,
      competitorPrice: avgComp || avgPrice,
      sampleSize:      T.length,
    });

    // Build response — shape matches AnalysisLab.jsx expectations
    const experimentId = `exp_${Date.now()}`;
    const result = {
      experiment_id:            experimentId,
      dataset_id,
      method,
      treatment_column:         treatmentCol,
      outcome_column:           outcomeCol,
      sample_size:              T.length,

      // Core DML outputs (what AnalysisLab reads)
      average_treatment_effect: dmlResult.causalTreatmentEffect,
      confidence_interval:      [
        dmlResult.confidenceInterval.lower,
        dmlResult.confidenceInterval.upper,
      ],
      price_elasticity:         dmlResult.priceElasticity,
      optimal_price:            dmlResult.optimalPrice,
      current_price:            avgPrice,

      // Chart data
      price_demand_curve:       dmlResult.priceDemandCurve,

      // Human-readable summary
      interpretation: dmlResult.interpretation || (
        `A 1-unit increase in ${treatmentCol} causally changes ${outcomeCol} by ` +
        `${dmlResult.causalTreatmentEffect?.toFixed(4)} units (95% CI: ` +
        `[${dmlResult.confidenceInterval?.lower?.toFixed(4)}, ` +
        `${dmlResult.confidenceInterval?.upper?.toFixed(4)}]). ` +
        `Estimated price elasticity: ${dmlResult.priceElasticity?.toFixed(3)}. ` +
        `Optimal price: $${dmlResult.optimalPrice?.toFixed(2)}.`
      ),
      recommendation: dmlResult.recommendation,
      status: 'complete',
      completedAt: new Date().toISOString(),
    };

    // Persist experiment
    if (isDbConnected()) {
      await PricingExperiment.create({
        productId:             dataset_id,
        productName:           dataset.originalName,
        method,
        causalTreatmentEffect: dmlResult.causalTreatmentEffect,
        priceElasticity:       dmlResult.priceElasticity,
        optimalPrice:          dmlResult.optimalPrice,
        currentPrice:          avgPrice,
        costPrice:             avgPrice * 0.45,
        confidenceInterval:    dmlResult.confidenceInterval,
        priceDemandCurve:      dmlResult.priceDemandCurve,
        recommendation:        dmlResult.recommendation,
        interpretation:        result.interpretation,
        sampleSize:            T.length,
        dataSource:            dataset.originalName,
        status:                'complete',
      });
    } else {
      memoryExperiments.set(experimentId, result);
    }

    return res.status(200).json({ success: true, result });
  } catch (err) {
    next(err);
  }
};

// ──────────────────────────────────────────────────────────────────────
// GET /api/reports/:id
// ──────────────────────────────────────────────────────────────────────
export const getReport = async (req, res, next) => {
  try {
    const { id } = req.params;

    if (isDbConnected()) {
      const exp = await PricingExperiment.findById(id).catch(() => null)
               || await PricingExperiment.findOne({ productId: id });
      if (exp) return res.status(200).json({ success: true, data: exp });
    }

    // Fallback: memory
    const mem = memoryExperiments.get(id);
    if (mem) return res.status(200).json({ success: true, data: mem });

    return res.status(404).json({ success: false, message: `Report ${id} not found` });
  } catch (err) {
    next(err);
  }
};
