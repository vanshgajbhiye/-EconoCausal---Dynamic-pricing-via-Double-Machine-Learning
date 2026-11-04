import { Product } from '../models/Product.js';
import { SalesRecord } from '../models/SalesRecord.js';
import { PricingExperiment } from '../models/PricingExperiment.js';
import { DMLSimulationService } from '../services/dmlSimulationService.js';
import { isConnectedToMongo } from '../config/db.js';
import mongoose from 'mongoose';

let memoryExperiments = [];

const isDbConnected = () => isConnectedToMongo && mongoose.connection.readyState === 1;

/**
 * POST /api/pricing/optimize
 * Run DML on a product's historical sales data.
 * Also called by the ML Python service for causal elasticity.
 *
 * Body: { productId?, title? }
 * Returns: full DML result with elasticity, optimal price, demand curve
 */
export const runDMLPricingAnalysis = async (req, res, next) => {
  try {
    const { productId, title = 'Causal DML Price Optimization' } = req.body;

    let product       = null;
    let historicalData = [];
    let resolvedProductId; // must be a real ObjectId for PricingExperiment

    if (isDbConnected()) {
      if (productId && mongoose.Types.ObjectId.isValid(productId)) {
        product           = await Product.findById(productId);
        historicalData    = await SalesRecord.find({ productId }).lean();
        resolvedProductId = product?._id;
      }

      // If no specific product given, pick the first one from DB
      if (!resolvedProductId) {
        const firstProduct = await Product.findOne({}).lean();
        resolvedProductId  = firstProduct?._id;
        if (!product && firstProduct) {
          product = firstProduct;
          historicalData = await SalesRecord.find({ productId: firstProduct._id }).lean();
        }
      }
    }

    // Pricing parameters — from DB product or sensible defaults
    const basePrice    = product?.basePrice    ?? 120;
    const costPrice    = product?.costPrice    ?? 55;
    const currentPrice = product?.currentPrice ?? 115;
    const minPrice     = product?.minPrice     ?? costPrice * 1.08;
    const maxPrice     = product?.maxPrice     ?? basePrice * 1.8;

    // Need ≥10 real rows for meaningful regression; otherwise use synthetic
    if (!historicalData || historicalData.length < 10) {
      historicalData = DMLSimulationService.generateSyntheticSales(
        productId || 'mock_prod', basePrice, 50
      );
    }

    // Run DML
    const analysis = DMLSimulationService.estimateCausalElasticity({
      historicalData,
      basePrice,
      costPrice,
      currentPrice,
      minPrice,
      maxPrice,
    });

    // Build experiment payload — matches PricingExperiment schema exactly
    const experimentPayload = {
      title,
      productId:                    resolvedProductId,
      status:                       'completed',
      methodology:                  analysis.methodology,
      sampleSize:                   analysis.sampleSize,
      causalTreatmentEffect:        analysis.causalTreatmentEffect,
      priceElasticity:              analysis.priceElasticity,
      confidenceInterval:           analysis.confidenceInterval,
      pVal:                         analysis.pValue,
      currentPrice:                 analysis.currentPrice,
      recommendedOptimalPrice:      analysis.recommendedOptimalPrice,
      projectedDemandAtOptimal:     analysis.projectedDemandAtOptimal,
      projectedRevenueAtOptimal:    analysis.projectedRevenueAtOptimal,
      projectedProfitLiftPercentage: analysis.projectedProfitLiftPercentage,
      controlledConfounders:        analysis.controlledConfounders,
      priceDemandCurve:             analysis.priceDemandCurve,
    };

    // Persist to MongoDB if connected and we have a valid ObjectId
    if (isDbConnected() && resolvedProductId) {
      const saved = await PricingExperiment.create(experimentPayload);
      return res.status(200).json({
        success: true,
        message: 'DML analysis completed and saved to MongoDB',
        data: saved,
      });
    }

    // In-memory fallback
    const saved = { _id: `exp_${Date.now()}`, ...experimentPayload, createdAt: new Date() };
    memoryExperiments.unshift(saved);

    return res.status(200).json({
      success: true,
      message: 'DML analysis completed (in-memory mode)',
      data: saved,
    });
  } catch (err) {
    next(err);
  }
};

/**
 * POST /api/pricing/simulate
 * Simulate revenue/demand across custom price points.
 * Used by the frontend BudgetPlanner and by the ML service.
 *
 * Body: { basePrice, costPrice, currentPrice, elasticity, minPrice, maxPrice, steps }
 */
export const simulatePricingScenarios = async (req, res, next) => {
  try {
    const {
      basePrice    = 100,
      costPrice    = 40,
      currentPrice = 95,
      elasticity   = -1.75,
      minPrice     = 50,
      maxPrice     = 180,
      steps        = 15,
    } = req.body;

    const curve = DMLSimulationService.generatePriceDemandCurve({
      basePrice:    Number(basePrice),
      costPrice:    Number(costPrice),
      currentPrice: Number(currentPrice),
      elasticity:   Number(elasticity),
      minPrice:     Number(minPrice),
      maxPrice:     Number(maxPrice),
      steps:        Number(steps),
    });

    // Best profit point on the curve
    const optimalScenario = curve.reduce(
      (max, pt) => (pt.predictedProfit > max.predictedProfit ? pt : max),
      curve[0]
    );

    return res.status(200).json({
      success: true,
      data: {
        parameters:      { basePrice, costPrice, currentPrice, elasticity },
        optimalPricePoint: optimalScenario,
        simulationCurve: curve,
      },
    });
  } catch (err) {
    next(err);
  }
};

/**
 * GET /api/pricing/experiments
 * List all stored DML experiments.
 */
export const getExperiments = async (req, res, next) => {
  try {
    if (isDbConnected()) {
      const experiments = await PricingExperiment.find().sort({ createdAt: -1 });
      return res.status(200).json({ success: true, count: experiments.length, data: experiments });
    }
    return res.status(200).json({ success: true, count: memoryExperiments.length, data: memoryExperiments });
  } catch (err) {
    next(err);
  }
};
