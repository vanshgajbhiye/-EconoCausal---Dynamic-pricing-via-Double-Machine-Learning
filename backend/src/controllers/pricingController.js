import { Product } from '../models/Product.js';
import { SalesRecord } from '../models/SalesRecord.js';
import { PricingExperiment } from '../models/PricingExperiment.js';
import { DMLSimulationService } from '../services/dmlSimulationService.js';
import { isConnectedToMongo } from '../config/db.js';
import mongoose from 'mongoose';

let memoryExperiments = [];

const isDbConnected = () => isConnectedToMongo && mongoose.connection.readyState === 1;


/**
 * Execute DML Causal Price Elasticity estimation
 */
export const runDMLPricingAnalysis = async (req, res, next) => {
  try {
    const { productId, title = 'Causal DML Price Optimization' } = req.body;

    let product = null;
    let historicalData = [];

    if (isDbConnected()) {
      if (productId && mongoose.Types.ObjectId.isValid(productId)) {
        product = await Product.findById(productId);
        historicalData = await SalesRecord.find({ productId }).lean();
      }
    }

    // Default parameters if product not found in DB
    const basePrice = product ? product.basePrice : 120;
    const costPrice = product ? product.costPrice : 55;
    const currentPrice = product ? product.currentPrice : 115;
    const minPrice = product ? product.minPrice : costPrice * 1.08;
    const maxPrice = product ? product.maxPrice : basePrice * 1.8;

    // If no real historical data yet, generate synthetic historical sample for DML fitting
    if (!historicalData || historicalData.length < 10) {
      historicalData = DMLSimulationService.generateSyntheticSales(productId || 'mock_prod', basePrice, 50);
    }

    // Run Causal DML Estimation
    const analysis = DMLSimulationService.estimateCausalElasticity({
      historicalData,
      basePrice,
      costPrice,
      currentPrice,
      minPrice,
      maxPrice,
    });

    const experimentPayload = {
      title,
      productId: productId || '650000000000000000000001',
      status: 'completed',
      methodology: analysis.methodology,
      sampleSize: analysis.sampleSize,
      causalTreatmentEffect: analysis.causalTreatmentEffect,
      priceElasticity: analysis.priceElasticity,
      confidenceInterval: analysis.confidenceInterval,
      pVal: analysis.pValue,
      currentPrice: analysis.currentPrice,
      recommendedOptimalPrice: analysis.recommendedOptimalPrice,
      projectedDemandAtOptimal: analysis.projectedDemandAtOptimal,
      projectedRevenueAtOptimal: analysis.projectedRevenueAtOptimal,
      projectedProfitLiftPercentage: analysis.projectedProfitLiftPercentage,
      controlledConfounders: analysis.controlledConfounders,
      priceDemandCurve: analysis.priceDemandCurve,
    };

    if (isDbConnected() && productId && mongoose.Types.ObjectId.isValid(productId)) {
      const savedExperiment = await PricingExperiment.create(experimentPayload);
      return res.status(200).json({
        success: true,
        message: 'Causal DML analysis completed and saved',
        data: savedExperiment,
      });
    }

    const savedExperiment = {
      _id: `exp_${Date.now()}`,
      ...experimentPayload,
      createdAt: new Date(),
    };
    memoryExperiments.unshift(savedExperiment);

    return res.status(200).json({
      success: true,
      message: 'Causal DML analysis calculated successfully',
      data: savedExperiment,
    });
  } catch (err) {
    next(err);
  }
};

/**
 * Simulate dynamic pricing scenarios for arbitrary custom inputs
 */
export const simulatePricingScenarios = async (req, res, next) => {
  try {
    const {
      basePrice = 100,
      costPrice = 40,
      currentPrice = 95,
      elasticity = -1.75,
      minPrice = 50,
      maxPrice = 180,
      steps = 15,
    } = req.body;

    const curve = DMLSimulationService.generatePriceDemandCurve({
      basePrice: Number(basePrice),
      costPrice: Number(costPrice),
      currentPrice: Number(currentPrice),
      elasticity: Number(elasticity),
      minPrice: Number(minPrice),
      maxPrice: Number(maxPrice),
      steps: Number(steps),
    });

    // Find price that maximizes profit
    const optimalScenario = curve.reduce(
      (max, pt) => (pt.predictedProfit > max.predictedProfit ? pt : max),
      curve[0]
    );

    return res.status(200).json({
      success: true,
      data: {
        parameters: { basePrice, costPrice, currentPrice, elasticity },
        optimalPricePoint: optimalScenario,
        simulationCurve: curve,
      },
    });
  } catch (err) {
    next(err);
  }
};

/**
 * Get all past causal pricing experiments
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
