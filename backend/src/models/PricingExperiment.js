import mongoose from 'mongoose';

const pricingExperimentSchema = new mongoose.Schema(
  {
    title: {
      type: String,
      required: true,
      trim: true,
    },
    productId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Product',
      required: true,
    },
    status: {
      type: String,
      enum: ['draft', 'running', 'completed', 'applied'],
      default: 'completed',
    },
    methodology: {
      type: String,
      default: 'Double Machine Learning (DML) - Orthogonal Residualization',
    },
    sampleSize: {
      type: Number,
      default: 0,
    },
    causalTreatmentEffect: {
      type: Number, // Elasticity or beta coefficient dQ / dP
      required: true,
    },
    priceElasticity: {
      type: Number,
      required: true,
    },
    confidenceInterval: {
      lower: { type: Number, required: true },
      upper: { type: Number, required: true },
    },
    pVal: {
      type: Number,
      default: 0.001,
    },
    currentPrice: {
      type: Number,
      required: true,
    },
    recommendedOptimalPrice: {
      type: Number,
      required: true,
    },
    projectedDemandAtOptimal: {
      type: Number,
      required: true,
    },
    projectedRevenueAtOptimal: {
      type: Number,
      required: true,
    },
    projectedProfitLiftPercentage: {
      type: Number,
      required: true,
    },
    controlledConfounders: [
      {
        type: String,
      },
    ],
    priceDemandCurve: [
      {
        price: Number,
        predictedDemand: Number,
        predictedRevenue: Number,
        predictedProfit: Number,
      },
    ],
  },
  {
    timestamps: true,
  }
);

export const PricingExperiment = mongoose.model(
  'PricingExperiment',
  pricingExperimentSchema
);
