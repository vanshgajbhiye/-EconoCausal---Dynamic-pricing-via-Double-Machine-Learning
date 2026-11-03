import mongoose from 'mongoose';

/**
 * PricingExperiment Schema
 * ========================
 * Stores the result of each DML causal pricing analysis run.
 * Written by: POST /api/pricing/optimize
 * Read by:    GET  /api/pricing/experiments
 *             GET  /reports/:id (future)
 */
const PricingExperimentSchema = new mongoose.Schema(
  {
    productId: { type: mongoose.Schema.Types.Mixed, index: true },
    productName: { type: String },
    method: {
      type: String,
      enum: ['DML', 'LinearDML', 'CausalForest', 'IV'],
      default: 'DML',
    },

    // DML outputs
    causalTreatmentEffect: { type: Number }, // θ — dQ/dP
    priceElasticity:       { type: Number }, // ε
    optimalPrice:          { type: Number }, // P* via Lerner rule
    currentPrice:          { type: Number },
    basePrice:             { type: Number },
    costPrice:             { type: Number },
    marginalCost:          { type: Number },

    // Statistical confidence
    confidenceInterval: {
      lower: { type: Number },
      upper: { type: Number },
    },
    rSquared:  { type: Number },
    pValue:    { type: Number },
    sampleSize: { type: Number },

    // Demand curve points for chart rendering
    priceDemandCurve: [
      {
        price:    { type: Number },
        demand:   { type: Number },
        revenue:  { type: Number },
        profit:   { type: Number },
      },
    ],

    // Business impact
    revenueImpact:    { type: Number },
    profitImpact:     { type: Number },
    recommendation:   { type: String },
    interpretation:   { type: String },

    // Experiment metadata
    dataSource:   { type: String, default: 'ml/data/raw/retail_price.csv' },
    status: {
      type: String,
      enum: ['running', 'complete', 'failed'],
      default: 'complete',
    },
  },
  {
    timestamps: true,
    toJSON: { virtuals: true },
  }
);

export const PricingExperiment =
  mongoose.models.PricingExperiment ||
  mongoose.model('PricingExperiment', PricingExperimentSchema);

