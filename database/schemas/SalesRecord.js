import mongoose from 'mongoose';

/**
 * SalesRecord Schema
 * ==================
 * One record per product per month.
 * Source: ml/data/raw/retail_price.csv (676 rows).
 *
 * Columns mapped:
 *   product_id     → productId (ref to Product._id)
 *   month_year     → date
 *   unit_price     → price          (treatment T in DML)
 *   qty            → unitsSold      (outcome Y in DML)
 *   total_price    → revenue
 *   freight_price  → freightPrice
 *   comp_1/2/3     → competitor prices (confounders X)
 *   customers      → market size confounder
 *   holiday        → promotion flag
 *   weekday/weekend→ seasonality
 *   lag_price      → lagged price (instrument candidate)
 */
const SalesRecordSchema = new mongoose.Schema(
  {
    productId: {
      type: mongoose.Schema.Types.Mixed, // ObjectId or string (for in-memory mode)
      required: true,
      index: true,
    },
    date: { type: Date, required: true, index: true },

    // Core financials
    price:       { type: Number, required: true }, // unit_price — treatment T
    unitsSold:   { type: Number, required: true }, // qty — outcome Y
    revenue:     { type: Number },                 // total_price
    freightPrice:{ type: Number, default: 0 },

    // Promotion
    discountPercentage:  { type: Number, default: 0 },
    isPromotionActive:   { type: Boolean, default: false }, // holiday > 0

    // Competitor prices — confounders X
    competitorPrice: { type: Number }, // comp_1
    comp2:           { type: Number }, // comp_2
    comp3:           { type: Number }, // comp_3
    ps1:             { type: Number }, // product score competitor 1
    ps2:             { type: Number },
    ps3:             { type: Number },
    fp1:             { type: Number }, // freight price competitor 1
    fp2:             { type: Number },
    fp3:             { type: Number },

    // Market & seasonality — confounders X
    customers:        { type: Number }, // market size
    weekday:          { type: Number },
    weekend:          { type: Number },
    holiday:          { type: Number },
    month:            { type: Number },
    year:             { type: Number },
    volume:           { type: Number }, // market volume
    seasonalityIndex: { type: Number, default: 1.0 },
    inventoryLevel:   { type: Number },

    // Product metadata
    productScore: { type: Number },

    // Lag price — potential instrument for DML
    lagPrice: { type: Number },
  },
  {
    timestamps: true,
  }
);

SalesRecordSchema.index({ productId: 1, date: -1 });

export const SalesRecord =
  mongoose.models.SalesRecord || mongoose.model('SalesRecord', SalesRecordSchema);
