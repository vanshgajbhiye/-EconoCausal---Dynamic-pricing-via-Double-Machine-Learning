import mongoose from 'mongoose';

/**
 * Product Schema
 * ==============
 * Represents a unique product in the EconoCausal catalog.
 * Source: ml/data/raw/retail_price.csv → distinct product_id rows.
 */
const ProductSchema = new mongoose.Schema(
  {
    name: { type: String, required: true, trim: true },
    sku:  { type: String, required: true, unique: true, uppercase: true, trim: true },
    category: { type: String, trim: true, default: 'General' },

    // Pricing
    basePrice:        { type: Number, required: true, min: 0 },
    costPrice:        { type: Number, required: true, min: 0 },
    currentPrice:     { type: Number, min: 0 },
    competitorPrice:  { type: Number, min: 0 },
    minPrice:         { type: Number, min: 0 },
    maxPrice:         { type: Number, min: 0 },

    // Inventory & market
    inventory:            { type: Number, default: 100, min: 0 },
    estimatedElasticity:  { type: Number, default: -1.75 },

    // Metadata from CSV
    productScore:          { type: Number },
    productWeightG:        { type: Number },
    productNameLength:     { type: Number },
    productDescriptionLength: { type: Number },
    productPhotosQty:      { type: Number },
    volume:                { type: Number },
  },
  {
    timestamps: true,
    toJSON: { virtuals: true },
  }
);

// Auto-set min/maxPrice if not provided
ProductSchema.pre('save', function (next) {
  if (!this.minPrice) this.minPrice = parseFloat((this.costPrice * 1.05).toFixed(2));
  if (!this.maxPrice) this.maxPrice = parseFloat((this.basePrice * 2.2).toFixed(2));
  if (!this.currentPrice) this.currentPrice = this.basePrice;
  next();
});

export const Product =
  mongoose.models.Product || mongoose.model('Product', ProductSchema);

