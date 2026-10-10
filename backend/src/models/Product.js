import mongoose from 'mongoose';

const productSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: [true, 'Product name is required'],
      trim: true,
    },
    sku: {
      type: String,
      required: [true, 'SKU is required'],
      unique: true,
      trim: true,
      uppercase: true,
    },
    category: {
      type: String,
      default: 'General',
    },
    basePrice: {
      type: Number,
      required: [true, 'Base price is required'],
      min: [0, 'Base price cannot be negative'],
    },
    costPrice: {
      type: Number,
      required: [true, 'Cost price (Marginal Cost) is required'],
      min: [0, 'Cost price cannot be negative'],
    },
    currentPrice: {
      type: Number,
      required: true,
      min: 0,
    },
    competitorPrice: {
      type: Number,
      default: 0,
    },
    minPrice: {
      type: Number,
      default: function () {
        return this.costPrice * 1.05; // At least 5% above cost price
      },
    },
    maxPrice: {
      type: Number,
      default: function () {
        return this.basePrice * 2.0;
      },
    },
    inventory: {
      type: Number,
      default: 100,
      min: 0,
    },
    estimatedElasticity: {
      type: Number,
      default: -1.75, // Default moderately elastic demand
    },
  },
  {
    timestamps: true,
  }
);

export const Product = mongoose.model('Product', productSchema);
