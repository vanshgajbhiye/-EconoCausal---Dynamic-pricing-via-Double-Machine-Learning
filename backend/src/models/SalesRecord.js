import mongoose from 'mongoose';

const salesRecordSchema = new mongoose.Schema(
  {
    productId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Product',
      required: true,
    },
    date: {
      type: Date,
      default: Date.now,
      index: true,
    },
    price: {
      type: Number,
      required: true,
    },
    unitsSold: {
      type: Number,
      required: true,
      min: 0,
    },
    revenue: {
      type: Number,
      required: true,
    },
    discountPercentage: {
      type: Number,
      default: 0,
    },
    isPromotionActive: {
      type: Boolean,
      default: false,
    },
    competitorPrice: {
      type: Number,
      default: 0,
    },
    inventoryLevel: {
      type: Number,
      default: 0,
    },
    seasonalityIndex: {
      type: Number,
      default: 1.0, // 1.0 = normal, >1 = peak season, <1 = low season
    },
    dayOfWeek: {
      type: Number, // 0 = Sunday, 6 = Saturday
      default: () => new Date().getDay(),
    },
  },
  {
    timestamps: true,
  }
);

export const SalesRecord = mongoose.model('SalesRecord', salesRecordSchema);
