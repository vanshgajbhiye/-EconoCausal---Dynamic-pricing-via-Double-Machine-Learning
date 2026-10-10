import { SalesRecord } from '../models/SalesRecord.js';
import { Product } from '../models/Product.js';
import { DMLSimulationService } from '../services/dmlSimulationService.js';
import { isConnectedToMongo } from '../config/db.js';
import mongoose from 'mongoose';

let memorySales = [];

const isDbConnected = () => isConnectedToMongo && mongoose.connection.readyState === 1;


export const recordSale = async (req, res, next) => {
  try {
    const { productId, price, unitsSold, discountPercentage, isPromotionActive, competitorPrice } = req.body;

    if (!productId || price === undefined || unitsSold === undefined) {
      return res.status(400).json({ success: false, message: 'productId, price, and unitsSold are required' });
    }

    const revenue = price * unitsSold;

    if (isDbConnected()) {
      const record = await SalesRecord.create({
        productId,
        price,
        unitsSold,
        revenue,
        discountPercentage: discountPercentage || 0,
        isPromotionActive: !!isPromotionActive,
        competitorPrice: competitorPrice || price,
      });
      return res.status(201).json({ success: true, data: record });
    }

    const record = {
      _id: `sale_${Date.now()}`,
      productId,
      date: new Date(),
      price: Number(price),
      unitsSold: Number(unitsSold),
      revenue: Number(revenue.toFixed(2)),
      discountPercentage: Number(discountPercentage || 0),
      isPromotionActive: !!isPromotionActive,
      competitorPrice: Number(competitorPrice || price),
      inventoryLevel: 150,
      seasonalityIndex: 1.0,
      dayOfWeek: new Date().getDay(),
      createdAt: new Date(),
    };
    memorySales.unshift(record);
    return res.status(201).json({ success: true, data: record });
  } catch (err) {
    next(err);
  }
};

export const getSalesHistory = async (req, res, next) => {
  try {
    const { productId } = req.query;

    if (isDbConnected()) {
      const filter = productId ? { productId } : {};
      const sales = await SalesRecord.find(filter).sort({ date: -1 }).limit(100);
      return res.status(200).json({ success: true, count: sales.length, data: sales });
    }

    let results = memorySales;
    if (productId) {
      results = memorySales.filter((s) => s.productId.toString() === productId.toString());
    }
    return res.status(200).json({ success: true, count: results.length, data: results });
  } catch (err) {
    next(err);
  }
};

export const seedSampleSalesData = async (req, res, next) => {
  try {
    const { productId, basePrice = 120, days = 45 } = req.body;
    const targetId = productId || 'prod_101';

    const syntheticData = DMLSimulationService.generateSyntheticSales(targetId, Number(basePrice), Number(days));

    if (isDbConnected() && mongoose.Types.ObjectId.isValid(targetId)) {
      await SalesRecord.deleteMany({ productId: targetId });
      await SalesRecord.insertMany(syntheticData);
      return res.status(200).json({
        success: true,
        message: `Successfully seeded ${syntheticData.length} historical sales records in MongoDB`,
        count: syntheticData.length,
      });
    }

    // Memory seed
    memorySales = [...syntheticData, ...memorySales.filter((s) => s.productId !== targetId)];
    return res.status(200).json({
      success: true,
      message: `Successfully seeded ${syntheticData.length} historical sales records in memory`,
      count: syntheticData.length,
      sample: syntheticData.slice(0, 3),
    });
  } catch (err) {
    next(err);
  }
};
