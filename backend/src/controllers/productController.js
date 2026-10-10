import { Product } from '../models/Product.js';
import { isConnectedToMongo } from '../config/db.js';
import mongoose from 'mongoose';

// In-memory fallback if MongoDB is not connected
let memoryProducts = [
  {
    _id: 'prod_101',
    name: 'Ergonomic Mechanical Keyboard',
    sku: 'TECH-KB-001',
    category: 'Electronics',
    basePrice: 120,
    costPrice: 55,
    currentPrice: 115,
    competitorPrice: 119,
    minPrice: 65,
    maxPrice: 180,
    inventory: 340,
    estimatedElasticity: -1.82,
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    _id: 'prod_102',
    name: 'Wireless Noise-Canceling Headphones',
    sku: 'AUDIO-HP-002',
    category: 'Electronics',
    basePrice: 199,
    costPrice: 85,
    currentPrice: 189,
    competitorPrice: 195,
    minPrice: 100,
    maxPrice: 280,
    inventory: 215,
    estimatedElasticity: -2.15,
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    _id: 'prod_103',
    name: 'Cold-Brew Coffee Blend (1kg)',
    sku: 'BEV-CB-003',
    category: 'Beverages',
    basePrice: 28,
    costPrice: 12,
    currentPrice: 26,
    competitorPrice: 29,
    minPrice: 15,
    maxPrice: 42,
    inventory: 500,
    estimatedElasticity: -1.35,
    createdAt: new Date(),
    updatedAt: new Date(),
  },
];

const isDbConnected = () => isConnectedToMongo && mongoose.connection.readyState === 1;


export const getProducts = async (req, res, next) => {
  try {
    if (isDbConnected()) {
      const products = await Product.find().sort({ createdAt: -1 });
      return res.status(200).json({ success: true, count: products.length, data: products });
    }
    return res.status(200).json({ success: true, count: memoryProducts.length, data: memoryProducts });
  } catch (err) {
    next(err);
  }
};

export const getProductById = async (req, res, next) => {
  try {
    const { id } = req.params;
    if (isDbConnected()) {
      const product = await Product.findById(id);
      if (!product) {
        return res.status(404).json({ success: false, message: 'Product not found' });
      }
      return res.status(200).json({ success: true, data: product });
    }

    const product = memoryProducts.find((p) => p._id.toString() === id);
    if (!product) {
      return res.status(404).json({ success: false, message: 'Product not found' });
    }
    return res.status(200).json({ success: true, data: product });
  } catch (err) {
    next(err);
  }
};

export const createProduct = async (req, res, next) => {
  try {
    const { name, sku, category, basePrice, costPrice, currentPrice, competitorPrice, inventory } = req.body;

    if (!name || !sku || basePrice === undefined || costPrice === undefined) {
      return res.status(400).json({
        success: false,
        message: 'name, sku, basePrice, and costPrice are required',
      });
    }

    if (isDbConnected()) {
      const product = await Product.create({
        name,
        sku,
        category,
        basePrice,
        costPrice,
        currentPrice: currentPrice || basePrice,
        competitorPrice: competitorPrice || basePrice,
        inventory: inventory || 100,
      });
      return res.status(201).json({ success: true, data: product });
    }

    const newProd = {
      _id: `prod_${Date.now()}`,
      name,
      sku,
      category: category || 'General',
      basePrice: Number(basePrice),
      costPrice: Number(costPrice),
      currentPrice: Number(currentPrice || basePrice),
      competitorPrice: Number(competitorPrice || basePrice),
      minPrice: Number(costPrice) * 1.05,
      maxPrice: Number(basePrice) * 2,
      inventory: Number(inventory || 100),
      estimatedElasticity: -1.75,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    memoryProducts.unshift(newProd);
    return res.status(201).json({ success: true, data: newProd });
  } catch (err) {
    next(err);
  }
};

export const updateProductPrice = async (req, res, next) => {
  try {
    const { id } = req.params;
    const { newPrice } = req.body;

    if (newPrice === undefined || newPrice <= 0) {
      return res.status(400).json({ success: false, message: 'Valid newPrice is required' });
    }

    if (isDbConnected()) {
      const product = await Product.findByIdAndUpdate(
        id,
        { currentPrice: newPrice },
        { new: true, runValidators: true }
      );
      if (!product) {
        return res.status(404).json({ success: false, message: 'Product not found' });
      }
      return res.status(200).json({ success: true, message: 'Price updated successfully', data: product });
    }

    const product = memoryProducts.find((p) => p._id.toString() === id);
    if (!product) {
      return res.status(404).json({ success: false, message: 'Product not found' });
    }
    product.currentPrice = Number(newPrice);
    product.updatedAt = new Date();
    return res.status(200).json({ success: true, message: 'Price updated successfully', data: product });
  } catch (err) {
    next(err);
  }
};

export const seedProducts = async (req, res, next) => {
  try {
    if (isDbConnected()) {
      await Product.deleteMany({});
      const created = await Product.insertMany(
        memoryProducts.map(({ _id, ...rest }) => rest)
      );
      return res.status(201).json({
        success: true,
        message: `Successfully seeded ${created.length} products to MongoDB`,
        data: created,
      });
    }

    return res.status(200).json({
      success: true,
      message: 'Memory products already initialized',
      data: memoryProducts,
    });
  } catch (err) {
    next(err);
  }
};

