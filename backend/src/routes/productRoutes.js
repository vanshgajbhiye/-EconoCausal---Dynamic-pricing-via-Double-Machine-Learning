import express from 'express';
import {
  getProducts,
  getProductById,
  createProduct,
  updateProductPrice,
  seedProducts,
} from '../controllers/productController.js';

const router = express.Router();

router.post('/seed', seedProducts);

router.route('/')
  .get(getProducts)
  .post(createProduct);

router.route('/:id')
  .get(getProductById);

router.route('/:id/price')
  .patch(updateProductPrice);

export default router;

