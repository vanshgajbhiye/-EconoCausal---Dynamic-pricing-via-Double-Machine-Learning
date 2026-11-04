import express from 'express';
import { upload }                        from '../middleware/upload.js';
import { uploadDataset, getDataset }     from '../controllers/datasetController.js';

const router = express.Router();

// POST /api/datasets       — upload CSV
router.post('/', upload.single('file'), uploadDataset);

// GET  /api/datasets/:id   — get dataset metadata + preview
router.get('/:id', getDataset);

export default router;
