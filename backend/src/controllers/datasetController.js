/**
 * Dataset Controller
 * ==================
 * Handles CSV uploads from the frontend's DatasetStudio page.
 *
 * POST /api/datasets        → uploadDataset
 * GET  /api/datasets/:id    → getDataset
 *
 * Datasets are held in-memory (Map) keyed by a generated ID.
 * In production this would persist to MongoDB GridFS or S3.
 */

// In-memory store: datasetId → { headers, rows, mapping, originalName, uploadedAt }
const datasetStore = new Map();

// ── CSV parser (no deps — works on Buffer from multer memoryStorage) ──
function parseCSVBuffer(buffer) {
  const text    = buffer.toString('utf8');
  const lines   = text.split(/\r?\n/).filter(l => l.trim());
  if (lines.length < 2) throw new Error('CSV must have a header row and at least one data row');

  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
  const rows    = lines.slice(1).map(line => {
    const vals = line.split(',');
    const obj  = {};
    headers.forEach((h, i) => { obj[h] = (vals[i] || '').trim().replace(/^"|"$/g, ''); });
    return obj;
  });
  return { headers, rows };
}

function generateId() {
  return `ds_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
}

// ──────────────────────────────────────────────────────────────────────
// POST /api/datasets
// Body: multipart/form-data  { file: CSV, mapping: JSON string }
// ──────────────────────────────────────────────────────────────────────
export const uploadDataset = async (req, res, next) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: 'No CSV file uploaded' });
    }

    const { headers, rows } = parseCSVBuffer(req.file.buffer);

    // Parse mapping from FormData (sent as JSON string by api.js)
    let mapping = {};
    if (req.body.mapping) {
      try { mapping = JSON.parse(req.body.mapping); } catch (_) { /* ignore bad mapping */ }
    }

    const id = generateId();
    datasetStore.set(id, {
      id,
      originalName: req.file.originalname,
      headers,
      rows,
      rowCount: rows.length,
      mapping,
      uploadedAt: new Date().toISOString(),
    });

    return res.status(201).json({
      success: true,
      dataset_id: id,
      originalName: req.file.originalname,
      headers,
      rowCount: rows.length,
      mapping,
      message: `Dataset uploaded: ${rows.length} rows, ${headers.length} columns`,
    });
  } catch (err) {
    next(err);
  }
};

// ──────────────────────────────────────────────────────────────────────
// GET /api/datasets/:id
// ──────────────────────────────────────────────────────────────────────
export const getDataset = (req, res) => {
  const { id } = req.params;
  const dataset = datasetStore.get(id);

  if (!dataset) {
    return res.status(404).json({ success: false, message: `Dataset ${id} not found` });
  }

  // Return metadata + first 10 rows preview (not full data)
  return res.status(200).json({
    success: true,
    data: {
      id:           dataset.id,
      originalName: dataset.originalName,
      headers:      dataset.headers,
      rowCount:     dataset.rowCount,
      mapping:      dataset.mapping,
      uploadedAt:   dataset.uploadedAt,
      preview:      dataset.rows.slice(0, 10),
    },
  });
};

// Exported for use by analysisController
export { datasetStore };
