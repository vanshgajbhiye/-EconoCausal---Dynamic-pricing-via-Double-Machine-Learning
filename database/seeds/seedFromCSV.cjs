/**
 * EconoCausal — DB Seeder
 * ========================
 * Reads ml/data/raw/retail_price.csv and populates MongoDB.
 *
 * Run from repo root:
 *   node database/seeds/seedFromCSV.cjs
 *
 * Prerequisites:
 *   - MongoDB running at localhost:27017
 *   - node_modules present in backend/ (mongoose installed)
 */

const fs   = require('fs');
const path = require('path');

// Resolve mongoose from backend's node_modules (database/ has no node_modules)
const mg = require(path.resolve(__dirname, '../../backend/node_modules/mongoose'));

const MONGO_URI = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/econocausal';
const CSV_PATH  = path.resolve(__dirname, '../../ml/data/raw/retail_price.csv');

// ── Inline schemas (CJS compat) ─────────────────────────────────

const ProductSchema = new mg.Schema({
  name:                 String,
  sku:                  { type: String, unique: true },
  category:             String,
  basePrice:            Number,
  costPrice:            Number,
  currentPrice:         Number,
  competitorPrice:      Number,
  minPrice:             Number,
  maxPrice:             Number,
  inventory:            Number,
  estimatedElasticity:  { type: Number, default: -1.75 },
  productScore:         Number,
  productWeightG:       Number,
  productNameLength:    Number,
  productDescriptionLength: Number,
  productPhotosQty:     Number,
  volume:               Number,
}, { timestamps: true });

const SalesSchema = new mg.Schema({
  productId:       mg.Schema.Types.Mixed,
  date:            Date,
  price:           Number,
  unitsSold:       Number,
  revenue:         Number,
  freightPrice:    Number,
  discountPercentage: { type: Number, default: 0 },
  isPromotionActive:  Boolean,
  competitorPrice: Number,
  comp2:           Number,
  comp3:           Number,
  ps1: Number, ps2: Number, ps3: Number,
  fp1: Number, fp2: Number, fp3: Number,
  customers:        Number,
  weekday:          Number,
  weekend:          Number,
  holiday:          Number,
  month:            Number,
  year:             Number,
  volume:           Number,
  seasonalityIndex: Number,
  inventoryLevel:   Number,
  productScore:     Number,
  lagPrice:         Number,
}, { timestamps: true });

// ── Helpers ──────────────────────────────────────────────────────

function parseCSV(filePath) {
  const lines   = fs.readFileSync(filePath, 'utf8').split('\n').filter(l => l.trim());
  const headers = lines[0].split(',').map(h => h.trim());
  return lines.slice(1).map(line => {
    const vals = line.split(',');
    const obj  = {};
    headers.forEach((h, i) => { obj[h] = (vals[i] || '').trim(); });
    return obj;
  });
}

const n = (v, fb = 0) => { const x = parseFloat(v); return isNaN(x) ? fb : x; };

function parseDate(monthYear) {
  const [d, m, y] = monthYear.split('-');
  return new Date(`${y}-${m}-${d}`);
}

// ── Main ─────────────────────────────────────────────────────────

async function seed() {
  console.log('\n🌱  EconoCausal DB Seeder');
  console.log('═══════════════════════════════════════════════');
  console.log(`    Source : ${CSV_PATH}`);
  console.log(`    Target : ${MONGO_URI}`);
  console.log('═══════════════════════════════════════════════\n');

  // ── Connect ──
  await mg.connect(MONGO_URI, { serverSelectionTimeoutMS: 5000 });
  console.log('✅  Connected to MongoDB\n');

  const Product     = mg.model('Product',     ProductSchema);
  const SalesRecord = mg.model('SalesRecord', SalesSchema);

  // ── Parse ──
  const rows = parseCSV(CSV_PATH);
  console.log(`📂  Loaded ${rows.length} rows from CSV`);

  // ── Build Products (unique per product_id) ──
  const productMap = new Map();
  rows.forEach(row => {
    if (productMap.has(row.product_id)) return;
    const base = n(row.unit_price);
    const cost = parseFloat((base * 0.45).toFixed(2));
    productMap.set(row.product_id, {
      name:     row.product_id.replace(/_/g, ' ').toUpperCase(),
      sku:      `EC-${row.product_id.toUpperCase()}-${row.product_category_name.substring(0, 3).toUpperCase()}`,
      category: row.product_category_name.replace(/_/g, ' '),
      basePrice:       base,
      costPrice:       cost,
      currentPrice:    base,
      competitorPrice: n(row.comp_1, base),
      minPrice:        parseFloat((cost * 1.05).toFixed(2)),
      maxPrice:        parseFloat((base * 2.2).toFixed(2)),
      inventory:       Math.max(1, Math.floor(n(row.volume, 500) / 10)),
      estimatedElasticity: -1.75,
      productScore:        n(row.product_score),
      productWeightG:      n(row.product_weight_g),
      productNameLength:   n(row.product_name_lenght),
      productDescriptionLength: n(row.product_description_lenght),
      productPhotosQty:    n(row.product_photos_qty),
      volume:              n(row.volume),
    });
  });

  // ── Seed Products ──
  console.log(`\n🗑   Clearing Products collection ...`);
  await Product.deleteMany({});
  const productDocs = Array.from(productMap.values());
  console.log(`📦  Inserting ${productDocs.length} products ...`);
  const inserted = await Product.insertMany(productDocs, { ordered: false });
  console.log(`✅  Products → ${inserted.length} inserted`);

  // Build productId lookup: raw csv product_id → mongo _id
  const idMap = new Map();
  inserted.forEach(p => {
    const csvId = p.sku.split('-')[1].toLowerCase(); // EC-BED1-BED → bed1
    idMap.set(csvId, p._id.toString());
  });

  // ── Build SalesRecords ──
  const salesDocs = rows.map(row => {
    const total   = n(row.weekday) + n(row.weekend);
    const season  = total > 0 ? parseFloat((n(row.weekend) / total * 2).toFixed(4)) : 1.0;
    const mongoId = idMap.get(row.product_id) || row.product_id;
    return {
      productId:        mongoId,
      date:             parseDate(row.month_year),
      price:            n(row.unit_price),
      unitsSold:        n(row.qty, 1),
      revenue:          n(row.total_price),
      freightPrice:     n(row.freight_price),
      discountPercentage: 0,
      isPromotionActive: n(row.holiday) > 0,
      competitorPrice:  n(row.comp_1),
      comp2:            n(row.comp_2),
      comp3:            n(row.comp_3),
      ps1: n(row.ps1), ps2: n(row.ps2), ps3: n(row.ps3),
      fp1: n(row.fp1), fp2: n(row.fp2), fp3: n(row.fp3),
      customers:        n(row.customers),
      weekday:          n(row.weekday),
      weekend:          n(row.weekend),
      holiday:          n(row.holiday),
      month:            n(row.month),
      year:             n(row.year),
      volume:           n(row.volume),
      seasonalityIndex: season,
      inventoryLevel:   Math.floor(n(row.volume, 500) / 10),
      productScore:     n(row.product_score),
      lagPrice:         n(row.lag_price),
    };
  });

  // ── Seed SalesRecords ──
  console.log(`\n🗑   Clearing SalesRecords collection ...`);
  await SalesRecord.deleteMany({});
  console.log(`  Inserting ${salesDocs.length} sales records ...`);
  const insertedSales = await SalesRecord.insertMany(salesDocs, { ordered: false });
  console.log(` SalesRecords → ${insertedSales.length} inserted`);

  // ── Summary ──
  console.log('\n═══════════════════════════════════════════════');
  console.log('  Seeding complete!');
  console.log(`    Products     : ${inserted.length}`);
  console.log(`    SalesRecords : ${insertedSales.length}`);
  console.log(`    Database     : econocausal @ localhost:27017`);
  console.log('═══════════════════════════════════════════════\n');

  await mg.disconnect();
  process.exit(0);
}

seed().catch(err => {
  console.error('\n❌  Seeder failed:', err.message);
  process.exit(1);
});
