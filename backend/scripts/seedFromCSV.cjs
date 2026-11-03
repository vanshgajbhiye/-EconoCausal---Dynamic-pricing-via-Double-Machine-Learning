/**
 * EconoCausal — Seed MongoDB from ml/data/raw/retail_price.csv
 * Run from repo root: node backend/scripts/seedFromCSV.cjs
 */
const mongoose = require('mongoose');
const fs = require('fs');
const path = require('path');

const MONGO_URI = 'mongodb://127.0.0.1:27017/econocausal';
const CSV_PATH = path.resolve(__dirname, '../../ml/data/raw/retail_price.csv');

// ── Inline Schemas (avoid ESM import issues in CJS seed script) ──

const ProductSchema = new mongoose.Schema({
  name: String,
  sku: { type: String, unique: true },
  category: String,
  basePrice: Number,
  costPrice: Number,
  currentPrice: Number,
  competitorPrice: Number,
  minPrice: Number,
  maxPrice: Number,
  inventory: Number,
  estimatedElasticity: { type: Number, default: -1.75 },
}, { timestamps: true });

const SalesRecordSchema = new mongoose.Schema({
  productId: String,
  date: Date,
  price: Number,
  unitsSold: Number,
  revenue: Number,
  freightPrice: Number,
  discountPercentage: { type: Number, default: 0 },
  isPromotionActive: { type: Boolean, default: false },
  competitorPrice: Number,
  comp2: Number,
  comp3: Number,
  inventoryLevel: Number,
  seasonalityIndex: Number,
  customers: Number,
  weekday: Number,
  weekend: Number,
  holiday: Number,
  month: Number,
  year: Number,
  volume: Number,
  productScore: Number,
  lagPrice: Number,
}, { timestamps: true });

// ── CSV Parser (no external deps) ──
function parseCSV(filePath) {
  const text = fs.readFileSync(filePath, 'utf8');
  const lines = text.split('\n').filter(l => l.trim());
  const headers = lines[0].split(',').map(h => h.trim());
  return lines.slice(1).map(line => {
    const vals = line.split(',');
    const obj = {};
    headers.forEach((h, i) => { obj[h] = (vals[i] || '').trim(); });
    return obj;
  });
}

function num(v, fallback = 0) {
  const n = parseFloat(v);
  return isNaN(n) ? fallback : n;
}

function parseDate(monthYear) {
  // format: "01-05-2017" → day-month-year
  const parts = monthYear.split('-');
  if (parts.length === 3) {
    return new Date(`${parts[2]}-${parts[1]}-${parts[0]}`);
  }
  return new Date();
}

async function seed() {
  console.log('\n🌱  EconoCausal DB Seeder');
  console.log('═══════════════════════════════════════');

  // ── Connect ──
  console.log(`\n🔌  Connecting to ${MONGO_URI} ...`);
  await mongoose.connect(MONGO_URI, { serverSelectionTimeoutMS: 5000 });
  console.log('✅  MongoDB connected: econocausal');

  const Product = mongoose.model('Product', ProductSchema);
  const SalesRecord = mongoose.model('SalesRecord', SalesRecordSchema);

  // ── Parse CSV ──
  console.log(`\n📂  Reading CSV: ${CSV_PATH}`);
  const rows = parseCSV(CSV_PATH);
  console.log(`    Found ${rows.length} rows`);

  // ── Build unique products from product_id × category ──
  const productMap = new Map();
  rows.forEach(row => {
    if (!productMap.has(row.product_id)) {
      const base = num(row.unit_price);
      const cost = base * 0.45; // ~45% cost ratio for retail
      const comp1 = num(row.comp_1, base);
      productMap.set(row.product_id, {
        name: row.product_id.replace(/_/g, ' ').toUpperCase(),
        sku: `EC-${row.product_id.toUpperCase()}-${row.product_category_name.substring(0,3).toUpperCase()}`,
        category: row.product_category_name.replace(/_/g, ' '),
        basePrice: base,
        costPrice: parseFloat(cost.toFixed(2)),
        currentPrice: base,
        competitorPrice: comp1,
        minPrice: parseFloat((cost * 1.05).toFixed(2)),
        maxPrice: parseFloat((base * 2.2).toFixed(2)),
        inventory: Math.floor(num(row.volume, 500) / 10),
        estimatedElasticity: -1.75,
      });
    }
  });

  const productDocs = Array.from(productMap.values());

  // ── Wipe & Insert Products ──
  console.log(`\n🗑   Clearing existing Products ...`);
  await Product.deleteMany({});
  console.log(`📦  Inserting ${productDocs.length} products ...`);
  const insertedProducts = await Product.insertMany(productDocs, { ordered: false });
  console.log(`✅  Products inserted: ${insertedProducts.length}`);

  // Build productId lookup (sku suffix → _id)
  const skuToId = new Map();
  insertedProducts.forEach(p => {
    // extract original product_id from sku: "EC-BED1-BED" → "bed1"
    const parts = p.sku.split('-');
    const rawId = parts[1].toLowerCase(); // "bed1"
    skuToId.set(rawId, p._id.toString());
  });

  // ── Build SalesRecords ──
  const salesDocs = rows.map(row => {
    const qty = num(row.qty, 1);
    const price = num(row.unit_price);
    const freight = num(row.freight_price);
    const comp1 = num(row.comp_1, price);
    const totalDays = num(row.weekday) + num(row.weekend);
    const seasonality = totalDays > 0
      ? parseFloat((num(row.weekend) / totalDays * 2).toFixed(4))
      : 1.0;

    return {
      productId: skuToId.get(row.product_id) || row.product_id,
      date: parseDate(row.month_year),
      price,
      unitsSold: qty,
      revenue: parseFloat((num(row.total_price)).toFixed(2)),
      freightPrice: freight,
      discountPercentage: 0,
      isPromotionActive: num(row.holiday) > 0,
      competitorPrice: comp1,
      comp2: num(row.comp_2),
      comp3: num(row.comp_3),
      inventoryLevel: Math.floor(num(row.volume, 500) / 10),
      seasonalityIndex: seasonality,
      customers: num(row.customers),
      weekday: num(row.weekday),
      weekend: num(row.weekend),
      holiday: num(row.holiday),
      month: num(row.month),
      year: num(row.year),
      volume: num(row.volume),
      productScore: num(row.product_score),
      lagPrice: num(row.lag_price),
    };
  });

  // ── Wipe & Insert SalesRecords ──
  console.log(`\n🗑   Clearing existing SalesRecords ...`);
  await SalesRecord.deleteMany({});
  console.log(` Inserting ${salesDocs.length} sales records ...`);
  const insertedSales = await SalesRecord.insertMany(salesDocs, { ordered: false });
  console.log(` Sales records inserted: ${insertedSales.length}`);

  // ── Summary ──
  console.log('\n═══════════════════════════════════════');
  console.log('  Seeding complete!');
  console.log(`    Products    : ${insertedProducts.length}`);
  console.log(`    SalesRecords: ${insertedSales.length}`);
  console.log('    DB          : econocausal @ localhost:27017');
  console.log('═══════════════════════════════════════\n');

  await mongoose.disconnect();
  process.exit(0);
}

seed().catch(err => {
  console.error('\n❌  Seeder failed:', err.message);
  process.exit(1);
});
