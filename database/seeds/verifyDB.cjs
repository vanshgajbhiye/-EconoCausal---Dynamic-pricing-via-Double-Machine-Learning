/**
 * EconoCausal — DB Verification
 * ==============================
 * Quick check of what's in MongoDB after seeding.
 *
 * Run from repo root:
 *   node database/seeds/verifyDB.cjs
 */

const path = require('path');
const mg   = require(path.resolve(__dirname, '../../backend/node_modules/mongoose'));

const MONGO_URI = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/econocausal';

async function verify() {
  await mg.connect(MONGO_URI, { serverSelectionTimeoutMS: 5000 });

  const Product     = mg.model('Product',     new mg.Schema({}, { strict: false }));
  const SalesRecord = mg.model('SalesRecord', new mg.Schema({}, { strict: false }));

  const pc     = await Product.countDocuments();
  const sc     = await SalesRecord.countDocuments();
  const cats   = await Product.distinct('category');
  const sample = await Product.findOne({}, { name: 1, sku: 1, basePrice: 1, category: 1, _id: 0 });

  console.log('\n📊  EconoCausal DB — Verification');
  console.log('════════════════════════════════════════');
  console.log(`  Products      : ${pc}`);
  console.log(`  SalesRecords  : ${sc}`);
  console.log(`  Categories    : ${cats.join(', ')}`);
  console.log(`  Sample        : ${JSON.stringify(sample)}`);
  console.log('════════════════════════════════════════\n');

  await mg.disconnect();
  process.exit(0);
}

verify().catch(e => { console.error('❌', e.message); process.exit(1); });
