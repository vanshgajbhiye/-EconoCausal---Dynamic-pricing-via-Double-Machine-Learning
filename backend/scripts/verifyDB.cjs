const mongoose = require('mongoose');
const MONGO_URI = 'mongodb://127.0.0.1:27017/econocausal';

const ProductSchema = new mongoose.Schema({}, { strict: false });
const SalesSchema   = new mongoose.Schema({}, { strict: false });

async function verify() {
  await mongoose.connect(MONGO_URI, { serverSelectionTimeoutMS: 5000 });
  const P = mongoose.model('Product', ProductSchema);
  const S = mongoose.model('SalesRecord', SalesSchema);

  const pc   = await P.countDocuments();
  const sc   = await S.countDocuments();
  const cats = await P.distinct('category');
  const sample = await P.findOne({}, { name: 1, sku: 1, basePrice: 1, category: 1 });

  console.log('\n📊  EconoCausal DB — Verification');
  console.log('══════════════════════════════════');
  console.log(`  Products      : ${pc}`);
  console.log(`  SalesRecords  : ${sc}`);
  console.log(`  Categories    : ${cats.join(', ')}`);
  console.log(`  Sample product: ${JSON.stringify(sample)}`);
  console.log('══════════════════════════════════\n');

  await mongoose.disconnect();
  process.exit(0);
}

verify().catch(e => { console.error('❌', e.message); process.exit(1); });
