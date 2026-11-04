/**
 * EconoCausal — Backend API Integration Test
 * Run: node backend/test/apiTest.cjs  (server must be running on :5000)
 */
const http = require('http');
const fs   = require('fs');
const path = require('path');

function request(options, body = null) {
  return new Promise((resolve, reject) => {
    const req = http.request(options, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        try { resolve({ status: res.statusCode, body: JSON.parse(data) }); }
        catch { resolve({ status: res.statusCode, body: data }); }
      });
    });
    req.on('error', reject);
    if (body) req.write(body);
    req.end();
  });
}

function pass(label) { console.log(`  ✅  ${label}`); }
function fail(label, detail) { console.log(`  ❌  ${label}: ${detail}`); process.exitCode = 1; }

async function run() {
  console.log('\n🧪  EconoCausal Backend — API Tests');
  console.log('══════════════════════════════════════\n');

  // 1. Health
  const h = await request({ hostname: 'localhost', port: 5000, path: '/api/health' });
  h.body?.status === 'online' ? pass('GET /api/health → online') : fail('health', JSON.stringify(h.body));

  // 2. Products (from real MongoDB data)
  const p = await request({ hostname: 'localhost', port: 5000, path: '/api/products' });
  p.body?.count >= 1 ? pass(`GET /api/products → ${p.body.count} products`) : fail('products', JSON.stringify(p.body));

  // 3. Sales
  const s = await request({ hostname: 'localhost', port: 5000, path: '/api/sales' });
  s.body?.count >= 1 ? pass(`GET /api/sales → ${s.body.count} records`) : fail('sales', JSON.stringify(s.body));

  // 4. DML Pricing optimize
  const dmlBody = JSON.stringify({ productId: 'prod_101' });
  const dml = await request({
    hostname: 'localhost', port: 5000, path: '/api/pricing/optimize',
    method: 'POST', headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(dmlBody) }
  }, dmlBody);
  dml.body?.success ? pass(`POST /api/pricing/optimize → ATE: ${dml.body.data?.causalTreatmentEffect}`) : fail('pricing/optimize', JSON.stringify(dml.body));

  // 5. Dataset upload (simulate multipart with real CSV)
  const csvContent = [
    'unit_price,qty,comp_1,comp_2,customers,month,holiday',
    '45.95,3,89.9,215,57,5,1','50.00,2,92.0,220,60,6,0','40.00,5,85.0,200,70,7,1',
    '55.00,1,95.0,230,50,8,0','42.00,4,88.0,210,65,9,0','48.00,3,91.0,218,58,10,1',
    '38.00,6,82.0,195,80,11,0','52.00,2,93.0,225,55,12,1','46.00,3,90.0,216,59,1,0',
    '44.00,4,87.0,208,62,2,0','51.00,2,94.0,228,53,3,1','39.00,5,83.0,198,78,4,0',
  ].join('\n') + '\n';
  const boundary = '----TestBoundary123';
  const multipart = [
    `--${boundary}`,
    'Content-Disposition: form-data; name="file"; filename="test.csv"',
    'Content-Type: text/csv',
    '',
    csvContent,
    `--${boundary}`,
    'Content-Disposition: form-data; name="mapping"',
    '',
    JSON.stringify({ treatment: 'unit_price', outcome: 'qty' }),
    `--${boundary}--`,
  ].join('\r\n');
  const ds = await request({
    hostname: 'localhost', port: 5000, path: '/api/datasets',
    method: 'POST',
    headers: { 'Content-Type': `multipart/form-data; boundary=${boundary}`, 'Content-Length': Buffer.byteLength(multipart) }
  }, multipart);
  const datasetId = ds.body?.dataset_id;
  datasetId ? pass(`POST /api/datasets → id: ${datasetId}, rows: ${ds.body.rowCount}`) : fail('datasets upload', JSON.stringify(ds.body));

  // 6. Get dataset
  if (datasetId) {
    const gd = await request({ hostname: 'localhost', port: 5000, path: `/api/datasets/${datasetId}` });
    gd.body?.success ? pass(`GET /api/datasets/${datasetId} → ${gd.body.data?.rowCount} rows`) : fail('datasets get', JSON.stringify(gd.body));
  }

  // 7. Run analysis
  if (datasetId) {
    const aBody = JSON.stringify({ dataset_id: datasetId, treatment: 'unit_price', outcome: 'qty', method: 'DML' });
    const a = await request({
      hostname: 'localhost', port: 5000, path: '/api/analysis',
      method: 'POST', headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(aBody) }
    }, aBody);
    a.body?.success
      ? pass(`POST /api/analysis → ATE: ${a.body.result?.average_treatment_effect?.toFixed(4)}, P*: $${a.body.result?.optimal_price?.toFixed(2)}`)
      : fail('analysis', JSON.stringify(a.body));
  }

  console.log('\n══════════════════════════════════════');
  if (process.exitCode) {
    console.log('❌  Some tests failed — see above\n');
  } else {
    console.log('🎉  All tests passed!\n');
  }
}

run().catch(e => { console.error('❌ Test runner error:', e.message); process.exit(1); });
