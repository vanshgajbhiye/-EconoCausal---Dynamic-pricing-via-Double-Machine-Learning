// Quick health check — CJS, runs from any directory
const http = require('http');

function get(url) {
  return new Promise((resolve, reject) => {
    http.get(url, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => resolve({ status: res.statusCode, body: data }));
    }).on('error', reject);
  });
}

async function main() {
  // Give server a moment
  await new Promise(r => setTimeout(r, 3000));
  try {
    const r = await get('http://localhost:5000/api/health');
    console.log('Health status:', r.status);
    console.log('Body:', r.body);

    const p = await get('http://localhost:5000/api/products');
    const pJson = JSON.parse(p.body);
    console.log('\nProducts count:', pJson.count);
    console.log('First product:', JSON.stringify(pJson.data[0]?.name));

    const s = await get('http://localhost:5000/api/sales?limit=3');
    const sJson = JSON.parse(s.body);
    console.log('\nSales count:', sJson.count);
  } catch (e) {
    console.error('Check failed:', e.message);
  }
  process.exit(0);
}

main();
