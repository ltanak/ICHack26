const fs = require('fs');
const path = require('path');

const src = path.join(__dirname, '..', 'public', 'countries-110m.json');
const out = path.join(__dirname, '..', 'public', 'countries-filtered.json');

const allowed = new Set([
  'Canada',
  'United States of America',
  'Mexico',
  'Belize',
  'Guatemala',
  'El Salvador',
  'Honduras',
  'Nicaragua',
  'Costa Rica',
  'Panama',
  'Argentina',
  'Bolivia',
  'Brazil',
  'Chile',
  'Colombia',
  'Ecuador',
  'Guyana',
  'Paraguay',
  'Peru',
  'Suriname',
  'Uruguay',
  'Venezuela',
  'Falkland Is.'
]);

const raw = fs.readFileSync(src, 'utf8');
const topo = JSON.parse(raw);
if (!topo.objects || !topo.objects.countries) {
  console.error('Unexpected TopoJSON structure: missing objects.countries');
  process.exit(1);
}

const filtered = topo.objects.countries.geometries.filter(g => {
  const name = g.properties && g.properties.name;
  return allowed.has(name);
});

const outTopo = Object.assign({}, topo, {
  objects: {
    countries: {
      type: 'GeometryCollection',
      geometries: filtered
    }
  }
});

fs.writeFileSync(out, JSON.stringify(outTopo, null, 2), 'utf8');
console.log('Wrote', out, 'with', filtered.length, 'geometries');
