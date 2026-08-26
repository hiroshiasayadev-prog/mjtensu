import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { deflateSync } from 'node:zlib';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const publicDirectory = join(dirname(scriptDirectory), 'public');
await mkdir(publicDirectory, { recursive: true });

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <rect width="512" height="512" rx="92" fill="#090b10"/>
  <g fill="#f5f7fa" stroke="#d4aa3a" stroke-width="8">
    <rect x="74" y="100" width="78" height="112" rx="10"/>
    <rect x="166" y="100" width="78" height="112" rx="10"/>
    <rect x="258" y="100" width="78" height="112" rx="10"/>
    <rect x="350" y="100" width="78" height="112" rx="10"/>
    <rect x="74" y="244" width="78" height="112" rx="10"/>
    <rect x="166" y="244" width="78" height="112" rx="10"/>
    <rect x="258" y="244" width="78" height="112" rx="10"/>
    <rect x="350" y="244" width="78" height="112" rx="10"/>
  </g>
  <path d="M92 402h328" stroke="#57e389" stroke-width="28" stroke-linecap="round"/>
</svg>`;
await writeFile(join(publicDirectory, 'app-icon.svg'), svg, 'utf8');

for (const size of [192, 512]) {
  const pixels = new Uint8Array(size * size * 4);
  fill(pixels, size, [9, 11, 16, 255]);
  const scale = size / 512;
  for (const y of [100, 244]) {
    for (const x of [74, 166, 258, 350]) {
      rect(pixels, size, x, y, 78, 112, [245, 247, 250, 255], scale);
      border(pixels, size, x, y, 78, 112, 8, [212, 170, 58, 255], scale);
    }
  }
  rect(pixels, size, 92, 388, 328, 28, [87, 227, 137, 255], scale);
  await writeFile(join(publicDirectory, `app-icon-${size}.png`), png(size, pixels));
}

function fill(data, size, rgba) {
  for (let index = 0; index < size * size; index += 1) set(data, size, index % size, Math.floor(index / size), rgba);
}
function rect(data, size, x, y, width, height, rgba, scale) {
  const left = Math.round(x * scale), top = Math.round(y * scale);
  const right = Math.round((x + width) * scale), bottom = Math.round((y + height) * scale);
  for (let row = top; row < bottom; row += 1) for (let column = left; column < right; column += 1) set(data, size, column, row, rgba);
}
function border(data, size, x, y, width, height, thickness, rgba, scale) {
  rect(data, size, x, y, width, thickness, rgba, scale);
  rect(data, size, x, y + height - thickness, width, thickness, rgba, scale);
  rect(data, size, x, y, thickness, height, rgba, scale);
  rect(data, size, x + width - thickness, y, thickness, height, rgba, scale);
}
function set(data, size, x, y, rgba) {
  if (x < 0 || y < 0 || x >= size || y >= size) return;
  const offset = (y * size + x) * 4;
  data.set(rgba, offset);
}
function png(size, rgba) {
  const stride = size * 4 + 1;
  const raw = Buffer.alloc(stride * size);
  for (let row = 0; row < size; row += 1) {
    const destination = row * stride;
    raw[destination] = 0;
    Buffer.from(rgba.buffer, rgba.byteOffset + row * size * 4, size * 4).copy(raw, destination + 1);
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(size, 0); header.writeUInt32BE(size, 4);
  header[8] = 8; header[9] = 6;
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}
function chunk(type, data) {
  const typeBytes = Buffer.from(type, 'ascii');
  const length = Buffer.alloc(4); length.writeUInt32BE(data.length);
  const checksum = Buffer.alloc(4); checksum.writeUInt32BE(crc32(Buffer.concat([typeBytes, data])));
  return Buffer.concat([length, typeBytes, data, checksum]);
}
function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) crc = (crc & 1) ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
  }
  return (crc ^ 0xffffffff) >>> 0;
}
