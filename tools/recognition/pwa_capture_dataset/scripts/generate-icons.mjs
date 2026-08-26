import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { deflateSync } from 'node:zlib';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const publicDirectory = join(dirname(scriptDirectory), 'public');
await mkdir(publicDirectory, { recursive: true });

for (const size of [192, 512]) {
  const pixels = new Uint8Array(size * size * 4);
  fill(pixels, size, [9, 11, 16, 255]);
  const scale = size / 512;
  border(pixels, size, 72, 104, 368, 304, 22, [87, 227, 137, 255], scale);
  rect(pixels, size, 120, 156, 210, 60, [245, 247, 250, 255], scale);
  rect(pixels, size, 120, 238, 210, 60, [245, 247, 250, 255], scale);
  rect(pixels, size, 350, 156, 54, 142, [245, 247, 250, 255], scale);
  circle(pixels, size, 256, 352, 34, [229, 72, 77, 255], scale);
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
function circle(data, size, x, y, radius, rgba, scale) {
  const cx = x * scale, cy = y * scale, r = radius * scale;
  for (let row = Math.floor(cy - r); row <= Math.ceil(cy + r); row += 1) {
    for (let column = Math.floor(cx - r); column <= Math.ceil(cx + r); column += 1) {
      if ((column - cx) ** 2 + (row - cy) ** 2 <= r ** 2) set(data, size, column, row, rgba);
    }
  }
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
