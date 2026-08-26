import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { deflateSync } from 'node:zlib';

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = (value & 1) !== 0 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    }
    table[index] = value >>> 0;
  }
  return table;
})();

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const publicDirectory = join(dirname(scriptDirectory), 'public');
await mkdir(publicDirectory, { recursive: true });

for (const size of [192, 512]) {
  const pixels = new Uint8Array(size * size * 4);
  fill(pixels, size, [9, 11, 16, 255]);

  const scale = size / 512;
  drawBorder(pixels, size, 82 * scale, 118 * scale, 348 * scale, 276 * scale, 22 * scale, [87, 227, 137, 255]);
  drawRect(pixels, size, 142 * scale, 174 * scale, 84 * scale, 164 * scale, [244, 247, 249, 255]);
  drawRect(pixels, size, 286 * scale, 174 * scale, 84 * scale, 164 * scale, [244, 247, 249, 255]);
  drawCircle(pixels, size, 184 * scale, 256 * scale, 22 * scale, [214, 63, 63, 255]);
  drawRect(pixels, size, 312 * scale, 216 * scale, 32 * scale, 80 * scale, [27, 109, 179, 255]);
  drawRect(pixels, size, 288 * scale, 240 * scale, 80 * scale, 32 * scale, [27, 109, 179, 255]);

  await writeFile(join(publicDirectory, `app-icon-${size}.png`), encodePng(size, size, pixels));
}

console.log('Generated 192px and 512px PWA icons.');

function fill(pixels, size, rgba) {
  for (let index = 0; index < size * size; index += 1) {
    const offset = index * 4;
    pixels[offset] = rgba[0];
    pixels[offset + 1] = rgba[1];
    pixels[offset + 2] = rgba[2];
    pixels[offset + 3] = rgba[3];
  }
}

function drawBorder(pixels, size, x, y, width, height, thickness, rgba) {
  drawRect(pixels, size, x, y, width, thickness, rgba);
  drawRect(pixels, size, x, y + height - thickness, width, thickness, rgba);
  drawRect(pixels, size, x, y, thickness, height, rgba);
  drawRect(pixels, size, x + width - thickness, y, thickness, height, rgba);
}

function drawRect(pixels, size, x, y, width, height, rgba) {
  const startX = clamp(Math.round(x), 0, size);
  const startY = clamp(Math.round(y), 0, size);
  const endX = clamp(Math.round(x + width), 0, size);
  const endY = clamp(Math.round(y + height), 0, size);
  for (let row = startY; row < endY; row += 1) {
    for (let column = startX; column < endX; column += 1) {
      setPixel(pixels, size, column, row, rgba);
    }
  }
}

function drawCircle(pixels, size, centerX, centerY, radius, rgba) {
  const startX = clamp(Math.floor(centerX - radius), 0, size);
  const startY = clamp(Math.floor(centerY - radius), 0, size);
  const endX = clamp(Math.ceil(centerX + radius), 0, size);
  const endY = clamp(Math.ceil(centerY + radius), 0, size);
  const radiusSquared = radius * radius;
  for (let row = startY; row < endY; row += 1) {
    for (let column = startX; column < endX; column += 1) {
      const deltaX = column + 0.5 - centerX;
      const deltaY = row + 0.5 - centerY;
      if (deltaX * deltaX + deltaY * deltaY <= radiusSquared) {
        setPixel(pixels, size, column, row, rgba);
      }
    }
  }
}

function setPixel(pixels, size, x, y, rgba) {
  const offset = (y * size + x) * 4;
  pixels[offset] = rgba[0];
  pixels[offset + 1] = rgba[1];
  pixels[offset + 2] = rgba[2];
  pixels[offset + 3] = rgba[3];
}

function encodePng(width, height, rgbaPixels) {
  const scanlineLength = width * 4 + 1;
  const raw = Buffer.alloc(scanlineLength * height);
  for (let row = 0; row < height; row += 1) {
    const rowOffset = row * scanlineLength;
    raw[rowOffset] = 0;
    const sourceOffset = row * width * 4;
    Buffer.from(rgbaPixels.buffer, rgbaPixels.byteOffset + sourceOffset, width * 4).copy(raw, rowOffset + 1);
  }

  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 6;
  header[10] = 0;
  header[11] = 0;
  header[12] = 0;

  return Buffer.concat([
    signature,
    pngChunk('IHDR', header),
    pngChunk('IDAT', deflateSync(raw, { level: 9 })),
    pngChunk('IEND', Buffer.alloc(0)),
  ]);
}

function pngChunk(type, data) {
  const typeBuffer = Buffer.from(type, 'ascii');
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);
  const checksum = Buffer.alloc(4);
  checksum.writeUInt32BE(crc32(Buffer.concat([typeBuffer, data])), 0);
  return Buffer.concat([length, typeBuffer, data, checksum]);
}

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}
