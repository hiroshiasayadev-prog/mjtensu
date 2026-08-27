import { mkdir, rm, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const frontendRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const outputDir = join(frontendRoot, 'public', 'tiles');

const upstreamRepository = 'FluffyStuff/riichi-mahjong-tiles';
const defaultUpstreamRef = '26e127ba2117f45cdce5ea0225748cc0cfad3169';
const upstreamRef = process.env.MAHJONG_TILE_ASSET_REF ?? defaultUpstreamRef;
const upstreamBase = `https://raw.githubusercontent.com/${upstreamRepository}/${upstreamRef}/Regular`;

const FACE_SCALE = 0.79;

const tileSources = {
  '1m': 'Man1.svg',
  '2m': 'Man2.svg',
  '3m': 'Man3.svg',
  '4m': 'Man4.svg',
  '5m': 'Man5.svg',
  '5m-red': 'Man5-Dora.svg',
  '6m': 'Man6.svg',
  '7m': 'Man7.svg',
  '8m': 'Man8.svg',
  '9m': 'Man9.svg',
  '1p': 'Pin1.svg',
  '2p': 'Pin2.svg',
  '3p': 'Pin3.svg',
  '4p': 'Pin4.svg',
  '5p': 'Pin5.svg',
  '5p-red': 'Pin5-Dora.svg',
  '6p': 'Pin6.svg',
  '7p': 'Pin7.svg',
  '8p': 'Pin8.svg',
  '9p': 'Pin9.svg',
  '1s': 'Sou1.svg',
  '2s': 'Sou2.svg',
  '3s': 'Sou3.svg',
  '4s': 'Sou4.svg',
  '5s': 'Sou5.svg',
  '5s-red': 'Sou5-Dora.svg',
  '6s': 'Sou6.svg',
  '7s': 'Sou7.svg',
  '8s': 'Sou8.svg',
  '9s': 'Sou9.svg',
  '1z': 'Ton.svg',
  '2z': 'Nan.svg',
  '3z': 'Shaa.svg',
  '4z': 'Pei.svg',
  '5z': 'Haku.svg',
  '6z': 'Hatsu.svg',
  '7z': 'Chun.svg',
};

async function fetchText(filename) {
  const url = `${upstreamBase}/${filename}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

function parseViewBox(svg, filename) {
  const viewBoxMatch = svg.match(/\bviewBox\s*=\s*["']([^"']+)["']/i);
  if (viewBoxMatch !== null) {
    const numbers = viewBoxMatch[1].trim().split(/[\s,]+/).map(Number);
    if (numbers.length === 4 && numbers.every(Number.isFinite)) {
      return numbers;
    }
  }

  const widthMatch = svg.match(/\bwidth\s*=\s*["']([0-9.]+)(?:px)?["']/i);
  const heightMatch = svg.match(/\bheight\s*=\s*["']([0-9.]+)(?:px)?["']/i);
  if (widthMatch !== null && heightMatch !== null) {
    return [0, 0, Number(widthMatch[1]), Number(heightMatch[1])];
  }

  throw new Error(`Could not determine SVG dimensions for ${filename}`);
}

function prefixSvgIds(svg, prefix) {
  const ids = new Set();
  const idPattern = /\bid\s*=\s*(["'])([^"']+)\1/g;
  for (const match of svg.matchAll(idPattern)) {
    ids.add(match[2]);
  }

  let result = svg.replace(idPattern, (_match, quote, id) => `id=${quote}${prefix}${id}${quote}`);

  for (const id of ids) {
    const escapedId = id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    result = result
      .replace(new RegExp(`url\\(#${escapedId}\\)`, 'g'), `url(#${prefix}${id})`)
      .replace(
        new RegExp(`((?:xlink:)?href\\s*=\\s*["'])#${escapedId}(["'])`, 'g'),
        `$1#${prefix}${id}$2`,
      );
  }

  return result;
}

function asNestedSvg(svg, prefix, box) {
  const normalized = prefixSvgIds(
    svg
      .replace(/^\uFEFF/, '')
      .replace(/<\?xml[\s\S]*?\?>/gi, '')
      .replace(/<!DOCTYPE[\s\S]*?>/gi, '')
      .trim(),
    prefix,
  );

  const openMatch = normalized.match(/<svg\b([^>]*)>/i);
  const closeIndex = normalized.toLowerCase().lastIndexOf('</svg>');
  if (openMatch === null || closeIndex < 0) {
    throw new Error('Invalid SVG document');
  }

  const rootAttributes = openMatch[1]
    .replace(/\s(?:x|y|width|height)\s*=\s*(["'])[^"']*\1/gi, '')
    .trim();
  const innerStart = openMatch.index + openMatch[0].length;
  const inner = normalized.slice(innerStart, closeIndex);

  return `<svg ${rootAttributes} x="${box.x}" y="${box.y}" width="${box.width}" height="${box.height}" preserveAspectRatio="xMidYMid meet">${inner}</svg>`;
}

function composeTileSvg(frontSvg, faceSvg, outputName) {
  const [minX, minY, width, height] = parseViewBox(frontSvg, 'Front.svg');
  const faceWidth = width * FACE_SCALE;
  const faceHeight = height * FACE_SCALE;
  const faceX = minX + (width - faceWidth) / 2;
  const faceY = minY + (height - faceHeight) / 2;
  const safePrefix = outputName.replace(/[^a-z0-9]+/gi, '-');
  const face = asNestedSvg(faceSvg, `face-${safePrefix}-`, {
    x: faceX,
    y: faceY,
    width: faceWidth,
    height: faceHeight,
  });

  const normalizedFront = frontSvg
    .replace(/^\uFEFF/, '')
    .replace(/<\?xml[\s\S]*?\?>/gi, '')
    .replace(/<!DOCTYPE[\s\S]*?>/gi, '')
    .trim();
  const closeIndex = normalizedFront.toLowerCase().lastIndexOf('</svg>');
  if (closeIndex < 0) {
    throw new Error('Invalid Front.svg document');
  }

  // Keep Front.svg as the outer SVG instead of nesting/rebuilding it. Inkscape
  // filters and edge highlights can change when the front artwork becomes a
  // nested SVG viewport, so only the transparent face artwork is injected.
  return `${normalizedFront.slice(0, closeIndex)}\n${face}\n${normalizedFront.slice(closeIndex)}\n`;
}

async function main() {
  console.log(`Fetching tile artwork from ${upstreamRepository}@${upstreamRef}...`);

  const frontPromise = fetchText('Front.svg');
  const backPromise = fetchText('Back.svg');
  const sourceEntries = Object.entries(tileSources);
  const facePromises = sourceEntries.map(async ([outputName, filename]) => [
    outputName,
    filename,
    await fetchText(filename),
  ]);

  const [frontSvg, backSvg, faces] = await Promise.all([
    frontPromise,
    backPromise,
    Promise.all(facePromises),
  ]);

  await rm(outputDir, { recursive: true, force: true });
  await mkdir(outputDir, { recursive: true });

  for (const [outputName, _filename, faceSvg] of faces) {
    const composed = composeTileSvg(frontSvg, faceSvg, outputName);
    await writeFile(join(outputDir, `${outputName}.svg`), composed, 'utf8');
  }

  await writeFile(join(outputDir, 'back.svg'), backSvg, 'utf8');

  await writeFile(
    join(outputDir, 'SOURCE.txt'),
    [
      'Generated by scripts/generate-tile-assets.mjs.',
      `Source: https://github.com/${upstreamRepository}`,
      `Ref: ${upstreamRef}`,
      'Variant: Regular',
      'License: public domain / CC0 1.0',
      'https://creativecommons.org/publicdomain/zero/1.0/',
      '',
    ].join('\n'),
    'utf8',
  );

  console.log(`Generated ${sourceEntries.length} face tile SVGs + back.svg in ${outputDir}`);
}

await main();
