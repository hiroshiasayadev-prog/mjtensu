#!/usr/bin/env node

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const [, , inputArg, outputArg] = process.argv;

if (!inputArg) {
  console.error(
    'Usage: node scripts/extract-recognition-debug.mjs <debug.json> [output-dir]',
  );
  process.exit(1);
}

const inputPath = path.resolve(inputArg);
const outputDir = outputArg
  ? path.resolve(outputArg)
  : path.join(
      path.dirname(inputPath),
      `${path.basename(inputPath, path.extname(inputPath))}-extracted`,
    );

const raw = await readFile(inputPath, 'utf8');
const payload = JSON.parse(raw);

await mkdir(outputDir, { recursive: true });
await mkdir(path.join(outputDir, 'regions'), { recursive: true });
await mkdir(path.join(outputDir, 'tensors'), { recursive: true });

await extractDataUrlPng(
  payload?.capture?.images?.sourcePngDataUrl,
  path.join(outputDir, 'source.png'),
);
await extractDataUrlPng(
  payload?.capture?.images?.compositePngDataUrl,
  path.join(outputDir, 'composite.png'),
);

for (const [region, dataUrl] of Object.entries(
  payload?.capture?.images?.regionPngDataUrls ?? {},
)) {
  await extractDataUrlPng(
    dataUrl,
    path.join(outputDir, 'regions', `${safeName(region)}.png`),
  );
}

const capture = payload?.capture ?? {};
const snapshot = capture.snapshot ?? null;
const detections = Array.isArray(capture.detections) ? capture.detections : [];

await writeJson(path.join(outputDir, 'snapshot.json'), snapshot);
await writeJson(path.join(outputDir, 'detections.json'), detections);
await writeJson(path.join(outputDir, 'metadata.json'), {
  schemaVersion: payload?.schemaVersion ?? null,
  capture: {
    schemaVersion: capture.schemaVersion ?? null,
    createdAtIso: capture.createdAtIso ?? null,
    capturedAtMs: capture.capturedAtMs ?? null,
    modelSetVersion: capture.modelSetVersion ?? null,
    sourceSize: capture.sourceSize ?? null,
    regions: capture.regions ?? null,
    sourceRegionRects: capture.sourceRegionRects ?? null,
  },
  runtimeDiagnostics: payload?.runtimeDiagnostics ?? null,
  environment: payload?.environment ?? null,
});

await writeDetectionsCsv(
  path.join(outputDir, 'detections.csv'),
  detections,
);

await writeJson(
  path.join(outputDir, 'meld-analysis.json'),
  buildMeldAnalysis(snapshot),
);

await extractTensor(
  capture.detectorInput,
  path.join(outputDir, 'tensors', 'detector-input.f32'),
  path.join(outputDir, 'tensors', 'detector-input.json'),
);
await extractTensor(
  capture.detectorOutput,
  path.join(outputDir, 'tensors', 'detector-output.f32'),
  path.join(outputDir, 'tensors', 'detector-output.json'),
);

console.log(`Extracted recognition debug capture to:\n${outputDir}`);
console.log('Useful files for meld diagnosis:');
console.log(`  ${path.join(outputDir, 'source.png')}`);
console.log(`  ${path.join(outputDir, 'regions', 'melds.png')}`);
console.log(`  ${path.join(outputDir, 'meld-analysis.json')}`);
console.log(`  ${path.join(outputDir, 'detections.csv')}`);

async function extractDataUrlPng(dataUrl, outputPath) {
  if (typeof dataUrl !== 'string') {
    return;
  }

  const match = /^data:image\/png;base64,(.+)$/s.exec(dataUrl);
  if (!match) {
    throw new Error(`Expected PNG data URL for ${outputPath}`);
  }

  await writeFile(outputPath, Buffer.from(match[1], 'base64'));
}

async function extractTensor(tensor, binaryPath, metadataPath) {
  if (
    tensor === null ||
    typeof tensor !== 'object' ||
    tensor.encoding !== 'base64-f32-le' ||
    typeof tensor.data !== 'string'
  ) {
    return;
  }

  await writeFile(binaryPath, Buffer.from(tensor.data, 'base64'));
  await writeJson(metadataPath, {
    shape: tensor.shape ?? tensor.dims ?? null,
    dims: tensor.dims ?? tensor.shape ?? null,
    type: tensor.type ?? 'float32',
    encoding: tensor.encoding,
  });
}

function buildMeldAnalysis(snapshotValue) {
  if (snapshotValue === null || typeof snapshotValue !== 'object') {
    return null;
  }

  const observations = Array.isArray(snapshotValue.observations)
    ? snapshotValue.observations
    : [];

  const meldObservations = observations
    .filter((observation) => observation?.region === 'melds')
    .map((observation) => {
      const bbox = observation?.bbox ?? null;
      return {
        id: observation?.id ?? null,
        classification: observation?.classification ?? null,
        bbox,
        center:
          bbox &&
          Number.isFinite(bbox.x) &&
          Number.isFinite(bbox.y) &&
          Number.isFinite(bbox.width) &&
          Number.isFinite(bbox.height)
            ? {
                x: bbox.x + bbox.width / 2,
                y: bbox.y + bbox.height / 2,
              }
            : null,
      };
    });

  return {
    commitEligibility: snapshotValue.commitEligibility ?? null,
    meldGroups: snapshotValue.meldGroups ?? [],
    draftMeldGroups: snapshotValue.draft?.meldGroups ?? [],
    meldObservations,
  };
}

async function writeDetectionsCsv(outputPath, values) {
  const rows = [
    [
      'id',
      'detectionIndex',
      'confidence',
      'region',
      'classificationKind',
      'tileKind',
      'red',
      'sourceX',
      'sourceY',
      'sourceWidth',
      'sourceHeight',
      'compositeX',
      'compositeY',
      'compositeWidth',
      'compositeHeight',
    ],
  ];

  for (const detection of values) {
    const classification = detection?.classification ?? {};
    rows.push([
      detection?.id ?? '',
      detection?.detectionIndex ?? '',
      detection?.confidence ?? '',
      detection?.region ?? '',
      classification?.kind ?? '',
      classification?.tile?.kind ?? '',
      classification?.tile?.red ?? '',
      detection?.sourceBox?.x ?? '',
      detection?.sourceBox?.y ?? '',
      detection?.sourceBox?.width ?? '',
      detection?.sourceBox?.height ?? '',
      detection?.compositeBox?.x ?? '',
      detection?.compositeBox?.y ?? '',
      detection?.compositeBox?.width ?? '',
      detection?.compositeBox?.height ?? '',
    ]);
  }

  const csv = rows
    .map((row) => row.map(csvCell).join(','))
    .join('\n');
  await writeFile(outputPath, `${csv}\n`, 'utf8');
}

function csvCell(value) {
  const text = String(value);
  if (!/[",\n\r]/.test(text)) {
    return text;
  }
  return `"${text.replaceAll('"', '""')}"`;
}

function safeName(value) {
  return String(value).replaceAll(/[^a-zA-Z0-9._-]/g, '_');
}

async function writeJson(outputPath, value) {
  await writeFile(outputPath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}
