import { copyFile, mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const appDirectory = dirname(scriptDirectory);
const repositoryRoot = resolve(appDirectory, '..', '..', '..');
const publicDirectory = join(appDirectory, 'public');
const modelDirectory = join(publicDirectory, 'models');

const defaultModelPath = join(
  repositoryRoot,
  '.local',
  'recognition',
  'nanodet_runs',
  'E1_plus_m_320_composite_augmented_amp40_seed42',
  'model_best',
  'nanodet-plus-m-320-composite-augmented.onnx',
);
const sourceModelPath = process.env.MJTENSU_CAPTURE_MODEL
  ? resolve(process.env.MJTENSU_CAPTURE_MODEL)
  : defaultModelPath;
const outputModelName = 'nanodet-plus-m-320-composite-augmented.onnx';
const outputModelPath = join(modelDirectory, outputModelName);
const layoutSource = join(repositoryRoot, 'tools', 'recognition', 'capture_layout.v1.json');

await mkdir(modelDirectory, { recursive: true });
await copyFile(sourceModelPath, outputModelPath);
await copyFile(layoutSource, join(publicDirectory, 'capture_layout.v1.json'));

const modelBytes = await readFile(sourceModelPath);
const modelStat = await stat(sourceModelPath);
const metadata = {
  name: outputModelName,
  source: sourceModelPath,
  sizeBytes: modelStat.size,
  sha256: createHash('sha256').update(modelBytes).digest('hex'),
  inputShape: [1, 3, 320, 320],
  outputShape: [1, 2125, 33],
  generatedAt: new Date().toISOString(),
};
await writeFile(
  join(modelDirectory, 'nanodet-plus-m-320-composite-augmented.metadata.json'),
  `${JSON.stringify(metadata, null, 2)}\n`,
  'utf8',
);

console.log(`Prepared capture model: ${outputModelPath}`);
