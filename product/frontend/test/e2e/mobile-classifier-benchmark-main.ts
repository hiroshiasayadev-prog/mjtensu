import * as ort from 'onnxruntime-web/wasm';

const MODELS = [
  {
    id: 'plain',
    label: 'Plain e150',
    path: 'tile-plain-gray35-random360-e150.onnx',
  },
  {
    id: 'mobile-standard',
    label: 'MobileNetV3 standard',
    path: 'tile-mobilenet-v3-small-1.0x-random360-e150.onnx',
  },
  {
    id: 'f8-r1',
    label: 'Mobile tile f8-r1',
    path: 'mobile-tile-f8-r1.onnx',
  },
  {
    id: 'f8-r2',
    label: 'Mobile tile f8-r2',
    path: 'mobile-tile-f8-r2.onnx',
  },
] as const;

const BATCH_SIZES = [1, 4, 8, 16, 24] as const;
const WARMUP_RUNS = 10;
const MEASUREMENT_RUNS = 50;
const IMAGE_SIZE = 64;
const CLASS_COUNT = 35;

type ModelId = (typeof MODELS)[number]['id'];

interface TimingSummary {
  readonly medianMs: number;
  readonly p95Ms: number;
  readonly meanMs: number;
  readonly msPerImageMedian: number;
}

interface ModelBenchmark {
  readonly model: ModelId;
  readonly label: string;
  readonly path: string;
  readonly inputName: string;
  readonly outputName: string;
  readonly batches: Readonly<Record<string, TimingSummary>>;
}

interface BenchmarkReport {
  readonly provider: 'wasm-simd';
  readonly numThreads: 1;
  readonly wasmProxy: false;
  readonly hardwareConcurrency: number;
  readonly crossOriginIsolated: boolean;
  readonly secureContext: boolean;
  readonly userAgent: string;
  readonly warmupRuns: number;
  readonly measurementRuns: number;
  readonly batchSizes: readonly number[];
  readonly models: readonly ModelBenchmark[];
}

const environmentNode = requireElement('environment');
const runButton = requireElement('run') as HTMLButtonElement;
const statusNode = requireElement('status');
const resultsNode = requireElement('results');
const jsonNode = requireElement('json');

configureOrt();
environmentNode.textContent = [
  'provider=wasm-simd',
  'numThreads=1',
  `hardwareConcurrency=${navigator.hardwareConcurrency || 1}`,
  `crossOriginIsolated=${globalThis.crossOriginIsolated === true}`,
  `secureContext=${globalThis.isSecureContext === true}`,
].join(' | ');

runButton.addEventListener('click', () => {
  void runBenchmark();
});

async function runBenchmark(): Promise<void> {
  runButton.disabled = true;
  resultsNode.replaceChildren();
  jsonNode.textContent = '';
  const sessions: Array<{
    model: ModelId;
    label: string;
    path: string;
    session: ort.InferenceSession;
  }> = [];
  try {
    for (const descriptor of MODELS) {
      statusNode.textContent = `loading ${descriptor.label}...`;
      sessions.push({
        model: descriptor.id,
        label: descriptor.label,
        path: descriptor.path,
        session: await createSession(descriptor.path),
      });
    }

    const models: ModelBenchmark[] = [];
    for (const item of sessions) {
      statusNode.textContent = `benchmarking ${item.label}...`;
      models.push(
        await benchmarkSession(
          item.model,
          item.label,
          item.path,
          item.session,
        ),
      );
    }

    const report: BenchmarkReport = {
      provider: 'wasm-simd',
      numThreads: 1,
      wasmProxy: false,
      hardwareConcurrency: navigator.hardwareConcurrency || 1,
      crossOriginIsolated: globalThis.crossOriginIsolated === true,
      secureContext: globalThis.isSecureContext === true,
      userAgent: navigator.userAgent,
      warmupRuns: WARMUP_RUNS,
      measurementRuns: MEASUREMENT_RUNS,
      batchSizes: [...BATCH_SIZES],
      models,
    };
    renderResults(report);
    jsonNode.textContent = JSON.stringify(report, null, 2);
    statusNode.textContent = 'completed';
  } catch (error) {
    statusNode.textContent = `failed: ${error instanceof Error ? error.message : String(error)}`;
    throw error;
  } finally {
    await Promise.allSettled(sessions.map(({ session }) => session.release()));
    runButton.disabled = false;
  }
}

function configureOrt(): void {
  ort.env.logLevel = 'warning';
  ort.env.wasm.proxy = false;
  ort.env.wasm.simd = true;
  ort.env.wasm.numThreads = 1;
}

async function createSession(path: string): Promise<ort.InferenceSession> {
  const url = `${import.meta.env.BASE_URL}${path}`;
  return ort.InferenceSession.create(url, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    executionMode: 'sequential',
  });
}

async function benchmarkSession(
  model: ModelId,
  label: string,
  path: string,
  session: ort.InferenceSession,
): Promise<ModelBenchmark> {
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  if (inputName === undefined || outputName === undefined) {
    throw new Error(`${label} has no input/output metadata`);
  }

  const batches: Record<string, TimingSummary> = {};
  for (const batchSize of BATCH_SIZES) {
    const tensor = makeInputTensor(batchSize);
    const feeds = { [inputName]: tensor };
    for (let index = 0; index < WARMUP_RUNS; index += 1) {
      const output = await session.run(feeds);
      validateOutput(label, outputName, output, batchSize);
    }

    const samples: number[] = [];
    for (let index = 0; index < MEASUREMENT_RUNS; index += 1) {
      const started = performance.now();
      const output = await session.run(feeds);
      samples.push(performance.now() - started);
      validateOutput(label, outputName, output, batchSize);
    }
    batches[String(batchSize)] = summarize(samples, batchSize);
    statusNode.textContent = `benchmarking ${label}: N=${batchSize} complete`;
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  }

  return { model, label, path, inputName, outputName, batches };
}

function validateOutput(
  label: string,
  outputName: string,
  output: ort.InferenceSession.OnnxValueMapType,
  batchSize: number,
): void {
  const logits = output[outputName];
  if (logits === undefined) {
    throw new Error(`${label} did not return ${outputName}`);
  }
  if (logits.dims[0] !== batchSize || logits.dims[1] !== CLASS_COUNT) {
    throw new Error(
      `${label} output shape ${logits.dims.join('x')} != ${batchSize}x${CLASS_COUNT}`,
    );
  }
}

function makeInputTensor(batchSize: number): ort.Tensor {
  const elements = batchSize * IMAGE_SIZE * IMAGE_SIZE;
  const values = new Float32Array(elements);
  for (let index = 0; index < elements; index += 1) {
    values[index] =
      Math.sin(index * 0.013) * 0.75 + Math.cos(index * 0.007) * 0.25;
  }
  return new ort.Tensor('float32', values, [
    batchSize,
    1,
    IMAGE_SIZE,
    IMAGE_SIZE,
  ]);
}

function summarize(
  samples: readonly number[],
  batchSize: number,
): TimingSummary {
  const ordered = [...samples].sort((left, right) => left - right);
  const medianMs = percentile(ordered, 0.5);
  const p95Ms = percentile(ordered, 0.95);
  const meanMs =
    samples.reduce((sum, value) => sum + value, 0) / samples.length;
  return {
    medianMs,
    p95Ms,
    meanMs,
    msPerImageMedian: medianMs / batchSize,
  };
}

function percentile(ordered: readonly number[], fraction: number): number {
  if (ordered.length === 0) {
    throw new Error('No timing samples');
  }
  const index = Math.min(
    ordered.length - 1,
    Math.max(0, Math.ceil(ordered.length * fraction) - 1),
  );
  return ordered[index] ?? ordered[ordered.length - 1] ?? 0;
}

function renderResults(report: BenchmarkReport): void {
  const plain = report.models.find((model) => model.model === 'plain');
  if (plain === undefined) {
    throw new Error('Plain baseline result is missing');
  }

  const table = document.createElement('table');
  const header = document.createElement('tr');
  for (const heading of [
    'model',
    'N',
    'median ms',
    'p95 ms',
    'median ms/image',
    'speedup vs Plain',
  ]) {
    const cell = document.createElement('th');
    cell.textContent = heading;
    header.append(cell);
  }
  const thead = document.createElement('thead');
  thead.append(header);
  table.append(thead);

  const body = document.createElement('tbody');
  for (const model of report.models) {
    for (const batchSize of report.batchSizes) {
      const timing = model.batches[String(batchSize)];
      const plainTiming = plain.batches[String(batchSize)];
      if (timing === undefined || plainTiming === undefined) {
        continue;
      }
      const row = document.createElement('tr');
      appendCell(row, model.label);
      appendCell(row, String(batchSize));
      appendCell(row, timing.medianMs.toFixed(3));
      appendCell(row, timing.p95Ms.toFixed(3));
      appendCell(row, timing.msPerImageMedian.toFixed(3));
      appendCell(row, `${(plainTiming.medianMs / timing.medianMs).toFixed(2)}x`);
      body.append(row);
    }
  }
  table.append(body);
  resultsNode.replaceChildren(table);
}

function appendCell(row: HTMLTableRowElement, value: string): void {
  const cell = document.createElement('td');
  cell.textContent = value;
  row.append(cell);
}

function requireElement(id: string): HTMLElement {
  const element = document.getElementById(id);
  if (element === null) {
    throw new Error(`Missing #${id}`);
  }
  return element;
}
