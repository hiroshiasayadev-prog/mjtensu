import * as ort from 'onnxruntime-web/wasm';

const MODEL_PATHS = {
  plain: 'tile-plain-gray35-random360-e150.onnx',
  mobile: 'tile-mobilenet-v3-small-1.0x-random360-e150.onnx',
} as const;

const BATCH_SIZES = [1, 4, 8, 16, 24] as const;
const WARMUP_RUNS = 10;
const MEASUREMENT_RUNS = 50;
const IMAGE_SIZE = 64;
const CLASS_COUNT = 35;

interface TimingSummary {
  readonly medianMs: number;
  readonly p95Ms: number;
  readonly meanMs: number;
  readonly msPerImageMedian: number;
}

interface ModelBenchmark {
  readonly model: keyof typeof MODEL_PATHS;
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
].join(' | ');

runButton.addEventListener('click', () => {
  void runBenchmark();
});

async function runBenchmark(): Promise<void> {
  runButton.disabled = true;
  resultsNode.replaceChildren();
  jsonNode.textContent = '';
  try {
    statusNode.textContent = 'loading models...';
    const sessions = await Promise.all(
      (Object.entries(MODEL_PATHS) as Array<[
        keyof typeof MODEL_PATHS,
        string,
      ]>).map(async ([model, path]) => ({
        model,
        path,
        session: await createSession(path),
      })),
    );

    const models: ModelBenchmark[] = [];
    for (const item of sessions) {
      statusNode.textContent = `benchmarking ${item.model}...`;
      models.push(
        await benchmarkSession(item.model, item.path, item.session),
      );
    }

    const report: BenchmarkReport = {
      provider: 'wasm-simd',
      numThreads: 1,
      wasmProxy: false,
      hardwareConcurrency: navigator.hardwareConcurrency || 1,
      crossOriginIsolated: globalThis.crossOriginIsolated === true,
      userAgent: navigator.userAgent,
      warmupRuns: WARMUP_RUNS,
      measurementRuns: MEASUREMENT_RUNS,
      batchSizes: [...BATCH_SIZES],
      models,
    };
    renderResults(report);
    jsonNode.textContent = JSON.stringify(report, null, 2);
    statusNode.textContent = 'completed';

    await Promise.allSettled(sessions.map(({ session }) => session.release()));
  } catch (error) {
    statusNode.textContent = `failed: ${error instanceof Error ? error.message : String(error)}`;
    throw error;
  } finally {
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
  model: keyof typeof MODEL_PATHS,
  path: string,
  session: ort.InferenceSession,
): Promise<ModelBenchmark> {
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  if (inputName === undefined || outputName === undefined) {
    throw new Error(`${model} has no input/output metadata`);
  }

  const batches: Record<string, TimingSummary> = {};
  for (const batchSize of BATCH_SIZES) {
    const tensor = makeInputTensor(batchSize);
    const feeds = { [inputName]: tensor };
    for (let index = 0; index < WARMUP_RUNS; index += 1) {
      await session.run(feeds);
    }

    const samples: number[] = [];
    for (let index = 0; index < MEASUREMENT_RUNS; index += 1) {
      const started = performance.now();
      const output = await session.run(feeds);
      samples.push(performance.now() - started);
      const logits = output[outputName];
      if (logits === undefined) {
        throw new Error(`${model} did not return ${outputName}`);
      }
      if (logits.dims[0] !== batchSize || logits.dims[1] !== CLASS_COUNT) {
        throw new Error(
          `${model} output shape ${logits.dims.join('x')} != ${batchSize}x${CLASS_COUNT}`,
        );
      }
    }
    batches[String(batchSize)] = summarize(samples, batchSize);
    statusNode.textContent = `benchmarking ${model}: N=${batchSize} complete`;
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  }

  return { model, path, inputName, outputName, batches };
}

function makeInputTensor(batchSize: number): ort.Tensor {
  const elements = batchSize * IMAGE_SIZE * IMAGE_SIZE;
  const values = new Float32Array(elements);
  for (let index = 0; index < elements; index += 1) {
    values[index] = Math.sin(index * 0.013) * 0.75 + Math.cos(index * 0.007) * 0.25;
  }
  return new ort.Tensor('float32', values, [batchSize, 1, IMAGE_SIZE, IMAGE_SIZE]);
}

function summarize(samples: readonly number[], batchSize: number): TimingSummary {
  const ordered = [...samples].sort((left, right) => left - right);
  const medianMs = percentile(ordered, 0.5);
  const p95Ms = percentile(ordered, 0.95);
  const meanMs = samples.reduce((sum, value) => sum + value, 0) / samples.length;
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
  const table = document.createElement('table');
  const header = document.createElement('tr');
  for (const heading of ['model', 'N', 'median ms', 'p95 ms', 'median ms/image']) {
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
      if (timing === undefined) {
        continue;
      }
      const row = document.createElement('tr');
      appendCell(row, model.model);
      appendCell(row, String(batchSize));
      appendCell(row, timing.medianMs.toFixed(3));
      appendCell(row, timing.p95Ms.toFixed(3));
      appendCell(row, timing.msPerImageMedian.toFixed(3));
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
