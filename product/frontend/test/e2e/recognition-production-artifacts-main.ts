import {
  FIXED_COMPOSITE_LAYOUT,
  PRODUCTION_RECOGNITION_MODEL_SET,
  createBrowserRecognitionModelAssets,
  createProductionRecognitionRuntime,
  type RecognitionFrame,
} from '@/recognition';
import {
  BASE_CLASSIFIER_LABELS,
  RED_FIVE_CLASSIFIER_LABELS,
} from '@/recognition/classifier/labels';
import {
  makeBaseClassifierTensor,
  makeRedFiveClassifierTensor,
  type ClassifierCropImage,
  type ClassifierTensor,
} from '@/recognition/classifier/preprocessing';
import { createOnnxRecognitionSessionFactory } from '@/recognition/model-runtime/onnx-session-factory';
import { createRecognitionModelRuntime } from '@/recognition/model-runtime/runtime';
import { getRecognitionClassifierNormalization } from '@/recognition/model-runtime/runtime-specs';

interface ProductionArtifactDiagnostics {
  status: 'running' | 'ready' | 'failed';
  phase?: string;
  modelSetVersion?: string;
  providers?: readonly {
    role: string;
    runtimeSpec: string;
    selectedProvider?: string;
    failedProviders: readonly string[];
  }[];
  baseFixture?: {
    label: string;
    logits: readonly number[];
  };
  redFiveFixture?: {
    label: string;
    logits: readonly number[];
  };
  blankFrameSnapshot?: unknown;
  error?: string;
}

declare global {
  interface Window {
    __MJTENSU_RECOGNITION_ARTIFACTS__: ProductionArtifactDiagnostics;
  }
}

const diagnostics: ProductionArtifactDiagnostics = { status: 'running' };
window.__MJTENSU_RECOGNITION_ARTIFACTS__ = diagnostics;
const statusElement = document.getElementById('status');

void runVerification();

async function runVerification(): Promise<void> {
  const productionRuntime = createProductionRecognitionRuntime({
    manifest: PRODUCTION_RECOGNITION_MODEL_SET,
  });
  const inspectionRuntime = createRecognitionModelRuntime({
    manifest: PRODUCTION_RECOGNITION_MODEL_SET,
    assets: createBrowserRecognitionModelAssets(),
    sessions: createOnnxRecognitionSessionFactory(),
  });

  try {
    diagnostics.phase = 'production-runtime-initialize';
    await productionRuntime.initialize();
    diagnostics.phase = 'blank-frame-pipeline';
    const pipeline = productionRuntime.createPipeline();
    const blankFrameSnapshot = await pipeline.evaluate(createBlankFrame());
    await pipeline.dispose();

    diagnostics.phase = 'inspection-runtime-initialize';
    await inspectionRuntime.initialize();
    const baseNormalization = getRecognitionClassifierNormalization('c8-tile-35-v1');
    const redFiveNormalization = getRecognitionClassifierNormalization('c8-red-five-v1');
    if (baseNormalization === null || redFiveNormalization === null) {
      throw new Error('Classifier runtime normalization is unavailable.');
    }

    diagnostics.phase = 'classifier-fixtures';
    const crop = createDeterministicClassifierCrop();
    const baseLogits = await runClassifierFixture(
      inspectionRuntime.getInitializedModel('tile-classifier').session,
      makeBaseClassifierTensor(crop, baseNormalization),
      BASE_CLASSIFIER_LABELS.length,
    );
    const redFiveLogits = await runClassifierFixture(
      inspectionRuntime.getInitializedModel('red-five-classifier').session,
      makeRedFiveClassifierTensor(crop, redFiveNormalization),
      RED_FIVE_CLASSIFIER_LABELS.length,
    );

    Object.assign(diagnostics, {
      status: 'ready' as const,
      phase: 'complete',
      modelSetVersion: PRODUCTION_RECOGNITION_MODEL_SET.modelSetVersion,
      providers: inspectionRuntime.getDiagnostics(),
      baseFixture: {
        label: BASE_CLASSIFIER_LABELS[argmax(baseLogits)] ?? 'unknown',
        logits: rounded(baseLogits),
      },
      redFiveFixture: {
        label: RED_FIVE_CLASSIFIER_LABELS[argmax(redFiveLogits)] ?? 'unknown',
        logits: rounded(redFiveLogits),
      },
      blankFrameSnapshot,
    });
    if (statusElement !== null) {
      statusElement.textContent = 'ready';
    }
  } catch (error) {
    Object.assign(diagnostics, {
      status: 'failed' as const,
      error: formatError(error),
    });
    if (statusElement !== null) {
      statusElement.textContent = `failed: ${diagnostics.error ?? 'unknown error'}`;
    }
  } finally {
    await Promise.allSettled([
      productionRuntime.dispose(),
      inspectionRuntime.dispose(),
    ]);
  }
}

function createBlankFrame(): RecognitionFrame {
  const canvas = document.createElement('canvas');
  canvas.width = FIXED_COMPOSITE_LAYOUT.width;
  canvas.height = FIXED_COMPOSITE_LAYOUT.height;
  const context = canvas.getContext('2d', { alpha: false });
  if (context === null) {
    throw new Error('Blank fixture canvas context is unavailable.');
  }
  context.fillStyle = 'rgb(0, 0, 0)';
  context.fillRect(0, 0, canvas.width, canvas.height);

  const normalized = (region: {
    readonly x: number;
    readonly y: number;
    readonly width: number;
    readonly height: number;
  }) => ({
    x: region.x / canvas.width,
    y: region.y / canvas.height,
    width: region.width / canvas.width,
    height: region.height / canvas.height,
  });

  return {
    source: canvas,
    sourceSize: { width: canvas.width, height: canvas.height },
    regions: {
      'completed-hand': normalized(FIXED_COMPOSITE_LAYOUT.regions.completed_hand),
      'dora-indicators': normalized(FIXED_COMPOSITE_LAYOUT.regions.dora_indicators),
      melds: normalized(FIXED_COMPOSITE_LAYOUT.regions.melds),
    },
    capturedAtMs: 1,
  };
}

function createDeterministicClassifierCrop(): ClassifierCropImage {
  const size = 64;
  const data = new Uint8Array(size * size * 3);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const offset = (y * size + x) * 3;
      data[offset] = (x * 4) & 0xff;
      data[offset + 1] = (y * 4) & 0xff;
      data[offset + 2] = ((x + y) * 2) & 0xff;
    }
  }
  return { width: size, height: size, channels: 3, data };
}

async function runClassifierFixture(
  session: {
    readonly inputNames: readonly string[];
    readonly outputNames: readonly string[];
    createFloat32Tensor(data: Float32Array, dims: readonly number[]): unknown;
    run(feeds: Readonly<Record<string, unknown>>): Promise<Readonly<Record<string, unknown>>>;
  },
  input: ClassifierTensor,
  expectedLogits: number,
): Promise<Float32Array> {
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  if (inputName === undefined || outputName === undefined) {
    throw new Error('Classifier ONNX input/output names are missing.');
  }
  const tensor = session.createFloat32Tensor(input.data, input.shape);
  const outputs = await session.run({ [inputName]: tensor });
  const output = outputs[outputName];
  if (
    typeof output !== 'object' ||
    output === null ||
    !('data' in output) ||
    !(output.data instanceof Float32Array) ||
    output.data.length !== expectedLogits
  ) {
    throw new Error(`Classifier ONNX output is incompatible; expected ${expectedLogits} logits.`);
  }
  return output.data;
}

function argmax(values: Float32Array): number {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index] ?? Number.NEGATIVE_INFINITY;
    if (value > bestValue) {
      bestIndex = index;
      bestValue = value;
    }
  }
  return bestIndex;
}

function rounded(values: Float32Array): readonly number[] {
  return Array.from(values, (value) => Number(value.toFixed(6)));
}

function formatError(error: unknown): string {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`;
  }
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}
