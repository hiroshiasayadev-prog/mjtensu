import type { TileClassification } from './classifier/labels';
import type {
  ClassifierCropImage,
  ClassifierNormalization,
  ClassifierTensor,
} from './classifier/preprocessing';
import {
  createC8ClassifierRuntime,
  type ClassifierSession,
} from './classifier/runtime';
import {
  createProductionNanoDetPostprocessor,
  type NanoDetDetectionPostprocessor,
} from './detector/detection-postprocessor';
import { buildFixedComposite } from './detector/fixed-composite';
import {
  NANODET_INPUT_SIZE,
  preprocessCompositeCanvas,
} from './detector/nanodet';
import type {
  CaptureRegions,
  Rect,
  RegionDetection,
  SemanticRegion,
  TensorOutput,
} from './detector/types';
import type {
  RecognitionDebugCapture,
  RecognitionEvaluationTiming,
  RecognitionFrame,
  RecognitionPipeline,
} from './contracts';
import type {
  RecognitionInferenceSession,
  RecognitionModelRuntimeInspection,
} from './model-runtime/runtime';
import { getRecognitionClassifierNormalization } from './model-runtime/runtime-specs';
import {
  isRecognitionRuntimeError,
  type RecognitionModelRole,
  type RecognitionModelRuntimeSpec,
  type RecognitionRuntimeError,
} from './model-runtime/types';
import { buildFrameRecognitionSnapshot } from './semantics/frame-semantics';
import type {
  ClassifiedRecognitionCandidate,
  FrameRecognitionSnapshot,
  NormalizedRect,
  RecognitionRegion,
} from './semantics/types';

export interface ProductionClassifierNormalization {
  readonly base: ClassifierNormalization;
  readonly redFive: ClassifierNormalization;
}

export interface RecognitionPipelinePlatform {
  buildComposite(frame: RecognitionFrame, regions: CaptureRegions): HTMLCanvasElement;
  preprocessComposite(composite: HTMLCanvasElement): Float32Array;
  extractCrop(frame: RecognitionFrame, box: Rect): ClassifierCropImage;
}

export interface RecognitionDebugCaptureBuilderInput {
  readonly frame: RecognitionFrame;
  readonly composite: HTMLCanvasElement;
  readonly captureRegions: CaptureRegions;
  readonly detectorInput: Float32Array;
  readonly detectorOutput: TensorOutput;
  readonly detections: readonly RegionDetection[];
  readonly classifications: readonly TileClassification[];
  readonly snapshot: FrameRecognitionSnapshot;
  readonly modelSetVersion?: string;
}

export type RecognitionDebugCaptureBuilder = (
  input: RecognitionDebugCaptureBuilderInput,
) => RecognitionDebugCapture;

export interface ProductionRecognitionPipelineDependencies {
  readonly modelRuntime: RecognitionModelRuntimeInspection;
  /** Test-only/internal seam. Production composition resolves this from runtimeSpec. */
  readonly classifierNormalizationOverride?: ProductionClassifierNormalization;
  readonly detectorPostprocessor?: NanoDetDetectionPostprocessor;
  readonly platform?: RecognitionPipelinePlatform;
  readonly now?: () => number;
  readonly onEvaluationTiming?: (timing: RecognitionEvaluationTiming) => void;
  readonly modelSetVersion?: string;
  readonly claimDebugCapture?: () => boolean;
  readonly onDebugCapture?: (capture: RecognitionDebugCapture) => void;
  readonly onDebugCaptureFailure?: (error: unknown) => void;
  readonly debugCaptureBuilder?: RecognitionDebugCaptureBuilder;
}

export function createProductionRecognitionPipeline(
  dependencies: ProductionRecognitionPipelineDependencies,
): RecognitionPipeline {
  const detectorModel = dependencies.modelRuntime.getInitializedModel('detector');
  const baseClassifierModel = dependencies.modelRuntime.getInitializedModel('tile-classifier');
  const redFiveClassifierModel = dependencies.modelRuntime.getInitializedModel(
    'red-five-classifier',
  );
  const detector = detectorModel.session;
  const baseClassifier = baseClassifierModel.session;
  const redFiveClassifier = redFiveClassifierModel.session;
  const classifierNormalization =
    dependencies.classifierNormalizationOverride ??
    resolveClassifierNormalization(
      baseClassifierModel.runtimeSpec,
      redFiveClassifierModel.runtimeSpec,
    );
  const detectorPostprocessor =
    dependencies.detectorPostprocessor ?? createProductionNanoDetPostprocessor();
  const platform = dependencies.platform ?? browserPipelinePlatform;
  const now = dependencies.now ?? monotonicNow;
  const debugCaptureBuilder =
    dependencies.debugCaptureBuilder ?? buildBrowserRecognitionDebugCapture;
  const classifier = createC8ClassifierRuntime({
    baseClassifier: createClassifierSessionAdapter(
      baseClassifier,
      'tile-classifier',
      35,
    ),
    redFiveClassifier: createClassifierSessionAdapter(
      redFiveClassifier,
      'red-five-classifier',
      2,
    ),
    baseNormalization: classifierNormalization.base,
    redFiveNormalization: classifierNormalization.redFive,
    now,
  });

  const inFlight = new Set<Promise<FrameRecognitionSnapshot>>();
  let disposalRequested = false;
  let disposalInFlight: Promise<void> | null = null;

  const evaluateFrame = async (
    frame: RecognitionFrame,
    debugRequested: boolean,
  ): Promise<FrameRecognitionSnapshot> => {
    validateFrame(frame);
    const evaluationStartedAt = now();

    const captureRegions = toCaptureRegions(frame);
    const detectorPreprocessingStartedAt = now();
    let detectorInput: Float32Array;
    let composite: HTMLCanvasElement;
    try {
      composite = platform.buildComposite(frame, captureRegions);
      detectorInput = platform.preprocessComposite(composite);
    } catch (error) {
      if (isRecognitionRuntimeError(error)) {
        throw error;
      }
      throw inferenceFailure('detector', error);
    }
    const detectorPreprocessingFinishedAt = now();

    const detectorInferenceStartedAt = now();
    const detectorOutput = await runDetector(detector, detectorInput);
    const detectorInferenceFinishedAt = now();

    const detectorPostprocessingStartedAt = now();
    let detections: readonly RegionDetection[];
    try {
      detections = detectorPostprocessor.process(
        detectorOutput,
        captureRegions,
      );
    } catch (error) {
      if (isRecognitionRuntimeError(error)) {
        throw error;
      }
      throw modelIncompatible('detector');
    }
    const detectorPostprocessingFinishedAt = now();

    const cropExtractionStartedAt = now();
    let crops: readonly ClassifierCropImage[];
    try {
      crops = detections.map((detection) =>
        platform.extractCrop(frame, detection.sourceBox),
      );
    } catch (error) {
      if (isRecognitionRuntimeError(error)) {
        throw error;
      }
      throw inferenceFailure('tile-classifier', error);
    }
    const cropExtractionFinishedAt = now();

    let classifications: readonly TileClassification[];
    try {
      classifications = await classifier.classifyBatch(crops);
    } catch (error) {
      if (isRecognitionRuntimeError(error)) {
        throw error;
      }
      throw inferenceFailure('tile-classifier', error);
    }
    if (classifications.length !== detections.length) {
      throw modelIncompatible('tile-classifier');
    }

    const candidates: ClassifiedRecognitionCandidate[] = detections.map(
      (detection, index) => {
        const classification = classifications[index];
        if (classification === undefined) {
          throw modelIncompatible('tile-classifier');
        }
        return {
          id: detection.id,
          region: toRecognitionRegion(detection.region),
          bbox: normalizeRect(detection.sourceBox, frame),
          classification,
        };
      },
    );
    const snapshot = buildFrameRecognitionSnapshot(candidates);
    const classifierTiming = classifier.getLastBatchTiming();
    const evaluationFinishedAt = now();

    if (classifierTiming !== null) {
      reportEvaluationTiming(dependencies.onEvaluationTiming, {
        totalMs: evaluationFinishedAt - evaluationStartedAt,
        candidateCount: classifierTiming.candidateCount,
        redFiveCandidateCount: classifierTiming.redFiveCandidateCount,
        detectorPreprocessingMs:
          detectorPreprocessingFinishedAt - detectorPreprocessingStartedAt,
        detectorInferenceMs:
          detectorInferenceFinishedAt - detectorInferenceStartedAt,
        detectorPostprocessingMs:
          detectorPostprocessingFinishedAt - detectorPostprocessingStartedAt,
        cropExtractionMs: cropExtractionFinishedAt - cropExtractionStartedAt,
        baseClassifierPreprocessingMs: classifierTiming.basePreprocessingMs,
        baseClassifierInferenceMs: classifierTiming.baseInferenceMs,
        redFiveClassifierPreprocessingMs: classifierTiming.redFivePreprocessingMs,
        redFiveClassifierInferenceMs: classifierTiming.redFiveInferenceMs,
      });
    }

    if (debugRequested) {
      try {
        dependencies.onDebugCapture?.(
          debugCaptureBuilder({
            frame,
            composite,
            captureRegions,
            detectorInput,
            detectorOutput,
            detections,
            classifications,
            snapshot,
            ...(dependencies.modelSetVersion === undefined
              ? {}
              : { modelSetVersion: dependencies.modelSetVersion }),
          }),
        );
      } catch (error) {
        dependencies.onDebugCaptureFailure?.(error);
      }
    }

    return snapshot;
  };

  return {
    evaluate(frame) {
      if (disposalRequested) {
        return Promise.reject(new Error('Recognition pipeline has been disposed.'));
      }
      const debugRequested = dependencies.claimDebugCapture?.() === true;
      const evaluation = evaluateFrame(frame, debugRequested);
      inFlight.add(evaluation);
      void evaluation.then(
        () => inFlight.delete(evaluation),
        (error: unknown) => {
          inFlight.delete(evaluation);
          if (debugRequested) {
            dependencies.onDebugCaptureFailure?.(error);
          }
        },
      );
      return evaluation;
    },

    dispose() {
      if (disposalInFlight !== null) {
        return disposalInFlight;
      }
      disposalRequested = true;
      disposalInFlight = Promise.allSettled([...inFlight]).then(() => undefined);
      return disposalInFlight;
    },
  };
}

const browserPipelinePlatform: RecognitionPipelinePlatform = {
  buildComposite(frame, regions) {
    return buildFixedComposite(frame.source, regions);
  },

  preprocessComposite(composite) {
    return preprocessCompositeCanvas(composite);
  },

  extractCrop(frame, box) {
    const width = Math.max(1, Math.round(box.width));
    const height = Math.max(1, Math.round(box.height));
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', {
      alpha: false,
      willReadFrequently: true,
    });
    if (context === null) {
      throw new Error('Recognition crop 2D context is unavailable.');
    }
    context.drawImage(
      frame.source,
      box.x,
      box.y,
      box.width,
      box.height,
      0,
      0,
      width,
      height,
    );
    return {
      width,
      height,
      channels: 4,
      data: context.getImageData(0, 0, width, height).data,
    };
  },
};

function buildBrowserRecognitionDebugCapture(
  input: RecognitionDebugCaptureBuilderInput,
): RecognitionDebugCapture {
  const sourceRect: Rect = {
    x: 0,
    y: 0,
    width: input.frame.sourceSize.width,
    height: input.frame.sourceSize.height,
  };
  const sourceCanvas = drawSourceRect(input.frame.source, sourceRect);
  const completedHand = input.captureRegions.completed_hand.sourceRect;
  const doraIndicators = input.captureRegions.dora_indicators.sourceRect;
  const melds = input.captureRegions.melds.sourceRect;

  return {
    schemaVersion: 1,
    createdAtIso: new Date().toISOString(),
    capturedAtMs: input.frame.capturedAtMs,
    ...(input.modelSetVersion === undefined
      ? {}
      : { modelSetVersion: input.modelSetVersion }),
    sourceSize: { ...input.frame.sourceSize },
    regions: {
      'completed-hand': { ...input.frame.regions['completed-hand'] },
      'dora-indicators': { ...input.frame.regions['dora-indicators'] },
      melds: { ...input.frame.regions.melds },
    },
    sourceRegionRects: {
      'completed-hand': { ...completedHand },
      'dora-indicators': { ...doraIndicators },
      melds: { ...melds },
    },
    images: {
      sourcePngDataUrl: sourceCanvas.toDataURL('image/png'),
      compositePngDataUrl: input.composite.toDataURL('image/png'),
      regionPngDataUrls: {
        'completed-hand': drawSourceRect(
          input.frame.source,
          completedHand,
        ).toDataURL('image/png'),
        'dora-indicators': drawSourceRect(
          input.frame.source,
          doraIndicators,
        ).toDataURL('image/png'),
        melds: drawSourceRect(input.frame.source, melds).toDataURL('image/png'),
      },
    },
    detectorInput: {
      shape: [1, 3, 320, 320],
      encoding: 'base64-f32-le',
      data: float32ToBase64(input.detectorInput),
    },
    detectorOutput: {
      dims: [...input.detectorOutput.dims],
      type: input.detectorOutput.type,
      encoding: 'base64-f32-le',
      data: float32ToBase64(requireFloat32TensorData(input.detectorOutput)),
    },
    detections: input.detections.map((detection, index) => {
      const classification = input.classifications[index];
      if (classification === undefined) {
        throw modelIncompatible('tile-classifier');
      }
      return {
        id: detection.id,
        detectionIndex: detection.detectionIndex,
        confidence: detection.confidence,
        region: toRecognitionRegion(detection.region),
        compositeBox: { ...detection.box },
        sourceBox: { ...detection.sourceBox },
        classification,
      };
    }),
    snapshot: input.snapshot,
  };
}

function drawSourceRect(source: CanvasImageSource, rect: Rect): HTMLCanvasElement {
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { alpha: false });
  if (context === null) {
    throw new Error('Recognition debug 2D context is unavailable.');
  }
  context.drawImage(
    source,
    rect.x,
    rect.y,
    rect.width,
    rect.height,
    0,
    0,
    width,
    height,
  );
  return canvas;
}

function requireFloat32TensorData(output: TensorOutput): Float32Array {
  if (!(output.data instanceof Float32Array)) {
    throw new Error('Recognition debug detector output is not Float32Array.');
  }
  return output.data;
}

function float32ToBase64(values: Float32Array): string {
  const bytes = new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  );
  const chunkSize = 0x8000;
  const chunks: string[] = [];
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const chunk = bytes.subarray(offset, Math.min(offset + chunkSize, bytes.length));
    let binary = '';
    for (let index = 0; index < chunk.length; index += 1) {
      binary += String.fromCharCode(chunk[index] ?? 0);
    }
    chunks.push(binary);
  }
  return btoa(chunks.join(''));
}

async function runDetector(
  session: RecognitionInferenceSession,
  input: Float32Array,
): Promise<TensorOutput> {
  if (input.length !== 3 * NANODET_INPUT_SIZE * NANODET_INPUT_SIZE) {
    throw modelIncompatible('detector');
  }
  const inputName = firstName(session.inputNames, 'detector');
  const outputName = firstName(session.outputNames, 'detector');
  let outputs: Readonly<Record<string, unknown>>;
  try {
    const tensor = session.createFloat32Tensor(input, [
      1,
      3,
      NANODET_INPUT_SIZE,
      NANODET_INPUT_SIZE,
    ]);
    outputs = await session.run({ [inputName]: tensor });
  } catch (error) {
    if (isRecognitionRuntimeError(error)) {
      throw error;
    }
    throw inferenceFailure('detector', error);
  }

  const output = outputs[outputName];
  if (!isTensorOutput(output)) {
    throw modelIncompatible('detector');
  }
  return output;
}

function createClassifierSessionAdapter(
  session: RecognitionInferenceSession,
  role: 'tile-classifier' | 'red-five-classifier',
  expectedLogits: number,
): ClassifierSession {
  return {
    async run(input: ClassifierTensor): Promise<ArrayLike<number>> {
      const inputName = firstName(session.inputNames, role);
      const outputName = firstName(session.outputNames, role);

      let outputs: Readonly<Record<string, unknown>>;
      try {
        const tensor = session.createFloat32Tensor(input.data, input.shape);
        outputs = await session.run({ [inputName]: tensor });
      } catch (error) {
        if (isRecognitionRuntimeError(error)) {
          throw error;
        }
        throw inferenceFailure(role, error);
      }

      const output = outputs[outputName];
      const batchSize = input.shape[0];
      if (
        !Number.isInteger(batchSize) ||
        batchSize < 1 ||
        !hasFloatData(output) ||
        output.data.length !== expectedLogits * batchSize
      ) {
        throw modelIncompatible(role);
      }
      return output.data;
    },
  };
}

function toCaptureRegions(frame: RecognitionFrame): CaptureRegions {
  return {
    completed_hand: {
      enabled: true,
      sourceRect: denormalizeRect(frame.regions['completed-hand'], frame),
    },
    dora_indicators: {
      enabled: true,
      sourceRect: denormalizeRect(frame.regions['dora-indicators'], frame),
    },
    melds: {
      enabled: true,
      sourceRect: denormalizeRect(frame.regions.melds, frame),
    },
  };
}

function denormalizeRect(rect: NormalizedRect, frame: RecognitionFrame): Rect {
  return {
    x: rect.x * frame.sourceSize.width,
    y: rect.y * frame.sourceSize.height,
    width: rect.width * frame.sourceSize.width,
    height: rect.height * frame.sourceSize.height,
  };
}

function normalizeRect(rect: Rect, frame: RecognitionFrame): NormalizedRect {
  return {
    x: rect.x / frame.sourceSize.width,
    y: rect.y / frame.sourceSize.height,
    width: rect.width / frame.sourceSize.width,
    height: rect.height / frame.sourceSize.height,
  };
}

function toRecognitionRegion(region: SemanticRegion): RecognitionRegion {
  switch (region) {
    case 'completed_hand':
      return 'completed-hand';
    case 'dora_indicators':
      return 'dora-indicators';
    case 'melds':
      return 'melds';
  }
}

function validateFrame(frame: RecognitionFrame): void {
  if (
    !Number.isFinite(frame.sourceSize.width) ||
    !Number.isFinite(frame.sourceSize.height) ||
    frame.sourceSize.width <= 0 ||
    frame.sourceSize.height <= 0
  ) {
    throw new Error('Recognition frame source size must be finite and positive.');
  }
  if (!Number.isFinite(frame.capturedAtMs)) {
    throw new Error('Recognition frame capture time must be finite.');
  }

  for (const region of [
    'completed-hand',
    'dora-indicators',
    'melds',
  ] as const satisfies readonly RecognitionRegion[]) {
    validateNormalizedRect(frame.regions[region], region);
  }
}

function validateNormalizedRect(rect: NormalizedRect, label: string): void {
  const right = rect.x + rect.width;
  const bottom = rect.y + rect.height;
  if (
    !Number.isFinite(rect.x) ||
    !Number.isFinite(rect.y) ||
    !Number.isFinite(rect.width) ||
    !Number.isFinite(rect.height) ||
    rect.x < 0 ||
    rect.y < 0 ||
    rect.width <= 0 ||
    rect.height <= 0 ||
    right > 1 ||
    bottom > 1
  ) {
    throw new Error(`${label} recognition region must be a normalized rectangle.`);
  }
}

function firstName(
  names: readonly string[],
  role: RecognitionModelRole,
): string {
  const name = names[0];
  if (name === undefined || name.length === 0) {
    throw modelIncompatible(role);
  }
  return name;
}

function isTensorOutput(value: unknown): value is TensorOutput {
  return (
    typeof value === 'object' &&
    value !== null &&
    'dims' in value &&
    Array.isArray(value.dims) &&
    'data' in value &&
    'type' in value &&
    typeof value.type === 'string'
  );
}

function hasFloatData(
  value: unknown,
): value is { readonly data: Float32Array } {
  return (
    typeof value === 'object' &&
    value !== null &&
    'data' in value &&
    value.data instanceof Float32Array
  );
}

function resolveClassifierNormalization(
  baseRuntimeSpec: RecognitionModelRuntimeSpec,
  redFiveRuntimeSpec: RecognitionModelRuntimeSpec,
): ProductionClassifierNormalization {
  const base = getRecognitionClassifierNormalization(baseRuntimeSpec);
  if (base === null) {
    throw modelIncompatible('tile-classifier');
  }
  const redFive = getRecognitionClassifierNormalization(redFiveRuntimeSpec);
  if (redFive === null) {
    throw modelIncompatible('red-five-classifier');
  }
  return { base, redFive };
}

function inferenceFailure(
  model: RecognitionModelRole,
  cause: unknown,
): RecognitionRuntimeError {
  return { kind: 'inference-failure', model, cause };
}

function modelIncompatible(model: RecognitionModelRole): RecognitionRuntimeError {
  return { kind: 'model-incompatible', model };
}

function reportEvaluationTiming(
  report: ((timing: RecognitionEvaluationTiming) => void) | undefined,
  timing: RecognitionEvaluationTiming,
): void {
  if (report === undefined) {
    return;
  }
  try {
    report(timing);
  } catch {
    // Diagnostics must never change Recognition behavior.
  }
}

function monotonicNow(): number {
  return globalThis.performance?.now() ?? Date.now();
}
