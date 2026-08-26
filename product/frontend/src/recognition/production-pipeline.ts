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
import type { RecognitionFrame, RecognitionPipeline } from './contracts';
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

export interface ProductionRecognitionPipelineDependencies {
  readonly modelRuntime: RecognitionModelRuntimeInspection;
  /** Test-only/internal seam. Production composition resolves this from runtimeSpec. */
  readonly classifierNormalizationOverride?: ProductionClassifierNormalization;
  readonly detectorPostprocessor?: NanoDetDetectionPostprocessor;
  readonly platform?: RecognitionPipelinePlatform;
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
  });

  const inFlight = new Set<Promise<FrameRecognitionSnapshot>>();
  let disposalRequested = false;
  let disposalInFlight: Promise<void> | null = null;

  const evaluateFrame = async (
    frame: RecognitionFrame,
  ): Promise<FrameRecognitionSnapshot> => {
    validateFrame(frame);

    const captureRegions = toCaptureRegions(frame);
    let detectorInput: Float32Array;
    try {
      const composite = platform.buildComposite(frame, captureRegions);
      detectorInput = platform.preprocessComposite(composite);
    } catch (error) {
      if (isRecognitionRuntimeError(error)) {
        throw error;
      }
      throw inferenceFailure('detector', error);
    }
    const detectorOutput = await runDetector(detector, detectorInput);
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

    const candidates: ClassifiedRecognitionCandidate[] = [];
    for (const detection of detections) {
      const classification = await classifyDetection(
        frame,
        detection,
        classifier.classify,
        platform,
      );
      candidates.push({
        id: detection.id,
        region: toRecognitionRegion(detection.region),
        bbox: normalizeRect(detection.sourceBox, frame),
        classification,
      });
    }

    return buildFrameRecognitionSnapshot(candidates);
  };

  return {
    evaluate(frame) {
      if (disposalRequested) {
        return Promise.reject(new Error('Recognition pipeline has been disposed.'));
      }
      const evaluation = evaluateFrame(frame);
      inFlight.add(evaluation);
      void evaluation.then(
        () => inFlight.delete(evaluation),
        () => inFlight.delete(evaluation),
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
      if (!hasFloatData(output) || output.data.length !== expectedLogits) {
        throw modelIncompatible(role);
      }
      return output.data;
    },
  };
}

async function classifyDetection(
  frame: RecognitionFrame,
  detection: RegionDetection,
  classify: (crop: ClassifierCropImage) => Promise<TileClassification>,
  platform: RecognitionPipelinePlatform,
): Promise<TileClassification> {
  try {
    return await classify(platform.extractCrop(frame, detection.sourceBox));
  } catch (error) {
    if (isRecognitionRuntimeError(error)) {
      throw error;
    }
    throw inferenceFailure('tile-classifier', error);
  }
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
