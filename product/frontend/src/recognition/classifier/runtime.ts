import type { TileClassification } from './labels';
import {
  isRedFiveSpecialistBaseKind,
  mapBaseClassifierLogits,
  mapRedFiveLogits,
} from './labels';
import {
  CLASSIFIER_IMAGE_SIZE,
  makeBaseClassifierBatchTensor,
  makeRedFiveClassifierBatchTensor,
  type ClassifierCropImage,
  type ClassifierNormalization,
  type ClassifierTensor,
} from './preprocessing';

const BASE_CLASS_COUNT = 35;
const RED_FIVE_CLASS_COUNT = 2;

export interface ClassifierSession {
  run(input: ClassifierTensor): Promise<ArrayLike<number>>;
}

export interface ClassifierBatchTiming {
  readonly candidateCount: number;
  readonly redFiveCandidateCount: number;
  readonly basePreprocessingMs: number;
  readonly baseInferenceMs: number;
  readonly redFivePreprocessingMs: number;
  readonly redFiveInferenceMs: number;
}

export interface C8ClassifierRuntimeOptions {
  readonly baseClassifier: ClassifierSession;
  readonly redFiveClassifier: ClassifierSession;
  readonly baseNormalization: ClassifierNormalization;
  readonly redFiveNormalization: ClassifierNormalization;
  readonly imageSize?: number;
  readonly now?: () => number;
}

export interface C8ClassifierRuntime {
  classify(crop: ClassifierCropImage): Promise<TileClassification>;
  classifyBatch(crops: readonly ClassifierCropImage[]): Promise<readonly TileClassification[]>;
  getLastBatchTiming(): ClassifierBatchTiming | null;
}

export function createC8ClassifierRuntime(
  options: C8ClassifierRuntimeOptions,
): C8ClassifierRuntime {
  const imageSize = options.imageSize ?? CLASSIFIER_IMAGE_SIZE;
  const now = options.now ?? monotonicNow;
  let lastBatchTiming: ClassifierBatchTiming | null = null;

  const classifyBatch = async (
    crops: readonly ClassifierCropImage[],
  ): Promise<readonly TileClassification[]> => {
    if (crops.length === 0) {
      lastBatchTiming = {
        candidateCount: 0,
        redFiveCandidateCount: 0,
        basePreprocessingMs: 0,
        baseInferenceMs: 0,
        redFivePreprocessingMs: 0,
        redFiveInferenceMs: 0,
      };
      return [];
    }

    const basePreprocessingStartedAt = now();
    const baseTensor = makeBaseClassifierBatchTensor(
      crops,
      options.baseNormalization,
      imageSize,
    );
    const basePreprocessingFinishedAt = now();
    const baseLogits = await options.baseClassifier.run(baseTensor);
    const baseInferenceFinishedAt = now();
    assertBatchLogitCount(
      baseLogits,
      crops.length,
      BASE_CLASS_COUNT,
      'base classifier',
    );

    const results = Array.from({ length: crops.length }, (_, index) =>
      mapBaseClassifierLogits(
        sliceLogits(baseLogits, index, BASE_CLASS_COUNT),
      ),
    );
    const redFiveCandidateIndices = results.flatMap((result, index) =>
      result.kind === 'tile' && isRedFiveSpecialistBaseKind(result.tile.kind)
        ? [index]
        : [],
    );

    let redFivePreprocessingMs = 0;
    let redFiveInferenceMs = 0;
    if (redFiveCandidateIndices.length > 0) {
      const redFivePreprocessingStartedAt = now();
      const redFiveCrops = redFiveCandidateIndices.map((index) => {
        const crop = crops[index];
        if (crop === undefined) {
          throw new Error(`Missing classifier crop at index ${index}`);
        }
        return crop;
      });
      const redFiveTensor = makeRedFiveClassifierBatchTensor(
        redFiveCrops,
        options.redFiveNormalization,
        imageSize,
      );
      const redFivePreprocessingFinishedAt = now();
      const redFiveLogits = await options.redFiveClassifier.run(redFiveTensor);
      const redFiveInferenceFinishedAt = now();
      assertBatchLogitCount(
        redFiveLogits,
        redFiveCandidateIndices.length,
        RED_FIVE_CLASS_COUNT,
        'red-five classifier',
      );

      redFiveCandidateIndices.forEach((candidateIndex, redFiveIndex) => {
        const baseResult = results[candidateIndex];
        if (baseResult?.kind !== 'tile') {
          throw new Error('Red-five candidate lost its base classification');
        }
        results[candidateIndex] = {
          kind: 'tile',
          tile: mapRedFiveLogits(
            baseResult.tile,
            sliceLogits(redFiveLogits, redFiveIndex, RED_FIVE_CLASS_COUNT),
          ),
        };
      });

      redFivePreprocessingMs =
        redFivePreprocessingFinishedAt - redFivePreprocessingStartedAt;
      redFiveInferenceMs = redFiveInferenceFinishedAt - redFivePreprocessingFinishedAt;
    }

    lastBatchTiming = {
      candidateCount: crops.length,
      redFiveCandidateCount: redFiveCandidateIndices.length,
      basePreprocessingMs: basePreprocessingFinishedAt - basePreprocessingStartedAt,
      baseInferenceMs: baseInferenceFinishedAt - basePreprocessingFinishedAt,
      redFivePreprocessingMs,
      redFiveInferenceMs,
    };
    return results;
  };

  return {
    async classify(crop) {
      const results = await classifyBatch([crop]);
      const result = results[0];
      if (result === undefined) {
        throw new Error('Classifier returned no result for one crop');
      }
      return result;
    },
    classifyBatch,
    getLastBatchTiming() {
      return lastBatchTiming;
    },
  };
}

function assertBatchLogitCount(
  logits: ArrayLike<number>,
  batchSize: number,
  classCount: number,
  classifierName: string,
): void {
  const expected = batchSize * classCount;
  if (logits.length !== expected) {
    throw new Error(
      `${classifierName} returned ${logits.length} logits for batch ${batchSize}; expected ${expected}`,
    );
  }
}

function sliceLogits(
  logits: ArrayLike<number>,
  batchIndex: number,
  classCount: number,
): ArrayLike<number> {
  const start = batchIndex * classCount;
  if (logits instanceof Float32Array) {
    return logits.subarray(start, start + classCount);
  }
  return Array.from(
    { length: classCount },
    (_, offset) => logits[start + offset] ?? Number.NaN,
  );
}

function monotonicNow(): number {
  return globalThis.performance?.now() ?? Date.now();
}
