import type { TileClassification } from './labels';
import {
  isRedFiveSpecialistBaseKind,
  mapBaseClassifierLogits,
  mapRedFiveLogits,
} from './labels';
import {
  CLASSIFIER_IMAGE_SIZE,
  makeBaseClassifierTensor,
  makeRedFiveClassifierTensor,
  type ClassifierCropImage,
  type ClassifierNormalization,
  type ClassifierTensor,
} from './preprocessing';

export interface ClassifierSession {
  run(input: ClassifierTensor): Promise<ArrayLike<number>>;
}

export interface C8ClassifierRuntimeOptions {
  readonly baseClassifier: ClassifierSession;
  readonly redFiveClassifier: ClassifierSession;
  readonly baseNormalization: ClassifierNormalization;
  readonly redFiveNormalization: ClassifierNormalization;
  readonly imageSize?: number;
}

export interface C8ClassifierRuntime {
  classify(crop: ClassifierCropImage): Promise<TileClassification>;
}

export function createC8ClassifierRuntime(
  options: C8ClassifierRuntimeOptions,
): C8ClassifierRuntime {
  const imageSize = options.imageSize ?? CLASSIFIER_IMAGE_SIZE;

  return {
    async classify(crop) {
      const baseTensor = makeBaseClassifierTensor(
        crop,
        options.baseNormalization,
        imageSize,
      );
      const baseResult = mapBaseClassifierLogits(
        await options.baseClassifier.run(baseTensor),
      );

      if (
        baseResult.kind === 'invalid' ||
        !isRedFiveSpecialistBaseKind(baseResult.tile.kind)
      ) {
        return baseResult;
      }

      const redFiveTensor = makeRedFiveClassifierTensor(
        crop,
        options.redFiveNormalization,
        imageSize,
      );
      return {
        kind: 'tile',
        tile: mapRedFiveLogits(
          baseResult.tile,
          await options.redFiveClassifier.run(redFiveTensor),
        ),
      };
    },
  };
}
