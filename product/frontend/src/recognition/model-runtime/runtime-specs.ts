import {
  RECOGNITION_MODEL_RUNTIME_SPECS,
  type RecognitionModelRole,
  type RecognitionModelRuntimeSpec,
} from './types';

export interface RuntimeSpecClassifierNormalization {
  readonly mean: readonly number[];
  readonly std: readonly number[];
}

export interface RecognitionRuntimeSpecDescriptor {
  readonly runtimeSpec: RecognitionModelRuntimeSpec;
  readonly role: RecognitionModelRole;
  readonly classifierNormalization?: RuntimeSpecClassifierNormalization | null;
}

const RUNTIME_SPEC_DESCRIPTORS = {
  'nanodet-plus-m-320-v1': {
    runtimeSpec: 'nanodet-plus-m-320-v1',
    role: 'detector',
  },
  'rotated-fcos-nano-320-v1': {
    runtimeSpec: 'rotated-fcos-nano-320-v1',
    role: 'detector',
  },
  'gray64-tile-35-v1': {
    runtimeSpec: 'gray64-tile-35-v1',
    role: 'tile-classifier',
    classifierNormalization: {
      mean: [0.6815832403977466],
      std: [0.2725553681973969],
    },
  },
  'gray64-tile-35-v2': {
    runtimeSpec: 'gray64-tile-35-v2',
    role: 'tile-classifier',
    classifierNormalization: {
      mean: [0.6816653769847909],
      std: [0.2714782333298719],
    },
  },
  'c8-red-five-v1': {
    runtimeSpec: 'c8-red-five-v1',
    role: 'red-five-classifier',
    classifierNormalization: {
      mean: [0.66025093606229934, 0.69172744263865471, 0.6489080530422624],
      std: [0.30491469480493394, 0.24924454491506576, 0.27107025824445752],
    },
  },
} as const satisfies Record<
  RecognitionModelRuntimeSpec,
  RecognitionRuntimeSpecDescriptor
>;

export function isRecognitionModelRuntimeSpec(
  value: unknown,
): value is RecognitionModelRuntimeSpec {
  return (
    typeof value === 'string' &&
    RECOGNITION_MODEL_RUNTIME_SPECS.includes(
      value as RecognitionModelRuntimeSpec,
    )
  );
}

export function getRecognitionRuntimeSpecDescriptor(
  runtimeSpec: RecognitionModelRuntimeSpec,
): RecognitionRuntimeSpecDescriptor {
  return RUNTIME_SPEC_DESCRIPTORS[runtimeSpec];
}

export function getRecognitionClassifierNormalization(
  runtimeSpec: RecognitionModelRuntimeSpec,
): RuntimeSpecClassifierNormalization | null {
  const descriptor = RUNTIME_SPEC_DESCRIPTORS[runtimeSpec];
  if (
    descriptor.role !== 'tile-classifier' &&
    descriptor.role !== 'red-five-classifier'
  ) {
    return null;
  }
  return descriptor.classifierNormalization;
}
