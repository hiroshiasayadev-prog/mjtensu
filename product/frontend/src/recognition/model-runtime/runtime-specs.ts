import {
  RECOGNITION_MODEL_RUNTIME_SPECS,
  type RecognitionModelRole,
  type RecognitionModelRuntimeSpec,
} from './types';

export interface RecognitionRuntimeSpecDescriptor {
  readonly runtimeSpec: RecognitionModelRuntimeSpec;
  readonly role: RecognitionModelRole;
}

const RUNTIME_SPEC_DESCRIPTORS = {
  'nanodet-plus-m-320-v1': {
    runtimeSpec: 'nanodet-plus-m-320-v1',
    role: 'detector',
  },
  'c8-tile-35-v1': {
    runtimeSpec: 'c8-tile-35-v1',
    role: 'tile-classifier',
  },
  'c8-red-five-v1': {
    runtimeSpec: 'c8-red-five-v1',
    role: 'red-five-classifier',
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
